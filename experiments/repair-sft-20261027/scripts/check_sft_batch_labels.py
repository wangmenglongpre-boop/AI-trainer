"""检查真实 SFTTrainer dataloader 的 labels；不启动训练。"""
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from trl import SFTConfig, SFTTrainer

from inspect_sft_data import inspect_files, render_labels


ROOT = Path(__file__).resolve().parents[1]
MAX_LENGTH = 1024
BATCH_SIZE = 2


def prepare_dataset(path, tokenizer, expected_split):
    """把已验证的监督边界转换为 TRL 使用的 completion_mask。"""
    rows, _ = inspect_files([path])
    records = []
    expected = {}

    for row in rows:
        if row["split"] != expected_split:
            raise ValueError(f"错误的集合：{row['id']}")

        feature = render_labels(row, tokenizer, MAX_LENGTH)
        ids = feature["input_ids"]
        prompt_length = feature["prompt_tokens"]

        # 只传入 token 和边界，不把预制 labels 交给 Trainer。
        # 实际 labels 必须由 Trainer 的默认 collator 生成。
        records.append({
            "input_ids": ids,
            "completion_mask": (
                [0] * prompt_length
                + [1] * (len(ids) - prompt_length)
            ),
        })

        key = tuple(ids)
        if key in expected:
            raise ValueError("存在完全相同的 token 序列，无法唯一匹配样本")

        expected[key] = {
            "id": row["id"],
            "prompt_length": prompt_length,
            "answer": row["messages"][-1]["content"],
        }

    return Dataset.from_list(records), expected


def inspect_loader(loader, expected, tokenizer, pad_id, name):
    """逐条检查真实 batch；通过 token 序列匹配，兼容训练集洗牌。"""
    seen = set()
    supervised_total = 0
    padding_total = 0
    batch_count = 0

    for batch_index, batch in enumerate(loader):
        batch_count += 1
        ids = batch["input_ids"].cpu()
        attention = batch["attention_mask"].cpu()
        labels = batch["labels"].cpu()

        assert ids.shape == attention.shape == labels.shape
        assert ids.ndim == 2

        for i in range(ids.shape[0]):
            assert torch.all((attention[i] == 0) | (attention[i] == 1))

            length = int(attention[i].sum().item())
            width = ids.shape[1]

            # 当前配置使用右侧 padding：前面为真实输入，后面为补齐。
            assert torch.all(attention[i, :length] == 1)
            assert torch.all(attention[i, length:] == 0)

            real_ids = ids[i, :length].tolist()
            key = tuple(real_ids)
            assert key in expected, "Trainer 改变或截断了输入"

            info = expected[key]
            sample_id = info["id"]
            assert sample_id not in seen, "同一遍遍历中出现重复样本"
            seen.add(sample_id)

            p = info["prompt_length"]

            # 提示保留为输入，但所有提示目标都应被忽略。
            assert torch.all(labels[i, :p] == -100)

            # 回答部分的 labels 应与同位置 input_ids 一致。
            assert torch.equal(labels[i, p:length], ids[i, p:length])

            # padding 的 token ID 和监督屏蔽分别检查。
            assert torch.all(ids[i, length:] == pad_id)
            assert torch.all(labels[i, length:] == -100)

            # 独立检查参与监督的文本，不能混入提示或遗漏回答。
            supervised = labels[i][labels[i] != -100].tolist()
            decoded = tokenizer.decode(
                supervised, skip_special_tokens=False
            )
            wanted = info["answer"] + "<|im_end|>\n"
            assert decoded == wanted, (
                f"{sample_id} 监督文本不一致：{decoded!r}"
            )

            # Qwen 的真实结束符必须参与监督。
            eos_positions = (
                ids[i, p:length] == tokenizer.eos_token_id
            )
            assert eos_positions.any(), "回答没有真实结束符"
            assert torch.all(
                labels[i, p:length][eos_positions]
                == tokenizer.eos_token_id
            )

            supervised_total += len(supervised)
            padding_total += width - length

            # 展示第一批，避免打印全部数据。
            if batch_index == 0:
                left = max(0, p - 4)
                right = min(length, p + 6)

                print(f"\n[{name}] {sample_id}")
                print("batch shape:", tuple(ids.shape))
                print("prompt tokens:", p)
                print("answer tokens:", len(supervised))
                print("padding tokens:", width - length)
                print("边界 input_ids:", ids[i, left:right].tolist())
                print("边界 labels:   ", labels[i, left:right].tolist())
                print("参与 loss 的文本:\n" + decoded)

    assert seen == {v["id"] for v in expected.values()}
    assert padding_total > 0, "本轮没有实际覆盖 padding"

    return {
        "samples": len(seen),
        "batches": batch_count,
        "supervised_tokens": supervised_total,
        "padding_tokens": padding_total,
    }


def run_check(model, tokenizer, datasets, expectations, tag, pad_token):
    """使用 SFTTrainer 默认 collator，覆盖 train 和 dev dataloader。"""
    config = SFTConfig(
        output_dir=str(ROOT / "outputs/batch-label-audit" / tag),
        use_cpu=True,
        bf16=False,
        fp16=False,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to=[],
        save_strategy="no",
        disable_tqdm=True,
        seed=42,
        max_length=MAX_LENGTH,
        packing=False,
        padding_free=False,
        completion_only_loss=True,
        pad_token=pad_token,
        # 数据已经过严格长度检查并完成 tokenization。
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        # 不传 data_collator：验证 TRL 自己的默认实现。
    )

    print(f"\n=== {tag} ===")
    print("collator:", type(trainer.data_collator).__name__)

    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    results = {}

    for split, loader in (
        ("train", trainer.get_train_dataloader()),
        ("dev", trainer.get_eval_dataloader()),
    ):
        results[split] = inspect_loader(
            loader,
            expectations[split],
            tokenizer,
            pad_id,
            f"{tag}/{split}",
        )

    assert results["train"]["supervised_tokens"] == 1152
    assert results["dev"]["supervised_tokens"] == 207
    return results


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models/qwen2.5-coder-1.5b-tokenizer",
        local_files_only=True,
    )
    tokenizer.padding_side = "right"

    datasets = {}
    expectations = {}
    for split in ("train", "dev"):
        datasets[split], expectations[split] = prepare_dataset(
            ROOT / f"data/repair/{split}.jsonl",
            tokenizer,
            split,
        )

    # 仅用于初始化真实 Trainer；不做 forward/backward/train。
    model_config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=MAX_LENGTH,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(model_config)

    results = {
        "normal_padding": run_check(
            model, tokenizer, datasets, expectations,
            "normal_padding",
            tokenizer.pad_token or tokenizer.eos_token,
        ),
        # 压力检查：padding 与真实结束符使用相同 token ID。
        "pad_equals_eos": run_check(
            model, tokenizer, datasets, expectations,
            "pad_equals_eos",
            tokenizer.eos_token,
        ),
    }

    report = {
        "passed": True,
        "actual_sfttrainer_dataloaders_checked": True,
        "training_started": False,
        "model": "tiny random Qwen2, CPU",
        "results": results,
    }
    report_path = ROOT / "reports/sft_batch_labels_check.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nPASS: 实际 Trainer batch 的 labels 全部通过")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("报告:", report_path)


if __name__ == "__main__":
    main()
