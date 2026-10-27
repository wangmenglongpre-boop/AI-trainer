"""Shared model identity, template, data and decoding for baseline/SFT."""
import hashlib
import json
from pathlib import Path

from download_repair_base import MODEL, REVISION, WEIGHT_SHA, WEIGHT_SIZE
from inspect_sft_data import inspect_files, render_labels

ROOT = Path(__file__).resolve().parents[1]
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 512


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_base(path):
    manifest = json.loads((path / "full_model_manifest.json").read_text())
    if manifest["model_id"] != MODEL or manifest["revision"] != REVISION:
        raise ValueError("Unexpected base revision")
    if (manifest["files"]["model.safetensors"]["sha256"] != WEIGHT_SHA
            or (path / "model.safetensors").stat().st_size != WEIGHT_SIZE):
        raise ValueError("Unexpected/incomplete base weights; run download --verify-only")
    for name, info in manifest["files"].items():
        if not (path / name).is_file() or (path / name).stat().st_size != info["bytes"]:
            raise ValueError(f"Missing or incomplete base file: {name}")
    return manifest


def verify_data():
    manifest = json.loads((ROOT / "data/repair/manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        current = hashlib.sha256((ROOT / "data/repair" / name).read_bytes()).hexdigest()
        if current != checksum:
            raise ValueError(f"Data changed: {name}")
    return manifest


def load_model(path, device="cuda"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    verify_base(path)
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA/bf16 unavailable")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "right"
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    # Official config disables sliding-window attention. Make full attention explicit.
    config.use_sliding_window = False
    config.sliding_window = None
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, local_files_only=True, use_safetensors=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    return model, tokenizer


def supervised_dataset(split, tokenizer):
    from datasets import Dataset
    if split not in ("train", "dev"):
        raise ValueError("SFT only accepts explicit train/dev")
    rows, _ = inspect_files([ROOT / f"data/repair/{split}.jsonl"])
    features = [render_labels(row, tokenizer, MAX_LENGTH) for row in rows]
    records = [dict(input_ids=f["input_ids"],
                    completion_mask=[0] * f["prompt_tokens"] + [1] * f["answer_tokens"])
               for f in features]
    return Dataset.from_list(records), features


def make_trainer(model, tokenizer, output, device="cuda", steps=10, batch_size=1):
    from trl import SFTConfig, SFTTrainer
    train, train_features = supervised_dataset("train", tokenizer)
    dev, dev_features = supervised_dataset("dev", tokenizer)
    config = SFTConfig(
        output_dir=str(output), use_cpu=device == "cpu", bf16=device == "cuda",
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, max_steps=steps, learning_rate=1e-4,
        lr_scheduler_type="constant", max_grad_norm=1.0, optim="adamw_torch",
        seed=42, data_seed=42, report_to=[], logging_steps=1, save_strategy="no",
        eval_strategy="no", packing=False, padding_free=False, completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True}, max_length=MAX_LENGTH,
        remove_unused_columns=False, label_names=["labels"], dataloader_num_workers=0,
        gradient_checkpointing=device == "cuda",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    trainer = SFTTrainer(model=model, args=config, processing_class=tokenizer,
                         train_dataset=train, eval_dataset=dev)
    # Inspect the actual runtime batch again using this production configuration.
    lengths = {tuple(f["input_ids"]): f for f in train_features}
    for batch in trainer.get_train_dataloader():
        for ids, attention, labels in zip(batch["input_ids"], batch["attention_mask"], batch["labels"]):
            n = int(attention.sum().item())
            expected = lengths[tuple(ids[:n].tolist())]
            if labels[:n].tolist() != expected["labels"] or any(x != -100 for x in labels[n:].tolist()):
                raise ValueError("Actual runtime batch labels changed")
    return trainer


def generation_protocol():
    return dict(base_model_id=MODEL, revision=REVISION, split="dev",
                decoding="greedy", max_new_tokens=MAX_NEW_TOKENS, max_prompt_tokens=MAX_LENGTH,
                num_beams=1, repetition_penalty=1.0,
                parsing="Only raw Python; surrounding whitespace stripped; fences rejected",
                metric="all hidden cases passed, numerator/denominator",
                attention="full sdpa, sliding window disabled", seed=42)
