"""验证 agi-post：依赖、CUDA/bf16、微型随机 Qwen 的 LoRA 更新与重载。

不下载基座权重；随机微型模型仅用于环境验证，不是正式任务训练。
"""
import argparse
import importlib.metadata
import inspect
import json
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    import torch
    import bitsandbytes
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from trl import SFTConfig, SFTTrainer

    expected = dict(torch="2.6.0+cu124", transformers="4.51.3", peft="0.15.2",
                    trl="0.17.0", accelerate="1.6.0", datasets="3.5.0",
                    bitsandbytes="0.45.5", numpy="1.26.4", tokenizers="0.21.1",
                    wandb="0.19.11")
    versions = {name: importlib.metadata.version(name) for name in expected}
    assert versions == expected, (versions, expected)
    assert sys.version_info[:2] == (3, 11), sys.version
    assert torch.cuda.is_available(), "CUDA 不可用"
    assert torch.cuda.is_bf16_supported(), "bf16 不可用"
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(x @ x.T).all()
    config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=64,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = "sdpa"
    torch.manual_seed(42)
    base = Qwen2ForCausalLM(config).cuda()
    base.eval()
    ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 2]], device="cuda")
    labels = ids.clone()
    labels[:, :4] = -100
    with torch.no_grad():
        initial_logits = base(ids).logits.clone()
    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM"))
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, initial_logits, atol=1e-6)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()
               if p.requires_grad}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=1e-3)
    model.train()
    loss = model(input_ids=ids, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
        else:
            assert param.grad is None, name
    optimizer.step()
    changed = [name for name, p in model.named_parameters()
               if p.requires_grad and not torch.equal(initial[name], p)]
    assert changed, "LoRA 参数未更新"
    model.eval()
    with torch.no_grad():
        trained_logits = model(ids).logits.clone()
        with model.disable_adapter():
            assert torch.allclose(model(ids).logits, initial_logits, atol=1e-6)
    with tempfile.TemporaryDirectory(prefix="lora-env-check-") as temp:
        model.save_pretrained(temp)
        torch.manual_seed(42)
        reloaded_base = Qwen2ForCausalLM(config).cuda()
        reloaded = PeftModel.from_pretrained(reloaded_base, temp)
        reloaded.eval()
        with torch.no_grad():
            reload_difference = (reloaded(ids).logits - trained_logits).abs().max().item()
            assert reload_difference < 1e-6, reload_difference
    fields = inspect.signature(SFTConfig).parameters
    report = dict(
        status="passed", python=sys.version, executable=sys.executable, packages=versions,
        gpu=torch.cuda.get_device_name(0), bf16_supported=True,
        toy_lora_loss=loss.item(), updated_parameter_tensors=len(changed),
        adapter_reload_max_difference=reload_difference,
        trl_sft_config_fields={name: name in fields for name in
                               ("max_length", "max_seq_length", "completion_only_loss",
                                "dataset_kwargs")},
        scope="Imports, CUDA and manual LoRA update/reload; no real-model SFT training.",
    )
    (args.report_dir / "post_environment_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
