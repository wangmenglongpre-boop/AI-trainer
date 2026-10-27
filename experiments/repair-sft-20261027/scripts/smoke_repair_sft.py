"""10 optimizer-update LoRA pipeline check; not a final model or efficacy claim."""
import argparse
import hashlib
import json
from pathlib import Path

from repair_model_common import (ROOT, generation_protocol, load_model, make_trainer,
                                 verify_data, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    if not 10 <= args.steps <= 20:
        raise ValueError("Smoke run is limited to 10–20 optimizer updates")
    baseline = json.loads(args.baseline.read_text())
    protocol = generation_protocol()
    data_manifest = verify_data()
    if (not baseline.get("complete") or baseline.get("adapter") is not None
            or not baseline["summary"]["valid_comparison"]
            or baseline["protocol"] != protocol or baseline["data_manifest"] != data_manifest):
        raise ValueError("Need a complete, valid, original-model dev baseline on these exact artifacts")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import TrainerCallback
    torch.manual_seed(42)
    model, tokenizer = load_model(args.model)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
    ))
    model.print_trainable_parameters()
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    initial = {n: p.detach().cpu().clone() for n, p in trainable.items()}
    trainer = make_trainer(model, tokenizer, args.output, steps=args.steps)
    gradient_checks = []

    class CheckGradients(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            norms = {}
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    if parameter.grad is not None:
                        raise ValueError("Frozen base received a gradient")
                    continue
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise ValueError(f"Missing/nonfinite LoRA gradient: {name}")
                norms[name] = parameter.grad.float().norm().item()
            if not any(v > 0 for v in norms.values()):
                raise ValueError("All LoRA gradients are zero")
            gradient_checks.append(dict(step=state.global_step + 1, max_norm=max(norms.values())))

    trainer.add_callback(CheckGradients())
    loss_before = trainer.evaluate()["eval_loss"]
    result = trainer.train()
    loss_after = trainer.evaluate()["eval_loss"]
    assert len(gradient_checks) == args.steps
    changed = [n for n, p in trainable.items() if not torch.equal(p.detach().cpu(), initial[n])]
    assert changed, "No LoRA parameter changed"
    assert torch.isfinite(torch.tensor([loss_before, loss_after, result.training_loss])).all()
    # Save the inference adapter only; this smoke run is not for resumable formal training.
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    batch = next(iter(trainer.get_eval_dataloader()))
    batch = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
    model.eval()
    with torch.inference_mode():
        before_reload = model(**batch).logits.float().cpu()
    # Reuse frozen base to avoid holding two complete 1.5B models on GPU.
    frozen_base = model.unload()
    reloaded = PeftModel.from_pretrained(frozen_base, args.output, is_trainable=False)
    reloaded.eval()
    with torch.inference_mode():
        after_reload = reloaded(**batch).logits.float().cpu()
    difference = (before_reload - after_reload).abs().max().item()
    assert torch.allclose(before_reload, after_reload, atol=1e-4, rtol=1e-4), difference
    manifest = dict(protocol=protocol, data_manifest=data_manifest, steps=args.steps,
                    baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                    lora=dict(r=8, alpha=16, dropout=0.05,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
                    loss_before=loss_before, loss_after=loss_after,
                    training_loss=result.training_loss, changed_lora_tensors=len(changed),
                    gradient_checks=gradient_checks, reload_max_logit_difference=difference,
                    scope="Smoke check only; restart from original base for formal SFT")
    write_json(args.output / "smoke_manifest.json", manifest)
    write_json(ROOT / "reports/repair_sft_smoke.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
