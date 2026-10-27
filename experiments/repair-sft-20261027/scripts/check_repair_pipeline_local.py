"""CPU/tiny-model integration checks plus real Docker scoring. No full-base SFT."""
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from repair_candidate_sandbox import case_passed, run_candidate
from repair_model_common import ROOT, generation_protocol, make_trainer, verify_data, write_json


def main():
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, TrainerCallback
    torch.set_num_threads(2)
    image = subprocess.check_output(["docker", "image", "inspect", "python:3.11-slim",
                                     "--format", "{{.Id}}"], text=True).strip()
    sources = json.loads((ROOT / "configs/repair_source_catalog.json").read_text())
    references = 0
    for task in sources:
        if task["split"] != "dev":
            continue
        function = task["signature"].split("(")[0].split()[-1]
        actual = run_candidate(task["reference"], function, task["cases"], image)
        assert all(case_passed(a, c) for a, c in zip(actual, task["cases"])), task["family"]
        buggy = run_candidate(task["bugs"][0]["code"], function, task["cases"][:1], image)
        assert not case_passed(buggy[0], task["cases"][0]), task["family"]
        references += len(task["cases"])
    adversarial = [
        ("def f():\n    print('spoof')\n    return 1", [], 1, False),
        ("def f():\n    raise RuntimeError()", [], 1, False),
        ("def f():\n    while True:\n        pass", [], 1, False),
        ("def f():\n    print('x' * 100000)\n    return 1", [], 1, False),
        ("def f(x):\n    x.append(1)\n    return 1", [[]], 1, True),
    ]
    for code, args, value, mutation in adversarial:
        case = dict(args=args, expected=value)
        actual = run_candidate(code, "f", [case], image)
        assert not case_passed(actual[0], case, no_mutation=mutation), actual
    print("Docker reference/bug/output/timeout/mutation checks passed", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "outputs/qwen2.5-coder-1.5b-tokenizer",
                                              local_files_only=True)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                        max_position_embeddings=1024, tie_word_embeddings=True,
                        use_sliding_window=False, sliding_window=None)
    torch.manual_seed(42)
    model = get_peft_model(Qwen2ForCausalLM(config), LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"))
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    gradient_events = []

    class Check(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            grads = []
            for n, p in model.named_parameters():
                if p.requires_grad:
                    assert "lora_" in n and p.grad is not None and torch.isfinite(p.grad).all()
                    grads.append(p.grad.norm().item())
                else:
                    assert p.grad is None
            assert any(v > 0 for v in grads)
            gradient_events.append(True)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        trainer = make_trainer(model, tokenizer, path / "trainer", device="cpu", steps=1, batch_size=2)
        trainer.add_callback(Check())
        evaluation_loss = trainer.evaluate()["eval_loss"]
        assert torch.isfinite(torch.tensor(evaluation_loss))
        batch = next(iter(trainer.get_eval_dataloader()))
        model.eval()
        with torch.inference_mode():
            output = model(**batch)
        manual = torch.nn.functional.cross_entropy(
            output.logits[:, :-1].reshape(-1, output.logits.size(-1)),
            batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
        assert torch.allclose(manual, output.loss, atol=1e-5, rtol=1e-5)
        training = trainer.train()
        assert gradient_events == [True]
        assert any(not torch.equal(p, initial[n]) for n, p in model.named_parameters() if n in initial)
        model.save_pretrained(path / "adapter")
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        model.eval()
        with torch.inference_mode():
            before = model(**inputs).logits
        restored = PeftModel.from_pretrained(model.unload(), path / "adapter", is_trainable=False)
        restored.eval()
        with torch.inference_mode():
            after = restored(**inputs).logits
        difference = (before - after).abs().max().item()
        assert torch.allclose(before, after, atol=1e-5, rtol=1e-5)
        # Exercise the full 10-step smoke CLI on a tiny CPU model with a synthetic
        # baseline gate fixture. This does not represent the real base's dev score.
        import smoke_repair_sft
        fixture = path / "baseline_fixture.json"
        write_json(fixture, dict(complete=True, adapter=None, summary=dict(valid_comparison=True),
                                 protocol=generation_protocol(), data_manifest=verify_data()))
        tiny = Qwen2ForCausalLM(config)

        def cpu_trainer(model, tokenizer, output, steps):
            return make_trainer(model, tokenizer, output, device="cpu", steps=steps)

        argv = ["smoke_repair_sft.py", "--model", str(path / "tiny"),
                "--baseline", str(fixture), "--output", str(path / "smoke"), "--steps", "10"]
        with patch("sys.argv", argv), patch.object(smoke_repair_sft, "ROOT", path), \
                patch.object(smoke_repair_sft, "load_model", return_value=(tiny, tokenizer)), \
                patch.object(smoke_repair_sft, "make_trainer", side_effect=cpu_trainer):
            smoke_repair_sft.main()
        smoke = json.loads((path / "smoke/smoke_manifest.json").read_text())
        assert smoke["steps"] == 10 and len(smoke["gradient_checks"]) == 10
    remaining = subprocess.check_output(["docker", "ps", "-a", "--filter", "name=repair-eval-",
                                        "--format", "{{.Names}}"], text=True).strip()
    assert not remaining, remaining
    import transformers, trl, peft
    write_json(ROOT / "reports/repair_pipeline_local_check.json",
               dict(passed=True, device="CPU", model="tiny random Qwen2; real official tokenizer",
                    versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                  trl=trl.__version__, peft=peft.__version__),
                    docker_image=image, dev_reference_cases=references,
                    adversarial_checks=len(adversarial), manual_shifted_loss=float(manual),
                    eval_loss=evaluation_loss,
                    training_loss=training.training_loss, reload_max_logit_difference=difference,
                    smoke_cli_cpu_steps=smoke["steps"],
                    real_base_gpu_checked=False))
    print("PASS: local integration checks; full base/GPU remains user execution")


if __name__ == "__main__":
    main()
