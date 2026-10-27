"""Greedy dev baseline or adapter evaluation; never consumes assistant references."""
import argparse
import ast
import json
import subprocess
import time
from pathlib import Path

from repair_candidate_sandbox import InfrastructureError, case_passed, run_candidate
from repair_model_common import (ROOT, MAX_LENGTH, MAX_NEW_TOKENS, generation_protocol,
                                 load_model, verify_data, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    data_manifest = verify_data()
    image = data_manifest["image_id"]
    subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True)
    subprocess.run(["docker", "info"], check=True, capture_output=True)
    protocol = generation_protocol()
    rows = [json.loads(line) for line in (ROOT / "data/repair/dev_prompts.jsonl").read_text().splitlines()]
    evaluator = json.loads((ROOT / "data/repair/evaluator_cases.json").read_text())
    import torch
    torch.manual_seed(42)
    model, tokenizer = load_model(args.model)
    if args.adapter:
        from peft import PeftModel
        adapter_manifest = json.loads((args.adapter / "smoke_manifest.json").read_text())
        if adapter_manifest["protocol"] != protocol:
            raise ValueError("Adapter protocol mismatch")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    from transformers import GenerationConfig
    decoding = GenerationConfig(
        do_sample=False, num_beams=1, repetition_penalty=1.0,
        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    results = []
    for row in rows:
        if row["split"] != "dev" or [m["role"] for m in row["messages"]] != ["system", "user"]:
            raise ValueError("Only held-out dev prompts accepted")
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                              add_generation_prompt=True)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
        n = inputs["input_ids"].shape[1]
        if n > MAX_LENGTH:
            raise ValueError("Prompt exceeds frozen budget")
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=decoding, use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        new_ids = generated[0, n:].tolist()
        raw = tokenizer.decode(new_ids, skip_special_tokens=False)
        code = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        ended = bool(new_ids and new_ids[-1] == tokenizer.eos_token_id)
        entry = dict(id=row["id"], raw=raw, code=code, latency_seconds=elapsed,
                     output_tokens=len(new_ids), truncated=not ended and len(new_ids) == MAX_NEW_TOKENS,
                     hidden_passed=False, public_passed=False, infrastructure_error=None)
        tests = evaluator[row["source_task_id"]]
        try:
            tree = ast.parse(code)
            if "```" in code or not any(isinstance(node, ast.FunctionDef) and node.name == tests["function"]
                                       for node in tree.body):
                raise ValueError("Expected raw Python with required function")
            entry["format_valid"] = True
            cases = tests["cases"]
            actual = run_candidate(code, tests["function"], cases, image)
            no_mutation = "not modify" in row["messages"][1]["content"]
            passed = [case_passed(a, c, no_mutation) for a, c in zip(actual, cases)]
            entry["public_passed"] = all(ok for ok, c in zip(passed, cases) if c["visibility"] == "public")
            entry["hidden_passed"] = all(ok for ok, c in zip(passed, cases) if c["visibility"] == "hidden")
            entry["case_statuses"] = [a["status"] for a in actual]
        except (SyntaxError, ValueError) as e:
            entry["format_valid"] = False
            entry["format_error"] = str(e)[:200]
        except InfrastructureError as e:
            entry["infrastructure_error"] = str(e)
        results.append(entry)
        write_json(args.output, dict(protocol=protocol, adapter=str(args.adapter) if args.adapter else None,
                                    complete=False, results=results))
        print(f"{row['id']}: hidden={entry['hidden_passed']} format={entry['format_valid']} "
              f"tokens={len(new_ids)} seconds={elapsed:.2f}", flush=True)
    infra = sum(r["infrastructure_error"] is not None for r in results)
    summary = dict(total=len(results), hidden_passed=sum(r["hidden_passed"] for r in results),
                   public_passed=sum(r["public_passed"] for r in results),
                   format_valid=sum(r["format_valid"] for r in results),
                   infrastructure_failures=infra, valid_comparison=infra == 0)
    write_json(args.output, dict(protocol=protocol, data_manifest=data_manifest,
                                model_path=str(args.model), adapter=str(args.adapter) if args.adapter else None,
                                complete=True, summary=summary, results=results))
    print(json.dumps(summary, indent=2))
    if infra:
        raise SystemExit("Infrastructure failure: fix and rerun with a new output path")


if __name__ == "__main__":
    main()
