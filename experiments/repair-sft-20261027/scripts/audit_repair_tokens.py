"""Offline actual-template checks; does not claim Trainer collator validation."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from build_repair_data import write_json
from inspect_sft_data import inspect_files, render_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows, counts = inspect_files([root / f"data/repair/{s}.jsonl" for s in ("train", "dev", "test")])
    manifest = json.loads((root / "data/repair/manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((root / "data/repair" / name).read_bytes()).hexdigest() == expected, name
    stats = defaultdict(list)
    for row in rows:
        feature = render_labels(row, tokenizer, 1024)
        p = feature["prompt_tokens"]
        assert all(x == -100 for x in feature["labels"][:p])
        assert feature["labels"][p:] == feature["input_ids"][p:]
        text = tokenizer.decode(feature["input_ids"][p:], skip_special_tokens=False)
        assert text == row["messages"][-1]["content"] + "<|im_end|>\n", row["id"]
        stats[row["split"]].append(dict(id=row["id"], tokens=len(feature["input_ids"]),
                                       answer_tokens=feature["answer_tokens"]))
        if row["split"] != "train":
            prompt_rows = [json.loads(line) for line in
                           (root / f"data/repair/{row['split']}_prompts.jsonl").read_text().splitlines()]
            prompt_row = next(r for r in prompt_rows if r["id"] == row["id"])
            assert prompt_row["messages"] == row["messages"][:-1]
    summary = {split: dict(samples=len(values), total_tokens=sum(v["tokens"] for v in values),
                           supervised_tokens=sum(v["answer_tokens"] for v in values),
                           max_tokens=max(v["tokens"] for v in values))
               for split, values in stats.items()}
    write_json(root / "reports/repair_token_audit.json",
               dict(summary=summary, rows=stats,
                    checks=["artifact SHA256", "source-group and exact prompt separation",
                            "actual offline Qwen chat template", "assistant and end-marker labels",
                            "no truncation at 1024", "held-out prompt exports omit assistant"],
                    trainer_batch_verified=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
