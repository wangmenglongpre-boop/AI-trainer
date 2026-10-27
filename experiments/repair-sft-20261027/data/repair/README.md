# Python repair corpus — authored-v1

This small corpus is for SFT pipeline validation. It contains original tasks
authored with AI assistance in this session, not MBPP or a public benchmark.
The corpus content is offered under CC0-1.0; this notice does not change the
license of other project files.

| Split | Source tasks | Bug variants | Use |
| --- | ---: | ---: | --- |
| train | 10 | 30 | SFT and few-shot examples |
| dev | 5 | 5 | Validation loss and model selection |
| test | 5 | 5 | Final evaluation after choices are frozen |

Sources are assigned before bug expansion. No source family crosses splits.
All expectations are specified separately from reference execution. These are
finite, AI-authored tests and still benefit from independent human review.

- `train.jsonl`: the only SFT training input.
- `dev.jsonl`: labeled validation input only.
- `test.jsonl`: audit-only reference labels, never an SFT/selection input.
- `dev_prompts.jsonl`, `test_prompts.jsonl`: inference inputs with no assistant.
- `evaluator_cases.json`: scorer-owned public/hidden inputs and expectations;
  never load it into model prompts or the training dataset.
- `source_partition.json`: frozen source assignment.
- `manifest.json`: artifact hashes, counts, provenance and Docker image identity.

Do not use a glob such as `data/repair/*.jsonl` to construct training data.
Public witnesses are the only tests shown in a user message. Hidden cases are
kept in evaluator artifacts and are not appended to candidate code. For data
verification, only arguments enter a restricted Docker container; expected
values and scoring remain outside it.

Rebuild on the remote host with Docker:

```bash
python scripts/build_repair_data.py
python scripts/verify_repair_data.py
python scripts/audit_repair_tokens.py --tokenizer models/qwen2.5-coder-1.5b-tokenizer
python scripts/check_repair_sandbox.py
```

Reference pass counts establish data consistency, not model performance.
Pretraining contamination is unknown. With five test tasks, one task changes
the reported pass rate by 20 percentage points. Expand independent source
families and tests before drawing a reliable claim about SFT effectiveness.
The verification harness is intended for these inspected, authored programs;
evaluation of arbitrary model outputs needs additional output-protocol and
candidate-process hardening.
