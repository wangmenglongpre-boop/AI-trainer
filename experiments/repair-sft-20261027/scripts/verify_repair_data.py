"""Execute corpus code only in restricted Docker; publish only after all gates pass."""
import argparse
import ast
import difflib
import hashlib
import itertools
import json
import subprocess
import tempfile
import uuid
from collections import Counter
from pathlib import Path

from build_repair_data import SYSTEM, digest, write_json
from inspect_sft_data import inspect_files

ROOT = Path(__file__).resolve().parents[1]
# Only inputs enter the container; expected values and scoring stay on the host.
WORKER = """
import copy, json, sys
payload = json.load(open('/payload.json'))
scope = {}
exec(compile(payload['code'], '<candidate>', 'exec'), scope)
fn = scope[payload['function']]
results = []
for args in payload['args']:
    original = copy.deepcopy(args)
    try:
        value = fn(*args)
        results.append(dict(value=value, mutated=args != original))
    except Exception as e:
        results.append(dict(error=type(e).__name__))
print(json.dumps(results, allow_nan=False))
"""


def sandbox(code, function, cases, image):
    name = "repair-check-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as temp:
        payload = json.dumps(dict(code=code, function=function,
                                  args=[c["args"] for c in cases]))
        worker = WORKER.replace("json.load(open('/payload.json'))",
                                f"json.loads({payload!r})")
        cmd = ["docker", "create", "--name", name, "--network", "none",
               "--user", "65534:65534", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--cpus", "1",
               "--memory", "128m", "--memory-swap", "128m", "--pids-limit", "32",
               "--ulimit", "nofile=64:64", "--ulimit", "fsize=1024:1024",
               "--log-driver", "json-file", "--log-opt", "max-size=64k",
               "--log-opt", "max-file=1", "--tmpfs", "/tmp:rw,noexec,nosuid,size=8m",
               image, "python", "-I", "-B", "-c", worker]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, timeout=10)
            subprocess.run(["docker", "start", name], check=True,
                           stdout=subprocess.DEVNULL, timeout=10)
            exit_code = subprocess.check_output(["docker", "wait", name], timeout=12).strip()
            # Files avoid unbounded memory capture; RLIMIT_FSIZE also caps host outputs.
            import resource
            def limits():
                resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                result = subprocess.run(["docker", "logs", name], stdout=stdout, stderr=stderr,
                                        timeout=10, preexec_fn=limits)
                stdout.seek(0)
                stderr.seek(0)
                out, err = stdout.read(65537), stderr.read(65537)
            if exit_code != b"0" or result.returncode or len(out) > 65536:
                raise RuntimeError(f"Sandbox infrastructure/output failure: {exit_code!r} {err[:300]!r}")
            values = json.loads(out)
            if len(values) != len(cases):
                raise RuntimeError("Sandbox returned wrong number of cases")
            return values
        finally:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)


def canonical(code):
    tree = ast.parse(code)
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = names.setdefault(node.id, f"v{len(names)}")
        elif isinstance(node, ast.arg):
            node.arg = names.setdefault(node.arg, f"v{len(names)}")
        elif isinstance(node, ast.FunctionDef):
            node.name = "function"
    return ast.dump(tree, include_attributes=False)


def passed(actual, case, no_mutation):
    return ("error" not in actual and actual["value"] == case["expected"]
            and not (no_mutation and actual["mutated"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="python:3.11-slim")
    args = parser.parse_args()
    image_info = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image]))[0]
    image = image_info["Id"]
    tasks = json.loads((ROOT / "configs/repair_source_catalog.json").read_text())
    partition = json.loads((ROOT / "data/repair/source_partition.json").read_text())
    assert len(partition) == len(tasks)
    rows, evaluations, near = [], [], []
    for t, other in itertools.combinations(tasks, 2):
        if t["split"] == other["split"]:
            continue
        a, b = canonical(t["reference"]), canonical(other["reference"])
        if a == b or t["requirement"] == other["requirement"]:
            raise ValueError("Cross-split exact/canonical duplicate")
        score = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        if score >= 0.75:
            near.append(dict(left=t["family"], right=other["family"], similarity=round(score, 3)))
    for t in tasks:
        assert partition[t["source_task_id"]] == t["split"]
        function = ast.parse(t["reference"]).body[0].name
        no_mutation = "not modify" in t["requirement"]
        expected = t["cases"]
        ref = sandbox(t["reference"], function, expected, image)
        if not all(passed(a, c, no_mutation) for a, c in zip(ref, expected)):
            raise ValueError(f"Reference failed: {t['family']} {ref}")
        for i, bug in enumerate(t["bugs"], 1):
            actual = sandbox(bug["code"], function, expected, image)
            public_index = bug.get("public_case_index", 0)
            public = expected[public_index]
            observed = actual[public_index]
            assert public["visibility"] == "public"
            if passed(observed, public, no_mutation):
                raise ValueError(f"Bug lacks public witness: {t['family']} {bug['kind']}")
            sid = t["source_task_id"] + f"-bug{i}"
            failure = (f"{function}(*{public['args']!r}) "
                       + (f"raised {observed['error']}" if "error" in observed
                          else f"returned {observed['value']!r}")
                       + f"; expected {public['expected']!r}.")
            prompt = (t["requirement"] + "\nBuggy code:\n" + bug["code"]
                      + "\nPublic failure: " + failure)
            rows.append(dict(id=sid, source_task_id=t["source_task_id"], split=t["split"],
                             messages=[dict(role="system", content=SYSTEM),
                                       dict(role="user", content=prompt),
                                       dict(role="assistant", content=t["reference"])]))
            evaluations.append(dict(id=sid, bug_kind=bug["kind"], reference_passed=len(expected),
                                    bug_failed=sum(not passed(a, c, no_mutation)
                                                   for a, c in zip(actual, expected)),
                                    public_observed=observed))
        print(f"Verified {t['split']}: {t['family']}", flush=True)
    # Write held-out inference files without answers; test labels are audit-only.
    for split in ("train", "dev", "test"):
        selected = [r for r in rows if r["split"] == split]
        path = ROOT / f"data/repair/{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
        if split != "train":
            prompts = [dict(id=r["id"], source_task_id=r["source_task_id"],
                            split=split, messages=r["messages"][:-1]) for r in selected]
            (ROOT / f"data/repair/{split}_prompts.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prompts))
    inspect_files([ROOT / f"data/repair/{s}.jsonl" for s in ("train", "dev", "test")])
    write_json(ROOT / "data/repair/evaluator_cases.json",
               {t["source_task_id"]: dict(function=ast.parse(t["reference"]).body[0].name,
                                          cases=t["cases"]) for t in tasks})
    paths = list((ROOT / "data/repair").glob("*"))
    manifest = dict(version="authored-v1", provenance="Original tasks authored in this session; no external dataset.",
                    license="CC0-1.0", base_pretraining_contamination="unknown",
                    source_catalog_sha256=digest(tasks), image_id=image,
                    image_repo_digests=image_info.get("RepoDigests", []),
                    sources=dict(Counter(t["split"] for t in tasks)),
                    samples=dict(Counter(r["split"] for r in rows)),
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(paths) if p.is_file() and p.name != "manifest.json"})
    write_json(ROOT / "data/repair/manifest.json", manifest)
    write_json(ROOT / "reports/repair_data_verification.json",
               dict(manifest=manifest, reference_cases=sum(len(t["cases"]) for t in tasks),
                    evaluations=evaluations, cross_split_ast_similarity_candidates=near,
                    limitations=["Finite tests do not prove correctness.",
                                 "Authored expectations need independent human review.",
                                 "AST similarity is a heuristic; manual semantic review remains necessary.",
                                 "Do not train on dev/test or expose evaluator_cases to model prompts."]))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
