"""Probe actual container restrictions and timeout cleanup, without host execution."""
import json
import subprocess
from pathlib import Path

from build_repair_data import write_json
from verify_repair_data import sandbox

PROBE = """def probe():
    import os
    status = open('/proc/self/status').read()
    try:
        open('/forbidden', 'w').write('test')
        readonly = False
    except OSError:
        readonly = True
    return {
        'uid': os.getuid(),
        'readonly': readonly,
        'capabilities_dropped': 'CapEff:\\t0000000000000000' in status,
        'no_new_privs': 'NoNewPrivs:\\t1' in status,
        'interfaces': sorted(os.listdir('/sys/class/net'))
    }
"""


def main():
    image = subprocess.check_output(["docker", "image", "inspect", "python:3.11-slim",
                                     "--format", "{{.Id}}"], text=True).strip()
    results = sandbox(PROBE, "probe", [dict(args=[])], image)
    expected = dict(uid=65534, readonly=True, capabilities_dropped=True,
                    no_new_privs=True, interfaces=["lo"])
    assert results[0]["value"] == expected, results
    try:
        sandbox("def hang():\n    while True:\n        pass", "hang", [dict(args=[])], image)
        raise AssertionError("Infinite loop was not timed out")
    except subprocess.TimeoutExpired:
        timeout_ok = True
    remaining = subprocess.check_output(["docker", "ps", "-a", "--filter",
                                        "name=repair-check-", "--format", "{{.Names}}"],
                                       text=True).strip()
    assert not remaining, remaining
    write_json(Path(__file__).resolve().parents[1] / "reports/repair_sandbox_probe.json",
               dict(image_id=image, observed=expected, timeout_cleanup_passed=timeout_ok,
                    note="Corpus validation harness; adversarial model-output evaluator still requires hardening."))
    print(json.dumps(expected), "timeout cleanup passed")


if __name__ == "__main__":
    main()
