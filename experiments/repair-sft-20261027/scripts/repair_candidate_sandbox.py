"""Run generated code in Docker, in a child process separate from the case driver."""
import json
import subprocess
import tempfile
import uuid

# No expected values are present inside the container. Each child receives one input.
CHILD = r"""
import contextlib, copy, json, resource, sys
resource.setrlimit(resource.RLIMIT_FSIZE, (32768, 32768))
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
p = json.loads(sys.argv[1])
args = p['args']
original = copy.deepcopy(args)
try:
    scope = {}
    # Ordinary debug prints go to bounded stderr, separate from return serialization.
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(p['code'], '<candidate>', 'exec'), scope)
        value = scope[p['function']](*args)
    result = dict(status='ok', value=value, mutated=args != original)
    output = json.dumps(result, allow_nan=False)
    if len(output.encode()) > 16384:
        raise ValueError('OutputTooLarge')
    print(output)
except BaseException as e:
    print(json.dumps(dict(status='exception', error=type(e).__name__)))
"""

DRIVER = r"""
import json, subprocess, sys, tempfile
p = json.loads(sys.argv[1])
child = sys.argv[2]
results = []
for args in p['args']:
    case = dict(code=p['code'], function=p['function'], args=args)
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            proc = subprocess.run([sys.executable, '-I', '-B', '-c', child, json.dumps(case)],
                                  stdout=out, stderr=err, timeout=3)
            out.seek(0)
            raw = out.read(32769)
            if proc.returncode or len(raw) > 32768:
                results.append(dict(status='process_error'))
                continue
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError()
            if result.get('status') == 'ok':
                if set(result) != {'status', 'value', 'mutated'} or type(result['mutated']) is not bool:
                    raise ValueError()
            elif set(result) != {'status', 'error'} or result['status'] != 'exception':
                raise ValueError()
            results.append(result)
        except subprocess.TimeoutExpired:
            results.append(dict(status='timeout'))
        except (ValueError, UnicodeError):
            results.append(dict(status='invalid_output'))
print(json.dumps(results, allow_nan=False))
"""


class InfrastructureError(RuntimeError):
    pass


def run_candidate(code, function, cases, image):
    name = "repair-eval-" + uuid.uuid4().hex
    payload = json.dumps(dict(code=code, function=function, args=[c["args"] for c in cases]))
    if len(payload.encode()) > 65536:
        raise ValueError("Candidate request exceeds input budget")
    command = ["docker", "create", "--name", name, "--network", "none",
               "--user", "65534:65534", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--cpus", "1",
               "--memory", "256m", "--memory-swap", "256m", "--pids-limit", "32",
               "--ulimit", "nofile=64:64", "--ulimit", "fsize=32768:32768",
               "--log-driver", "json-file", "--log-opt", "max-size=64k",
               "--log-opt", "max-file=1", "--tmpfs", "/tmp:rw,noexec,nosuid,size=8m",
               image, "python", "-I", "-B", "-c", DRIVER, payload, CHILD]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=15)
        subprocess.run(["docker", "start", name], check=True, capture_output=True, timeout=15)
        try:
            exit_code = subprocess.check_output(["docker", "wait", name],
                                                timeout=5 + 4 * len(cases)).strip()
        except subprocess.TimeoutExpired:
            return [dict(status="container_timeout") for _ in cases]
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            proc = subprocess.run(["docker", "logs", name], stdout=out, stderr=err, timeout=15)
            out.seek(0)
            raw = out.read(65537)
        if exit_code != b"0":
            return [dict(status="container_process_error") for _ in cases]
        if proc.returncode:
            raise InfrastructureError("Docker logs failed")
        if len(raw) > 65536:
            return [dict(status="output_limit") for _ in cases]
        try:
            values = json.loads(raw)
            if not isinstance(values, list) or len(values) != len(cases):
                raise ValueError()
            return values
        except (ValueError, UnicodeError):
            return [dict(status="invalid_driver_output") for _ in cases]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise InfrastructureError(f"Docker infrastructure failure: {type(e).__name__}") from e
    finally:
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=15)


def case_passed(result, case, no_mutation=False):
    if result.get("status") != "ok" or (no_mutation and result.get("mutated")):
        return False
    # JSON comparison avoids accepting True in place of integer 1.
    return json.dumps(result["value"], sort_keys=True, allow_nan=False) == json.dumps(
        case["expected"], sort_keys=True, allow_nan=False)
