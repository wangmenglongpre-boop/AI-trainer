"""Download the pinned official checkpoint using only Python's standard library."""
import argparse
import fcntl
import hashlib
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
WEIGHT_SHA = "c1b9b30e907950516ba3c646bdf570d8084c25a6410a0cdca80cf04b11bc13a8"
WEIGHT_SIZE = 3087467144
FILES = ["LICENSE", "README.md", "config.json", "generation_config.json",
         "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json",
         "model.safetensors"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, target, expected_size=None):
    partial = target.with_name(target.name + ".partial")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = response.status == 206 and offset > 0
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected byte range")
                with partial.open("ab" if resumed else "wb") as stream:
                    total = offset if resumed else 0
                    next_print = total + 128 * 1024 * 1024
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        total += len(chunk)
                        if total >= next_print:
                            print(f"{target.name}: {total / 1e9:.2f} GB", flush=True)
                            next_print = total + 128 * 1024 * 1024
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise ValueError("Incomplete download; retry to resume")
            partial.replace(target)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def download_weight(url, target, workers):
    """Resume a verified-order prefix, download independent ranges, then append in order."""
    partial = target.with_name(target.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > WEIGHT_SIZE:
        raise ValueError("Partial file is larger than the official weight")
    chunks = target.with_name(target.name + ".chunks")
    chunks.mkdir(exist_ok=True)
    ranges = [(start, min(start + 64 * 1024 * 1024, WEIGHT_SIZE) - 1)
              for start in range(offset, WEIGHT_SIZE, 64 * 1024 * 1024)]

    def fetch(bounds):
        start, end = bounds
        path = chunks / f"{start}-{end}.part"
        size = end - start + 1
        if path.exists() and path.stat().st_size == size:
            return path
        for attempt in range(3):
            try:
                done = path.stat().st_size if path.exists() else 0
                if done == size:
                    return path
                if done > size:
                    raise ValueError("Range cache is larger than its promised size")
                request_start = start + done
                request = urllib.request.Request(url, headers={"Range": f"bytes={request_start}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected_range = f"bytes {request_start}-{end}/{WEIGHT_SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Server did not honor pinned byte range")
                    with path.open("ab" if done else "wb") as stream:
                        remaining = size - done
                        while remaining:
                            block = response.read(min(4 * 1024 * 1024, remaining))
                            if not block:
                                raise ValueError("Incomplete range download")
                            stream.write(block)
                            remaining -= len(block)
                return path
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map preserves byte-range order even though requests complete concurrently.
        with partial.open("ab") as stream:
            for path in pool.map(fetch, ranges):
                with path.open("rb") as chunk:
                    for block in iter(lambda: chunk.read(4 * 1024 * 1024), b""):
                        stream.write(block)
                stream.flush()
                path.unlink()
                print(f"{target.name}: {stream.tell() / 1e9:.2f}/{WEIGHT_SIZE / 1e9:.2f} GB", flush=True)
    if partial.stat().st_size != WEIGHT_SIZE:
        raise ValueError("Incomplete checkpoint")
    partial.replace(target)
    if not any(chunks.iterdir()):
        chunks.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".download.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another download/verification process is using this output directory")
    manifest_path = args.output / "full_model_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and (previous["model_id"] != MODEL or previous["revision"] != REVISION):
        raise ValueError("Output already belongs to a different checkpoint")
    files = {}
    for name in FILES:
        path = args.output / name
        if not path.exists():
            if args.verify_only:
                raise FileNotFoundError(path)
            print(f"Downloading {name}", flush=True)
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            if name == "model.safetensors":
                download_weight(url, path, args.workers)
            else:
                download(url, path)
        checksum = sha256(path)
        if previous and checksum != previous["files"][name]["sha256"]:
            raise ValueError(f"Checksum changed: {name}")
        if name == "model.safetensors":
            if checksum != WEIGHT_SHA or path.stat().st_size != WEIGHT_SIZE:
                raise ValueError("Weight checksum/size does not match official HF LFS metadata")
        files[name] = dict(sha256=checksum, bytes=path.stat().st_size)
    manifest = dict(model_id=MODEL, revision=REVISION, files=files,
                    weight_sha_source="Official pinned HF API siblings LFS metadata",
                    official_weight_sha256=WEIGHT_SHA)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Remove only numeric range-cache files generated by this downloader.
    cache = args.output / "model.safetensors.chunks"
    if cache.exists():
        for path in cache.iterdir():
            if path.is_file() and re.fullmatch(r"\d+-\d+\.part", path.name):
                path.unlink()
        if not any(cache.iterdir()):
            cache.rmdir()
    lock.close()
    print("PASS: complete checkpoint and SHA256 verified", flush=True)


if __name__ == "__main__":
    main()
