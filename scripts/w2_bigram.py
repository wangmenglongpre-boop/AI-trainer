"""W2 Day 1: Bigram character LM on names.txt."""
import os, random
from pathlib import Path
import torch

LAB_ROOT = Path(os.environ.get("LAB_ROOT", Path(__file__).resolve().parent.parent))
NAMES_PATH = LAB_ROOT / "data" / "raw" / "names.txt"
SEED = 42

def main():
    torch.manual_seed(SEED)
    chars = ["."] + [chr(ord("a") + i) for i in range(26)]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars)

    words = [w.strip() for w in NAMES_PATH.read_text().splitlines() if w.strip()]
    print(f"[load] total names: {len(words)}  first 5: {words[:5]}")

    rng = random.Random(SEED)
    shuffled = words[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    train = shuffled[:n_train]
    dev = shuffled[n_train:n_train + n_dev]
    test = shuffled[n_train + n_dev:]
    print(f"[split] train={len(train)}  dev={len(dev)}  test={len(test)}")

    N = torch.zeros((V, V), dtype=torch.int64)
    for w in train:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            N[stoi[a], stoi[b]] += 1
    print(f"[counts] total train pairs = {N.sum().item()}")

    P = (N + 1).float()
    P = P / P.sum(dim=1, keepdim=True)
    assert torch.allclose(P.sum(dim=1), torch.ones(V))

    def nll(words):
        s, k = 0.0, 0
        for w in words:
            chs = ["."] + list(w) + ["."]
            for a, b in zip(chs, chs[1:]):
                s += torch.log(P[stoi[a], stoi[b]]).item()
                k += 1
        return -s / k, k

    for name, ws in [("train", train), ("dev", dev), ("test", test)]:
        loss, k = nll(ws)
        print(f"[nll] {name:5s} = {loss:.4f}  ({k} pairs)")

    g = torch.Generator().manual_seed(SEED)
    print("[sample] 10 names from Bigram:")
    for _ in range(10):
        ix, out = 0, []
        for _ in range(20):
            ix = torch.multinomial(P[ix], 1, replacement=True, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        print("  -", "".join(out))

if __name__ == "__main__":
    main()
