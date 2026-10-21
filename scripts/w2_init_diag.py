"""W2 Day 4: initialization diagnostics - bad vs good init."""
import os, random
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_ROOT = Path(os.environ.get("LAB_ROOT", Path(__file__).resolve().parent.parent))
REPORTS = LAB_ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42
CONTEXT = 3; EMB_DIM = 10; HIDDEN = 128; V = 27
BATCH = 32; STEPS = 2000; LR = 0.1; EVAL_EVERY = 100
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {DEV}")

random.seed(SEED); torch.manual_seed(SEED)
words = open(LAB_ROOT / "data/raw/names.txt").read().splitlines()
chars = sorted(set("".join(words)))
stoi = {s: i+1 for i, s in enumerate(chars)}; stoi["."] = 0

def build(ws):
    X, Y = [], []
    for w in ws:
        ctx = [0]*CONTEXT
        for ch in w + ".":
            ix = stoi[ch]; X.append(ctx); Y.append(ix); ctx = ctx[1:]+[ix]
    return torch.tensor(X).to(DEV), torch.tensor(Y).to(DEV)

random.shuffle(words); n1 = int(0.8*len(words)); n2 = int(0.9*len(words))
Xtr, Ytr = build(words[:n1]); Xdv, Ydv = build(words[n1:n2])

def init(mode):
    g = torch.Generator(device=DEV).manual_seed(SEED)
    C  = torch.randn(V, EMB_DIM, generator=g, device=DEV)
    W1 = torch.randn(CONTEXT*EMB_DIM, HIDDEN, generator=g, device=DEV)
    b1 = torch.zeros(HIDDEN, device=DEV)
    W2 = torch.randn(HIDDEN, V, generator=g, device=DEV)
    b2 = torch.zeros(V, device=DEV)
    if mode == "good":
        W1 = W1 * (5/3) / ((CONTEXT*EMB_DIM)**0.5)
        W2 = W2 * 0.01
    for p in (C, W1, b1, W2, b2): p.requires_grad_(True)
    return [C, W1, b1, W2, b2]

def fwd(ps, X):
    C, W1, b1, W2, b2 = ps
    emb = C[X].view(X.shape[0], -1)
    pre = emb @ W1 + b1; h = torch.tanh(pre)
    return h @ W2 + b2, pre, h

def run(mode):
    print(f"\n=== {mode.upper()} INIT ===")
    ps = init(mode)
    with torch.no_grad():
        lg, pre, h = fwd(ps, Xtr[:1000])
        l0 = F.cross_entropy(lg, Ytr[:1000]).item()
        sat = (h.abs() > 0.99).float().mean().item() * 100
        print(f"[init] loss={l0:.3f} (ref log27=3.296)  tanh|.|>0.99 = {sat:.1f}%")
    hist = {"s": [], "dv": [], "gW1": [], "gW2": []}
    for s in range(1, STEPS+1):
        idx = torch.randint(0, Xtr.shape[0], (BATCH,), device=DEV)
        lg, _, _ = fwd(ps, Xtr[idx])
        loss = F.cross_entropy(lg, Ytr[idx])
        for p in ps: p.grad = None
        loss.backward()
        if s % EVAL_EVERY == 0 or s == 1:
            with torch.no_grad():
                dv = F.cross_entropy(fwd(ps, Xdv)[0], Ydv).item()
            hist["s"].append(s); hist["dv"].append(dv)
            hist["gW1"].append(ps[1].grad.norm().item())
            hist["gW2"].append(ps[3].grad.norm().item())
        with torch.no_grad():
            for p in ps: p -= LR * p.grad
    print(f"[final] dev={hist['dv'][-1]:.4f}")
    return hist, sat

bad_h, bad_sat = run("bad")
good_h, good_sat = run("good")

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
with torch.no_grad():
    bl, _, bh = fwd(init("bad"), Xtr[:500])
    gl, _, gh = fwd(init("good"), Xtr[:500])
ax[0,0].hist(bl.cpu().flatten().numpy(), bins=50, alpha=0.5, label=f"bad std={bl.std():.1f}")
ax[0,0].hist(gl.cpu().flatten().numpy(), bins=50, alpha=0.5, label=f"good std={gl.std():.2f}")
ax[0,0].set_title("Initial logits"); ax[0,0].legend()
ax[0,1].hist(bh.cpu().flatten().numpy(), bins=50, alpha=0.5, label=f"bad sat={bad_sat:.0f}%")
ax[0,1].hist(gh.cpu().flatten().numpy(), bins=50, alpha=0.5, label=f"good sat={good_sat:.0f}%")
ax[0,1].set_title("Initial tanh"); ax[0,1].legend()
ax[1,0].plot(bad_h["s"], bad_h["dv"], label="bad"); ax[1,0].plot(good_h["s"], good_h["dv"], label="good")
ax[1,0].set_title("Dev loss"); ax[1,0].legend(); ax[1,0].set_xlabel("step")
ax[1,1].plot(bad_h["s"], bad_h["gW1"], label="bad W1"); ax[1,1].plot(good_h["s"], good_h["gW1"], label="good W1")
ax[1,1].plot(bad_h["s"], bad_h["gW2"], "--", label="bad W2"); ax[1,1].plot(good_h["s"], good_h["gW2"], "--", label="good W2")
ax[1,1].set_yscale("log"); ax[1,1].set_title("Gradient norms"); ax[1,1].legend()
plt.tight_layout(); plt.savefig(REPORTS / "w2_init_diag.png", dpi=140)
print(f"[plot] {REPORTS/'w2_init_diag.png'}")
