"""W2 Day 5: temperature sampling + Bigram vs MLP on the same split, vocab and targets."""
import os, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_ROOT = Path(os.environ.get("LAB_ROOT", Path(__file__).resolve().parent.parent))
REPORTS = LAB_ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)
OUTPUTS = LAB_ROOT / "outputs"; OUTPUTS.mkdir(parents=True, exist_ok=True)

SEED = 42
CONTEXT = 3; EMB_DIM = 10; HIDDEN = 128; V = 27
BATCH = 32; LR = 1e-3; MAX_STEPS = 20000; EVAL_EVERY = 200; PATIENCE = 10
PROMPTS = ["", "a", "ma", "jo", "kay"]   # "" = start from the boundary only
TEMPS = [0.7, 1.0]
N_PER_PROMPT = 10; MAX_NEW = 20
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {DEV}")

# ---- data: same split / vocab as Day 1-4 ----
words = [w.strip() for w in (LAB_ROOT / "data/raw/names.txt").read_text().splitlines() if w.strip()]
print(f"[data] names={len(words)}  unique={len(set(words))}")
random.Random(SEED).shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
train, dev, test = words[:n1], words[n1:n2], words[n2:]
train_set = set(train)
itos = ["."] + [chr(ord("a") + i) for i in range(26)]
stoi = {c: i for i, c in enumerate(itos)}

def build(ws):
    X, Y = [], []
    for w in ws:
        ctx = [0] * CONTEXT
        for ch in w + ".":
            ix = stoi[ch]; X.append(ctx); Y.append(ix); ctx = ctx[1:] + [ix]
    return torch.tensor(X, device=DEV), torch.tensor(Y, device=DEV)

Xtr, Ytr = build(train); Xdv, Ydv = build(dev); Xte, Yte = build(test)
print(f"[split] names = {len(train)}/{len(dev)}/{len(test)}  targets = {len(Ytr)}/{len(Ydv)}/{len(Yte)}")

# ---- Bigram: last context column = previous char, so (X[:, -1], Y) are exactly the bigram pairs ----
N = torch.zeros(V, V, device=DEV)
N.index_put_((Xtr[:, -1], Ytr), torch.ones(len(Ytr), device=DEV), accumulate=True)
P_bi = (N + 1) / (N + 1).sum(1, keepdim=True)
assert torch.allclose(P_bi.sum(1), torch.ones(V, device=DEV))
logP_bi = P_bi.log()

# ---- MLP: Day 3 recipe (good init + Adam + early stopping) ----
torch.manual_seed(SEED)
C = torch.randn(V, EMB_DIM, device=DEV)
W1 = torch.randn(CONTEXT * EMB_DIM, HIDDEN, device=DEV) * (5 / 3) / (CONTEXT * EMB_DIM) ** 0.5
b1 = torch.zeros(HIDDEN, device=DEV)
W2 = torch.randn(HIDDEN, V, device=DEV) * 0.01
b2 = torch.zeros(V, device=DEV)
params = [C, W1, b1, W2, b2]
for p in params: p.requires_grad_(True)

def mlp_logits(X):
    h = torch.tanh(C[X].view(X.shape[0], -1) @ W1 + b1)
    return h @ W2 + b2

@torch.no_grad()
def logits_of(model, X):
    return logP_bi[X[:, -1]] if model == "bigram" else mlp_logits(X)

@torch.no_grad()
def nll(model, X, Y, T=1.0):
    return F.cross_entropy(logits_of(model, X) / T, Y).item()

@torch.no_grad()
def entropy(model, X, T):
    p = F.softmax(logits_of(model, X) / T, dim=-1)
    return -(p * p.clamp_min(1e-12).log()).sum(-1).mean().item()

opt = torch.optim.Adam(params, lr=LR)
best, best_step, wait, best_state = float("inf"), 0, 0, None
t0 = time.time()
for step in range(1, MAX_STEPS + 1):
    idx = torch.randint(0, Xtr.shape[0], (BATCH,), device=DEV)
    loss = F.cross_entropy(mlp_logits(Xtr[idx]), Ytr[idx])
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % EVAL_EVERY == 0:
        dv = nll("mlp", Xdv, Ydv)
        if dv < best:
            best, best_step, wait = dv, step, 0
            best_state = [p.detach().clone() for p in params]
        else:
            wait += 1
            if wait >= PATIENCE: break
with torch.no_grad():
    for p, s in zip(params, best_state): p.copy_(s)
print(f"[mlp] stop @ {step}, restored best dev={best:.4f} @ {best_step}  ({time.time() - t0:.1f}s)")
torch.save({"params": [p.detach().cpu() for p in params], "itos": itos,
            "config": dict(CONTEXT=CONTEXT, EMB_DIM=EMB_DIM, HIDDEN=HIDDEN, SEED=SEED, best_step=best_step)},
           OUTPUTS / "w2_mlp_h128.pt")

# ---- 1) NLL under the same targets (incl. end boundary) ----
splits = {"train": (Xtr, Ytr), "dev": (Xdv, Ydv), "test": (Xte, Yte)}
res = {m: {s: nll(m, X, Y) for s, (X, Y) in splits.items()} for m in ("bigram", "mlp")}
for m, r in res.items():
    print(f"[nll] {m:6s} train={r['train']:.4f}  dev={r['dev']:.4f}  test={r['test']:.4f}")

# ---- 2) temperature sampling: weights frozen, same prompts and seed ----
@torch.no_grad()
def sample(model, prompt, T, g):
    ctx = [0] * CONTEXT
    for ch in prompt: ctx = ctx[1:] + [stoi[ch]]
    out = prompt
    for _ in range(MAX_NEW):
        p = F.softmax(logits_of(model, torch.tensor([ctx], device=DEV))[0] / T, dim=-1).cpu()
        ix = torch.multinomial(p, 1, generator=g).item()
        if ix == 0: return out, False
        out += itos[ix]; ctx = ctx[1:] + [ix]
    return out, True   # hit MAX_NEW without the end boundary

rows, samples = [], {}
for m in ("bigram", "mlp"):
    for T in TEMPS:
        g = torch.Generator().manual_seed(SEED)   # same random stream for every (model, T)
        outs = []
        for pr in PROMPTS:
            got = [sample(m, pr, T, g) for _ in range(N_PER_PROMPT)]
            samples[(m, T, pr)] = [n for n, _ in got]
            outs += got
        ns = [n for n, _ in outs]
        rows.append(dict(model=m, T=T, dev_nll=nll(m, Xdv, Ydv, T), entropy=entropy(m, Xdv, T),
                         unique=len(set(ns)) / len(ns), in_train=sum(n in train_set for n in ns) / len(ns),
                         avg_len=sum(map(len, ns)) / len(ns), trunc=sum(t for _, t in outs) / len(outs)))

print("\nmodel   T    devNLL@T  entropy  unique  in_train  avg_len  trunc")
for r in rows:
    print(f"{r['model']:6s} {r['T']:.1f}  {r['dev_nll']:.4f}    {r['entropy']:.3f}    {r['unique']:.2f}    {r['in_train']:.2f}      {r['avg_len']:.2f}     {r['trunc']:.2f}")
for pr in PROMPTS:
    print(f"\n[prompt {pr!r}]")
    for m in ("bigram", "mlp"):
        for T in TEMPS:
            print(f"  {m:6s} T={T}: " + ", ".join(samples[(m, T, pr)][:6]))

# ---- 3) plot: next-char distribution after "ma" + dev NLL / entropy per (model, T) ----
with torch.no_grad():
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ctx = torch.tensor([[0, stoi["m"], stoi["a"]]], device=DEV)
    xs = torch.arange(V).float()
    for k, T in enumerate(TEMPS):
        p = F.softmax(mlp_logits(ctx)[0] / T, dim=-1).cpu()
        ax[0].bar((xs + (k - 0.5) * 0.4).numpy(), p.numpy(), width=0.4, label=f"T={T}")
    ax[0].set_xticks(range(V)); ax[0].set_xticklabels(itos)
    ax[0].set_title("MLP next-char probs after prompt 'ma'"); ax[0].legend()
    ax[1].bar(range(len(rows)), [r["dev_nll"] for r in rows],
              color=["C0" if r["model"] == "bigram" else "C1" for r in rows])
    for i, r in enumerate(rows):
        ax[1].text(i, r["dev_nll"] + 0.02, f"H={r['entropy']:.2f}", ha="center")
    ax[1].set_xticks(range(len(rows))); ax[1].set_xticklabels([f"{r['model']}\nT={r['T']}" for r in rows])
    ax[1].set_ylim(0, max(r["dev_nll"] for r in rows) * 1.15)
    ax[1].set_ylabel("dev NLL"); ax[1].set_title("Dev NLL under temperature (H = mean entropy, nats)")
    plt.tight_layout(); plt.savefig(REPORTS / "w2_day5_temperature.png", dpi=150)
print(f"\n[plot] {REPORTS / 'w2_day5_temperature.png'}")

# ---- 4) report ----
lines = ["# W2 Day 5: temperature sampling + Bigram vs MLP", "",
         f"- split seed={SEED}, names train/dev/test = {len(train)}/{len(dev)}/{len(test)}; same vocab and targets (incl. end boundary)",
         f"- MLP: context={CONTEXT}, emb={EMB_DIM}, hidden={HIDDEN}, Adam lr={LR}, batch={BATCH}, best step={best_step}",
         f"- sampling: prompts={PROMPTS}, {N_PER_PROMPT} names per prompt, max {MAX_NEW} new chars, seed={SEED}",
         "", "| model | train | dev | test |", "|---|---|---|---|"]
lines += [f"| {m} | {r['train']:.4f} | {r['dev']:.4f} | {r['test']:.4f} |" for m, r in res.items()]
lines += ["", "| model | T | dev NLL@T | entropy | unique | in_train | avg_len | truncated |",
          "|---|---|---|---|---|---|---|---|"]
lines += [f"| {r['model']} | {r['T']} | {r['dev_nll']:.4f} | {r['entropy']:.3f} | {r['unique']:.2f} | "
          f"{r['in_train']:.2f} | {r['avg_len']:.2f} | {r['trunc']:.2f} |" for r in rows]
lines += ["", "## Samples (first 6 per prompt)", ""]
for pr in PROMPTS:
    for m in ("bigram", "mlp"):
        for T in TEMPS:
            lines.append(f"- prompt={pr!r} {m} T={T}: " + ", ".join(samples[(m, T, pr)][:6]))
(REPORTS / "w2_day5_report.md").write_text("\n".join(lines) + "\n")
print(f"[report] {REPORTS / 'w2_day5_report.md'}")
