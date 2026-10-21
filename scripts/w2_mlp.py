"""W2 Day 3 (GPU): MLP character LM with Adam + early stopping."""
import os, random, time
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAB_ROOT = Path(os.environ.get("LAB_ROOT", Path(__file__).resolve().parent.parent))
NAMES_PATH = LAB_ROOT / "data" / "raw" / "names.txt"
REPORTS = LAB_ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42
CONTEXT = 3
EMB_DIM = 10
HIDDEN = 128
BATCH = 32
MAX_STEPS = 20000
LR = 1e-3
EVAL_EVERY = 200
PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using {DEVICE}  (torch={torch.__version__})")
if DEVICE.type == "cuda":
    print(f"[device] {torch.cuda.get_device_name(0)}  bf16={torch.cuda.is_bf16_supported()}")

def build_dataset(words, stoi, context=CONTEXT):
    X, Y = [], []
    for w in words:
        ctx = [0] * context
        for ch in list(w) + ["."]:
            ix = stoi[ch]
            X.append(ctx); Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

def main():
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    chars = ["."] + [chr(ord("a") + i) for i in range(26)]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars)

    words = [w.strip() for w in NAMES_PATH.read_text().splitlines() if w.strip()]
    rng = random.Random(SEED); rng.shuffle(words)
    n = len(words); n_tr = int(n * 0.8); n_dv = int(n * 0.1)
    train, dev, test = words[:n_tr], words[n_tr:n_tr+n_dv], words[n_tr+n_dv:]

    Xtr, Ytr = build_dataset(train, stoi); Xtr, Ytr = Xtr.to(DEVICE), Ytr.to(DEVICE)
    Xdv, Ydv = build_dataset(dev,   stoi); Xdv, Ydv = Xdv.to(DEVICE), Ydv.to(DEVICE)
    Xte, Yte = build_dataset(test,  stoi); Xte, Yte = Xte.to(DEVICE), Yte.to(DEVICE)
    print(f"[data] Xtr={tuple(Xtr.shape)}  Xdv={tuple(Xdv.shape)}  Xte={tuple(Xte.shape)}")

    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    C  = torch.randn((V, EMB_DIM), generator=g, device=DEVICE, requires_grad=True)
    W1 = torch.randn((CONTEXT*EMB_DIM, HIDDEN), generator=g, device=DEVICE)
    W1 = (W1 * (5/3) / ((CONTEXT*EMB_DIM)**0.5)).detach().requires_grad_(True)
    b1 = torch.zeros(HIDDEN, device=DEVICE, requires_grad=True)
    W2 = (torch.randn((HIDDEN, V), generator=g, device=DEVICE) * 0.01).detach().requires_grad_(True)
    b2 = torch.zeros(V, device=DEVICE, requires_grad=True)
    params = [C, W1, b1, W2, b2]
    print(f"[model] params={sum(p.numel() for p in params)}")
    opt = torch.optim.Adam(params, lr=LR)

    def forward(X):
        emb = C[X]
        h = torch.tanh(emb.view(emb.size(0), -1) @ W1 + b1)
        return h @ W2 + b2

    @torch.no_grad()
    def eval_loss(X, Y, chunk=8192):
        total, k = 0.0, 0
        for i in range(0, len(X), chunk):
            total += F.cross_entropy(forward(X[i:i+chunk]), Y[i:i+chunk], reduction="sum").item()
            k += len(Y[i:i+chunk])
        return total / k

    steps_log, train_log, dev_log = [], [], []
    best_dev, best_step, bad = float("inf"), 0, 0
    best_state = None
    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        ix = torch.randint(0, len(Xtr), (BATCH,), generator=g, device=DEVICE)
        loss = F.cross_entropy(forward(Xtr[ix]), Ytr[ix])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            tr = eval_loss(Xtr, Ytr); dv = eval_loss(Xdv, Ydv)
            steps_log.append(step); train_log.append(tr); dev_log.append(dv)
            print(f"[step {step:5d}] train={tr:.4f}  dev={dv:.4f}  (batch={loss.item():.4f})")
            if dv < best_dev - 1e-4:
                best_dev, best_step, bad = dv, step, 0
                best_state = [p.detach().clone() for p in params]
            else:
                bad += 1
                if bad >= PATIENCE:
                    print(f"[early-stop] best dev={best_dev:.4f} @ step {best_step}")
                    break
    print(f"[time] trained {step} steps in {time.time()-t0:.1f}s")

    if best_state is not None:
        with torch.no_grad():
            for p, s in zip(params, best_state): p.copy_(s)
        print(f"[restore] step {best_step}")

    print(f"\n[final] train = {eval_loss(Xtr, Ytr):.4f}")
    final_dv = eval_loss(Xdv, Ydv); final_te = eval_loss(Xte, Yte)
    print(f"[final] dev   = {final_dv:.4f}")
    print(f"[final] test  = {final_te:.4f}")
    print(f"[compare] Bigram ~ 2.45,  MLP target < 2.20")

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "legend.fontsize": 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax in (ax1, ax2):
        ax.plot(steps_log, train_log, label="train NLL", color="#1f77b4", marker="o", markersize=3)
        ax.plot(steps_log, dev_log,   label="dev NLL",   color="#ff7f0e", marker="s", markersize=3)
        ax.axhline(2.4548, color="#2ca02c", linestyle="--", linewidth=1.3, label="Bigram baseline 2.45")
        ax.set_xlabel("step"); ax.set_ylabel("NLL"); ax.grid(alpha=0.3); ax.legend(loc="upper right")
    ax1.set_title("(a) full curve")
    tail_start = max(steps_log[0], steps_log[-1] // 4)
    tail_idx = [i for i, s in enumerate(steps_log) if s >= tail_start]
    tail_vals = [train_log[i] for i in tail_idx] + [dev_log[i] for i in tail_idx]
    ax2.set_xlim(tail_start, steps_log[-1])
    ax2.set_ylim(min(tail_vals)-0.02, max(tail_vals)+0.02)
    ax2.set_title(f"(b) zoom-in  (train vs dev gap)")
    fig.suptitle(f"W2 Day 3 MLP (GPU) | ctx={CONTEXT}, emb={EMB_DIM}, hidden={HIDDEN}, batch={BATCH}, Adam lr={LR}",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = REPORTS / "w2_mlp_curve.png"
    plt.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] saved {out}")

    g2 = torch.Generator(device=DEVICE).manual_seed(SEED)
    print("[sample] 15 names from MLP:")
    with torch.no_grad():
        for _ in range(15):
            ctx = [0] * CONTEXT; out_chs = []
            for _ in range(20):
                probs = F.softmax(forward(torch.tensor([ctx], device=DEVICE)), dim=1)[0]
                ix = torch.multinomial(probs, 1, replacement=True, generator=g2).item()
                if ix == 0: break
                out_chs.append(itos[ix])
                ctx = ctx[1:] + [ix]
            print("  -", "".join(out_chs))

if __name__ == "__main__":
    main()
