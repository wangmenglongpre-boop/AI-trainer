"""W2 Day 2: Bigram as 1-layer NN with gradient descent + dual-panel NLL curve."""
import os, random
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
STEPS = 300
LR = 50.0
EVAL_EVERY = 10

def build_pairs(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            xs.append(stoi[a]); ys.append(stoi[b])
    return torch.tensor(xs), torch.tensor(ys)

def main():
    torch.manual_seed(SEED)
    chars = ["."] + [chr(ord("a") + i) for i in range(26)]
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars)

    words = [w.strip() for w in NAMES_PATH.read_text().splitlines() if w.strip()]
    rng = random.Random(SEED); rng.shuffle(words)
    n = len(words); n_tr = int(n * 0.8); n_dv = int(n * 0.1)
    train, dev, test = words[:n_tr], words[n_tr:n_tr+n_dv], words[n_tr+n_dv:]

    xs_tr, ys_tr = build_pairs(train, stoi)
    xs_dv, ys_dv = build_pairs(dev, stoi)
    xs_te, ys_te = build_pairs(test, stoi)
    print(f"[data] train pairs={len(xs_tr)}  dev={len(xs_dv)}  test={len(xs_te)}")

    g = torch.Generator().manual_seed(SEED)
    W = torch.randn((V, V), generator=g, requires_grad=True)

    def loss_on(xs, ys):
        with torch.no_grad():
            return F.cross_entropy(F.one_hot(xs, V).float() @ W, ys).item()

    steps_log, train_log, dev_log = [], [], []
    def record(step):
        steps_log.append(step)
        train_log.append(loss_on(xs_tr, ys_tr))
        dev_log.append(loss_on(xs_dv, ys_dv))

    record(0)
    print(f"[eval] step   0: train={train_log[-1]:.4f}  dev={dev_log[-1]:.4f}")
    xenc_tr = F.one_hot(xs_tr, V).float()
    for step in range(1, STEPS + 1):
        loss = F.cross_entropy(xenc_tr @ W, ys_tr)
        W.grad = None; loss.backward()
        with torch.no_grad(): W -= LR * W.grad
        if step % EVAL_EVERY == 0 or step == STEPS:
            record(step)
            if step % 50 == 0 or step == STEPS:
                print(f"[train] step {step:3d}: train={train_log[-1]:.4f}  dev={dev_log[-1]:.4f}")

    final_tr = loss_on(xs_tr, ys_tr); final_dv = loss_on(xs_dv, ys_dv); final_te = loss_on(xs_te, ys_te)
    print(f"\n[final] train = {final_tr:.4f}\n[final] dev   = {final_dv:.4f}\n[final] test  = {final_te:.4f}")
    print(f"[compare] counting bigram train ~ 2.4548")

    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13, "legend.fontsize": 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax in (ax1, ax2):
        ax.plot(steps_log, train_log, label="train NLL", color="#1f77b4", marker="o", markersize=4, linewidth=1.8)
        ax.plot(steps_log, dev_log,   label="dev NLL",   color="#ff7f0e", marker="s", markersize=4, linewidth=1.8, alpha=0.85)
        ax.axhline(2.4548, color="#2ca02c", linestyle="--", linewidth=1.5, label="counting baseline 2.4548")
        ax.set_xlabel("training step"); ax.set_ylabel("NLL (cross-entropy)")
        ax.grid(alpha=0.3); ax.legend(loc="upper right")

    ax1.axhline(3.2958, color="#d62728", linestyle=":", linewidth=1.5, label="uniform log(27)=3.30")
    ax1.set_title("(a) full view: step 0 -> 300")
    ax1.legend(loc="upper right")
    ax1.annotate(f"start = {train_log[0]:.3f}",
                 xy=(0, train_log[0]), xytext=(40, train_log[0] - 0.05),
                 fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))
    ax1.annotate(f"end = {final_tr:.3f}",
                 xy=(STEPS, final_tr), xytext=(STEPS - 120, final_tr + 0.25),
                 fontsize=10, arrowprops=dict(arrowstyle="->", color="gray"))

    tail_start = 50
    tail_mask = [i for i, s in enumerate(steps_log) if s >= tail_start]
    tail_vals = [train_log[i] for i in tail_mask] + [dev_log[i] for i in tail_mask] + [2.4548]
    y_min, y_max = min(tail_vals) - 0.005, max(tail_vals) + 0.015
    ax2.set_xlim(tail_start, STEPS); ax2.set_ylim(y_min, y_max)
    ax2.set_title(f"(b) zoom-in: step {tail_start} -> {STEPS}  (convergence detail)")
    for s, v in [(steps_log[-1], final_tr), (steps_log[-1], final_dv)]:
        ax2.annotate(f"{v:.4f}", xy=(s, v), xytext=(s - 55, v + 0.003), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    fig.suptitle(f"W2 Day 2  |  NN Bigram training  |  lr={LR}, steps={STEPS}, seed={SEED}",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    out_png = REPORTS / "w2_bigram_nn_curve.png"
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_png}  (size ~ {out_png.stat().st_size // 1024} KB)")

    g2 = torch.Generator().manual_seed(SEED)
    print("[sample] 10 names from trained NN bigram:")
    with torch.no_grad():
        for _ in range(10):
            ix, out = 0, []
            for _ in range(20):
                probs = F.softmax(F.one_hot(torch.tensor([ix]), V).float() @ W, dim=1)[0]
                ix = torch.multinomial(probs, 1, replacement=True, generator=g2).item()
                if ix == 0: break
                out.append(itos[ix])
            print("  -", "".join(out))

if __name__ == "__main__":
    main()
