"""
W1 · Day 1 张量热身 + 线性数据构造 + 单样本梯度对照
=====================================================
学习目标（对齐 W1 手册 Day 1 步骤 7）：
  A. 张量基础：创建 / 矩阵乘法 / 广播 / 索引 / reshape，每步都打印"输入形状 → 输出形状"。
  B. 生成 200 点线性数据 y = 3x + 2 + 噪声，固定 seed=42，按 160/20/20 划分并落盘。
  C. 画散点图（train/val/test 用不同颜色），保存到 outputs/w1_scatter.png。
  D. 取 1 个训练样本，用 autograd 算出 MSE 对 w、b 的梯度，作为"手算对照的标准答案"。

运行方式（在服务器、agi-foundation 环境下）：
    cd "$LAB_ROOT" && python scripts/w1_warmup.py

注意：这是学习脚本。请先自己在纸上/markdown 里推导 D 部分的梯度，
再运行本脚本核对，不要直接抄结果。
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 服务器无图形界面，用非交互后端，直接存文件
import matplotlib.pyplot as plt

# 让所有产物落到 LAB_ROOT 下；本地测试时回退到脚本上一级目录
LAB_ROOT = os.environ.get("LAB_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(LAB_ROOT, "data")
OUT_DIR = os.path.join(LAB_ROOT, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42


def section(title):
    print("\n" + "=" * 56)
    print(title)
    print("=" * 56)


# ============================================================
# A. 张量基础：重点是"看清每一步的形状怎么变的"
# ============================================================
def tensor_warmup():
    section("A. 张量基础（关注 shape 的变化）")

    # 1) 创建：一个 2x3 矩阵
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    print(f"a = arange(6).reshape(2,3)            -> shape {tuple(a.shape)}")
    print(a)

    # 2) 矩阵乘法：(2x3) @ (3x4) = (2x4)。内维 3 必须相等，外维决定结果形状
    b = torch.ones(3, 4)
    c = a @ b
    print(f"\n矩阵乘法 (2,3) @ (3,4)                -> shape {tuple(c.shape)}  （内维3相消，外维2和4保留）")

    # 3) 广播：(2,3) 加上 (3,) —— 后者被"复制"到每一行，无需显式扩展
    bias = torch.tensor([10.0, 20.0, 30.0])
    d = a + bias
    print(f"\n广播 (2,3) + (3,)                     -> shape {tuple(d.shape)}  （(3,) 自动对齐到每一行）")
    print(d)

    # 4) 索引 / 切片：取第 0 行、第 1 列
    print(f"\n索引 a[0]  (第0行)                    -> shape {tuple(a[0].shape)}  值={a[0].tolist()}")
    print(f"索引 a[:, 1] (第1列)                  -> shape {tuple(a[:, 1].shape)}  值={a[:, 1].tolist()}")

    # 5) reshape / view：元素总数不变(6)，重排形状
    e = a.reshape(3, 2)
    f = a.reshape(-1)  # -1 让 PyTorch 自动推断这一维 = 6
    print(f"\nreshape (2,3)->(3,2)                  -> shape {tuple(e.shape)}")
    print(f"reshape (2,3)->(-1) 拉平              -> shape {tuple(f.shape)}  （-1 自动算成 6）")

    print("\n✅ A 完成：你应该能对每一步说出'为什么是这个形状'。")


# ============================================================
# B. 生成线性数据 y = 3x + 2 + 噪声，并按 160/20/20 划分
# ============================================================
def make_data():
    section("B. 生成线性数据 + 划分 (seed=42, 160/20/20)")

    rng = np.random.default_rng(SEED)  # 固定随机源，保证可复现
    true_w, true_b = 3.0, 2.0
    n = 200

    x = rng.standard_normal(n).astype(np.float32)          # 200 个 x ~ N(0,1)
    noise = (0.5 * rng.standard_normal(n)).astype(np.float32)
    y = (true_w * x + true_b + noise).astype(np.float32)    # 真实关系 + 噪声
    print(f"x shape {x.shape}, y shape {y.shape}, 真值 w={true_w}, b={true_b}, 噪声std=0.5")

    # 打乱后切分：前160训练 / 中20验证 / 后20测试
    idx = rng.permutation(n)
    tr, va = idx[:160], idx[160:180]
    te = idx[180:]
    split = {
        "x_train": x[tr], "y_train": y[tr],
        "x_val": x[va],   "y_val": y[va],
        "x_test": x[te],  "y_test": y[te],
    }
    for k, v in split.items():
        print(f"  {k:8s} -> shape {v.shape}")

    save_path = os.path.join(DATA_DIR, "linreg_split.npz")
    np.savez(save_path, seed=SEED, true_w=true_w, true_b=true_b, **split)
    print(f"\n已保存: {save_path}")
    print("✅ B 完成：数据可复现（seed=42 已写入文件）。")
    return split


# ============================================================
# C. 散点图：train/val/test 三色
# ============================================================
def plot_scatter(split):
    section("C. 画散点图 -> outputs/w1_scatter.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(split["x_train"], split["y_train"], s=12, alpha=0.6, label="train (160)")
    plt.scatter(split["x_val"],   split["y_val"],   s=24, label="val (20)")
    plt.scatter(split["x_test"],  split["y_test"],  s=24, label="test (20)")
    # 叠加真实直线 y = 3x + 2 作参照
    xs = np.linspace(split["x_train"].min(), split["x_train"].max(), 50)
    plt.plot(xs, 3 * xs + 2, "k--", lw=1, label="真实 y=3x+2")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.title("W1 线性数据")
    out_path = os.path.join(OUT_DIR, "w1_scatter.png")
    plt.tight_layout(); plt.savefig(out_path, dpi=120)
    print(f"已保存: {out_path}")
    print("✅ C 完成：把图打开看一眼，点应大致沿虚线分布。")


# ============================================================
# D. 单样本梯度：MSE 对 w、b 的梯度（autograd 给标准答案）
# ============================================================
def grad_check_one_sample(split):
    section("D. 单样本梯度对照（先自己手算，再看这里）")

    # 取训练集第 0 个样本
    x0 = float(split["x_train"][0])
    y0 = float(split["y_train"][0])
    print(f"样本: x={x0:.4f}, y={y0:.4f}")

    # 假设当前参数（随便给的初值，方便手算核对）
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # 前向：预测 + 单样本 MSE   L = (w*x + b - y)^2
    y_hat = w * x0 + b
    loss = (y_hat - y0) ** 2
    loss.backward()

    print(f"当前 w=1.0, b=0.0 时:")
    print(f"  预测 ŷ = w*x+b = {y_hat.item():.4f}")
    print(f"  loss = (ŷ-y)^2 = {loss.item():.4f}")
    print(f"  autograd: dL/dw = {w.grad.item():.4f}")
    print(f"  autograd: dL/db = {b.grad.item():.4f}")
    print("\n📝 手算对照（自己先推，再看这两行是否一致）：")
    print("   dL/dw = 2*(ŷ-y)*x   ;   dL/db = 2*(ŷ-y)*1")
    manual_dw = 2 * (y_hat.item() - y0) * x0
    manual_db = 2 * (y_hat.item() - y0)
    print(f"   代入: dL/dw = {manual_dw:.4f} , dL/db = {manual_db:.4f}")
    ok = abs(manual_dw - w.grad.item()) < 1e-4 and abs(manual_db - b.grad.item()) < 1e-4
    print(f"\n{'✅' if ok else '❌'} 手算公式与 autograd {'一致' if ok else '不一致，检查推导'}。")


if __name__ == "__main__":
    print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
    tensor_warmup()
    split = make_data()
    plot_scatter(split)
    grad_check_one_sample(split)
    print("\n全部完成。产物: data/linreg_split.npz, outputs/w1_scatter.png")
