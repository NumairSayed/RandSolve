import sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
BUILD = os.path.join(ROOT, "build")

sys.path.insert(0, ROOT)
sys.path.insert(0, BUILD)

if not os.path.exists(BUILD):
    raise RuntimeError("Build directory not found. Run CMake first.")

import rsvd_backend
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import random
import matplotlib.pyplot as plt

torch.manual_seed(42)
random.seed(42)

# CONFIG GRID

CONFIGS = [
    (192, 64), (192, 96), (192, 128),
    (256, 64), (256, 96), (256, 128),
    (512, 64), (512, 96), (512, 128),
]

# BACKEND RSVD

def backend_rsvd(W, r, sketch="gaussian"):
    t0 = time.time()

    U, S, V = rsvd_backend.rsvd(
        W.detach().cpu().numpy(),
        r,
        sketch
    )

    t1 = time.time()

    return (
        torch.from_numpy(U).to(W.device).float(),
        torch.from_numpy(S).to(W.device).float(),
        torch.from_numpy(V).to(W.device).float(),
        t1 - t0
    )

# GRADIENT COMPRESSION

def compress_gradient(g, r):
    if g.ndim < 2:
        return g, 0.0

    orig_shape = g.shape
    g2d = g.reshape(g.shape[0], -1)

    U, S, V, t_rsvd = backend_rsvd(g2d, r)
    g_hat = (U * S.unsqueeze(0)) @ V

    return g_hat.reshape(orig_shape), t_rsvd

# STEP SCHEDULING

def get_steps(d, L):
    scale = (d / 256) * (L / 96)

    base = 800

    steps = int(base / scale)

    return max(80, min(150, steps))

# DATASET

class CharDataset:
    def __init__(self, path, seq_len):
        text = open(path, 'r', encoding='utf-8').read()
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def get_batch(self, batch_size):
        ix = torch.randint(0, len(self.data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([self.data[i:i+self.seq_len] for i in ix])
        y = torch.stack([self.data[i+1:i+self.seq_len+1] for i in ix])
        return x, y
# LoRA

class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=2, randlora=False, sketch="gaussian"):
        super().__init__()
        self.sketch = sketch
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.A = nn.Parameter(torch.zeros(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.r = r

        if randlora:
            self.init_randlora()
        else:
            self.init_baseline()

    def init_baseline(self):
        nn.init.zeros_(self.A)
        nn.init.zeros_(self.B)

    def init_randlora(self):
        with torch.no_grad():
            U, S, V, _ = backend_rsvd(self.weight, self.r, self.sketch)
            sqrt_S = torch.sqrt(S)
            self.B.copy_(U * sqrt_S.unsqueeze(0))
            self.A.copy_(sqrt_S.unsqueeze(1) * V)

    def forward(self, x):
        return x @ self.weight.T + x @ self.A.T @ self.B.T

# MODEL

class SelfAttention(nn.Module):
    def __init__(self, d, L, r, randlora, sketch):
        super().__init__()
        self.h = d // 64

        self.q = LoRALinear(d, d, r, randlora, sketch)
        self.k = nn.Linear(d, d)
        self.v = LoRALinear(d, d, r, randlora, sketch)
        self.o = nn.Linear(d, d)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(L, L)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, L, D = x.shape

        q = self.q(x).view(B, L, self.h, 64).transpose(1, 2)
        k = self.k(x).view(B, L, self.h, 64).transpose(1, 2)
        v = self.v(x).view(B, L, self.h, 64).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(64)
        attn = attn.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o(out)

class Block(nn.Module):
    def __init__(self, d, L, r, randlora, sketch):
        super().__init__()
        self.attn = SelfAttention(d, L, r, randlora, sketch)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4*d),
            nn.GELU(),
            nn.Linear(4*d, d)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab, d, L, r=2, randlora=False, sketch="gaussian"):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(d, L, r, randlora, sketch) for _ in range(4)
        ])
        self.token = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, L, d))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        x = self.token(x) + self.pos[:, :x.shape[1], :]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

# TRAIN

def train_one(mode, d, L, data_path):
    dataset = CharDataset(data_path, L)

    use_rand = "randlora" in mode
    use_comp = "compress" in mode

    sketch = "gaussian"
    if "count" in mode:
        sketch = "count"
    elif "hadamard" in mode:
        sketch = "hadamard"

    model = Transformer(dataset.vocab_size, d, L, r=2,
                        randlora=use_rand, sketch=sketch)

    opt = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    losses, times = [], []
    start = time.time()

    total_overhead = 0.0

    for step in range(get_steps(d, L)):
        x, y = dataset.get_batch(8)

        logits = model(x)
        loss = loss_fn(
            logits.reshape(-1, dataset.vocab_size),
            y.reshape(-1)
        )

        opt.zero_grad()
        loss.backward()

        if use_comp and step % 5 == 0:
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                if ("A" in name or "B" in name):
                    if p.grad.ndim == 2 and p.grad.numel() > 1024:
                        with torch.no_grad():
                            p.grad, cost = compress_gradient(p.grad, r=1)
                            total_overhead += cost

        opt.step()

        losses.append(loss.item())
        times.append(time.time() - start)

    return losses, times, model, total_overhead

# UTILS

def moving_avg(x, w=10):
    x = torch.tensor(x)
    return torch.conv1d(
        x.view(1,1,-1),
        torch.ones(1,1,w)/w
    ).view(-1).numpy()

def time_to_target(losses, times, target):
    for l, t in zip(losses, times):
        if l <= target:
            return t
    return times[-1]

# RUN

def run(data_path):
    results = {}
    BASE = os.path.dirname(__file__)

    MODES = [
        "baseline",
        "randlora_gaussian",
        "randlora_count",
        "randlora_hadamard",
        "randlora_compress"
    ]

    for d, L in CONFIGS:
        print(f"Running {d}/{L}")

        results[(d, L)] = {}

        for mode in MODES:
            print(f" -> {mode}")

            key = mode.replace("randlora_", "")

            losses, times, model, overhead = train_one(mode, d, L, data_path)

            results[(d, L)][key] = (losses, times, overhead)

            torch.save(
                (losses, times),
                os.path.join(BASE, f"curves_{mode}_{d}_{L}.pt")
            )

            torch.save(
                model.state_dict(),
                os.path.join(BASE, f"model_{mode}_{d}_{L}.pt")
            )

    return results

# PLOT

def plot(results, save_dir):
    COLORS = {
        "baseline": "#1f77b4",
        "gaussian": "#ff7f0e",
        "count": "#2ca02c",
        "hadamard": "#d62728",
        "compress": "#9467bd"
    }

    ORDER = ["baseline", "gaussian", "count", "hadamard", "compress"]

    # LOSS SUBPLOTS

    n = len(results)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ((d, L), data) in enumerate(results.items()):
        ax = axes[i]

        for k in ORDER:
            style = '--' if k == "baseline" else '-'
            ax.plot(
                moving_avg(data[k][0]),
                style,
                color=COLORS[k],
                linewidth=1.5,
                label=k
            )

        ax.set_title(f"d={d}, L={L}")
        ax.grid(alpha=0.2)

        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_subplots.png"))
    plt.close()

    # SPEEDUP SUBPLOTS

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten()

    for i, ((d, L), data) in enumerate(results.items()):
        ax = axes[i]

        lb, tb, ob = data["baseline"]
        lg, tg, og = data["gaussian"]
        lc, tc, oc = data["count"]
        lh, th, oh = data["hadamard"]
        lcmp, tcmp, ocmp = data["compress"]

        target = min(lb[-1], lg[-1], lc[-1], lh[-1], lcmp[-1]) * 1.05

        b_adj   = max(1e-6, time_to_target(lb, tb, target) - ob)
        g_adj   = max(1e-6, time_to_target(lg, tg, target) - og)
        c_adj   = max(1e-6, time_to_target(lc, tc, target) - oc)
        h_adj   = max(1e-6, time_to_target(lh, th, target) - oh)
        cmp_adj = max(1e-6, time_to_target(lcmp, tcmp, target) - ocmp)

        # speedups
        labels = ["gaussian", "count", "hadamard", "compressed"]
        values = [
            b_adj / g_adj,
            b_adj / c_adj,
            b_adj / h_adj,
            b_adj / cmp_adj
        ]

        colors = [
            COLORS["gaussian"],
            COLORS["count"],
            COLORS["hadamard"],
            COLORS["compress"]
        ]

        y = range(len(labels))

        ax.barh(y, values, color=colors)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"d={d}, L={L}", fontsize=10)
        ax.set_xlabel("Speedup (×)")
        ax.axvline(1.0, linestyle='--', color='gray')  # baseline reference
        ax.grid(axis='x', alpha=0.3)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "runtime_speedup.png"))
    plt.close()

    print("\n===== PERFORMANCE (RUNTIME + SPEEDUP) =====")
    print(f"{'d':>5} {'L':>5} "
          f"{'baseline':>14} {'gaussian':>18} {'count':>18} {'hadamard':>18} {'compressed':>18}")

    for (d, L), data in results.items():
        lb, tb, ob = data["baseline"]
        lg, tg, og = data["gaussian"]
        lc, tc, oc = data["count"]
        lh, th, oh = data["hadamard"]
        lcmp, tcmp, ocmp = data["compress"]

        target = min(lb[-1], lg[-1], lc[-1], lh[-1], lcmp[-1]) * 1.05

        b_adj   = time_to_target(lb, tb, target) - ob
        g_adj   = time_to_target(lg, tg, target) - og
        c_adj   = time_to_target(lc, tc, target) - oc
        h_adj   = time_to_target(lh, th, target) - oh
        cmp_adj = time_to_target(lcmp, tcmp, target) - ocmp

        # speedups
        sg = b_adj / g_adj
        sc = b_adj / c_adj
        sh = b_adj / h_adj
        sp = b_adj / cmp_adj

        def fmt(t, s):
            return f"{t:6.2f} ({s:4.2f}×)"

        print(f"{d:5} {L:5} "
              f"{b_adj:14.2f} "
              f"{fmt(g_adj, sg):>18} "
              f"{fmt(c_adj, sc):>18} "
              f"{fmt(h_adj, sh):>18} "
              f"{fmt(cmp_adj, sp):>18}")

# MAIN

if __name__ == "__main__":
    BASE = os.path.dirname(__file__)
    DATA = os.path.join(BASE, "input.txt")

    results = run(DATA)
    plot(results, BASE)

    print("Done.")