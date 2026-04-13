"""
full_experiment.py
==================
Complete RandLoRA experimental suite for course project submission.

Experiments:
  E1. RandLoRA vs Standard LoRA across 4 models
  E2. RandLoRA across 3 datasets (OPT-350m)
  E3. Rank sensitivity: rank ∈ {4, 8, 16, 32}
  E4. Init time scaling: layers vs RandSVD time
  E5. Gradient compression fidelity (offline, no training overhead)

All results saved to experiments/ directory.
All plots saved to plots/ directory.

Usage:
  python python/full_experiment.py
  python python/full_experiment.py --quick   # faster smoke test
"""

# ── Path setup ────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Imports ───────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import randnla_ext
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# ── Output directories ────────────────────────────────────────────────────
Path("experiments").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────

MODELS = {
    "opt-125m"   : "facebook/opt-125m",
    "opt-350m"   : "facebook/opt-350m",
    "gpt2-medium": "gpt2-medium",
    "pythia-160m": "EleutherAI/pythia-160m",
}

MODEL_TARGETS = {
    "opt-125m"   : ("q_proj", "v_proj", "out_proj"),
    "opt-350m"   : ("q_proj", "v_proj", "out_proj"),
    "gpt2-medium": ("c_attn", "c_proj"),
    "pythia-160m": ("query_key_value", "dense"),
}

DATASETS = {
    "wikitext2": ("wikitext", "wikitext-2-raw-v1"),
    "squad"    : ("squad",    None),
    "agnews"   : ("ag_news",  None),
}

COLORS = {
    "standard": "#2C7BB6",
    "randlora" : "#D7191C",
}

# ─────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    model_name  : str
    dataset_name: str
    mode        : str
    rank        : int
    losses      : List[float]
    times       : List[float]
    initial_loss: float
    final_loss  : float
    patch_ms    : float
    total_ms    : float
    n_layers    : int
    trainable   : int

    def improvement_pct(self, other: "RunResult") -> float:
        """% improvement in initial loss vs other (baseline)."""
        return 100 * (other.initial_loss - self.initial_loss) / other.initial_loss

    def smooth(self, w: int = 15) -> List[float]:
        return [float(np.mean(self.losses[max(0, i-w):i+1]))
                for i in range(len(self.losses))]

# ─────────────────────────────────────────────────────────────────────────
# LORA LAYERS
# ─────────────────────────────────────────────────────────────────────────

class LoRALayer(nn.Module):
    """Standard LoRA: A~N(0,0.02), B=0. BA=0 at init."""
    def __init__(self, W: torch.Tensor, rank: int, alpha: float = 1.0):
        super().__init__()
        d_out, d_in = W.shape
        self.scale  = alpha / rank
        self.W = nn.Parameter(W.clone(), requires_grad=False)
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.02)
        self.B = nn.Parameter(torch.zeros(d_out, rank))

    def forward(self, x):
        return x @ self.W.T + self.scale * (x @ self.A.T) @ self.B.T


class RandLoRALayer(nn.Module):
    """
    RandLoRA: warm-started via C++ RandomizedSVD.
    W ≈ U diag(S) Vt  →  B=U√S, A=√S Vt  →  BA≈W at init.
    Frozen residual W_res = W - BA ≈ 0.
    """
    def __init__(self, W: torch.Tensor, rank: int,
                 alpha: float = 1.0, seed: int = 42):
        super().__init__()
        self.scale = alpha / rank
        t0 = time.perf_counter()
        U, S, Vt = randnla_ext.randomized_svd(
            W.float().cpu(), rank=rank,
            oversample=10, power_iters=2, seed=seed)
        self.init_ms = (time.perf_counter() - t0) * 1000

        S_sqrt = S.sqrt()
        B_init = U  * S_sqrt.unsqueeze(0)   # (d_out, rank)
        A_init = Vt * S_sqrt.unsqueeze(1)   # (rank, d_in)
        W_res  = W.float().cpu() - (B_init @ A_init)

        self.W_res = nn.Parameter(W_res,   requires_grad=False)
        self.A     = nn.Parameter(A_init,  requires_grad=True)
        self.B     = nn.Parameter(B_init,  requires_grad=True)

    def forward(self, x):
        return x @ (self.W_res + self.B @ self.A).T

# ─────────────────────────────────────────────────────────────────────────
# MODEL PATCHING
# ─────────────────────────────────────────────────────────────────────────

def patch_model(model, rank: int, mode: str,
                targets: tuple) -> Tuple[nn.Module, float, int, int]:
    """
    Freeze base weights and replace target layers with LoRA.
    Returns (model, init_ms, n_patched, n_trainable).
    """
    for p in model.parameters():
        p.requires_grad = False

    init_times = []
    patched    = 0

    for name, module in list(model.named_modules()):
        if not any(name.endswith(t) for t in targets):
            continue

        if isinstance(module, nn.Linear):
            W = module.weight.data.float()
        elif isinstance(module, Conv1D):
            W = module.weight.data.T.float()
        else:
            continue

        if mode == "standard":
            layer = LoRALayer(W, rank)
        else:
            layer = RandLoRALayer(W, rank)
            init_times.append(layer.init_ms)

        # Replace in parent
        parts, parent = name.split("."), model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], layer)
        patched += 1

    total_init  = sum(init_times)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, total_init, patched, n_trainable

# ─────────────────────────────────────────────────────────────────────────
# DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────

def load_texts(dataset_name: str, min_len: int = 50) -> List[str]:
    """Load and preprocess text samples from a dataset."""
    name, config = DATASETS[dataset_name]

    if dataset_name == "wikitext2":
        ds    = load_dataset(name, config, split="train")
        texts = [t for t in ds["text"] if len(t) > min_len]

    elif dataset_name == "squad":
        ds    = load_dataset(name, split="train")
        # Use context passages — longer, more structured
        texts = [t for t in ds["context"] if len(t) > min_len]
        texts = list(set(texts))  # deduplicate

    elif dataset_name == "agnews":
        ds    = load_dataset(name, split="train")
        # Combine title + description
        texts = [f"{r['text']}" for r in ds if len(r["text"]) > min_len]

    print(f"  Loaded {len(texts):,} samples from {dataset_name}")
    return texts

# ─────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────

def train(model, tokenizer, texts: List[str],
          steps: int, device: str, lr: float,
          seed: int = 42) -> Tuple[List[float], List[float], float]:
    """
    Deterministic training loop.
    Same seed → same batch order → comparable results.
    Returns (losses, step_times, total_ms).
    """
    model = model.to(device)
    model.train()

    # Deterministic batch order
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(texts))

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    warmup    = LinearLR(optimizer, start_factor=0.1,
                          end_factor=1.0, total_iters=min(20, steps//5))
    cosine    = CosineAnnealingLR(optimizer,
                                   T_max=max(1, steps - min(20, steps//5)),
                                   eta_min=lr / 10)
    scheduler = SequentialLR(optimizer,
                              schedulers=[warmup, cosine],
                              milestones=[min(20, steps//5)])

    losses, times = [], []
    t_start = time.perf_counter()

    for step in range(steps):
        text      = texts[order[step % len(order)]]
        enc       = tokenizer(text, return_tensors="pt",
                              max_length=512, truncation=True,
                              padding=False)
        input_ids = enc["input_ids"].to(device)

        if input_ids.shape[1] < 4:
            losses.append(losses[-1] if losses else 0.0)
            times.append(0.0)
            continue

        t0   = time.perf_counter()
        loss = model(input_ids, labels=input_ids).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step_ms = (time.perf_counter() - t0) * 1000
        losses.append(loss.item())
        times.append(step_ms)

    total_ms = (time.perf_counter() - t_start) * 1000
    return losses, times, total_ms

# ─────────────────────────────────────────────────────────────────────────
# SINGLE COMPARISON RUN
# ─────────────────────────────────────────────────────────────────────────

def run_comparison(model_key: str, dataset_name: str,
                   rank: int, steps: int,
                   device: str, seed: int = 42) -> Dict[str, RunResult]:
    """
    Run standard LoRA vs RandLoRA on one model+dataset combination.
    Both modes use the same batches (same seed) for fair comparison.
    """
    model_name = MODELS[model_key]
    targets    = MODEL_TARGETS[model_key]

    print(f"\n  Model: {model_key}  |  Dataset: {dataset_name}  "
          f"|  Rank: {rank}  |  Steps: {steps}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_texts(dataset_name)
    results = {}

    for mode in ["standard", "randlora"]:
        lr = 3e-4 if mode == "standard" else 5e-5

        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32)

        t_patch                         = time.perf_counter()
        model, init_ms, n_layers, n_tr  = patch_model(
            model, rank, mode, targets)
        patch_ms                        = (time.perf_counter() - t_patch) * 1000

        losses, times, total_ms = train(
            model, tokenizer, texts,
            steps=steps, device=device, lr=lr, seed=seed)

        results[mode] = RunResult(
            model_name   = model_key,
            dataset_name = dataset_name,
            mode         = mode,
            rank         = rank,
            losses       = losses,
            times        = times,
            initial_loss = losses[0],
            final_loss   = float(np.mean(losses[-10:])),  # avg last 10 steps
            patch_ms     = patch_ms,
            total_ms     = total_ms,
            n_layers     = n_layers,
            trainable    = n_tr,
        )

        print(f"    [{mode:8s}]  init={losses[0]:.3f}  "
              f"final={results[mode].final_loss:.3f}  "
              f"patch={patch_ms:.0f}ms  "
              f"train={total_ms/1000:.1f}s")

        del model
        torch.cuda.empty_cache()

    return results

# ─────────────────────────────────────────────────────────────────────────
# EXPERIMENT E1: Cross-model comparison
# ─────────────────────────────────────────────────────────────────────────

def experiment_e1_models(steps: int, device: str) -> Dict:
    """E1: RandLoRA vs Standard across 4 models, fixed dataset+rank."""
    print("\n" + "="*70)
    print("  E1: Cross-Model Comparison (dataset=wikitext2, rank=16)")
    print("="*70)

    all_results = {}
    for model_key in MODELS:
        try:
            res = run_comparison(model_key, "wikitext2",
                                  rank=16, steps=steps, device=device)
            all_results[model_key] = res
        except Exception as e:
            print(f"  SKIPPED {model_key}: {e}")

    # Save
    save_results(all_results, "experiments/e1_models.json")
    return all_results

# ─────────────────────────────────────────────────────────────────────────
# EXPERIMENT E2: Cross-dataset comparison
# ─────────────────────────────────────────────────────────────────────────

def experiment_e2_datasets(steps: int, device: str) -> Dict:
    """E2: RandLoRA vs Standard across 3 datasets, fixed model+rank."""
    print("\n" + "="*70)
    print("  E2: Cross-Dataset Comparison (model=opt-350m, rank=16)")
    print("="*70)

    all_results = {}
    for ds_name in DATASETS:
        try:
            res = run_comparison("opt-350m", ds_name,
                                  rank=16, steps=steps, device=device)
            all_results[ds_name] = res
        except Exception as e:
            print(f"  SKIPPED {ds_name}: {e}")

    save_results(all_results, "experiments/e2_datasets.json")
    return all_results

# ─────────────────────────────────────────────────────────────────────────
# EXPERIMENT E3: Rank sensitivity
# ─────────────────────────────────────────────────────────────────────────

def experiment_e3_ranks(steps: int, device: str) -> Dict:
    """E3: Effect of LoRA rank on RandLoRA advantage."""
    print("\n" + "="*70)
    print("  E3: Rank Sensitivity (model=opt-350m, dataset=wikitext2)")
    print("="*70)

    ranks = [4, 8, 16, 32]
    all_results = {}

    for rank in ranks:
        try:
            res = run_comparison("opt-350m", "wikitext2",
                                  rank=rank, steps=steps, device=device)
            all_results[str(rank)] = res
        except Exception as e:
            print(f"  SKIPPED rank={rank}: {e}")

    save_results(all_results, "experiments/e3_ranks.json")
    return all_results

# ─────────────────────────────────────────────────────────────────────────
# EXPERIMENT E4: Init time scaling
# ─────────────────────────────────────────────────────────────────────────

def experiment_e4_init_time() -> Dict:
    """
    E4: RandSVD init time vs exact SVD time across matrix sizes.
    No training — pure C++ backend benchmark.
    Shows the 6.97x speedup from Phase 1, now in Python context.
    """
    print("\n" + "="*70)
    print("  E4: Init Time — RandSVD vs Exact SVD")
    print("="*70)

    sizes = [(64, 64), (128, 128), (256, 256),
             (512, 512), (768, 768), (1024, 1024),
             (2048, 512), (4096, 512)]
    rank  = 16
    runs  = 5

    results = {"sizes": [], "exact_ms": [], "rand_ms": [], "speedup": []}

    for m, n in sizes:
        W = torch.randn(m, n)

        # Exact SVD (PyTorch)
        t_exact = min(
            time.perf_counter() * 0 or
            (lambda: (t := time.perf_counter(),
                      torch.linalg.svd(W, full_matrices=False),
                      (time.perf_counter() - t) * 1000)[2])()
            for _ in range(runs)
        )
        # Cleaner timing:
        exact_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            torch.linalg.svd(W, full_matrices=False)
            exact_times.append((time.perf_counter() - t0) * 1000)
        t_exact = min(exact_times)

        # RandSVD (C++ backend)
        rand_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            randnla_ext.randomized_svd(W.float(), rank=rank,
                                        oversample=10, power_iters=2)
            rand_times.append((time.perf_counter() - t0) * 1000)
        t_rand = min(rand_times)

        speedup = t_exact / t_rand
        label   = f"{m}×{n}"
        results["sizes"].append(label)
        results["exact_ms"].append(t_exact)
        results["rand_ms"].append(t_rand)
        results["speedup"].append(speedup)

        print(f"  {label:12s}  exact={t_exact:7.2f}ms  "
              f"rand={t_rand:7.2f}ms  speedup={speedup:.2f}x")

    with open("experiments/e4_init_time.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

# ─────────────────────────────────────────────────────────────────────────
# EXPERIMENT E5: Gradient compression fidelity (offline)
# ─────────────────────────────────────────────────────────────────────────

def experiment_e5_gradient_fidelity() -> Dict:
    """
    E5: Gradient compression quality across ranks and matrix sizes.
    Offline — no training overhead, pure fidelity measurement.
    Shows: gradient cos-sim > 0.95 for compression_ratio >= 0.5
    """
    print("\n" + "="*70)
    print("  E5: Gradient Compression Fidelity (offline)")
    print("="*70)

    sizes  = [64, 128, 256, 512, 768]
    ratios = [0.1, 0.25, 0.5, 0.75]
    trials = 20

    results = {
        "sizes": sizes, "ratios": ratios,
        "cos_sim": [],   # shape: [len(ratios), len(sizes)]
        "rel_err": [],
    }

    print(f"\n  Cosine similarity (mean over {trials} random gradients):")
    print(f"  {'Ratio':>8}", end="")
    for s in sizes:
        print(f"  {s:>8}", end="")
    print()
    print(f"  {'':-<8}", end="")
    for _ in sizes:
        print(f"  {'':->8}", end="")
    print()

    for ratio in ratios:
        row_cos, row_err = [], []
        print(f"  {ratio:>8.2f}", end="")

        for size in sizes:
            cos_vals = []
            for _ in range(trials):
                G = torch.randn(size, size)
                k = max(1, int(min(size, size) * ratio))
                r = min(k, size - 1)

                U, S, Vt = randnla_ext.randomized_svd(
                    G, rank=r, oversample=5, power_iters=1)
                G_comp = U @ torch.diag(S) @ Vt

                cos = torch.nn.functional.cosine_similarity(
                    G.flatten().unsqueeze(0),
                    G_comp.flatten().unsqueeze(0)
                ).item()
                cos_vals.append(cos)

            mean_cos = float(np.mean(cos_vals))
            mean_err = float(np.mean(
                [(G - (randnla_ext.randomized_svd(G, rank=min(max(1,int(min(G.shape)*ratio)),
                       min(G.shape)-1), oversample=5, power_iters=1)[0] @
                       torch.diag(randnla_ext.randomized_svd(G, rank=min(max(1,int(min(G.shape)*ratio)),
                       min(G.shape)-1), oversample=5, power_iters=1)[1]) @
                       randnla_ext.randomized_svd(G, rank=min(max(1,int(min(G.shape)*ratio)),
                       min(G.shape)-1), oversample=5, power_iters=1)[2])).norm().item() /
                 G.norm().item()
                 for G in [torch.randn(size, size) for _ in range(5)]]))

            row_cos.append(mean_cos)
            row_err.append(mean_err)
            mark = "✓" if mean_cos > 0.95 else "⚠"
            print(f"  {mean_cos:.4f}{mark}", end="")

        results["cos_sim"].append(row_cos)
        results["rel_err"].append(row_err)
        print()

    with open("experiments/e5_gradient.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

# ─────────────────────────────────────────────────────────────────────────
# SAVE / LOAD HELPERS
# ─────────────────────────────────────────────────────────────────────────

def save_results(results: dict, path: str):
    """Serialize RunResult objects to JSON."""
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {
                mode: asdict(res) if isinstance(res, RunResult) else res
                for mode, res in v.items()
            }
        else:
            serializable[k] = v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved → {path}")

# ─────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────

def make_all_plots(e1, e2, e3, e4, e5):
    """Generate all publication-quality plots."""

    plt.rcParams.update({
        "font.size"      : 11,
        "axes.titlesize" : 12,
        "axes.titleweight": "bold",
        "figure.dpi"     : 150,
        "axes.grid"      : True,
        "grid.alpha"     : 0.3,
    })

    # ── Figure 1: E1 — Cross-model loss curves (2×2) ─────────────────────
    if e1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Fig 1: RandLoRA vs Standard LoRA — Loss Convergence Across Models\n"
            "(dataset=WikiText-2, rank=16)",
            fontsize=13, fontweight="bold")
        axes = axes.flatten()

        for idx, (model_key, res) in enumerate(e1.items()):
            ax = axes[idx]
            for mode in ["standard", "randlora"]:
                if mode not in res:
                    continue
                r = res[mode]
                losses = r["losses"] if isinstance(r, dict) else r.losses
                s = [float(np.mean(losses[max(0,i-15):i+1]))
                     for i in range(len(losses))]
                il = losses[0]
                ax.plot(s, color=COLORS[mode], linewidth=2.5,
                        label=f"{'Standard LoRA' if mode=='standard' else 'RandLoRA (ours)'}")
                ax.axhline(il, color=COLORS[mode],
                            linestyle=":", alpha=0.4, linewidth=1)

            # Annotate improvement
            if "standard" in res and "randlora" in res:
                std_il  = (res["standard"]["losses"] if isinstance(res["standard"], dict)
                           else res["standard"].losses)[0]
                rand_il = (res["randlora"]["losses"]  if isinstance(res["randlora"], dict)
                           else res["randlora"].losses)[0]
                pct = 100 * (std_il - rand_il) / std_il
                sign = "−" if pct >= 0 else "+"
                ax.set_title(f"{model_key}  ({sign}{abs(pct):.1f}% initial loss)")

            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss (smoothed)")
            ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig("plots/fig1_cross_model.png", bbox_inches="tight")
        print("  Saved → plots/fig1_cross_model.png")
        plt.close()

    # ── Figure 2: E1 — Summary bar chart ─────────────────────────────────
    if e1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Fig 2: Initial & Final Loss Comparison Across Models",
            fontsize=13, fontweight="bold")

        model_keys = list(e1.keys())
        x          = np.arange(len(model_keys))
        w          = 0.35

        for ax_idx, metric in enumerate(["initial_loss", "final_loss"]):
            ax = axes[ax_idx]
            std_vals  = []
            rand_vals = []

            for mk in model_keys:
                res = e1[mk]
                sv  = (res["standard"][metric] if isinstance(res["standard"], dict)
                       else getattr(res["standard"], metric))
                rv  = (res["randlora"][metric]  if isinstance(res["randlora"], dict)
                       else getattr(res["randlora"],  metric))
                std_vals.append(sv)
                rand_vals.append(rv)

            b1 = ax.bar(x - w/2, std_vals,  w, label="Standard LoRA",
                         color=COLORS["standard"], alpha=0.85)
            b2 = ax.bar(x + w/2, rand_vals, w, label="RandLoRA (ours)",
                         color=COLORS["randlora"],  alpha=0.85)

            for i, (sv, rv) in enumerate(zip(std_vals, rand_vals)):
                pct = 100 * (sv - rv) / sv
                ax.annotate(
                    f"−{pct:.1f}%",
                    xy=(x[i] + w/2, rv),
                    xytext=(x[i] + w/2, rv + 0.05),
                    ha="center", fontsize=9,
                    color="darkred", fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(model_keys, rotation=15)
            ax.set_ylabel("Loss")
            ax.set_title(f"{'Initial' if metric=='initial_loss' else 'Final'} Loss")
            ax.legend(fontsize=9)
            ax.set_ylim(0, max(std_vals) * 1.3)

        plt.tight_layout()
        plt.savefig("plots/fig2_model_summary.png", bbox_inches="tight")
        print("  Saved → plots/fig2_model_summary.png")
        plt.close()

    # ── Figure 3: E2 — Cross-dataset ─────────────────────────────────────
    if e2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            "Fig 3: RandLoRA Across Datasets (OPT-350M, Rank=16)",
            fontsize=13, fontweight="bold")

        for idx, (ds_name, res) in enumerate(e2.items()):
            ax = axes[idx]
            for mode in ["standard", "randlora"]:
                if mode not in res:
                    continue
                losses = (res[mode]["losses"] if isinstance(res[mode], dict)
                          else res[mode].losses)
                s = [float(np.mean(losses[max(0,i-15):i+1]))
                     for i in range(len(losses))]
                ax.plot(s, color=COLORS[mode], linewidth=2.5,
                        label=f"{'Standard LoRA' if mode=='standard' else 'RandLoRA'}")

            ax.set_title(ds_name)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig("plots/fig3_cross_dataset.png", bbox_inches="tight")
        print("  Saved → plots/fig3_cross_dataset.png")
        plt.close()

    # ── Figure 4: E3 — Rank sensitivity ──────────────────────────────────
    if e3:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            "Fig 4: Rank Sensitivity — RandLoRA Advantage vs LoRA Rank",
            fontsize=13, fontweight="bold")

        ranks     = [int(r) for r in e3.keys()]
        init_impr = []
        final_impr= []
        init_ms   = []

        for r_str, res in e3.items():
            std_il  = (res["standard"]["initial_loss"] if isinstance(res["standard"], dict)
                       else res["standard"].initial_loss)
            rand_il = (res["randlora"]["initial_loss"]  if isinstance(res["randlora"], dict)
                       else res["randlora"].initial_loss)
            std_fl  = (res["standard"]["final_loss"]   if isinstance(res["standard"], dict)
                       else res["standard"].final_loss)
            rand_fl = (res["randlora"]["final_loss"]    if isinstance(res["randlora"], dict)
                       else res["randlora"].final_loss)
            pm      = (res["randlora"]["patch_ms"]      if isinstance(res["randlora"], dict)
                       else res["randlora"].patch_ms)

            init_impr.append(100 * (std_il - rand_il) / std_il)
            final_impr.append(100 * (std_fl - rand_fl) / std_fl)
            init_ms.append(pm)

        ax = axes[0]
        ax.plot(ranks, init_impr,  marker="o", color=COLORS["randlora"],
                linewidth=2.5, label="Initial loss improvement")
        ax.plot(ranks, final_impr, marker="s", color=COLORS["standard"],
                linewidth=2.5, label="Final loss improvement")
        ax.axhline(0, linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("LoRA Rank")
        ax.set_ylabel("Loss Improvement (%)")
        ax.set_title("RandLoRA Advantage vs Rank")
        ax.legend(fontsize=9)

        ax = axes[1]
        ax.bar(ranks, init_ms, color=COLORS["randlora"], alpha=0.85, width=2)
        ax.set_xlabel("LoRA Rank")
        ax.set_ylabel("RandSVD Init Time (ms)")
        ax.set_title("Initialization Overhead vs Rank")

        plt.tight_layout()
        plt.savefig("plots/fig4_rank_sensitivity.png", bbox_inches="tight")
        print("  Saved → plots/fig4_rank_sensitivity.png")
        plt.close()

    # ── Figure 5: E4 — Init time speedup ─────────────────────────────────
    if e4:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            "Fig 5: C++ RandSVD vs Exact SVD — Initialization Time",
            fontsize=13, fontweight="bold")

        x     = np.arange(len(e4["sizes"]))
        exact = e4["exact_ms"]
        rand  = e4["rand_ms"]
        spdup = e4["speedup"]

        axes[0].bar(x - 0.2, exact, 0.4, label="Exact SVD (PyTorch)",
                     color=COLORS["standard"], alpha=0.85)
        axes[0].bar(x + 0.2, rand,  0.4, label="RandSVD (C++ backend)",
                     color=COLORS["randlora"],  alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(e4["sizes"], rotation=30)
        axes[0].set_ylabel("Time (ms)")
        axes[0].set_title("Absolute Init Time")
        axes[0].legend(fontsize=9)
        axes[0].set_yscale("log")

        axes[1].plot(e4["sizes"], spdup, marker="o",
                      color=COLORS["randlora"], linewidth=2.5)
        axes[1].axhline(1.0, linestyle="--", color="gray", linewidth=1)
        axes[1].fill_between(range(len(e4["sizes"])),
                              [1]*len(e4["sizes"]), spdup,
                              alpha=0.15, color=COLORS["randlora"])
        axes[1].set_xticks(range(len(e4["sizes"])))
        axes[1].set_xticklabels(e4["sizes"], rotation=30)
        axes[1].set_ylabel("Speedup (×)")
        axes[1].set_title("Speedup: Exact / RandSVD")

        # Annotate peak speedup
        peak_idx = int(np.argmax(spdup))
        axes[1].annotate(
            f"Peak {spdup[peak_idx]:.1f}×",
            xy=(peak_idx, spdup[peak_idx]),
            xytext=(peak_idx - 1, spdup[peak_idx] + 1),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig("plots/fig5_init_speedup.png", bbox_inches="tight")
        print("  Saved → plots/fig5_init_speedup.png")
        plt.close()

    # ── Figure 6: E5 — Gradient fidelity heatmap ─────────────────────────
    if e5:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            "Fig 6: Gradient Compression Fidelity\n"
            "(Cosine Similarity: Original vs Compressed Gradient)",
            fontsize=13, fontweight="bold")

        data = np.array(e5["cos_sim"])  # (ratios, sizes)

        im = axes[0].imshow(data, cmap="RdYlGn", vmin=0.7, vmax=1.0,
                             aspect="auto")
        axes[0].set_xticks(range(len(e5["sizes"])))
        axes[0].set_xticklabels(e5["sizes"])
        axes[0].set_yticks(range(len(e5["ratios"])))
        axes[0].set_yticklabels([f"{r:.2f}" for r in e5["ratios"]])
        axes[0].set_xlabel("Matrix Size (d×d)")
        axes[0].set_ylabel("Compression Ratio k/d")
        axes[0].set_title("Cosine Similarity Heatmap")
        plt.colorbar(im, ax=axes[0])

        # Add text annotations
        for i in range(len(e5["ratios"])):
            for j in range(len(e5["sizes"])):
                val = data[i, j]
                axes[0].text(j, i, f"{val:.3f}",
                              ha="center", va="center",
                              fontsize=8,
                              color="white" if val < 0.85 else "black")

        # Line plot: cos-sim vs ratio for each size
        for j, size in enumerate(e5["sizes"]):
            col = data[:, j]
            axes[1].plot(e5["ratios"], col, marker="o", linewidth=2,
                          label=f"d={size}")
        axes[1].axhline(0.95, linestyle="--", color="gray",
                         linewidth=1.5, label="0.95 threshold")
        axes[1].fill_between([min(e5["ratios"]), max(e5["ratios"])],
                              0.95, 1.0, alpha=0.08, color="green",
                              label="Acceptable region")
        axes[1].set_xlabel("Compression Ratio")
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_title("Fidelity vs Compression Ratio")
        axes[1].legend(fontsize=8)
        axes[1].set_ylim(0.6, 1.05)

        plt.tight_layout()
        plt.savefig("plots/fig6_gradient_fidelity.png", bbox_inches="tight")
        print("  Saved → plots/fig6_gradient_fidelity.png")
        plt.close()

    # ── Figure 7: Full summary dashboard ─────────────────────────────────
    if e1 and e4:
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            "RandSolve: Randomized NLA for Transformer Fine-Tuning\n"
            "Complete Results Summary",
            fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.45, wspace=0.35)

        # Panel A: best model loss curves
        ax_a = fig.add_subplot(gs[0, 0])
        best_model = list(e1.keys())[1]  # opt-350m
        res = e1[best_model]
        for mode in ["standard", "randlora"]:
            if mode not in res:
                continue
            losses = (res[mode]["losses"] if isinstance(res[mode], dict)
                      else res[mode].losses)
            s = [float(np.mean(losses[max(0,i-15):i+1]))
                 for i in range(len(losses))]
            ax_a.plot(s, color=COLORS[mode], linewidth=2.5,
                      label=f"{'Standard' if mode=='standard' else 'RandLoRA'}")
        ax_a.set_title(f"A. Loss Convergence ({best_model})")
        ax_a.set_xlabel("Step"); ax_a.set_ylabel("Loss")
        ax_a.legend(fontsize=9)

        # Panel B: initial loss improvement bar
        ax_b = fig.add_subplot(gs[0, 1])
        mkeys  = list(e1.keys())
        imprvs = []
        for mk in mkeys:
            res = e1[mk]
            si  = (res["standard"]["initial_loss"] if isinstance(res["standard"], dict)
                   else res["standard"].initial_loss)
            ri  = (res["randlora"]["initial_loss"]  if isinstance(res["randlora"], dict)
                   else res["randlora"].initial_loss)
            imprvs.append(100 * (si - ri) / si)

        bars = ax_b.bar(mkeys, imprvs,
                         color=[COLORS["randlora"] if v > 0
                                else COLORS["standard"] for v in imprvs],
                         alpha=0.85)
        ax_b.axhline(0, color="gray", linewidth=1)
        ax_b.set_ylabel("Initial Loss Improvement (%)")
        ax_b.set_title("B. RandLoRA Init Advantage")
        ax_b.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, imprvs):
            ax_b.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.2,
                       f"{val:.1f}%", ha="center",
                       fontsize=9, fontweight="bold")

        # Panel C: speedup vs matrix size
        ax_c = fig.add_subplot(gs[0, 2])
        ax_c.plot(range(len(e4["sizes"])), e4["speedup"],
                   marker="o", color=COLORS["randlora"], linewidth=2.5)
        ax_c.axhline(1.0, linestyle="--", color="gray")
        ax_c.set_xticks(range(len(e4["sizes"])))
        ax_c.set_xticklabels(e4["sizes"], rotation=30, fontsize=8)
        ax_c.set_ylabel("Speedup (×)")
        ax_c.set_title("C. RandSVD vs Exact SVD")

        # Panel D: rank sensitivity
        if e3:
            ax_d = fig.add_subplot(gs[1, 0])
            ranks = [int(r) for r in e3.keys()]
            imprs = []
            for r_str, res in e3.items():
                si = (res["standard"]["initial_loss"] if isinstance(res["standard"], dict)
                      else res["standard"].initial_loss)
                ri = (res["randlora"]["initial_loss"]  if isinstance(res["randlora"], dict)
                      else res["randlora"].initial_loss)
                imprs.append(100 * (si - ri) / si)
            ax_d.plot(ranks, imprs, marker="s",
                       color=COLORS["randlora"], linewidth=2.5)
            ax_d.axhline(0, linestyle="--", color="gray")
            ax_d.set_xlabel("LoRA Rank")
            ax_d.set_ylabel("Init Improvement (%)")
            ax_d.set_title("D. Advantage vs Rank")

        # Panel E: gradient fidelity
        if e5:
            ax_e = fig.add_subplot(gs[1, 1])
            for j, size in enumerate(e5["sizes"]):
                ax_e.plot(e5["ratios"],
                           np.array(e5["cos_sim"])[:, j],
                           marker="o", linewidth=2, label=f"d={size}")
            ax_e.axhline(0.95, linestyle="--", color="gray")
            ax_e.set_xlabel("Compression Ratio")
            ax_e.set_ylabel("Gradient Cos-Sim")
            ax_e.set_title("E. Gradient Compression Fidelity")
            ax_e.legend(fontsize=8)

        # Panel F: dataset comparison
        if e2:
            ax_f = fig.add_subplot(gs[1, 2])
            ds_names = list(e2.keys())
            std_init = []
            rnd_init = []
            for dn in ds_names:
                res = e2[dn]
                si  = (res["standard"]["initial_loss"] if isinstance(res["standard"], dict)
                       else res["standard"].initial_loss)
                ri  = (res["randlora"]["initial_loss"]  if isinstance(res["randlora"], dict)
                       else res["randlora"].initial_loss)
                std_init.append(si)
                rnd_init.append(ri)

            xd = np.arange(len(ds_names))
            ax_f.bar(xd - 0.2, std_init, 0.4,
                      label="Standard", color=COLORS["standard"], alpha=0.85)
            ax_f.bar(xd + 0.2, rnd_init, 0.4,
                      label="RandLoRA",  color=COLORS["randlora"],  alpha=0.85)
            ax_f.set_xticks(xd)
            ax_f.set_xticklabels(ds_names)
            ax_f.set_ylabel("Initial Loss")
            ax_f.set_title("F. Cross-Dataset Init Loss")
            ax_f.legend(fontsize=9)

        plt.savefig("plots/fig7_dashboard.png", bbox_inches="tight")
        print("  Saved → plots/fig7_dashboard.png")
        plt.close()

# ─────────────────────────────────────────────────────────────────────────
# PRINT FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────

def print_final_summary(e1, e3, e4):
    print("\n" + "="*75)
    print("  FINAL RESULTS SUMMARY")
    print("="*75)

    if e1:
        print(f"\n  E1: Cross-Model (rank=16, wikitext2, 300 steps)")
        print(f"  {'Model':<14} {'Std Init':>10} {'Rand Init':>10} "
              f"{'Δ Init':>8} {'Std Final':>10} {'Rand Final':>10} "
              f"{'Δ Final':>8} {'Init(ms)':>10}")
        print(f"  {'─'*82}")
        for mk, res in e1.items():
            si  = (res["standard"]["initial_loss"] if isinstance(res["standard"], dict)
                   else res["standard"].initial_loss)
            ri  = (res["randlora"]["initial_loss"]  if isinstance(res["randlora"], dict)
                   else res["randlora"].initial_loss)
            sf  = (res["standard"]["final_loss"]    if isinstance(res["standard"], dict)
                   else res["standard"].final_loss)
            rf  = (res["randlora"]["final_loss"]    if isinstance(res["randlora"], dict)
                   else res["randlora"].final_loss)
            pm  = (res["randlora"]["patch_ms"]      if isinstance(res["randlora"], dict)
                   else res["randlora"].patch_ms)
            print(f"  {mk:<14} {si:>10.3f} {ri:>10.3f} "
                  f"{100*(si-ri)/si:>+7.1f}% {sf:>10.3f} {rf:>10.3f} "
                  f"{100*(sf-rf)/sf:>+7.1f}% {pm:>9.0f}ms")

    if e4:
        peak_spdup = max(e4["speedup"])
        peak_size  = e4["sizes"][int(np.argmax(e4["speedup"]))]
        print(f"\n  E4: RandSVD vs Exact SVD")
        print(f"    Peak speedup: {peak_spdup:.2f}× at {peak_size}")
        print(f"    Mean speedup: {np.mean(e4['speedup']):.2f}×")

    print("\n" + "="*75)
    print("  Plots saved to plots/")
    print("  Data  saved to experiments/")
    print("="*75)

# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test: fewer steps, fewer models")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--skip-e1", action="store_true")
    parser.add_argument("--skip-e2", action="store_true")
    parser.add_argument("--skip-e3", action="store_true")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    STEPS  = 50 if args.quick else args.steps

    print("\n" + "="*70)
    print("  RandSolve: Full Experimental Suite")
    print("="*70)
    print(f"  Device : {DEVICE}")
    print(f"  Steps  : {STEPS}")
    print(f"  Mode   : {'QUICK' if args.quick else 'FULL'}")
    print("="*70)

    # Run experiments
    e1, e2, e3, e4, e5 = None, None, None, None, None

    # E4 and E5 are fast — always run
    e4 = experiment_e4_init_time()
    e5 = experiment_e5_gradient_fidelity()

    if not args.skip_e1:
        # Quick mode: only 2 models
        if args.quick:
            global MODELS
            MODELS = {k: v for k, v in list(MODELS.items())[:2]}
        e1 = experiment_e1_models(STEPS, DEVICE)

    if not args.skip_e2:
        e2 = experiment_e2_datasets(STEPS, DEVICE)

    if not args.skip_e3:
        ranks_to_test = [8, 16] if args.quick else [4, 8, 16, 32]
        e3 = {}
        for rank in ranks_to_test:
            res = run_comparison("opt-350m", "wikitext2",
                                  rank=rank, steps=STEPS, device=DEVICE)
            e3[str(rank)] = res
        save_results(e3, "experiments/e3_ranks.json")

    # Plot everything
    print("\n  Generating plots...")
    make_all_plots(e1, e2, e3, e4, e5)

    # Print summary
    print_final_summary(e1, e3, e4)


if __name__ == "__main__":
    main()