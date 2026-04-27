import sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
BUILD = os.path.join(ROOT, "build")

sys.path.insert(0, ROOT)
sys.path.insert(0, BUILD)

if not os.path.exists(BUILD):
    raise RuntimeError("Build directory not found. Run CMake first.")

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import Transformer, CharDataset

CONFIGS = [
    (192, 64), (192, 96), (192, 128),
    (256, 64), (256, 96), (256, 128),
    (512, 64), (512, 96), (512, 128),
]

VARIANTS = [
    "baseline",
    "randlora_gaussian",
    "randlora_count",
    "randlora_hadamard",
    "randlora_compress"
]



# EVALUATION

def evaluate(model, dataset, steps=200):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    with torch.no_grad():
        for _ in range(steps):
            x, y = dataset.get_batch(8)
            logits = model(x)
            loss = loss_fn(
                logits.reshape(-1, dataset.vocab_size),
                y.reshape(-1)
            )
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)

    return avg_loss, ppl

# TIME TO TARGET

def time_to_target(losses, times, target):
    for l, t in zip(losses, times):
        if l <= target:
            return t
    return times[-1]

# LOAD TRAINING CURVES

def load_training_curves(base_dir):
    curves = {}

    for d, L in CONFIGS:
        curves[(d, L)] = {}

        for v in VARIANTS:
            path = os.path.join(base_dir, f"curves_{v}_{d}_{L}.pt")

            if not os.path.exists(path):
                continue

            losses, times = torch.load(path, map_location="cpu")
            curves[(d, L)][v] = (losses, times)

    return curves

# LOAD MODELS + PPL

def compute_ppl(base_dir):

    data_path = os.path.join(base_dir, "input.txt")

    ppl_results = {}

    for d, L in CONFIGS:
        ppl_results[(d, L)] = {}

        dataset = CharDataset(data_path, seq_len=L)

        for v in VARIANTS:

            ckpt = os.path.join(base_dir, f"model_{v}_{d}_{L}.pt")

            if not os.path.exists(ckpt):
                continue

            sketch = "gaussian"
            if "count" in v:
                sketch = "count"
            elif "hadamard" in v:
                sketch = "hadamard"

            model = Transformer(
                dataset.vocab_size,
                d,
                L,
                r=2,
                randlora=("randlora" in v),
                sketch=sketch
            )

            model.load_state_dict(torch.load(ckpt, map_location="cpu"))

            loss, ppl = evaluate(model, dataset)
            ppl_results[(d, L)][v] = (loss, ppl)

    return ppl_results

# BUILD FINAL TABLE

def build_table(curves, ppl_results):

    rows = []

    required = VARIANTS

    for (d, L) in CONFIGS:

        c = curves.get((d, L), {})
        p = ppl_results.get((d, L), {})

        if not all(v in c for v in required):
            continue
        if not all(v in p for v in required):
            continue

        lb, tb = c["baseline"]
        lg, tg = c["randlora_gaussian"]
        lc, tc = c["randlora_count"]
        lh, th = c["randlora_hadamard"]
        lcmp, tcmp = c["randlora_compress"]

        target = min(lb[-1], lg[-1], lc[-1], lh[-1], lcmp[-1]) * 1.05

        rows.append({
            "d": d,
            "L": L,
            "Baseline Time": round(time_to_target(lb, tb, target), 2),
            "Gaussian Time": round(time_to_target(lg, tg, target), 2),
            "Count Time": round(time_to_target(lc, tc, target), 2),
            "Hadamard Time": round(time_to_target(lh, th, target), 2),
            "Compressed Time": round(time_to_target(lcmp, tcmp, target), 2),
            "Baseline PPL": round(p["baseline"][1], 2),
            "Gaussian PPL": round(p["randlora_gaussian"][1], 2),
            "Count PPL": round(p["randlora_count"][1], 2),
            "Hadamard PPL": round(p["randlora_hadamard"][1], 2),
            "Compressed PPL": round(p["randlora_compress"][1], 2),
        })

    return pd.DataFrame(rows)

# REPORT PLOT

def report_plot(df, save_dir):

    labels = [f"{d}/{L}" for d, L in zip(df["d"], df["L"])]
    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(12,6))
    width = 0.15

    ax1.bar(x - 2*width, df["Baseline Time"], width, label="Baseline", alpha=0.85)
    ax1.bar(x - width, df["Gaussian Time"], width, label="Gaussian", alpha=0.85)
    ax1.bar(x, df["Count Time"], width, label="Count", alpha=0.85)
    ax1.bar(x + width, df["Hadamard Time"], width, label="Hadamard", alpha=0.85)
    ax1.bar(x + 2*width, df["Compressed Time"], width, label="Compressed", alpha=0.85)

    ax1.set_ylabel("Time (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.grid(axis='y', alpha=0.2)

    ax2 = ax1.twinx()

    ax2.plot(x, df["Baseline PPL"], '--o', linewidth=1.5, markersize=4)
    ax2.plot(x, df["Gaussian PPL"], '-o', linewidth=1.5, markersize=4)
    ax2.plot(x, df["Count PPL"], '-o', linewidth=1.5, markersize=4)
    ax2.plot(x, df["Hadamard PPL"], '-o', linewidth=1.5, markersize=4)
    ax2.plot(x, df["Compressed PPL"], ':o', linewidth=1.5, markersize=4)

    ax2.set_ylabel("Perplexity")

    ax1.legend(loc="upper left", fontsize=8)

    plt.title("Runtime vs Perplexity Tradeoff")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    plt.savefig(os.path.join(save_dir, "report_plot.png"))
    plt.close()

# INFERENCE SNAPSHOT

def generate_snapshot(model, dataset, prompt="The ", steps=50, temperature=1.0):
    model.eval()

    x = torch.tensor([[dataset.stoi[c] for c in prompt]], dtype=torch.long)

    generated = prompt
    tokens = []

    with torch.no_grad():
        for _ in range(steps):
            logits = model(x)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            token_char = dataset.itos[token_id]

            tokens.append({
                "id": token_id,
                "char": token_char,
                "max_prob": probs.max().item()
            })

            generated += token_char
            x = torch.cat([x, next_token], dim=1)

    return generated, tokens


def inference_comparison(base_dir, d=256, L=96, prompt="The "):

    data_path = os.path.join(base_dir, "input.txt")
    dataset = CharDataset(data_path, seq_len=L)

    outputs = {}

    for v in VARIANTS:
        ckpt = os.path.join(base_dir, f"model_{v}_{d}_{L}.pt")
        if not os.path.exists(ckpt):
            continue

        sketch = "gaussian"
        if "count" in v:
            sketch = "count"
        elif "hadamard" in v:
            sketch = "hadamard"

        model = Transformer(
            dataset.vocab_size,
            d,
            L,
            r=2,
            randlora=("randlora" in v),
            sketch=sketch
        )

        model.load_state_dict(torch.load(ckpt, map_location="cpu"))

        text, tokens = generate_snapshot(model, dataset, prompt)

        outputs[v] = {
            "text": text,
            "tokens": tokens
        }

    return outputs


def print_snapshot(outputs):
    print("\n===== INFERENCE SNAPSHOT =====\n")

    for v, out in outputs.items():
        print(f"[{v}]")
        print(out["text"])
        print("----")

# MAIN

if __name__ == "__main__":

    BASE = os.path.dirname(__file__)

    print("Loading training curves...")
    curves = load_training_curves(BASE)

    print("Computing perplexity...")
    ppl_results = compute_ppl(BASE)

    print("Building table...")
    df = build_table(curves, ppl_results)

    print("\n===== FINAL REPORT TABLE =====")
    print(df.to_string(index=False))

    df.to_csv(os.path.join(BASE, "final_report_table.csv"), index=False)

    print("Generating plot...")
    report_plot(df, BASE)

    print("\nRunning inference snapshot...")
    snap = inference_comparison(BASE, d=256, L=96, prompt="The ")
    print_snapshot(snap)

    print("\nSaved:")
    print("- final_report_table.csv")
    print("- report_plot.png")