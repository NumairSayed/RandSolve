#!/usr/bin/env python3
import subprocess
import os
import matplotlib.pyplot as plt

# ----------------------------
# Binary path 
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(SCRIPT_DIR, "..", "build", "benchmark")


# ----------------------------
# Run C++ binary
# ----------------------------
def run_binary():
    if not os.path.exists(BINARY):
        print(f"Error: {BINARY} not found")
        exit(1)

    print("Using binary:", BINARY)
    print("Running benchmark...\n")

    result = subprocess.run([BINARY], capture_output=True, text=True, check=True)
    return result.stdout


# ----------------------------
# Parse output
# ----------------------------
def parse_output(text):
    data = {
        "regression": {"Gaussian": [], "Sparse": []},
        "attention": {"Gaussian": [], "Sparse": []},
        "pca": {}
    }

    current = None

    for line in text.splitlines():
        line = line.strip()

        if "=== Linear Regression ===" in line:
            current = "regression"
            continue
        elif "=== Attention" in line:
            current = "attention"
            continue

        # Capture exact times
        if "Exact:" in line and current == "regression":
            data["regression"]["exact_time"] = float(line.split()[1])
        elif "Exact:" in line and current == "attention":
            data["attention"]["exact_time"] = float(line.split()[1])

        # PCA parsing
        if "Exact SVD" in line:
            data["pca"]["exact_time"] = float(line.split()[2])
        elif "RandSVD" in line:
            parts = line.split()
            data["pca"]["rand_time"] = float(parts[1])
            data["pca"]["err"] = float(parts[-1].split("=")[1])

        # Sketch lines
        if "[" in line and "time=" in line:
            try:
                parts = line.split()

                kind = parts[0][1:]
                k = int(parts[1].split("=")[1][:-1])
                time = float(parts[2].split("=")[1])
                err = float(parts[-1].split("=")[1])

                if current:
                    data[current][kind].append((k, time, err))

            except:
                pass

    return data


# ----------------------------
# Annotate values
# ----------------------------
def annotate(xs, ys):
    for x, y in zip(xs, ys):
        label = f"{y:.2e}" if y < 1e-2 else f"{y:.3f}"
        plt.annotate(label, (x, y),
                     textcoords="offset points",
                     xytext=(0, 6),
                     ha='center',
                     fontsize=8)


# ----------------------------
# Plot everything
# ----------------------------
def plot_all(data):

    plt.figure(figsize=(12, 10))

    # ---------- Regression Error ----------
    plt.subplot(3, 2, 1)
    for kind in ["Gaussian", "Sparse"]:
        vals = sorted(data["regression"][kind])
        if vals:
            ks = [v[0] for v in vals]
            errs = [v[2] for v in vals]
            plt.plot(ks, errs, marker='o', label=kind)
            annotate(ks, errs)

    plt.title("Regression Error")
    plt.xlabel("k")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.grid()

    # ---------- Regression Time ----------
    plt.subplot(3, 2, 2)
    all_ks = []
    for kind in ["Gaussian", "Sparse"]:
        vals = sorted(data["regression"][kind])
        if vals:
            ks = [v[0] for v in vals]
            times = [v[1] for v in vals]
            plt.plot(ks, times, marker='o', label=kind)
            annotate(ks, times)
            all_ks.extend(ks)

    if "exact_time" in data["regression"]:
        ks_all = sorted(set(all_ks))
        plt.plot(ks_all,
                 [data["regression"]["exact_time"]] * len(ks_all),
                 linestyle='--',
                 color='black',
                 label="Exact")

    plt.title("Regression Time")
    plt.xlabel("k")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid()

    # ---------- Attention Error ----------
    plt.subplot(3, 2, 3)
    for kind in ["Gaussian", "Sparse"]:
        vals = sorted(data["attention"][kind])
        if vals:
            ks = [v[0] for v in vals]
            errs = [v[2] for v in vals]
            plt.plot(ks, errs, marker='o', label=kind)
            annotate(ks, errs)

    plt.title("Attention Error")
    plt.xlabel("k")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.grid()

    # ---------- Attention Time ----------
    plt.subplot(3, 2, 4)
    all_ks = []
    for kind in ["Gaussian", "Sparse"]:
        vals = sorted(data["attention"][kind])
        if vals:
            ks = [v[0] for v in vals]
            times = [v[1] for v in vals]
            plt.plot(ks, times, marker='o', label=kind)
            annotate(ks, times)
            all_ks.extend(ks)

    if "exact_time" in data["attention"]:
        ks_all = sorted(set(all_ks))
        plt.plot(ks_all,
                 [data["attention"]["exact_time"]] * len(ks_all),
                 linestyle='--',
                 color='black',
                 label="Exact")

    plt.title("Attention Time")
    plt.xlabel("k")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.grid()

    # ---------- PCA ----------
    plt.subplot(3, 1, 3)
    if data["pca"]:
        labels = ["Exact SVD", "RandSVD"]
        times = [data["pca"]["exact_time"], data["pca"]["rand_time"]]

        plt.bar(labels, times)

        for i, v in enumerate(times):
            plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')

        plt.title(f"PCA Time (err={data['pca']['err']:.2e})")
        plt.ylabel("Time (ms)")
        plt.grid(axis='y')

    plt.tight_layout()
    plt.show()


# ----------------------------
# MAIN
# ----------------------------
def main():
    output = run_binary()

    print("\n--- RAW OUTPUT ---\n")
    print(output)

    data = parse_output(output)

    print("\n--- PARSED DATA ---\n")
    print(data)

    plot_all(data)


if __name__ == "__main__":
    main()