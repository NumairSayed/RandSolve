"""
attention_exp_plots.py
Robust plotting script for attention_benchmark_results.csv
Handles variable-column-count rows written by different experiments.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
from io import StringIO

# ── 1. Load raw lines and split by experiment tag ──────────────────────────
raw_rows = {
    'scaling'     : [],
    'ablation'    : [],
    'sketch_type' : [],
    'fidelity'    : [],
    'multihead'   : [],
    'entropy'     : [],
}

with open('./attention_benchmark_results.csv', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Skip header rows (first field == 'exp')
        if line.startswith('exp,'):
            continue
        tag = line.split(',')[0]
        if tag in raw_rows:
            raw_rows[tag].append(line)

# ── 2. Parse each experiment with its known schema ─────────────────────────

def parse(rows, columns):
    if not rows:
        return pd.DataFrame(columns=columns)
    text = ','.join(columns) + '\n' + '\n'.join(rows)
    return pd.read_csv(StringIO(text))

scaling = parse(raw_rows['scaling'],
    ['exp','L','d','k','method','time_ms','speedup'])

ablation = parse(raw_rows['ablation'],
    ['exp','L','d','k','k_over_d',
     'output_cos_sim','output_rel_err',
     'attn_cos_sim','attn_rel_err','time_ms','speedup'])

sketch_type = parse(raw_rows['sketch_type'],
    ['exp','L','d','k','sketch_type',
     'output_cos_sim','output_rel_err','time_ms','speedup'])

fidelity = parse(raw_rows['fidelity'],
    ['exp','trial','k','cos_sim','rel_err'])

multihead = parse(raw_rows['multihead'],
    ['exp','L','d_model','num_heads','d_head',
     'sketch_dim','method','time_ms','speedup'])

entropy = parse(raw_rows['entropy'],
    ['exp','trial','k','method','entropy'])

# ── 3. Plot ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Sketched Attention Benchmark Results', fontsize=16, fontweight='bold')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
ax1, ax2, ax3, ax4, ax5, ax6 = axes

COLORS = {'exact':'#2C7BB6', 'gauss_sketch':'#D7191C', 'sparse_sketch':'#1A9641'}
MARKERS= {'exact':'o',       'gauss_sketch':'s',       'sparse_sketch':'^'}

# ── Plot 1: Time vs Sequence Length ───────────────────────────────────────
if not scaling.empty:
    for method, grp in scaling.groupby('method'):
        grp = grp.sort_values('L')
        color  = COLORS.get(method, None)
        marker = MARKERS.get(method, 'o')
        label  = {'exact':'Exact O(L²d)',
                  'gauss_sketch':'Gaussian sketch',
                  'sparse_sketch':'CountSketch'}.get(method, method)
        ax1.plot(grp['L'], grp['time_ms'],
                 marker=marker, color=color, label=label, linewidth=2)
    ax1.set_xlabel('Sequence Length L', fontsize=11)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.set_title('① Attention Time vs L', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

# ── Plot 2: Quality vs Sketch Size ────────────────────────────────────────
if not ablation.empty:
    abl = ablation.sort_values('k_over_d')
    ax2.plot(abl['k_over_d'], abl['output_cos_sim'],
             marker='s', color='#D7191C', linewidth=2, label='Cosine sim (output)')
    ax2.plot(abl['k_over_d'], abl['attn_cos_sim'],
             marker='^', color='#1A9641', linewidth=2,
             linestyle='--', label='Cosine sim (attn weights)')
    ax2.axhline(1.0, linestyle=':', color='#2C7BB6', linewidth=1.5, label='Exact = 1.0')
    ax2.axhline(0.99, linestyle='--', color='gray', linewidth=1,
                alpha=0.6, label='0.99 threshold')
    ax2.set_xlabel('k / d  (sketch dimension ratio)', fontsize=11)
    ax2.set_ylabel('Cosine Similarity', fontsize=11)
    ax2.set_title('② Output Quality vs Sketch Size', fontsize=12, fontweight='bold')
    ax2.set_ylim(0.85, 1.02)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

# ── Plot 3: Sketch Type Comparison ────────────────────────────────────────
if not sketch_type.empty:
    type_colors = {
        'Gaussian': '#2C7BB6',
        'CountSketch': '#D7191C',
        'OSNAP s=2': '#1A9641',
        'OSNAP s=4': '#FF7F00',
    }
    sketch_type['base_type'] = sketch_type['sketch_type'].str.extract(
        r'(Gaussian|CountSketch|OSNAP s=\d+)')
    for btype, grp in sketch_type.groupby('base_type'):
        grp = grp.sort_values('k')
        color = type_colors.get(btype, None)
        ax3.plot(grp['k'], grp['output_cos_sim'],
                 marker='o', color=color, label=btype, linewidth=2)
    ax3.axhline(1.0, linestyle=':', color='gray', linewidth=1.5)
    ax3.set_xlabel('Sketch dimension k', fontsize=11)
    ax3.set_ylabel('Cosine Similarity (output)', fontsize=11)
    ax3.set_title('③ Sketch Type Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

# ── Plot 4: Fidelity Distribution ────────────────────────────────────────
if not fidelity.empty:
    fid_colors = plt.cm.viridis(np.linspace(0.2, 0.85, fidelity['k'].nunique()))
    for (k_val, grp), color in zip(fidelity.groupby('k'), fid_colors):
        ax4.hist(grp['cos_sim'], bins=8, alpha=0.65,
                 color=color, label=f'k={int(k_val)}', edgecolor='white')
    ax4.axvline(1.0, linestyle='--', color='black', linewidth=1.5, label='Perfect = 1.0')
    ax4.set_xlabel('Cosine Similarity', fontsize=11)
    ax4.set_ylabel('Count (over 20 trials)', fontsize=11)
    ax4.set_title('④ Output Fidelity Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

# ── Plot 5: Multi-head Speedup ────────────────────────────────────────────
if not multihead.empty:
    mh_exact    = multihead[multihead['method'] == 'exact']
    mh_sketched = multihead[multihead['method'] == 'sketched']
    if not mh_exact.empty and not mh_sketched.empty:
        heads = sorted(mh_exact['num_heads'].unique())
        exact_times    = [mh_exact[mh_exact['num_heads']==h]['time_ms'].values[0]
                          for h in heads if len(mh_exact[mh_exact['num_heads']==h]) > 0]
        sketched_times = [mh_sketched[mh_sketched['num_heads']==h]['time_ms'].values[0]
                          for h in heads if len(mh_sketched[mh_sketched['num_heads']==h]) > 0]
        x = np.arange(len(heads))
        w = 0.35
        ax5.bar(x - w/2, exact_times,    w, label='Exact',    color='#2C7BB6', alpha=0.85)
        ax5.bar(x + w/2, sketched_times, w, label='Sketched', color='#D7191C', alpha=0.85)
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'{h} head{"s" if h>1 else ""}' for h in heads])
        ax5.set_ylabel('Time (ms)', fontsize=11)
        ax5.set_title('⑤ Multi-Head Scaling', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')

# ── Plot 6: Attention Entropy ─────────────────────────────────────────────
if not entropy.empty:
    ent_colors = {'exact': '#2C7BB6'}
    unique_methods = entropy['method'].unique()
    palette = plt.cm.Reds(np.linspace(0.4, 0.9, len(unique_methods)))
    for i, m in enumerate(unique_methods):
        if m == 'exact':
            color = '#2C7BB6'
        else:
            color = palette[i]
        grp = entropy[entropy['method'] == m]
        ax6.scatter([m] * len(grp), grp['entropy'],
                    alpha=0.6, color=color, s=40)
        mean_val = grp['entropy'].mean()
        ax6.axhline(mean_val, color=color, linewidth=1.5, alpha=0.8)
    ax6.set_xlabel('Method', fontsize=11)
    ax6.set_ylabel('Attention Entropy', fontsize=11)
    ax6.set_title('⑥ Attention Entropy Preservation', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='x', rotation=30)
    ax6.grid(True, alpha=0.3, axis='y')

plt.savefig('attention_benchmark.png', dpi=150, bbox_inches='tight')
print("Saved → attention_benchmark.png")
plt.show()