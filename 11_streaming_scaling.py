#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_streaming_scaling.py
=======================
Power-Law Scaling of Streaming Distributions & Fractal Dimension

Input:  04_clean_lyrics_data/04_lyrics_cleaned.csv
Output: 11_streaming_scaling/
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
INPUT_FILE = "04_clean_lyrics_data/04_lyrics_cleaned.csv"
OUTPUT_DIR = "11_streaming_scaling"
LANGUAGES = {"en": "English", "de": "German", "es": "Spanish"}
SEED = 42
# =======================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def gini(x):
    x = np.sort(np.array(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n


def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñäöß\w\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def box_counting_1d(data, n_boxes=20):
    data = np.array(data, dtype=float)
    if len(data) < 10:
        return None

    data_range = data.max() - data.min()
    if data_range == 0:
        return None

    data_norm = (data - data.min()) / data_range

    epsilons = np.logspace(-3, 0, n_boxes)
    N_eps = []
    valid_eps = []

    for eps in epsilons:
        bins = np.arange(0, 1 + eps, eps)
        counts, _ = np.histogram(data_norm, bins=bins)
        n_occupied = np.sum(counts > 0)
        if n_occupied > 0:
            N_eps.append(n_occupied)
            valid_eps.append(eps)

    if len(valid_eps) < 4:
        return None

    log_eps = np.log(1 / np.array(valid_eps))
    log_N = np.log(np.array(N_eps))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_eps, log_N
    )

    return {
        "D": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "log_eps": log_eps,
        "log_N": log_N,
    }


def box_counting_2d(points, n_boxes=20):
    points = np.array(points)
    if len(points) < 10:
        return None

    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    max_range = max(x_range, y_range)
    if max_range == 0:
        return None

    pts_norm = (points - points.min(axis=0)) / max_range

    epsilons = np.logspace(-2, 0, n_boxes)
    N_eps = []
    valid_eps = []

    for eps in epsilons:
        grid_x = np.floor(pts_norm[:, 0] / eps).astype(int)
        grid_y = np.floor(pts_norm[:, 1] / eps).astype(int)
        occupied = set(zip(grid_x, grid_y))
        n_occ = len(occupied)
        if n_occ > 0:
            N_eps.append(n_occ)
            valid_eps.append(eps)

    if len(valid_eps) < 4:
        return None

    log_eps = np.log(1 / np.array(valid_eps))
    log_N = np.log(np.array(N_eps))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_eps, log_N
    )

    return {
        "D": slope,
        "r_squared": r_value ** 2,
        "log_eps": log_eps,
        "log_N": log_N,
    }


# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("11 — STREAMING SCALING & FRACTAL DIMENSION ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
df = df[df["language"].isin(LANGUAGES.keys())].copy()
print(f"Loaded {len(df)} songs (EN/DE/ES)")
for lang, name in LANGUAGES.items():
    print(f"  {name}: {len(df[df['language'] == lang])} songs")

# Check for streams column
stream_col = None
for col in ["streams", "Streams", "stream_count", "play_count"]:
    if col in df.columns:
        stream_col = col
        break

if stream_col is None:
    print("[!] No streaming column found. Will skip streaming analysis.")
    HAS_STREAMS = False
else:
    df[stream_col] = pd.to_numeric(df[stream_col], errors="coerce")
    df = df.dropna(subset=[stream_col])
    HAS_STREAMS = True
    print(f"  Streaming column: '{stream_col}'")

# ============================================================
# 2. RANK-FREQUENCY ANALYSIS OF STREAMING COUNTS
# ============================================================
stream_results = {}

if HAS_STREAMS:
    print("\n--- Rank-frequency analysis of streaming counts ---")

    for lang, name in LANGUAGES.items():
        subset = df[df["language"] == lang].sort_values(
            stream_col, ascending=False
        ).reset_index(drop=True)
        streams = subset[stream_col].values
        ranks = np.arange(1, len(streams) + 1)

        valid = streams > 0
        log_rank = np.log10(ranks[valid])
        log_streams = np.log10(streams[valid])

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_rank, log_streams
        )

        stream_results[lang] = {
            "language": name,
            "n_songs": len(streams),
            "zipf_exponent": -slope,
            "r_squared": r_value ** 2,
            "std_err": std_err,
            "p_value": p_value,
            "log_rank": log_rank,
            "log_streams": log_streams,
            "intercept": intercept,
            "slope": slope,
            "max_streams": streams.max(),
            "median_streams": np.median(streams),
            "gini": gini(streams),
        }
        print(f"  {name}: Zipf exponent = {-slope:.4f} "
              f"(R2 = {r_value**2:.4f})")

    stream_df = pd.DataFrame([{
        "language": r["language"],
        "n_songs": r["n_songs"],
        "zipf_exponent": r["zipf_exponent"],
        "r_squared": r["r_squared"],
        "std_err": r["std_err"],
        "max_streams": r["max_streams"],
        "median_streams": r["median_streams"],
        "gini": r["gini"],
    } for r in stream_results.values()])
    stream_df.to_csv(os.path.join(OUTPUT_DIR, "streaming_exponents_table.csv"),
                      index=False)

# ============================================================
# 3. BOX-COUNTING FRACTAL DIMENSION ON WORD-LENGTH DISTRIBUTIONS
# ============================================================
print("\n--- Box-counting fractal dimension of word-length distributions ---")

bc_results_1d = {}
for lang, name in LANGUAGES.items():
    subset = df[df["language"] == lang]
    all_lengths = []
    for lyrics in subset["plain_lyrics"]:
        all_lengths.extend([len(t) for t in tokenize(lyrics)])

    result = box_counting_1d(all_lengths)
    if result is not None:
        bc_results_1d[lang] = result
        print(f"  {name}: D = {result['D']:.4f} "
              f"(R2 = {result['r_squared']:.4f})")

# ============================================================
# 4. BOX-COUNTING ON RANK-FREQUENCY SPACE (2D)
# ============================================================
print("\n--- Box-counting on rank-frequency space (2D) ---")

bc_results_2d = {}
for lang, name in LANGUAGES.items():
    subset = df[df["language"] == lang]
    all_tokens = []
    for lyrics in subset["plain_lyrics"]:
        all_tokens.extend(tokenize(lyrics))
    freq = Counter(all_tokens)
    counts = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1)

    points = np.column_stack([np.log10(ranks), np.log10(counts)])

    result = box_counting_2d(points)
    if result is not None:
        bc_results_2d[lang] = result
        print(f"  {name}: D_2D = {result['D']:.4f} "
              f"(R2 = {result['r_squared']:.4f})")

# Save fractal dimension results
fd_rows = []
for lang, name in LANGUAGES.items():
    row = {"language": name}
    if lang in bc_results_1d:
        row["D_wordlen_1d"] = bc_results_1d[lang]["D"]
        row["R2_wordlen_1d"] = bc_results_1d[lang]["r_squared"]
    if lang in bc_results_2d:
        row["D_rankfreq_2d"] = bc_results_2d[lang]["D"]
        row["R2_rankfreq_2d"] = bc_results_2d[lang]["r_squared"]
    fd_rows.append(row)

fd_df = pd.DataFrame(fd_rows)
fd_df.to_csv(os.path.join(OUTPUT_DIR, "fractal_dim_topic_space.csv"),
              index=False)

# ============================================================
# 5. PLOTS
# ============================================================
print("\n--- Generating plots ---")

colors = {"en": "#1f77b4", "de": "#d62728", "es": "#2ca02c"}

# --- Plot 1: Streaming rank-frequency ---
if HAS_STREAMS:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (lang, name) in enumerate(LANGUAGES.items()):
        ax = axes[i]
        r = stream_results[lang]
        ax.scatter(r["log_rank"], r["log_streams"], s=8, alpha=0.5,
                   color=colors[lang])
        x_fit = np.linspace(r["log_rank"].min(), r["log_rank"].max(), 50)
        y_fit = r["intercept"] + r["slope"] * x_fit
        ax.plot(x_fit, y_fit, "k--", linewidth=1.5,
                label=f"alpha = {r['zipf_exponent']:.3f}\nR2 = {r['r_squared']:.4f}")
        ax.set_xlabel("log10(Rank)", fontsize=11)
        ax.set_ylabel("log10(Streams)", fontsize=11)
        ax.set_title(f"{name} (n = {r['n_songs']})", fontsize=12,
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Streaming Rank-Frequency Distribution by Language",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "streaming_rank_frequency.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: streaming_rank_frequency.png")

# --- Plot 2: Box-counting log-log ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for lang, name in LANGUAGES.items():
    if lang in bc_results_1d:
        r = bc_results_1d[lang]
        ax.plot(r["log_eps"], r["log_N"], "o-", color=colors[lang],
                label=f"{name}: D={r['D']:.3f}", markersize=4, linewidth=1.5)
ax.set_xlabel("log(1/eps)", fontsize=11)
ax.set_ylabel("log(N(eps))", fontsize=11)
ax.set_title("Box-Counting: Word Length Distribution",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
for lang, name in LANGUAGES.items():
    if lang in bc_results_2d:
        r = bc_results_2d[lang]
        ax.plot(r["log_eps"], r["log_N"], "o-", color=colors[lang],
                label=f"{name}: D={r['D']:.3f}", markersize=4, linewidth=1.5)
ax.set_xlabel("log(1/eps)", fontsize=11)
ax.set_ylabel("log(N(eps))", fontsize=11)
ax.set_title("Box-Counting: Rank-Frequency Space (2D)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle("Fractal Dimension of Linguistic Distributions",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "box_counting_topic_space.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: box_counting_topic_space.png")

# --- Plot 3: Power-law comparison ---
if HAS_STREAMS:
    fig, ax = plt.subplots(figsize=(8, 5))
    names_list = [LANGUAGES[l] for l in LANGUAGES]
    exps = [stream_results[l]["zipf_exponent"] for l in LANGUAGES]
    errs = [stream_results[l]["std_err"] for l in LANGUAGES]
    bars = ax.bar(names_list, exps, yerr=errs, capsize=5,
                  color=[colors[l] for l in LANGUAGES],
                  edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.set_ylabel("Streaming Zipf Exponent", fontsize=12)
    ax.set_title("Streaming Distribution Scaling Exponents",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, exp in zip(bars, exps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{exp:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "streaming_power_law_fit.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: streaming_power_law_fit.png")

# ============================================================
# 6. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)
if HAS_STREAMS:
    print("\nStreaming Scaling Exponents:")
    print(stream_df.to_string(index=False))
print("\nFractal Dimensions:")
print(fd_df.to_string(index=False))
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("Done!")
