#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_hurst_long_range.py
======================
Long-Range Dependence Analysis: Hurst Exponent & DFA

Input:  04_clean_lyrics_data/04_lyrics_cleaned.csv
Output: 10_hurst_long_range/
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
OUTPUT_DIR = "10_hurst_long_range"
LANGUAGES = {"en": "English", "de": "German", "es": "Spanish"}
SEED = 42
N_SURROGATES = 50
# =======================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)


# ============================================================
# DFA IMPLEMENTATION
# ============================================================
def dfa(series, n_scales=20, order=1):
    N = len(series)
    if N < 50:
        return None

    Y = np.cumsum(series - np.mean(series))

    min_scale = max(10, order + 2)
    max_scale = N // 4
    if max_scale <= min_scale:
        return None
    scales = np.unique(np.logspace(
        np.log10(min_scale), np.log10(max_scale), n_scales
    ).astype(int))
    scales = scales[scales >= min_scale]

    if len(scales) < 4:
        return None

    F = np.zeros(len(scales))
    for si, s in enumerate(scales):
        n_seg = N // s
        if n_seg < 1:
            F[si] = np.nan
            continue

        var_list = []
        for v in range(n_seg):
            segment = Y[v * s : (v + 1) * s]
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            var_list.append(np.mean((segment - trend) ** 2))

        for v in range(n_seg):
            segment = Y[N - (v + 1) * s : N - v * s]
            x = np.arange(s)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            var_list.append(np.mean((segment - trend) ** 2))

        var_array = np.array(var_list)
        var_array = var_array[var_array > 0]
        if len(var_array) > 0:
            F[si] = np.sqrt(np.mean(var_array))
        else:
            F[si] = np.nan

    valid = np.isfinite(F) & (F > 0)
    if valid.sum() < 3:
        return None

    log_s = np.log2(scales[valid])
    log_F = np.log2(F[valid])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_s, log_F)

    return {
        "H": slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
        "scales": scales[valid],
        "F": F[valid],
        "log_s": log_s,
        "log_F": log_F,
        "intercept": intercept,
    }


# ============================================================
# TOKENIZATION
# ============================================================
def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñäöß\w\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("10 — HURST EXPONENT & LONG-RANGE DEPENDENCE ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
df = df[df["language"].isin(LANGUAGES.keys())].copy()
print(f"Loaded {len(df)} songs (EN/DE/ES)")
for lang, name in LANGUAGES.items():
    print(f"  {name}: {len(df[df['language'] == lang])} songs")

# ============================================================
# 2. BUILD WORD-LENGTH TIME SERIES PER LANGUAGE
# ============================================================
print("\n--- Building word-length time series ---")

series_by_lang = {}
for lang, name in LANGUAGES.items():
    subset = df[df["language"] == lang]
    all_lengths = []
    for lyrics in subset["plain_lyrics"]:
        tokens = tokenize(lyrics)
        all_lengths.extend([len(t) for t in tokens])
    series_by_lang[lang] = np.array(all_lengths, dtype=float)
    print(f"  {name}: {len(all_lengths):,} word lengths")

# ============================================================
# 3. BUILD WORD FREQUENCY RANK SERIES
# ============================================================
print("\n--- Building word frequency rank series ---")

rank_series_by_lang = {}
for lang, name in LANGUAGES.items():
    subset = df[df["language"] == lang]
    all_tokens = []
    for lyrics in subset["plain_lyrics"]:
        all_tokens.extend(tokenize(lyrics))
    freq = Counter(all_tokens)
    sorted_words = sorted(freq.keys(), key=lambda w: -freq[w])
    rank_dict = {w: i + 1 for i, w in enumerate(sorted_words)}
    rank_series = np.array([np.log(rank_dict[t]) for t in all_tokens])
    rank_series_by_lang[lang] = rank_series
    print(f"  {name}: {len(rank_series):,} tokens (log-rank encoded)")

# ============================================================
# 4. DFA ON WORD-LENGTH SERIES
# ============================================================
print("\n--- DFA on word-length series ---")

dfa_wordlen = {}
for lang, name in LANGUAGES.items():
    result = dfa(series_by_lang[lang])
    if result is not None:
        dfa_wordlen[lang] = result
        print(f"  {name}: H = {result['H']:.4f} "
              f"(R2 = {result['r_squared']:.4f})")
    else:
        print(f"  {name}: DFA failed")

# ============================================================
# 5. DFA ON WORD FREQUENCY RANK SERIES
# ============================================================
print("\n--- DFA on word frequency rank series ---")

dfa_rank = {}
for lang, name in LANGUAGES.items():
    result = dfa(rank_series_by_lang[lang])
    if result is not None:
        dfa_rank[lang] = result
        print(f"  {name}: H = {result['H']:.4f} "
              f"(R2 = {result['r_squared']:.4f})")
    else:
        print(f"  {name}: DFA failed")

# ============================================================
# 6. SURROGATE TEST (SHUFFLE TO DESTROY CORRELATIONS)
# ============================================================
print(f"\n--- Surrogate test ({N_SURROGATES} shuffled replicates) ---")

surrogate_H = {lang: {"wordlen": [], "rank": []} for lang in LANGUAGES}

for b in range(N_SURROGATES):
    if (b + 1) % 10 == 0:
        print(f"    Surrogate {b+1}/{N_SURROGATES}...")
    for lang in LANGUAGES:
        # Word length
        shuffled = series_by_lang[lang].copy()
        np.random.shuffle(shuffled)
        result = dfa(shuffled, n_scales=15)
        if result is not None:
            surrogate_H[lang]["wordlen"].append(result["H"])

        # Rank
        shuffled = rank_series_by_lang[lang].copy()
        np.random.shuffle(shuffled)
        result = dfa(shuffled, n_scales=15)
        if result is not None:
            surrogate_H[lang]["rank"].append(result["H"])

print("\n  Surrogate comparison:")
for lang, name in LANGUAGES.items():
    for series_type in ["wordlen", "rank"]:
        surr = np.array(surrogate_H[lang][series_type])
        dfa_dict = dfa_wordlen if series_type == "wordlen" else dfa_rank
        orig = dfa_dict[lang]["H"] if lang in dfa_dict else np.nan
        if len(surr) > 0:
            p_val = np.mean(surr >= orig)
            print(f"    {name} ({series_type}): "
                  f"H_orig={orig:.4f}, H_surr={surr.mean():.4f}+/-{surr.std():.4f}, "
                  f"p={p_val:.4f}")

# ============================================================
# 7. SAVE RESULTS
# ============================================================
print("\n--- Saving results ---")

rows = []
for lang, name in LANGUAGES.items():
    row = {"language": name, "n_songs": len(df[df["language"] == lang])}
    if lang in dfa_wordlen:
        row["H_wordlen"] = dfa_wordlen[lang]["H"]
        row["R2_wordlen"] = dfa_wordlen[lang]["r_squared"]
    if lang in dfa_rank:
        row["H_rank"] = dfa_rank[lang]["H"]
        row["R2_rank"] = dfa_rank[lang]["r_squared"]
    rows.append(row)

results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(OUTPUT_DIR, "hurst_results_table.csv"),
                   index=False)

surr_rows = []
for lang, name in LANGUAGES.items():
    for st in ["wordlen", "rank"]:
        for val in surrogate_H[lang][st]:
            surr_rows.append({"language": name, "series_type": st,
                              "H_surrogate": val})
surr_df = pd.DataFrame(surr_rows)
surr_df.to_csv(os.path.join(OUTPUT_DIR, "surrogate_test_results.csv"),
                index=False)

# ============================================================
# 8. PLOTS
# ============================================================
print("\n--- Generating plots ---")

colors = {"en": "#1f77b4", "de": "#d62728", "es": "#2ca02c"}

# --- Plot 1: DFA log-log plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (dfa_dict, title_type) in enumerate([
    (dfa_wordlen, "Word Length"),
    (dfa_rank, "Word Frequency Rank"),
]):
    ax = axes[i]
    for lang, name in LANGUAGES.items():
        if lang in dfa_dict:
            r = dfa_dict[lang]
            ax.scatter(r["log_s"], r["log_F"], s=30, color=colors[lang],
                       alpha=0.7, zorder=3)
            x_fit = np.linspace(r["log_s"].min(), r["log_s"].max(), 50)
            y_fit = r["intercept"] + r["H"] * x_fit
            ax.plot(x_fit, y_fit, "--", color=colors[lang], linewidth=1.5,
                    label=f"{name}: H={r['H']:.3f}")
    ax.set_xlabel("log2(scale)", fontsize=11)
    ax.set_ylabel("log2(F(s))", fontsize=11)
    ax.set_title(f"DFA: {title_type} Series", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Detrended Fluctuation Analysis by Language",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "dfa_log_log_plots.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: dfa_log_log_plots.png")

# --- Plot 2: Hurst exponent comparison ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (dfa_dict, title_type) in enumerate([
    (dfa_wordlen, "Word Length"),
    (dfa_rank, "Word Frequency Rank"),
]):
    ax = axes[i]
    names_plot = []
    h_vals = []
    errs = []
    c_list = []
    for lang, name in LANGUAGES.items():
        if lang in dfa_dict:
            names_plot.append(name)
            h_vals.append(dfa_dict[lang]["H"])
            errs.append(dfa_dict[lang]["std_err"])
            c_list.append(colors[lang])
    bars = ax.bar(names_plot, h_vals, yerr=errs, capsize=5,
                  color=c_list, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.6,
               label="H=0.5 (uncorrelated)")
    ax.set_ylabel("Hurst Exponent (H)", fontsize=11)
    ax.set_title(f"Hurst Exponent: {title_type}", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, h in zip(bars, h_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "hurst_comparison.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: hurst_comparison.png")

# --- Plot 3: Surrogate test ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, series_type in enumerate(["wordlen", "rank"]):
    ax = axes[i]
    dfa_dict = dfa_wordlen if series_type == "wordlen" else dfa_rank
    label = "Word Length" if series_type == "wordlen" else "Frequency Rank"

    for lang, name in LANGUAGES.items():
        surr = np.array(surrogate_H[lang][series_type])
        orig_H = dfa_dict[lang]["H"] if lang in dfa_dict else np.nan
        if len(surr) > 0:
            ax.hist(surr, bins=15, alpha=0.4, color=colors[lang],
                    label=f"{name} surrogates", density=True)
            ax.axvline(orig_H, color=colors[lang], linewidth=2,
                       linestyle="--", label=f"{name} observed: {orig_H:.3f}")

    ax.axvline(0.5, color="gray", linewidth=1, linestyle=":",
               label="H=0.5 (expected)")
    ax.set_xlabel("Hurst Exponent (H)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Surrogate Test: {label} Series", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "hurst_surrogate_test.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: hurst_surrogate_test.png")

# ============================================================
# 9. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)
print("\nHurst Exponents:")
print(results_df.to_string(index=False))
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("Done!")
