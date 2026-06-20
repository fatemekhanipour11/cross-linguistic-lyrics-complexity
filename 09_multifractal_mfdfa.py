#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_multifractal_mfdfa.py
========================
Multifractal Detrended Fluctuation Analysis (MF-DFA) on Lyrics

Input:  04_clean_lyrics_data/04_lyrics_cleaned.csv
Output: 09_multifractal_mfdfa/
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
OUTPUT_DIR = "09_multifractal_mfdfa"
LANGUAGES = {"en": "English", "de": "German", "es": "Spanish"}
SEED = 42
N_BOOTSTRAP = 100
Q_RANGE = np.arange(-5, 5.5, 0.5)
Q_RANGE = Q_RANGE[Q_RANGE != 0]
MIN_SERIES_LEN = 200
# =======================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)


# ============================================================
# MF-DFA IMPLEMENTATION
# ============================================================
def mfdfa(series, q_range, n_scales=20, order=1):
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

    Fq = np.zeros((len(q_range), len(scales)))

    for si, s in enumerate(scales):
        n_seg = N // s
        if n_seg < 1:
            Fq[:, si] = np.nan
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

        if len(var_array) < 2:
            Fq[:, si] = np.nan
            continue

        for qi, q in enumerate(q_range):
            if q == 0:
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(var_array)))
            else:
                Fq[qi, si] = (np.mean(var_array ** (q / 2.0))) ** (1.0 / q)

    log_scales = np.log2(scales)
    hq = np.zeros(len(q_range))
    for qi in range(len(q_range)):
        valid = np.isfinite(Fq[qi, :]) & (Fq[qi, :] > 0)
        if valid.sum() >= 3:
            log_fq = np.log2(Fq[qi, valid])
            slope, _, r, _, _ = stats.linregress(log_scales[valid], log_fq)
            hq[qi] = slope
        else:
            hq[qi] = np.nan

    valid_hq = np.isfinite(hq)
    if valid_hq.sum() < 4:
        return None

    q_valid = q_range[valid_hq]
    h_valid = hq[valid_hq]

    tau_q = q_valid * h_valid - 1
    alpha = np.gradient(tau_q, q_valid)
    f_alpha = q_valid * alpha - tau_q

    delta_alpha = np.nanmax(alpha) - np.nanmin(alpha)

    H_idx = np.argmin(np.abs(q_range - 2))

    return {
        "scales": scales,
        "Fq": Fq,
        "q_range": q_range,
        "hq": hq,
        "tau_q": tau_q,
        "alpha": alpha,
        "f_alpha": f_alpha,
        "delta_alpha": delta_alpha,
        "H": hq[H_idx] if np.isfinite(hq[H_idx]) else np.nan,
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


def lyrics_to_series(tokens, freq_dict):
    if not tokens:
        return np.array([])
    sorted_words = sorted(freq_dict.keys(), key=lambda w: -freq_dict[w])
    rank_dict = {w: i + 1 for i, w in enumerate(sorted_words)}
    series = np.array([np.log(rank_dict.get(t, len(rank_dict)))
                       for t in tokens])
    return series


# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("09 — MULTIFRACTAL DETRENDED FLUCTUATION ANALYSIS (MF-DFA)")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
df = df[df["language"].isin(LANGUAGES.keys())].copy()
print(f"Loaded {len(df)} songs (EN/DE/ES)")
for lang, name in LANGUAGES.items():
    print(f"  {name}: {len(df[df['language'] == lang])} songs")

# ============================================================
# 2. BUILD CORPUS FREQUENCY DICTS
# ============================================================
print("\n--- Building corpus frequency dictionaries ---")

corpus_freq = {}
for lang, name in LANGUAGES.items():
    subset = df[df["language"] == lang]
    all_tokens = []
    for lyrics in subset["plain_lyrics"]:
        all_tokens.extend(tokenize(lyrics))
    corpus_freq[lang] = Counter(all_tokens)
    print(f"  {name}: {len(all_tokens):,} tokens, "
          f"{len(corpus_freq[lang]):,} types")

# ============================================================
# 3. CONCATENATED CORPUS MF-DFA
# ============================================================
print("\n--- Running MF-DFA on concatenated corpus per language ---")

mfdfa_results = {}

for lang, name in LANGUAGES.items():
    subset = df[df["language"] == lang]
    all_tokens = []
    for lyrics in subset["plain_lyrics"]:
        all_tokens.extend(tokenize(lyrics))

    series = lyrics_to_series(all_tokens, corpus_freq[lang])
    print(f"  {name}: series length = {len(series):,}")

    result = mfdfa(series, Q_RANGE)
    if result is not None:
        mfdfa_results[lang] = result
        print(f"    Delta_alpha = {result['delta_alpha']:.4f}, "
              f"H(q=2) = {result['H']:.4f}")
    else:
        print(f"    [!] MF-DFA failed for {name}")

# ============================================================
# 4. PER-SONG MF-DFA (for distribution of Delta_alpha)
# ============================================================
print("\n--- Running per-song MF-DFA ---")

song_results = {lang: [] for lang in LANGUAGES}

for lang, name in LANGUAGES.items():
    subset = df[df["language"] == lang]
    n_valid = 0
    for _, row in subset.iterrows():
        tokens = tokenize(row["plain_lyrics"])
        if len(tokens) < MIN_SERIES_LEN:
            continue
        series = lyrics_to_series(tokens, corpus_freq[lang])
        result = mfdfa(series, Q_RANGE, n_scales=12)
        if result is not None:
            song_results[lang].append(result["delta_alpha"])
            n_valid += 1
    print(f"  {name}: {n_valid} songs with valid MF-DFA "
          f"(>={MIN_SERIES_LEN} tokens)")
    if n_valid > 0:
        arr = np.array(song_results[lang])
        print(f"    Delta_alpha: mean={arr.mean():.4f}, "
              f"median={np.median(arr):.4f}, std={arr.std():.4f}")

# ============================================================
# 5. BOOTSTRAP SUBSAMPLING TEST
# ============================================================
print(f"\n--- Bootstrap subsampling (n={N_BOOTSTRAP} resamples) ---")
print("  Testing if multifractality ranking holds when "
      "controlling for sample size")

min_n = min(len(df[df["language"] == lang]) for lang in LANGUAGES)
print(f"  Subsampling each language to n = {min_n} songs")

bootstrap_results = {lang: [] for lang in LANGUAGES}

for b in range(N_BOOTSTRAP):
    if (b + 1) % 20 == 0:
        print(f"    Bootstrap {b+1}/{N_BOOTSTRAP}...")
    for lang in LANGUAGES:
        subset = df[df["language"] == lang].sample(n=min_n,
                                                     random_state=SEED + b)
        all_tokens = []
        for lyrics in subset["plain_lyrics"]:
            all_tokens.extend(tokenize(lyrics))
        series = lyrics_to_series(all_tokens, corpus_freq[lang])
        result = mfdfa(series, Q_RANGE, n_scales=15)
        if result is not None:
            bootstrap_results[lang].append(result["delta_alpha"])

print("\n  Bootstrap Delta_alpha distributions:")
for lang, name in LANGUAGES.items():
    arr = np.array(bootstrap_results[lang])
    if len(arr) > 0:
        print(f"    {name}: mean={arr.mean():.4f} +/- {arr.std():.4f} "
              f"(n={len(arr)} successful)")

# Pairwise comparisons
print("\n  Pairwise Mann-Whitney U tests:")
lang_list = list(LANGUAGES.keys())
for i in range(len(lang_list)):
    for j in range(i + 1, len(lang_list)):
        l1, l2 = lang_list[i], lang_list[j]
        a1 = np.array(bootstrap_results[l1])
        a2 = np.array(bootstrap_results[l2])
        if len(a1) > 0 and len(a2) > 0:
            U, p = stats.mannwhitneyu(a1, a2, alternative="two-sided")
            print(f"    {LANGUAGES[l1]} vs {LANGUAGES[l2]}: "
                  f"U={U:.0f}, p={p:.4e}")

# ============================================================
# 6. SAVE RESULTS
# ============================================================
print("\n--- Saving results ---")

results_table = []
for lang, name in LANGUAGES.items():
    row = {"language": name, "n_songs": len(df[df["language"] == lang])}
    if lang in mfdfa_results:
        r = mfdfa_results[lang]
        row["delta_alpha_corpus"] = r["delta_alpha"]
        row["H_q2"] = r["H"]
    if song_results[lang]:
        arr = np.array(song_results[lang])
        row["delta_alpha_song_mean"] = arr.mean()
        row["delta_alpha_song_std"] = arr.std()
        row["delta_alpha_song_median"] = np.median(arr)
    if bootstrap_results[lang]:
        arr = np.array(bootstrap_results[lang])
        row["delta_alpha_bootstrap_mean"] = arr.mean()
        row["delta_alpha_bootstrap_std"] = arr.std()
    results_table.append(row)

results_df = pd.DataFrame(results_table)
results_df.to_csv(os.path.join(OUTPUT_DIR, "mfdfa_results_table.csv"),
                   index=False)

boot_rows = []
for lang, name in LANGUAGES.items():
    for val in bootstrap_results[lang]:
        boot_rows.append({"language": name, "delta_alpha": val})
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(os.path.join(OUTPUT_DIR, "mfdfa_bootstrap_results.csv"),
                index=False)

# ============================================================
# 7. PLOTS
# ============================================================
print("\n--- Generating plots ---")

colors = {"en": "#1f77b4", "de": "#d62728", "es": "#2ca02c"}

# --- Plot 1: Generalized Hurst exponent h(q) ---
fig, ax = plt.subplots(figsize=(8, 5))
for lang, name in LANGUAGES.items():
    if lang in mfdfa_results:
        r = mfdfa_results[lang]
        valid = np.isfinite(r["hq"])
        ax.plot(r["q_range"][valid], r["hq"][valid], "o-",
                color=colors[lang], label=name, markersize=4, linewidth=1.5)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5,
           label="H=0.5 (random)")
ax.set_xlabel("q (Moment Order)", fontsize=12)
ax.set_ylabel("h(q) (Generalized Hurst Exponent)", fontsize=12)
ax.set_title("Generalized Hurst Exponent h(q)\n"
             "(Non-constant h(q) = Multifractal)", fontsize=13,
             fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "mfdfa_hq_curves.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: mfdfa_hq_curves.png")

# --- Plot 2: Singularity spectrum f(alpha) ---
fig, ax = plt.subplots(figsize=(8, 5))
for lang, name in LANGUAGES.items():
    if lang in mfdfa_results:
        r = mfdfa_results[lang]
        valid = np.isfinite(r["alpha"]) & np.isfinite(r["f_alpha"])
        ax.plot(r["alpha"][valid], r["f_alpha"][valid], "o-",
                color=colors[lang],
                label=f"{name} (Delta_alpha = {r['delta_alpha']:.3f})",
                markersize=4, linewidth=1.5)
ax.set_xlabel("alpha (Singularity Strength)", fontsize=12)
ax.set_ylabel("f(alpha) (Singularity Spectrum)", fontsize=12)
ax.set_title("Multifractal Singularity Spectrum f(alpha)\n"
             "(Wider spectrum = Stronger multifractality)", fontsize=13,
             fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "mfdfa_singularity_spectrum.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: mfdfa_singularity_spectrum.png")

# --- Plot 3: Spectrum width comparison ---
fig, ax = plt.subplots(figsize=(8, 5))
names_list = []
widths = []
color_list = []
for lang, name in LANGUAGES.items():
    if lang in mfdfa_results:
        names_list.append(name)
        widths.append(mfdfa_results[lang]["delta_alpha"])
        color_list.append(colors[lang])
bars = ax.bar(names_list, widths, color=color_list,
              edgecolor="black", linewidth=0.8, alpha=0.85)
ax.set_ylabel("Delta_alpha (Multifractal Spectrum Width)", fontsize=12)
ax.set_title("Multifractality Comparison Across Languages",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
for bar, w in zip(bars, widths):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{w:.3f}", ha="center", va="bottom", fontsize=11,
            fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "mfdfa_spectrum_width_comparison.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: mfdfa_spectrum_width_comparison.png")

# --- Plot 4: Bootstrap test ---
fig, ax = plt.subplots(figsize=(8, 5))
bp_data = []
bp_labels = []
bp_colors = []
for lang, name in LANGUAGES.items():
    arr = np.array(bootstrap_results[lang])
    if len(arr) > 0:
        bp_data.append(arr)
        bp_labels.append(name)
        bp_colors.append(colors[lang])

if bp_data:
    bp = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], bp_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

ax.set_ylabel("Delta_alpha (Multifractal Spectrum Width)", fontsize=12)
ax.set_title(f"Bootstrap Subsampling Test (n={min_n} songs each, "
             f"{N_BOOTSTRAP} resamples)\nControlling for Sample Size",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "mfdfa_bootstrap_test.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: mfdfa_bootstrap_test.png")

# ============================================================
# 8. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)
print("\nCorpus-level MF-DFA:")
print(results_df.to_string(index=False))
print(f"\nBootstrap subsampling: n={min_n} songs per language, "
      f"{N_BOOTSTRAP} resamples")
for lang, name in LANGUAGES.items():
    arr = np.array(bootstrap_results[lang])
    if len(arr) > 0:
        print(f"  {name}: Delta_alpha = {arr.mean():.4f} +/- {arr.std():.4f}")
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("Done!")
