#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_zipf_power_law.py
====================
Zipf's Law & Power-Law Analysis of Word Frequency Distributions
per Language (English, German, Spanish)

Input:  04_clean_lyrics_data/04_lyrics_cleaned.csv
Output: 08_zipf_power_law/
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
OUTPUT_DIR = "08_zipf_power_law"
LANGUAGES = {"en": "English", "de": "German", "es": "Spanish"}
SEED = 42
# =======================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("08 — ZIPF'S LAW & POWER-LAW ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} songs total")

df = df[df["language"].isin(LANGUAGES.keys())].copy()
print(f"After filtering to EN/DE/ES: {len(df)} songs")
for lang, name in LANGUAGES.items():
    n = len(df[df["language"] == lang])
    print(f"  {name}: {n} songs")

# ============================================================
# 2. TOKENIZE AND BUILD WORD FREQUENCY DISTRIBUTIONS
# ============================================================
print("\n--- Building word frequency distributions ---")

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñäöß\w\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]

freq_by_lang = {}
tokens_by_lang = {}

for lang, name in LANGUAGES.items():
    subset = df[df["language"] == lang]
    all_tokens = []
    for lyrics in subset["plain_lyrics"]:
        all_tokens.extend(tokenize(lyrics))
    freq = Counter(all_tokens)
    freq_by_lang[lang] = freq
    tokens_by_lang[lang] = all_tokens
    print(f"  {name}: {len(all_tokens):,} total tokens, "
          f"{len(freq):,} unique types")

# ============================================================
# 3. ZIPF'S LAW: LOG-LOG REGRESSION
# ============================================================
print("\n--- Fitting Zipf's Law (log rank vs log frequency) ---")

zipf_results = {}

for lang, name in LANGUAGES.items():
    freq = freq_by_lang[lang]
    counts = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1)
    log_rank = np.log10(ranks)
    log_freq = np.log10(counts)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_rank, log_freq
    )
    zipf_results[lang] = {
        "language": name,
        "zipf_exponent": -slope,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
        "intercept": intercept,
        "n_types": len(counts),
        "n_tokens": len(tokens_by_lang[lang]),
        "ranks": ranks,
        "counts": counts,
        "log_rank": log_rank,
        "log_freq": log_freq,
    }
    print(f"  {name}: Zipf exponent α = {-slope:.4f} "
          f"(R² = {r_value**2:.4f}, p = {p_value:.2e})")

# ============================================================
# 4. VOCABULARY RICHNESS METRICS
# ============================================================
print("\n--- Computing vocabulary richness metrics ---")

vocab_metrics = []
for lang, name in LANGUAGES.items():
    freq = freq_by_lang[lang]
    tokens = tokens_by_lang[lang]
    n_tokens = len(tokens)
    n_types = len(freq)
    hapax = sum(1 for c in freq.values() if c == 1)
    dis_legomena = sum(1 for c in freq.values() if c == 2)

    ttr = n_types / n_tokens if n_tokens > 0 else 0
    hapax_ratio = hapax / n_types if n_types > 0 else 0
    herdan_c = np.log(n_types) / np.log(n_tokens) if n_tokens > 1 else 0
    brunet_w = n_tokens ** (n_types ** -0.172) if n_types > 0 else 0

    row = {
        "language": name,
        "n_songs": len(df[df["language"] == lang]),
        "n_tokens": n_tokens,
        "n_types": n_types,
        "ttr": ttr,
        "herdan_c": herdan_c,
        "brunet_w": brunet_w,
        "hapax_legomena": hapax,
        "hapax_ratio": hapax_ratio,
        "dis_legomena": dis_legomena,
    }
    vocab_metrics.append(row)
    print(f"  {name}: TTR={ttr:.4f}, Herdan C={herdan_c:.4f}, "
          f"hapax={hapax} ({hapax_ratio:.2%} of types)")

vocab_df = pd.DataFrame(vocab_metrics)
vocab_df.to_csv(os.path.join(OUTPUT_DIR, "vocabulary_metrics.csv"), index=False)

# ============================================================
# 5. POWER-LAW FIT: CLAUSET-SHALIZI-NEWMAN METHOD
# ============================================================
print("\n--- Power-law fitting (Clauset-Shalizi-Newman) ---")

try:
    import powerlaw
    HAS_POWERLAW = True
except ImportError:
    print("  [!] 'powerlaw' package not installed. "
          "Install with: pip install powerlaw")
    print("  [!] Falling back to manual MLE estimation.")
    HAS_POWERLAW = False

pl_results = []

for lang, name in LANGUAGES.items():
    freq = freq_by_lang[lang]
    counts = np.array(sorted(freq.values(), reverse=True), dtype=float)

    if HAS_POWERLAW:
        fit = powerlaw.Fit(counts, discrete=True, verbose=False)
        alpha = fit.power_law.alpha
        xmin = fit.power_law.xmin
        sigma = fit.power_law.sigma
        R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
        R_exp, p_exp = fit.distribution_compare("power_law", "exponential")
        ks_D = fit.power_law.D

        row = {
            "language": name,
            "alpha_mle": alpha,
            "xmin": xmin,
            "sigma": sigma,
            "ks_D": ks_D,
            "R_vs_lognormal": R_ln,
            "p_vs_lognormal": p_ln,
            "R_vs_exponential": R_exp,
            "p_vs_exponential": p_exp,
        }
    else:
        xmin = 1
        above = counts[counts >= xmin]
        alpha = 1 + len(above) / np.sum(np.log(above / (xmin - 0.5)))
        row = {
            "language": name,
            "alpha_mle": alpha,
            "xmin": xmin,
            "sigma": np.nan,
            "ks_D": np.nan,
            "R_vs_lognormal": np.nan,
            "p_vs_lognormal": np.nan,
            "R_vs_exponential": np.nan,
            "p_vs_exponential": np.nan,
        }

    pl_results.append(row)
    print(f"  {name}: α_MLE = {row['alpha_mle']:.4f}, "
          f"x_min = {row['xmin']}")

pl_df = pd.DataFrame(pl_results)
pl_df.to_csv(os.path.join(OUTPUT_DIR, "powerlaw_fit_summary.csv"), index=False)

# ============================================================
# 6. SAVE ZIPF EXPONENTS TABLE
# ============================================================
zipf_table = []
for lang, name in LANGUAGES.items():
    r = zipf_results[lang]
    zipf_table.append({
        "language": name,
        "n_songs": len(df[df["language"] == lang]),
        "n_tokens": r["n_tokens"],
        "n_types": r["n_types"],
        "zipf_exponent": r["zipf_exponent"],
        "r_squared": r["r_squared"],
        "std_err": r["std_err"],
    })
zipf_df = pd.DataFrame(zipf_table)
zipf_df.to_csv(os.path.join(OUTPUT_DIR, "zipf_exponents_table.csv"), index=False)

# ============================================================
# 7. PLOTS
# ============================================================
print("\n--- Generating plots ---")

colors = {"en": "#1f77b4", "de": "#d62728", "es": "#2ca02c"}

# --- Plot 1: Zipf's Law per language ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (lang, name) in enumerate(LANGUAGES.items()):
    ax = axes[i]
    r = zipf_results[lang]
    ax.scatter(r["log_rank"], r["log_freq"], s=2, alpha=0.4,
               color=colors[lang], label="Data")
    x_fit = np.linspace(r["log_rank"].min(), r["log_rank"].max(), 100)
    y_fit = r["intercept"] + (-r["zipf_exponent"]) * x_fit
    ax.plot(x_fit, y_fit, "k--", linewidth=1.5,
            label=f"α = {r['zipf_exponent']:.3f}\nR² = {r['r_squared']:.4f}")
    ax.set_xlabel("log10(Rank)", fontsize=11)
    ax.set_ylabel("log10(Frequency)", fontsize=11)
    ax.set_title(f"{name} (n = {r['n_types']:,} types)", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

fig.suptitle("Zipf's Law in Song Lyrics by Language", fontsize=14,
             fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "zipf_all_languages.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: zipf_all_languages.png")

# --- Plot 2: Exponent comparison bar chart ---
fig, ax = plt.subplots(figsize=(8, 5))
names_list = [LANGUAGES[l] for l in LANGUAGES]
exponents = [zipf_results[l]["zipf_exponent"] for l in LANGUAGES]
errs = [zipf_results[l]["std_err"] for l in LANGUAGES]
bars = ax.bar(names_list, exponents, yerr=errs, capsize=5,
              color=[colors[l] for l in LANGUAGES], edgecolor="black",
              linewidth=0.8, alpha=0.85)
ax.set_ylabel("Zipf Exponent (alpha)", fontsize=12)
ax.set_title("Zipf Exponent Comparison Across Languages", fontsize=13,
             fontweight="bold")
ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5,
           label="alpha = 1 (classic Zipf)")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

for bar, exp in zip(bars, exponents):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{exp:.3f}", ha="center", va="bottom", fontsize=11,
            fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "zipf_exponents_comparison.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: zipf_exponents_comparison.png")

# --- Plot 3: Vocabulary richness comparison ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

metrics = [("ttr", "Type-Token Ratio"), ("herdan_c", "Herdan's C"),
           ("hapax_ratio", "Hapax Ratio")]
for i, (col, label) in enumerate(metrics):
    ax = axes[i]
    vals = [vocab_df[vocab_df["language"] == LANGUAGES[l]][col].values[0]
            for l in LANGUAGES]
    bars = ax.bar([LANGUAGES[l] for l in LANGUAGES], vals,
                  color=[colors[l] for l in LANGUAGES], edgecolor="black",
                  linewidth=0.8, alpha=0.85)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)

fig.suptitle("Vocabulary Richness Metrics by Language", fontsize=14,
             fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "vocabulary_richness.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: vocabulary_richness.png")

# ============================================================
# 8. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)
print("\nZipf Exponents:")
print(zipf_df.to_string(index=False))
print("\nVocabulary Metrics:")
print(vocab_df.to_string(index=False))
if HAS_POWERLAW:
    print("\nPower-Law Fits:")
    print(pl_df.to_string(index=False))
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print("Done!")
