#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_vader_on_lrclib.py
=====================
Same-source VADER vs BERT comparison.

Reviewer 1 (Comment 2) flagged that the original paper compared:
  - VADER on Genius lyrics
  - BERT on LRCLIB lyrics
which conflates source and method. To address this, we re-run VADER on the
LRCLIB lyrics (the same source already used by BERT in script 07) and report
correlations between VADER (LRCLIB) and BERT (LRCLIB) per language.

Inputs:
  04_clean_lyrics_data/04_lyrics_cleaned.csv     (LRCLIB lyrics + language)
  07_bert_sentiment/07_bert_sentiment_results.csv (BERT scores)

Outputs:
  12_vader_lrclib/12_vader_lrclib_results.csv      VADER scores on LRCLIB
  12_vader_lrclib/12_vader_vs_bert_comparison.csv  per-language stats + correlations
  12_vader_lrclib/12_vader_lrclib_kde.png          VADER (LRCLIB) KDE by language
  12_vader_lrclib/12_vader_vs_bert_scatter.png     scatter plot
  12_vader_lrclib/12_summary.txt
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============ CONFIG ============
LYRICS_FILE = '04_clean_lyrics_data/04_lyrics_cleaned.csv'
BERT_FILE   = '07_bert_sentiment/07_bert_sentiment_results.csv'
OUTPUT_DIR  = '12_vader_lrclib'
LANGUAGES   = {'en': 'English', 'es': 'Spanish', 'de': 'German'}
LANG_COLORS = {'en': '#2171B5', 'es': '#238B45', 'de': '#CB181D'}
MIN_WORDS   = 50  # same filter as script 07
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ ACADEMIC PLOT STYLE ============
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

print("=" * 70)
print("12 - VADER ON LRCLIB LYRICS (same-source comparison with BERT)")
print("=" * 70)

# ============ STEP 1: LOAD LRCLIB LYRICS ============
print(f"\nStep 1: Loading LRCLIB lyrics from {LYRICS_FILE}")
df = pd.read_csv(LYRICS_FILE)
print(f"  Loaded {len(df):,} rows")

# Filter top-3 languages, non-null lyrics, >= 50 words (match script 07)
df = df[df['language'].isin(LANGUAGES.keys())].copy()
df = df[df['plain_lyrics'].notna()].copy()
df['_word_count'] = df['plain_lyrics'].astype(str).str.split().str.len()
df = df[df['_word_count'] >= MIN_WORDS].copy()
df = df.drop(columns=['_word_count'])
print(f"  After top-3 filter & >={MIN_WORDS} words: {len(df):,} tracks")
for lang, name in LANGUAGES.items():
    print(f"    {name}: {(df['language'] == lang).sum():,}")

# ============ STEP 2: VADER ON LRCLIB plain_lyrics ============
print("\nStep 2: Running VADER on LRCLIB plain_lyrics ...")
analyzer = SentimentIntensityAnalyzer()

def vader_scores(text):
    if not isinstance(text, str) or text.strip() == '':
        return (np.nan, np.nan, np.nan, np.nan)
    s = analyzer.polarity_scores(text)
    return (s['compound'], s['pos'], s['neg'], s['neu'])

vader_out = df['plain_lyrics'].apply(vader_scores)
df['vader_compound_lrclib'] = [v[0] for v in vader_out]
df['vader_pos_lrclib']      = [v[1] for v in vader_out]
df['vader_neg_lrclib']      = [v[2] for v in vader_out]
df['vader_neu_lrclib']      = [v[3] for v in vader_out]
df['vader_class_lrclib']    = np.where(
    df['vader_pos_lrclib'] > df['vader_neg_lrclib'], 'positive', 'negative'
)

# Save raw VADER-on-LRCLIB results
out_cols = ['track_name', 'artist', 'language', 'plain_lyrics',
            'vader_compound_lrclib', 'vader_pos_lrclib',
            'vader_neg_lrclib', 'vader_neu_lrclib', 'vader_class_lrclib']
df_out = df[out_cols].copy()
df_out.to_csv(os.path.join(OUTPUT_DIR, '12_vader_lrclib_results.csv'),
              index=False, encoding='utf-8')
print(f"  Saved: 12_vader_lrclib_results.csv ({len(df_out):,} rows)")

# ============ STEP 3: MERGE WITH BERT (script 07) ============
print(f"\nStep 3: Merging with BERT scores from {BERT_FILE}")
bert = pd.read_csv(BERT_FILE)
print(f"  Loaded BERT: {len(bert):,} rows")

# Merge on track_name + artist
merge_keys = ['track_name', 'artist']
merged = df.merge(
    bert[merge_keys + ['language', 'BERT_Composite']].rename(
        columns={'language': 'language_bert'}
    ),
    on=merge_keys, how='inner'
)
# language consistency check
merged = merged[merged['language'] == merged['language_bert']].copy()
merged = merged.drop(columns=['language_bert'])
print(f"  Merged (inner, same language): {len(merged):,} tracks")

# ============ STEP 4: PER-LANGUAGE STATS + CORRELATIONS ============
print("\nStep 4: Per-language descriptive stats and correlations")
print(f"  {'Lang':>8s}  {'n':>5s}  {'VADER_M':>9s}  {'VADER_SD':>9s}  "
      f"{'BERT_M':>9s}  {'BERT_SD':>9s}  {'Pearson':>8s}  {'Spearman':>9s}")

rows = []
for lang, name in LANGUAGES.items():
    sub = merged[merged['language'] == lang].copy()
    sub = sub.dropna(subset=['vader_compound_lrclib', 'BERT_Composite'])
    n = len(sub)
    if n < 2:
        continue
    v = sub['vader_compound_lrclib'].values
    b = sub['BERT_Composite'].values

    pearson_r, pearson_p   = stats.pearsonr(v, b)
    spearman_r, spearman_p = stats.spearmanr(v, b)

    rows.append({
        'language': name,
        'n': n,
        'vader_lrclib_M':  float(np.mean(v)),
        'vader_lrclib_SD': float(np.std(v, ddof=1)),
        'vader_lrclib_Mdn': float(np.median(v)),
        'bert_M':  float(np.mean(b)),
        'bert_SD': float(np.std(b, ddof=1)),
        'bert_Mdn': float(np.median(b)),
        'pearson_r':  float(pearson_r),
        'pearson_p':  float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'pct_neg_vader_lrclib': float((v < 0).mean() * 100),
        'pct_neg_bert':         float((b < 0).mean() * 100),
    })
    print(f"  {name:>8s}  {n:>5d}  "
          f"{np.mean(v):>9.4f}  {np.std(v, ddof=1):>9.4f}  "
          f"{np.mean(b):>9.4f}  {np.std(b, ddof=1):>9.4f}  "
          f"{pearson_r:>8.4f}  {spearman_r:>9.4f}")

# Pooled (all 3 languages together)
merged_clean = merged.dropna(subset=['vader_compound_lrclib', 'BERT_Composite'])
v_all = merged_clean['vader_compound_lrclib'].values
b_all = merged_clean['BERT_Composite'].values
pearson_r_all, pearson_p_all   = stats.pearsonr(v_all, b_all)
spearman_r_all, spearman_p_all = stats.spearmanr(v_all, b_all)
rows.append({
    'language': 'Pooled (3 langs)',
    'n': len(v_all),
    'vader_lrclib_M':  float(np.mean(v_all)),
    'vader_lrclib_SD': float(np.std(v_all, ddof=1)),
    'vader_lrclib_Mdn': float(np.median(v_all)),
    'bert_M':  float(np.mean(b_all)),
    'bert_SD': float(np.std(b_all, ddof=1)),
    'bert_Mdn': float(np.median(b_all)),
    'pearson_r':  float(pearson_r_all),
    'pearson_p':  float(pearson_p_all),
    'spearman_r': float(spearman_r_all),
    'spearman_p': float(spearman_p_all),
    'pct_neg_vader_lrclib': float((v_all < 0).mean() * 100),
    'pct_neg_bert':         float((b_all < 0).mean() * 100),
})
print(f"  {'Pooled':>8s}  {len(v_all):>5d}  "
      f"{np.mean(v_all):>9.4f}  {np.std(v_all, ddof=1):>9.4f}  "
      f"{np.mean(b_all):>9.4f}  {np.std(b_all, ddof=1):>9.4f}  "
      f"{pearson_r_all:>8.4f}  {spearman_r_all:>9.4f}")

comp_df = pd.DataFrame(rows)
comp_df.to_csv(os.path.join(OUTPUT_DIR, '12_vader_vs_bert_comparison.csv'),
               index=False, encoding='utf-8')
print(f"  Saved: 12_vader_vs_bert_comparison.csv")

# ============ STEP 5: KDE PLOT ============
print("\nStep 5: Generating KDE plot")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
for lang, name in LANGUAGES.items():
    sub = merged.loc[merged['language'] == lang,
                     'vader_compound_lrclib'].dropna().values
    if len(sub) > 0:
        sns.kdeplot(data=sub, fill=True, alpha=0.15,
                    label=f"{name} (n = {len(sub):,}, M = {sub.mean():.2f})",
                    color=LANG_COLORS[lang], linewidth=2.5, ax=ax)
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_title('VADER Compound Sentiment on LRCLIB Lyrics by Language',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('VADER Compound (LRCLIB)', fontsize=13, fontweight='bold')
ax.set_ylabel('Density', fontsize=13, fontweight='bold')
ax.set_xlim(-1.30, 1.30)
ax.legend(loc='upper center', fontsize=11, frameon=True, framealpha=0.9,
          edgecolor='gray', ncol=3, bbox_to_anchor=(0.5, -0.12))
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(os.path.join(OUTPUT_DIR, '12_vader_lrclib_kde.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: 12_vader_lrclib_kde.png")

# Scatter: VADER (LRCLIB) vs BERT, colored by language
fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
for lang, name in LANGUAGES.items():
    sub = merged[merged['language'] == lang].dropna(
        subset=['vader_compound_lrclib', 'BERT_Composite'])
    ax.scatter(sub['vader_compound_lrclib'], sub['BERT_Composite'],
               s=15, alpha=0.5, color=LANG_COLORS[lang],
               label=f"{name} (n={len(sub):,})", edgecolors='none')
ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)
ax.set_xlabel('VADER Compound (LRCLIB)', fontsize=13, fontweight='bold')
ax.set_ylabel('BERT Composite (LRCLIB)', fontsize=13, fontweight='bold')
ax.set_title(f"VADER vs BERT on Same Source (LRCLIB)\n"
             f"Pooled Pearson r = {pearson_r_all:.3f}, "
             f"Spearman rho = {spearman_r_all:.3f}",
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.legend(loc='lower right', fontsize=10, frameon=True)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '12_vader_vs_bert_scatter.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: 12_vader_vs_bert_scatter.png")

# ============ STEP 6: SUMMARY ============
summary = f"""VADER ON LRCLIB - SAME-SOURCE COMPARISON WITH BERT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 60}

PURPOSE:
  Address Reviewer 1 Comment 2: VADER and BERT must be applied to the
  same lyrics source for a fair cross-method comparison. Original paper:
    VADER on Genius, BERT on LRCLIB.
  This script: VADER and BERT both on LRCLIB plain_lyrics.

INPUTS:
  Lyrics: {LYRICS_FILE}
  BERT:   {BERT_FILE}

PROCESSING:
  - Top-3 language filter (en/es/de)
  - >= {MIN_WORDS} words (same filter as script 07)
  - VADER applied to plain_lyrics from LRCLIB
  - Inner-merged with BERT scores on (track_name, artist)

PER-LANGUAGE STATISTICS AND CORRELATIONS:
"""
for r in rows:
    summary += (
        f"\n  {r['language']}:\n"
        f"    n = {r['n']}\n"
        f"    VADER (LRCLIB):  M = {r['vader_lrclib_M']:.4f}, "
        f"SD = {r['vader_lrclib_SD']:.4f}, Mdn = {r['vader_lrclib_Mdn']:.4f}, "
        f"% neg = {r['pct_neg_vader_lrclib']:.1f}\n"
        f"    BERT (LRCLIB):   M = {r['bert_M']:.4f}, "
        f"SD = {r['bert_SD']:.4f}, Mdn = {r['bert_Mdn']:.4f}, "
        f"% neg = {r['pct_neg_bert']:.1f}\n"
        f"    Pearson  r = {r['pearson_r']:.4f}  (p = {r['pearson_p']:.2e})\n"
        f"    Spearman rho = {r['spearman_r']:.4f}  (p = {r['spearman_p']:.2e})\n"
    )

summary += f"""
OUTPUT FILES:
  12_vader_lrclib_results.csv      - VADER scores on LRCLIB per track
  12_vader_vs_bert_comparison.csv  - per-language stats and correlations
  12_vader_lrclib_kde.png          - KDE by language
  12_vader_vs_bert_scatter.png     - scatter VADER (LRCLIB) vs BERT
  12_summary.txt                   - this summary
"""

with open(os.path.join(OUTPUT_DIR, '12_summary.txt'), 'w',
          encoding='utf-8') as f:
    f.write(summary)
print(f"\n  Saved: 12_summary.txt")
print("\n" + "=" * 70)
print("SCRIPT 12 COMPLETE")
print("=" * 70)
