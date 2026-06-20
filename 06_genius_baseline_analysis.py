#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_genius_baseline_analysis.py
Baseline exploratory analysis: merge VADER sentiment with lyrics data,
filter to top-3 languages, produce KDE distribution plot.

FIX: Two-stage encoding normalization before merge to prevent
silent data loss from Latin-1 vs UTF-8 character mismatches.
Original script lost 112 tracks; this version recovers them.

Input:  04_clean_lyrics_data/04_lyrics_cleaned.csv  (UTF-8, from script 04)
        05_download_genius_data/track_sentiment_analysis(GeniusData).csv (from script 05)
Output: 06_genius_baseline_analysis/
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import unicodedata
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ================= CONFIG =================
INPUT_GENIUS = '05_download_genius_data/track_sentiment_analysis(GeniusData).csv'
INPUT_LYRICS = '04_clean_lyrics_data/04_lyrics_cleaned.csv'
OUTPUT_FOLDER = '06_genius_baseline_analysis'

TOP_N_LANGUAGES = 3
SENTIMENT_COLS = ['compound_score', 'classification', 'pos_score', 'neg_score', 'neu_score']

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= ACADEMIC PLOT STYLE =================
def set_academic_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 15,
        'axes.labelsize': 13,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'black',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 2.5,
    })

set_academic_style()

# ================= MERGE KEY FUNCTIONS =================
def normalize_stage1(s):
    """Stage 1: Unicode NFKD decomposition + strip combining marks + lowercase."""
    if pd.isna(s):
        return ''
    s = str(s).strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', s).strip().lower()


def normalize_stage2(s):
    """Stage 2: Strip ALL non-alphanumeric characters (handles ? artifacts)."""
    if pd.isna(s):
        return ''
    s = str(s).strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-zA-Z0-9]', '', s)
    return s.lower()


# ================= UTILS =================
def save_plot(fig, name):
    path = os.path.join(OUTPUT_FOLDER, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  ✓ Saved: {name}.png")


def save_csv(df, name):
    path = os.path.join(OUTPUT_FOLDER, f"{name}.csv")
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"  ✓ Saved: {name}.csv")


# ================= LOAD =================
print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

lyrics = pd.read_csv(INPUT_LYRICS, encoding='utf-8')
print(f"  Lyrics: {len(lyrics):,} records")

try:
    genius = pd.read_csv(INPUT_GENIUS, encoding='utf-8')
except UnicodeDecodeError:
    genius = pd.read_csv(INPUT_GENIUS, encoding='latin-1')
    print("  (Genius loaded with Latin-1 fallback)")
print(f"  Genius: {len(genius):,} records")

genius_with_sent = genius[genius['compound_score'].notna()].copy()
print(f"  Genius tracks with sentiment: {len(genius_with_sent):,}")

# ================= TWO-STAGE MERGE =================
print("\n" + "=" * 70)
print("TWO-STAGE ENCODING-SAFE MERGE")
print("=" * 70)

avail_sent_cols = [c for c in SENTIMENT_COLS if c in genius.columns]

# --- STAGE 1: Normalized string match ---
lyrics['_s1_t'] = lyrics['track_name'].apply(normalize_stage1)
lyrics['_s1_a'] = lyrics['artist'].apply(normalize_stage1)
genius_with_sent['_s1_t'] = genius_with_sent['track_name'].apply(normalize_stage1)
genius_with_sent['_s1_a'] = genius_with_sent['artist'].apply(normalize_stage1)

g1 = genius_with_sent[['_s1_t', '_s1_a'] + avail_sent_cols].drop_duplicates(
    subset=['_s1_t', '_s1_a'], keep='first')

merged = lyrics.merge(g1, on=['_s1_t', '_s1_a'], how='left')
stage1_matched = merged['compound_score'].notna().sum()
print(f"  Stage 1 (accent-stripped): {stage1_matched:,} matched")

# --- STAGE 2: For unmatched rows, try aggressive normalization ---
unmatched_mask = merged['compound_score'].isna()
unmatched_idx = merged[unmatched_mask].index
stage2_count = 0

if unmatched_idx.size > 0:
    unmatched_tracks = merged.loc[unmatched_idx, 'track_name'].apply(normalize_stage2)
    unmatched_artists = merged.loc[unmatched_idx, 'artist'].apply(normalize_stage2)

    genius_with_sent['_s2_t'] = genius_with_sent['track_name'].apply(normalize_stage2)
    genius_with_sent['_s2_a'] = genius_with_sent['artist'].apply(normalize_stage2)

    g2 = genius_with_sent[['_s2_t', '_s2_a'] + avail_sent_cols].drop_duplicates(
        subset=['_s2_t', '_s2_a'], keep='first')
    g2_dict = {(t, a): row for (t, a), row in g2.set_index(['_s2_t', '_s2_a'])[avail_sent_cols].iterrows()}

    for idx, s2_t, s2_a in zip(unmatched_idx, unmatched_tracks, unmatched_artists):
        key = (s2_t, s2_a)
        if key in g2_dict:
            for col in avail_sent_cols:
                merged.at[idx, col] = g2_dict[key][col]
            stage2_count += 1

    print(f"  Stage 2 (aggressive normalization): {stage2_count:,} additional matches")

# Clean up temp columns
merged.drop(columns=[c for c in merged.columns if c.startswith('_s')], inplace=True)

total_matched = merged['compound_score'].notna().sum()
total_unmatched = len(merged) - total_matched
print(f"\n  TOTAL matched: {total_matched:,} / {len(merged):,} ({total_matched/len(merged)*100:.1f}%)")
print(f"  Unmatched: {total_unmatched:,}")
print(f"  (Genius had sentiment for {len(genius_with_sent):,} tracks; "
      f"{len(genius_with_sent) - total_matched:,} could not be linked)")

df = merged.copy()

# ================= TOP-3 LANGUAGE FILTER =================
print("\n" + "=" * 70)
print("TOP-3 LANGUAGE ANALYSIS")
print("=" * 70)

LANG_NAMES = {'en': 'English', 'es': 'Spanish', 'de': 'German'}
top_languages = df['language'].value_counts().head(TOP_N_LANGUAGES).index.tolist()
df_top3 = df[df['language'].isin(top_languages)].copy()

print(f"  Top-3 languages: {', '.join(f'{LANG_NAMES.get(l,l)} ({l})' for l in top_languages)}")
print(f"  Top-3 subset: {len(df_top3):,} / {len(df):,} ({len(df_top3)/len(df)*100:.1f}%)")

for lang in top_languages:
    sub = df_top3[df_top3['language'] == lang]
    sub_sent = sub['compound_score'].dropna()
    total_lang = len(sub)
    with_sent = len(sub_sent)
    print(f"  {LANG_NAMES.get(lang, lang):>8s}: n={total_lang:,}, "
          f"with sentiment={with_sent:,} ({with_sent/total_lang*100:.1f}%), "
          f"M={sub_sent.mean():.4f}, SD={sub_sent.std():.4f}, Mdn={sub_sent.median():.4f}")

save_csv(df_top3, '05_top3_languages_baseline_dataset')

# ================= PLOT: KDE COMPOUND DISTRIBUTION =================
print("\n" + "=" * 70)
print("GENERATING PLOT")
print("=" * 70)

LANG_COLORS = {'en': '#2171B5', 'es': '#238B45', 'de': '#CB181D'}

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

for lang in top_languages:
    subset = df_top3[df_top3['language'] == lang]['compound_score'].dropna()
    if len(subset) > 0:
        label = LANG_NAMES.get(lang, lang)
        n = len(subset)
        m = subset.mean()
        sns.kdeplot(
            data=subset,
            fill=True,
            alpha=0.15,
            label=f"{label} (n = {n:,}, M = {m:.2f})",
            color=LANG_COLORS.get(lang, '#999999'),
            linewidth=2.5,
            ax=ax
        )

ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

ax.set_title('Distribution of VADER Compound Sentiment by Language',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Compound Score', fontsize=13, fontweight='bold')
ax.set_ylabel('Density', fontsize=13, fontweight='bold')
ax.set_xlim(-1.30, 1.30)

ax.legend(loc='upper center', fontsize=11, frameon=True, framealpha=0.9,
          edgecolor='gray', ncol=3, bbox_to_anchor=(0.5, -0.12))

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

fig.tight_layout(rect=[0, 0.08, 1, 1])
save_plot(fig, '01_compound_sentiment_kde_by_language')

# ================= SUMMARY =================
print("\n" + "=" * 70)
print("GENERATING SUMMARY")
print("=" * 70)

total_with_sentiment = df['compound_score'].notna().sum()
classified = df['classification'].dropna()
pos_n = (classified == 'positive').sum()
neg_n = (classified == 'negative').sum()
total_class = pos_n + neg_n

summary = f"""BASELINE SENTIMENT ANALYSIS SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

DATASET OVERVIEW:
  Total songs: {len(df):,}
  Unique languages: {df['language'].nunique()}
  Songs with VADER sentiment (after merge): {total_with_sentiment:,}
  Sentiment coverage: {total_with_sentiment/len(df)*100:.1f}%

MERGE QUALITY:
  Genius script 05 had sentiment for: {len(genius_with_sent):,} tracks
  Stage 1 matched (accent-normalized): {stage1_matched:,}
  Stage 2 matched (aggressive normalization): {stage2_count:,}
  Total matched: {total_matched:,}
  Unlinked (different tracks by same artist): {len(genius_with_sent) - total_matched:,}

TOP-3 LANGUAGE FOCUS:
  Languages: {', '.join(f'{LANG_NAMES.get(l,l)} ({l})' for l in top_languages)}
  Subset size: {len(df_top3):,} ({len(df_top3)/len(df)*100:.1f}%)

CLASSIFICATION (of {total_class:,} classified songs):
  Positive: {pos_n:,} ({pos_n/total_class*100:.1f}%)
  Negative: {neg_n:,} ({neg_n/total_class*100:.1f}%)

COMPOUND SCORE BY LANGUAGE:"""

for lang in top_languages:
    sub = df_top3[df_top3['language'] == lang]['compound_score'].dropna()
    if len(sub) > 0:
        summary += f"""
  {LANG_NAMES.get(lang, lang):>8s}: n={len(sub):,}, M={sub.mean():.4f}, SD={sub.std():.4f}, Mdn={sub.median():.4f}"""

summary += f"""

OUTPUT FILES:
  01_compound_sentiment_kde_by_language.png
  05_top3_languages_baseline_dataset.csv
  06_analysis_summary.txt
"""

summary_path = os.path.join(OUTPUT_FOLDER, '06_analysis_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)
print(f"  ✓ Summary saved")
print(summary)

print("=" * 70)
print("✓ Baseline analysis completed!")
print("=" * 70)
