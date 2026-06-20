"""
07_sentiment_preprocessing.py
Multilingual BERT sentiment analysis on cleaned lyrics.

Performs:
- Lyrics text cleaning (stopword removal, lowercasing, special character removal)
  NOTE: lyrics_cleaned is produced for downstream Zipf/DFA scripts (08-11),
        but BERT scores the original plain_lyrics to preserve natural sentence
        structure that the transformer model requires for accurate inference.
- Sentiment scoring using nlptown/bert-base-multilingual-uncased-sentiment
- Outputs 5-star probabilities, composite score, and weighted average per track
- Generates one publication-quality plot: BERT sentiment distribution by language

Input:  06_genius_baseline_analysis/05_top3_languages_baseline_dataset.csv
Output: 07_bert_sentiment/
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import re
import string
import json
import time
import numpy as np
import pandas as pd
import polars as pl
import nltk
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = '06_genius_baseline_analysis/05_top3_languages_baseline_dataset.csv'
OUTPUT_FOLDER = '07_bert_sentiment'
BERT_MODEL = 'nlptown/bert-base-multilingual-uncased-sentiment'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================================
# ACADEMIC PLOT STYLE
# ============================================================================
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

# ============================================================================
# LYRICS PREPROCESSING
# ============================================================================
class LyricsPreprocessor:
    """Multilingual text preprocessing for lyrics analysis."""

    def __init__(self):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        self.all_stopwords = (
            set(stopwords.words('english'))
            | set(stopwords.words('spanish'))
            | set(stopwords.words('german'))
            | {
                'yeah', 'oh', 'uh', 'hmm', 'ah', 'ahh', 'ooh', 'chorus',
                'verse', 'intro', 'outro', 'like', 'know', 'wanna', 'gonna',
                'got', 'em', 'im', 'thats', 'ain', 'ja', 'woo', 'que', 'pa',
                'yo', 'si', 'na', 'ey', 'mhh', 'hab', 'uhh', 'ohoh', 'ye',
                'haan', 'ayy', 'noo', 'eh', 'va', 'vvs', 'yeh', 'brr', 'uah',
                'dj', 'wuh', 'get', 'go', 'say', 'make', 'want', 'querer',
                'saber', 'hacer', 'immer', 'komm', 'el', 'der', 'la', 'los',
                'las', 'und', 'die', 'das',
            }
        )

    def clean(self, lyrics: str) -> str:
        """Clean and preprocess lyrics text."""
        if not isinstance(lyrics, str) or lyrics.strip() == '':
            return ''

        text = lyrics.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Keep only letters and accented characters
        text = re.sub(r'[^a-z\u00e4\u00f6\u00fc\u00df\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1 ]', ' ', text)
        # Remove punctuation and digits
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        # Remove repetitive patterns
        text = re.sub(r'\b(\w{2,})\1{1,}\b', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove stopwords
        words = [w for w in text.split() if w not in self.all_stopwords and len(w) > 1]
        return ' '.join(words)


# ============================================================================
# BERT SENTIMENT ANALYZER
# ============================================================================
class BERTSentimentAnalyzer:
    """BERT-based multilingual sentiment scoring."""

    def __init__(self, model_name: str = BERT_MODEL):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Loading BERT model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        print(f"  Model loaded: {model_name}")

        self.struct_dtype = pl.Struct([
            pl.Field('BERT_1star', pl.Float64),
            pl.Field('BERT_2star', pl.Float64),
            pl.Field('BERT_3star', pl.Float64),
            pl.Field('BERT_4star', pl.Float64),
            pl.Field('BERT_5star', pl.Float64),
            pl.Field('BERT_Composite', pl.Float64),
            pl.Field('BERT_WeightedAvg', pl.Float64),
        ])

    def score(self, text: str) -> dict:
        """Return 5-star probabilities and composite scores for one text."""
        if not text or text.strip() == '':
            return {k: np.nan for k in [
                'BERT_1star','BERT_2star','BERT_3star','BERT_4star','BERT_5star',
                'BERT_Composite','BERT_WeightedAvg']}

        try:
            inputs = self.tokenizer(
                text, return_tensors='pt', truncation=True,
                max_length=512, padding=True, add_special_tokens=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()

            return {
                'BERT_1star': float(probs[0]),
                'BERT_2star': float(probs[1]),
                'BERT_3star': float(probs[2]),
                'BERT_4star': float(probs[3]),
                'BERT_5star': float(probs[4]),
                'BERT_Composite': float(np.dot(probs, [-1.0, -0.5, 0.0, 0.5, 1.0])),
                'BERT_WeightedAvg': float(np.dot(probs, [1, 2, 3, 4, 5]) / 5),
            }
        except Exception as e:
            print(f"  [WARN] BERT failed on text ({len(text)} chars): {e}")
            return {k: np.nan for k in [
                'BERT_1star','BERT_2star','BERT_3star','BERT_4star','BERT_5star',
                'BERT_Composite','BERT_WeightedAvg']}


# ============================================================================
# PLOT: BERT SENTIMENT BY LANGUAGE
# ============================================================================
LANG_NAMES = {'en': 'English', 'es': 'Spanish', 'de': 'German'}
LANG_COLORS = {'en': '#2171B5', 'es': '#238B45', 'de': '#CB181D'}

def plot_bert_by_language(df, output_folder):
    """KDE plot of BERT_Composite by language."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for lang in ['en', 'es', 'de']:
        subset = df.filter(pl.col('language') == lang)['BERT_Composite'].drop_nulls().to_numpy()
        if len(subset) > 0:
            label = f"{LANG_NAMES[lang]} (n = {len(subset):,}, M = {subset.mean():.2f})"
            sns.kdeplot(data=subset, fill=True, alpha=0.15, label=label,
                        color=LANG_COLORS[lang], linewidth=2.5, ax=ax)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title('Distribution of BERT Composite Sentiment by Language',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('BERT Composite Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Density', fontsize=13, fontweight='bold')
    ax.set_xlim(-1.25, 1.25)
    ax.legend(loc='upper center', fontsize=11, frameon=True, framealpha=0.9,
              edgecolor='gray', ncol=3, bbox_to_anchor=(0.5, -0.12))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(output_folder, '01_bert_sentiment_kde_by_language.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    start_time = time.time()

    print('=' * 70)
    print('BERT SENTIMENT ANALYSIS PIPELINE')
    print('=' * 70)

    # --- Step 1: Load data ---
    print(f"\nStep 1: Loading data from {INPUT_FILE}")
    df = pl.read_csv(INPUT_FILE)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # --- Step 2: Clean lyrics (for downstream Zipf/DFA scripts 08-11) ---
    print("\nStep 2: Cleaning lyrics (for downstream fractal analyses)...")
    preprocessor = LyricsPreprocessor()
    df = df.with_columns(
        pl.col('plain_lyrics')
        .map_elements(preprocessor.clean, return_dtype=pl.Utf8)
        .alias('lyrics_cleaned')
    )

    original_non_empty = df['plain_lyrics'].is_not_null().sum()
    cleaned_non_empty = df['lyrics_cleaned'].str.len_chars().gt(0).sum()
    print(f"  Non-empty lyrics: {original_non_empty:,} -> {cleaned_non_empty:,} after cleaning")

    # --- Step 2b: Filter short tracks using plain_lyrics word count ---
    # Use plain_lyrics for filtering since BERT will score plain_lyrics
    MIN_WORDS = 50
    df = df.with_columns(
        pl.col('plain_lyrics')
        .fill_null('')
        .str.split(' ')
        .list.len()
        .alias('_word_count')
    )

    before_filter = len(df)
    df = df.filter(
        (pl.col('plain_lyrics').is_not_null()) &
        (pl.col('plain_lyrics').str.len_chars() > 0) &
        (pl.col('_word_count') >= MIN_WORDS)
    )
    df = df.drop('_word_count')
    print(f"  Filtered: removed {before_filter - len(df):,} tracks with < {MIN_WORDS} words")
    print(f"  Retained: {len(df):,} tracks for BERT analysis")

    # --- Step 3: BERT sentiment analysis on ORIGINAL lyrics ---
    # NOTE: BERT is a contextual transformer model pretrained on natural text.
    #       Feeding it stopword-removed text degrades performance because the
    #       model relies on grammatical structure and function words for context.
    #       We therefore score plain_lyrics (original text), NOT lyrics_cleaned.
    #       lyrics_cleaned is retained in the output for scripts 08-11 (Zipf, DFA, MF-DFA).
    print("\nStep 3: Running BERT sentiment analysis on original lyrics...")
    analyzer = BERTSentimentAnalyzer()

    df = df.with_columns(
        pl.col('plain_lyrics')
        .map_elements(analyzer.score, return_dtype=analyzer.struct_dtype)
        .alias('_bert_struct')
    ).unnest('_bert_struct')

    # Drop rows where BERT failed
    df = df.filter(pl.col('BERT_Composite').is_not_null())
    print(f"  BERT scores computed for {len(df):,} tracks")

    # --- Step 4: Summary statistics ---
    print("\nStep 4: Summary statistics")
    for lang in ['en', 'es', 'de']:
        sub = df.filter(pl.col('language') == lang)
        comp = sub['BERT_Composite'].to_numpy()
        wavg = sub['BERT_WeightedAvg'].to_numpy()
        print(f"  {LANG_NAMES[lang]:>8s}: n={len(sub):,}, "
              f"Composite M={comp.mean():.4f} SD={comp.std():.4f} Mdn={np.median(comp):.4f}, "
              f"WeightedAvg M={wavg.mean():.4f} SD={wavg.std():.4f}")

    # Star distribution
    print("\n  Star rating distribution (mean probabilities):")
    for s in range(1, 6):
        col = f'BERT_{s}star'
        m = df[col].mean()
        print(f"    {s}-star: {m:.4f} ({m*100:.1f}%)")

    # --- Step 5: Generate plot ---
    print("\nStep 5: Generating plot")
    plot_bert_by_language(df, OUTPUT_FOLDER)

    # --- Step 6: Save outputs ---
    print("\nStep 6: Saving outputs")

    # Main CSV output (21 core columns only)
    core_cols = [
        'track_name', 'artist', 'position', 'streams', 'year',
        'plain_lyrics', 'instrumental', 'language',
        'classification', 'pos_score', 'neg_score', 'neu_score', 'compound_score',
        'lyrics_cleaned',
        'BERT_1star', 'BERT_2star', 'BERT_3star', 'BERT_4star', 'BERT_5star',
        'BERT_Composite', 'BERT_WeightedAvg',
    ]
    existing_core = [c for c in core_cols if c in df.columns]
    df_out = df.select(existing_core)

    csv_path = os.path.join(OUTPUT_FOLDER, '07_bert_sentiment_results.csv')
    df_out.write_csv(csv_path)
    print(f"  CSV saved: {csv_path} ({len(df_out):,} rows, {len(df_out.columns)} columns)")

    # Summary text
    elapsed = time.time() - start_time
    summary = f"""BERT SENTIMENT ANALYSIS SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

MODEL: {BERT_MODEL}
DEVICE: {analyzer.device}

NOTE: BERT scores were computed on original plain_lyrics (not stopword-removed
lyrics_cleaned) to preserve natural sentence structure for the transformer model.
The lyrics_cleaned column is retained for downstream Zipf/DFA analyses (scripts 08-11).

DATASET:
  Input tracks: {original_non_empty:,}
  After cleaning: {cleaned_non_empty:,}
  With BERT scores: {len(df):,}

BERT COMPOSITE SCORE BY LANGUAGE:
"""
    for lang in ['en', 'es', 'de']:
        sub = df.filter(pl.col('language') == lang)['BERT_Composite']
        arr = sub.to_numpy()
        summary += (f"  {LANG_NAMES[lang]:>8s}: n={len(sub):,}, "
                    f"M={arr.mean():.4f}, SD={arr.std():.4f}, Mdn={np.median(arr):.4f}\n")

    summary += f"""
STAR RATING DISTRIBUTION (mean probability):
"""
    for s in range(1, 6):
        m = df[f'BERT_{s}star'].mean()
        summary += f"  {s}-star: {m:.4f} ({m*100:.1f}%)\n"

    summary += f"""
PROCESSING TIME: {elapsed:.1f} seconds

OUTPUT FILES:
  07_bert_sentiment_results.csv
  01_bert_sentiment_kde_by_language.png
  07_summary.txt
"""

    summary_path = os.path.join(OUTPUT_FOLDER, '07_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  Summary saved: {summary_path}")

    print(f"\n{'='*70}")
    print(f"BERT SENTIMENT ANALYSIS COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
