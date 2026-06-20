"""
04_clean_lyrics.py
Clean lyrics data and detect language.

Performs:
- Language detection on lyrics using langdetect
- Column renaming and cleanup
- Removal of unnecessary LRCLIB metadata columns

Input: 03_download_lyrics_lrclib/03_lyrics_complete.csv
Output: 04_clean_lyrics_data/
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import polars as pl
import os
import time
import re
from datetime import datetime
from langdetect import detect, DetectorFactory

# ============ CONFIGURATION ============
INPUT_FILE = '03_download_lyrics_lrclib/03_lyrics_complete.csv'
OUTPUT_FOLDER = '04_clean_lyrics_data'
OTHER_THRESHOLD = 10
# =======================================

# Create output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Helper functions
def save_csv(df, file_name):
    """Save DataFrame to CSV."""
    output_path = os.path.join(OUTPUT_FOLDER, f"{file_name}.csv")
    df.to_csv(output_path, index=False)
    print(f"✓ CSV saved: {output_path}")

# def save_plot(fig, plot_name):
#     """Save matplotlib figure to PNG."""
#     output_path = os.path.join(OUTPUT_FOLDER, f"{plot_name}.png")
#     fig.savefig(output_path, dpi=150, bbox_inches='tight')
#     print(f"✓ Plot saved: {output_path}")

def save_summary(text, filename='05_summary.txt'):
    """Append summary text to the summary file."""
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write("\n" + "="*60 + "\n")
    print(f"✓ Summary saved: {output_path}")

# Fix randomness for langdetect
DetectorFactory.seed = 0

print("=" * 70)
print("LYRICS ANALYSIS PIPELINE")
print("=" * 70)

# ----------------------------
# STEP 1: Load CSV
# ----------------------------
print(f"\nStep 1: Loading data from {INPUT_FILE}")
start = time.time()
try:
    df = pl.read_csv(INPUT_FILE)
    print(f"  ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns in {time.time()-start:.2f} seconds")
except Exception as e:
    print(f"  ✗ Error loading file: {e}")
    exit(1)

# ----------------------------
# STEP 2: Cleaning + Language Detection
# ----------------------------
print("\nStep 2: Cleaning lyrics and detecting languages...")
start = time.time()

def clean_and_detect_language(text):
    """Clean text and detect language."""
    if not text:
        return "unknown"
    text = str(text)
    non_lyrical_phrases = r'(Uh|Yeah(\,\s*yeah)?|Oh no|Singing\s*\:)'
    text = re.sub(non_lyrical_phrases, '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return "unknown"
    try:
        return detect(text)
    except:
        return "unknown"

languages = [clean_and_detect_language(text) for text in df["plain_lyrics"]]
df = df.with_columns(pl.Series("language", languages))
print(f"  ✓ Language detection completed in {time.time()-start:.2f} seconds")

# ----------------------------
# STEP 3: Rename columns
# ----------------------------
print("\nStep 3: Renaming columns...")
if "original_track_name" in df.columns:
    df = df.rename({
        "original_track_name": "track_name",
        "original_artist_name": "artist"
    })
    print("  ✓ Columns renamed")

# ----------------------------
# STEP 4: Drop unnecessary columns
# ----------------------------
print("\nStep 4: Dropping unnecessary columns...")
columns_to_drop = [
    'lrclib_id', 'track_name_matched', 'artist_name_matched',
    'album_name_matched', 'duration_matched', 'synced_lyrics'
]
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
if existing_cols_to_drop:
    df = df.drop(existing_cols_to_drop)
    print(f"  ✓ Dropped: {existing_cols_to_drop}")

# ----------------------------
# STEP 5: Language statistics
# ----------------------------
print("\nStep 5: Computing language statistics...")
num_languages = df['language'].n_unique()
num_artists = df['artist'].n_unique()

lang_counts = df.group_by("language").agg(pl.len().alias("count")).sort("count", descending=True)
lang_df = lang_counts.to_pandas()

print(f"  Total rows: {len(df)}")
print(f"  Unique languages: {num_languages}")
print(f"  Unique artists: {num_artists}")

# Save cleaned DataFrame
save_csv(df.to_pandas(), file_name="04_lyrics_cleaned")

summary_text = f"""
LYRICS ANALYSIS SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

DATASET OVERVIEW:
- Total tracks: {len(df)}
- Unique languages detected: {num_languages}
- Unique artists: {num_artists}

LANGUAGE DISTRIBUTION (Top 10):
{lang_df.head(10).to_string(index=False)}

OUTPUT FILES:
- 04_lyrics_cleaned.csv: All tracks with language detection and cleaned columns
"""
save_summary(summary_text)

print("\n" + "=" * 70)
print("LYRICS ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nResults saved to: {OUTPUT_FOLDER}/")
print(f"\n  CSVs (3):")
print(f"    - 04_lyrics_cleaned.csv")

print(f"\n  Summary:")
print(f"    - 04_summary.txt")
