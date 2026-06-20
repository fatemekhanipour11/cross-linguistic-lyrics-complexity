"""
04_clean_spotify_data.py
Clean and prepare Spotify charts data by removing unnecessary columns and creating unique tracks list.

Input: 01_download_spotify_charts/01_spotify_charts_complete.csv
Output: 02_clean_spotify_data/
"""

import sys
import io
# Set UTF-8 encoding for stdout to handle special characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import os
from datetime import datetime

# ============ CONFIGURATION ============
INPUT_FILE = '01_download_spotify_charts/01_spotify_charts_complete.csv'
OUTPUT_FOLDER = '02_clean_spotify_data'
# =======================================

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 70)
print("SPOTIFY DATA CLEANING AND PREPARATION")
print("=" * 70)

# Load raw dataset
print(f"\nLoading data from: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
except Exception as e:
    print(f"✗ Error loading file: {e}")
    exit(1)

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Clean the dataset
print("\n" + "-" * 70)
print("CLEANING DATA")
print("-" * 70)

# Remove unnecessary URL column if exists
if "URL" in df.columns:
    df.drop(columns=["URL"], inplace=True)
    print("✓ Removed 'URL' column")

# Calculate statistics
num_unique_tracks = df["Track Name"].nunique()
num_unique_artists = df["Artist"].nunique()

print(f"\nStatistics:")
print(f"  Total rows: {len(df)}")
print(f"  Unique track names: {num_unique_tracks}")
print(f"  Unique artists: {num_unique_artists}")

# Create unique song DataFrame (Track Name + Artist)
print("\n" + "-" * 70)
print("CREATING UNIQUE TRACKS LIST")
print("-" * 70)

unique_songs = df.drop_duplicates(subset=["Track Name", "Artist"])
num_unique_songs = len(unique_songs)

print(f"✓ Unique songs (Track + Artist): {num_unique_songs}")
print(f"  Columns: {list(unique_songs.columns)}")

# Save cleaned unique tracks
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

output_file = os.path.join(OUTPUT_FOLDER, "02_unique_tracks.csv")
unique_songs.to_csv(output_file, index=False)
print(f"✓ Unique tracks saved: {output_file}")

# Save summary
summary_text = f"""
SPOTIFY DATA CLEANING SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

Total rows in raw dataset: {len(df)}
Unique track names: {num_unique_tracks}
Unique artists: {num_unique_artists}
Unique songs (Track + Artist): {num_unique_songs}

Columns in cleaned dataset:
{list(unique_songs.columns)}

This unique tracks list ensures we do not attempt to collect 
the same song lyrics multiple times, optimizing the lyric 
scraping process.
"""

summary_file = os.path.join(OUTPUT_FOLDER, "02_summary.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write(summary_text)
print(f"✓ Summary saved: {summary_file}")

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

