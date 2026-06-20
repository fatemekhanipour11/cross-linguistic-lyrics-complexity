#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_sentiment_analysis_genius.py
Perform sentiment analysis on song lyrics using Genius API and VADER.

Fetches lyrics from Genius API and analyzes sentiment using VADER sentiment analyzer.
Input: 04_clean_lyrics_data/04_lyrics_cleaned.csv
Output: 03_sentiment_analysis_genius/03_track_sentiment_analysis.xlsx
"""

import sys
import io
# Set UTF-8 encoding for stdout to handle special characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import lyricsgenius
from pathlib import Path

# ============ CONFIGURATION ============
GENIUS_ACCESS_TOKEN = 'xxxxxx'
INPUT_FILE = '04_clean_lyrics_data/04_lyrics_cleaned.csv'  # Input from lrclib script output
OUTPUT_FOLDER = '05_download_genius_data'
OUTPUT_FILE = f'{OUTPUT_FOLDER}/03_track_sentiment_analysis.xlsx'
SAVE_INTERVAL = 50  # Save every N tracks to prevent data loss on crashes
# ======================================

# Initialize Genius API
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
genius.verbose = False  # Suppress status messages
genius.remove_section_headers = True  # Clean lyrics

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Create output folder
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# Load your tracks
df = pd.read_csv(INPUT_FILE)

# Detect column names - handle both original_track_name and track_name formats
lower_map = {c.lower(): c for c in df.columns}
track_col = (lower_map.get("track_name") or lower_map.get("track name") or
             lower_map.get("original_track_name") or lower_map.get("track") or 
             lower_map.get("song") or lower_map.get("title"))
artist_col = (lower_map.get("artist") or lower_map.get("artist_name") or
              lower_map.get("original_artist_name") or lower_map.get("artist name"))

if not track_col or not artist_col:
    raise ValueError(f"Couldn't find track/artist columns. Found columns: {list(df.columns)}")

# Process all tracks
df_test = df.copy()
df_test = df_test.rename(columns={track_col: "track_name", artist_col: "artist"})
df_test["track_name"] = df_test["track_name"].astype(str).str.strip()
df_test["artist"] = df_test["artist"].astype(str).str.strip()

def get_lyrics(track_name, artist):
    """Fetch lyrics from Genius"""
    try:
        song = genius.search_song(track_name, artist)
        if song:
            return song.lyrics
        return None
    except Exception as e:
        print(f"Error fetching lyrics for {track_name} by {artist}: {e}")
        return None

def analyze_sentiment(lyrics):
    """
    Analyze sentiment using VADER (exactly like the paper)
    Returns: pos, neg, neu, compound scores
    """
    if not lyrics or lyrics.strip() == "":
        return None, None, None, None
    
    scores = analyzer.polarity_scores(lyrics)
    return scores['pos'], scores['neg'], scores['neu'], scores['compound']

def classify_song(pos_score, neg_score):
    """
    Paper's method: A song is positive if pos > neg
    """
    if pos_score is None or neg_score is None:
        return None
    return "positive" if pos_score > neg_score else "negative"

# Load existing progress if available (to resume after crashes)
processed_tracks = set()
records = []
if Path(OUTPUT_FILE).exists():
    try:
        df_existing = pd.read_excel(OUTPUT_FILE, engine='openpyxl')
        print(f"Found existing file with {len(df_existing)} tracks. Resuming...")
        for _, row in df_existing.iterrows():
            key = (str(row['track_name']).strip(), str(row['artist']).strip())
            processed_tracks.add(key)
        records = df_existing.to_dict('records')
        print(f"Resuming from track {len(records) + 1}")
    except Exception as e:
        print(f"Could not load existing file: {e}. Starting fresh...")
        records = []

found_lyrics = sum(1 for r in records if r.get('lyrics_found', False))
no_lyrics = sum(1 for r in records if not r.get('lyrics_found', False))

print(f"Processing {len(df_test)} tracks...")
print(f"Already processed: {len(processed_tracks)} tracks")

for idx, row in df_test.iterrows():
    track_name = row['track_name']
    artist = row['artist']
    
    # Skip if already processed
    track_key = (track_name.strip(), artist.strip())
    if track_key in processed_tracks:
        continue
    
    print(f"\n[{idx+1}/{len(df_test)}] {track_name} - {artist}")
    
    # Get lyrics
    lyrics = get_lyrics(track_name, artist)
    
    if not lyrics:
        print(f"  [X] No lyrics found")
        records.append({
            "track_name": track_name,
            "artist": artist,
            "lyrics_found": False,
            "classification": None,
            "pos_score": None,
            "neg_score": None,
            "neu_score": None,
            "compound_score": None
        })
        no_lyrics += 1
        processed_tracks.add(track_key)
        
        # Save progress periodically to prevent data loss
        if len(records) % SAVE_INTERVAL == 0:
            df_temp = pd.DataFrame.from_records(records)
            df_temp.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
            print(f"\n[PROGRESS SAVED] Processed {len(records)}/{len(df_test)} tracks")
        
        time.sleep(0.5)  # Be nice to Genius API
        continue
    
    found_lyrics += 1
    print(f"  [OK] Lyrics found ({len(lyrics)} chars)")
    
    # Analyze sentiment
    pos, neg, neu, compound = analyze_sentiment(lyrics)
    classification = classify_song(pos, neg)
    
    print(f"  [SENTIMENT] {classification} (pos={pos:.3f}, neg={neg:.3f})")
    
    records.append({
        "track_name": track_name,
        "artist": artist,
        "lyrics_found": True,
        "lyrics": lyrics[:500],  # Store first 500 chars for reference
        "classification": classification,
        "pos_score": pos,
        "neg_score": neg,
        "neu_score": neu,
        "compound_score": compound
    })
    
    processed_tracks.add(track_key)
    
    # Save progress periodically to prevent data loss
    if len(records) % SAVE_INTERVAL == 0:
        df_temp = pd.DataFrame.from_records(records)
        df_temp.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
        print(f"\n[PROGRESS SAVED] Processed {len(records)}/{len(df_test)} tracks")
    
    time.sleep(0.5)  # Be nice to Genius API

# Final save
df_results = pd.DataFrame.from_records(records)
df_results.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")

# Final stats are already calculated during processing

print("\n" + "="*50)
print("[OK] Done!")
print(f"Total tracks processed: {len(records)}")
print(f"Lyrics found: {found_lyrics}")
print(f"No lyrics: {no_lyrics}")
if len(records) > 0:
    df_final = pd.DataFrame.from_records(records)
    positive_count = len(df_final[df_final['classification']=='positive']) if 'classification' in df_final.columns else 0
    negative_count = len(df_final[df_final['classification']=='negative']) if 'classification' in df_final.columns else 0
    print(f"Positive songs: {positive_count}")
    print(f"Negative songs: {negative_count}")
print(f"Wrote: {Path(OUTPUT_FILE).resolve()}")

