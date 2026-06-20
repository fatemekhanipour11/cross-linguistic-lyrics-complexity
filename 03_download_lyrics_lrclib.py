"""
02_download_lyrics_lrclib.py
Download lyrics from LRCLIB (free community data) for Spotify chart tracks.

Input: CSV with columns: 'Position', 'Track Name', 'Artist', 'Streams', 'year'
       Reads from: 02_clean_spotify_data/02_unique_tracks.csv
Output: 03_lyrics_lrclib/
"""

import pandas as pd
import requests
import time
import os
from datetime import datetime

# ============ CONFIGURATION ============
INPUT_FILE = '02_clean_spotify_data/02_unique_tracks.csv'  # Input from wayback script output
OUTPUT_FOLDER = '03_lyrics_lrclib'
# =======================================

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# LRCLIB API base URL (no API key needed!)
LRCLIB_BASE_URL = 'https://lrclib.net/api'

def search_lyrics_lrclib(track_name, artist_name, album_name=None, duration=None):
    """Search for lyrics on LRCLIB"""
    params = {
        'track_name': track_name,
        'artist_name': artist_name
    }
    
    try:
        response = requests.get(
            f"{LRCLIB_BASE_URL}/search",
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            
            if results and len(results) > 0:
                # Get first match
                match = results[0]
                return {
                    'lrclib_id': match.get('id'),
                    'track_name_matched': match.get('trackName'),
                    'artist_name_matched': match.get('artistName'),
                    'album_name_matched': match.get('albumName'),
                    'duration_matched': match.get('duration'),
                    'plain_lyrics': match.get('plainLyrics'),
                    'synced_lyrics': match.get('syncedLyrics'),
                    'instrumental': match.get('instrumental', False)
                }
    except Exception as e:
        pass
    
    return None

# ============ MAIN EXECUTION ============
print("=" * 70)
print("LRCLIB LYRICS DOWNLOADER")
print("=" * 70)

# Load metadata
print(f"\nLoading metadata from: {INPUT_FILE}")
try:
    df_metadata = pd.read_csv(INPUT_FILE)
    
    # Check if we need to create unique tracks list first
    # If input has duplicates, create unique tracks
    if 'Track Name' in df_metadata.columns and 'Artist' in df_metadata.columns:
        # Create unique tracks if needed
        unique_tracks = df_metadata.drop_duplicates(subset=['Track Name', 'Artist'])
        if len(unique_tracks) < len(df_metadata):
            print(f"  Found {len(df_metadata)} total tracks, {len(unique_tracks)} unique")
            df_metadata = unique_tracks
    
    # Validate required columns
    required_cols = ['Track Name', 'Artist']
    if not all(col in df_metadata.columns for col in required_cols):
        print(f"✗ Error: Input file must contain columns: {required_cols}")
        print(f"  Found columns: {list(df_metadata.columns)}")
        exit(1)
        
    print(f"✓ Loaded {len(df_metadata)} tracks")
except Exception as e:
    print(f"✗ Error loading file: {e}")
    exit(1)

print("\n✓ No API key required for LRCLIB!")
print("Starting lyrics download...")
print("-" * 70)

lyrics_data = []
processed = 0
found = 0
with_lyrics = 0
with_synced = 0
instrumental_count = 0
not_found = 0

for idx, row in df_metadata.iterrows():
    track_name = row['Track Name']
    artist_name = row['Artist']
    album_name = None 
    duration = None
    
    # Search for lyrics
    lyrics_info = search_lyrics_lrclib(track_name, artist_name, album_name, duration)
    
    # Build entry structure
    entry = {
        'original_track_name': track_name,
        'original_artist_name': artist_name,
        'position': row.get('Position'),
        'streams': row.get('Streams'),
        'year': row.get('year')
    }
    
    if lyrics_info:
        entry.update(lyrics_info)
        found += 1
        
        if lyrics_info.get('instrumental'):
            instrumental_count += 1
        elif lyrics_info.get('plain_lyrics'):
            with_lyrics += 1
            if lyrics_info.get('synced_lyrics'):
                with_synced += 1
    else:
        not_found += 1
        entry['lrclib_id'] = None
        entry['plain_lyrics'] = None
        entry['synced_lyrics'] = None
        # Add matched columns for consistency
        entry['track_name_matched'] = None
        entry['artist_name_matched'] = None
        entry['album_name_matched'] = None
        entry['duration_matched'] = None
        entry['instrumental'] = False

    
    lyrics_data.append(entry)
    processed += 1
    
    # Progress update
    if processed % 25 == 0:
        print(f"  Progress: {processed}/{len(df_metadata)} | Found: {found} | With lyrics: {with_lyrics}")
    
    # Rate limiting (be respectful to free API)
    time.sleep(0.3)

print(f"\n✓ Processing complete!")
print(f"  Total processed: {processed}")
print(f"  Tracks found: {found}")
print(f"  With plain lyrics: {with_lyrics}")
print(f"  With synced lyrics: {with_synced}")
print(f"  Instrumental: {instrumental_count}")
print(f"  Not found: {not_found}")

# Save results
if lyrics_data:
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    df_lyrics = pd.DataFrame(lyrics_data)
    
    # Save complete data
    output_file = f"{OUTPUT_FOLDER}/03_lyrics_complete.csv"
    df_lyrics.to_csv(output_file, index=False)
    print(f"\n✓ Lyrics data saved: {output_file}")
    
    # Save only tracks with lyrics
    df_with_lyrics = df_lyrics[df_lyrics['plain_lyrics'].notna() & ~df_lyrics['instrumental']]
    lyrics_only_file = f"{OUTPUT_FOLDER}/03_lyrics_plain_only.csv"
    df_with_lyrics.to_csv(lyrics_only_file, index=False)
    print(f"✓ Plain lyrics only: {lyrics_only_file} ({len(df_with_lyrics)} tracks)")
    
    # Save lyrics as individual text files
    lyrics_text_folder = f"{OUTPUT_FOLDER}/03_lyrics_text"
    os.makedirs(lyrics_text_folder, exist_ok=True)
    
    saved_count = 0
    for idx, row in df_with_lyrics.iterrows():
        if row['plain_lyrics']:
            # Create safe filename
            safe_track = "".join(c for c in str(row['original_track_name']) if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_artist = "".join(c for c in str(row['original_artist_name']) if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"{safe_artist} - {safe_track}.txt"
            
            filepath = os.path.join(lyrics_text_folder, filename[:200])
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Track: {row['original_track_name']}\n")
                    f.write(f"Artist: {row['original_artist_name']}\n")
                    f.write(f"Position: {row.get('position', 'N/A')}\n")
                    f.write(f"Streams: {row.get('streams', 'N/A')}\n")
                    f.write(f"Source: LRCLIB (ID: {row.get('lrclib_id', 'N/A')})\n")
                    f.write(f"\n{'-'*50}\n\n")
                    f.write(row['plain_lyrics'])
                saved_count += 1
            except Exception as e:
                pass
    
    print(f"✓ Individual lyrics files: {lyrics_text_folder}/ ({saved_count} files)")
    
    # Generate summary
    summary_stats = {
        'total_tracks_processed': len(df_lyrics),
        'tracks_found': found,
        'tracks_with_plain_lyrics': with_lyrics,
        'tracks_with_synced_lyrics': with_synced,
        'instrumental_tracks': instrumental_count,
        'tracks_not_found': not_found,
        'coverage_rate': f"{(with_lyrics/len(df_lyrics)*100):.1f}%" if len(df_lyrics) > 0 else '0.0%'
    }
    
    # Save summary
    summary_file = f"{OUTPUT_FOLDER}/03_lyrics_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"LRCLIB Lyrics Download Summary\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Input Columns: Position, Track Name, Artist, Streams, year\n")
        f.write(f"=" * 50 + "\n\n")
        
        f.write("STATISTICS:\n")
        for key, value in summary_stats.items():
            f.write(f"  {key}: {value}\n")
        
    print(f"✓ Summary saved: {summary_file}")
    
else:
    print("\n✗ ERROR: No data collected!")

print("\n" + "=" * 70)
print("LYRICS DOWNLOAD COMPLETE")
print("=" * 70)

