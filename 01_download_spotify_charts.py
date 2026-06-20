"""
01_download_spotify_charts.py
Download historical Spotify chart data from Wayback Machine.

Finds Wayback snapshots for spotifycharts.com CSV download URLs and downloads CSVs.
Outputs to: 01_download_spotify_charts/
"""

import requests
import time
import os
import json
import pandas as pd
from urllib.parse import quote_plus, urlparse
from datetime import datetime

OUTPUT_DIR = "01_download_spotify_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Markets and path variants to search
MARKETS = ['global', 'us', 'gb', 'de', 'fr', 'es', 'pl']
# choose 'weekly' or 'daily' depending on what you need
PERIOD = 'weekly'   # or 'daily'

# CDX API base
CDX_URL = "http://web.archive.org/cdx/search/cdx"

# helper: query CDX for snapshots whose original URL contains '/regional/{market}/{period}/' and '/download'
def query_wayback_for_market(market, from_year=2019, to_year=2023):
    # We'll request prefix matches for the weekly listing path, then filter for '/download'
    prefix = f"spotifycharts.com/regional/{market}/{PERIOD}/"
    params = {
        'url': prefix,
        'matchType': 'prefix',
        'output': 'json',
        'from': f"{from_year}",
        'to': f"{to_year}",
        'filter': 'statuscode:200',
        'collapse': 'original'  # avoid exact duplicate originals
    }
    resp = requests.get(CDX_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()  # first row is header
    header, rows = data[0], data[1:]
    results = []
    # rows are like: [original, timestamp, ...] depending on fields
    # but with default fields: urlkey original timestamp ...
    for row in rows:
        # row[1] is original (depends; check length). Safer to parse as dict with header
        try:
            # build dict
            entry = dict(zip(header, row))
        except Exception:
            # fallback: some CDX endpoints may return slightly different header ordering
            entry = {'url': row[0], 'timestamp': row[1]}  # best effort
        orig = entry.get('original') or entry.get('url') or entry.get('urlkey')
        timestamp = entry.get('timestamp')
        if not orig or not timestamp:
            continue
        # only keep those original URLs that end with '/download' or contain 'download'
        if '/download' in orig:
            results.append({'original': orig, 'timestamp': timestamp})
    return results

# helper: construct wayback snapshot URL
def snapshot_url(original, timestamp):
    return f"https://web.archive.org/web/{timestamp}/{original}"

# download CSV from snapshot and save
def download_snapshot_csv(snap_url, out_path):
    try:
        r = requests.get(snap_url, timeout=30)
        if r.status_code == 200:
            content_type = r.headers.get('Content-Type','')
            # sometimes Wayback serves text/html that contains redirect link to real CSV,
            # but often the snapshot contains the CSV directly (text/csv)
            if 'text/csv' in content_type or snap_url.endswith('.csv'):
                with open(out_path, 'wb') as f:
                    f.write(r.content)
                return True
            else:
                # If HTML, try to find direct link to CSV inside the HTML (simple heuristic)
                text = r.text
                # find any link that contains '/download' or '.csv'
                import re
                m = re.search(r'href="([^"]+(?:/download|\.csv)[^"]*)"', text)
                if m:
                    linked = m.group(1)
                    # make absolute if needed
                    if linked.startswith('/web/'):
                        # already a wayback absolute path
                        csv_url = 'https://web.archive.org' + linked
                    elif linked.startswith('http'):
                        csv_url = linked
                    else:
                        # relative to original snapshot domain
                        parsed = urlparse(snap_url)
                        base = f"{parsed.scheme}://{parsed.netloc}"
                        csv_url = base + linked
                    r2 = requests.get(csv_url, timeout=30)
                    if r2.status_code == 200:
                        with open(out_path, 'wb') as f:
                            f.write(r2.content)
                        return True
    except Exception as e:
        print("   download error:", e)
    return False

# MAIN routine: for each market find snapshots and download
def run(markets=MARKETS, from_year=2019, to_year=2023, limit_per_market=50):
    for m in markets:
        print(f"\n=== Market: {m} ===")
        entries = query_wayback_for_market(m, from_year, to_year)
        print(f"Found {len(entries)} candidate originals containing '/download' for {m}")
        count = 0
        for e in entries:
            if count >= limit_per_market:
                break
            orig = e['original']
            ts = e['timestamp']
            snap = snapshot_url(orig, ts)
            date_str = datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%Y-%m-%d")
            safe_name = f"{m}_{date_str}.csv".replace('/','-')
            out_path = os.path.join(OUTPUT_DIR, safe_name)
            if os.path.exists(out_path):
                print("  already have:", out_path)
                continue
            print("  trying:", snap)
            ok = download_snapshot_csv(snap, out_path)
            if ok:
                print("    saved:", out_path)
                count += 1
            else:
                print("    failed to extract CSV from snapshot")
            time.sleep(1.0)  # be polite
        print(f"Downloaded {count} CSVs for {m}")

def combine_all_csvs(output_dir=OUTPUT_DIR, output_filename="01_spotify_charts_complete.csv"):
    """
    Combine all downloaded CSV files into a single CSV file.
    """
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and f != output_filename]
    
    if not csv_files:
        print(f"\nNo CSV files found in {output_dir} to combine.")
        return
    
    print(f"\nCombining {len(csv_files)} CSV files...")
    all_data = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(output_dir, csv_file)
        try:
            df = pd.read_csv(csv_path)
            all_data.append(df)
            print(f"  ✓ Loaded {csv_file} ({len(df)} rows)")
        except Exception as e:
            print(f"  ✗ Error loading {csv_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Remove duplicates based on Track Name, Artist, and URL if available
        if 'Track Name' in combined_df.columns and 'Artist' in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['Track Name', 'Artist'], keep='first')
            after_dedup = len(combined_df)
            if before_dedup != after_dedup:
                print(f"  Removed {before_dedup - after_dedup} duplicate tracks")
        
        output_path = os.path.join(output_dir, output_filename)
        combined_df.to_csv(output_path, index=False)
        print(f"\n✓ Combined CSV saved: {output_path}")
        print(f"  Total tracks: {len(combined_df)}")
    else:
        print("\n✗ No valid CSV data to combine.")

if __name__ == "__main__":
    run()
    # Optionally combine all downloaded CSVs into one file
    # Uncomment the line below if you want to auto-combine after downloading
    # combine_all_csvs()

