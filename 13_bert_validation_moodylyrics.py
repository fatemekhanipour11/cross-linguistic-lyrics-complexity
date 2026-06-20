"""
13_bert_validation_moodylyrics.py
Validates nlptown/bert-base-multilingual-uncased-sentiment on MoodyLyrics
(Cano & Morisio 2017). Lyrics fetched from Genius API (lyricsgenius package),
matching the same source used for VADER scoring in the original paper (script 05).

Note: LRCLIB was the original choice for source consistency with the BERT
analysis (script 07), but LRCLIB rate-limiting blocked bulk validation queries.
Genius lyrics are textually equivalent for sentiment scoring purposes
(both are user-transcribed song words; cf. paper Section 3.5).
"""

import os, sys, time, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lyricsgenius
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
from scipy.stats import pointbiserialr, spearmanr

# ----- Paths and constants -----
ROOT          = os.path.dirname(os.path.abspath(__file__))
OUT_DIR       = os.path.join(ROOT, "13_bert_validation")
INPUT_CSV     = os.path.join(OUT_DIR, "moodylyrics_raw.csv")
LYRICS_CACHE  = os.path.join(OUT_DIR, "13_moodylyrics_with_lyrics.csv")
SCORED_CSV    = os.path.join(OUT_DIR, "13_bert_validation_scored.csv")
METRICS_CSV   = os.path.join(OUT_DIR, "13_bert_validation_metrics.csv")
QUADRANT_CSV  = os.path.join(OUT_DIR, "13_by_quadrant.csv")
CM_PNG        = os.path.join(OUT_DIR, "13_confusion_matrix.png")
SUMMARY_TXT   = os.path.join(OUT_DIR, "13_summary.txt")

GENIUS_ACCESS_TOKEN = '7inbAnRbcSLh3oKSxDAmNuTIO58olsa-WtPJ58XPrR6XoPYqkx-uX_xJ5cFuO3L2'
GENIUS_TIMEOUT = 10
GENIUS_DELAY   = 0.50
MIN_WORDS      = 50

VALENCE_MAP = {"happy": 1, "relaxed": 1, "sad": 0, "angry": 0}

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def log(msg):
    print(msg, flush=True)

# ----- Step 1: Load MoodyLyrics -----
log("=" * 70)
log("13 - BERT VALIDATION ON MOODYLYRICS (via Genius API)")
log("=" * 70)
log(f"Step 1: Loading MoodyLyrics from {INPUT_CSV}")
ml = pd.read_csv(INPUT_CSV)
ml.columns = [str(c).strip() for c in ml.columns]
log(f"  Loaded {len(ml):,} rows; columns={list(ml.columns)}")

cl = {c.lower(): c for c in ml.columns}
artist_col = cl.get("artist")
song_col   = cl.get("title") or cl.get("song")
mood_col   = cl.get("mood") or cl.get("emotion")
if not (artist_col and song_col and mood_col):
    raise SystemExit(f"ERROR: cannot detect Artist/Title/Mood. Got {list(ml.columns)}")

ml = ml[[artist_col, song_col, mood_col]].copy()
ml.columns = ["artist", "title", "mood"]
ml["mood"] = ml["mood"].astype(str).str.lower().str.strip()
ml = ml[ml["mood"].isin(VALENCE_MAP)].reset_index(drop=True)
ml["valence_label"] = ml["mood"].map(VALENCE_MAP).astype(int)
log(f"  4-mood subset: {len(ml):,}; dist={ml['mood'].value_counts().to_dict()}")

# ----- Step 2: Fetch from Genius -----
log(f"\nStep 2: Fetching lyrics from Genius (timeout={GENIUS_TIMEOUT}s, delay={GENIUS_DELAY}s)")
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=GENIUS_TIMEOUT,
                             retries=2, sleep_time=GENIUS_DELAY)
genius.verbose = False
genius.remove_section_headers = True
genius.skip_non_songs = True

def fetch_genius(artist, title):
    """Return (lyrics, status). Never raises."""
    try:
        song = genius.search_song(title, artist)
        if song is None:
            return None, "not_found"
        lyrics = song.lyrics
        if not lyrics or not lyrics.strip():
            return None, "empty"
        return lyrics, "ok"
    except Exception as e:
        return None, f"err:{type(e).__name__}"

done_keys, rows_buf = set(), []
if os.path.exists(LYRICS_CACHE):
    cached = pd.read_csv(LYRICS_CACHE)
    done_keys = set(zip(cached["artist"].astype(str), cached["title"].astype(str)))
    rows_buf = cached.to_dict("records")
    log(f"  Resuming: {len(done_keys):,} already cached")

n_total, fetched, found, skipped = len(ml), 0, 0, 0
t0 = time.time()
for _, row in ml.iterrows():
    key = (str(row["artist"]), str(row["title"]))
    if key in done_keys:
        skipped += 1
        continue
    lyrics, status = fetch_genius(row["artist"], row["title"])
    fetched += 1
    if lyrics:
        found += 1
    rows_buf.append({"artist": row["artist"], "title": row["title"],
                     "mood": row["mood"], "valence_label": row["valence_label"],
                     "plain_lyrics": lyrics, "fetch_status": status})
    if fetched % 25 == 0:
        elapsed = time.time() - t0
        rate = fetched / max(elapsed, 0.01)
        eta = (n_total - skipped - fetched) / max(rate, 0.01)
        log(f"  progress: {fetched + skipped}/{n_total} (found {found}, "
            f"last={status}, rate={rate:.2f}/s, eta={eta/60:.1f}min)")
    if fetched % 100 == 0:
        pd.DataFrame(rows_buf).to_csv(LYRICS_CACHE, index=False, encoding="utf-8")
    time.sleep(GENIUS_DELAY)

pd.DataFrame(rows_buf).to_csv(LYRICS_CACHE, index=False, encoding="utf-8")
log(f"  Done: {fetched:,} fetched this run, {found:,} with lyrics, {skipped:,} cached")

# ----- Step 3: Filter -----
log(f"\nStep 3: Filtering >= {MIN_WORDS} words")
df = pd.read_csv(LYRICS_CACHE)
df = df[df["plain_lyrics"].notna()].copy()
df["word_count"] = df["plain_lyrics"].astype(str).str.split().str.len()
df = df[df["word_count"] >= MIN_WORDS].reset_index(drop=True)
log(f"  Usable: {len(df):,}; mood dist={df['mood'].value_counts().to_dict()}")
if len(df) < 50:
    raise SystemExit(f"ERROR: only {len(df)} usable tracks.")

# ----- Step 4: BERT scoring -----
log(f"\nStep 4: Loading BERT (nlptown/bert-base-multilingual-uncased-sentiment)")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"  Device: {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

@torch.no_grad()
def score_text(text):
    enc = tokenizer(text, truncation=True, max_length=512,
                    return_tensors="pt", padding=True).to(device)
    logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    stars = np.array([1, 2, 3, 4, 5])
    expected = float((probs * stars).sum())
    composite = (expected - 3.0) / 2.0
    return int(stars[np.argmax(probs)]), composite

log(f"  Scoring {len(df):,} tracks...")
stars_list, comp_list = [], []
for i, txt in enumerate(df["plain_lyrics"].astype(str).tolist()):
    s, c = score_text(txt)
    stars_list.append(s); comp_list.append(c)
    if (i + 1) % 200 == 0:
        log(f"    scored {i+1}/{len(df)}")

df["bert_star"] = stars_list
df["bert_composite"] = comp_list
df["bert_pred_pos"] = (df["bert_composite"] > 0).astype(int)
df.to_csv(SCORED_CSV, index=False, encoding="utf-8")
log(f"  Saved: {SCORED_CSV}")

# ----- Step 5: Metrics -----
log(f"\nStep 5: Metrics")
y_true  = df["valence_label"].values
y_pred  = df["bert_pred_pos"].values
y_score = df["bert_composite"].values
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                   average="binary",
                                                   pos_label=1, zero_division=0)
try:
    auc = roc_auc_score(y_true, y_score)
except Exception:
    auc = float("nan")
pb_r, pb_p = pointbiserialr(y_true, y_score)
sp_r, sp_p = spearmanr(y_true, y_score)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

metrics_df = pd.DataFrame([{
    "n": int(len(df)), "accuracy": round(acc, 4),
    "precision_pos": round(prec, 4), "recall_pos": round(rec, 4),
    "f1_pos": round(f1, 4),
    "roc_auc": round(auc, 4) if not math.isnan(auc) else "NA",
    "pointbiserial_r": round(pb_r, 4), "pointbiserial_p": f"{pb_p:.2e}",
    "spearman_rho": round(sp_r, 4), "spearman_p": f"{sp_p:.2e}",
    "TN": int(cm[0,0]), "FP": int(cm[0,1]),
    "FN": int(cm[1,0]), "TP": int(cm[1,1]),
}])
metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8")
log(f"  Saved: {METRICS_CSV}")
log(f"\n  --- TOP-LINE METRICS ---")
for k, v in metrics_df.iloc[0].items():
    log(f"    {k}: {v}")

quad = (df.groupby("mood")
          .agg(n=("bert_composite", "size"),
               bert_mean=("bert_composite", "mean"),
               bert_sd=("bert_composite", "std"),
               bert_median=("bert_composite", "median"),
               star_mean=("bert_star", "mean"))
          .round(4).reset_index())
quad.to_csv(QUADRANT_CSV, index=False, encoding="utf-8")
log(f"\n  --- BY QUADRANT ---")
log(quad.to_string(index=False))

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
            color="white" if v > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Neg (pred)", "Pos (pred)"])
ax.set_yticklabels(["Neg (true)", "Pos (true)"])
ax.set_xlabel("BERT prediction (composite > 0)")
ax.set_ylabel("MoodyLyrics ground truth")
ax.set_title(f"Confusion matrix (n={len(df)}, acc={acc:.3f})")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout(); plt.savefig(CM_PNG, dpi=200); plt.close()
log(f"  Saved: {CM_PNG}")

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write("BERT validation on MoodyLyrics (Genius API)\n" + "=" * 50 + "\n\n")
    f.write(f"Source: Cano & Morisio (2017), ml_raw.xlsx\n")
    f.write(f"Lyrics: Genius API (lyricsgenius package, search_song).\n")
    f.write(f"Model: {MODEL_NAME}\n\n")
    f.write(f"MoodyLyrics tracks: {len(ml):,}\n")
    f.write(f"Usable (>={MIN_WORDS} words): {len(df):,}\n\n")
    f.write("Top-line metrics:\n")
    for k, v in metrics_df.iloc[0].items():
        f.write(f"  {k}: {v}\n")
    f.write("\nBy quadrant:\n" + quad.to_string(index=False))
    f.write("\n\nConfusion matrix [rows=true, cols=pred]:\n")
    f.write(f"          Neg(pred)  Pos(pred)\n")
    f.write(f"Neg(true) {cm[0,0]:>9d}  {cm[0,1]:>9d}\n")
    f.write(f"Pos(true) {cm[1,0]:>9d}  {cm[1,1]:>9d}\n")
log(f"  Saved: {SUMMARY_TXT}")
log("\n" + "=" * 70 + "\nSCRIPT 13 COMPLETE\n" + "=" * 70)