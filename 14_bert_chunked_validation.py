"""
14_bert_chunked_validation.py
Robustness check for BERT 512-token truncation.
Re-scores each track by splitting full lyrics into <=500-token chunks,
running BERT on each chunk, and averaging the composite scores.
Compares the chunk-averaged 'full-song' score with the original
opening-only score from script 07.
"""

import os, sys, math, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----- Paths -----
ROOT       = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(ROOT, "07_bert_sentiment", "07_bert_sentiment_results.csv")
OUT_DIR    = os.path.join(ROOT, "14_bert_chunked")
os.makedirs(OUT_DIR, exist_ok=True)

SCORED_CSV = os.path.join(OUT_DIR, "14_bert_chunked_scored.csv")
METRICS_CSV = os.path.join(OUT_DIR, "14_bert_chunked_metrics.csv")
BY_LANG_CSV = os.path.join(OUT_DIR, "14_by_language.csv")
SCATTER_PNG = os.path.join(OUT_DIR, "14_chunked_vs_opening_scatter.png")
SUMMARY_TXT = os.path.join(OUT_DIR, "14_summary.txt")

MODEL_NAME    = "nlptown/bert-base-multilingual-uncased-sentiment"
CHUNK_TOKENS  = 500          # leave room for special tokens within 512
TOP3          = {"en", "es", "de"}

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def log(m):
    print(m, flush=True)

# ----- Load -----
log("=" * 70)
log("14 - BERT CHUNKED ROBUSTNESS CHECK (full-song vs opening-only)")
log("=" * 70)
log(f"Loading: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
log(f"  Loaded {len(df):,} rows")

# Filter same as script 07 (top-3, with lyrics, BERT-scored)
df = df[df["language"].isin(TOP3)].copy()
df = df[df["plain_lyrics"].notna()].copy()
df = df[df["BERT_Composite"].notna()].copy()
df = df.reset_index(drop=True)
log(f"  After top-3 + non-null filter: {len(df):,}")

# ----- BERT -----
log(f"\nLoading BERT model")
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"  Device: {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

@torch.no_grad()
def score_chunk(token_ids):
    """Score a single chunk of token IDs. Returns (predicted_star, composite)."""
    # token_ids: 1D tensor of input IDs (no special tokens yet)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is not None and sep_id is not None:
        ids = torch.cat([torch.tensor([cls_id]), token_ids, torch.tensor([sep_id])])
    else:
        ids = token_ids
    ids = ids.unsqueeze(0).to(device)
    attn = torch.ones_like(ids).to(device)
    logits = model(input_ids=ids, attention_mask=attn).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    stars = np.array([1, 2, 3, 4, 5])
    expected = float((probs * stars).sum())
    composite = (expected - 3.0) / 2.0
    return int(stars[np.argmax(probs)]), composite, probs.tolist()

@torch.no_grad()
def score_full_song(text):
    """Tokenize full text, split into <=CHUNK_TOKENS chunks, score each, average."""
    enc = tokenizer(text, add_special_tokens=False, return_tensors="pt", truncation=False)
    ids_all = enc["input_ids"][0]
    n_tok_total = int(ids_all.shape[0])
    # Split into chunks
    chunks = [ids_all[i:i + CHUNK_TOKENS] for i in range(0, n_tok_total, CHUNK_TOKENS)]
    n_chunks = len(chunks)
    composites = []
    stars = []
    for c in chunks:
        s, comp, _ = score_chunk(c)
        composites.append(comp)
        stars.append(s)
    return {
        "n_tokens_total":   n_tok_total,
        "n_chunks":         n_chunks,
        "chunked_composite": float(np.mean(composites)),
        "chunked_star":     float(np.mean(stars)),
        "chunk_min":        float(np.min(composites)),
        "chunk_max":        float(np.max(composites)),
        "chunk_sd":         float(np.std(composites)) if len(composites) > 1 else 0.0,
    }

# ----- Score all tracks -----
log(f"\nScoring {len(df):,} tracks (chunk size = {CHUNK_TOKENS} tokens)...")
t0 = time.time()
results = []
for i, txt in enumerate(df["plain_lyrics"].astype(str).tolist()):
    r = score_full_song(txt)
    results.append(r)
    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 0.01)
        eta = (len(df) - i - 1) / max(rate, 0.01)
        log(f"  {i+1}/{len(df)} ({rate:.2f}/s, eta {eta/60:.1f}min)")

res_df = pd.DataFrame(results)
df = pd.concat([df.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)
df["chunked_pred_pos"] = (df["chunked_composite"] > 0).astype(int)
df["opening_pred_pos"] = (df["BERT_Composite"] > 0).astype(int)
df["abs_diff"] = (df["chunked_composite"] - df["BERT_Composite"]).abs()

# ----- Save -----
keep = ["track_name", "artist", "language", "BERT_Composite",
        "chunked_composite", "chunked_star", "n_tokens_total", "n_chunks",
        "chunk_min", "chunk_max", "chunk_sd", "abs_diff",
        "opening_pred_pos", "chunked_pred_pos"]
df[keep].to_csv(SCORED_CSV, index=False, encoding="utf-8")
log(f"\nSaved per-track: {SCORED_CSV}")

# ----- Overall metrics -----
def metrics_for(sub):
    n = len(sub)
    if n < 2:
        return dict(n=n)
    pr, pp = pearsonr(sub["BERT_Composite"], sub["chunked_composite"])
    sr, sp = spearmanr(sub["BERT_Composite"], sub["chunked_composite"])
    mad = sub["abs_diff"].mean()
    rmse = math.sqrt((sub["abs_diff"] ** 2).mean())
    agree = accuracy_score(sub["opening_pred_pos"], sub["chunked_pred_pos"])
    cm = confusion_matrix(sub["opening_pred_pos"], sub["chunked_pred_pos"], labels=[0, 1])
    flips = int(cm[0, 1] + cm[1, 0])
    n_trunc = int((sub["n_tokens_total"] >= 512).sum())
    return dict(
        n=n, n_truncated=n_trunc,
        pct_truncated=round(100 * n_trunc / n, 1),
        pearson_r=round(pr, 4), pearson_p=f"{pp:.2e}",
        spearman_rho=round(sr, 4), spearman_p=f"{sp:.2e}",
        mean_abs_diff=round(mad, 4),
        rmse=round(rmse, 4),
        sign_agreement=round(agree, 4),
        n_flipped=flips,
        pct_flipped=round(100 * flips / n, 2),
    )

overall = metrics_for(df)
overall_df = pd.DataFrame([dict(scope="overall", **overall)])

per_lang_rows = []
for lang in ["en", "es", "de"]:
    sub = df[df["language"] == lang]
    per_lang_rows.append(dict(scope=lang, **metrics_for(sub)))
per_lang_df = pd.DataFrame(per_lang_rows)

all_metrics = pd.concat([overall_df, per_lang_df], ignore_index=True)
all_metrics.to_csv(METRICS_CSV, index=False, encoding="utf-8")
log(f"Saved metrics: {METRICS_CSV}")

# ----- By-language descriptive table -----
by_lang = (df.groupby("language")
             .agg(n=("chunked_composite", "size"),
                  opening_M=("BERT_Composite", "mean"),
                  opening_SD=("BERT_Composite", "std"),
                  chunked_M=("chunked_composite", "mean"),
                  chunked_SD=("chunked_composite", "std"),
                  median_n_tokens=("n_tokens_total", "median"),
                  median_n_chunks=("n_chunks", "median"))
             .round(4).reset_index())
by_lang.to_csv(BY_LANG_CSV, index=False, encoding="utf-8")

# ----- Plot -----
fig, ax = plt.subplots(figsize=(6.5, 6))
colors = {"en": "#2E86AB", "es": "#E63946", "de": "#FFB000"}
for lang in ["en", "es", "de"]:
    sub = df[df["language"] == lang]
    ax.scatter(sub["BERT_Composite"], sub["chunked_composite"],
               s=14, alpha=0.45, color=colors[lang],
               label=f"{lang.upper()} (n={len(sub)})", edgecolor="none")
ax.plot([-1, 1], [-1, 1], color="gray", linestyle="--", linewidth=1, label="y = x")
ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
ax.set_xlabel("Opening-window BERT composite (script 07)")
ax.set_ylabel("Full-song chunked-average BERT composite")
ax.set_title(f"Chunked vs. opening-only BERT (n = {len(df):,}; "
             f"r = {overall['pearson_r']}, MAD = {overall['mean_abs_diff']})")
ax.legend(loc="lower right", frameon=True)
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(SCATTER_PNG, dpi=200); plt.close()
log(f"Saved plot: {SCATTER_PNG}")

# ----- Summary text -----
with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write("BERT chunked vs. opening-only robustness check\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Tracks scored: {len(df):,}\n")
    f.write(f"Chunk size: {CHUNK_TOKENS} tokens\n\n")
    f.write("Overall metrics:\n")
    for k, v in overall.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nBy-language metrics:\n")
    f.write(per_lang_df.to_string(index=False))
    f.write("\n\nBy-language descriptive (opening vs chunked):\n")
    f.write(by_lang.to_string(index=False))
    f.write("\n")
log(f"Saved summary: {SUMMARY_TXT}")

log("\n" + "=" * 70)
log("DONE")
log("=" * 70)
log(f"  n = {len(df):,}, truncated = {overall['n_truncated']} "
    f"({overall['pct_truncated']}%)")
log(f"  Pearson r (chunked vs opening) = {overall['pearson_r']}")
log(f"  Mean absolute difference        = {overall['mean_abs_diff']}")
log(f"  Sign agreement (pos/neg)        = {overall['sign_agreement']*100:.1f}%")
log(f"  Tracks that flipped class       = {overall['n_flipped']} "
    f"({overall['pct_flipped']}%)")