"""
compute_trendiness.py

Computes the trendiness composite (0–3) for each video in an enriched CSV.
Requires data/music_lookup.csv to be populated manually first — see CLAUDE.md.

Usage
-----
    python compute_trendiness.py                                        # accounts
    python compute_trendiness.py --input data/hashtag_enriched.csv \
                                 --output data/hashtag_trendiness.csv  # hashtags

Configuration
-------------
Set MUSIC_THRESHOLD after examining the music_usage_count distribution.
Set TRENDINESS_THRESHOLD to change the is_new_form cutoff (default: 2).

Output
------
    data/trendiness.csv — full dataset with trendiness scores and all variables
"""

import argparse
import os
import re

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Set empirically after examining music_usage_count distribution.
# None = condition will not be computed until this is set.
MUSIC_THRESHOLD = 10000

# is_new_form = trendiness_score >= TRENDINESS_THRESHOLD
# Run sensitivity checks at 1 and 3.
TRENDINESS_THRESHOLD = 2


def load_music_lookup(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found — cond_sound will be set to False for all videos.")
        return pd.DataFrame(columns=["music_id", "music_usage_count", "music_title"])
    return pd.read_csv(path, dtype={"music_id": str}).drop_duplicates(subset=["music_id"])


def is_original_sound(music_title, username):
    if pd.isna(music_title):
        return False
    t = str(music_title).lower()
    if "original sound" in t:
        return True
    if pd.isna(username):
        return False
    # Whole-word match on the username to avoid spurious substring hits
    # (e.g. a 3-char username matching a fragment of a song title).
    u = str(username).lower()
    return re.search(rf"\b{re.escape(u)}\b", t) is not None


def compute(df, music):
    df["music_id"] = df["music_id"].astype(str)
    df = df.merge(music[["music_id", "music_usage_count", "music_title"]], on="music_id", how="left")

    df["_is_original_sound"] = df.apply(
        lambda r: is_original_sound(r.get("music_title"), r.get("username")), axis=1
    )

    if MUSIC_THRESHOLD is None:
        print("Warning: MUSIC_THRESHOLD is not set — cond_sound = False for all videos.")
        print("Examine music_usage_count distribution and set MUSIC_THRESHOLD in this script.")
        df["cond_sound"] = False
    else:
        df["cond_sound"] = (
            (df["music_usage_count"] > MUSIC_THRESHOLD) & ~df["_is_original_sound"]
        )

    df["cond_platform_native"] = df["has_platform_native_feature"].fillna(False)
    df["cond_hashtag"]         = df["has_virality_hashtag"].fillna(False)

    df["trendiness_score"] = (
        df["cond_sound"].astype(int) +
        df["cond_platform_native"].astype(int) +
        df["cond_hashtag"].astype(int)
    )
    df["is_new_form"] = df["trendiness_score"] >= TRENDINESS_THRESHOLD

    df = df.drop(columns=["_is_original_sound"], errors="ignore")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=os.path.join(DATA_DIR, "enriched", "accounts_enriched.csv"),
                        help="Input CSV (default: data/enriched/accounts_enriched.csv).")
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "trendiness", "accounts_trendiness.csv"),
                        help="Output CSV (default: data/trendiness/accounts_trendiness.csv).")
    parser.add_argument("--music-lookup", default=os.path.join(DATA_DIR, "music_lookup", "accounts_music_lookup.csv"),
                        help="Music lookup CSV (default: data/music_lookup/accounts_music_lookup.csv).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Not found: {args.input}")
        print("Run enrich.py first.")
        return

    # IMPORTANT: read music_id and id as str. If a NaN is present pandas would
    # otherwise coerce the column to float64, mangling 19-digit IDs into
    # scientific notation and silently breaking the music_lookup merge.
    df    = pd.read_csv(args.input, dtype={"music_id": str, "id": str})
    music = load_music_lookup(args.music_lookup)

    print(f"Loaded {len(df)} videos from {args.input}")
    if not music.empty:
        print(f"Loaded {len(music)} music IDs from {args.music_lookup}")

    df = compute(df, music)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")

    print(f"\nTrendiness score distribution:")
    print(df["trendiness_score"].value_counts().sort_index().to_string())
    print(f"\nis_new_form (score >= {TRENDINESS_THRESHOLD}): {df['is_new_form'].sum()} / {len(df)} videos")

    if "_party" in df.columns:
        print(f"\nMean trendiness score by party:")
        print(df.groupby("_party")["trendiness_score"].mean().sort_values(ascending=False).to_string())

    if MUSIC_THRESHOLD is None:
        print(f"\nNext step: set MUSIC_THRESHOLD.")
        if "music_usage_count" in df.columns:
            print(f"\nmusic_usage_count summary (excluding NaN):")
            print(df["music_usage_count"].dropna().describe().to_string())


if __name__ == "__main__":
    main()
