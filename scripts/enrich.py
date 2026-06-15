"""
enrich.py

Reads a combined CSV and adds all derived fields needed for the trendiness
composite. Also outputs data/hashtag_frequency.csv for manual review of the
virality hashtag seed list.

Usage
-----
    python enrich.py                                              # accounts (default)
    python enrich.py --input data/hashtags_combined.csv \
                     --output data/hashtags_enriched.csv         # hashtags

Output
------
    data/accounts_enriched.csv  — accounts_combined.csv + derived fields
    data/hashtag_frequency.csv  — all hashtags ranked by frequency
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Update this set after reviewing data/hashtag_frequency.csv
VIRALITY_HASHTAGS = {
    "fyp", "foryou", "foryoupage", "foryourpage",
    "viral", "trending", "tiktok", "xyzbca",
}


def parse_list_field(val):
    """Parse a field that may be stored as a Python list literal or JSON string."""
    if isinstance(val, list):
        return val
    if pd.isna(val) or val == "" or val == "[]":
        return []
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def enrich(df):
    df["caption_length"] = df["video_description"].fillna("").str.len()

    desc = df["video_description"].fillna("").str.lower()
    df["is_duet"]   = desc.str.match(r"#?duet with @")
    df["is_stitch"] = desc.str.match(r"#?stitch with @")

    effects = df.get("effect_ids", pd.Series(["[]"] * len(df))).apply(parse_list_field)
    df["has_effects"] = effects.apply(lambda x: any(e for e in x))

    df["has_platform_native_feature"] = df["has_effects"] | df["is_duet"] | df["is_stitch"]

    hashtags = df["hashtag_names"].apply(parse_list_field)
    df["has_virality_hashtag"] = hashtags.apply(
        lambda tags: any(
            isinstance(t, str) and t.lower() in VIRALITY_HASHTAGS for t in tags
        )
    )

    # Temporal and reach metrics
    if "scrape_timestamp" in df.columns and "create_time" in df.columns:
        scrape_dt  = pd.to_datetime(df["scrape_timestamp"], utc=True, errors="coerce")
        create_dt  = pd.to_datetime(df["create_time"], unit="s", utc=True, errors="coerce")
        df["days_since_posting"] = (scrape_dt - create_dt).dt.total_seconds() / 86400
        df["views_per_day"] = df["view_count"] / df["days_since_posting"].replace(0, np.nan)

    if "follower_count" in df.columns:
        df["VPF"]     = df["view_count"] / df["follower_count"].replace(0, np.nan)
        df["log_VPF"] = np.log(df["VPF"].clip(lower=0) + 1)

    return df, hashtags


def hashtag_frequency(hashtag_series):
    from collections import Counter
    counts = Counter(
        tag.lower()
        for tags in hashtag_series
        for tag in tags
        if isinstance(tag, str)
    )
    return pd.DataFrame(counts.most_common(), columns=["hashtag", "count"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=os.path.join(DATA_DIR, "combined", "accounts_combined.csv"),
                        help="Input CSV (default: data/combined/accounts_combined.csv). "
                             "Use data/combined/hashtags_combined.csv for hashtag-collected data.")
    parser.add_argument("--output", default=os.path.join(DATA_DIR, "enriched", "accounts_enriched.csv"),
                        help="Output CSV (default: data/enriched/accounts_enriched.csv).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Not found: {args.input}")
        print("Run collect_accounts.py --merge or collect_hashtags.py --merge first.")
        return

    df = pd.read_csv(args.input, dtype={"music_id": str, "id": str})
    print(f"Loaded {len(df)} videos from {args.input}")

    df, hashtag_series = enrich(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved enriched data → {args.output}")

    freq_path = os.path.join(DATA_DIR, "supplementary", "hashtag_frequency.csv")
    os.makedirs(os.path.dirname(freq_path), exist_ok=True)
    freq = hashtag_frequency(hashtag_series)
    freq.to_csv(freq_path, index=False)
    print(f"Saved hashtag frequency table → {freq_path}")
    print(f"\nTop 20 hashtags:")
    print(freq.head(20).to_string(index=False))
    print(f"\nReview {freq_path} and update VIRALITY_HASHTAGS in enrich.py if needed.")

    print(f"\nDuets: {df['is_duet'].sum()}, Stitches: {df['is_stitch'].sum()}")
    print(f"Has effects: {df['has_effects'].sum()}")
    print(f"Has virality hashtag: {df['has_virality_hashtag'].sum()}")


if __name__ == "__main__":
    main()
