"""
partition_music_work.py

Splits unprocessed music_ids into N shards so multiple fetch_music_lookup.py
instances can run in parallel.

Usage
-----
    python partition_music_work.py --shards 3
    python partition_music_work.py --shards 3 --input data/enriched/hashtags_enriched.csv

After partitioning, open N terminal windows and run the printed commands.
When all shards finish, run with --merge to combine into music_lookup.csv.

    python partition_music_work.py --merge
"""

import argparse
import os

import pandas as pd

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
ENRICHED    = os.path.join(DATA_DIR, "enriched", "hashtags_enriched.csv")
LOOKUP_OUT  = os.path.join(DATA_DIR, "music_lookup", "hashtags_music_lookup.csv")
SHARDS_DIR  = os.path.join(DATA_DIR, "music_lookup", "shards")


def partition(input_path, n_shards):
    df = pd.read_csv(input_path, dtype={"music_id": str, "id": str})

    # Find unprocessed music_ids
    processed = set()
    if os.path.exists(LOOKUP_OUT):
        existing = pd.read_csv(LOOKUP_OUT, dtype={"music_id": str})
        processed = set(existing["music_id"].astype(str))

    # Build video_map for unprocessed IDs only
    def _is_int(val):
        try:
            int(str(val).strip())
            return True
        except (ValueError, TypeError):
            return False

    valid = df[df["music_id"].apply(_is_int) & df["id"].apply(_is_int)]
    video_map = (
        valid.dropna(subset=["music_id", "id", "username"])
             .drop_duplicates(subset=["music_id"])
             .set_index("music_id")[["id", "username"]]
    )
    remaining = [mid for mid in video_map.index if str(mid) not in processed]
    print(f"{len(processed)} already complete, {len(remaining)} remaining → splitting into {n_shards} shards\n")

    os.makedirs(SHARDS_DIR, exist_ok=True)

    chunk_size = len(remaining) // n_shards + 1
    for i in range(n_shards):
        shard_ids  = remaining[i * chunk_size : (i + 1) * chunk_size]
        if not shard_ids:
            continue
        shard_rows = video_map.loc[shard_ids].reset_index()
        # Keep only the columns fetch_music_lookup.py needs
        shard_input  = os.path.join(SHARDS_DIR, f"shard_{i}_input.csv")
        shard_output = os.path.join(SHARDS_DIR, f"shard_{i}_output.csv")
        shard_rows.to_csv(shard_input, index=False)
        print(f"Shard {i}: {len(shard_ids)} IDs → {shard_input}")
        print(f"  python3 fetch_music_lookup.py --input {shard_input} --output {shard_output}")

    print(f"\nWhen all shards finish, merge with:")
    print(f"  python3 partition_music_work.py --merge")


def merge():
    frames = [pd.read_csv(LOOKUP_OUT, dtype={"music_id": str})] if os.path.exists(LOOKUP_OUT) else []

    shard_files = sorted(
        os.path.join(SHARDS_DIR, f)
        for f in os.listdir(SHARDS_DIR)
        if f.startswith("shard_") and f.endswith("_output.csv")
    ) if os.path.exists(SHARDS_DIR) else []

    for path in shard_files:
        if os.path.exists(path):
            frames.append(pd.read_csv(path, dtype={"music_id": str}))
            print(f"  + {path} ({len(pd.read_csv(path))} rows)")

    if not frames:
        print("Nothing to merge.")
        return

    combined = (
        pd.concat(frames, ignore_index=True)
          .drop_duplicates(subset=["music_id"], keep="last")
    )
    _tmp = LOOKUP_OUT + ".tmp"
    combined.to_csv(_tmp, index=False)
    os.replace(_tmp, LOOKUP_OUT)
    print(f"\nMerged {len(combined)} unique music_ids → {LOOKUP_OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--input",  default=ENRICHED)
    parser.add_argument("--merge",  action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge()
    else:
        partition(args.input, args.shards)


if __name__ == "__main__":
    main()
