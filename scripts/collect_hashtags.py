"""
collect_hashtags.py

Collects TikTok videos matching election-related hashtags via the Research API.
Queries are chunked into ≤30-day windows to stay within API limits.

Usage
-----
    python collect_hashtags.py              # collect all date chunks
    python collect_hashtags.py --status     # show progress without collecting
    python collect_hashtags.py --merge      # merge per-chunk CSVs into hashtag_combined.csv

Configuration
-------------
    START_DATE, END_DATE, CHUNK_DAYS — define the collection window.
    HASHTAGS — list of hashtags to match (OR logic: any video with any of these).

Output
------
    data/hashtag_collected/<YYYYMMDD>_<YYYYMMDD>.csv  — per-chunk raw CSVs
    data/hashtag_collected/progress.json               — tracks completed chunks + daily quota
    data/hashtag_combined.csv                          — merged deduplicated (after --merge)
"""

import argparse
import json
import os
import re
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

CLIENT_KEY    = os.getenv("TIKTOK_CLIENT_KEY")
CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET")

TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
QUERY_URL = "https://open.tiktokapis.com/v2/research/video/query/"

VIDEO_FIELDS = ",".join([
    "id", "create_time", "username", "video_description",
    "music_id", "like_count", "comment_count", "share_count",
    "view_count", "favorites_count", "effect_ids", "hashtag_names",
    "video_duration", "voice_to_text", "video_label",
    "video_mention_list", "region_code",
])

# Canonical column order — every page is reindexed to this before writing.
# Prevents column misalignment when the API omits fields for some videos.
COLUMNS = [
    "id", "create_time", "username", "video_description",
    "music_id", "like_count", "comment_count", "share_count",
    "view_count", "favorites_count", "effect_ids", "hashtag_names",
    "video_duration", "voice_to_text", "video_label",
    "video_mention_list", "region_code", "scrape_timestamp",
]

# UK 2024 election: called 22 May, polling day 4 July, +2-week post-election window.
START_DATE = "2024-05-22"
END_DATE   = "2024-07-18"
CHUNK_DAYS = 28   # ≤30 to stay within the API's per-request date-range limit

HASHTAGS = [
    # General election (non-partisan)
    "ukelection2024", "ukelection", "ge2024", "ge2024uk",
    "ukge2024", "ukgeneralelection", "election2024uk", "voteuk", "ukvote2024",
    "generalelection2024", "ukpolitics",
    "rishisunakandkeirstarmer",
    "leadersdebate", "leadersdebate2024",
    "tacticalvoting", "tacticalvote",
    "parliament", "election2024", "government",

    # Labour
    "labourdoorstep", "votelabour", "keirstarmer", "toriesout",
    "labour", "labourparty", "starmer",
    "angelarayner", "rachelreeves", "labourgovernment", "teamlabour",

    # Conservative
    "toryparty", "rishisunak", "voteconservative", "neverlabour",
    "tories", "tory", "conservative", "conservatives", "sunak",
    "conservativeparty", "toryleadership",

    # Reform UK
    "reformuk", "nigelfarage", "votereform",
    "farage", "reform", "richardtice",
    "reformparty", "reformukparty",

    # SNP
    "snp", "votesnp",
    "johnswinney", "scottishindependence", "indyref",

    # Lib Dem
    "libdems", "votelibdem",
    "liberaldemocrats", "libdem", "eddavey",

    # Green
    "greenparty", "votegreen",
    "carladenyer", "adrianramsay", "greenwave",

    # Plaid Cymru
    "plaidcymru", "voteplaid", "lukefletcher",

    # Workers' Party
    "workersparty", "georgegalloway",
]

# Excluded (global/generic volume — too noisy):
#   "generalelection"
# Reinstated with GB filter:
#   "election2024", "conservatives", "parliament", "government", "reformparty", "reformukparty"
# Reinstated (date window limits noise sufficiently):
#   "generalelection2024", "ukpolitics" — added back May 2026
# Excluded (non-major-6 parties):
#   "plaidcymru"

MAX_PER_REQUEST  = 100
REQUESTS_PER_DAY = 1000
SLEEP_BETWEEN    = 6.0

OUTPUT_DIR    = os.path.join(ROOT_DIR, "data", "raw", "hashtags")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #

def get_access_token():
    resp = requests.post(TOKEN_URL, data={
        "client_key":    CLIENT_KEY,
        "client_secret": CLIENT_SECRET,
        "grant_type":    "client_credentials",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


# --------------------------------------------------------------------------- #
# Date chunking
# --------------------------------------------------------------------------- #

def date_chunks(start_str, end_str, chunk_days):
    start = date.fromisoformat(start_str)
    end   = date.fromisoformat(end_str)
    cur   = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def chunk_key(start, end):
    return f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"


# --------------------------------------------------------------------------- #
# Video collection
# --------------------------------------------------------------------------- #

def fetch_and_save_chunk(token, start, end, key, scrape_ts, progress, gb_only=False, hashtags=None):
    """
    Fetch all videos for a chunk, writing to disk and updating progress after
    every page. Never loses data — any interruption leaves the CSV intact and
    the next run resumes from where it left off.

    On resume, attempts to continue from the saved cursor + search_id (works for
    quick restarts within the same API session). If the search_id has expired the
    API will return 0 results; the code detects this and falls back to fetching
    from cursor=0, skipping pages where all IDs are already on disk.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    token_fetched_at = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{key}.csv")

    # Resume: load IDs already saved from a previous partial run of this chunk
    seen_ids    = set()
    first_write = True
    if os.path.exists(path):
        seen_ids = set(pd.read_csv(path, usecols=["id"], dtype={"id": str})["id"])
        first_write = False
        print(f"\n    Resuming — {len(seen_ids)} videos already on disk", end=" ", flush=True)

    total_saved = len(seen_ids)
    completed   = False  # set True only when API signals no more pages

    # Attempt to resume from last saved position (avoids re-fetching already-seen pages).
    # Falls back to cursor=0 if the search_id has expired.
    partial      = progress.get("partial", {}).get(key, {})
    cursor       = partial.get("cursor", 0)
    search_id    = partial.get("search_id", "")
    resumed_mid  = bool(search_id and seen_ids)

    active_hashtags = hashtags if hashtags else HASHTAGS
    hashtag_conditions = [
        {"operation": "EQ", "field_name": "hashtag_name", "field_values": [tag]}
        for tag in active_hashtags
    ]

    try:
        while True:
            if time.time() - token_fetched_at > 5400:
                token = get_access_token()
                token_fetched_at = time.time()
                headers["Authorization"] = f"Bearer {token}"

            query = {"or": hashtag_conditions}
            if gb_only:
                query["and"] = [{"operation": "EQ", "field_name": "region_code", "field_values": ["GB"]}]

            payload = {
                "query":      query,
                "start_date": start.strftime("%Y%m%d"),
                "end_date":   end.strftime("%Y%m%d"),
                "max_count":  MAX_PER_REQUEST,
                "cursor":     cursor,
            }
            if search_id:
                payload["search_id"] = search_id

            # Retry loop — handles HTTP errors AND network-level failures
            resp = None
            for attempt in range(5):
                try:
                    resp = requests.post(QUERY_URL, params={"fields": VIDEO_FIELDS},
                                         headers=headers, json=payload, timeout=30)
                    progress["request_count_today"] += 1
                    save_progress(progress)
                except requests.exceptions.RequestException as e:
                    wait = 5 * (2 ** attempt)
                    if attempt < 4:
                        print(f"    Network error ({e}) — retrying in {wait}s (attempt {attempt+1}/5)")
                        time.sleep(wait)
                        continue
                    else:
                        raise

                if resp.status_code == 429:
                    print(f"    Rate limited — sleeping 90s (attempt {attempt+1}/5)")
                    time.sleep(90)
                elif resp.status_code == 500:
                    wait = 5 * (2 ** attempt)
                    if attempt < 4:
                        print(f"    500 error — retrying in {wait}s (attempt {attempt+1}/5)")
                        time.sleep(wait)
                    else:
                        print("    500 error — giving up on this page")
                        break
                else:
                    break  # success or unrecoverable status — handle below

            if resp is None:
                break  # all network retries exhausted

            if resp.status_code == 500:
                break  # gave up on this page, move on

            if resp.status_code == 429:
                print(f"\n    Rate limit exhausted after 5 attempts — stopping to protect quota")
                break

            if resp.status_code in (401, 403):
                print(f"\n    {resp.status_code} — session expired ({total_saved} videos saved so far)")
                break

            resp.raise_for_status()
            data   = resp.json().get("data", {})
            videos = data.get("videos", [])

            # Detect expired search_id: resumed mid-chunk but got 0 results on first attempt.
            # Reset to cursor=0 and retry this iteration without a search_id.
            if resumed_mid and not videos:
                print(f"\n    search_id expired — restarting from page 1 (dedup will skip seen videos)", end=" ", flush=True)
                cursor      = 0
                search_id   = ""
                resumed_mid = False
                continue

            resumed_mid = False  # only relevant for the very first page after a resume

            # Deduplicate within the page first, then against already-seen IDs
            seen_this_page = {}
            for v in videos:
                vid_id = str(v.get("id"))
                if vid_id not in seen_ids and vid_id not in seen_this_page:
                    seen_this_page[vid_id] = v
            new_videos = list(seen_this_page.values())
            if new_videos:
                df = pd.DataFrame(new_videos)
                # Stringify music_id from original Python dicts (exact ints from JSON)
                # before pandas can float64-coerce large IDs when any row has a null music_id.
                if "music_id" in df.columns:
                    df["music_id"] = [str(v["music_id"]) if v.get("music_id") is not None else "" for v in new_videos]
                df["scrape_timestamp"] = scrape_ts
                df = df.reindex(columns=COLUMNS)
                df.to_csv(path, mode="a", header=first_write, index=False)
                first_write = False
                seen_ids.update(str(v["id"]) for v in new_videos)
                total_saved += len(new_videos)

            search_id = data.get("search_id", search_id)
            has_more  = data.get("has_more", False)
            cursor    = data.get("cursor", 0)

            # Persist cursor position so a resume can attempt to skip already-fetched pages
            progress.setdefault("partial", {})[key] = {"cursor": cursor, "search_id": search_id}
            save_progress(progress)

            time.sleep(SLEEP_BETWEEN)

            if not has_more or not videos:
                completed = True
                break

    except Exception as e:
        print(f"\n    Unexpected error: {e} — {total_saved} videos saved so far")

    return total_saved, completed


# --------------------------------------------------------------------------- #
# Progress tracking
# --------------------------------------------------------------------------- #

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "request_count_today": 0, "date": str(datetime.now(timezone.utc).date())}


def save_progress(p):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)


def reset_daily_count(p):
    today = str(datetime.now(timezone.utc).date())
    if p.get("date") != today:
        p["request_count_today"] = 0
        p["date"] = today
    return p


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #

def validate_chunk(path, key):
    """
    Check that a collected chunk CSV is structurally clean.

    Verifies:
      - Header exactly matches COLUMNS
      - Every data row has exactly len(COLUMNS) fields
      - Every data row has a valid video ID (15-20 digit integer) in the id column

    Returns True if clean, False otherwise. Prints a one-line summary either way.
    """
    import csv as _csv

    id_col = COLUMNS.index("id")
    wrong_field_count = 0
    bad_id = 0
    total = 0

    with open(path, newline="") as f:
        reader = _csv.reader(f)
        header = next(reader, None)
        if header != COLUMNS:
            print(f"\n  ✗ VALIDATION FAILED for {key}: header mismatch")
            print(f"    Expected: {COLUMNS}")
            print(f"    Got:      {header}")
            return False
        for row in reader:
            total += 1
            if len(row) != len(COLUMNS):
                wrong_field_count += 1
            elif not re.fullmatch(r"\d{15,20}", row[id_col].strip()):
                bad_id += 1

    if wrong_field_count or bad_id:
        print(f"\n  ✗ VALIDATION FAILED for {key}: "
              f"{wrong_field_count} rows with wrong field count, "
              f"{bad_id} rows with invalid id  (out of {total} total)")
        return False

    print(f"    ✓ Validated: {total} rows, all clean")
    return True


def merge_all():
    frames = []
    skipped = []
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if not f.endswith(".csv"):
            continue
        path = os.path.join(OUTPUT_DIR, f)
        key  = f[:-4]
        if validate_chunk(path, key):
            frames.append(pd.read_csv(path, dtype={"id": str, "music_id": str}))
        else:
            skipped.append(f)

    if skipped:
        print(f"\nWARNING: skipped {len(skipped)} invalid chunk(s) from merge: {skipped}")
        print("Re-run collection to fix these before merging.\n")

    if not frames:
        print("No valid CSVs found to merge.")
        return

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["id"])
        .sort_values("create_time")
        .reset_index(drop=True)
    )
    out_path = os.path.join(ROOT_DIR, "data", "combined", "hashtags_combined.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Merged {len(combined)} unique videos → {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status",     action="store_true")
    parser.add_argument("--merge",      action="store_true")
    parser.add_argument("--gb-only",    action="store_true",
                        help="Filter to GB region_code only (saves ~38%% of requests)")
    parser.add_argument("--start-date", default=None,
                        help="Override start date (YYYY-MM-DD). Use with --end-date to collect a custom window.")
    parser.add_argument("--end-date",   default=None,
                        help="Override end date (YYYY-MM-DD).")
    parser.add_argument("--hashtags",   nargs="+", default=None,
                        help="Run only these specific hashtags instead of the full list.")
    args = parser.parse_args()

    start_date = args.start_date or START_DATE
    end_date   = args.end_date   or END_DATE
    chunks = list(date_chunks(start_date, end_date, CHUNK_DAYS))

    if args.merge:
        merge_all()
        return

    progress = reset_daily_count(load_progress())

    if args.status:
        done = len(progress["completed"])
        print(f"Progress: {done}/{len(chunks)} chunks collected")
        print(f"Requests today: {progress['request_count_today']}/{REQUESTS_PER_DAY}\n")
        for start, end in chunks:
            key    = chunk_key(start, end)
            status = "✓" if key in progress["completed"] else "○"
            print(f"  {status} {start} → {end}  ({key})")
        return

    if not CLIENT_KEY or not CLIENT_SECRET:
        print("Error: TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET must be set in ../.env")
        return

    active_hashtags = args.hashtags if args.hashtags else HASHTAGS
    print(f"Collecting {len(chunks)} date chunks ({start_date} → {end_date})")
    print(f"Hashtags ({len(active_hashtags)}): {', '.join(active_hashtags)}")
    if args.gb_only:
        print("Region filter: GB only")
    print(f"Requests used today: {progress['request_count_today']}/{REQUESTS_PER_DAY}\n")

    token            = get_access_token()
    token_fetched_at = time.time()
    scrape_ts        = datetime.now(timezone.utc).isoformat()

    key_suffix = "_extra" if args.hashtags else ""

    for start, end in chunks:
        key = chunk_key(start, end) + key_suffix

        if key in progress["completed"]:
            print(f"  ✓ {key} (already done)")
            continue

        if progress["request_count_today"] >= REQUESTS_PER_DAY:
            print("\nDaily request limit reached. Resume tomorrow.")
            save_progress(progress)
            return

        if time.time() - token_fetched_at > 5400:
            token            = get_access_token()
            token_fetched_at = time.time()

        print(f"  Fetching {start} → {end}...", end=" ", flush=True)
        n_saved, completed = fetch_and_save_chunk(token, start, end, key, scrape_ts, progress,
                                                   gb_only=args.gb_only,
                                                   hashtags=args.hashtags if args.hashtags else None)
        csv_path = os.path.join(OUTPUT_DIR, f"{key}.csv")
        print(f"{n_saved} videos ({progress['request_count_today']} requests today)")

        if not os.path.exists(csv_path):
            print(f"  WARNING: no file written for {key} — will retry next run")
        elif not completed:
            print(f"  WARNING: collection stopped early for {key} (500 error or interruption).")
            print(f"  Chunk NOT marked complete — re-run to resume from where it stopped.")
        elif not validate_chunk(csv_path, key):
            print(f"  Deleting corrupted file so next run re-collects it cleanly.")
            os.remove(csv_path)
        else:
            progress["completed"].append(key)
            progress.get("partial", {}).pop(key, None)
            save_progress(progress)

    print("\nCollection complete. Run with --merge to combine chunk files.")
    merge_all()


if __name__ == "__main__":
    main()
