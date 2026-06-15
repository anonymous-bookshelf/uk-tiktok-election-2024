"""
collect_partisan_accounts.py

Collects all election-window videos for every GB account that posted ≥1
partisan video in the hashtag corpus. Used as the foundation for a clean
within-account amplification analysis.

Why
---
The hashtag corpus only captures videos that used one of the 63 tracked
hashtags. An account's apolitical videos will mostly be absent. This script
does a full timeline pull for qualifying accounts so we have their complete
election-window output — not just the slice that appeared in hashtag searches.

Qualifying accounts
-------------------
GB-coded accounts in data/classified/hashtags_classified.csv with ≥1 video
classified as a named party (Labour / Conservative / Reform UK / Green / SNP
/ Lib Dem). No minimum on apolitical videos — the point is to collect
everything so we can compute apolitical mean views from the full timeline.

Usage
-----
    python scripts/collect_partisan_accounts.py           # collect
    python scripts/collect_partisan_accounts.py --status  # check progress
    python scripts/collect_partisan_accounts.py --merge   # merge to combined CSV

Output
------
    data/raw/partisan_accounts/<username>.csv     — election-window videos per account
    data/raw/partisan_accounts/progress.json      — resume tracking
    data/combined/partisan_accounts_combined.csv  — merged file (after --merge)
"""

import argparse
import json
import os
import time
from datetime import date, timedelta, datetime, timezone

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

ELECTION_START = "2024-05-22"
ELECTION_END   = "2024-07-18"
CHUNK_DAYS     = 28

PARTIES = ["Labour", "Conservative", "Reform UK", "Green", "SNP", "Lib Dem"]

MAX_PER_REQUEST  = 100
REQUESTS_PER_DAY = 1000
SLEEP_BETWEEN    = 2.0

OUTPUT_DIR    = os.path.join(ROOT_DIR, "data", "raw", "partisan_accounts")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
COMBINED_OUT  = os.path.join(ROOT_DIR, "data", "combined", "partisan_accounts_combined.csv")

CLASSIFIED_PATH = os.path.join(ROOT_DIR, "data", "classified", "hashtags_classified.csv")


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
# Identify qualifying accounts
# --------------------------------------------------------------------------- #

def get_qualifying_accounts():
    """Return sorted list of GB usernames with ≥1 partisan video, plus a random
    sample of 500 accounts with ≥1 non-partisan video (seed=42)."""
    import random

    ht = pd.read_csv(os.path.join(ROOT_DIR, "data", "combined", "hashtags_combined.csv"), dtype={"id": str})
    cl = pd.read_csv(CLASSIFIED_PATH, dtype={"id": str})[["id", "predicted_party"]].drop_duplicates("id")
    df = ht.merge(cl, on="id", how="left")
    gb = df[df["region_code"].str.lower() == "gb"]

    # Partisan accounts (original set)
    partisan_usernames = set(
        gb[gb["predicted_party"].isin(PARTIES)]["username"].dropna().unique()
    )

    # Non-partisan accounts not already in partisan set
    nonpartisan_candidates = []
    for username, grp in gb.groupby("username"):
        if username in partisan_usernames:
            continue
        if (grp["predicted_party"] == "Non-partisan").any():
            nonpartisan_candidates.append(username)

    random.seed(42)
    sample = set(random.sample(nonpartisan_candidates, min(500, len(nonpartisan_candidates))))

    all_accounts = sorted(partisan_usernames | sample)
    print(f"  Partisan accounts: {len(partisan_usernames):,}")
    print(f"  Non-partisan sample: {len(sample):,}")
    print(f"  Total: {len(all_accounts):,}")
    return all_accounts


# --------------------------------------------------------------------------- #
# Video collection
# --------------------------------------------------------------------------- #

def date_chunks(start_str, end_str, chunk_days):
    start = date.fromisoformat(start_str)
    end   = date.fromisoformat(end_str)
    cur   = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def fetch_chunk(token, username, start, end, progress):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    videos    = []
    cursor    = 0
    search_id = ""

    while True:
        payload = {
            "query": {
                "and": [
                    {"operation": "EQ", "field_name": "username", "field_values": [username]},
                ]
            },
            "start_date": start.strftime("%Y%m%d"),
            "end_date":   end.strftime("%Y%m%d"),
            "max_count":  MAX_PER_REQUEST,
            "cursor":     cursor,
        }
        if search_id:
            payload["search_id"] = search_id

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
                    print(f"    Network error ({e}) — retrying in {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    raise

            if resp.status_code == 429:
                if attempt < 4:
                    print(f"    Rate limited — sleeping 90s (attempt {attempt+1}/5)")
                    time.sleep(90)
                else:
                    print("    Rate limit exhausted — stopping")
                    return videos, True
            elif resp.status_code == 500:
                wait = 5 * (2 ** attempt)
                if attempt < 4:
                    print(f"    500 error — retrying in {wait}s (attempt {attempt+1}/5)")
                    time.sleep(wait)
                else:
                    break
            elif resp.status_code == 401:
                return videos, False  # caller handles token refresh
            else:
                break

        if resp is None or resp.status_code in (429, 500):
            break

        resp.raise_for_status()
        data = resp.json().get("data", {})
        page = data.get("videos", [])
        videos.extend(page)
        search_id = data.get("search_id", search_id)
        has_more  = data.get("has_more", False)
        cursor    = data.get("cursor", 0)
        time.sleep(SLEEP_BETWEEN)
        if not has_more or not page:
            break

    return videos, False


def fetch_account(token, username, progress):
    all_videos = []
    for start, end in date_chunks(ELECTION_START, ELECTION_END, CHUNK_DAYS):
        videos, rate_limited = fetch_chunk(token, username, start, end, progress)
        all_videos.extend(videos)
        if rate_limited:
            return all_videos, True
    return all_videos, False


def save_account(videos, username, scrape_ts):
    if not videos:
        return 0
    df = pd.DataFrame(videos)
    if "music_id" in df.columns:
        df["music_id"] = [str(v.get("music_id", "")) for v in videos]
    df["_source_username"] = username
    df["scrape_timestamp"]  = scrape_ts
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{username}.csv")
    df.to_csv(path, index=False)
    return len(df)


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

def merge_all():
    frames = []
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if not f.endswith(".csv"):
            continue
        frames.append(pd.read_csv(os.path.join(OUTPUT_DIR, f), dtype={"id": str}))
    if not frames:
        print("No CSVs to merge.")
        return
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
    os.makedirs(os.path.dirname(COMBINED_OUT), exist_ok=True)
    combined.to_csv(COMBINED_OUT, index=False)
    print(f"Merged {len(combined):,} election-window videos → {COMBINED_OUT}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Show collection progress")
    parser.add_argument("--merge",  action="store_true", help="Merge per-account CSVs")
    args = parser.parse_args()

    if args.merge:
        merge_all()
        return

    accounts = get_qualifying_accounts()
    print(f"Qualifying accounts: {len(accounts):,}")

    progress = reset_daily_count(load_progress())

    if args.status:
        done = len(progress["completed"])
        remaining = [a for a in accounts if a not in progress["completed"]]
        print(f"Collected: {done}/{len(accounts)} ({len(remaining)} remaining)")
        print(f"Requests today: {progress['request_count_today']}/{REQUESTS_PER_DAY}")
        return

    if not CLIENT_KEY or not CLIENT_SECRET:
        raise SystemExit("Set TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET in .env")

    print(f"Election window: {ELECTION_START} → {ELECTION_END}")
    print(f"Requests used today: {progress['request_count_today']}/{REQUESTS_PER_DAY}\n")

    token = get_access_token()
    token_fetched_at = time.time()
    scrape_ts = datetime.now(timezone.utc).isoformat()

    for username in accounts:
        if username in progress["completed"]:
            continue

        if progress["request_count_today"] >= REQUESTS_PER_DAY:
            done = len(progress["completed"])
            print(f"Daily limit reached ({done}/{len(accounts)} done). Resume tomorrow.")
            save_progress(progress)
            return

        # Refresh token every 90 minutes
        if time.time() - token_fetched_at > 5400:
            token = get_access_token()
            token_fetched_at = time.time()

        print(f"  @{username}...", end=" ", flush=True)
        try:
            videos, rate_limited = fetch_account(token, username, progress)
            n = save_account(videos, username, scrape_ts)
            progress["completed"].append(username)
            save_progress(progress)
            print(f"{n} videos")
            if rate_limited:
                done = len(progress["completed"])
                print(f"Rate limit hit ({done}/{len(accounts)} done). Resume tomorrow.")
                return
        except Exception as e:
            print(f"ERROR: {e}")
            save_progress(progress)

    print(f"\nCollection complete — {len(accounts):,} accounts.")
    merge_all()


if __name__ == "__main__":
    main()
