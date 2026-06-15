"""
collect_baseline_accounts.py

Collects pre-election videos for accounts that appear in the hashtag corpus
amplification analysis. Used to build a richer, cleaner baseline for the
within-account amplification proxy.

How it works
------------
Rather than using apolitical videos from *within* the election window as the
baseline (which may themselves be affected by the election atmosphere), this
script collects each qualifying account's videos from a pre-election window
(1 Jan 2024 – 21 May 2024). Mean views from that period = the account's
"normal" reach before the election campaign started.

The amplification ratio then becomes:
    ratio = mean views of partisan election-window video(s)
          / mean views of pre-election videos

This cleanly separates the baseline from the treatment period.

Qualifying accounts
-------------------
GB-coded accounts in the hashtag corpus that have:
  - ≥ 3 total videos in the hashtag corpus
  - ≥ 1 video classified as a named party (Labour / Conservative / Reform UK /
    Green / SNP / Lib Dem)
  - ≥ 1 video classified as Apolitical

Usage
-----
    python collect_baseline_accounts.py           # collect
    python collect_baseline_accounts.py --status  # check progress
    python collect_baseline_accounts.py --merge   # merge to combined CSV
    python collect_baseline_accounts.py --analyse # run amplification analysis

Output
------
    data/raw/baseline_accounts/<username>.csv     — pre-election videos per account
    data/raw/baseline_accounts/progress.json      — resume tracking
    data/combined/baseline_accounts_combined.csv  — merged file (after --merge)
"""

import argparse
import json
import os
import time
from datetime import date, timedelta, datetime, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy import stats

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

# Pre-election baseline window — before the election was called (22 May 2024)
BASELINE_START = "2024-01-01"
BASELINE_END   = "2024-05-21"
CHUNK_DAYS     = 28

# Election window (for the amplification analysis)
ELECTION_START = "2024-05-22"
ELECTION_END   = "2024-07-18"

PARTIES = ["Labour", "Conservative", "Reform UK", "Green", "SNP", "Lib Dem"]

MIN_TOTAL_VIDEOS  = 3   # minimum videos in hashtag corpus to qualify
MAX_PER_REQUEST   = 100
REQUESTS_PER_DAY  = 1000
SLEEP_BETWEEN     = 2.0

OUTPUT_DIR    = os.path.join(ROOT_DIR, "data", "raw", "baseline_accounts")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
COMBINED_OUT  = os.path.join(ROOT_DIR, "data", "combined", "baseline_accounts_combined.csv")

HASHTAGS_PATH    = os.path.join(ROOT_DIR, "data", "combined", "hashtags_combined.csv")
CLASSIFIED_PATH  = os.path.join(ROOT_DIR, "data", "classified", "hashtags_classified.csv")
PARTISAN_COMBINED = os.path.join(ROOT_DIR, "data", "combined", "partisan_accounts_combined.csv")


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
    """Return list of GB account usernames that qualify for the pre-election baseline.

    Primary source: partisan_accounts_combined.csv (full election-window timelines).
    Accounts with ≥ MIN_TOTAL_VIDEOS in that file are likely to have enough data
    for the analysis and are worth collecting a pre-election baseline for.

    Fallback: if the full timeline pull hasn't been done yet, derive accounts from
    the hashtag classified corpus using the old ≥1-partisan + ≥1-apolitical logic.
    """
    if os.path.exists(PARTISAN_COMBINED):
        df = pd.read_csv(PARTISAN_COMBINED, dtype={"id": str})
        counts = df.groupby("_source_username").size()
        qualifying = sorted(counts[counts >= MIN_TOTAL_VIDEOS].index.tolist())
        print(f"(Using partisan_accounts_combined.csv — {len(qualifying)} accounts with ≥{MIN_TOTAL_VIDEOS} election-window videos)")
        return qualifying

    # Fallback: old logic using hashtag corpus + classified labels
    print("(partisan_accounts_combined.csv not found — falling back to hashtag corpus)")
    df = pd.read_csv(HASHTAGS_PATH, dtype={"id": str})
    cl = pd.read_csv(CLASSIFIED_PATH, dtype={"id": str})[["id", "predicted_party"]].drop_duplicates("id")
    df = df.merge(cl, on="id", how="left")
    df = df[df["region_code"].str.lower() == "gb"]

    qualifying = []
    for username, grp in df.groupby("username"):
        if len(grp) < MIN_TOTAL_VIDEOS:
            continue
        has_partisan   = grp["predicted_party"].isin(PARTIES).any()
        has_apolitical = (grp["predicted_party"] == "Apolitical").any()
        if has_partisan and has_apolitical:
            qualifying.append(username)

    return qualifying


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
                    return videos, True   # True = hit limit
            elif resp.status_code == 500:
                wait = 5 * (2 ** attempt)
                if attempt < 4:
                    print(f"    500 error — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    break
            elif resp.status_code == 401:
                return videos, False   # signal token expiry to caller
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
    for start, end in date_chunks(BASELINE_START, BASELINE_END, CHUNK_DAYS):
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
        if f.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(OUTPUT_DIR, f), dtype={"id": str}))
    if not frames:
        print("No CSVs to merge.")
        return
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])
    os.makedirs(os.path.dirname(COMBINED_OUT), exist_ok=True)
    combined.to_csv(COMBINED_OUT, index=False)
    print(f"Merged {len(combined):,} pre-election videos → {COMBINED_OUT}")


# --------------------------------------------------------------------------- #
# Amplification analysis
# --------------------------------------------------------------------------- #

def run_analysis():
    if not os.path.exists(COMBINED_OUT):
        print("No baseline data found. Run collection and --merge first.")
        return

    # Load election-window hashtag corpus with labels
    ht = pd.read_csv(HASHTAGS_PATH, dtype={"id": str})
    cl = pd.read_csv(CLASSIFIED_PATH, dtype={"id": str})[["id", "predicted_party"]].drop_duplicates("id")
    ht = ht.merge(cl, on="id", how="left")
    ht = ht[ht["region_code"] == "gb"]

    # Load pre-election baseline
    bl = pd.read_csv(COMBINED_OUT, dtype={"id": str})

    print(f"Baseline videos: {len(bl):,} from {bl['_source_username'].nunique():,} accounts")
    print(f"Election-window classified videos: {ht['predicted_party'].notna().sum():,}\n")

    for party in PARTIES:
        ratios = []
        for username, grp_el in ht.groupby("username"):
            partisan  = grp_el[grp_el["predicted_party"] == party]
            if len(partisan) == 0:
                continue

            # Pre-election baseline for this account
            grp_bl = bl[bl["_source_username"] == username]
            if len(grp_bl) == 0:
                continue

            baseline = grp_bl["view_count"].mean()
            if baseline == 0:
                continue

            ratio = partisan["view_count"].mean() / baseline
            ratios.append(ratio)

        if len(ratios) < 5:
            print(f"{party:<15} insufficient data (n={len(ratios)})")
            continue

        median_r = np.median(ratios)
        mean_r   = np.mean(ratios)
        _, p = stats.wilcoxon([r - 1 for r in ratios])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{party:<15} n={len(ratios):>4}  median={median_r:.2f}x  mean={mean_r:.2f}x  p={p:.4f} {sig}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status",  action="store_true", help="Show progress")
    parser.add_argument("--merge",   action="store_true", help="Merge CSVs")
    parser.add_argument("--analyse", action="store_true", help="Run amplification analysis")
    args = parser.parse_args()

    if args.merge:
        merge_all()
        return

    if args.analyse:
        run_analysis()
        return

    accounts = get_qualifying_accounts()
    print(f"Qualifying accounts: {len(accounts):,}")

    progress = reset_daily_count(load_progress())

    if args.status:
        done = len(progress["completed"])
        print(f"Collected: {done}/{len(accounts)}")
        print(f"Requests today: {progress['request_count_today']}/{REQUESTS_PER_DAY}")
        return

    if not CLIENT_KEY or not CLIENT_SECRET:
        raise SystemExit("Set TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET in .env")

    print(f"Baseline window: {BASELINE_START} → {BASELINE_END}")
    print(f"Requests used today: {progress['request_count_today']}/{REQUESTS_PER_DAY}\n")

    token = get_access_token()
    token_fetched_at = time.time()
    scrape_ts = datetime.now(timezone.utc).isoformat()

    for username in accounts:
        if username in progress["completed"]:
            continue

        if progress["request_count_today"] >= REQUESTS_PER_DAY:
            print("Daily limit reached. Resume tomorrow.")
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
            print(f"{n} pre-election videos")
            if rate_limited:
                print("Rate limit hit — resume tomorrow.")
                return
        except Exception as e:
            print(f"ERROR: {e}")
            save_progress(progress)

    print("\nCollection complete.")
    merge_all()


if __name__ == "__main__":
    main()
