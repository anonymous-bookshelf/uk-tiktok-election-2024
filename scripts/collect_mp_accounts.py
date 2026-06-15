"""
collect_mp_accounts.py

Collects TikTok videos from verified MP accounts via the Research API.
Separate from party-account collection — outputs to data/raw/mp_accounts/.

Input:  verified_mps.csv  (columns: person_id, first_name, last_name,
                                     tiktok_username, party, constituency)
Output: data/raw/mp_accounts/<username>.csv  — per-account raw CSVs
        data/raw/mp_accounts/progress.json   — resume state + daily quota
        data/combined/mp_accounts_combined.csv — merged (after --merge)

Usage
-----
    python collect_mp_accounts.py           # collect all MP accounts
    python collect_mp_accounts.py --status  # check progress
    python collect_mp_accounts.py --merge   # merge into mp_accounts_combined.csv
"""

import argparse
import json
import os
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
USER_URL  = "https://open.tiktokapis.com/v2/research/user/info/"

VIDEO_FIELDS = ",".join([
    "id", "create_time", "username", "video_description",
    "music_id", "like_count", "comment_count", "share_count",
    "view_count", "favorites_count", "effect_ids", "hashtag_names",
    "video_duration", "voice_to_text", "video_label",
    "video_mention_list", "region_code",
])

USER_FIELDS = "follower_count,following_count,video_count,is_verified"

START_DATE = "2024-05-22"
END_DATE   = "2024-07-18"
CHUNK_DAYS = 28

MAX_PER_REQUEST  = 100
REQUESTS_PER_DAY = 1000
SLEEP_BETWEEN    = 2.0

ACCOUNTS_FILE = os.path.join(ROOT_DIR, "data", "supplementary", "verified_mps.csv")
OUTPUT_DIR    = os.path.join(ROOT_DIR, "data", "raw", "mp_accounts")
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
# User info
# --------------------------------------------------------------------------- #

def fetch_user_info(token, username):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        resp = requests.post(USER_URL, params={"fields": USER_FIELDS},
                             headers=headers, json={"username": username}, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"    user_info network error ({e}) — continuing without user stats")
        return {}
    if resp.status_code != 200:
        return {}
    data = resp.json().get("data", {})
    return {
        "follower_count":  data.get("follower_count"),
        "following_count": data.get("following_count"),
        "video_count":     data.get("video_count"),
        "is_verified":     data.get("is_verified"),
    }


# --------------------------------------------------------------------------- #
# Video collection
# --------------------------------------------------------------------------- #

def fetch_videos_chunk(token, username, start, end, progress):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    all_videos = []
    cursor     = 0
    search_id  = ""

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
                break

        if resp is None or resp.status_code in (429, 500):
            break

        resp.raise_for_status()
        data = resp.json().get("data", {})

        videos = data.get("videos", [])
        all_videos.extend(videos)

        search_id = data.get("search_id", search_id)
        has_more  = data.get("has_more", False)
        cursor    = data.get("cursor", 0)

        time.sleep(SLEEP_BETWEEN)

        if not has_more or not videos:
            break

    return all_videos


def fetch_videos(token, username, progress):
    start = date.fromisoformat(START_DATE)
    end   = date.fromisoformat(END_DATE)
    all_videos = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        videos = fetch_videos_chunk(token, username, cur, chunk_end, progress)
        all_videos.extend(videos)
        cur = chunk_end + timedelta(days=1)
    return all_videos


def save_account(videos, username, mp_meta, user_info, scrape_ts):
    if not videos:
        return 0
    df = pd.DataFrame(videos)
    if "music_id" in df.columns:
        df["music_id"] = [str(v["music_id"]) if v.get("music_id") is not None else "" for v in videos]
    df["_party"]        = mp_meta["party"]
    df["_person_id"]    = mp_meta["person_id"]
    df["_first_name"]   = mp_meta["first_name"]
    df["_last_name"]    = mp_meta["last_name"]
    df["_constituency"] = mp_meta["constituency"]
    df["scrape_timestamp"] = scrape_ts
    for k, v in user_info.items():
        df[k] = v

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
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(OUTPUT_DIR, f), dtype={"id": str, "music_id": str}))
    if not frames:
        print("No CSVs found to merge.")
        return
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["id"])
        .sort_values("create_time")
        .reset_index(drop=True)
    )
    out_path = os.path.join(ROOT_DIR, "data", "combined", "mp_accounts_combined.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Merged {len(combined)} unique videos → {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def load_accounts():
    df = pd.read_csv(ACCOUNTS_FILE)
    df.columns = [c.strip() for c in df.columns]
    return df.dropna(subset=["tiktok_username"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--merge",  action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_all()
        return

    accounts = load_accounts()
    if accounts.empty:
        print("No accounts in verified_mps.csv — check the file and retry.")
        return

    progress = reset_daily_count(load_progress())

    if args.status:
        total = len(accounts)
        done  = len(progress["completed"])
        print(f"Progress: {done}/{total} MP accounts collected")
        print(f"Requests today: {progress['request_count_today']}/{REQUESTS_PER_DAY}\n")
        for _, row in accounts.iterrows():
            status = "✓" if row["tiktok_username"] in progress["completed"] else "○"
            print(f"  {status} @{row['tiktok_username']} — {row['first_name']} {row['last_name']} ({row['party']}, {row['constituency']})")
        return

    if not CLIENT_KEY or not CLIENT_SECRET:
        print("Error: TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET must be set in ../.env")
        return

    print(f"Collecting {len(accounts)} MP accounts ({START_DATE} → {END_DATE})")
    print(f"Requests used today: {progress['request_count_today']}/{REQUESTS_PER_DAY}\n")

    token            = get_access_token()
    token_fetched_at = time.time()
    scrape_ts        = datetime.now(timezone.utc).isoformat()

    for _, row in accounts.iterrows():
        username = row["tiktok_username"]
        mp_meta  = {
            "party":        row["party"],
            "person_id":    row["person_id"],
            "first_name":   row["first_name"],
            "last_name":    row["last_name"],
            "constituency": row["constituency"],
        }

        if username in progress["completed"]:
            print(f"  ✓ @{username} (already done)")
            continue

        if progress["request_count_today"] >= REQUESTS_PER_DAY:
            print(f"\nDaily request limit reached. Resume tomorrow.")
            save_progress(progress)
            return

        if time.time() - token_fetched_at > 5400:
            token            = get_access_token()
            token_fetched_at = time.time()

        print(f"  Fetching @{username} ({mp_meta['first_name']} {mp_meta['last_name']}, {mp_meta['party']})...", end=" ", flush=True)
        requests_before = progress["request_count_today"]
        try:
            user_info = fetch_user_info(token, username)
            videos    = fetch_videos(token, username, progress)
            n_saved   = save_account(videos, username, mp_meta, user_info, scrape_ts)
            progress["completed"].append(username)
            save_progress(progress)
            n_req = progress["request_count_today"] - requests_before
            print(f"{n_saved} videos ({n_req} requests, {progress['request_count_today']} today)")
        except Exception as e:
            print(f"ERROR: {e}")
            save_progress(progress)

    print("\nCollection complete. Run with --merge to combine per-account files.")
    merge_all()


if __name__ == "__main__":
    main()
