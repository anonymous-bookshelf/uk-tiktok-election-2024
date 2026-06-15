"""
fetch_amplification_user_info.py

Fetches follower_count, following_count, video_count, is_verified
for all unique accounts in the amplification master CSV.

Saves incrementally — safe to interrupt and resume.

Usage
-----
    python scripts/fetch_amplification_user_info.py

Output
------
    data/supplementary/amplification_user_info.csv
"""

import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

CLIENT_KEY    = os.getenv("TIKTOK_CLIENT_KEY")
CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET")

TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
USER_URL  = "https://open.tiktokapis.com/v2/research/user/info/"
USER_FIELDS = "follower_count,following_count,video_count,is_verified"

MASTER_CSV = "/Users/sjdh/Desktop/projects/Dissertation/organised_dataset/Amplification/amplification_master.csv"
OUT_CSV    = os.path.join(ROOT_DIR, "data", "supplementary", "amplification_user_info.csv")

SLEEP_BETWEEN = 1.5   # seconds between requests


def get_access_token():
    resp = requests.post(TOKEN_URL, data={
        "client_key":    CLIENT_KEY,
        "client_secret": CLIENT_SECRET,
        "grant_type":    "client_credentials",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_user_info(token, username):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        resp = requests.post(
            USER_URL,
            params={"fields": USER_FIELDS},
            headers=headers,
            json={"username": username},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        print(f"  network error for {username}: {e}")
        return None
    if resp.status_code == 401:
        return "expired"
    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code} for {username}")
        return None
    data = resp.json().get("data", {})
    return {
        "username":        username,
        "follower_count":  data.get("follower_count"),
        "following_count": data.get("following_count"),
        "video_count":     data.get("video_count"),
        "is_verified":     data.get("is_verified"),
    }


def main():
    # ── load all usernames ────────────────────────────────────────────────────
    df = pd.read_csv(MASTER_CSV, dtype={"id": str})
    all_usernames = sorted(df["username"].dropna().unique().tolist())
    print(f"Total unique accounts: {len(all_usernames)}")

    # ── load already-fetched results ──────────────────────────────────────────
    if os.path.exists(OUT_CSV):
        done_df = pd.read_csv(OUT_CSV)
        done = set(done_df["username"].tolist())
        rows = done_df.to_dict("records")
    else:
        done = set()
        rows = []

    remaining = [u for u in all_usernames if u not in done]
    print(f"Already done: {len(done)}  |  Remaining: {len(remaining)}")

    if not remaining:
        print("All accounts fetched.")
        return

    # ── fetch ─────────────────────────────────────────────────────────────────
    token = get_access_token()
    print("Token obtained. Starting fetch...\n")

    for i, username in enumerate(remaining, 1):
        result = fetch_user_info(token, username)

        if result == "expired":
            print("  Token expired — refreshing...")
            token = get_access_token()
            result = fetch_user_info(token, username)

        if result and result != "expired":
            rows.append(result)
        else:
            rows.append({"username": username, "follower_count": None,
                         "following_count": None, "video_count": None,
                         "is_verified": None})

        if i % 50 == 0 or i == len(remaining):
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(f"  [{i}/{len(remaining)}] saved — last: {username}")

        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. Output: {OUT_CSV}")
    final = pd.read_csv(OUT_CSV)
    n_ok = final["follower_count"].notna().sum()
    print(f"Follower count retrieved for {n_ok}/{len(final)} accounts.")


if __name__ == "__main__":
    main()
