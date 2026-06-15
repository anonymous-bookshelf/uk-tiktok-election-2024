"""
fetch_music_lookup.py

Automatically populates data/music_lookup.csv by scraping TikTok sound pages
using Playwright (full browser rendering) so that dynamically-loaded video
counts are captured.

For each unique music_id in data/enriched.csv it:
  1. Loads the representative video page (requests) to get the music title and
     original-sound flag from the embedded JSON.
  2. Constructs the sound page URL and loads it with Playwright to get the
     video count from the rendered page.

Usage
-----
    python fetch_music_lookup.py
    python fetch_music_lookup.py --browser chrome   # cookie source (default: chrome)
    python fetch_music_lookup.py --headless false   # show the browser window

Requirements
------------
    pip install pyktok playwright browser-cookie3 beautifulsoup4
    playwright install chromium
    Must be logged into TikTok in the specified browser.

Output
------
    data/music_lookup.csv — music_id, music_title, music_usage_count, is_original_sound
"""

import argparse
import asyncio
import json
import os
import re
import time

import browser_cookie3
import pandas as pd
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
ENRICHED   = os.path.join(DATA_DIR, "enriched", "accounts_enriched.csv")
LOOKUP_OUT = os.path.join(DATA_DIR, "music_lookup", "hashtags_music_lookup.csv")

SLEEP_REQUESTS  = 2.0
SLEEP_PLAYWRIGHT = 3.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_cookies(browser_name):
    try:
        raw = getattr(browser_cookie3, browser_name)(domain_name=".tiktok.com")
        return {c.name: c.value for c in raw}
    except Exception as e:
        print(f"Warning: could not load {browser_name} cookies: {e}")
        return {}


def parse_count(text):
    """Convert TikTok display counts like '2.3M', '45.1K', '123' to int."""
    text = text.strip().upper().replace(",", "")
    m = re.search(r"([\d.]+)([MKB]?)", text)
    if not m:
        return 0
    num = float(m.group(1))
    suffix = m.group(2)
    multipliers = {"M": 1_000_000, "K": 1_000, "B": 1_000_000_000}
    return int(num * multipliers.get(suffix, 1))


def make_slug(title):
    return re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-")


# --------------------------------------------------------------------------- #
# Step 1: get music title + original flag from video page (fast, requests)
# --------------------------------------------------------------------------- #

def get_music_info_from_video(video_url, cookies):
    try:
        resp = requests.get(video_url, headers=HEADERS, cookies=cookies, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        tag  = soup.find("script", attrs={"id": "__UNIVERSAL_DATA_FOR_REHYDRATION__"})
        if not tag:
            return None
        data   = json.loads(tag.string)
        detail = (
            data.get("__DEFAULT_SCOPE__", {})
                .get("webapp.video-detail", {})
                .get("itemInfo", {})
                .get("itemStruct", {})
        )
        music = detail.get("music")
        if not music:
            return None
        return {
            "music_title":       music.get("title", ""),
            "is_original_sound": bool(music.get("original", False)),
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Step 2: get video count from sound page (Playwright)
# --------------------------------------------------------------------------- #

async def get_video_count(page, music_id, title):
    slug = make_slug(title) if isinstance(title, str) and title else str(music_id)
    url  = f"https://www.tiktok.com/music/{slug}-{music_id}"
    try:
        await page.goto(url, wait_until="networkidle", timeout=30000)
        content = await page.content()
        # TikTok renders counts like "2.3M videos" or "45.1K videos"
        m = re.search(r"([\d.,]+[MKBmkb]?)\s*[Vv]ideo", content)
        if m:
            return parse_count(m.group(1))
    except Exception as e:
        print(f"    Playwright error for {music_id}: {e}")
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _is_int(val):
    try:
        int(str(val).strip())
        return True
    except (ValueError, TypeError):
        return False


def build_video_map(df):
    valid = df[df["music_id"].apply(_is_int) & df["id"].apply(_is_int)]
    return (
        valid.dropna(subset=["music_id", "id", "username"])
             .drop_duplicates(subset=["music_id"])
             .set_index("music_id")[["id", "username"]]
    )


async def run(input_path, browser_name, headless, output_path=None):
    out = output_path or LOOKUP_OUT
    df        = pd.read_csv(input_path, dtype={"music_id": str, "id": str})
    video_map = build_video_map(df)
    print(f"{len(video_map)} unique music_ids to look up")

    existing_complete = {}   # have a usage count
    existing_partial  = {}   # in file but count is null — will retry
    if os.path.exists(out):
        ex = pd.read_csv(out, dtype={"music_id": str}).drop_duplicates(subset=["music_id"])
        _tmp = out + ".tmp"
        ex.to_csv(_tmp, index=False)
        os.replace(_tmp, out)
        for mid, row in ex.set_index("music_id").iterrows():
            count = row.get("music_usage_count")
            is_orig = row.get("is_original_sound")
            if pd.isna(count):
                existing_partial[str(mid)] = row.to_dict()
            elif count == 1 and not is_orig:
                # Count of 1 for a non-original sound is likely a scraping artefact
                # (Playwright landed on the wrong page). Retry to get the real count.
                existing_partial[str(mid)] = row.to_dict()
            else:
                existing_complete[str(mid)] = row.to_dict()
        uncertain = sum(1 for r in existing_partial.values()
                        if not pd.isna(r.get("music_usage_count")))
        print(f"{len(existing_complete)} complete, {len(existing_partial)} to retry "
              f"({uncertain} uncertain count=1, rest missing)\n")

    cookies = load_cookies(browser_name)
    pw_cookies = [
        {"name": k, "value": v, "domain": ".tiktok.com", "path": "/"}
        for k, v in cookies.items()
    ]

    # Seed results with completes AND partials, so that mid-run interruption
    # never drops a partial from disk. Partials will be overwritten in-place
    # below as they are re-attempted (see results_by_id).
    results = (
        [{"music_id": k, **v} for k, v in existing_complete.items()]
        + [{"music_id": k, **v} for k, v in existing_partial.items()]
    )
    results_by_id = {r["music_id"]: r for r in results}
    todo    = [(str(mid), row) for mid, row in video_map.iterrows() if str(mid) not in existing_complete]
    print(f"{len(todo)} to fetch\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=(headless.lower() != "false"))
        context = await browser.new_context(user_agent=HEADERS["User-Agent"])
        await context.add_cookies(pw_cookies)
        page = await context.new_page()

        for music_id, row in todo:
            prev = results_by_id.get(music_id, {})
            mt = prev.get("music_title")
            prev_title       = mt if isinstance(mt, str) else ""
            prev_is_original = prev.get("is_original_sound")

            try:
                video_url = f"https://www.tiktok.com/@{row['username']}/video/{int(row['id'])}"
            except (ValueError, TypeError):
                print(f"  {music_id}... skipping (malformed id)")
                rec = {"music_id": music_id, "music_title": prev_title,
                       "music_usage_count": None, "is_original_sound": prev_is_original}
                results_by_id[music_id] = rec
                _tmp = out + ".tmp"
                pd.DataFrame(list(results_by_id.values())).to_csv(_tmp, index=False)
                os.replace(_tmp, out)
                continue
            print(f"  {music_id}...", end=" ", flush=True)

            # Step 1: title + original flag — preserve previously-known values on failure
            info = get_music_info_from_video(video_url, cookies)
            title       = info["music_title"]       if info else prev_title
            is_original = info["is_original_sound"] if info else prev_is_original
            time.sleep(SLEEP_REQUESTS)

            # Step 2: video count from sound page
            count = await get_video_count(page, music_id, title)
            await asyncio.sleep(SLEEP_PLAYWRIGHT)

            results_by_id[music_id] = {
                "music_id":          music_id,
                "music_title":       title,
                "music_usage_count": count,
                "is_original_sound": is_original,
            }

            _tmp = out + ".tmp"
            pd.DataFrame(list(results_by_id.values())).to_csv(_tmp, index=False)
            os.replace(_tmp, out)

            flag = " [ORIGINAL]" if is_original else ""
            cnt  = f"{count:,}" if count is not None else "?"
            print(f"'{title}' — {cnt} uses{flag}")

        await browser.close()

    final_rows = list(results_by_id.values())
    print(f"\nDone. {len(final_rows)} rows saved → {out}")
    valid = pd.DataFrame(final_rows)["music_usage_count"].dropna()
    if not valid.empty:
        print("\nUsage count distribution:")
        print(valid.describe().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default=ENRICHED,
                        help="Enriched CSV to read music_ids from (default: accounts_enriched.csv).")
    parser.add_argument("--output",   default=None,
                        help="Output CSV path (default: data/supplementary/music_lookup.csv).")
    parser.add_argument("--browser",  default="chrome")
    parser.add_argument("--headless", default="true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Not found: {args.input} — run enrich.py first.")
        return

    asyncio.run(run(args.input, args.browser, args.headless, output_path=args.output))


if __name__ == "__main__":
    main()
