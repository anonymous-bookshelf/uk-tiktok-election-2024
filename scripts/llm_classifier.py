"""
llm_classifier.py

Classifies TikTok video transcripts by UK political party using GPT-4o-mini.
Filters to videos with transcripts only. Saves incrementally so a crash can
be resumed.

Usage
-----
    python llm_classifier.py                                      # hashtag corpus (default)
    python llm_classifier.py --input data/combined/partisan_accounts_combined.csv \
                             --output data/classified/partisan_accounts_classified.csv

Install dependencies once:
    pip install openai python-dotenv pandas tqdm

API key must be set in .env (project root):
    OPENAI_API_KEY=...
"""

import argparse
import os
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

DATA_DIR = os.path.join(ROOT_DIR, "data")

DEFAULT_INPUT_PATHS = [
    os.path.join(DATA_DIR, "combined", "accounts_combined.csv"),
    os.path.join(DATA_DIR, "combined", "hashtags_combined.csv"),
]

DEFAULT_OUTPUT_PATH = os.path.join(DATA_DIR, "classified", "hashtags_classified.csv")

MODEL             = "gpt-4o-mini"
MAX_TRANSCRIPT_CHARS = 1800
SLEEP_BETWEEN_CALLS  = 0.5
MAX_RETRIES          = 3

LABEL_NORMALIZER = {
    "labour":                 "Labour",
    "conservative":           "Conservative",
    "conservatives":          "Conservative",
    "tory":                   "Conservative",
    "tories":                 "Conservative",
    "lib dem":                "Lib Dem",
    "lib dems":               "Lib Dem",
    "liberal democrats":      "Lib Dem",
    "liberal democrat":       "Lib Dem",
    "reform uk":              "Reform UK",
    "reform":                 "Reform UK",
    "reform uk party":        "Reform UK",
    "green":                  "Green",
    "green party":            "Green",
    "snp":                    "SNP",
    "scottish national party":"SNP",
    "non-partisan":           "Non-partisan",
    "nonpartisan":            "Non-partisan",
    "non partisan":           "Non-partisan",
    "none":                   "Non-partisan",
    "uncertain":              "Apolitical",
    "unclear":                "Apolitical",
    "n/a":                    "Apolitical",
    "no party":               "Non-partisan",
    "neutral":                "Non-partisan",
    "apolitical":             "Apolitical",
    "not political":          "Apolitical",
    "non-political":          "Apolitical",
    "nonpolitical":           "Apolitical",
    "no political content":   "Apolitical",
}

SYSTEM_PROMPT = """\
You are an expert classifier of UK political content from TikTok videos.

Your task is to assign exactly one label to a TikTok video transcript. \
There are three tiers of labels:

THE CRITICAL DISTINCTION — READ THIS FIRST:
There are two completely different kinds of uncertainty and they lead to opposite labels.

UNCERTAINTY ABOUT WHETHER THE CONTENT IS POLITICAL AT ALL → Apolitical. \
If you watch the transcript and genuinely cannot tell whether it has anything to do \
with politics, elections, parties, or public affairs — use Apolitical. \
When in doubt about whether it is even political, default to Apolitical.

UNCERTAINTY ABOUT WHICH PARTY (given you can see it IS political) → Non-partisan. \
If you can clearly see the content is about politics, elections, or parties, but you \
cannot pin it to a specific party's advocacy — use Non-partisan.

TIER 1 — APOLITICAL
Use "Apolitical" when the content has no clear political dimension, OR when you \
genuinely cannot tell whether it is political. Examples: lifestyle content, \
entertainment, cooking, sport, personal vlogs. If after reading the transcript you \
are still unsure whether it touches on politics at all, use Apolitical.

TIER 2 — POLITICAL BUT NOT PARTISAN
Use "Non-partisan" when the content is political in any way but does not explicitly \
advocate for a specific party. This includes:
- Neutral news reporting or journalism about what parties are doing or saying
- Satire, comedy, or parody about politicians or the election
- Criticising a party without explicitly endorsing an alternative
- Anti-Tory or anti-Sunak content that does not call for a Labour vote
- Generic get-out-the-vote messages with no party named
- TV debate clips, interview footage, or panel discussions
- Commentary or analysis discussing parties without advocating for one
- Tactical voting content that names multiple parties
- ANY case where you think a video might lean toward a party but you are not certain

TIER 3 — PARTISAN ADVOCACY
Use a party label only when the content explicitly and unambiguously advocates \
for that specific party. The video must be clearly calling for support for that \
party — not merely mentioning it, reporting on it, or criticising its opponents.

PARTIES:
- Labour: Centre-left, led by Keir Starmer in 2024. NHS, workers' rights, \
progressive taxation.
- Conservative: Centre-right, led by Rishi Sunak in 2024. Lower taxes, \
managed immigration, traditional institutions.
- Lib Dem: Liberal Democrats, led by Ed Davey. Civil liberties, proportional \
representation, closer EU ties.
- Reform UK: Right-wing populist, led by Nigel Farage. Anti-immigration, \
anti-establishment, Brexit-aligned.
- Green: Left-wing. Climate action, environmental protection, social justice.
- SNP: Scottish National Party. Scottish independence, centre-left.

RULES:
1. If there is no political content, or you genuinely cannot tell whether it is \
political → Apolitical
2. If the content is clearly political but you cannot identify which party it \
advocates for → Non-partisan (NOT Apolitical)
3. If political but no explicit party endorsement → Non-partisan
4. Only assign a party label if the video unambiguously calls for support \
for that party. If in any doubt → Non-partisan.
5. Criticising one party does NOT imply support for another — unless the creator \
explicitly names an alternative party they are advocating for, in which case \
label it as that party.
6. Reporting on a party is journalism, not advocacy.
7. TACTICAL VOTING: If the creator argues that viewers should vote for a specific \
party for strategic or tactical reasons — even if they personally prefer another \
— label it as that party. Recommending a vote is advocacy regardless of the \
reason given.
8. CANVASSING AND RALLIES: If the creator is actively door-knocking, \
phone-banking, attending a rally, or otherwise campaigning on behalf of a named \
party, label it as that party.
9. DEFENDING A PARTY'S RECORD: If a politician, candidate, or party supporter is \
defending their own party's record or policies against criticism — not a \
journalist or neutral commentator — label it as that party.

Respond with EXACTLY ONE label from this list — nothing else, no punctuation, \
no explanation:
Apolitical
Non-partisan
Labour
Conservative
Lib Dem
Reform UK
Green
SNP"""


def load_data(input_paths) -> pd.DataFrame:
    frames = []
    for path in input_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found — skipping.")
            continue
        df = pd.read_csv(path, dtype={"id": str})
        df["_source"] = os.path.basename(path).replace("_combined.csv", "")
        frames.append(df)
    if not frames:
        raise SystemExit("No input CSVs found.")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["id"])

    has_transcript = (
        combined["voice_to_text"].notna() &
        (combined["voice_to_text"].str.strip() != "")
    )
    combined = combined[has_transcript].copy()
    combined["voice_to_text"] = combined["voice_to_text"].str.strip().str[:MAX_TRANSCRIPT_CHARS]

    print(f"Loaded {len(combined)} videos with transcripts")
    return combined.reset_index(drop=True)


def parse_label(raw: str) -> str:
    return LABEL_NORMALIZER.get(raw.strip().lower().rstrip("."), "Apolitical")


def classify(client, transcript: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=16,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Transcript:\n{transcript}"},
                ],
            )
            return parse_label(response.choices[0].message.content)
        except Exception as e:
            err = str(e).lower()
            is_rate = "rate" in err or "429" in err
            if attempt >= MAX_RETRIES - 1:
                print(f"\n  Giving up after {MAX_RETRIES} attempts — last error: {e}")
                return None
            if is_rate:
                wait = 60 * (attempt + 1)
                print(f"\n  Rate limited — sleeping {wait}s")
                time.sleep(wait)
            else:
                time.sleep(5 * (attempt + 1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  nargs="+", default=None,
                        help="Input CSV path(s). Defaults to hashtag + accounts combined CSVs.")
    parser.add_argument("--output", default=None,
                        help="Output CSV path. Defaults to data/classified/hashtags_classified.csv.")
    args = parser.parse_args()

    input_paths = args.input if args.input else DEFAULT_INPUT_PATHS
    output_path = args.output if args.output else DEFAULT_OUTPUT_PATH

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("Set OPENAI_API_KEY in .env before running.")

    try:
        import openai as openai_lib
        client = openai_lib.OpenAI(api_key=openai_key)
    except ImportError:
        raise SystemExit("Run: pip install openai")

    df = load_data(input_paths)

    # Resume: skip already-classified rows
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    done_ids = set()
    if os.path.exists(output_path):
        done = pd.read_csv(output_path, dtype={"id": str})
        done_ids = set(done["id"])
        print(f"Resuming — {len(done_ids)} already classified\n")

    todo = df[~df["id"].isin(done_ids)]
    print(f"{len(todo)} to classify\n")

    first_write = not os.path.exists(output_path)
    n_failed = 0
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=MODEL, dynamic_ncols=True):
        label = classify(client, row["voice_to_text"])
        if label is None:
            n_failed += 1
            continue
        out = row.to_frame().T.copy()
        out["predicted_party"] = label
        out.to_csv(output_path, mode="a", header=first_write, index=False)
        first_write = False
        time.sleep(SLEEP_BETWEEN_CALLS)

    if n_failed:
        print(f"\n  WARNING: {n_failed} videos skipped due to repeated API errors. Re-run to retry.")

    print(f"\nDone. Results → {output_path}")

    results = pd.read_csv(output_path, dtype={"id": str})
    print(f"\nPrediction breakdown ({len(results)} videos):")
    for label, n in results["predicted_party"].value_counts().items():
        print(f"  {label:<14} {n:>5}  ({100*n/len(results):.1f}%)")


if __name__ == "__main__":
    main()
