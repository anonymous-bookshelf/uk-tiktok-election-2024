# The UK 2024 Election on TikTok

Code repository for the MPhil dissertation:

> **"The UK 2024 Election on TikTok: An Investigation of The Affordances and Amplification of Political Content"**
> Leverhulme Centre for the Future of Intelligence, University of Cambridge, 2025

---

## Overview

This project investigates whether UK political actors adopted TikTok's platform-native affordances during the 2024 general election campaign (22 May – 18 July 2024), and how the platform's recommendation algorithm treated political content.

Four research questions are addressed:

- **RQ1** — How far did UK political actors adopt TikTok affordances?
- **RQ2** — Is affordance adoption associated with greater engagement?
- **RQ3** — Does the algorithm amplify or suppress political content relative to apolitical content from the same accounts?
- **RQ4** — Does account size moderate the amplification of political content?

---

## Data

Three corpora were collected via the [TikTok Research API](https://developers.tiktok.com/products/research-api/):

| Corpus | Description | N videos |
|--------|-------------|----------|
| Account corpus | 9 official party accounts + 15 MP accounts | 1,141 |
| Hashtag corpus | 64 election-related hashtags | 108,070 |
| Amplification corpus | 1,056 politically-active accounts, with baseline apolitical content | 51,388 |

### Data availability

Raw collected data is not distributed in this repository due to size and TikTok's terms of service. Small lookup files are included in `data/supplementary/`:

- `accounts.csv` — party account list (username, party)
- `verified_mps.csv` — 40 verified MP accounts (person_id, name, tiktok_username, party, constituency)
- `amplification_user_info.csv` — account-level metadata for the amplification corpus

---

## Pipeline

```
scripts/collect_accounts.py
scripts/collect_hashtags.py         →  data/combined/
scripts/collect_mp_accounts.py

scripts/enrich.py                   →  data/enriched/

scripts/fetch_music_lookup.py       →  data/music_lookup/
  (Playwright-based scraper; queries TikTok sound pages for
   usage counts and original-sound flags)

scripts/compute_trendiness.py       →  data/trendiness/

scripts/llm_classifier.py           →  data/classified/
  (GPT-4o-mini; political/apolitical + party label for hashtag corpus)

scripts/collect_baseline_accounts.py \
scripts/collect_partisan_accounts.py  →  amplification corpus
scripts/fetch_amplification_user_info.py
scripts/amplification_bootstrap.py

analysis/build_dissertation_figures.py  →  figures/
```

---

## Trendiness Composite

Three binary conditions, each contributing 1 point (range 0–3):

| Condition | Variable | Operationalisation |
|-----------|----------|--------------------|
| Trending sound | `cond_sound` | Non-original sound with >10,000 uses on TikTok |
| Platform-native feature | `cond_platform_native` | `has_effects OR is_duet OR is_stitch` |
| Virality hashtag | `cond_hashtag` | Any of: `#fyp`, `#foryou`, `#foryoupage`, `#viral`, `#trending` |

```
trendiness_score = cond_sound + cond_platform_native + cond_hashtag
is_new_form      = trendiness_score >= 2
```

Sound usage counts are scraped per-video from TikTok sound pages using Playwright (`scripts/fetch_music_lookup.py`).

---

## LLM Classification

Videos in the hashtag corpus with a voice-to-text transcript (~29% of 108,070) were classified using GPT-4o-mini. Labels: `Apolitical`, `Non-partisan`, `Labour`, `Conservative`, `Lib Dem`, `Reform UK`, `Green`, `SNP`. GPT-4o-mini achieved 88.0% accuracy in model selection evaluation (237 manually annotated ground-truth labels; Gwet's AC1 = .910 inter-coder reliability).

---

## Amplification Analysis

For each account in the amplification corpus, a within-account ratio is computed:

```
ratio = mean(political video views) / mean(apolitical video views)
```

Bootstrapped (1,000 resamples) per account. Accounts with fewer than 3 apolitical videos are excluded. Wilcoxon signed-rank test on log-transformed ratios. Key finding: median ratio = 0.67× for partisan content (p < .001) — the algorithm suppresses political content, and suppresses it more for larger accounts ("equalisation by suppression").

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Copy `.env.example` to `.env` and add credentials:

```
TIKTOK_CLIENT_KEY=...
TIKTOK_CLIENT_SECRET=...
OPENAI_API_KEY=...
```

The Playwright sound scraper (`fetch_music_lookup.py`) requires being logged into TikTok in Chrome or Firefox.

---

## Figures

Pre-generated figures are in `figures/` (PDF for Overleaf, PNG for preview). To regenerate:

```bash
python analysis/build_dissertation_figures.py        # all figures
python analysis/build_dissertation_figures.py fig7   # single figure
```

Requires the processed data files in `data/trendiness/`, `data/classified/`, and the amplification master CSV. Path constants are at the top of `build_dissertation_figures.py`.
