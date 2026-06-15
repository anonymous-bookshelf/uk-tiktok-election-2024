"""
amplification_bootstrap.py

Bootstrapped within-account amplification analysis.

For each qualifying account, bootstraps the apolitical baseline to produce
a distribution of amplification ratios, accounting for baseline uncertainty.
Aggregates across accounts using the median bootstrap ratio per account.
"""

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = "/Users/sjdh/Desktop/projects/Dissertation/affordance_classifier"
CLASSIFIED_PATH = f"{ROOT_DIR}/data/classified/partisan_accounts_classified.csv"

PARTIES = ["Labour", "Conservative", "Reform UK", "Green", "SNP", "Lib Dem"]
N_BOOTSTRAP = 1000
MIN_APOLITICAL = 3
MIN_PARTISAN = 1
SEED = 42


def bootstrap_ratio(partisan_views, apolitical_views, n_boot=N_BOOTSTRAP, rng=None):
    """Return median bootstrapped ratio for one account."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    ratios = []
    for _ in range(n_boot):
        baseline = rng.choice(apolitical_views, size=len(apolitical_views), replace=True).mean()
        if baseline == 0:
            continue
        ratios.append(partisan_views.mean() / baseline)
    return np.median(ratios) if ratios else None


def run(cl, party, rng):
    ratios = []
    for username, grp in cl.groupby("username"):
        partisan  = grp[grp["predicted_party"] == party]["view_count"].values
        apolitical = grp[grp["predicted_party"] == "Apolitical"]["view_count"].values
        if len(partisan) < MIN_PARTISAN or len(apolitical) < MIN_APOLITICAL:
            continue
        r = bootstrap_ratio(partisan, apolitical, rng=rng)
        if r is not None:
            ratios.append(r)
    if len(ratios) < 5:
        print(f"  {party:<15} insufficient data (n={len(ratios)})")
        return
    median_r = np.median(ratios)
    mean_r   = np.mean(ratios)
    ci_lo, ci_hi = np.percentile(ratios, [2.5, 97.5])
    _, p = stats.wilcoxon([r - 1 for r in ratios])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {party:<15} n={len(ratios):>4}  median={median_r:.2f}x  "
          f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]  p={p:.4f} {sig}")


def main():
    cl = pd.read_csv(CLASSIFIED_PATH, dtype={"id": str})
    rng = np.random.default_rng(SEED)

    print(f"Bootstrap amplification analysis (n_boot={N_BOOTSTRAP})\n")
    print("=== PARTISAN vs APOLITICAL ===")
    for party in PARTIES:
        run(cl, party, rng)

    print("\n=== NON-PARTISAN vs APOLITICAL ===")
    ratios = []
    for username, grp in cl.groupby("username"):
        nonpartisan = grp[grp["predicted_party"] == "Non-partisan"]["view_count"].values
        apolitical  = grp[grp["predicted_party"] == "Apolitical"]["view_count"].values
        if len(nonpartisan) < 1 or len(apolitical) < 1:
            continue
        r = bootstrap_ratio(nonpartisan, apolitical, rng=rng)
        if r is not None:
            ratios.append(r)
    median_r = np.median(ratios)
    ci_lo, ci_hi = np.percentile(ratios, [2.5, 97.5])
    _, p = stats.wilcoxon([r - 1 for r in ratios])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"  {'Non-partisan':<15} n={len(ratios):>4}  median={median_r:.2f}x  "
          f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]  p={p:.4f} {sig}")


if __name__ == "__main__":
    main()
