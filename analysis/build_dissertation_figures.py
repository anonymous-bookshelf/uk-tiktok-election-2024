"""
build_dissertation_figures.py

Recreates fig1..fig11 (originally in cowork_dissertation/analysis/) with
LaTeX-consistent (Computer Modern) styling for inclusion in the Overleaf
draft. Same chart designs/content as plot_all_figures.py, recomputed
directly from the organised_dataset master CSVs, with two fixes:
  - drop the font.serif=["cmr10"] override (it lacks the e-acute glyph,
    causing "Sinn F[]in" tofu in the originals) -- default serif renders
    "Sinn Féin" correctly while mathtext.fontset="cm" still gives CM math.
  - "Sinn Féin" (proper UTF-8) didn't match the PARTY_COLOURS keys
    ("Sinn Fein" / "Sinn F\\'ein") and fell back to grey; added the
    correct UTF-8 key so it gets its intended teal (#326760).

Run:
  cd cowork_dissertation/results_analysis
  python3 build_dissertation_figures.py [fig1 fig2 ...]   # default: all
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# STYLE (Computer Modern via mathtext -- no LaTeX install required)
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "text.usetex":                  False,
    "font.family":                  "serif",
    "mathtext.fontset":             "cm",
    "axes.formatter.use_mathtext":  True,
    "font.size":                    10,
    "axes.titlesize":               10,
    "axes.labelsize":               10,
    "xtick.labelsize":              9,
    "ytick.labelsize":              9,
    "legend.fontsize":              9,
    "figure.facecolor":             "white",
    "axes.facecolor":               "white",
    "savefig.bbox":                 "tight",
    "savefig.pad_inches":           0.03,
})

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
ROOT         = "/Users/sjdh/Desktop/projects/Dissertation/organised_dataset"
PARTY_PATH   = f"{ROOT}/Parties/party_accounts_master.csv"
MP_PATH      = f"{ROOT}/MPs/mp_accounts_master.csv"
HASHTAG_PATH = f"{ROOT}/Hashtags/hashtags_master.csv"
AMP_PATH     = f"{ROOT}/Amplification/amplification_master.csv"
OUT = "/Users/sjdh/Desktop/projects/Dissertation/cowork_dissertation/results_analysis/figures_latex"

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
MUSIC_THRESHOLD      = 10_000
N_BOOTSTRAP          = 300
MIN_APOLITICAL       = 3
MIN_APOLITICAL_SPLIT = 4
SEED                 = 42

PARTY_COLOURS = {
    "Labour":             "#E4003B",
    "Conservative":       "#0087DC",
    "Reform UK":          "#12B6CF",
    "Liberal Democrats":  "#FAA61A",
    "Lib Dem":            "#FAA61A",
    "SNP":                "#F5A800",
    "Green":              "#02A95B",
    "Plaid Cymru":        "#3F8428",
    "Sinn Fein":          "#326760",
    "Sinn F\\'ein":       "#326760",
    "Sinn Féin":          "#326760",   # fix: master CSV uses proper UTF-8
    "DUP":                "#D46A4C",
    "Non-partisan":       "#888888",
}
DEFAULT_COLOUR   = "#aaaaaa"
PARTISAN_LABELS  = {"Labour", "Conservative", "Reform UK", "Green", "SNP", "Lib Dem"}
POLITICAL_LABELS = PARTISAN_LABELS | {"Non-partisan"}
SHARED_COLS = ["view_count", "like_count", "comment_count", "share_count",
               "favorites_count", "video_duration", "music_title",
               "music_usage_count", "has_platform_native_feature",
               "has_virality_hashtag"]


def pcolour(name):
    return PARTY_COLOURS.get(name, DEFAULT_COLOUR)


def clean(s):
    return str(s).replace("Sinn F\\'ein", "Sinn Féin").replace("Sinn Fein", "Sinn Féin")


DISPLAY_NAME = {"Liberal Democrats": "Lib Dem"}


def display(s):
    s = clean(s)
    return DISPLAY_NAME.get(s, s)


# ══════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════

def compute_trendiness(df):
    df = df.copy()
    df["music_usage_count"] = pd.to_numeric(df["music_usage_count"], errors="coerce")
    df["is_original_sound"]    = df["music_title"].isna() | (df["music_usage_count"] == 1)
    df["cond_sound"]           = (df["music_usage_count"] > MUSIC_THRESHOLD) & (~df["is_original_sound"])
    df["cond_platform_native"] = df["has_platform_native_feature"].fillna(False).astype(bool)
    df["cond_hashtag"]         = df["has_virality_hashtag"].fillna(False).astype(bool)
    df["any_affordance"]       = df["cond_sound"] | df["cond_platform_native"] | df["cond_hashtag"]
    df["trendiness_score"]     = (df["cond_sound"].astype(int)
                                  + df["cond_platform_native"].astype(int)
                                  + df["cond_hashtag"].astype(int))
    return df


def compute_pca_engagement(df):
    v = df["view_count"].replace(0, np.nan)
    df = df.copy()
    df["_lr"] = df["like_count"]      / v
    df["_cr"] = df["comment_count"]   / v
    df["_sr"] = df["share_count"]     / v
    df["_fr"] = df["favorites_count"] / v
    feats = ["_lr", "_cr", "_sr", "_fr"]
    X = df[feats].dropna()
    X_std = (X - X.mean()) / X.std()
    _, _, Vt = np.linalg.svd(X_std.values, full_matrices=False)
    df.loc[X.index, "pc1"] = X_std.values @ Vt[0]
    for c in feats:
        df.drop(columns=c, inplace=True, errors="ignore")
    return df


def load_accounts(exclude_workers=True):
    pa = pd.read_csv(PARTY_PATH)
    if exclude_workers:
        pa = pa[pa["_party"] != "Workers Party of Britain"]
    pa = pa.copy()
    pa["party"] = pa["_party"]
    mp = pd.read_csv(MP_PATH)
    if exclude_workers:
        mp = mp[mp["_party"] != "Independent"]
    mp = mp.copy()
    mp["party"] = mp["_party"]
    df = pd.concat([pa[SHARED_COLS + ["party"]], mp[SHARED_COLS + ["party"]]], ignore_index=True)
    df = compute_trendiness(df)
    df = compute_pca_engagement(df)
    print(f"  Accounts: {len(df):,}")
    return df


def load_hashtags(drop_apolitical=True):
    ht = pd.read_csv(HASHTAG_PATH)
    if drop_apolitical:
        ht = ht[ht["predicted_party"] != "Apolitical"].copy()
    else:
        ht = ht.copy()
    ht["party"] = ht["predicted_party"]
    ht = compute_trendiness(ht)
    ht = compute_pca_engagement(ht)
    print(f"  Hashtags: {len(ht):,} (drop_apolitical={drop_apolitical})")
    return ht


def load_amplification():
    df = pd.read_csv(AMP_PATH, dtype={"id": str})
    df = df[df["region_code"].str.upper() == "GB"].copy()
    df = compute_trendiness(df)
    print(f"  Amplification: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════
# AMPLIFICATION HELPERS
# ══════════════════════════════════════════════════════════════

def bootstrap_ratio(pol, apol, n_boot=N_BOOTSTRAP, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    ratios = []
    for _ in range(n_boot):
        base = rng.choice(apol, size=len(apol), replace=True).mean()
        if base > 0:
            ratios.append(pol.mean() / base)
    return np.median(ratios) if ratios else None


def collect_ratios(df, pol_label, min_pol=1):
    if isinstance(pol_label, str):
        pol_label = {pol_label}
    rng = np.random.default_rng(SEED)
    records = []
    for username, grp in df.groupby("username"):
        apol = grp[grp["predicted_party"] == "Apolitical"]["view_count"].values
        if len(apol) < MIN_APOLITICAL:
            continue
        pol = grp[grp["predicted_party"].isin(pol_label)]["view_count"].values
        if len(pol) < min_pol:
            continue
        r = bootstrap_ratio(pol, apol, rng=rng)
        if r is not None:
            records.append({"username": username, "ratio": r, "baseline_mean": apol.mean()})
    return pd.DataFrame(records)


def wilcoxon_log_p(r):
    r = np.asarray(r)
    if len(r) < 5:
        return None
    _, p = stats.wilcoxon(np.log10(np.clip(r, 1e-9, None)))
    return p


def sig_plain(p):
    if p is None:  return "n.s."
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "n.s."


def bootstrap_median_ci(r, n_boot=500, seed=SEED):
    rng  = np.random.default_rng(seed)
    meds = [np.median(rng.choice(r, size=len(r), replace=True)) for _ in range(n_boot)]
    return tuple(np.percentile(meds, [2.5, 97.5]))


# ══════════════════════════════════════════════════════════════
# SHARED BAR STYLE
# ══════════════════════════════════════════════════════════════

def style_vbar(ax, ylabel=None, ylim=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="x", length=0, colors="#333333")
    ax.tick_params(axis="y", colors="#555555")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color="#333333")
    if ylim:
        ax.set_ylim(ylim)


# ══════════════════════════════════════════════════════════════
# SHARED RAINCLOUD
# ══════════════════════════════════════════════════════════════

def draw_raincloud(ax, fig, rows,
                   x_min=0.04, x_max=25,
                   spacing=1.0, kde_height=0.35,
                   strip_half=0.12, box_half=0.07):
    log_min  = np.log10(x_min)
    log_max  = np.log10(x_max)
    n_rows   = len(rows)
    rng_plot = np.random.default_rng(SEED)

    for i, row in enumerate(rows):
        y_centre = (n_rows - 1 - i) * spacing
        colour   = row["colour"]
        ratios   = np.asarray(row["ratios"])
        n        = len(ratios)
        if n == 0:
            continue
        log_ratios = np.log10(np.clip(ratios, x_min, x_max))
        median_r   = np.median(ratios)

        kde_x = np.linspace(log_min, log_max, 400)
        try:
            kde   = gaussian_kde(log_ratios, bw_method=0.35)
            kde_y = kde(kde_x)
            kde_y = kde_y / kde_y.max() * kde_height
            ax.fill_between(kde_x, y_centre, y_centre + kde_y,
                            color=colour, alpha=0.55, linewidth=0)
            ax.plot(kde_x, y_centre + kde_y, color=colour, linewidth=0.8, alpha=0.8)
        except Exception:
            pass

        jitter = rng_plot.uniform(-strip_half, strip_half, size=n)
        ax.scatter(log_ratios, y_centre - 0.05 + jitter,
                   color=colour, alpha=0.35, s=12, linewidths=0, zorder=2)

        y_box     = y_centre - 0.28
        q1, q3    = np.percentile(ratios, [25, 75])
        lo5, hi95 = np.percentile(ratios, [5, 95])
        log_q1    = np.log10(max(q1,      x_min))
        log_q3    = np.log10(max(q3,      x_min))
        log_med   = np.log10(max(median_r, x_min))
        log_lo    = np.log10(max(lo5,     x_min))
        log_hi    = np.log10(max(hi95,    x_min))

        ax.plot([log_lo, log_hi], [y_box, y_box], color=colour, linewidth=1.2, zorder=3)
        rect = mpatches.FancyBboxPatch(
            (log_q1, y_box - box_half), log_q3 - log_q1, box_half * 2,
            boxstyle="square,pad=0", linewidth=0, facecolor=colour, alpha=0.35, zorder=4)
        ax.add_patch(rect)
        ax.plot([log_med, log_med], [y_box - box_half, y_box + box_half],
                color=colour, linewidth=2.0, zorder=5)

        ci = row.get("ci")
        if ci is not None:
            ax.plot([np.log10(max(ci[0], x_min)), np.log10(max(ci[1], x_min))],
                    [y_box, y_box],
                    color=colour, linewidth=3.5, alpha=0.5, zorder=3, solid_capstyle="butt")

        sig   = sig_plain(row.get("p"))
        label = clean(row["label"])
        ax.text(log_min - 0.05, y_centre - 0.05,
                f"{label}\n(n={n}, med={median_r:.3f}" + r"$\times$" + f", {sig})",
                va="center", ha="right", fontsize=9.5,
                color="#333333", linespacing=1.4, clip_on=False)

    ax.axvline(log_min, color="#333333", linewidth=1.2, zorder=1)
    ax.axvline(0,       color="#888888", linestyle="--", linewidth=1.1, zorder=1)

    tick_vals = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    ax.set_xticks([np.log10(v) for v in tick_vals])
    ax.set_xticklabels([f"${v}\\times$" for v in tick_vals], fontsize=9.5)
    ax.set_xlim(log_min, log_max)
    ax.set_ylim(-0.7, (n_rows - 1) * spacing + kde_height + 0.15)
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="x", colors="#555555")

    fig.canvas.draw()
    trans = ax.transData + fig.transFigure.inverted()
    for i, row in enumerate(rows):
        y_centre = (n_rows - 1 - i) * spacing
        x_fig, y_fig = trans.transform((log_min, y_centre - 0.05))
        tick_w = 0.006
        fig.add_artist(plt.Line2D([x_fig - tick_w, x_fig], [y_fig, y_fig],
                                   transform=fig.transFigure,
                                   color="#333333", linewidth=1.5, clip_on=False))


def savefig(fname):
    plt.savefig(f"{OUT}/{fname}.pdf")
    plt.savefig(f"{OUT}/{fname}.png", dpi=200, facecolor="white")
    plt.close()
    print(f"  {fname} done")


# ══════════════════════════════════════════════════════════════
# FIG 1 -- affordance adoption by party (account corpus)
# ══════════════════════════════════════════════════════════════

def fig1():
    accounts = load_accounts()
    agg = (accounts.groupby("party")["any_affordance"]
           .agg(pct=lambda x: x.mean() * 100, n="count")
           .reset_index().sort_values("pct", ascending=False))
    print(agg.to_string(index=False))

    parties = agg["party"].tolist()
    pcts    = agg["pct"].values
    ns      = agg["n"].values
    colours = [pcolour(p) for p in parties]

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(parties))
    ax.bar(x, pcts, width=0.65, color=colours, alpha=0.85, zorder=3)
    for xi, pct in zip(x, pcts):
        ax.text(xi, pct + 0.4, f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color="#333333")

    labels = [f"{display(p)}\n(n={n})" for p, n in zip(parties, ns)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title("Affordance adoption by party — party + MP accounts",
                 fontsize=10, color="#222222", pad=8, loc="left", fontweight="bold")
    style_vbar(ax, ylabel=r"Videos using $\geq$1 affordance (%)",
               ylim=(0, pcts.max() * 1.25))
    plt.tight_layout()
    savefig("fig1")


# ══════════════════════════════════════════════════════════════
# FIG 2 -- affordance condition usage by party (account corpus)
# ══════════════════════════════════════════════════════════════

def fig2():
    accounts = load_accounts()
    party_order = (accounts.groupby("party")["any_affordance"]
                   .mean().sort_values(ascending=False).index.tolist())
    conditions = [
        ("cond_sound",           "Trending sound"),
        ("cond_hashtag",         "Virality hashtag"),
        ("cond_platform_native", "Effect/duet/stitch"),
    ]
    n_p = len(party_order)
    n_c = len(conditions)
    bw  = 0.8 / n_p
    x_c = np.arange(n_c)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for pi, party in enumerate(party_order):
        colour  = pcolour(party)
        pcts    = [accounts[accounts["party"] == party][col].mean() * 100
                   for col, _ in conditions]
        offsets = x_c + (pi - n_p / 2 + 0.5) * bw
        ax.bar(offsets, pcts, width=bw * 0.92, color=colour, alpha=0.85,
               label=clean(party), zorder=3)
        print(f"  {clean(party):20s} " + "  ".join(f"{lbl}={p:5.2f}%" for (_, lbl), p in zip(conditions, pcts)))

    ax.set_xticks(x_c)
    ax.set_xticklabels([label for _, label in conditions], fontsize=10)
    ax.set_title("Affordance condition usage by party — party + MP accounts",
                 fontsize=10, color="#222222", pad=8, loc="left", fontweight="bold")
    style_vbar(ax, ylabel="% of videos")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7.5, frameon=False,
              ncol=3, loc="upper right", columnspacing=0.8, handlelength=1.0)
    plt.tight_layout()
    savefig("fig2")


# ══════════════════════════════════════════════════════════════
# FIG 3 -- affordance adoption by party (hashtag corpus, apolitical excl.)
# ══════════════════════════════════════════════════════════════

def fig3():
    ht = load_hashtags(drop_apolitical=True)
    agg = (ht.groupby("party")["any_affordance"]
           .agg(pct=lambda x: x.mean() * 100, n="count")
           .reset_index().sort_values("pct", ascending=False))
    print(agg.to_string(index=False))

    parties = agg["party"].tolist()
    pcts    = agg["pct"].values
    ns      = agg["n"].values
    colours = [pcolour(p) for p in parties]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(parties))
    ax.bar(x, pcts, width=0.65, color=colours, alpha=0.85, zorder=3)
    for xi, pct in zip(x, pcts):
        ax.text(xi, pct + 0.4, f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color="#333333")

    labels = [f"{display(p)}\n(n={n:,})" for p, n in zip(parties, ns)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Affordance adoption by party — hashtag corpus (apolitical excluded)",
                 fontsize=10, color="#222222", pad=8, loc="left", fontweight="bold")
    style_vbar(ax, ylabel=r"Videos using $\geq$1 affordance (%)",
               ylim=(0, pcts.max() * 1.25))
    plt.tight_layout()
    savefig("fig3")


# ══════════════════════════════════════════════════════════════
# FIG 4 -- affordance condition usage by party (hashtag corpus)
# ══════════════════════════════════════════════════════════════

def fig4():
    ht = load_hashtags(drop_apolitical=True)
    party_order = (ht.groupby("party")["any_affordance"]
                   .mean().sort_values(ascending=False).index.tolist())
    conditions = [
        ("cond_sound",           "Trending sound"),
        ("cond_hashtag",         "Virality hashtag"),
        ("cond_platform_native", "Effect/duet/stitch"),
    ]
    n_p = len(party_order)
    n_c = len(conditions)
    bw  = 0.8 / n_p
    x_c = np.arange(n_c)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for pi, party in enumerate(party_order):
        colour  = pcolour(party)
        pcts    = [ht[ht["party"] == party][col].mean() * 100 for col, _ in conditions]
        offsets = x_c + (pi - n_p / 2 + 0.5) * bw
        ax.bar(offsets, pcts, width=bw * 0.92, color=colour, alpha=0.85,
               label=display(party), zorder=3)
        print(f"  {display(party):14s} " + "  ".join(f"{lbl}={p:5.2f}%" for (_, lbl), p in zip(conditions, pcts)))

    ax.set_xticks(x_c)
    ax.set_xticklabels([label for _, label in conditions], fontsize=10)
    ax.set_title("Affordance condition usage by party — hashtag corpus (apolitical excluded)",
                 fontsize=10, color="#222222", pad=8, loc="left", fontweight="bold")
    style_vbar(ax, ylabel="% of videos")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, frameon=False,
              ncol=2, loc="upper right", columnspacing=0.8, handlelength=1.0)
    plt.tight_layout()
    savefig("fig4")


# ══════════════════════════════════════════════════════════════
# FIG 5 -- official accounts vs hashtag corpus (partisan parties)
# ══════════════════════════════════════════════════════════════

def fig5():
    accounts = load_accounts()
    ht       = load_hashtags(drop_apolitical=True)

    shared = ["Labour", "Conservative", "Reform UK", "SNP", "Green", "Lib Dem"]

    acc = accounts.copy()
    acc["party_std"] = acc["party"].apply(display)
    acc_stats = (acc[acc["party_std"].isin(shared)]
                 .groupby("party_std")["any_affordance"].mean() * 100)
    ht_stats  = (ht[ht["party"].isin(shared)]
                 .groupby("party")["any_affordance"].mean() * 100)

    order   = ht_stats.reindex(shared).sort_values(ascending=False).index.tolist()
    colours = [pcolour(p) for p in order]
    acc_pcts = [acc_stats.get(p, 0) for p in order]
    ht_pcts  = [ht_stats.get(p,  0) for p in order]
    for p, ap, hp in zip(order, acc_pcts, ht_pcts):
        print(f"  {p:14s} account={ap:5.1f}%  hashtag={hp:5.1f}%")

    x  = np.arange(len(order))
    bw = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, (ap, hp, colour) in enumerate(zip(acc_pcts, ht_pcts, colours)):
        ax.bar(x[i] - bw / 2, ap, width=bw * 0.9, color=colour, alpha=0.85, zorder=3)
        ax.bar(x[i] + bw / 2, hp, width=bw * 0.9, color=colour, alpha=0.38,
               hatch="///", edgecolor=colour, linewidth=0.6, zorder=3)
        ax.text(x[i] - bw / 2, ap + 0.5, f"{ap:.1f}%", ha="center", va="bottom",
                fontsize=8, color=colour)
        ax.text(x[i] + bw / 2, hp + 0.5, f"{hp:.1f}%", ha="center", va="bottom",
                fontsize=8, color=colour, alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels([display(p) for p in order], fontsize=9.5)

    solid = mpatches.Patch(color="#555555", alpha=0.85, label="Official party account")
    hatch = mpatches.Patch(facecolor="#555555", alpha=0.38, hatch="///",
                            edgecolor="#555555", label="Hashtag corpus (creator content)")
    ax.legend(handles=[solid, hatch], fontsize=9, frameon=False, loc="upper right")

    ax.set_title(
        "Affordance use: official party accounts vs. hashtag corpus\n(partisan videos only)",
        fontsize=10, color="#222222", pad=8, loc="left", fontweight="bold")
    style_vbar(ax, ylabel="% of videos using any affordance",
               ylim=(0, max(max(acc_pcts), max(ht_pcts)) * 1.25))

    for tick, colour in zip(ax.get_xticklabels(), colours):
        tick.set_color(colour)

    plt.tight_layout()
    savefig("fig5")


# ══════════════════════════════════════════════════════════════
# FIG 6 -- engagement composite vs affordance score
# ══════════════════════════════════════════════════════════════

def fig6():
    accounts = load_accounts()
    ht_all   = load_hashtags(drop_apolitical=False)

    def panel(ax, df, colour, label, max_score=None):
        df = df.dropna(subset=["pc1", "trendiness_score"])
        x_all = df["trendiness_score"].values.astype(float)
        y_all = df["pc1"].values

        scores = sorted(df["trendiness_score"].unique())
        if max_score is not None:
            scores = [s for s in scores if s <= max_score]

        x_med  = np.array(scores)
        y_med  = [df[df["trendiness_score"] == s]["pc1"].median() for s in scores]
        y_q25  = [df[df["trendiness_score"] == s]["pc1"].quantile(0.25) for s in scores]
        y_q75  = [df[df["trendiness_score"] == s]["pc1"].quantile(0.75) for s in scores]

        slope, intercept, r, p, se = stats.linregress(x_all, y_all)
        n  = len(x_all)
        r2 = r**2
        print(f"  {label}: beta={slope:.4f}, p={p:.6f}, n={n:,}, R2={r2:.4f}")

        x_fit = np.linspace(x_med.min(), x_med.max(), 100)
        y_fit = slope * x_fit + intercept
        x_mean = x_all.mean()
        ss_x   = ((x_all - x_mean)**2).sum()
        y_pred_all = slope * x_all + intercept
        resid_std  = np.std(y_all - y_pred_all)
        ci_lo = y_fit - 1.96 * resid_std * np.sqrt(1/n + (x_fit - x_mean)**2 / ss_x)
        ci_hi = y_fit + 1.96 * resid_std * np.sqrt(1/n + (x_fit - x_mean)**2 / ss_x)

        err_lo = np.array(y_med) - np.array(y_q25)
        err_hi = np.array(y_q75) - np.array(y_med)

        ax.fill_between(x_fit, ci_lo, ci_hi, color=colour, alpha=0.15, zorder=1)
        ax.plot(x_fit, y_fit, color=colour, linewidth=1.8, zorder=2)
        ax.errorbar(x_med, y_med, yerr=[err_lo, err_hi],
                    fmt="o", color=colour, markersize=6,
                    elinewidth=1.2, capsize=3, zorder=3)
        ax.axhline(0, color="#bbbbbb", linewidth=0.8, linestyle="--")

        sign = "+" if slope >= 0 else ""
        p_str = "$p < 0.0001$" if p < 0.0001 else f"$p = {p:.4f}$"
        title = f"{label}\n$\\beta = {sign}{slope:.3f}$, {p_str}, $n = {n:,}$, $R^2 = {r2:.2f}$"
        ax.set_title(title, fontsize=9, color="#333333", pad=5, loc="center")
        ax.set_xlabel("Affordance score", fontsize=9.5, color="#333333")
        ax.set_ylabel("Engagement composite (PC1)", fontsize=9.5, color="#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.spines["left"].set_color("#cccccc")
        ax.tick_params(colors="#555555")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    panel(ax1, accounts, "#4472C4", "Account level", max_score=2)
    panel(ax2, ht_all,   "#70AD47", "Hashtag corpus", max_score=3)
    plt.tight_layout(pad=2.0)
    savefig("fig6")


# ══════════════════════════════════════════════════════════════
# FIG 7 -- amplification raincloud, partisan vs non-partisan
# ══════════════════════════════════════════════════════════════

def fig7():
    amp = load_amplification()
    print("  computing fig7...")
    p_df  = collect_ratios(amp, PARTISAN_LABELS)
    np_df = collect_ratios(amp, "Non-partisan")
    r_p   = p_df["ratio"].values
    r_np  = np_df["ratio"].values
    print(f"    partisan n={len(r_p)}, med={np.median(r_p):.3f}")
    print(f"    non-partisan n={len(r_np)}, med={np.median(r_np):.3f}")

    rows = [
        {"label":  "Partisan",    "ratios": r_p,
         "colour": "#4472C4",
         "ci":     bootstrap_median_ci(r_p),
         "p":      wilcoxon_log_p(r_p)},
        {"label":  "Non-partisan", "ratios": r_np,
         "colour": "#70AD47",
         "ci":     bootstrap_median_ci(r_np),
         "p":      wilcoxon_log_p(r_np)},
    ]
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    fig.subplots_adjust(left=0.20, right=0.985, top=0.97, bottom=0.26)
    draw_raincloud(ax, fig, rows)
    ax.set_xlabel("Amplification ratio  (mean partisan views / mean apolitical views, log scale)",
                  fontsize=10, color="#444444", labelpad=10)
    savefig("fig7")


# ══════════════════════════════════════════════════════════════
# FIG 8 -- amplification raincloud by party
# ══════════════════════════════════════════════════════════════

def fig8():
    amp = load_amplification()
    print("  computing fig8...")
    PARTIES = ["Labour", "Reform UK", "SNP", "Green", "Conservative", "Lib Dem"]
    rows = []
    for party in PARTIES:
        df_r = collect_ratios(amp, {party})
        r    = df_r["ratio"].values
        ci   = bootstrap_median_ci(r) if len(r) >= 5 else None
        p    = wilcoxon_log_p(r)
        pstr = f"{p:.4f}" if p is not None else "NA"
        print(f"    {party}: n={len(r)}, med={np.median(r):.3f}, p={pstr}")
        rows.append({"label": party, "ratios": r, "colour": pcolour(party), "ci": ci, "p": p})

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    fig.subplots_adjust(left=0.20, right=0.985, top=0.98, bottom=0.14)
    draw_raincloud(ax, fig, rows)
    ax.set_xlabel("Amplification ratio  (mean partisan views / mean apolitical views, log scale)",
                  fontsize=10, color="#444444", labelpad=10)
    savefig("fig8")


# ══════════════════════════════════════════════════════════════
# FIG 9 -- affordance-conditioned amplification dumbbell
# ══════════════════════════════════════════════════════════════

def fig9():
    amp = load_amplification()
    print("  computing fig9...")

    CONDITIONS = [
        ("Trending sound",          "cond_sound"),
        ("Platform-native feature", "cond_platform_native"),
        ("Virality hashtag",        "cond_hashtag"),
    ]
    COLOURS_9 = {
        "Trending sound":           "#E4003B",
        "Platform-native feature":  "#0087DC",
        "Virality hashtag":         "#02A95B",
    }

    def paired_ratios(df, cond_col):
        rng = np.random.default_rng(SEED)
        records = []
        for username, grp in df.groupby("username"):
            apol = grp[grp["predicted_party"] == "Apolitical"]["view_count"].values
            if len(apol) < MIN_APOLITICAL:
                continue
            pol = grp[grp["predicted_party"].isin(POLITICAL_LABELS)]
            if len(pol) == 0:
                continue
            used    = pol[pol[cond_col] == True]["view_count"].values
            notused = pol[pol[cond_col] == False]["view_count"].values
            if len(used) < 1 or len(notused) < 1:
                continue
            ru = bootstrap_ratio(used,    apol, rng=rng)
            rn = bootstrap_ratio(notused, apol, rng=rng)
            if ru is not None and rn is not None:
                records.append({"r_used": ru, "r_notused": rn})
        return pd.DataFrame(records)

    results = {}
    for label, col in CONDITIONS:
        pr = paired_ratios(amp, col)
        n  = len(pr)
        mu = np.median(pr["r_used"])
        mn = np.median(pr["r_notused"])
        ld = np.log10(pr["r_used"].values) - np.log10(pr["r_notused"].values)
        _, p = stats.wilcoxon(ld)
        results[label] = {"n": n, "med_used": mu, "med_notused": mn, "p": p}
        print(f"    {label}: n={n}, used={mu:.3f}, not_used={mn:.3f}, p={p:.4f}")

    x_min, x_max = 0.1, 3.0
    log_min = np.log10(x_min)
    log_max = np.log10(x_max)
    n_conds = len(CONDITIONS)
    spacing = 1.4
    y_pos   = {lb: (n_conds - 1 - i) * spacing for i, (lb, _) in enumerate(CONDITIONS)}

    fig, ax = plt.subplots(figsize=(9.0, 4.05))
    fig.subplots_adjust(left=0.24, right=0.985, top=0.97, bottom=0.16)
    ax.axvline(0, color="#aaaaaa", linestyle="--", linewidth=1.0, zorder=1)

    for label, _ in CONDITIONS:
        r      = results[label]
        y      = y_pos[label]
        colour = COLOURS_9[label]
        log_u  = np.log10(r["med_used"])
        log_nu = np.log10(r["med_notused"])
        sig    = sig_plain(r["p"])
        n      = r["n"]

        ax.plot([log_nu, log_u], [y, y], color=colour,
                linewidth=2.2, alpha=0.5, zorder=2, solid_capstyle="round")
        ax.scatter([log_nu], [y], color="white", s=110, zorder=4,
                   edgecolors=colour, linewidths=2.2)
        ax.scatter([log_u], [y], color=colour, s=110, zorder=4,
                   edgecolors="white", linewidths=0.8)
        ax.text(log_u,  y - 0.28, f"{r['med_used']:.2f}" + r"$\times$",
                va="top", ha="center", fontsize=8.5, color=colour, fontweight="bold")
        ax.text(log_nu, y + 0.25, f"{r['med_notused']:.2f}" + r"$\times$",
                va="bottom", ha="center", fontsize=8.5, color=colour)
        ax.text(log_min - 0.06, y, f"{label}\n(n={n}, {sig})",
                va="center", ha="right", fontsize=10, color="#333333",
                linespacing=1.5, clip_on=False)

    ax.axvline(log_min, color="#333333", linewidth=1.2, zorder=1)

    used_p    = mlines.Line2D([], [], marker="o", color="#555555",
                               markerfacecolor="#555555", markersize=8,
                               label="Affordance used", linewidth=0)
    notused_p = mlines.Line2D([], [], marker="o", color="#555555",
                               markerfacecolor="white", markersize=8,
                               label="Affordance not used", linewidth=0,
                               markeredgewidth=2.0)
    ax.legend(handles=[used_p, notused_p], fontsize=9, frameon=False,
              loc="upper left",
              bbox_to_anchor=(log_min + 0.02, (n_conds - 1) * spacing + 0.7),
              bbox_transform=ax.transData)

    tick_vals = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    ax.set_xticks([np.log10(v) for v in tick_vals])
    ax.set_xticklabels([f"${v}\\times$" for v in tick_vals], fontsize=9.5)
    ax.set_xlim(log_min, log_max)
    ax.set_ylim(-0.8, (n_conds - 1) * spacing + 0.8)
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="x", colors="#666666")
    ax.set_xlabel("Amplification ratio  (mean political views / mean apolitical views, log scale)",
                  fontsize=10, color="#444444", labelpad=8)
    savefig("fig9")


# ══════════════════════════════════════════════════════════════
# FIG 10 -- amplification by account-size quartile (raincloud)
# ══════════════════════════════════════════════════════════════

def fig10():
    amp = load_amplification()
    print("  computing fig10...")
    TIER_COLOURS = {"Q1": "#9ecae1", "Q2": "#6baed6", "Q3": "#2171b5", "Q4": "#08306b"}
    TIER_ORDER   = ["Q1", "Q2", "Q3", "Q4"]

    all_r = collect_ratios(amp, POLITICAL_LABELS)
    all_r["tier"] = pd.qcut(all_r["baseline_mean"], q=4, labels=TIER_ORDER)

    rows = []
    for tier in TIER_ORDER:
        sub = all_r[all_r["tier"] == tier]
        r   = sub["ratio"].values
        bl  = sub["baseline_mean"].values
        p   = wilcoxon_log_p(r)
        ci  = (np.percentile(r, 2.5), np.percentile(r, 97.5))
        label = f"{tier}  ({bl.min():.0f}--{bl.max():.0f} views)"
        pstr  = f"{p:.4f}" if p is not None else "NA"
        print(f"    {tier}: n={len(r)}, med={np.median(r):.3f}, p={pstr}")
        rows.append({"label": label, "ratios": r,
                     "colour": TIER_COLOURS[tier], "ci": ci, "p": p})

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    fig.subplots_adjust(left=0.24, right=0.985, top=0.97, bottom=0.18)
    draw_raincloud(ax, fig, rows)
    ax.set_xlabel("Amplification ratio  (mean political views / mean apolitical views, log scale)",
                  fontsize=10, color="#444444", labelpad=10)
    savefig("fig10")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

FIGS = {
    "fig1": fig1,
    "fig2": fig2,
    "fig3": fig3,
    "fig4": fig4,
    "fig5": fig5,
    "fig6": fig6,
    "fig7": fig7,
    "fig8": fig8,
    "fig9": fig9,
    "fig10": fig10,
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(FIGS.keys())
    for t in targets:
        print(f"\n{t}:")
        FIGS[t]()
