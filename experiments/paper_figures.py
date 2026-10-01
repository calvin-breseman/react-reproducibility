"""Candidate figures for the REACT paper, from the final run of 2026-09-30.

Reads `outputs/final/` (records, summaries, accuracy, q-sweep) and the known-truth simulation
records copied to `data/sim/`, and writes PNGs in the light theme of the REACT Results page
(IBM Plex Sans from `data/fonts/`, falling back to Helvetica).

    .venv/bin/python experiments/paper_figures.py                 # all figures
    .venv/bin/python experiments/paper_figures.py --only fig04,fig09
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs" / "final"
RUNS = {
    "eindhoven": FINAL / "eindhoven_floorA",
    "eindhoven_nofloor": FINAL / "eindhoven_nofloor",
    "ditl": FINAL / "ditl_floor1cm",
    "ditl_2cm": FINAL / "ditl_floor2cm",
}
DATASET_LABEL = {"eindhoven": "Eindhoven", "ditl": "Day in the Life"}
SIM = ROOT / "data" / "sim"
HEAD_INFO = 4.2
HZ = 10
T_MAX = 150  # 15 s window

# ---------- theme (REACT Results page, light tokens) ----------
INK, INK2, INK3, RULE, PANEL, ACCENT = "#1c2026", "#4d5562", "#7c8491", "#d8dce2", "#f2f4f6", "#1f4e79"
MODELS = ["ConstantVelocity", "CTRV", "IMM", "GRUPosition", "GRUOrientation", "Trajectron_mix"]
NAME = {
    "ConstantVelocity": "CV", "CTRV": "CTRV", "IMM": "IMM", "GRUPosition": "GRU (position)",
    "GRUOrientation": "GRU (heading)", "Trajectron_mix": "Trajectron++", "Trajectron": "Trajectron++ (moment-matched)",
    "oracle_B": "Oracle B", "Climatology": "Climatology",
}
COLOR = {
    "ConstantVelocity": "#c98a00", "CTRV": "#3d9bd6", "IMM": "#00896a", "GRUPosition": "#0b5f9c",
    "GRUOrientation": "#b25f98", "Trajectron_mix": "#cf4f00", "Trajectron": "#cf4f00", "oracle_B": INK,
    "Climatology": INK3,
}
LEARNED = {"GRUPosition", "GRUOrientation", "Trajectron_mix"}
OUTCOME = [  # (key, label, colour), best to worst
    ("better", "Anticipated, confirmed", "#1f4e79"),
    ("indistinguishable", "Anticipated, unconfirmed", "#a9c1da"),
    ("rec_early", "Recovered within 15 frames", "#c9ced6"),
    ("rec_late", "Recovered after 15 frames", "#8a929e"),
    ("censored", "Censored (never recovered)", "#b5482a"),
]
TARGET_LABEL = {0.0: "climatology margin", 0.1: "REACT@0.1 m", 0.3: "REACT@0.3 m"}
FULL_W, HALF_W = 6.5, 3.2
HEIGHT = {}  # paper mode: per-figure height overrides (inches)
RENAME = {}  # paper mode: output file names
PAPER = False  # paper mode: no footnotes or seconds axes (captions carry them)


def note(fig, *args, **kw):
    if not PAPER:
        fig.text(*args, **kw)


def ht(key, default):
    return HEIGHT.get(key, default)


def setup_theme():
    family = "Helvetica"
    for f in sorted((ROOT / "data" / "fonts").glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
        family = "IBM Plex Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "medium",
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": RULE, "axes.linewidth": 0.8,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": RULE, "grid.linewidth": 0.6, "grid.alpha": 0.8, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
    })


def panel_label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom", color=INK)


def seconds_axis(ax):
    if PAPER:
        return None
    sec = ax.secondary_xaxis("top", functions=(lambda f: f / HZ, lambda s: s * HZ))
    sec.tick_params(labelsize=6.5, colors=INK3, length=2)
    sec.spines["top"].set_color(RULE)
    return sec


def model_legend(fig, models, oracle=True, ncol=None, y=1.0, extra=()):
    h = [Line2D([], [], color=COLOR[m], lw=2, label=NAME[m]) for m in models]
    if oracle:
        h.append(Line2D([], [], color=INK, lw=1.4, ls=(0, (4, 2)), label="Oracle B (floor)"))
    h += list(extra)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(h), handlelength=2.2,
               columnspacing=1.4)


# ---------- data ----------
REC_COLS = ["lead", "info", "distortion", "model", "track", "t0", "event_type", "status", "onset", "react",
            "duration", "relapse", "relapse_after", "relapse3"]


@lru_cache(None)
def records(run: str, info: float | None = HEAD_INFO) -> pd.DataFrame:
    df = pd.read_csv(glob.glob(str(RUNS[run] / "records*.csv"))[0], usecols=REC_COLS)
    if info is not None:
        df = df[np.isclose(df["info"], info)]
    df["distortion"] = df["distortion"].round(2)
    return df.reset_index(drop=True)


@lru_cache(None)
def summary(run: str) -> pd.DataFrame:
    S = json.load(open(glob.glob(str(RUNS[run] / "summary*.json"))[0]))
    df = pd.DataFrame(S["results"])
    df["distortion"] = df["distortion"].fillna(0.0).astype(float).round(2)
    return df


def accuracy(run: str) -> dict:
    return json.load(open(RUNS[run] / "accuracy.json"))["models"]


def onset_set(g: pd.DataFrame) -> pd.DataFrame:
    """Events that fell behind (Oracle B never does; all of its events enter its curve)."""
    return g[g["status"].isin(["recovered", "censored"])]


def km(g: pd.DataFrame, t_max: int = T_MAX) -> np.ndarray:
    """Kaplan-Meier share not yet recovered, given onset: recovery at `duration`, censoring at `duration`."""
    o = onset_set(g)
    t = o["duration"].to_numpy(float)
    ev = (o["status"] == "recovered").to_numpy()
    s, out = 1.0, []
    for f in range(t_max + 1):
        risk = (t >= f).sum()
        hit = ((t == f) & ev).sum()
        if risk:
            s *= 1 - hit / risk
        out.append(s)
    return np.array(out)


def km_all(g: pd.DataFrame, t_max: int = T_MAX) -> np.ndarray:
    """KM share not yet recovered over all events; anticipated events count as recovered at frame 0."""
    st = g["status"].to_numpy()
    onset = np.isin(st, ("recovered", "censored"))
    t = np.where(onset, g["duration"].to_numpy(float), 0.0)
    ev = (st == "recovered") | ~onset
    s_, out = 1.0, []
    for f in range(t_max + 1):
        risk = (t >= f).sum()
        hit = ((t == f) & ev).sum()
        if risk:
            s_ *= 1 - hit / risk
        out.append(s_)
    return np.array(out)


def km_median(curve: np.ndarray) -> float:
    idx = np.flatnonzero(curve <= 0.5)
    return float(idx[0]) if len(idx) else np.nan


def outcome_shares(g: pd.DataFrame, cut: int = 15) -> dict:
    n = len(g)
    st = g["status"]
    rec = st == "recovered"
    return {
        "better": (st == "better").sum() / n, "indistinguishable": (st == "indistinguishable").sum() / n,
        "rec_early": (rec & (g["react"] <= cut)).sum() / n, "rec_late": (rec & (g["react"] > cut)).sum() / n,
        "censored": (st == "censored").sum() / n,
    }


def cond_rates(g: pd.DataFrame) -> dict:
    o = onset_set(g)
    rec = o[o["status"] == "recovered"]
    curve = km(g)  # recovered by K frames = 1 - KM share not yet recovered at K (handles short windows)
    return {
        "n": len(g), "n_onset": len(o), "anticipation": 1 - len(o) / len(g),
        "censored": (o["status"] == "censored").mean(), "by10": 1 - curve[10], "by15": 1 - curve[15],
        "by30": 1 - curve[30], "median": km_median(curve),
        "relapse": rec["relapse"].astype(bool).mean(), "relapse3": rec["relapse3"].astype(bool).mean(),
    }


def sel(df, lead=None, d=None, model=None):
    m = np.ones(len(df), bool)
    if lead is not None:
        m &= df["lead"].to_numpy() == lead
    if d is not None:
        m &= np.isclose(df["distortion"].to_numpy(), d)
    if model is not None:
        m &= df["model"].to_numpy() == model
    return df[m]


def km_lines(ax, run, lead, d, t_max=T_MAX, emphasize=None, models=MODELS, all_events=False):
    df = records(run)
    curve = km_all if all_events else km
    for m in models:
        g = sel(df, lead, d, m)
        if not len(g):
            continue
        c = curve(g, t_max)
        lw, alpha = 1.6, 1.0
        if emphasize is not None:
            lw, alpha = (2.4, 1.0) if m == emphasize else (1.2, 0.75)
        ax.step(np.arange(t_max + 1), c, where="post", color=COLOR[m], lw=lw, alpha=alpha)
    g = sel(df, lead, d, "oracle_B")
    ax.step(np.arange(t_max + 1), km(g, t_max), where="post", color=INK, lw=1.2, ls=(0, (4, 2)))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, t_max)


def save(fig, out: Path, name: str):
    name = RENAME.get(name, name)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", name)


# ---------- figures ----------
def fig01(out):
    """Illustrative timeline of one event and the outcome categories."""
    rng = np.random.default_rng(3)
    h, m = 5, 5
    t = np.arange(-12, 61)

    def smooth(x, k=3):
        xp = np.pad(x, k // 2, mode="edge")
        return np.convolve(xp, np.ones(k) / k, mode="valid")

    kw = [(-12, .06), (0, .07), (3, .5), (6, .99), (12, 1.0), (16, .9), (20, .3), (24, .06), (29, .06), (31, .99),
          (33, .99), (35, .1), (39, .05), (60, .04)]
    kb = [(-12, .55), (0, .5), (4, .08), (14, .05), (18, .3), (21, .85), (23, 1.0), (28, 1.0), (30, .5), (32, .05),
          (35, .1), (38, .85), (41, 1.0), (60, 1.0)]
    p_worse = np.clip(smooth(np.interp(t, *zip(*kw)) + rng.normal(0, 0.01, t.size)), 0, 1)
    p_better = np.clip(smooth(np.interp(t, *zip(*kb)) + rng.normal(0, 0.01, t.size)), 0, 1)
    t_on = int(t[(t >= 1) & (p_worse >= 0.95)][0])
    t_rec = int(t[(t > t_on) & (p_better >= 0.95)][0])
    t_rel = int(t[(t > t_rec) & (p_worse >= 0.95)][0])

    fig = plt.figure(figsize=(FULL_W, ht("fig01", 4.3)))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 1.0 if PAPER else 1.15], hspace=0.45 if PAPER else 0.55)
    ax = fig.add_subplot(gs[0])
    ax.axvspan(0, h + m, color=PANEL, zorder=0)
    wy, wl = (0.08, f"onset window ($h$+{m})") if PAPER else (1.2, f"onset window\n(lead $h$ + {m} frames)")
    (lambda *a, **k: None if PAPER else ax.text(*a, **k))((h + m) / 2, wy, wl, ha="center", va="center", fontsize=6 if PAPER else 7, color=INK2)
    ax.axvspan(t_rec, t_rec + 10, color="#f6ece8", zorder=0)
    (lambda *a, **k: None if PAPER else ax.text(*a, **k))(t_rec + 5, wy, "relapse window (10)" if PAPER else "relapse window\n(10 frames)", ha="center",
            va="center", fontsize=6 if PAPER else 7, color=INK2)
    ax.axhline(0.95, color=INK3, lw=0.8, ls=(0, (2, 2)))
    ax.text(60.5, 0.95, "$p$ = 0.95", va="center", ha="left", fontsize=7, color=INK3)
    ax.plot(t, p_worse, color="#b5482a", lw=1.8, label=r"$P(I_t < 0 \mid \mathrm{data})$: behind")
    ax.plot(t, p_better, color=ACCENT, lw=1.8, label=r"$P(I_t > \varepsilon \mid \mathrm{data})$: ahead by $\varepsilon$")
    ax.axvline(0, color=INK, lw=1.0)
    ax.text(-0.6, 1.03, "event $t_0$", ha="right", va="bottom", fontsize=7.5, color=INK)
    for x, lab, c in [(t_on, "onset", "#b5482a"), (t_rec, "recovery", ACCENT), (t_rel, "relapse", "#b5482a")]:
        y = p_worse[t == x][0] if c != ACCENT else p_better[t == x][0]
        ax.plot([x], [y], "o", ms=5, color=c, mec="white", mew=0.8, zorder=5)
        ax.text(x, 1.03, lab, fontsize=7.5, color=c, ha="center", va="bottom", fontweight="medium")
    ax.annotate("", xy=(t_rec, -0.2), xytext=(0, -0.2), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9),
                annotation_clip=False)
    ax.text(t_rec / 2, -0.23, f"REACT = {t_rec} frames", ha="center", va="top", fontsize=7.5, color=INK,
            fontweight="medium")
    ax.set_xlim(-12, 60)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 0.95])
    ax.set_ylabel("Posterior probability")
    ax.set_xlabel("Frames since event (10 Hz)", labelpad=24)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=7, handlelength=1.4)
    panel_label(ax, "a", y=1.1)

    ax2 = fig.add_subplot(gs[1])
    ax2.grid(False)
    rows = [
        ("Anticipated", [(0, 60, ACCENT, "never confidently worse within the onset window; confirmed if confidently better there")]),
        ("Recovered", [(0, 9, "#c9ced6", ""), (9, 26, "#e6b7a9", "behind"), (26, 60, ACCENT, "")]),
        ("Censored", [(0, 6, "#c9ced6", ""), (6, 60, "#e6b7a9", "never confidently better before the next event or 15 s")]),
        ("Relapsed", [(0, 8, "#c9ced6", ""), (8, 22, "#e6b7a9", ""), (22, 30, ACCENT, ""), (30, 60, "#e6b7a9", "")]),
    ]
    for i, (lab, segs) in enumerate(rows):
        y = len(rows) - 1 - i
        for a, b, c, txt in segs:
            ax2.barh(y, b - a, left=a, height=0.75 if PAPER else 0.55, color=c, edgecolor="white", lw=0.8)
            if txt:
                ax2.text(a + 1, y, txt, va="center", fontsize=5.6 if PAPER else 6.8, color="white" if c == ACCENT else INK)
        ax2.text(-2, y, lab, ha="right", va="center", fontsize=6.3 if PAPER else 7.5, color=INK, fontweight="medium")
    ax2.plot([26], [2], "|", color=INK, ms=10, mew=1.2)
    ax2.plot([22], [0], "|", color=INK, ms=10, mew=1.2)
    ax2.plot([30], [0], "|", color="#b5482a", ms=10, mew=1.2)
    ax2.set_xlim(-12, 60)
    ax2.set_ylim(-0.6, len(rows) - 0.4)
    ax2.set_yticks([])
    ax2.set_xticks([])
    for s in ("left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.axvline(0, color=INK, lw=1.0)
    ax2.legend(handles=[Patch(color=ACCENT, label="confidently better"), Patch(color="#e6b7a9", label="behind climatology"),
                        Patch(color="#c9ced6", label="not yet decided")], loc="upper center",
               bbox_to_anchor=(0.55, -0.02), ncol=3, fontsize=7)
    panel_label(ax2, "b", y=1.0)
    note(fig, 0.99, 0.005, "Illustrative traces, not data.", ha="right", fontsize=6.5, color=INK3, style="italic")
    save(fig, out, "fig01_react_schematic")


TYPE_GROUP = {"speed_change": "Speed change", "both": "Speed + turn", "start": "Start", "stop": "Stop",
              "turn": "Turn", "turn_change": "Turn", "turn_exit": "Turn", "sharp_turn": "Turn"}
TYPE_ORDER = ["Speed change", "Speed + turn", "Start", "Stop", "Turn"]
TYPE_TICK = ["Speed\nchange", "Speed\n+ turn", "Start", "Stop", "Turn"]
DS_COLOR = {"eindhoven": ACCENT, "ditl": "#9a6a2f"}


def events_table(run):
    g = sel(records(run), 5, 0.0, "oracle_B")[["track", "t0", "event_type"]].drop_duplicates()
    g = g.assign(kind=g["event_type"].map(TYPE_GROUP).fillna("Other"))
    g = g.sort_values(["track", "t0"])
    g["gap"] = g.groupby("track")["t0"].diff(-1).abs()
    return g


def fig02(out):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, ht("fig02", 2.6)), gridspec_kw=dict(width_ratios=[1.1, 1], wspace=0.3))
    ax = axes[0]
    w = 0.38
    for i, run in enumerate(["eindhoven", "ditl"]):
        g = events_table(run)
        share = g["kind"].value_counts(normalize=True).reindex(TYPE_ORDER).fillna(0)
        x = np.arange(len(TYPE_ORDER)) + (i - 0.5) * w
        ax.bar(x, share.values, w, color=DS_COLOR[run], label=f"{DATASET_LABEL[run]} (n = {len(g):,})")
    ax.set_xticks(np.arange(len(TYPE_ORDER)), TYPE_TICK)
    ax.set_ylabel("Share of evaluated events")
    ax.legend(loc="upper right")
    panel_label(ax, "a")
    ax = axes[1]
    for i, run in enumerate(["eindhoven", "ditl"]):
        gap = events_table(run)["gap"].dropna().to_numpy()
        xs = np.sort(np.minimum(gap, 400))
        ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=DS_COLOR[run], label=DATASET_LABEL[run])
        ax.text(390, 0.32 - 0.1 * i, f"{DATASET_LABEL[run]}: {np.mean(gap >= 150):.0%} of gaps ≥ 15 s",
                color=DS_COLOR[run], fontsize=6.8, ha="right")
    ax.axvline(150, color=INK3, lw=0.8, ls=(0, (2, 2)))
    ax.set_xlim(0, 400)
    ax.set_xlabel("Frames to the next evaluated event on the track")
    ax.set_ylabel("Cumulative share")
    ax.text(160, 0.04, "window cap (15 s)", fontsize=6.5, color=INK3)
    panel_label(ax, "b")
    save(fig, out, "fig02_event_corpus")


def fig03(out):
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, ht("fig03", 4.6)), gridspec_kw=dict(wspace=0.75, hspace=0.55, width_ratios=[1, 1, 0.8]))
    for r, run in enumerate(["eindhoven", "ditl"]):
        A = accuracy(run)
        names = [m for m in MODELS if m in A] + ["Climatology"]
        H = range(1, 16)
        for m in names:
            v = A[m]
            ls = (0, (3, 2)) if m == "Climatology" else "-"
            axes[r, 0].plot(H, [v[f"err{h}"] * 100 for h in H], color=COLOR[m], ls=ls)
            axes[r, 1].plot(H, [-v[f"ll{h}"] for h in H], color=COLOR[m], ls=ls)
        axes[r, 0].set_ylabel(f"{DATASET_LABEL[run]}\nDisplacement error (cm)")
        axes[r, 1].set_ylabel("NLL (nats)")
        ade = [A[m]["ade"] * 100 for m in names]
        fde = [A[m]["fde"] * 100 for m in names]
        y = np.arange(len(names))[::-1]
        ax = axes[r, 2]
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        for yi, a, f, m in zip(y, ade, fde, names):
            ax.plot([a, f], [yi, yi], color=RULE, lw=1.5, zorder=1)
            ax.plot(a, yi, "o", color=COLOR[m], ms=5, zorder=2)
            ax.plot(f, yi, "D", color=COLOR[m], ms=4.5, mfc="white", zorder=2)
        ax.set_yticks(y, [NAME[m] for m in names], fontsize=7)
        ax.set_xlabel("cm")
        for c in (0, 1):
            axes[r, c].set_xlabel("Lead (frames)")
            axes[r, c].set_xticks([1, 5, 10, 15])
    axes[0, 0].set_title("Error of forecast mean", loc="left")
    axes[0, 1].set_title("Negative log-likelihood", loc="left")
    axes[0, 2].set_title("ADE (filled), FDE (open)", loc="left")
    for a, s in zip(axes.flat, "abcdef"):
        panel_label(a, s)
    h = [Line2D([], [], color=COLOR[m], lw=2, label=NAME[m]) for m in MODELS]
    h.append(Line2D([], [], color=INK3, lw=1.6, ls=(0, (3, 2)), label="Climatology"))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=7, handlelength=1.8, columnspacing=1.1)
    note(fig, 0.5, -0.02, "All issue frames, not only event windows. NLL of the 2-D position in metres, so sharp forecasts "
             "give negative values; lower is better.", ha="center", fontsize=6.3, color=INK3)
    save(fig, out, "fig03_classical_accuracy")


def fig04(out, run="eindhoven", name="fig04_km_given_onset", d=0.0):
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.8), sharey=True, gridspec_kw=dict(hspace=0.62, wspace=0.12))
    for ax, lead, s in zip(axes.flat, [1, 5, 10, 15], "abcd"):
        km_lines(ax, run, lead, d)
        ax.set_title(f"Lead {lead} ({lead / HZ:g} s)", loc="left", pad=16)
        seconds_axis(ax)
        ax.set_xlabel("Frames since event")
        panel_label(ax, s, y=1.12)
        df = records(run)
        tail = sorted(((km(sel(df, lead, d, m))[-1], m) for m in MODELS), reverse=True)
        ax.text(T_MAX - 2, 0.97, "not recovered by 15 s:\n" + "\n".join(f"{NAME[m]} {v:.0%}" for v, m in tail[:3]),
                ha="right", va="top", fontsize=6.3, color=INK2, linespacing=1.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share not yet recovered\n(given onset)")
    model_legend(fig, MODELS, y=1.03, ncol=4)
    save(fig, out, name)


def fig05(out):
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.7), gridspec_kw=dict(width_ratios=[1.25, 1], hspace=0.6, wspace=0.3))
    df = records("eindhoven")
    for r, lead in enumerate([10, 15]):
        ax = axes[r, 0]
        ax.axvspan(0, lead + 5, color=PANEL, zorder=0)
        ax.text((lead + 5) / 2, 0.06, "onset window", ha="center", fontsize=6.5, color=INK3)
        km_lines(ax, "eindhoven", lead, 0.0, t_max=40)
        ax.axvline(15, color=INK3, lw=0.8, ls=(0, (2, 2)))
        ax.set_title(f"Lead {lead} ({lead / HZ:g} s): first 4 s after the event", loc="left", pad=16)
        seconds_axis(ax)
        ax.set_xlabel("Frames since event")
        ax.set_ylabel("Share not yet recovered\n(given onset)")
        ax = axes[r, 1]
        cuts = ["by10", "by15", "by30"]
        w = 0.13
        for i, m in enumerate(MODELS):
            c = cond_rates(sel(df, lead, 0.0, m))
            x = np.arange(3) + (i - 2.5) * w
            ax.bar(x, [c[k] for k in cuts], w, color=COLOR[m])
        ax.set_xticks(np.arange(3), ["≤ 10 frames", "≤ 15 frames", "≤ 30 frames"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Recovered, given onset")
        ax.set_title("Share recovered by", loc="left", pad=16)
        panel_label(axes[r, 0], "ab"[r] if r == 0 else "c", y=1.12)
        panel_label(axes[r, 1], "b" if r == 0 else "d", y=1.12)
    model_legend(fig, MODELS, y=1.03, ncol=4)
    save(fig, out, "fig05_early_window")


def outcome_bars(ax, run, lead, d, models, show_labels=True):
    df = records(run)
    y = np.arange(len(models))[::-1]
    for yi, m in zip(y, models):
        sh = outcome_shares(sel(df, lead, d, m))
        left = 0
        for k, _, c in OUTCOME:
            ax.barh(yi, sh[k], left=left, color=c, height=0.7, edgecolor="white", lw=0.5)
            if sh[k] >= 0.09:
                ax.text(left + sh[k] / 2, yi, f"{sh[k] * 100:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if k in ("better", "censored", "rec_late") else INK)
            left += sh[k]
    ax.set_yticks(y, [NAME[m] for m in models] if show_labels else [""] * len(models), fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50", "100%"])
    ax.grid(False)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def outcome_legend(fig, y):
    fig.legend(handles=[Patch(color=c, label=lab) for _, lab, c in OUTCOME], loc="upper center",
               bbox_to_anchor=(0.5, y), ncol=5, handlelength=1.2, columnspacing=1.0, fontsize=7)


def fig06(out, name="fig06_outcome_composition", d=0.0):
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.6), gridspec_kw=dict(wspace=0.16))
    for i, (ax, lead) in enumerate(zip(axes, [1, 5, 10, 15])):
        outcome_bars(ax, "eindhoven", lead, d, MODELS, show_labels=i == 0)
        ax.set_title(f"Lead {lead}", loc="left")
        ax.set_xlabel("Share of events")
    outcome_legend(fig, 1.1)
    save(fig, out, name)


SHORT = {"ConstantVelocity": "CV", "CTRV": "CTRV", "IMM": "IMM", "GRUPosition": "GRU-p", "GRUOrientation": "GRU-h",
         "Trajectron_mix": "T++"}


def fig07(out):
    """Anticipation against early recovery: upper right is good on both."""
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 2.5), sharey=True, gridspec_kw=dict(wspace=0.12))
    df = records("eindhoven")
    for ax, lead, s in zip(axes, [1, 5, 10, 15], "abcd"):
        ax.grid(axis="x")
        pts = {m: cond_rates(sel(df, lead, 0.0, m)) for m in MODELS}
        for m, c in pts.items():
            ax.plot(c["anticipation"], c["by15"], "o", color=COLOR[m], ms=6, mec="white", mew=0.7, zorder=3)
        ax.set_xlim(-0.02, 0.5)
        ax.set_ylim(0.2, 0.9)
        ax.set_title(f"Lead {lead}", loc="left")
        ax.set_xlabel("Anticipation rate")
        panel_label(ax, s)
    axes[0].set_ylabel("Recovered within 15 frames\n(given onset)")
    model_legend(fig, MODELS, oracle=False, y=1.1, ncol=6)
    fig.text(0.5, -0.1, "Each point is one forecaster, Eindhoven, 4.2 nats, climatology margin. Up: recovers quickly "
             "when it falls behind. Right: falls behind less often.", ha="center", fontsize=6.3, color=INK3)
    save(fig, out, "fig07_anticipation_vs_recovery")


def fig08(out, lead=10):
    A = accuracy("eindhoven")
    df = records("eindhoven")
    rates = {m: cond_rates(sel(df, lead, 0.0, m)) for m in MODELS}
    cols = [
        ("ADE", {m: A[m]["ade"] for m in MODELS}, True),
        ("FDE", {m: A[m]["fde"] for m in MODELS}, True),
        (f"NLL\n(lead {lead})", {m: -A[m][f"ll{lead}"] for m in MODELS}, True),
        ("Anticipation", {m: rates[m]["anticipation"] for m in MODELS}, False),
        ("Recovered\n≤ 15 fr.", {m: rates[m]["by15"] for m in MODELS}, False),
        ("Median\nREACT", {m: rates[m]["median"] for m in MODELS}, True),
        ("Censored", {m: rates[m]["censored"] for m in MODELS}, True),
        ("Relapse\n(3 frames)", {m: rates[m]["relapse3"] for m in MODELS}, True),
    ]
    ranks = {}
    for lab, vals, low_good in cols:
        order = sorted(MODELS, key=lambda m: (vals[m] if low_good else -vals[m], m))
        ranks[lab] = {m: order.index(m) + 1 for m in MODELS}
    fig, ax = plt.subplots(figsize=(FULL_W, ht("fig08", 3.1)))
    ax.grid(False)
    x = np.arange(len(cols))
    for m in MODELS:
        ys = [ranks[lab][m] for lab, _, _ in cols]
        lw = 2.2 if m in ("Trajectron_mix", "ConstantVelocity") else 1.4
        ax.plot(x, ys, "-o", color=COLOR[m], lw=lw, ms=5, mec="white", mew=0.7)
        ax.text(-0.15, ys[0], NAME[m], ha="right", va="center", fontsize=7, color=COLOR[m])
        ax.text(len(cols) - 1 + 0.6, ys[-1], NAME[m], ha="left", va="center", fontsize=7, color=COLOR[m])
        for xi, (lab, vals, _) in zip(x, cols):
            v = vals[m]
            txt = f"{v * 100:.1f}" if lab in ("ADE", "FDE") else (f"{v:.2f}" if "NLL" in lab else (
                f"{v:.0f}" if "Median" in lab else f"{v:.0%}"))
            ax.text(xi + 0.13, ranks[lab][m] - 0.2, txt, ha="left", va="center", fontsize=5.6, color=INK2,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85), zorder=4)
    ax.axvspan(-0.4, 2.4, color=PANEL, zorder=0)
    ax.text(1, 0.25, "classical metrics", ha="center", fontsize=7, color=INK2, fontweight="medium")
    ax.text(5, 0.25, f"REACT at lead {lead}, climatology margin", ha="center", fontsize=7, color=INK2,
            fontweight="medium")
    ax.set_xticks(x, [lab for lab, _, _ in cols], fontsize=7)
    if PAPER:
        short = ["ADE", "FDE", "NLL", "Antic.", "Rec.\n≤15 fr.", "Median\nREACT", "Cens.", "Relapse\n(3 fr.)"]
        ax.set_xticks(x, short, fontsize=6.5)
    ax.set_yticks(range(1, 7), [f"{i}" for i in range(1, 7)])
    ax.set_ylim(6.6, 0.0)
    ax.set_xlim(-1.4, len(cols) - 1 + 1.9)
    ax.set_ylabel("Rank (1 = best)")
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    note(fig, 0.5, -0.02, "ADE/FDE in cm over the 1.5 s horizon. Recovery, median REACT (frames), censoring and relapse are given onset. "
             "Small numbers are the values ranked.",
             ha="center", fontsize=6.3, color=INK3)
    if PAPER:
        ax.tick_params(axis="y", labelleft=False)
        ax.set_ylabel("Rank (best at top)")
    save(fig, out, "fig08_rank_discordance")


def fig09(out):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, ht("fig09", 2.7)), sharey=True, gridspec_kw=dict(wspace=0.1))
    df = records("eindhoven")
    for ax, lead, s in zip(axes, [5, 10, 15], "abc"):
        km_lines(ax, "eindhoven", lead, 0.1, emphasize="Trajectron_mix")
        ax.set_title(f"Lead {lead} ({lead / HZ:g} s)", loc="left", pad=16)
        seconds_axis(ax)
        ax.set_xlabel("Frames since event")
        g = sel(df, lead, 0.1, "Trajectron_mix")
        c = cond_rates(g)
        ax.text(T_MAX - 3, 0.97, f"Trajectron++\ncensored: {c['censored']:.0%}\nnot recovered by 15 s: {km(g)[-1]:.0%}",
                ha="right", va="top", fontsize=6.3, color=COLOR["Trajectron_mix"], linespacing=1.3)
        panel_label(ax, s, y=1.12)
    axes[0].set_ylabel("Share not yet recovered\n(given onset)")
    model_legend(fig, MODELS, y=1.1, ncol=7)
    save(fig, out, "fig09_km_react_at_0p1")


def fig10(out):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.6), sharey=True, gridspec_kw=dict(wspace=0.1))
    df = records("eindhoven")
    marks = {0.0: ("o", "climatology margin"), 0.3: ("s", "0.3 m"), 0.1: ("D", "0.1 m")}
    y = np.arange(len(MODELS))[::-1]
    for ax, lead, s in zip(axes, [5, 10, 15], "abc"):
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        for yi, m in zip(y, MODELS):
            v = {d: cond_rates(sel(df, lead, d, m))["censored"] for d in marks}
            ax.plot([min(v.values()), max(v.values())], [yi, yi], color=RULE, lw=2, zorder=1)
            for d, (mk, _) in marks.items():
                ax.plot(v[d], yi, mk, color=COLOR[m], ms=5.5 if mk != "D" else 5, mfc=COLOR[m] if d == 0.1 else "white",
                        mew=1.2, zorder=2)
        g = sel(df, lead, 0.1, "oracle_B")
        ax.axvline((g["status"] == "censored").mean(), color=INK, lw=1, ls=(0, (4, 2)))
        ax.set_xlim(-0.02, 0.75)
        ax.set_xticks([0, 0.2, 0.4, 0.6], ["0", "20", "40", "60%"])
        ax.set_title(f"Lead {lead}", loc="left")
        ax.set_xlabel("Censored, given onset")
        panel_label(ax, s)
    axes[0].set_yticks(y, [NAME[m] for m in MODELS], fontsize=7)
    axes[0].tick_params(axis="y", length=0)
    h = [Line2D([], [], marker=mk, color=INK2, ls="", mfc=INK2 if d == 0.1 else "white", label=lab)
         for d, (mk, lab) in marks.items()]
    h.append(Line2D([], [], color=INK, lw=1, ls=(0, (4, 2)), label="Oracle B at 0.1 m"))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, title="Target", title_fontsize=7)
    save(fig, out, "fig10_censoring_vs_target")


def sim_relapse_validation():
    """False-relapse rate and sensitivity of the two relapse rules on the known-truth simulation (lead 5, 4.2 nats),
    with exp10's 10-frame window after recovery. Pooled over oracle, lag5, lag20, under and CV (as in ana.py)."""
    logk = np.log(19.0)

    def fires(mask, start, end, k):
        return any(mask[j:j + k].all() for j in range(start, min(end, len(mask) - k + 1)))

    out = {}
    for tag, label in [("0.0", "none"), ("18.2", "1.82 cm\n(est.)"), ("23.7", "2.37 cm\n(true)")]:
        path = SIM / f"rec_I4.2_s{tag}.pkl"
        if not path.exists():
            continue
        st, dp = {1: [], 3: []}, {1: [], 3: []}
        for r in pickle.load(open(path, "rb")):
            if r["model"] == "over":
                continue
            rel = r["rel"].astype(int)
            I = r["trueI"]
            on_m = r["lo_on"][0.0] >= logk
            on = np.flatnonzero(on_m & (rel <= 10))
            if not len(on):
                continue
            rr = np.flatnonzero((r["lo_rec"] >= logk) & (np.arange(len(rel)) > on[0]))
            if not len(rr):
                continue
            start, end = rr[0] + 1, rr[0] + 11
            stable = not (I[start:end + 2] < 0).any()  # truly ahead through the window (and a run's extension)
            dip = fires(I < 0, start, end, 3)  # a real setback: three truly-worse frames in a row
            for k in (1, 3):
                f = fires(on_m, start, end, k)
                (st[k].append(f) if stable else None)
                (dp[k].append(f) if dip else None)
        out[label] = {k: (np.mean(st[k]), len(st[k]), np.mean(dp[k]), len(dp[k])) for k in (1, 3)}
    return out


def fig11(out):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, ht("fig11", 2.8)), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 0.78, 0.85]))
    df = records("eindhoven")
    w = 0.13
    for ax, d, leads, s in zip(axes[:2], [0.0, 0.1], [[1, 5, 10, 15], [5, 10, 15]], "ab"):
        for i, m in enumerate(MODELS):
            vals = [cond_rates(sel(df, L, d, m))["relapse3"] for L in leads]
            x = np.arange(len(leads)) + (i - 2.5) * w
            ax.bar(x, vals, w, color=COLOR[m])
        ax.set_xticks(np.arange(len(leads)), [str(L) for L in leads])
        ax.set_xlabel("Lead (frames)")
        ax.set_ylim(0, 0.8)
        ax.set_title(TARGET_LABEL[d].capitalize() if d == 0 else TARGET_LABEL[d], loc="left")
        panel_label(ax, s)
    axes[0].set_ylabel("Relapse rate, 3-frame rule\n(share of recoveries)")
    axes[1].set_yticklabels([])
    ax = axes[2]
    V = sim_relapse_validation()
    labs = list(V)
    x = np.arange(len(labs))
    for k, off, c in [(1, -0.18, "#c9ced6"), (3, 0.18, ACCENT)]:
        fr = [V[l][k][0] for l in labs]
        ax.bar(x + off, fr, 0.34, color=c, label="1 frame" if k == 1 else "3 frames")
        for xi, l in zip(x, labs):
            ax.text(xi + off, V[l][k][0] + 0.004, f"{V[l][k][0]:.1%}", ha="center", fontsize=5.6, color=INK2,
                    rotation=90, va="bottom")
    ax.set_xticks(x, labs, fontsize=6.2)
    ax.set_ylim(0, 0.16)
    ax.set_yticks([0, 0.05, 0.1, 0.15], ["0", "5", "10", "15%"])
    ax.set_xlabel("Noise floor assumed")
    ax.set_ylabel("False relapses\n(truly ahead after recovery)")
    ax.set_title("Validation: known truth", loc="left")
    ax.legend(title="Rule", title_fontsize=6.5, fontsize=6.5, loc="upper right")
    sens = ", ".join(f"{V[l][3][2]:.0%}" for l in labs)
    panel_label(ax, "c")
    model_legend(fig, MODELS, oracle=False, y=1.12, ncol=6)
    note(fig, 0.5, -0.14, "a–b: Eindhoven, 4.2 nats. Relapse = three consecutive frames with P(I < 0) ≥ 0.95 starting "
             "within 10 frames of recovery; lead 1 has the same margin for every target.",
             ha="center", fontsize=6.3, color=INK3)
    note(fig, 0.5, -0.19, f"c: simulated tracks, lead 5, 4.2 nats; the 3-frame rule still catches {sens} of real "
             "setbacks (three truly-worse frames) in the same window.", ha="center", fontsize=6.3, color=INK3)
    save(fig, out, "fig11_relapse")


def ditl_panel(out, d, name, emphasize=None):
    fig = plt.figure(figsize=(FULL_W, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.12, height_ratios=[1.15, 1])
    for j, lead in enumerate([5, 10, 15]):
        ax = fig.add_subplot(gs[0, j])
        km_lines(ax, "ditl", lead, d, emphasize=emphasize)
        ax.set_title(f"Lead {lead} ({lead / HZ:g} s)", loc="left", pad=16)
        seconds_axis(ax)
        ax.set_xlabel("Frames since event")
        if j == 0:
            ax.set_ylabel("Share not yet recovered\n(given onset)")
        else:
            ax.set_yticklabels([])
        panel_label(ax, "abc"[j], y=1.12)
        ax2 = fig.add_subplot(gs[1, j])
        outcome_bars(ax2, "ditl", lead, d, MODELS, show_labels=j == 0)
        ax2.set_title(f"Lead {lead}", loc="left")
        ax2.set_xlabel("Share of events")
        panel_label(ax2, "def"[j])
    model_legend(fig, MODELS, y=1.0, ncol=7)
    outcome_legend(fig, 0.47)
    fig.text(0.5, 0.995 + 0.03, f"Day in the Life (retail): learned models and IMM fine-tuned, CTRV hand-set; 4.2 nats, {TARGET_LABEL[d]}",
             ha="center", fontsize=8, color=INK2)
    save(fig, out, name)


def fig12(out):
    ditl_panel(out, 0.0, "fig12_ditl_replication")


def fig13(out):
    S = summary("eindhoven")
    N = summary("eindhoven_nofloor")
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.3), sharex=True, gridspec_kw=dict(hspace=0.35, wspace=0.3))
    for j, lead in enumerate([5, 10, 15]):
        for m in MODELS + ["oracle_B"]:
            g = S[(S["lead"] == lead) & np.isclose(S["distortion"], 0.0) & (S["model"] == m)].sort_values("info")
            cens = g["censored"] / (g["recovered"] + g["censored"])
            ls = (0, (4, 2)) if m == "oracle_B" else "-"
            axes[0, j].plot(g["info"], g["km_react"], ls=ls, marker="o", ms=3.5, color=COLOR[m], lw=1.3)
            axes[1, j].plot(g["info"], cens, ls=ls, marker="o", ms=3.5, color=COLOR[m], lw=1.3)
            n = N[(N["lead"] == lead) & np.isclose(N["distortion"], 0.0) & (N["model"] == m)]
            if len(n):
                axes[0, j].plot(n["info"] + 0.12, n["km_react"], marker="o", ms=4, mfc="white", color=COLOR[m], ls="")
                axes[1, j].plot(n["info"] + 0.12, n["censored"] / (n["recovered"] + n["censored"]), marker="o", ms=4,
                                mfc="white", color=COLOR[m], ls="")
        axes[0, j].set_title(f"Lead {lead}", loc="left")
        axes[1, j].set_xlabel("Assumed information (nats)")
        axes[1, j].set_xticks([3, 4.2, 6])
        axes[1, j].set_ylim(-0.02, 0.4)
    axes[0, 0].set_ylabel("Median REACT\ngiven onset (frames)")
    axes[1, 0].set_ylabel("Censored,\ngiven onset")
    for a, s in zip(axes.flat, "abcdef"):
        panel_label(a, s)
    extra = [Line2D([], [], color=INK2, marker="o", mfc="white", ls="", label="no measurement-noise floor (4.2)")]
    model_legend(fig, MODELS, y=1.05, ncol=4, extra=extra)
    save(fig, out, "fig13_info_sensitivity")


def fig14(out):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True, gridspec_kw=dict(wspace=0.1))
    st = {0.0: ("-", "climatology margin"), 0.3: ((0, (1, 1.5)), "0.3 m"), 0.1: ((0, (4, 2)), "0.1 m")}
    for ax, run, s in zip(axes, ["eindhoven", "ditl"], "ab"):
        df = records(run)
        for d, (ls, lab) in st.items():
            leads = [1, 5, 10, 15]
            v = [cond_rates(sel(df, L, d, "oracle_B")) for L in leads]
            ax.plot(leads, [x["censored"] for x in v], ls=ls, marker="o", ms=4, color=INK, lw=1.3, label=lab)
            for L, x in zip(leads, v):
                ax.annotate(f"{x['median']:.0f}", (L, x["censored"]), xytext=(0, 5), textcoords="offset points",
                            fontsize=6, color=INK3, ha="center")
        ax.set_xticks([1, 5, 10, 15])
        ax.set_xlabel("Lead (frames)")
        ax.set_title(DATASET_LABEL[run], loc="left")
        panel_label(ax, s)
    axes[0].set_ylabel("Oracle B never confirmed better")
    axes[0].legend(title="Target", title_fontsize=7, loc="upper left")
    fig.text(0.5, -0.1, "Small numbers: Oracle B's median REACT (frames), the earliest any forecast could be confirmed. "
             "4.2 nats with floor.", ha="center", fontsize=6.3, color=INK3)
    save(fig, out, "fig14_oracle_reachability")


def fig15(out):
    Q = json.load(open(RUNS["eindhoven"] / "qsweep_lead5_info4.2.json"))
    R = pd.DataFrame(Q["results"])
    panels = [("anticipation", "Anticipation rate"), ("react_median_uncensored", "Median REACT,\nrecovered events only (frames)"),
              ("censored", "Censored, share of all events"),
              ("pre_gain", "Pre-event gain over\nclimatology (nats/frame)")]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.2), sharex=True, gridspec_kw=dict(hspace=0.3, wspace=0.3))
    for ax, (k, lab), s in zip(axes.flat, panels, "abcd"):
        for m, g in R.groupby("model"):
            g = g.sort_values("q")
            qs = Q["qstar"][m]
            ax.plot(g["q"] / qs, g[k], "-o", ms=3, color=COLOR[m], lw=1.3)
            g1 = g.iloc[np.argmin(np.abs(g["q"] - 1.0))]
            ax.plot(g1["q"] / qs, g1[k], "s", ms=5.5, mfc="white", mec=COLOR[m], mew=1.3, zorder=4)
        ax.axvline(1, color=INK3, lw=0.8, ls=(0, (2, 2)))
        ax.set_xscale("log")
        ax.set_ylabel(lab)
        panel_label(ax, s)
    for ax in axes[1]:
        ax.set_xlabel("Covariance scale relative to best-calibrated $q^*$")
    ms = [m for m in Q["qstar"]]
    extra = [Line2D([], [], marker="s", mfc="white", mec=INK2, ls="", label="as issued ($q$ = 1)")]
    fig.legend(handles=[Line2D([], [], color=COLOR[m], lw=2, label=NAME[m]) for m in ms] + extra,
               loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=5)
    fig.text(0.5, -0.02, "Lead 5, 4.2 nats, climatology margin. Trajectron++ enters this sweep as its moment-matched "
             "Gaussian.", ha="center", fontsize=6.3, color=INK3)
    save(fig, out, "fig15_calibration_qsweep")


def fig16(out, lead=10):
    df = sel(records("eindhoven"), lead, 0.0)
    df = df.assign(kind=df["event_type"].map(TYPE_GROUP).fillna("Other"))
    M = np.full((len(MODELS), len(TYPE_ORDER)), np.nan)
    Cn = np.zeros_like(M)
    Ce = np.zeros_like(M)
    for i, m in enumerate(MODELS):
        for j, k in enumerate(TYPE_ORDER):
            g = df[(df["model"] == m) & (df["kind"] == k)]
            M[i, j] = km_median(km(g))
            Cn[i, j] = len(onset_set(g))
            Ce[i, j] = cond_rates(g)["censored"] if len(onset_set(g)) else np.nan
    fig, ax = plt.subplots(figsize=(FULL_W * 0.78, 2.9))
    ax.grid(False)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("react", ["#f4f6f9", "#a9c1da", ACCENT])
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=np.nanmax(M))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = "—" if not np.isfinite(v) else f"{v:.0f}"
            ax.text(j, i - 0.12, txt, ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(v) and v > 0.6 * np.nanmax(M) else INK, fontweight="medium")
            ax.text(j, i + 0.27, f"{Ce[i, j]:.0%} cens.", ha="center", va="center", fontsize=5.5,
                    color="white" if np.isfinite(v) and v > 0.6 * np.nanmax(M) else INK3)
    ax.set_xticks(range(len(TYPE_ORDER)), TYPE_ORDER)
    ax.set_yticks(range(len(MODELS)), [NAME[m] for m in MODELS])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    n_ev = df[df["model"] == "oracle_B"]["kind"].value_counts().reindex(TYPE_ORDER)
    ax.set_xticks(range(len(TYPE_ORDER)), [f"{k}\n(n = {n:,})" for k, n in zip(TYPE_ORDER, n_ev)], fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Median REACT given onset (frames)", fontsize=7)
    cb.outline.set_visible(False)
    ax.set_title(f"Eindhoven, lead {lead}, 4.2 nats, climatology margin", loc="left", fontsize=8, color=INK2)
    save(fig, out, "fig16_event_type")


KINEMATIC = ["ConstantVelocity", "CTRV", "IMM"]
LEARNED_MODELS = ["GRUPosition", "GRUOrientation", "Trajectron_mix"]
ADV_COLS = [  # (label, key, higher is better, unit scale, unit)
    ("Error at\nlead", "ade", False, 100, "cm"), ("NLL at\nlead", "nll", False, 1, "nats"), ("Anticipation", "anticipation", True, 100, "pp"),
    ("Recovered\n≤ 10 fr.", "by10", True, 100, "pp"), ("Recovered\n≤ 15 fr.", "by15", True, 100, "pp"),
    ("Recovered\n≤ 30 fr.", "by30", True, 100, "pp"), ("Censored", "censored", False, 100, "pp"),
    ("Relapse\n(3 frames)", "relapse3", False, 100, "pp"),
]


def advantage_rows():
    rows = []
    for run in ["eindhoven", "ditl"]:
        A = accuracy(run)
        df = records(run)
        for d in [0.0, 0.1]:
            for lead in ([1, 5, 10, 15] if d == 0 else [5, 10, 15]):
                vals = {}
                for m in MODELS:
                    c = cond_rates(sel(df, lead, d, m))
                    c["ade"] = A[m][f"err{lead}"]  # error of the forecast mean at this lead
                    c["nll"] = -A[m][f"ll{lead}"]
                    vals[m] = c
                rows.append((run, d, lead, vals))
    return rows


def advantage_map(out, name, challenger, title):
    """Cells: challenger minus the best kinematic model, signed so blue = challenger better."""
    rows = advantage_rows()
    M = np.full((len(rows), len(ADV_COLS)), np.nan)
    thin = np.zeros_like(M, bool)  # a compared model fell behind on fewer than 100 events
    for i, (_, _, _, vals) in enumerate(rows):
        for j, (_, k, hi, scale, _) in enumerate(ADV_COLS):
            pick = max if hi else min
            km_ = pick(KINEMATIC, key=lambda m: vals[m][k])
            ch_ = pick(challenger, key=lambda m: vals[m][k])
            kin, ch = vals[km_][k], vals[ch_][k]
            M[i, j] = (ch - kin if hi else kin - ch) * scale
            if k not in ("ade", "nll", "anticipation"):
                thin[i, j] = min(vals[km_]["n_onset"], vals[ch_]["n_onset"]) < 100
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("adv", ["#b5482a", "#f2d3c9", "#f7f7f7", "#c3d4e6", ACCENT])
    sat = np.array([{"cm": 5.0, "nats": 1.5, "pp": 25.0}[c[4]] for c in ADV_COLS])
    norm = np.clip(M / sat, -1, 1)
    fig, ax = plt.subplots(figsize=(FULL_W, 5.6))
    ax.grid(False)
    ax.imshow(norm, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            unit = ADV_COLS[j][4]
            v = round(M[i, j], 2 if unit == "nats" else (1 if unit == "cm" else 0)) + 0.0
            txt = f"{v:+.2f}" if unit == "nats" else (f"{v:+.1f}" if unit == "cm" else f"{v:+.0f}")
            ax.text(j, i, txt + ("†" if thin[i, j] else ""), ha="center", va="center", fontsize=6.6,
                    color="white" if abs(norm[i, j]) > 0.62 else INK)
    ax.set_xticks(range(len(ADV_COLS)), [f"{lab}\n({u})" for lab, _, _, _, u in ADV_COLS], fontsize=6.6)
    ax.xaxis.tick_top()
    ylab = [f"{DATASET_LABEL[r][:9] if r == 'eindhoven' else 'DitL'} · {'clim.' if d == 0 else '0.1 m'} · lead {L}"
            for r, d, L, _ in rows]
    ax.set_yticks(range(len(rows)), ylab, fontsize=6.8)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for y in [3.5, 6.5, 10.5]:
        ax.axhline(y, color="white", lw=2.5 if y == 6.5 else 1.2)
    ax.axvline(1.5, color="white", lw=2.5)
    ax.text(0.5, -1.9, "classical", ha="center", fontsize=7, color=INK2, fontweight="medium")
    ax.text(4.5, -1.9, "REACT, 4.2 nats (recovery, censoring and relapse given onset)", ha="center", fontsize=7,
            color=INK2, fontweight="medium")
    fig.text(0.5, 0.035, f"{title}. Blue: learned side better; red: best kinematic model better. Each cell compares "
             "the best model on each side for that measure.", ha="center", fontsize=6.3, color=INK3)
    fig.text(0.5, 0.012, "Shading saturates at ±5 cm, ±1.5 nats, ±25 pp (percentage points). † a compared model fell "
             "behind on fewer than 100 events (Day in the Life lead 1: CV anticipates 99%).", ha="center", fontsize=6.3,
             color=INK3)
    save(fig, out, name)


def fig18(out):
    advantage_map(out, "fig18_advantage_map_learned", LEARNED_MODELS,
                  "Best learned model (GRU position, GRU heading, Trajectron++) minus best of CV, CTRV and IMM, per cell")


def fig19(out):
    advantage_map(out, "fig19_advantage_map_trajectron", ["Trajectron_mix"],
                  "Trajectron++ minus best of CV, CTRV and IMM, per cell")


def fig20(out):
    ditl_panel(out, 0.1, "fig20_ditl_react_at_0p1", emphasize="Trajectron_mix")


SIM_MODELS = ["oracle", "lag5", "lag20", "over", "under", "CV"]
SIM_NAME = {"oracle": "Oracle", "lag5": "Oracle, 5 frames late", "lag20": "Oracle, 20 frames late",
            "over": "Overconfident oracle", "under": "Underconfident oracle", "CV": "Constant velocity"}
SIM_COLOR = {"oracle": INK, "lag5": "#5b7fa6", "lag20": "#9db3cc", "over": "#8c6bb1", "under": "#6b8f71",
             "CV": "#c98a00"}


SIM_SHORT = {"oracle": "Oracle", "lag5": "Late 5 fr.", "lag20": "Late 20 fr.", "over": "Overconfident",
             "under": "Underconfident", "CV": "CV"}


def fig17(out, sig="18.2"):
    names = SIM_SHORT if PAPER else SIM_NAME
    path = SIM / f"rec_I4.2_s{sig}.pkl"
    if not path.exists():
        print("skip fig17: simulation records missing at", path)
        return
    R = pickle.load(open(path, "rb"))
    logk = np.log(0.95 / 0.05)
    sig_ = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, ht("fig17", 2.7)), gridspec_kw=dict(wspace=0.95, width_ratios=[1, 1.25, 0.9]))
    # (a) reliability of the recovery test, pooled over forecasters
    p = np.concatenate([sig_(r["lo_rec"].astype(float)) for r in R])
    yv = np.concatenate([r["trueI"] > 0 for r in R])
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    pm = [p[idx == b].mean() for b in range(10)]
    om = [yv[idx == b].mean() for b in range(10)]
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=RULE, lw=1)
    ax.plot(pm, om, "-o", color=ACCENT, ms=3.5)
    ax.set_xlabel(r"$P(I_t > 0 \mid \mathrm{data})$")
    ax.set_ylabel(r"Observed share with true $I_t > 0$")
    ax.set_title("Recovery test is conservative", loc="left")
    panel_label(ax, "a")
    # (b) median true advantage over the first 60 frames after the event (shown while >= 30% of events remain)
    ax = axes[1]
    for m in SIM_MODELS:
        rs = [r for r in R if r["model"] == m]
        I = np.full((len(rs), 61), np.nan)
        for i, r in enumerate(rs):
            rel = r["rel"].astype(int)
            ok = (rel >= 0) & (rel <= 60)
            I[i, rel[ok]] = r["trueI"][ok]
        keep = np.mean(np.isfinite(I), 0) >= 0.3
        med = np.where(keep, np.nanmedian(I, 0), np.nan)
        ax.plot(np.arange(61), np.clip(med, -6, None), color=SIM_COLOR[m], lw=1.4)
    ax.axhline(0, color=INK3, lw=0.8)
    ax.set_ylim(-6.3, 5)
    ax.text(59, -6.1, "clipped at -6", ha="right", va="bottom", fontsize=6, color=INK3)
    ax.set_xlabel("Frames since event")
    ax.set_ylabel("Median true advantage $I_t$ (nats)")
    ax.set_title("Known truth", loc="left")
    panel_label(ax, "b")
    # (c) REACT measured vs true
    ax = axes[2]
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for k, m in enumerate(SIM_MODELS[::-1]):
        est, tru = [], []
        for r in (r for r in R if r["model"] == m):
            rel = r["rel"].astype(int)
            I = r["trueI"]
            on = np.flatnonzero((r["lo_on"][0.0] >= logk) & (rel <= 10))
            if len(on):
                rr = np.flatnonzero((r["lo_rec"] >= logk) & (np.arange(len(rel)) > on[0]))
                est.append(rel[rr[0]] if len(rr) else np.nan)
            ton = np.flatnonzero((I < 0) & (rel <= 10))
            if len(ton):  # true recovery: first later frame truly ahead for 10 frames (or to the window end), as ana.py
                t_rec = np.nan
                for j in range(ton[0] + 1, len(I)):
                    if I[j] > 0 and np.all(I[j:j + 10] > 0):
                        t_rec = rel[j]
                        break
                tru.append(t_rec)
        est, tru = np.array(est, float), np.array(tru, float)
        e = np.nanmedian(est) if np.isfinite(est).any() else np.nan
        t = np.nanmedian(tru) if np.isfinite(tru).any() else np.nan
        ax.plot([t, e], [k, k], color=RULE, lw=2, zorder=1)
        ax.plot(t, k, "o", mfc="white", color=SIM_COLOR[m], ms=5, mew=1.2, zorder=2)
        ax.plot(e, k, "o", color=SIM_COLOR[m], ms=5, zorder=3)
        ax.text(31, k - 0.32, f"confirmed {np.isfinite(est).sum()}/{len(est)} · true {np.isfinite(tru).sum()}/{len(tru)}",
                ha="right", va="center", fontsize=5.2, color=INK3)
    ax.set_xlim(0, 31)
    ax.set_ylim(-0.75, len(SIM_MODELS) - 0.5)
    ax.set_yticks(range(len(SIM_MODELS)), [names[m] for m in SIM_MODELS[::-1]], fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Median recovery frame (recovered events)")
    ax.set_title("REACT (filled) vs truth (open)", loc="left")
    panel_label(ax, "c")
    h = [Line2D([], [], color=SIM_COLOR[m], lw=2, label=names[m]) for m in SIM_MODELS]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.12 if PAPER else 1.1), ncol=6 if PAPER else 3)
    note(fig, 0.5, -0.12, f"Simulated tracks, lead 5, 4.2 nats; REACT assumes a noise floor of {float(sig) / 10:.2f} cm "
             "(estimated; the simulator's true noise is 2.37 cm). a: all frames, pooled over forecasters.",
             ha="center", fontsize=6.3, color=INK3)
    note(fig, 0.5, -0.17, "c: truth = first frame with true $I_t$ < 0 in the onset window, then the first frame truly ahead "
             "for 10 frames. Small text: recovered / fell behind. The oracle is rarely truly behind.",
             ha="center", fontsize=6.3, color=INK3)
    save(fig, out, "fig17_known_truth")


# ---------- compact figures for the manuscript (text width 5.9 in, shown at 100%) ----------
def p_km(out, run="eindhoven", d=0.0, name="paper_fig04_recovery"):
    """Recovery given onset: full 15 s window (top) and the first 4 s (bottom), leads 1/5/10/15."""
    leads = [1, 5, 10, 15]
    fig, axes = plt.subplots(2, 4, figsize=(FULL_W, 3.2), sharey=True, gridspec_kw=dict(hspace=0.32, wspace=0.1))
    for j, lead in enumerate(leads):
        km_lines(axes[0, j], run, lead, d, t_max=100, all_events=True)
        axes[0, j].set_title(f"Lead {lead} ({lead / HZ:g} s)", loc="left", fontsize=7.5)
        axes[0, j].set_xticks([0, 25, 50, 75])
        ax = axes[1, j]
        ax.axvspan(0, lead + 5, color=PANEL, zorder=0)
        km_lines(ax, run, lead, d, t_max=40)
        ax.axvline(15, color=INK3, lw=0.7, ls=(0, (2, 2)))
        ax.set_xticks([0, 10, 20, 30])
        ax.set_xlabel("Frames since event")
    axes[0, 0].set_ylabel("Not yet recovered\n(all events, 0–10 s)")
    axes[1, 0].set_ylabel("Not yet recovered\n(given onset, 0–4 s)")
    panel_label(axes[0, 0], "a")
    panel_label(axes[1, 0], "b")
    model_legend(fig, MODELS, y=1.01, ncol=7)
    save(fig, out, name)


def p_react01(out, name="paper_fig05_react_at_0p1"):
    """REACT@0.1 m on Eindhoven, Trajectron++ emphasised."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    df = records("eindhoven")
    for ax, lead, s_ in zip(axes, [5, 10, 15], "abc"):
        km_lines(ax, "eindhoven", lead, 0.1, emphasize="Trajectron_mix")
        ax.set_title(f"Lead {lead} ({lead / HZ:g} s)", loc="left", fontsize=7.5)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Frames since event")
        g = sel(df, lead, 0.1, "Trajectron_mix")
        ax.text(T_MAX - 3, 0.97, f"Trajectron++ censored {cond_rates(g)['censored']:.0%}", ha="right", va="top",
                fontsize=6.3, color=COLOR["Trajectron_mix"])
    axes[0].set_ylabel("Not yet recovered\n(given onset)")
    model_legend(fig, MODELS, y=1.1, ncol=7)
    save(fig, out, name)


def p_advantage(out, name="paper_fig08_advantage"):
    """Best learned model (a) and Trajectron++ (b) minus the best kinematic model, side by side."""
    keep = [0, 1, 2, 3, 5, 6, 7]  # drop "recovered <= 15" to fit both maps
    cols = [ADV_COLS[k] for k in keep]
    rows = advantage_rows()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("adv", ["#b5482a", "#f2d3c9", "#f7f7f7", "#c3d4e6", ACCENT])
    sat = np.array([{"cm": 5.0, "nats": 1.5, "pp": 25.0}[c[4]] for c in cols])
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.5), gridspec_kw=dict(wspace=0.04))
    for ax, challenger, title, s_ in zip(axes, [LEARNED_MODELS, ["Trajectron_mix"]],
                                         ["Best learned model", "Trajectron++"], "ab"):
        M = np.full((len(rows), len(cols)), np.nan)
        thin = np.zeros_like(M, bool)
        for i, (_, _, _, vals) in enumerate(rows):
            for j, (_, k, hi, scale, _) in enumerate(cols):
                pick = max if hi else min
                km_ = pick(KINEMATIC, key=lambda m: vals[m][k])
                ch_ = pick(challenger, key=lambda m: vals[m][k])
                M[i, j] = (vals[ch_][k] - vals[km_][k] if hi else vals[km_][k] - vals[ch_][k]) * scale
                if k not in ("ade", "nll", "anticipation"):
                    thin[i, j] = min(vals[km_]["n_onset"], vals[ch_]["n_onset"]) < 100
        norm = np.clip(M / sat, -1, 1)
        ax.grid(False)
        ax.imshow(norm, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                unit = cols[j][4]
                v = round(M[i, j], 2 if unit == "nats" else (1 if unit == "cm" else 0)) + 0.0
                txt = f"{v:+.2f}" if unit == "nats" else (f"{v:+.1f}" if unit == "cm" else f"{v:+.0f}")
                ax.text(j, i, txt + ("†" if thin[i, j] else ""), ha="center", va="center", fontsize=5.6,
                        color="white" if abs(norm[i, j]) > 0.62 else INK)
        short = {"ade": "Error\n(cm)", "nll": "NLL\n(nats)", "anticipation": "Antic.\n(pp)", "by10": "Rec.\n≤10 fr",
                 "by30": "Rec.\n≤30 fr", "censored": "Cens.\n(pp)", "relapse3": "Relap.\n(pp)"}
        ax.set_xticks(range(len(cols)), [short[c[1]] for c in cols], fontsize=6, rotation=0)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for y in [3.5, 6.5, 10.5]:
            ax.axhline(y, color="white", lw=2.2 if y == 6.5 else 1.0)
        ax.axvline(1.5, color="white", lw=2.2)
        ax.set_title(f"{s_}  {title} − best kinematic", loc="left", fontsize=7.5, pad=24)
        if s_ == "a":
            ax.set_yticks(range(len(rows)), [f"{'Eind.' if r == 'eindhoven' else 'DitL'} {'clim.' if d == 0 else '0.1 m'} "
                                             f"L{L}" for r, d, L, _ in rows], fontsize=6.3)
        else:
            ax.set_yticks([])
    save(fig, out, name)


def p_qsweep(out, name="paper_fig09_qsweep", lead=10):
    Q = json.load(open(RUNS["eindhoven"] / f"qsweep_lead{lead}_info4.2.json"))
    R_ = pd.DataFrame(Q["results"])
    panels = [("anticipation", "Anticipation"), ("react_median_uncensored", "Median REACT (fr.)"),
              ("censored", "Censored (all events)"), ("pre_gain", "Pre-event gain (nats)")]
    fig, axes = plt.subplots(1, 4, figsize=(FULL_W, 1.75), gridspec_kw=dict(wspace=0.35))
    for ax, (k, lab), s_ in zip(axes, panels, "abcd"):
        for m, g in R_.groupby("model"):
            g = g.sort_values("q")
            qs = Q["qstar"][m]
            ax.plot(g["q"] / qs, g[k], "-o", ms=2.2, color=COLOR[m], lw=1.1)
            g1 = g.iloc[np.argmin(np.abs(g["q"] - 1.0))]
            ax.plot(g1["q"] / qs, g1[k], "s", ms=4, mfc="white", mec=COLOR[m], mew=1.1, zorder=4)
        ax.axvline(1, color=INK3, lw=0.7, ls=(0, (2, 2)))
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2], ["0.5", "1", "2"])
        ax.minorticks_off()
        ax.set_xlabel("$q/q^*$")
        ax.set_title(lab.replace("\n", " "), loc="left", fontsize=6.5)
        panel_label(ax, s_, y=1.12)
    ms = list(Q["qstar"])
    extra = [Line2D([], [], marker="s", mfc="white", mec=INK2, ls="", label="as issued")]
    fig.legend(handles=[Line2D([], [], color=COLOR[m], lw=2, label=NAME[m].replace(" (moment-matched)", " (Gaussian)"))
                        for m in ms] + extra, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=5)
    save(fig, out, name)


def p_ditl(out, name="paper_fig11_ditl"):
    """Day in the Life: recovery given onset under the climatology margin (top) and REACT@0.1 m (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 3.3), sharey=True, sharex=True, gridspec_kw=dict(hspace=0.3, wspace=0.08))
    for i, d in enumerate([0.0, 0.1]):
        for j, lead in enumerate([5, 10, 15]):
            ax = axes[i, j]
            km_lines(ax, "ditl", lead, d, emphasize="Trajectron_mix" if d else None)
            if i == 0:
                ax.set_title(f"Lead {lead} ({lead / HZ:g} s)", loc="left", fontsize=7.5)
            else:
                ax.set_xlabel("Frames since event")
            ax.set_xticks([0, 50, 100])
        axes[i, 0].set_ylabel(("Climatology margin" if d == 0 else "REACT@0.1 m") + "\nnot yet recovered")
    panel_label(axes[0, 0], "a")
    panel_label(axes[1, 0], "b")
    model_legend(fig, MODELS, y=1.01, ncol=7)
    save(fig, out, name)


PAPER_HEIGHT = {"fig01": 2.25, "fig02": 1.8, "fig03": 2.5, "fig08": 2.35, "fig09": 2.05, "fig11": 1.6, "fig17": 1.9}
PAPER_NAMES = {
    "fig01_react_schematic": "paper_fig01_schematic", "fig02_event_corpus": "paper_fig02_events",
    "fig03_classical_accuracy": "paper_fig03_accuracy",
    "fig11_relapse": "paper_fig06_relapse", "fig08_rank_discordance": "paper_fig07_ranks",
    "fig17_known_truth": "paper_fig10_validation",
}


def paper_set(out):
    global FULL_W, PAPER
    FULL_W, PAPER = 5.9, True
    NAME.update({"GRUPosition": "GRU (pos)", "GRUOrientation": "GRU (head)"})
    HEIGHT.update(PAPER_HEIGHT)
    RENAME.update(PAPER_NAMES)
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "lines.linewidth": 1.3})
    for f in (fig01, fig02, fig03, p_km, p_react01, fig11, fig08, p_advantage, p_qsweep, fig17, p_ditl):
        f(out)


FIGS = {f"fig{i:02d}": f for i, f in enumerate(
    [fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12, fig13, fig14, fig15, fig16,
     fig17, fig18, fig19, fig20], start=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(FINAL / "figures"))
    ap.add_argument("--paper", action="store_true", help="compact manuscript set (5.9 in text width)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_theme()
    if a.paper:
        (out / "paper").mkdir(exist_ok=True)
        paper_set(out / "paper")
        return
    for k in (a.only.split(",") if a.only else FIGS):
        FIGS[k](out)


if __name__ == "__main__":
    main()
