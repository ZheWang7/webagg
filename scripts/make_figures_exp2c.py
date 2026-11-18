"""Exp 2-C paper figures, drawn from the result TABLES only (no ledgers,
no network): data/experiments/exp2c/results/{calibration,stops,shortfall,
grades}.csv -> fig_c1_calibration, fig_c2_coverage, fig_shortfall
(PDF for LaTeX, PNG for slides), always on a white background.

  Fig C1  claimed bound on the unseen share (U_hat + psi, webaggr.pdf
          Theorem 1) against the share actually unseen, along every run.
          A point BELOW the diagonal would be a bound that was too
          optimistic.
  Fig C2  per paper: the recall the certificate promised (1 - eps at the
          tightest pre-registered arm where it fires) against the recall
          measured against the withheld oracle.
  Fig S   truth citers never found, per paper (none was lost by stopping
          early in this experiment).

The notebook imports these same functions, so the figures it shows are
the paper figures.

Usage:
    python scripts/make_figures_exp2c.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                      # noqa: E402

RES = config.DATA_DIR / "experiments" / "exp2c" / "results"

# validated categorical pair (colorblind-safe on a light surface)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# short display names for the long paper titles (prefix of the stored key)
SHORT = {"clinical course": "COVID-19 clinical course",
         "edger": "edgeR", "kegg": "KEGG", "plink": "PLINK",
         "standards and guidelines": "Variant-interpretation guidelines"}


def short(name: str) -> str:
    for prefix, label in SHORT.items():
        if name.startswith(prefix):
            return label
    return name[:32]


def read(name: str) -> list[dict]:
    path = RES / name
    if not path.exists():
        raise SystemExit(f"{path} missing — run replay_exp2c.py and "
                         "grade_exp2c.py first.")
    return list(csv.DictReader(open(path, encoding="utf-8")))


def style() -> None:
    """Light, print-safe defaults regardless of the IDE's theme."""
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False})


def save(fig, stem: str, close: bool = True) -> None:
    """Write PDF + PNG. close=False leaves the figure open so a notebook
    can display it after saving."""
    RES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(RES / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    if close:
        plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def fig_c1(cal: list[dict], live_eps: float):
    runs = defaultdict(list)
    for r in cal:
        runs[r["player"]].append((int(r["step"]), float(r["realized_unseen"]),
                                  float(r["claimed_unseen"]),
                                  min(float(r["raw_U_hat"]), 1.0)))
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(0.80, 0.76, "claimed = actual", color=MUTED, fontsize=8,
            rotation=45, ha="center", va="center")
    ax.axhline(live_eps, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.text(0.99, live_eps + 0.012, f"stop threshold ε = {live_eps}",
            color=MUTED, fontsize=8, ha="right")
    for i, (_name, pts) in enumerate(sorted(runs.items())):
        pts.sort()
        x = [p[1] for p in pts]
        ax.plot(x, [p[3] for p in pts], color=BLUE, lw=1.4, alpha=0.9,
                label="estimate $\\hat{U}$ (capped at 1)" if i == 0 else None)
        ax.plot(x, [p[2] for p in pts], color=ORANGE, lw=1.8,
                label="certified bound $\\hat{U}+\\psi$" if i == 0 else None)
        # the end of each run: filled = certificate fired at the live eps
        fired = pts[-1][2] < live_eps
        ax.scatter([x[-1]], [pts[-1][2]], s=42, zorder=4, color=ORANGE,
                   facecolor=ORANGE if fired else "white", linewidth=1.6)
    ax.scatter([], [], s=42, color=ORANGE, label="run end: certified")
    ax.scatter([], [], s=42, color=ORANGE, facecolor="white", linewidth=1.6,
               label="run end: not certified")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("share of true citers actually unseen (1 − recall)")
    ax.set_ylabel("claimed share unseen")
    ax.legend(loc="lower right", fontsize=8.5)
    return fig


def fig_c2(stops: list[dict]):
    # tightest pre-registered arm at which each paper's certificate fires
    best = {}
    for r in stops:
        if r["stop_kind"] != "statistical":
            continue
        eps = float(r["eps"])
        if r["player"] not in best or eps < best[r["player"]][0]:
            best[r["player"]] = (eps, float(r["recall_at_stop"]))
    names = sorted(best, key=lambda p: (best[p][0], -best[p][1]), reverse=True)
    fig, ax = plt.subplots(figsize=(6.2, 0.55 * len(names) + 1.3))
    for y, p in enumerate(names):
        eps, rec = best[p]
        ax.plot([1 - eps, rec], [y, y], color=GRID, lw=3, zorder=1,
                solid_capstyle="round")
        ax.scatter([1 - eps], [y], s=60, color=BLUE, zorder=3,
                   label="recall promised (1 − ε)" if y == 0 else None)
        ax.scatter([rec], [y], s=60, color=ORANGE, zorder=3,
                   label="recall measured against the oracle" if y == 0 else None)
        ax.text(1 - eps - 0.02, y, f"ε = {eps:g}", ha="right", va="center",
                fontsize=8.5, color=MUTED)
        ax.text(rec + 0.02, y, f"{rec:.1%}", ha="left", va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([short(p) for p in names], color=INK)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("recall")
    ax.grid(axis="y", visible=False)
    if names:
        ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2,
                  fontsize=8.5)
    else:
        ax.text(0.5, 0.5, "no certificate fired at any arm", ha="center",
                va="center", transform=ax.transAxes, color=MUTED)
    return fig


def fig_shortfall(short_rows: list[dict], live_eps: float):
    rows = [r for r in short_rows if abs(float(r["eps"]) - live_eps) < 1e-9]
    lost = sum(int(r["lost_to_stopping"]) for r in rows)
    rows.sort(key=lambda r: int(r["never_discovered"]) / int(r["n_truth"]))
    fig, ax = plt.subplots(figsize=(6.2, 0.5 * len(rows) + 1.1))
    for y, r in enumerate(rows):
        n, t = int(r["never_discovered"]), int(r["n_truth"])
        ax.barh(y, 100 * n / t, height=0.55, color=BLUE, zorder=2)
        ax.text(100 * n / t + 0.12, y, f"{100 * n / t:.1f}%  ({n:,} of {t:,})",
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([short(r["player"]) for r in rows], color=INK)
    top = max((100 * int(r["never_discovered"]) / int(r["n_truth"])
               for r in rows), default=0.0)
    ax.set_xlim(0, max(top * 1.55, 1.0))
    ax.set_xlabel("true citers never found (% of the oracle's list)")
    ax.grid(axis="y", visible=False)
    if lost:
        ax.set_title(f"{lost:,} further citers were found only after the stop",
                     fontsize=9, loc="left")
    return fig


def main() -> None:
    plt.switch_backend("Agg")                    # files only, no window
    style()
    stops = read("stops.csv")
    live_eps = min(float(r["eps"]) for r in stops)   # the strictest arm
    print(f"figures from {RES} (live eps = {live_eps})")
    save(fig_c1(read("calibration.csv"), live_eps), "fig_c1_calibration")
    save(fig_c2(stops), "fig_c2_coverage")
    save(fig_shortfall(read("shortfall.csv"), live_eps), "fig_shortfall")


if __name__ == "__main__":
    main()
