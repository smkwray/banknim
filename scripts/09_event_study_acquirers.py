from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from nimscale.bank_panel import pick_first_existing, standardize_columns
from nimscale.io import ensure_dir
from nimscale.regression import fit_panel_fe, tidy_linearmodels
from nimscale.settings import load_config, project_root


def infer_event_columns(df: pd.DataFrame, cfg: dict) -> dict[str, str]:
    cc = cfg["fdic"]["column_candidates"]
    return {
        "event_date": pick_first_existing(df, cc["history_event_date"]),
        "acquirer": pick_first_existing(df, cc["history_acquirer_cert"]),
        "desc": pick_first_existing(df, cc["history_event_desc"]),
    }


def quarter_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
    later_p = pd.to_datetime(later).dt.to_period("Q")
    earlier_p = pd.to_datetime(earlier).dt.to_period("Q")
    return later_p.astype(int) - earlier_p.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Event study around acquisition-driven growth.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    history_path = project_root() / cfg["paths"]["raw"] / "fdic_history" / "history_events.csv"
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])
    fig_dir = ensure_dir(project_root() / cfg["paths"]["figures"])

    if not history_path.exists():
        raise FileNotFoundError(history_path)
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    hist = pd.read_csv(history_path, low_memory=False)
    hist = standardize_columns(hist)
    cols = infer_event_columns(hist, cfg)

    hist[cols["desc"]] = hist[cols["desc"]].astype(str)

    # Keep only institutional-level merger/acquisition events where
    # the acquirer cert is populated (excludes branch-level events).
    pat = "MERGER|ACQUIS|COMBINATION|ABSORPTION|ASSUMPTION|ABSORBTION|CONSOLIDAT"
    events = hist[hist[cols["desc"]].str.upper().str.contains(pat, regex=True, na=False)].copy()
    events[cols["event_date"]] = pd.to_datetime(events[cols["event_date"]], errors="coerce")
    events[cols["acquirer"]] = pd.to_numeric(events[cols["acquirer"]], errors="coerce")
    events = events.dropna(subset=[cols["event_date"], cols["acquirer"]])
    # Convert acquirer cert to int string for clean merge with panel
    events["_ACQ_CERT"] = events[cols["acquirer"]].astype(int).astype(str)
    print(f"institutional-level acquisition events: {len(events):,}")

    # First event per acquirer for a simple clean event study
    first_events = (
        events.sort_values(["_ACQ_CERT", cols["event_date"]])
        .groupby("_ACQ_CERT", as_index=False)
        .first()
        [["_ACQ_CERT", cols["event_date"]]]
    )
    first_events.columns = ["CERT", "EVENT_DATE"]
    print(f"unique acquirers with first event: {len(first_events):,}")

    panel = pd.read_parquet(panel_path)
    # Ensure CERT is an int string (no decimals) for clean merge
    panel["CERT"] = pd.to_numeric(panel["CERT"], errors="coerce").dropna().astype(int).astype(str)
    panel["REPDTE"] = pd.to_datetime(panel["REPDTE"])
    panel = panel.merge(first_events, on="CERT", how="inner")
    panel["EVENT_TIME_Q"] = quarter_diff(panel["REPDTE"], panel["EVENT_DATE"])

    window = args.window
    panel = panel[(panel["EVENT_TIME_Q"] >= -window) & (panel["EVENT_TIME_Q"] <= window)].copy()
    panel = panel.dropna(subset=["NIM"])
    print(f"event study panel: {len(panel):,} obs, {panel['CERT'].nunique():,} acquirers, window=[-{window},+{window}]")

    # Build event-time dummies excluding -1
    def event_name(k: int) -> str:
        return f"ET_m{abs(k)}" if k < 0 else f"ET_p{k}"

    for k in range(-window, window + 1):
        if k == -1:
            continue
        panel[event_name(k)] = (panel["EVENT_TIME_Q"] == k).astype(int)

    rhs_terms = " + ".join([event_name(k) for k in range(-window, window + 1) if k != -1])
    formula = f"NIM ~ 1 + {rhs_terms} + EntityEffects + TimeEffects"
    res = fit_panel_fe(panel, formula=formula, entity_col="CERT", time_col="REPDTE")
    tidy = tidy_linearmodels(res, "acquirer_event_study")
    tidy.to_csv(table_dir / "event_study_results.csv", index=False)

    plot_df = tidy[tidy["term"].str.startswith("ET_")].copy()

    def parse_event_time(term: str) -> int:
        x = term.replace("ET_", "")
        if x.startswith("m"):
            return -int(x[1:])
        if x.startswith("p"):
            return int(x[1:])
        raise ValueError(term)

    plot_df["event_time"] = plot_df["term"].map(parse_event_time)
    plot_df = plot_df.sort_values("event_time")

    plt.figure(figsize=(8, 4))
    plt.axhline(0, linewidth=1)
    plt.axvline(-1, linewidth=1, linestyle="--")
    plt.plot(plot_df["event_time"], plot_df["coef"], marker="o")
    plt.fill_between(plot_df["event_time"], plot_df["ci_low"], plot_df["ci_high"], alpha=0.2)
    plt.xlabel("Quarters relative to event")
    plt.ylabel("NIM effect")
    plt.title("Acquirer event study")
    plt.tight_layout()
    plt.savefig(fig_dir / "acquirer_event_study.png", dpi=160)
    plt.close()

    print(f"saved {table_dir / 'event_study_results.csv'}")


if __name__ == "__main__":
    main()
