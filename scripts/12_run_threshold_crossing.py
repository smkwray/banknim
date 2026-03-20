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

from nimscale.bank_panel import winsorize_series
from nimscale.io import ensure_dir
from nimscale.regression import fit_panel_fe, tidy_linearmodels
from nimscale.settings import load_config, project_root
from nimscale.validation import assert_nonempty_sample, require_columns, winsorize_required


def find_first_crossing(df: pd.DataFrame, bank_col: str, date_col: str, flag_col: str) -> pd.DataFrame:
    """Return DataFrame with (bank_col, CROSS_DATE) for the first quarter each bank crosses the threshold."""
    above = df[df[flag_col] == 1].copy()
    if above.empty:
        return pd.DataFrame(columns=[bank_col, "CROSS_DATE"])
    first = above.sort_values(date_col).groupby(bank_col, as_index=False)[date_col].first()
    first.columns = [bank_col, "CROSS_DATE"]
    return first


def quarter_diff(later: pd.Series, earlier: pd.Series) -> pd.Series:
    later_p = pd.to_datetime(later).dt.to_period("Q")
    earlier_p = pd.to_datetime(earlier).dt.to_period("Q")
    return later_p.astype(int) - earlier_p.astype(int)


def run_threshold_event_study(
    panel: pd.DataFrame,
    cross_events: pd.DataFrame,
    bank_col: str,
    date_col: str,
    window: int,
    model_name: str,
) -> pd.DataFrame | None:
    es = panel.merge(cross_events, on=bank_col, how="inner").copy()
    es["EVENT_TIME_Q"] = quarter_diff(es[date_col], es["CROSS_DATE"])
    es = es[(es["EVENT_TIME_Q"] >= -window) & (es["EVENT_TIME_Q"] <= window)]
    es = es.dropna(subset=["NIM_W"])

    if es[bank_col].nunique() < 10:
        print(f"  {model_name}: too few crossers ({es[bank_col].nunique()}), skipping")
        return None
    assert_nonempty_sample(es, model_name, min_rows=1, entity_col=bank_col, min_entities=10)

    print(f"  {model_name}: {len(es):,} obs, {es[bank_col].nunique():,} banks, window=[-{window},+{window}]")

    def ename(k: int) -> str:
        return f"ET_m{abs(k)}" if k < 0 else f"ET_p{k}"

    for k in range(-window, window + 1):
        if k == -1:
            continue
        es[ename(k)] = (es["EVENT_TIME_Q"] == k).astype(int)

    rhs = " + ".join(ename(k) for k in range(-window, window + 1) if k != -1)
    formula = f"NIM_W ~ 1 + {rhs} + EQ_RATIO_W + LOANS_SHARE_W + EntityEffects + TimeEffects"
    res = fit_panel_fe(es, formula=formula, entity_col=bank_col, time_col=date_col)
    return tidy_linearmodels(res, model_name)


def plot_event_study(tidy: pd.DataFrame, model_name: str, fig_path: Path) -> None:
    et = tidy[tidy["term"].str.startswith("ET_")].copy()
    if et.empty:
        return

    def parse_et(t: str) -> int:
        x = t.replace("ET_", "")
        return -int(x[1:]) if x.startswith("m") else int(x[1:])

    et["event_time"] = et["term"].map(parse_et)
    et = et.sort_values("event_time")

    plt.figure(figsize=(8, 4))
    plt.axhline(0, linewidth=1, color="gray")
    plt.axvline(-1, linewidth=1, linestyle="--", color="gray")
    plt.plot(et["event_time"], et["coef"], marker="o")
    plt.fill_between(et["event_time"], et["ci_low"], et["ci_high"], alpha=0.2)
    plt.xlabel("Quarters relative to threshold crossing")
    plt.ylabel("NIM effect")
    plt.title(model_name.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="M7: Threshold-crossing event studies.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(panel_path)

    table_dir = ensure_dir(project_root() / cfg["paths"]["tables"])
    fig_dir = ensure_dir(project_root() / cfg["paths"]["figures"])

    df = pd.read_parquet(panel_path).copy()
    df["CERT"] = df["CERT"].astype(str)
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])

    require_columns(df, ["NIM", "LN_ASSETS", "EQ_RATIO", "LOANS_SHARE"], "threshold-crossing models")
    wp = cfg["project"]["winsor_pct"]
    df["NIM_W"] = winsorize_series(df["NIM"], p=wp)
    df["EQ_RATIO_W"] = winsorize_required(df, "EQ_RATIO", p=wp, context="threshold-crossing models")
    df["LOANS_SHARE_W"] = winsorize_required(df, "LOANS_SHARE", p=wp, context="threshold-crossing models")

    results = []

    # Run event studies for each threshold
    thresholds = [
        ("GT_10B", "threshold_cross_10b"),
        ("GT_50B", "threshold_cross_50b"),
        ("GT_100B", "threshold_cross_100b"),
    ]

    for flag_col, model_name in thresholds:
        if flag_col not in df.columns:
            print(f"  {flag_col} not in panel, skipping")
            continue

        cross = find_first_crossing(df, "CERT", "REPDTE", flag_col)
        # Require the bank to have been below the threshold at some earlier point
        # (i.e., it actually crossed, not always above)
        always_above = df.groupby("CERT")[flag_col].min()
        true_crossers = always_above[always_above == 0].index
        cross = cross[cross["CERT"].isin(true_crossers)]
        print(f"{flag_col}: {len(cross):,} banks crossed threshold")

        tidy = run_threshold_event_study(df, cross, "CERT", "REPDTE", args.window, model_name)
        if tidy is not None:
            results.append(tidy)
            plot_event_study(tidy, model_name, fig_dir / f"{model_name}.png")

    if results:
        out = pd.concat(results, ignore_index=True)
        out.to_csv(table_dir / "threshold_crossing_results.csv", index=False)
        print(f"saved {table_dir / 'threshold_crossing_results.csv'}")
    else:
        print("no threshold models produced")


if __name__ == "__main__":
    main()
