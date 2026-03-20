from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from nimscale.settings import load_config, project_root
from nimscale.validation import assert_nonempty_sample, require_columns, winsorize_required


FAILURE_DESCS = {
    "Failure - Whole Institution",
    "Closing - Failure Payoff",
    "Bridge Bank Resolution",
}
ASSISTED_DESC = "Participated in FDIC Assisted Merger"


def prep_panel(cfg: dict) -> pd.DataFrame:
    panel_path = project_root() / cfg["paths"]["interim"] / "bank_panel.parquet"
    df = pd.read_parquet(panel_path).copy()
    df["CERT"] = pd.to_numeric(df["CERT"], errors="coerce").astype("Int64")
    df["REPDTE"] = pd.to_datetime(df["REPDTE"])
    df["QUARTER"] = df["REPDTE"].dt.to_period("Q").astype(str)
    wp = cfg["project"]["winsor_pct"]
    require_columns(df, ["NIM", "ROA", "EQ_RATIO", "LOANS_SHARE", "LN_ASSETS"], "distress prep")
    df["NIM_W"] = winsorize_required(df, "NIM", p=wp, context="distress prep")
    df["ROA_W"] = winsorize_required(df, "ROA", p=wp, context="distress prep")
    df["EQ_RATIO_W"] = winsorize_required(df, "EQ_RATIO", p=wp, context="distress prep")
    df["LOANS_SHARE_W"] = winsorize_required(df, "LOANS_SHARE", p=wp, context="distress prep")
    return df


def load_distress_events(cfg: dict, include_assisted: bool = False) -> pd.DataFrame:
    history_path = project_root() / cfg["paths"]["raw"] / "fdic_history" / "history_events.csv"
    hist = pd.read_csv(history_path, low_memory=False)
    hist["CHANGECODE_DESC"] = hist["CHANGECODE_DESC"].astype(str)
    event_descs = set(FAILURE_DESCS)
    if include_assisted:
        event_descs.add(ASSISTED_DESC)
    events = hist[hist["CHANGECODE_DESC"].isin(event_descs)].copy()
    events["EVENT_DATE"] = pd.to_datetime(events["EFFDATE"], errors="coerce")
    cert_cols = ["CERT", "SUR_CERT", "ACQ_CERT", "FRM_CERT"]
    for col in cert_cols:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    events["CERT_EVENT"] = events[cert_cols].bfill(axis=1).iloc[:, 0]
    events = events.dropna(subset=["EVENT_DATE", "CERT_EVENT"]).copy()
    events["CERT_EVENT"] = events["CERT_EVENT"].astype(int)
    events["EVENT_Q"] = events["EVENT_DATE"].dt.to_period("Q")
    events["EVENT_Q_NUM"] = events["EVENT_DATE"].dt.year * 4 + events["EVENT_DATE"].dt.quarter
    events = (
        events.sort_values(["CERT_EVENT", "EVENT_DATE"])
        .drop_duplicates(subset=["CERT_EVENT", "EVENT_Q"], keep="first")
        [["CERT_EVENT", "EVENT_DATE", "EVENT_Q", "EVENT_Q_NUM", "CHANGECODE_DESC"]]
    )
    return events


def build_forward_distress_sample(panel: pd.DataFrame, events: pd.DataFrame, horizon_q: int = 4) -> pd.DataFrame:
    event_q_map = events.groupby("CERT_EVENT")["EVENT_Q"].min().to_dict()
    event_q_num_map = events.groupby("CERT_EVENT")["EVENT_Q_NUM"].min().to_dict()

    sample = panel.copy()
    sample["EVENT_Q"] = sample["CERT"].map(event_q_map)
    sample["EVENT_Q_NUM"] = sample["CERT"].map(event_q_num_map)
    sample["REPDTE_Q"] = sample["REPDTE"].dt.to_period("Q")
    sample["REPDTE_Q_NUM"] = sample["REPDTE"].dt.year * 4 + sample["REPDTE"].dt.quarter

    qdiff = sample["EVENT_Q_NUM"] - sample["REPDTE_Q_NUM"]
    sample["DISTRESS_NEXT_4Q"] = ((qdiff >= 1) & (qdiff <= horizon_q)).astype(int)
    sample = sample[(sample["EVENT_Q"].isna()) | (qdiff >= 1)].copy()
    return sample


def tidy_glm(result, model_name: str) -> pd.DataFrame:
    ci = result.conf_int()
    return pd.DataFrame(
        {
            "model": model_name,
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "t_or_z": result.tvalues.values,
            "p_value": result.pvalues.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
            "nobs": float(result.nobs),
            "r2": float("nan"),
        }
    )


def upsert_extension_results(table_dir: Path, new_rows: pd.DataFrame) -> None:
    path = table_dir / "extension_results.csv"
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[~existing["model"].isin(new_rows["model"].unique())].copy()
        out = pd.concat([existing, new_rows], ignore_index=True)
    else:
        out = new_rows.copy()
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forward distress prediction logit.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--include-assisted",
        action="store_true",
        help="Also include assisted-merger resolution rows from FDIC history in the event label.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = project_root() / cfg["paths"]["tables"]

    panel = prep_panel(cfg)
    events = load_distress_events(cfg, include_assisted=args.include_assisted)
    sample = build_forward_distress_sample(panel, events, horizon_q=4)
    model_df = sample.dropna(subset=["DISTRESS_NEXT_4Q", "NIM_W", "ROA_W", "EQ_RATIO_W", "LOANS_SHARE_W", "LN_ASSETS"]).copy()
    assert_nonempty_sample(model_df, "distress sample", min_rows=1, entity_col="CERT", min_entities=2)

    formula = "DISTRESS_NEXT_4Q ~ NIM_W + ROA_W + EQ_RATIO_W + LOANS_SHARE_W + LN_ASSETS + C(QUARTER)"
    model = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": model_df["CERT"]})
    out = tidy_glm(res, "distress_logit")
    assert_nonempty_sample(out, "distress result export")
    upsert_extension_results(table_dir, out)

    event_rate = model_df["DISTRESS_NEXT_4Q"].mean()
    event_label = "failure+assisted" if args.include_assisted else "failure_only"
    print(
        f"distress_logit: NIM_W={res.params['NIM_W']:.6f} "
        f"(p={res.pvalues['NIM_W']:.4f}), event_rate={event_rate:.5f}, "
        f"nobs={res.nobs}, label={event_label}"
    )
    print(f"updated {table_dir / 'extension_results.csv'}")


if __name__ == "__main__":
    main()
