from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fit_between_ols(df: pd.DataFrame, formula: str):
    model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
    return model


def fit_panel_fe(df: pd.DataFrame, formula: str, entity_col: str, time_col: str):
    try:
        from linearmodels.panel import PanelOLS
    except ImportError as exc:  # pragma: no cover
        raise ImportError("linearmodels is required for panel fixed-effects models.") from exc

    panel = df.copy()
    panel[entity_col] = panel[entity_col].astype(str)
    panel[time_col] = pd.to_datetime(panel[time_col])
    panel = panel.set_index([entity_col, time_col]).sort_index()

    model = PanelOLS.from_formula(formula=formula, data=panel, drop_absorbed=True)
    res = model.fit(cov_type="clustered", cluster_entity=True)
    return res


def tidy_statsmodels(result, model_name: str) -> pd.DataFrame:
    ci = result.conf_int()
    out = pd.DataFrame(
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
            "r2": getattr(result, "rsquared", float("nan")),
        }
    )
    return out


def tidy_linearmodels(result, model_name: str) -> pd.DataFrame:
    ci = result.conf_int()
    out = pd.DataFrame(
        {
            "model": model_name,
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.std_errors.values,
            "t_or_z": result.tstats.values,
            "p_value": result.pvalues.values,
            "ci_low": ci.iloc[:, 0].values,
            "ci_high": ci.iloc[:, 1].values,
            "nobs": float(result.nobs),
            "r2_within": getattr(result, "rsquared_within", float("nan")),
            "r2_between": getattr(result, "rsquared_between", float("nan")),
            "r2_overall": getattr(result, "rsquared_overall", float("nan")),
        }
    )
    return out
