from __future__ import annotations

import pandas as pd
import pytest

from nimscale.validation import (
    ValidationError,
    assert_merge_coverage,
    assert_nonempty_sample,
    require_columns,
    winsorize_required,
)


def test_require_columns_raises_for_missing_inputs():
    df = pd.DataFrame({"A": [1], "B": [2]})
    with pytest.raises(ValidationError):
        require_columns(df, ["A", "C"], "validation test")


def test_winsorize_required_raises_when_column_has_no_usable_values():
    df = pd.DataFrame({"A": [None, None]})
    with pytest.raises(ValidationError):
        winsorize_required(df, "A", p=0.01, context="validation test")


def test_assert_merge_coverage_raises_for_incomplete_coverage():
    df = pd.DataFrame({"FEDFUNDS": [1.0, None], "SLOPE_10Y_3M": [0.5, 0.6]})
    with pytest.raises(ValidationError):
        assert_merge_coverage(df, ["FEDFUNDS", "SLOPE_10Y_3M"], "validation test")


def test_assert_nonempty_sample_checks_rows_and_entities():
    df = pd.DataFrame({"CERT": ["1"], "value": [1]})
    with pytest.raises(ValidationError):
        assert_nonempty_sample(df, "validation test", min_rows=2)
    with pytest.raises(ValidationError):
        assert_nonempty_sample(df, "validation test", entity_col="CERT", min_entities=2)
