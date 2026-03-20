from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .bank_panel import winsorize_series


class ValidationError(ValueError):
    """Raised when pipeline inputs fail required validation."""


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValidationError(f"{context}: missing required columns {missing}")


def winsorize_required(df: pd.DataFrame, source_col: str, p: float, context: str) -> pd.Series:
    require_columns(df, [source_col], context)
    series = pd.to_numeric(df[source_col], errors="coerce")
    if series.dropna().empty:
        raise ValidationError(f"{context}: column {source_col} has no usable values")
    return winsorize_series(series, p=p)


def assert_merge_coverage(
    df: pd.DataFrame,
    columns: Iterable[str],
    context: str,
    min_non_null_share: float = 1.0,
) -> None:
    require_columns(df, columns, context)
    failures: list[str] = []
    for col in columns:
        share = float(df[col].notna().mean())
        if share < min_non_null_share:
            failures.append(f"{col}={share:.3f}")
    if failures:
        threshold = f"{min_non_null_share:.3f}"
        raise ValidationError(
            f"{context}: non-null coverage below {threshold} for {', '.join(failures)}"
        )


def assert_nonempty_sample(
    df: pd.DataFrame,
    context: str,
    *,
    min_rows: int = 1,
    entity_col: str | None = None,
    min_entities: int | None = None,
) -> None:
    row_count = len(df)
    if row_count < min_rows:
        raise ValidationError(f"{context}: sample has {row_count} rows, need at least {min_rows}")

    if entity_col is not None and min_entities is not None:
        if entity_col not in df.columns:
            raise ValidationError(f"{context}: entity column {entity_col} is missing")
        entity_count = df[entity_col].nunique(dropna=True)
        if entity_count < min_entities:
            raise ValidationError(
                f"{context}: sample has {entity_count} entities, need at least {min_entities}"
            )

