from __future__ import annotations

import pandas as pd

from nimscale.bank_panel import pick_first_existing


def test_pick_first_existing_finds_first_match():
    df = pd.DataFrame(columns=["ABC", "CERT", "XYZ"])
    assert pick_first_existing(df, ["ID", "CERT", "OTHER"]) == "CERT"
