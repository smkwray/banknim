from __future__ import annotations

import pandas as pd

from nimscale.fdic_api import FDICClient


class DummyResponse:
    def __init__(self, text: str, headers: dict[str, str] | None = None, payload: dict | None = None) -> None:
        self.text = text
        self.headers = headers or {}
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        if self._payload is None:
            raise AssertionError("json() should not have been called")
        return self._payload


def test_get_csv_page_reads_csv_and_requests_csv_format(monkeypatch):
    captured = {}

    def fake_get(url, params, headers, timeout):
        captured["params"] = params
        return DummyResponse(
            text='"CERT","REPDTE"\n1,"20241231"\n',
            headers={"content-type": "text/csv; charset=utf-8"},
        )

    client = FDICClient(base_url="https://api.fdic.gov/banks")
    monkeypatch.setattr(client.session, "get", fake_get)

    out = client.get_csv_page("financials", {"filters": "CERT:1"})

    assert captured["params"]["format"] == "csv"
    assert out.to_dict(orient="records") == [{"CERT": 1, "REPDTE": 20241231}]


def test_get_csv_page_flattens_json_payload(monkeypatch):
    payload = {
        "meta": {"total": 1},
        "data": [
            {
                "data": {"CERT": 1, "REPDTE": "20241231", "ASSET": 100.0},
                "score": 0,
            }
        ],
    }

    def fake_get(url, params, headers, timeout):
        return DummyResponse(
            text='{"meta":{"total":1},"data":[{"data":{"CERT":1}}]}',
            headers={"content-type": "application/json; charset=utf-8"},
            payload=payload,
        )

    client = FDICClient(base_url="https://api.fdic.gov/banks")
    monkeypatch.setattr(client.session, "get", fake_get)

    out = client.get_csv_page("financials", {"filters": "CERT:1"})

    expected = pd.DataFrame([{"CERT": 1, "REPDTE": "20241231", "ASSET": 100.0}])
    pd.testing.assert_frame_equal(out, expected)
