from __future__ import annotations

import os
import time
from io import StringIO
from typing import Any

import pandas as pd
import requests


class FDICClient:
    """Tiny FDIC BankFind API client.

    Notes
    -----
    - Uses CSV output because it is easy to page and persist.
    - Keeps the API key optional.
    - Assumes the BankFind endpoints are rooted at:
      https://api.fdic.gov/banks/{endpoint}
    """

    def __init__(
        self,
        base_url: str,
        api_key_env: str | None = None,
        timeout: int = 120,
        pause_seconds: float = 0.15,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv(api_key_env) if api_key_env else None
        self.timeout = timeout
        self.pause_seconds = pause_seconds
        self.max_retries = max_retries
        self.session = requests.Session()

    def _request(self, endpoint: str, params: dict[str, Any]) -> requests.Response:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"Accept": "text/csv"}
        req_params = dict(params)
        req_params.setdefault("format", "csv")
        if self.api_key:
            req_params["api_key"] = self.api_key

        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                resp = self.session.get(url, params=req_params, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                return resp
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                time.sleep(self.pause_seconds)
        raise RuntimeError(f"FDIC request failed for {endpoint}: {last_error}")

    @staticmethod
    def _parse_json_payload(payload: dict[str, Any]) -> pd.DataFrame:
        records = payload.get("data", [])
        if not records:
            return pd.DataFrame()

        flattened: list[dict[str, Any]] = []
        for record in records:
            if isinstance(record, dict) and isinstance(record.get("data"), dict):
                flattened.append(record["data"])
            elif isinstance(record, dict):
                flattened.append(record)

        if not flattened:
            return pd.DataFrame()
        return pd.DataFrame(flattened)

    def get_csv_page(self, endpoint: str, params: dict[str, Any]) -> pd.DataFrame:
        resp = self._request(endpoint, params=params)
        text = resp.text.strip()
        if not text:
            return pd.DataFrame()
        content_type = resp.headers.get("content-type", "").lower()

        if "json" in content_type or text.startswith("{"):
            return self._parse_json_payload(resp.json())

        # BankFind CSV can come with BOM
        text = text.lstrip("\ufeff")
        return pd.read_csv(StringIO(text))

    def get_csv_paged(
        self,
        endpoint: str,
        params: dict[str, Any],
        page_size: int = 10000,
        max_pages: int = 1000,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        offset = 0
        for _page in range(max_pages):
            page_params = dict(params)
            page_params.update(
                {
                    "limit": page_size,
                    "offset": offset,
                    "download": "false",
                }
            )
            df = self.get_csv_page(endpoint, page_params)
            if df.empty:
                break
            frames.append(df)
            if len(df) < page_size:
                break
            offset += page_size
            time.sleep(self.pause_seconds)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out.columns = [str(c).upper() for c in out.columns]
        return out
