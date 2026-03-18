from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path

import requests

from nimscale.io import ensure_dir
from nimscale.settings import load_config, project_root


def download(url: str, path: Path) -> None:
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    print(f"saved {path} ({len(resp.content):,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official metadata and helper docs.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = ensure_dir(project_root() / cfg["paths"]["external"] / "metadata")

    urls = {
        "fdic_swagger.yaml": cfg["fdic"]["metadata_urls"]["swagger"],
        "fdic_common_financial_reports.xlsx": cfg["fdic"]["metadata_urls"]["common_financial_reports"],
        "fdic_sod_variables.csv": cfg["fdic"]["metadata_urls"]["sod_definitions"],
    }

    ffiec_user_guide_page = cfg["ffiec"]["user_guide_page"]
    urls["ffiec_user_guide_page.html"] = ffiec_user_guide_page

    for fname, url in urls.items():
        path = outdir / fname
        try:
            download(url, path)
        except Exception as exc:
            print(f"warning: failed to fetch {url}: {exc}")

    print("done")


if __name__ == "__main__":
    main()
