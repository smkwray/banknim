from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text_list(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def write_lines(path: str | Path, lines: Iterable[str]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
