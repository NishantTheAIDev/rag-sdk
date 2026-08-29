#!/usr/bin/env python
"""Download test fixtures for external tests.

Run with: uv run python tests/external/download_fixtures.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

FIXTURES = {
    "sample.pdf": "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",
    "sample.docx": "https://raw.githubusercontent.com/python-openxml/python-docx/main/docx/templates/default.docx",
    "sample.html": "https://www.w3.org/TR/html52/",
    "long_document.pdf": "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  Success: {dest.stat().st_size} bytes")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def main() -> int:
    print(f"Downloading fixtures to {FIXTURES_DIR}")
    success = 0
    for name, url in FIXTURES.items():
        dest = FIXTURES_DIR / name
        if dest.exists():
            print(f"Skipping {name} (already exists)")
            success += 1
            continue
        if download_file(url, dest):
            success += 1
    
    print(f"\nDownloaded {success}/{len(FIXTURES)} fixtures")
    return 0 if success == len(FIXTURES) else 1


if __name__ == "__main__":
    raise SystemExit(main())