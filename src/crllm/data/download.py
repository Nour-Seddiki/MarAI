"""Resumable downloads of Hub parquet shards over plain HTTP.

huggingface_hub downloads have hung on the laptop (Xet backend and plain HTTP alike);
this streams with Range-resume and retries, like `curl -C -`, and verifies the size.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)
API = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"


def shard_urls(repo: str, config: str, split: str) -> list[str]:
    r = requests.get(API.format(repo=repo, config=config, split=split), timeout=30)
    r.raise_for_status()
    return list(r.json())


def remote_size(url: str) -> int | None:
    r = requests.head(url, allow_redirects=True, timeout=30)
    if not r.ok:
        return None
    n = r.headers.get("Content-Length")
    return int(n) if n and n.isdigit() else None


def fetch(url: str, dest: Path, tries: int = 40, chunk: int = 1 << 20) -> Path:
    """Download `url` to `dest`, resuming a `.part` file; a complete `dest` is kept as is."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    total = remote_size(url)
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have >= total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(20, 60)) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
        except (requests.RequestException, OSError) as e:
            log.warning("%s: attempt %d at %.0f MB: %s", dest.name, attempt, have / 2**20, str(e)[:100])
            time.sleep(min(3 * attempt, 30))
            continue
        if total is None or part.stat().st_size >= total:
            break
    size = part.stat().st_size if part.exists() else 0
    if total is not None and size != total:
        raise RuntimeError(f"{dest.name}: got {size} of {total} bytes after {tries} attempts")
    part.replace(dest)
    log.info("downloaded %s (%.0f MB)", dest.name, size / 2**20)
    return dest


def fetch_shards(repo: str, config: str, split: str, out_dir: Path, max_shards: int | None = None) -> list[Path]:
    urls = shard_urls(repo, config, split)[:max_shards]
    safe = repo.replace("/", "__")
    return [fetch(u, out_dir / safe / config / f"{split}-{i:04d}.parquet") for i, u in enumerate(urls)]
