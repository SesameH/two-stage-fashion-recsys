"""Fetch product images for the articles the console can actually display.

Run: python -m src.serve.fetch_images        (needs ~/.kaggle/kaggle.json)

The competition's image set is ~35 GB for 105,542 articles. Everything the console *could*
display is 74,681 articles (~2.2 GB) — recommendations plus purchase history across every
evaluation customer. Everything it *will* display in a browsing session is far less, because
one session shows ~40 articles per customer clicked and sales are heavily skewed toward a small
head. Hence the default limit: the 3,000 most-sold articles cover most tiles at about
400 MB (measured ~133 KB per JPEG, not the 30 KB first assumed).

Kaggle throttles per-file competition downloads and reports the throttle as a 404, so the
script resumes: files already present are skipped, and a failure-heavy run prints a warning
rather than pretending those articles have no image.

Scope, deliberately: these images are for the LOCAL demo and for README screenshots. They are
H&M's commercial photography, and the competition data licence does not cover redistributing
them from a public deployment, so `.dockerignore` and `.gitignore` both exclude the output
directory. The console degrades to product-type tiles when a file is absent, which is what the
deployed version shows.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb

from src.serve.precompute import SERVING

COMPETITION = "h-and-m-personalized-fashion-recommendations"


def target_articles(limit: int | None) -> list[int]:
    """Articles worth downloading: everything the demo tables can surface.

    That is the union of what customers bought in the target week and what the retrieval layer
    proposes, capped by popularity so a `--limit` keeps the most-shown images.
    """
    con = duckdb.connect()
    rows = con.execute(
        f"""
        WITH shown AS (
            SELECT article_id FROM read_parquet('{SERVING / "demo_truth.parquet"}')
            UNION
            SELECT article_id FROM read_parquet('{SERVING / "demo_history.parquet"}')
            UNION
            SELECT article_id FROM read_parquet('{SERVING / "popularity_global.parquet"}')
            UNION
            SELECT article_id FROM read_parquet('{SERVING / "popularity_age.parquet"}')
            UNION
            SELECT article_id FROM read_parquet('{SERVING / "category_candidates.parquet"}')
            UNION
            SELECT article_id FROM read_parquet('{SERVING / "variants.parquet"}')
            UNION
            SELECT article_id FROM read_parquet('{SERVING / "repurchase.parquet"}')
        ),
        pop AS (
            SELECT article_id, n_sold
            FROM read_parquet('{SERVING / "article_features.parquet"}')
        )
        SELECT s.article_id
        FROM shown s LEFT JOIN pop p USING (article_id)
        ORDER BY coalesce(p.n_sold, 0) DESC, s.article_id
        {f"LIMIT {int(limit)}" if limit else ""}
        """
    ).df()["article_id"].tolist()
    con.close()
    return [int(a) for a in rows]


def _authenticated_api():
    import kaggle

    api = kaggle.KaggleApi()
    api.authenticate()
    return api


def _bump(counts: dict, lock: threading.Lock, key: str, total: int) -> None:
    with lock:
        counts[key] += 1
        counts["done"] += 1
        if counts["done"] % 250 == 0:
            print(
                f"  {counts['done']:,}/{total:,}  ok={counts['ok']:,} "
                f"skipped={counts['skipped']:,} failed={counts['failed']:,}",
                flush=True,
            )


def image_path(article_id: int) -> str:
    """Dataset layout: images/<first 3 digits of the zero-padded id>/<id>.jpg"""
    padded = f"{article_id:010d}"
    return f"images/{padded[:3]}/{padded}.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SERVING / "images")
    parser.add_argument(
        "--limit", type=int, default=3000,
        help="download the N most-sold articles (0 for all 74,681)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--sleep", type=float, default=0.2,
        help="delay before each request, per worker; raise if downloads start 404ing",
    )
    args = parser.parse_args()

    # kaggle 2.x accepts either the legacy kaggle.json or an access_token file, and env vars
    # override both. Check all three rather than hard-coding the older layout.
    import os

    kdir = Path.home() / ".kaggle"
    if not (
        (kdir / "kaggle.json").exists()
        or (kdir / "access_token").exists()
        or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    ):
        raise SystemExit(
            "no Kaggle credentials found. Create an API token at "
            "https://www.kaggle.com/settings and save it as ~/.kaggle/kaggle.json "
            "(or set KAGGLE_USERNAME and KAGGLE_KEY), then re-run."
        )

    articles = target_articles(args.limit or None)
    print(f"{len(articles):,} articles to fetch -> {args.out}", flush=True)

    api = _authenticated_api()
    lock = threading.Lock()
    errors: Counter[str] = Counter()
    counts = {"ok": 0, "skipped": 0, "failed": 0, "done": 0}

    def fetch(article_id: int) -> None:
        rel = image_path(article_id)
        dest_dir = args.out / Path(rel).parent.name
        dest = dest_dir / Path(rel).name
        if dest.exists():
            _bump(counts, lock, "skipped", len(articles))
            return
        dest_dir.mkdir(parents=True, exist_ok=True)
        if args.sleep:
            time.sleep(args.sleep)
        try:
            api.competition_download_file(COMPETITION, rel, path=str(dest_dir), force=True)
            _bump(counts, lock, "ok" if dest.exists() else "failed", len(articles))
        except Exception as exc:  # noqa: BLE001 — the error text is the diagnosis, keep it
            with lock:
                errors[f"{type(exc).__name__}: {str(exc)[:90]}"] += 1
            _bump(counts, lock, "failed", len(articles))

    # In-process API calls in a small thread pool. The first version spawned `python -m kaggle`
    # per file, which spent more time on interpreter startup than on the download itself.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, a) for a in articles]
        for f in as_completed(futures):
            f.result()

    ok, skipped, failed = counts["ok"], counts["skipped"], counts["failed"]
    print(f"done: ok={ok:,} skipped={skipped:,} failed={failed:,}")
    for msg, n in errors.most_common(3):
        print(f"  {n:,}x {msg}")

    # Kaggle throttles per-file competition downloads and signals it as 404, not 429 — a path
    # that downloaded successfully a minute earlier starts returning 404 once the quota is hit.
    # So a high failure rate means "slow down", not "the dataset lacks these images". The first
    # version of this script assumed the latter and reported 2,650 throttled requests as
    # missing files.
    if failed > ok:
        print(
            "\nMost requests failed. Kaggle returns 404 for throttled per-file downloads, so "
            "this is almost certainly a rate limit rather than missing images.\n"
            "Re-run later to resume — files already on disk are skipped. Lower --workers and "
            "raise --sleep to stay under the limit."
        )


if __name__ == "__main__":
    main()
