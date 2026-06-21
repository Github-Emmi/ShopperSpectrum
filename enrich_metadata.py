#!/usr/bin/env python3
"""
enrich_metadata.py
==================
Visual Metadata Bridge: online_retail.csv → product_metadata.json

Fetches product thumbnails, ratings, and review counts from Google
Shopping via Serper.dev (recommended) or SerpApi.  Falls back to
deterministic Picsum Photos when no API key is provided.

Usage
-----
    # Serper.dev — 2,500 FREE queries, no credit card required
    # Sign up at https://serper.dev  (takes ~1 minute)
    export SERPER_KEY="your_serper_key"
    python enrich_metadata.py

    # SerpApi (legacy; 250 free searches/month)
    export SERPAPI_KEY="your_serpapi_key"
    python enrich_metadata.py

    # Re-fetch all entries currently marked source='synthetic'
    export SERPER_KEY="your_serper_key"
    python enrich_metadata.py --replace-synthetic

    # Regenerate synthetic placeholders in-place (no API)
    python enrich_metadata.py --regen-synthetic

    # Synthetic mode — no API key, instant deterministic Picsum output
    python enrich_metadata.py --synthetic-only

    # Limit to first N products
    python enrich_metadata.py --limit 200

The script is fully idempotent: products already present in
product_metadata.json with a real source are skipped to preserve
API credits.  A checkpoint is saved every 50 new entries.

Output
------
product_metadata.json  (keyed by product Description string)
{
  "WHITE HANGING HEART T-LIGHT HOLDER": {
    "thumbnail":    "https://...",
    "rating":       4.7,
    "reviews":      2847,
    "unit_price":   2.55,
    "strike_price": 3.83,
    "source":       "serper" | "serpapi" | "synthetic"
  },
  ...
}
"""

import os
import json
import csv
import time
import random
import argparse
from pathlib import Path

DATA_FILE     = "online_retail.csv"
METADATA_FILE = "product_metadata.json"
SERPAPI_URL   = "https://serpapi.com/search.json"
SERPER_URL    = "https://google.serper.dev/shopping"

def _placeholder_url(description: str, seed: int = 0) -> str:
    """
    Return a deterministic Picsum Photos URL seeded by the product name.
    Every product gets a unique, consistent, beautiful photograph.
    picsum.photos/seed/{text}/WxH is free, no API key, and stable.
    """
    import re
    # Build a clean URL-safe slug from the description
    slug = description.lower()[:40]
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)   # strip non-alphanum except space/hyphen
    slug = re.sub(r"\s+", "-", slug.strip())     # spaces → hyphens
    slug = slug.strip("-") or "product"
    return f"https://picsum.photos/seed/{slug}/300/300"


def _synthetic(description: str, unit_price: float) -> dict:
    """
    Generate deterministic synthetic metadata.
    Same product name always produces the same values (seeded random).
    """
    seed = sum(ord(c) for c in description) % 100_000
    rng  = random.Random(seed)
    rating   = round(rng.uniform(3.8, 4.9), 1)
    reviews  = rng.randint(500, 25_000)
    # Strike price varies 30–80% above unit price for visual variety
    multiplier = rng.uniform(1.30, 1.80)
    strike   = round(unit_price * multiplier, 2)
    return {
        "thumbnail":    _placeholder_url(description, seed=seed),
        "rating":       rating,
        "reviews":      reviews,
        "unit_price":   round(unit_price, 2),
        "strike_price": strike,
        "source":       "synthetic",
    }


def _fetch_serper(description: str, unit_price: float, api_key: str,
                  max_retries: int = 4) -> dict:
    """
    Query Serper.dev Google Shopping API for real product thumbnails.
    2,500 free queries on signup — https://serper.dev
    Retries on 429 with exponential backoff.
    """
    try:
        import requests
        import re as _re
        headers = {
            "X-API-KEY":     api_key,
            "Content-Type":  "application/json",
        }
        payload = {"q": description, "gl": "gb", "hl": "en", "num": 3}

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    SERPER_URL, headers=headers, json=payload, timeout=15
                )
                if resp.status_code == 429:
                    wait = (2 ** attempt) * 10 + random.uniform(0, 3)
                    print(f"    Rate limited (429) — waiting {wait:.0f}s "
                          f"before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                print("    Timed out — using synthetic")
                return _synthetic(description, unit_price)
        else:
            print("    Exhausted retries (429) — using synthetic")
            return _synthetic(description, unit_price)

        results = resp.json().get("shopping", [])
        if not results:
            print("    No results — using synthetic")
            return _synthetic(description, unit_price)

        item      = results[0]
        thumbnail = item.get("imageUrl") or _placeholder_url(description)
        rating    = float(item.get("rating") or random.uniform(3.8, 4.9))
        reviews   = int(item.get("ratingCount") or random.randint(500, 15_000))

        # Price field is a string like "£3.99" or "$2.49" — extract float
        price_str = item.get("price", "")
        price_num = _re.sub(r"[^\d.]", "", price_str)
        ex_price  = float(price_num) if price_num else unit_price

        return {
            "thumbnail":    thumbnail,
            "rating":       round(min(5.0, max(1.0, rating)), 1),
            "reviews":      reviews,
            "unit_price":   round(ex_price, 2),
            "strike_price": round(ex_price * 1.5, 2),
            "source":       "serper",
        }
    except ImportError:
        print("    'requests' not installed — pip install requests")
        return _synthetic(description, unit_price)
    except Exception as exc:
        print(f"    Serper error: {exc} — using synthetic")
        return _synthetic(description, unit_price)


def _fetch_serpapi(description: str, unit_price: float, api_key: str,
                   max_retries: int = 4) -> dict:
    """
    Query SerpApi Google Shopping for product visual metadata.
    Retries on 429 with exponential backoff. Falls back to synthetic on
    persistent failure.
    """
    try:
        import requests  # optional dependency — only needed for SerpApi mode
        params = {
            "engine":  "google_shopping",
            "q":       description,
            "api_key": api_key,
            "num":     3,
            "gl":      "gb",   # UK locale
            "hl":      "en",
        }
        for attempt in range(max_retries):
            try:
                resp = requests.get(SERPAPI_URL, params=params, timeout=15)
                if resp.status_code == 429:
                    wait = (2 ** attempt) * 10 + random.uniform(0, 3)
                    print(f"    Rate limited (429) — waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                print(f"    Timed out after {max_retries} attempts — using synthetic")
                return _synthetic(description, unit_price)
        else:
            print(f"    Exhausted retries (429) — using synthetic")
            return _synthetic(description, unit_price)

        results = resp.json().get("shopping_results", [])
        if not results:
            print(f"    No results — using synthetic")
            return _synthetic(description, unit_price)

        item      = results[0]
        thumbnail = item.get("thumbnail") or _placeholder_url(description)
        rating    = float(item.get("rating") or random.uniform(3.8, 4.9))
        reviews   = int(item.get("reviews") or random.randint(500, 15_000))
        ex_price  = float(item.get("extracted_price") or unit_price)

        return {
            "thumbnail":    thumbnail,
            "rating":       round(min(5.0, max(1.0, rating)), 1),
            "reviews":      reviews,
            "unit_price":   round(ex_price, 2),
            "strike_price": round(ex_price * 1.5, 2),
            "source":       "serpapi",
        }
    except ImportError:
        print("    'requests' not installed — pip install requests")
        return _synthetic(description, unit_price)
    except Exception as exc:
        print(f"    SerpApi error: {exc} — using synthetic")
        return _synthetic(description, unit_price)


def _read_products(filepath: str) -> dict:
    """
    Return {description: avg_unit_price} for every valid product in the CSV.
    """
    acc: dict = {}
    with open(filepath, encoding="latin-1") as f:
        for row in csv.DictReader(f):
            desc  = row.get("Description", "").strip()
            price = row.get("UnitPrice", "").strip()
            if not desc or not price:
                continue
            try:
                p = float(price)
                if p > 0:
                    acc.setdefault(desc, []).append(p)
            except ValueError:
                pass
    return {d: round(sum(v) / len(v), 2) for d, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build product_metadata.json from online_retail.csv"
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N unique products (default: all)",
    )
    ap.add_argument(
        "--synthetic-only", action="store_true",
        help="Skip SerpApi entirely; generate deterministic synthetic metadata",
    )
    ap.add_argument(
        "--replace-synthetic", action="store_true",
        help="Re-fetch entries that currently have source='synthetic' using SerpApi",
    )
    ap.add_argument(
        "--regen-synthetic", action="store_true",
        help="Regenerate all source='synthetic' entries with fresh synthetic data "
             "(useful after changing placeholder URLs without needing to strip the file)",
    )
    ap.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to wait between SerpApi requests (default: 0.5)",
    )
    args = ap.parse_args()

    # ── Load existing cache ────────────────────────────────────────────────────
    cache: dict = {}
    cp = Path(METADATA_FILE)
    if cp.exists():
        with open(cp) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache):,} cached entries from {METADATA_FILE}")

    # ── Read CSV ───────────────────────────────────────────────────────────────
    if not Path(DATA_FILE).exists():
        print(f"ERROR: {DATA_FILE} not found in current directory.")
        return

    products = _read_products(DATA_FILE)
    print(f"Found {len(products):,} unique products in {DATA_FILE}")

    # ── Decide API provider ────────────────────────────────────────────────────
    # SERPER_KEY takes priority (2,500 free queries, no card required)
    # Fall back to SERPAPI_KEY if SERPER_KEY is not set.
    serper_key  = os.getenv("SERPER_KEY", "")
    serpapi_key = os.getenv("SERPAPI_KEY", "")

    if args.synthetic_only:
        use_api, api_provider, active_key = False, None, ""
        print("Mode: synthetic-only (--synthetic-only flag)")
    elif serper_key:
        use_api, api_provider, active_key = True, "serper", serper_key
        print(f"Mode: Serper.dev Google Shopping (key: {serper_key[:8]}...)")
    elif serpapi_key:
        use_api, api_provider, active_key = True, "serpapi", serpapi_key
        print(f"Mode: SerpApi Google Shopping (key: {serpapi_key[:8]}...)")
    else:
        use_api, api_provider, active_key = False, None, ""
        print("Mode: synthetic-only (no SERPER_KEY or SERPAPI_KEY set)")

    # ── Enrichment loop ────────────────────────────────────────────────────────
    items = list(products.items())
    if args.limit:
        items = items[: args.limit]

    # "real" sources that should NOT be re-fetched unless --replace-synthetic
    _real_sources = {"serper", "serpapi"}

    added = skipped = 0
    for idx, (desc, avg_price) in enumerate(items, 1):
        if desc in cache:
            # Regenerate in-place if --regen-synthetic and entry is synthetic
            if args.regen_synthetic and cache[desc].get("source") == "synthetic":
                cache[desc] = _synthetic(desc, cache[desc].get("unit_price", avg_price))
                added += 1
                if added % 50 == 0:
                    with open(cp, "w") as f:
                        json.dump(cache, f, indent=2)
                    print(f"    Checkpoint: {len(cache):,} entries saved.")
                continue
            # Skip real entries; skip synthetic unless --replace-synthetic + api
            is_real    = cache[desc].get("source") in _real_sources
            want_refetch = (
                args.replace_synthetic
                and cache[desc].get("source") == "synthetic"
                and use_api
            )
            if is_real or not want_refetch:
                skipped += 1
                continue

        print(f"  [{idx}/{len(items)}] {desc[:70]}")
        if use_api:
            if api_provider == "serper":
                cache[desc] = _fetch_serper(desc, avg_price, active_key)
            else:
                cache[desc] = _fetch_serpapi(desc, avg_price, active_key)
            time.sleep(args.delay)
        else:
            cache[desc] = _synthetic(desc, avg_price)
        added += 1

        # Incremental checkpoint every 50 new entries
        if added % 50 == 0:
            with open(cp, "w") as f:
                json.dump(cache, f, indent=2)
            print(f"    Checkpoint: {len(cache):,} entries saved.")

    # ── Final save ─────────────────────────────────────────────────────────────
    with open(cp, "w") as f:
        json.dump(cache, f, indent=2)

    print(
        f"\nDone!  Added: {added:,}  |  Skipped (cached): {skipped:,}  "
        f"|  Total in {METADATA_FILE}: {len(cache):,}"
    )
    if not use_api:
        print(
            "\nTo enrich with real product images, set SERPER_KEY and re-run:\n"
            "  export SERPER_KEY='your_key'   # 2,500 free at https://serper.dev\n"
            "  python enrich_metadata.py --replace-synthetic\n"
        )


if __name__ == "__main__":
    main()
