#!/usr/bin/env python3
"""
enrich_metadata.py
==================
Visual Metadata Bridge: online_retail.csv → product_metadata.json

Fetches product thumbnails, ratings, and review counts from SerpApi
Google Shopping for every unique product in the UK retail dataset.
Falls back to deterministic synthetic data when no API key is set.

Usage
-----
    # Full SerpApi enrichment (requires SERPAPI_KEY env var)
    export SERPAPI_KEY="your_serpapi_key_here"
    python enrich_metadata.py

    # Synthetic mode — no API key, instant, deterministic output
    python enrich_metadata.py --synthetic-only

    # Process only the first N unique products
    python enrich_metadata.py --limit 200

The script is fully idempotent: products already present in
product_metadata.json are skipped to preserve API credits.
A checkpoint is saved every 50 new entries.

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
    "source":       "serpapi" | "synthetic"
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

# ─── Category-aware placeholder images ────────────────────────────────────────
_PLACEHOLDERS = {
    "LIGHT":    "https://placehold.co/300x300/fff9c4/555555?text=Light",
    "LANTERN":  "https://placehold.co/300x300/fff9c4/555555?text=Lantern",
    "CANDLE":   "https://placehold.co/300x300/fff8e1/555555?text=Candle",
    "CLOCK":    "https://placehold.co/300x300/e3f2fd/555555?text=Clock",
    "ALARM":    "https://placehold.co/300x300/e3f2fd/555555?text=Clock",
    "BAG":      "https://placehold.co/300x300/fce4ec/555555?text=Bag",
    "JUMBO":    "https://placehold.co/300x300/f3e5f5/555555?text=Bag",
    "LUNCH":    "https://placehold.co/300x300/e1f5fe/555555?text=Lunch+Box",
    "BOX":      "https://placehold.co/300x300/f3e5f5/555555?text=Box",
    "CAKE":     "https://placehold.co/300x300/fbe9e7/555555?text=Cake",
    "HEART":    "https://placehold.co/300x300/fce4ec/555555?text=Heart",
    "JAM":      "https://placehold.co/300x300/fff9c4/555555?text=Jam",
    "VINTAGE":  "https://placehold.co/300x300/efebe9/555555?text=Vintage",
    "FRAME":    "https://placehold.co/300x300/f5f5f5/555555?text=Frame",
    "MUG":      "https://placehold.co/300x300/e8f5e9/555555?text=Mug",
    "CUP":      "https://placehold.co/300x300/e8f5e9/555555?text=Cup",
    "BASKET":   "https://placehold.co/300x300/efebe9/555555?text=Basket",
    "DOORMAT":  "https://placehold.co/300x300/efebe9/555555?text=Doormat",
    "CLOCK":    "https://placehold.co/300x300/e3f2fd/555555?text=Clock",
    "RETROSPOT":"https://placehold.co/300x300/ffebee/555555?text=Retrospot",
    "POLKADOT": "https://placehold.co/300x300/f3e5f5/555555?text=Polkadot",
    "DEFAULT":  "https://placehold.co/300x300/f5f5f5/999999?text=Product",
}


def _placeholder_url(description: str) -> str:
    """Return a category-matched placeholder image URL."""
    desc_up = description.upper()
    for keyword, url in _PLACEHOLDERS.items():
        if keyword in desc_up:
            return url
    return _PLACEHOLDERS["DEFAULT"]


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
        "thumbnail":    _placeholder_url(description),
        "rating":       rating,
        "reviews":      reviews,
        "unit_price":   round(unit_price, 2),
        "strike_price": strike,
        "source":       "synthetic",
    }


def _fetch_serpapi(description: str, unit_price: float, api_key: str) -> dict:
    """
    Query SerpApi Google Shopping for product visual metadata.
    Gracefully falls back to synthetic data on any error.
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
        resp = requests.get(SERPAPI_URL, params=params, timeout=12)
        resp.raise_for_status()
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

    # ── Decide API vs synthetic ────────────────────────────────────────────────
    api_key = os.getenv("SERPAPI_KEY", "")
    use_api = bool(api_key) and not args.synthetic_only
    if not use_api:
        mode = "synthetic-only (no SERPAPI_KEY set)" if not api_key else "synthetic-only (--synthetic-only flag)"
        print(f"Mode: {mode}")
    else:
        print(f"Mode: SerpApi enrichment (key: {api_key[:8]}...)")

    # ── Enrichment loop ────────────────────────────────────────────────────────
    items = list(products.items())
    if args.limit:
        items = items[: args.limit]

    added = skipped = 0
    for idx, (desc, avg_price) in enumerate(items, 1):
        if desc in cache:
            skipped += 1
            continue

        print(f"  [{idx}/{len(items)}] {desc[:70]}")
        if use_api:
            cache[desc] = _fetch_serpapi(desc, avg_price, api_key)
            time.sleep(0.5)   # respect SerpApi rate limits
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
            "\nTo enrich with real product images, set SERPAPI_KEY and re-run:\n"
            "  export SERPAPI_KEY='your_key'\n"
            "  python enrich_metadata.py"
        )


if __name__ == "__main__":
    main()
