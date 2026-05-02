#!/usr/bin/env python3
"""Convert all city data into one CSV per city, organised by state.

Two source formats are auto-detected:

  FORMAT A — nested year folders (*_scans_split):
    <City> -YYYY/
      1.cat.csv, 2.cat.csv, ...   columns: organization_category, social_organization, line_no, line

  FORMAT B — flat year CSVs (NN CityName dirs):
    CityName_YYYY.csv             columns: page, organization_name, category, [sub_category,] address, description

Output unified schema (written to <outdir>/<state>/<City>.csv):
  city, year, page, line_no, organization_name, category, sub_category, line, address, description

Usage:
  python3 scripts/preprocessing/merge_city_to_single_csv.py
  python3 scripts/merge_city_to_single_csv.py --root output/org_lines_cat --outdir output/city_merged
"""

from __future__ import annotations

import argparse
import re
import csv
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UNIFIED_COLS = [
    "city", "year", "page", "line_no",
    "organization_name", "category", "sub_category",
    "line", "address", "description",
]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _is_format_a(city_dir: Path) -> bool:
    """Format A: subdirectories named like 'City -YYYY'."""
    for sub in city_dir.iterdir():
        if sub.is_dir() and re.search(r"-\s*\d{4}\s*$", sub.name):
            return True
    return False


def _is_format_b(city_dir: Path) -> bool:
    """Format B: CSV files named like 'CityName_YYYY.csv'."""
    return any(re.search(r"_\d{4}\.csv$", f.name) for f in city_dir.glob("*.csv"))


# ---------------------------------------------------------------------------
# City / state name normalisation
# ---------------------------------------------------------------------------

def _state_from_dirname(dirname: str) -> str:
    """'al_txt' → 'al',  'fl' → 'fl'."""
    return dirname.removesuffix("_txt").lower()


def _city_from_dirname(dirname: str) -> str:
    """Extract a clean city name from a directory name.

    'Birmingham_scans_split'  → 'Birmingham'
    'Los Angeles_scans_split' → 'Los Angeles'
    'Kansas_City_scans_split' → 'Kansas City'
    '12 Jacksonville'         → 'Jacksonville'
    '40 Colorado Springs'     → 'Colorado Springs'
    """
    name = dirname
    # Format A: strip _scans_split suffix, then replace _ with space
    if name.endswith("_scans_split"):
        name = name[: -len("_scans_split")]
        name = name.replace("_", " ").strip()
        return name
    # Format B: strip leading number
    name = re.sub(r"^\d+\s*", "", name).strip()
    return name


# ---------------------------------------------------------------------------
# Format A reader
# ---------------------------------------------------------------------------

def _parse_year_from_dirname(name: str) -> int | None:
    m = re.search(r"(\d{4})\s*$", name)
    return int(m.group(1)) if m else None


def merge_format_a(city_dir: Path, city_name: str) -> pd.DataFrame:
    """Read all N.cat.csv files under year-subfolders; return unified DataFrame."""
    rows: list[dict] = []
    for year_dir in sorted(city_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        year = _parse_year_from_dirname(year_dir.name)
        if year is None:
            continue
        for cat_file in sorted(year_dir.glob("*.cat.csv")):
            if "reasons" in cat_file.name:
                continue
            page_num = re.match(r"(\d+)", cat_file.name)
            page = int(page_num.group(1)) if page_num else None
            try:
                raw = cat_file.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore")
                reader = csv.DictReader(raw.splitlines())
                if not reader.fieldnames or "social_organization" not in reader.fieldnames:
                    continue
                for row in reader:
                    rows.append({
                        "city": city_name,
                        "year": year,
                        "page": page,
                        "line_no": row.get("line_no", ""),
                        "organization_name": (row.get("social_organization") or "").strip(),
                        "category": (row.get("organization_category") or "").strip(),
                        "sub_category": "",
                        "line": (row.get("line") or "").strip(),
                        "address": "",
                        "description": "",
                    })
            except Exception as e:
                print(f"  Warning: skipping {cat_file.name}: {e}")
    return pd.DataFrame(rows, columns=UNIFIED_COLS)


# ---------------------------------------------------------------------------
# Format B reader
# ---------------------------------------------------------------------------

def _parse_year_from_filename(name: str) -> int | None:
    m = re.search(r"_(\d{4})\.csv$", name)
    return int(m.group(1)) if m else None


def merge_format_b(city_dir: Path, city_name: str) -> pd.DataFrame:
    rows: list[dict] = []
    for year_file in sorted(city_dir.glob("*.csv")):
        year = _parse_year_from_filename(year_file.name)
        if year is None:
            continue
        try:
            df = pd.read_csv(year_file, dtype=str).fillna("")
            for _, row in df.iterrows():
                rows.append({
                    "city": city_name,
                    "year": year,
                    "page": row.get("page", ""),
                    "line_no": "",
                    "organization_name": row.get("organization_name", "").strip(),
                    "category": row.get("category", "").strip(),
                    "sub_category": row.get("sub_category", "").strip(),
                    "line": "",
                    "address": row.get("address", "").strip(),
                    "description": row.get("description", "").strip(),
                })
        except Exception as e:
            print(f"  Warning: skipping {year_file.name}: {e}")
    return pd.DataFrame(rows, columns=UNIFIED_COLS)


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def discover_cities(root: Path) -> list[dict]:
    """Walk root and return a list of {state, city_name, city_dir, format} dicts."""
    cities = []
    for state_dir in sorted(root.iterdir()):
        if not state_dir.is_dir():
            continue
        state = _state_from_dirname(state_dir.name)
        for city_dir in sorted(state_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            city_name = _city_from_dirname(city_dir.name)
            if _is_format_a(city_dir):
                fmt = "A"
            elif _is_format_b(city_dir):
                fmt = "B"
            else:
                print(f"  Skipping {state}/{city_dir.name}: unrecognised format")
                continue
            cities.append({
                "state": state,
                "city_name": city_name,
                "city_dir": city_dir,
                "format": fmt,
            })
    return cities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT / "output/org_lines_cat"),
        help="Root directory containing state/city folder tree",
    )
    parser.add_argument(
        "--outdir",
        default=str(PROJECT_ROOT / "output/city_merged"),
        help="Output root; state subdirectories are created automatically",
    )
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)

    cities = discover_cities(root)
    print(f"Found {len(cities)} cities across {len({c['state'] for c in cities})} states.\n")

    total_rows = 0
    for spec in cities:
        state = spec["state"]
        city_name = spec["city_name"]
        city_dir = spec["city_dir"]
        fmt = spec["format"]

        state_outdir = outdir / state
        state_outdir.mkdir(parents=True, exist_ok=True)
        out_path = state_outdir / f"{city_name}.csv"

        print(f"[{state}] {city_name} (format {fmt}) ...", end=" ", flush=True)
        try:
            if fmt == "A":
                df = merge_format_a(city_dir, city_name)
            else:
                df = merge_format_b(city_dir, city_name)

            df = df[df["organization_name"].str.strip().astype(bool)]
            df.to_csv(out_path, index=False)
            years = sorted(df["year"].dropna().unique().tolist())
            print(f"{len(df):,} rows, {len(years)} years ({years[0]}–{years[-1]})")
            total_rows += len(df)
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone. {total_rows:,} total rows written to {outdir}/")


if __name__ == "__main__":
    main()
