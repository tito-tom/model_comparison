from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path


def read_last_row(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None

    row = rows[-1]
    row["source"] = path

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--out", default="outputs/summary.csv")

    args = parser.parse_args()

    files = glob.glob(os.path.join(args.root, "**", "*_metrics.csv"), recursive=True)

    rows = [r for r in (read_last_row(p) for p in files) if r is not None]

    if not rows:
        print("No metric CSV files found.")
        return

    keys = sorted(set().union(*[set(r.keys()) for r in rows]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.out} with {len(rows)} rows")


if __name__ == "__main__":
    main()