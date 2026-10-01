#!/usr/bin/env python
"""Export a representative sample of a cached Eindhoven parquet shard to CSV.

Cowork sessions cannot reach Zenodo (and PyPI is blocked by org egress policy, so
pyarrow cannot be installed there). This script runs on the machine that already has
the cache and the project venv, and writes a CSV that any environment can read.

Usage (from the reactmetric directory):

    .venv/bin/python scripts/export_eindhoven_csv.py --tracks 20000

Writes data/eindhoven_<n>tracks.csv in the bundled-sample schema
(time_ms, object_identifier, x_position_mm, y_position_mm). Load it with
``reactmetric.datasets.load_eindhoven(path=...)``, or with stdlib ``csv`` +
``datasets.eindhoven._from_rows`` in a pandas-less environment.

Sampling. Tracks are selected uniformly at random (seeded) from those meeting
``--min-length``, stratified across the shard's 24 hours by default so the sample
covers the whole day. The previous version of this script kept the first N track
ids it encountered, which is a chronological prefix: on day02 that was 2372 tracks
spanning only midnight to 09:28, i.e. the quietest part of the day.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

CACHE = os.path.expanduser("~/.cache/reactmetric/eindhoven")
DAY_FILES = {
    "day01": "Eindhoven_centraal_trajectories_01_10_day01.parquet",
    "day02": "Eindhoven_centraal_trajectories_01_10_day02.parquet",
    "01_10": "Eindhoven_centraal_trajectories_days_01_10.parquet",
}
COLS = ["time_ms", "object_identifier", "x_position_mm", "y_position_mm"]


def _select_ids(ids, times, n_tracks, min_length, stratify, seed):
    """Choose up to n_tracks eligible track ids, optionally spread across hours."""
    counts = defaultdict(int)
    first_t = {}
    for tid, t in zip(ids, times):
        counts[tid] += 1
        if tid not in first_t or t < first_t[tid]:
            first_t[tid] = t

    eligible = np.array([t for t, c in counts.items() if c >= min_length], dtype=np.int64)
    print(f"  {len(counts)} tracks in shard, {len(eligible)} with >= {min_length} frames")
    if len(eligible) <= n_tracks:
        print(f"  keeping all {len(eligible)} eligible tracks (asked for {n_tracks})")
        return set(eligible.tolist())

    rng = np.random.default_rng(seed)
    if not stratify:
        return set(rng.choice(eligible, size=n_tracks, replace=False).tolist())

    # Stratify by the hour each track starts, so the sample covers the whole shard.
    t0 = min(first_t.values())
    hour = {t: int((first_t[t] - t0) // 3_600_000) for t in eligible.tolist()}
    buckets = defaultdict(list)
    for t in eligible.tolist():
        buckets[hour[t]].append(t)

    chosen: list[int] = []
    taken: set = set()
    order = sorted(buckets)
    # Water-filling: equal share per hour, redistributing what sparse hours cannot supply.
    remaining, pool = n_tracks, list(order)
    while pool and remaining > 0:
        share = max(1, remaining // len(pool))
        nxt = []
        for h in pool:
            avail = [t for t in buckets[h] if t not in taken]
            take = min(share, len(avail), remaining)
            if take:
                picked = rng.choice(avail, size=take, replace=False).tolist()
                chosen.extend(picked)
                taken.update(picked)
                remaining -= take
            if len(avail) > take:
                nxt.append(h)
        if len(nxt) == len(pool) and share == 0:
            break
        pool = nxt
    print(f"  stratified across {len(order)} hours, {len(chosen)} tracks selected")
    return set(chosen[:n_tracks])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=None, help="parquet shard to read")
    ap.add_argument("--days", default="day02", choices=sorted(DAY_FILES),
                    help="which cached shard to use when --src is not given")
    ap.add_argument("--tracks", type=int, default=20000, help="number of tracks to keep")
    ap.add_argument("--min-length", type=int, default=40, help="drop shorter tracks")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--no-stratify", action="store_true",
                    help="sample uniformly instead of spreading across hours")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    src = args.src or os.path.join(CACHE, DAY_FILES[args.days])
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(here, "data", f"eindhoven_{args.tracks}tracks.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    pf = pq.ParquetFile(src)
    missing = set(COLS) - set(pf.schema.names)
    if missing:
        raise SystemExit(f"{src} is missing columns: {sorted(missing)}")

    print(f"reading {src}")
    table = pq.read_table(src, columns=COLS)
    ids = table.column("object_identifier").to_numpy()
    times = table.column("time_ms").to_numpy()

    keep = _select_ids(ids, times, args.tracks, args.min_length,
                       not args.no_stratify, args.seed)

    id_type = table.schema.field("object_identifier").type
    mask = pc.is_in(
        table.column("object_identifier"),
        value_set=pa.array(sorted(keep), type=id_type),
    )
    sub = table.filter(mask).sort_by(
        [("object_identifier", "ascending"), ("time_ms", "ascending")]
    )
    pacsv.write_csv(sub.select(COLS), out)

    t = sub.column("time_ms").to_numpy()
    lengths = np.unique(sub.column("object_identifier").to_numpy(), return_counts=True)[1]
    size_mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}")
    print(f"  {len(keep)} tracks, {sub.num_rows} rows, {size_mb:.1f} MB")
    print(f"  median track length {int(np.median(lengths))} frames")
    print(f"  time span {(t.max() - t.min()) / 3.6e6:.2f} h of the shard's 24 h")


if __name__ == "__main__":
    main()
