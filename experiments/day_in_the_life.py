"""Standard AI "Day in the Life" public release as a REACT dataset (kept outside the package for now).

Source: huggingface.co/datasets/standard-cognition/day-in-the-life, file canonical_tracks.parquet
(1,817 person tracks in one store over a day, 10 Hz, planar position in metres; the position is
the middle of the waist for 99.8% of rows). Download it to ~/.cache/reactmetric/day_in_the_life/.

Cleaning: rows with a missing or invalid position are gaps; gaps of up to MAX_FILL frames are
filled by linear interpolation and longer ones split the track. Pieces longer than MAX_LEN
frames are cut into consecutive pieces (the changepoint detector is quadratic in track length),
and pieces shorter than MIN_LEN are dropped. Piece ids are "<person_track_id>:<k>".
"""

from __future__ import annotations

import os

import numpy as np

DEFAULT_PATH = os.path.expanduser("~/.cache/reactmetric/day_in_the_life/canonical_tracks.parquet")
MAX_FILL, MAX_LEN, MIN_LEN, DT = 5, 6000, 30, 0.1


def load_day_in_the_life(path=DEFAULT_PATH, max_fill=MAX_FILL, max_len=MAX_LEN, min_len=MIN_LEN):
    import pyarrow.parquet as pq

    from reactmetric.core.trajectory import Trajectory, TrajectorySet

    cols = ["timestamp", "person_track_id", "x", "y", "position_valid"]
    df = pq.read_table(path, columns=cols).to_pandas().sort_values(["person_track_id", "timestamp"])
    out, stats = {}, {"tracks": 0, "filled_frames": 0, "splits": 0, "cuts": 0, "dropped": 0}
    for tid, g in df.groupby("person_track_id", sort=False):
        stats["tracks"] += 1
        # place rows on the 10 Hz grid so missing timestamps are gaps too
        t = (g["timestamp"] - g["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
        k = np.round(t / DT).astype(int)
        n = k[-1] + 1
        xy = np.full((n, 2), np.nan)
        ok = g["position_valid"].to_numpy() & np.isfinite(g["x"].to_numpy()) & np.isfinite(g["y"].to_numpy())
        xy[k[ok]] = g[["x", "y"]].to_numpy()[ok]
        bad = ~np.isfinite(xy[:, 0])
        # runs of missing frames
        pieces, start, i = [], 0, 0
        while i < n:
            if bad[i]:
                j = i
                while j < n and bad[j]:
                    j += 1
                if i > 0 and j < n and j - i <= max_fill:
                    for c in range(2):
                        xy[i:j, c] = np.interp(np.arange(i, j), [i - 1, j], [xy[i - 1, c], xy[j, c]])
                    stats["filled_frames"] += j - i
                else:
                    if i > start:
                        pieces.append((start, i))
                    stats["splits"] += 1
                    start = j
                i = j
            else:
                i += 1
        if start < n:
            pieces.append((start, n))
        for a, b in pieces:
            for s in range(a, b, max_len):
                e = min(s + max_len, b)
                if s > a:
                    stats["cuts"] += 1
                if e - s < min_len:
                    stats["dropped"] += 1
                    continue
                pid = f"{tid}:{len([q for q in out if q.startswith(str(tid) + ':')])}"
                out[pid] = Trajectory(id=pid, positions=xy[s:e].copy(), dt=DT)
    return TrajectorySet(out), stats
