"""Compare a (subset) exp10 run against the paper's reference outputs.

    python verify/compare.py NEW_RUN_DIR REFERENCE_RECORDS_CSV REFERENCE_SUMMARY_JSON

Records: the reference is filtered to the tracks present in the new run, both sides are
sorted on (lead, info, distortion, model, track, t0), strings/ints/bools must match exactly
and floats to rtol 1e-9 (NaN equals NaN). Summary: every section fitted on the calibration
tracks (reference_fit, model_fits, event_source, noise_floor, calibration) must match the
reference to the same tolerance; `results` is skipped because it aggregates over test tracks.
Exits 1 on any mismatch.
"""
import glob
import json
import math
import sys

import numpy as np
import pandas as pd

KEY = ["lead", "info", "distortion", "model", "track", "t0"]
SECTIONS = ["reference_fit", "model_fits", "event_source", "noise_floor", "calibration"]
RTOL = 1e-9


def close(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=0.0)


def diff_json(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if "wall" in str(k) or "clock" in str(k):
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present on one side only")
            else:
                diff_json(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_json(x, y, f"{path}[{i}]", out)
    elif not close(a, b):
        out.append(f"{path}: {a!r} vs {b!r}")


def main(new_dir, ref_records, ref_summary):
    failures = []
    new = pd.read_csv(glob.glob(f"{new_dir}/records_*.csv")[0])
    ref = pd.read_csv(ref_records)
    ref = ref[ref["track"].isin(set(new["track"]))]
    new, ref = (d.sort_values(KEY).reset_index(drop=True) for d in (new, ref))
    print(f"records: {len(new)} rows on {new['track'].nunique()} tracks; reference has {len(ref)} rows for them")
    if list(new.columns) != list(ref.columns):
        failures.append(f"columns differ: {set(new.columns) ^ set(ref.columns)}")
    elif len(new) != len(ref):
        failures.append(f"row count {len(new)} vs {len(ref)}")
    else:
        for c in new.columns:
            a, b = new[c].to_numpy(), ref[c].to_numpy()
            if new[c].dtype.kind == "f" and ref[c].dtype.kind == "f":
                bad = ~np.isclose(a, b, rtol=RTOL, atol=0.0, equal_nan=True)
            else:
                bad = ~((a == b) | (pd.isna(a) & pd.isna(b)))
            for i in np.flatnonzero(bad)[:5]:
                failures.append(f"records[{c}] row {dict(new.loc[i, KEY])}: {a[i]!r} vs {b[i]!r}")
            if bad.sum() > 5:
                failures.append(f"records[{c}]: {bad.sum()} mismatches in total")

    s_new = json.load(open(glob.glob(f"{new_dir}/summary_*.json")[0]))
    s_ref = json.load(open(ref_summary))
    for sec in SECTIONS:
        out = []
        diff_json(s_new.get(sec), s_ref.get(sec), sec, out)
        failures += out[:10]
    print(f"summary sections compared: {', '.join(SECTIONS)}")

    if failures:
        print("FAIL")
        print("\n".join(failures[:40]))
        sys.exit(1)
    print("PASS: identical to the paper's final run on these tracks")


if __name__ == "__main__":
    main(*sys.argv[1:4])
