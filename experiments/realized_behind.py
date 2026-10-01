"""Share of frames on which each forecaster's realized log score is below the climatology's.

s_t = log p_m(y_t) - log r_t(y_t) on every test issue frame of the exp10 seed split (same frames
and scoring as outputs/final/code/eval_oos_tpp.py; climatology scored as its full mixture), at
leads 1, 5, 10, 15. Frames are split by whether the target frame lies in an evaluated event
window (t0 < t <= t0 + min(150, frames to the next evaluated event)) from the run's records.

    ../.venv/bin/python realized_behind.py SUMMARY_JSON RECORDS_CSV OUT_JSON
"""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd, exp10_information_react as e10, react_information as ri
from concurrent.futures import ProcessPoolExecutor
from reactmetric.core.trajectory import TrajectorySet
from reactmetric.reference import Climatology

H = 15; LEADS = (1, 5, 10, 15); CAP = 150


def work(args):
    tracks, specs, ref, windows = args
    models = e10.build_models(specs)
    acc = {}
    def add(m, h, where, behind):
        a = acc.setdefault((m, h, where), [0, 0]); a[0] += int(np.sum(behind)); a[1] += int(np.size(behind))
    for traj in tracks:
        o = np.asarray(traj.positions, float); T = len(o)
        preds = {m: f.predict(traj) for m, f in models.items()}
        full = [m for m in specs if specs[m][2] is None]
        if any(preds[m] is None for m in full): continue
        common = set(preds[full[0]].issue_frames.tolist())
        for m in full[1:]: common &= set(preds[m].issue_frames.tolist())
        common = np.array(sorted(i for i in common if i + H - 1 < T and i - 1 - 5 >= 0))
        if len(common) == 0: continue
        tgt = common[:, None] + np.arange(H)[None, :]
        y = o[tgt]
        inwin = np.zeros(T, bool)
        for a, b in windows.get(str(traj.id) if hasattr(traj, "id") else "", []):
            inwin[a:b] = True
        clim = {}
        for h in LEADS:
            f, nll = ref.nll_trace(traj, h); mp = dict(zip(f.tolist(), (-nll).tolist()))
            clim[h] = np.array([mp[x] for x in tgt[:, h - 1]])
        for m, p in preds.items():
            if p is None or specs[m][2] is not None: continue
            idx = np.searchsorted(p.issue_frames, common)
            mu, S = p.means[idx], p.covs[idx] + 1e-9 * np.eye(2)
            ll = ri.gaussian_logpdf(y, mu, S)
            mix = getattr(p, "mix", None)
            if m.endswith("_mix") and mix is not None:
                jx = np.searchsorted(mix["issue_frames"], common)
                ll = np.stack([ri.MixtureForecast(mix["weights"][jx], mix["means"][jx, h], mix["covs"][jx, h]).logpdf(y[:, h])
                               for h in range(H)], 1)
            for h in LEADS:
                behind = ll[:, h - 1] < clim[h]
                w = inwin[tgt[:, h - 1]]
                add(m, h, "all", behind); add(m, h, "event", behind[w]); add(m, h, "calm", behind[~w])
    return acc


if __name__ == "__main__":
    summ = json.load(open(sys.argv[1])); rec_path = sys.argv[2]; out = sys.argv[3]
    a = summ["args"]
    if a.get("dataset") == "day-in-the-life":
        from day_in_the_life import load_day_in_the_life
        full, _ = load_day_in_the_life()
    else:
        from reactmetric.datasets.eindhoven import load_eindhoven
        full = load_eindhoven(path=a["csv"])
    ids = list(full.ids); rng = np.random.default_rng(a["seed"])
    data = TrajectorySet({t: full[t] for t in [ids[i] for i in rng.choice(len(ids), a["tracks"], replace=False)]})
    calib, test = data.split(fraction=0.3, seed=a["seed"])
    cap = not a.get("no_still_cap", False)
    src = e10.cap_still_runs(calib)[0] if cap else {t: np.asarray(calib[t].positions, float) for t in calib.ids}
    ref = Climatology.fit(src, leads=list(range(1, H + 1)), mode="kinematic", velocity_window=1, stop_model="fitted",
                          dt=calib.dt, seed=a["seed"])
    tpp = dict(tpp_dir=a["tpp_dir"], model_dir=a["tpp_model"], checkpoint=a["tpp_ckpt"]) if a.get("tpp_dir") else None
    specs, _ = e10.fit_models(calib, a["gru_dir"], a.get("gru_at5_dir"), a["imm_params"], tpp, a.get("ctrv_params"))
    ev = pd.read_csv(rec_path, usecols=["lead", "model", "track", "t0"])
    ev = ev[(ev["lead"] == 5) & (ev["model"] == "oracle_B")][["track", "t0"]].drop_duplicates().sort_values(["track", "t0"])
    windows = {}
    for tr, g in ev.groupby("track"):
        t0 = g["t0"].to_numpy(int); nxt = np.append(t0[1:], 10 ** 9)
        windows[str(tr)] = [(int(a_) + 1, int(min(a_ + CAP, b_)) + 1) for a_, b_ in zip(t0, nxt)]
    tl = []
    for t in test.ids:
        tr = test[t]
        try:
            tr.id = t
        except Exception:
            pass
        tl.append(tr)
    # windows keyed by the test id string; pass the id explicitly through the wrapper attribute above
    with ProcessPoolExecutor(4) as ex:
        parts = list(ex.map(work, [(tl[i::4], specs, ref, windows) for i in range(4)]))
    acc = {}
    for p in parts:
        for k, (b, n) in p.items(): x = acc.setdefault(k, [0, 0]); x[0] += b; x[1] += n
    res = {}
    for (m, h, where), (b, n) in sorted(acc.items()):
        res.setdefault(m, {}).setdefault(str(h), {})[where] = {"behind": b / n if n else None, "frames": n}
    json.dump({"leads": LEADS, "window_cap": CAP, "models": res}, open(out, "w"), indent=1)
    for m, r in res.items():
        print(m, " ".join(f"L{h}: all {r[h]['all']['behind']:.2f} event {r[h]['event']['behind']:.2f} "
                          f"calm {r[h]['calm']['behind']:.2f}" for h in r))
