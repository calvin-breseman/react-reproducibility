"""Out-of-sample accuracy of every exp10 forecaster: log-likelihood by lead, ADE and FDE.

Test tracks of the exp10 seed-0 split. Every model is scored on the same issue frames
(where all models issue a forecast and the full 15-frame horizon is observed). ADE/FDE use
the forecast mean over the full horizon; log-likelihood is the Gaussian predictive density
(the climatology's is its full mixture).
"""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, exp10_information_react as e10, react_information as ri
from concurrent.futures import ProcessPoolExecutor
from reactmetric.core.trajectory import TrajectorySet
from reactmetric.datasets.eindhoven import load_eindhoven
from reactmetric.reference import Climatology
H = 15; LEADS = list(range(1, H + 1)); REPORT = (1, 5, 10)

def work(args):
    tracks, specs, ref = args
    models = e10.build_models(specs)
    acc = {}
    def add(k, v):
        a = acc.setdefault(k, [0.0, 0]); a[0] += float(np.sum(v)); a[1] += int(np.size(v))
    for traj in tracks:
        o = np.asarray(traj.positions, float); T = len(o)
        preds = {m: f.predict(traj) for m, f in models.items()}
        full = [m for m in specs if specs[m][2] is None]
        if any(preds[m] is None for m in full): continue
        common = set(preds[full[0]].issue_frames.tolist())
        for m in full[1:]: common &= set(preds[m].issue_frames.tolist())
        common = np.array(sorted(i for i in common if i + H - 1 < T and i - 1 - 5 >= 0))
        if len(common) == 0: continue
        tgt = common[:, None] + np.arange(H)[None, :]            # (n, H) target frames
        y = o[tgt]                                                # (n, H, 2)
        for m, p in preds.items():
            if p is None: continue
            idx = np.searchsorted(p.issue_frames, common)
            mu, S = p.means[idx], p.covs[idx] + 1e-9 * np.eye(2)
            ll = ri.gaussian_logpdf(y, mu, S)                     # (n, H)
            mix = getattr(p, "mix", None)
            if m.endswith("_mix") and mix is not None:            # full mixture log density
                jx = np.searchsorted(mix["issue_frames"], common)
                ll = np.stack([ri.MixtureForecast(mix["weights"][jx], mix["means"][jx, h], mix["covs"][jx, h]).logpdf(y[:, h])
                               for h in range(H)], 1)
            err = np.linalg.norm(y - mu, axis=-1)
            leads = [5] if specs[m][2] is not None else LEADS
            for h in leads:
                add((m, "ll", h), ll[:, h - 1]); add((m, "err", h), err[:, h - 1])
            if specs[m][2] is None:
                add((m, "ade", 0), err.mean(1)); add((m, "fde", 0), err[:, -1])
        # climatology: full mixture log-lik, mixture mean for errors (anchor = issue - 1)
        cerr = np.zeros((len(common), H))
        for h in LEADS:
            f, nll = ref.nll_trace(traj, h); mix = dict(zip(f.tolist(), (-nll).tolist()))
            fr = e10.reference_frames(traj, ref, h); mm = dict(zip(fr[0].tolist(), range(len(fr[0]))))
            t = tgt[:, h - 1]
            add(("Climatology", "ll", h), [mix[x] for x in t])
            cerr[:, h - 1] = np.linalg.norm(o[t] - fr[2][[mm[x] for x in t]], axis=-1)
            add(("Climatology", "err", h), cerr[:, h - 1])
        add(("Climatology", "ade", 0), cerr.mean(1)); add(("Climatology", "fde", 0), cerr[:, -1])
    return acc

if __name__ == "__main__":
    summ = json.load(open(sys.argv[1])); out = sys.argv[2]
    a = summ["args"]; fits = summ["model_fits"]
    if a.get("dataset") == "day-in-the-life":
        from day_in_the_life import load_day_in_the_life
        full, _ = load_day_in_the_life()
    else:
        full = load_eindhoven(path=a["csv"])
    ids = list(full.ids); rng = np.random.default_rng(a["seed"])
    data = TrajectorySet({t: full[t] for t in [ids[i] for i in rng.choice(len(ids), a["tracks"], replace=False)]})
    calib, test = data.split(fraction=0.3, seed=a["seed"])
    cap = not a.get("no_still_cap", False)
    src = e10.cap_still_runs(calib)[0] if cap else {t: np.asarray(calib[t].positions, float) for t in calib.ids}
    ref = Climatology.fit(src, leads=LEADS, mode="kinematic", velocity_window=1, stop_model="fitted", dt=calib.dt, seed=a["seed"])
    tpp = dict(tpp_dir=a["tpp_dir"], model_dir=a["tpp_model"], checkpoint=a["tpp_ckpt"]) if a.get("tpp_dir") else None
    specs, _ = e10.fit_models(calib, a["gru_dir"], a.get("gru_at5_dir"), a["imm_params"], tpp, a.get("ctrv_params"))
    tl = [test[t] for t in test.ids]
    with ProcessPoolExecutor(4) as ex: parts = list(ex.map(work, [(tl[i::4], specs, ref) for i in range(4)]))
    acc = {}
    for p in parts:
        for k, (s_, n) in p.items(): x = acc.setdefault(k, [0.0, 0]); x[0] += s_; x[1] += n
    res = {}
    for (m, what, h), (s_, n) in acc.items():
        res.setdefault(m, {})[f"{what}{h if h else ''}"] = s_ / n
        res[m]["n"] = max(res[m].get("n", 0), n if what in ("ade", "ll") else 0)
    for m, r in res.items():
        lls = [r[f"ll{h}"] for h in LEADS if f"ll{h}" in r]
        if len(lls) == H: r["ll_mean"] = float(np.mean(lls))
    json.dump({"horizon_frames": H, "dt": calib.dt, "reference_still_cap": cap, "models": res}, open(out, "w"), indent=1)
    print(f"{'model':18s} {'LL@1':>7} {'LL@5':>7} {'LL@10':>7} {'LL 1-15':>8} {'ADE cm':>7} {'FDE cm':>7}")
    for m, r in res.items():
        g = lambda k, s=1: f"{r[k]*s:7.2f}" if k in r else "      —"
        print(f"{m:18s} {g('ll1')} {g('ll5')} {g('ll10')} {g('ll_mean') if 'll_mean' in r else '       —'} {g('ade',100)} {g('fde',100)}")
