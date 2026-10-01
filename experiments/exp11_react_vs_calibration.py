"""Exp 11: is REACT measuring something new? Forecast covariance sweep (TGF 2026 slide 21).

Scale a model's predictive covariance by q and measure, on the same events and the same shared
truth posterior as exp10 (below). The sweep is centred on each model's best-calibrated scale
q*, the scalar covariance scale that maximises its likelihood at this lead on the calibration
tracks (for a 2D Gaussian, q* = mean squared Mahalanobis distance / 2), and spans +-1 nat of
claimed information around it: scaling a 2D covariance by q changes the forecast's entropy by
log q nats, so q = q* e^(-u) claims u nats more information than the calibrated forecast. The
as-shipped model (q = 1) is added to the grid. The measures:

  * the pre-event mean information gain: mean realized score over the climatology (the real
    mixture) in the PRE_WINDOW frames before each event, the kind of number NLL reports;
  * REACT under the headline rule: mean and median over recovered events, the restricted mean
    over all events, and the anticipation rate.

If REACT tracked average information it would move with the pre-event gain as q varies. The
slide's finding was that it barely moves: REACT measures accuracy on the temporal axis.

Reads the exp10 summary for the reference, split, model fits and calibration, so it runs on
exactly the same setup. Run from here:

    ../.venv/bin/python exp11_react_vs_calibration.py \\
        ../outputs/exp10_mixture/summary_seed0_n1200_vw1_uncapped.json --jobs 8
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

warnings.filterwarnings("ignore")

import exp10_information_react as e10  # noqa: E402
import numpy as np  # noqa: E402
import react_information as ri  # noqa: E402

from reactmetric.core.trajectory import TrajectorySet  # noqa: E402
from reactmetric.metric.survival import kaplan_meier, km_median  # noqa: E402

NATS = np.linspace(-1.0, 1.0, 9)  # claimed information relative to the calibrated scale
# Trajectron is included only when the exp10 summary was made with --tpp-dir
MODELS = ("CTRV", "IMM", "GRUPosition", "Trajectron")


def models_in(specs):
    return tuple(m for m in MODELS if m in specs)


def qstar_chunk(args):
    """Sums of squared Mahalanobis distances and counts per model at one lead."""
    tracks, ref, specs, lead = args
    names = models_in(specs)
    models = e10.build_models({m: specs[m] for m in names})
    out = {m: [0.0, 0] for m in names}
    for traj in tracks:
        T = len(traj.positions)
        obs = np.asarray(traj.positions, float)
        for m in names:
            mf = e10.model_frames(models[m].predict(traj), T, lead, ref)
            if mf is None:
                continue
            tm, mu, S = mf
            d = obs[tm] - mu
            out[m][0] += float(np.einsum("ni,nij,nj->", d, np.linalg.inv(S), d))
            out[m][1] += len(d)
    return out


def work(args):
    tracks, ref, specs, lead, level, eps, log_k, table_dir, det_spec, decision, q_grid, sig2 = args
    info, c, rho_a, rho_b = level
    det = e10.make_detector(det_spec)
    names = models_in(specs)
    models = e10.build_models({m: specs[m] for m in names})
    nodes, gw = ri.gh_nodes_2d(e10.GH_NODES)
    table = ri.OverlapTable.load(e10.table_path(table_dir, lead, info))
    out = []
    for traj in tracks:
        events = sorted(det.detect(traj), key=lambda e: e.t0)
        if not events:
            continue
        T = len(traj.positions)
        rm = e10.reference_mixture(traj, ref, lead)
        if rm is None:
            continue
        target, y, wts, mus, covs = rm
        speed, heading, anchor = e10.frame_geometry(traj, ref, lead, target)
        mu_r, S_r = ri.mixture_moments(wts, mus, covs)
        w = e10.whiten(y, mu_r, S_r)
        # whitened variance of the measurement-noise floor sigma^2 I (as in exp10)
        sn_fr = sig2 * 0.5 * np.trace(np.linalg.inv(S_r), axis1=1, axis2=2)
        scale = np.exp(ri.mixture_entropy(wts, mus, covs, nodes, gw) - ri.gaussian_entropy(S_r))
        ref_lp = ri.mixture_logpdf(y[:, None], ri._mix_prep(wts, mus, covs))[:, 0]
        mfr = {m: e10.model_frames(models[m].predict(traj), T, lead, ref) for m in names}
        for k_ev, ev in enumerate(events):
            nxt = events[k_ev + 1].t0 if k_ev + 1 < len(events) else np.inf
            sel = (target > ev.t0) & (target < nxt) & (target <= ev.t0 + e10.WINDOW_CAP)
            if sel.sum() < e10.MIN_WINDOW:
                continue
            pre = (target >= ev.t0 - e10.PRE_WINDOW) & (target < ev.t0)
            if k_ev > 0:
                pre &= target > events[k_ev - 1].t0
            fw, mrw, Srw, ww = target[sel], mu_r[sel], S_r[sel], w[sel]
            ct = c * scale[sel]
            sn_t = sn_fr[sel]
            va = ri.knowable_share(ct, sn_t)[:, None, None]
            mf_, vf_, _, _ = ri.truth_posterior_het(fw, ww, rho_a, rho_b, ct, sn_t)
            pmu, pC = ri.to_world(mf_, vf_, mrw, Srw)
            tests = ri.RegimeTestsExact(
                pmu, pC, mrw, va * Srw, wts[sel], mus[sel], covs[sel], c,
                **e10.OVERLAP, table=table, geom=(speed[sel], heading[sel], anchor[sel]), sig2=sig2,
            )
            for m in names:
                if mfr[m] is None:
                    continue
                tm, mu_m, S_m = mfr[m]
                _, ia, ib = np.intersect1d(fw, tm, return_indices=True)
                if len(ia) < e10.MIN_WINDOW:
                    continue
                _, pa, pb = np.intersect1d(target[pre], tm, return_indices=True)
                for q in q_grid[m]:
                    lbf_on, lbf_rec, _ = tests.test(ia, mu_m[ib], q * S_m[ib], eps, decision=decision)
                    res = ri.classify(
                        lbf_on,
                        lbf_rec,
                        fw[ia],
                        ev.t0,
                        log_k,
                        onset_window=lead + e10.ONSET_MARGIN,
                        first_verdict=False,
                    )
                    pre_s = (
                        ri.gaussian_logpdf(y[pre][pa], mu_m[pb], q * S_m[pb]) - ref_lp[pre][pa]
                        if len(pa)
                        else np.array([])
                    )
                    out.append(
                        {
                            "model": m,
                            "q": float(q),
                            "status": res["status"],
                            "duration": res["duration"],
                            "pre_sum": float(pre_s.sum()),
                            "pre_n": int(len(pre_s)),
                        }
                    )
    return out


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], round(r["q"], 6)), []).append(r)
    res = []
    for (m, q), rs in sorted(groups.items()):
        rec = np.array([r["duration"] for r in rs if r["status"] == "recovered"], float)
        st = np.array([r["status"] for r in rs])
        onset = np.isin(st, ("recovered", "censored"))
        d = np.array([r["duration"] if r["duration"] is not None else 0.0 for r in rs], float)
        t_, s_ = kaplan_meier(np.where(onset, d, 0.0), (st == "recovered") | ~onset)
        t_c = np.append(np.minimum(t_, e10.WINDOW_CAP), e10.WINDOW_CAP)
        n_pre = sum(r["pre_n"] for r in rs)
        res.append(
            {
                "model": m,
                "q": q,
                "events": len(rs),
                "pre_gain": sum(r["pre_sum"] for r in rs) / n_pre if n_pre else float("nan"),
                "react_mean_uncensored": float(rec.mean()) if len(rec) else float("nan"),
                "react_median_uncensored": float(np.median(rec)) if len(rec) else float("nan"),
                "react_rmst": float(np.sum(np.diff(t_c) * s_)),
                "react_median_all": float(km_median(t_, s_)),
                "anticipation": float(np.mean(~onset)),
                "censored": float(np.mean(st == "censored")),
            }
        )
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("summary", help="exp10 summary JSON (reference, split, fits, calibration)")
    ap.add_argument("--lead", type=int, default=5)
    ap.add_argument("--info", type=float, default=2.0)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    S = json.load(open(args.summary))
    a = S["args"]
    from reactmetric.datasets.eindhoven import load_eindhoven

    full = load_eindhoven(path=a["csv"])
    ids = list(full.ids)
    rng = np.random.default_rng(a["seed"])
    data = TrajectorySet(
        {t: full[t] for t in [ids[i] for i in rng.choice(len(ids), a["tracks"], replace=False)]}
    )
    calib, test = data.split(fraction=0.3, seed=a["seed"])
    ref, _ = e10.fit_reference(
        calib,
        a["velocity_window"],
        a["seed"],
        cap_still=not a.get("no_still_cap"),
        stop_isotropic=not a.get("stop_anisotropic"),
    )
    tpp = (
        dict(tpp_dir=a["tpp_dir"], model_dir=a["tpp_model"], checkpoint=a["tpp_ckpt"])
        if a.get("tpp_dir")
        else None
    )
    specs, _ = e10.fit_models(calib, a["gru_dir"], None, a["imm_params"], tpp)
    names = models_in(specs)
    fit = S["calibration"][str(args.lead)]["fits"][f"{args.info:.3f}"]
    rho_b = ri.overlap_corr(args.lead, e10.CORR_LEN)
    level = (
        args.info,
        fit["c"],
        ri.knowable_corr(e10.CORR_LEN, fit["scale"], fit["tau"], fit["omega"]),
        rho_b,
    )
    cl = [calib[t] for t in calib.ids]
    with ProcessPoolExecutor(args.jobs) as ex:
        parts = list(ex.map(qstar_chunk, [(cl[i :: args.jobs], ref, specs, args.lead) for i in range(args.jobs)]))
    qstar = {
        m: sum(p[m][0] for p in parts) / (2.0 * max(sum(p[m][1] for p in parts), 1)) for m in names
    }
    q_grid = {m: np.unique(np.append(qstar[m] * np.exp(-NATS), 1.0)) for m in names}
    print("calibrated scales q*: " + ", ".join(f"{m} {q:.3f}" for m, q in qstar.items()), flush=True)
    tl = [test[t] for t in test.ids]
    ev = S.get("event_source", {"detector": "pelt-low"})
    det_spec = (
        (ev["detector"], ev["hazard"]) if ev["detector"] in ("bayes", "bayes-sg") else ("pelt", "low")
    )
    sig2 = (S.get("noise_floor", {}).get("sigma_cm", 0.0) / 100.0) ** 2
    decision = a.get("decision", "bayes-factor")
    log_k = float(np.log(a["p"] / (1 - a["p"])) if decision == "posterior" else np.log(a["k"]))
    jobs = [
        (tl[i :: args.jobs], ref, specs, args.lead, level, a["eps"], log_k,
         os.path.join(os.path.dirname(args.summary), "tables"), det_spec, decision, q_grid, sig2)
        for i in range(args.jobs)
    ]
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = [r for part in ex.map(work, jobs) for r in part]
    res = summarize(rows)
    out = os.path.join(
        os.path.dirname(args.summary), f"qsweep_lead{args.lead}_info{args.info:g}.json"
    )
    json.dump(
        {
            "lead": args.lead,
            "info": args.info,
            "qstar": qstar,
            "q_grid": {m: g.tolist() for m, g in q_grid.items()},
            "results": [
                {**r, "claimed_nats": float(-np.log(r["q"] / qstar[r["model"]]))} for r in res
            ],
        },
        open(out, "w"),
        indent=1,
    )
    for r in res:
        print(
            f"{r['model']:12s} q={r['q']:.3f} pre-gain {r['pre_gain']:+.2f}  "
            f"REACT recovered mean {r['react_mean_uncensored']:.1f} "
            f"median {r['react_median_uncensored']:.0f}  RMST {r['react_rmst']:.1f}  "
            f"anticip {r['anticipation']:.2f}"
        )
    print("wrote", out)


if __name__ == "__main__":
    main()
