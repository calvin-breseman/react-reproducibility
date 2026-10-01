"""Exp 10: REACT in the information formulation (../docs/react-spec.md), on Eindhoven.

Uses the same track draw, split and 'low'-preset events as exp09 seed 0.

Reference: the kinematic climatology conditioned on the velocity over the last
--velocity-window frames (default 1: the last observed step), with its stopped/moving share
fitted by EM rather than a hand-set ramp, on calibration frames whose still runs are cut at
WINDOW_CAP frames (stillness REACT can never score does not train the reference). The
5-frame version is a calibrated forecaster in its own right, so it is kept only as an option.

Models: the Kalman constant-velocity filter and the IMM have their noise parameters fitted by
maximum likelihood on the calibration tracks' one-step innovations (ConstantVelocity.fit,
IMM.fit). CTRV keeps its hand-set parameters. GRU checkpoints can be supplied with
--gru-dir (a directory holding gru_position_best.pt and gru_orientation_best.pt); otherwise
the shipped retail-trained checkpoints are used.
Trajectron++ (--tpp-dir, a clone of the repository; the Eindhoven run-1 checkpoint is in
../artifacts/eindhoven/trajectron_run1) enters twice from one network pass: "Trajectron"
(moment-matched Gaussian) and "Trajectron_mix" (all 25 components, no pruning or merging, in the
realized score and the tests; react_information.MixtureForecast / RegimeTestsMixture).
The base case, recorded as distortion 0.0 (label "clim"), has no target: the climatology's own
spread is the bar; distortions > 0 are the REACT@d targets.

  1. Calibration tracks. Each model's mean realized score on calm stretches (reported only).
     Climatology's whitened residual autocorrelation rho_w(l), and for each assumed
     information level I (--levels, nats) the knowable correlation rho_a fitted net of the
     structural overlap.
  2. Test tracks. For every event window: the truth posterior (shared by all models); per model
     the per-frame onset and recovery Bayes factors -> onset, REACT, status, relapse; the
     perfect-forecaster floor; mean realized score vs mean posterior advantage (consistency).

Needs the CSV first (../data/make_eindhoven_csv.py). Run from here:

    ../.venv/bin/python exp10_information_react.py --tracks 1200 --seed 0 --jobs 8 \
        --gru-dir ../artifacts/eindhoven/gru_10hz
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from itertools import repeat  # noqa: E402

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import react_information as ri  # noqa: E402
from trajectron_forecaster import add_cli_args as add_tpp_args  # noqa: E402

from reactmetric import baselines  # noqa: E402
from reactmetric.core.predictive import Predictive  # noqa: E402
from reactmetric.core.trajectory import TrajectorySet  # noqa: E402
import bayes_changepoints as bcp  # noqa: E402
from reactmetric.events import KinematicChangepoints  # noqa: E402
from reactmetric.metric.survival import kaplan_meier, km_median  # noqa: E402
from reactmetric.reference import Climatology  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_CSV = os.path.join(ROOT, "data", "eindhoven_20000tracks.csv")
DEFAULT_OUT = os.path.join(ROOT, "outputs", "exp10")
# label of the base case (distortion 0.0): no target, the climatology's own spread
CLIM = "clim"

LEADS = (1, 5, 10, 15)
# IMM fit bounds: the hand-set TGF_2026 values +- a physical margin, so the modes stay a stop
# mode, a straight mode and two turns (unbounded, the turn modes become catch-all noise modes).
IMM_BOUNDS = {
    "measurement_noise": (0.001, 0.038),
    "turn_rate": (0.4, 0.8),
    "sigma_a_cv": (0.2, 0.6),
    "sigma_a_ct": (0.5, 0.9),
    "sigma_a_stop": (0.0, 0.25),
    "stop_velocity_decay": (0.05, 0.25),
    "p_stay": (0.94, 0.9999),
    "measurement_noise_stop": (0.001, 0.038),
    "p_stay_stop": (0.94, 0.9999),
}
MASK_PRE, MASK_POST = 20, 60  # calm stretches exclude [t0 - 20, t0 + 60], as exp06
# MIN_WINDOW: events with fewer target frames before the next event (or the cap) are skipped;
# MIN_FRAMES: the shortest track stretch the per-track frame builders accept (calibration too)
WINDOW_CAP, MIN_WINDOW, MIN_FRAMES, MIN_RUN = 150, 25, 15, 12
# Outcome rule recorded per event (docs/react-spec.md section 7): an onset counts only within
# lead + ONSET_MARGIN frames of the event, the frames whose failure it can explain, and only
# those frames are tested for it. Without an onset the event is anticipated; "better" then means
# a recovery test is confidently ahead somewhere in those same frames (so an event that is ahead
# only later in its window counts as indistinguishable). Recovery is searched after the onset,
# and a renewed onset counts as a relapse only in the ri.RELAPSE_WINDOW frames after recovery.
ONSET_MARGIN = 5
# outcome fields also recorded under the entropy margin (suffix _H; onset does not depend on it)
_H_FIELDS = ("status", "react", "duration", "relapse", "relapse_after", "relapse3",
             "relapse3_after")
# Gauss-Hermite order for the mixture entropy (the entropic truth spread in the shared
# machinery); frames before an event used for the pre-event score
GH_NODES, PRE_WINDOW = 5, 30
# exact overlap term: Gauss-Hermite nodes per rule, and the polar-grid stride it is computed on
OVERLAP = dict(overlap_gh=9)
# overlap tables: uniform core of +-TABLE_CORE stopped-regime sds at TABLE_STEP sds, then
# geometric growth by TABLE_GROWTH per node out to TABLE_REACH moving-regime sds
TABLE_CORE, TABLE_STEP, TABLE_GROWTH, TABLE_REACH = 8.0, 1.0 / 6.0, 1.06, 10.0
# target distortions (RMS metres) for REACT@d: recovery requires the forecaster to beat the
# climatology by the information needed to localize the person within d. The base case,
# recorded as distortion 0.0 (label "clim"), has no target: the climatology's own spread is the
# bar, so any improvement over it counts
DISTORTIONS = (0.1, 0.2, 0.3)
N_LAGS, FIT_LAGS = 30, 20
CORR_LEN = WINDOW_CAP + 1


# ---------------------------------------------------------------------------
# Per-track frames
# ---------------------------------------------------------------------------


def reference_frames(traj, ref, lead):
    """Target frames with climatology's moment-matched Gaussian (moving + stopped components)."""
    obs = np.asarray(traj.positions, float)
    T = len(obs)
    vw = int(getattr(ref, "velocity_window", 5))
    target = np.arange(lead + vw, T)
    if len(target) < MIN_FRAMES:
        return None
    anchors = target - lead
    vel = (obs[anchors] - obs[anchors - vw]) / (vw * ref.dt)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    lam = ref.move_weight(lead, speed)
    stop = np.asarray(ref.stop_cov[lead], float)
    mu = np.empty((len(target), 2))
    S = np.empty((len(target), 2, 2))
    for i in range(len(target)):
        mw, cw = ref.conditional_gaussian(lead, float(speed[i]), float(heading[i]))
        md = lam[i] * mw
        S[i] = lam[i] * (cw + np.outer(mw, mw)) + (1.0 - lam[i]) * stop - np.outer(md, md)
        mu[i] = obs[anchors[i]] + md
    return target, obs[target], mu, S + 1e-8 * np.eye(2)


def reference_mixture(traj, ref, lead):
    """Target frames with the climatology as its real two-part mixture (moving, stopped).

    Returns target, y, weights (n, 2), means (n, 2, 2) absolute, covs (n, 2, 2, 2). The same
    density as ref.nll_trace; reference_frames is its moment-matched Gaussian.
    """
    obs = np.asarray(traj.positions, float)
    T = len(obs)
    vw = int(getattr(ref, "velocity_window", 5))
    target = np.arange(lead + vw, T)
    if len(target) < MIN_FRAMES:
        return None
    anchors = target - lead
    vel = (obs[anchors] - obs[anchors - vw]) / (vw * ref.dt)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    mu_b, cov_b = ref._predict_moving(lead, speed)
    cs, sn = np.cos(heading), np.sin(heading)
    R = np.stack([np.stack([cs, -sn], -1), np.stack([sn, cs], -1)], -2)
    mu_w = np.einsum("nij,nj->ni", R, mu_b)
    cov_w = np.einsum("nij,njk,nlk->nil", R, cov_b, R)
    lam = np.asarray(ref.move_weight(lead, speed), float)
    weights = np.column_stack([lam + 1e-12, 1.0 - lam + 1e-12])
    weights /= weights.sum(1, keepdims=True)
    means = np.stack([obs[anchors] + mu_w, obs[anchors]], 1)
    stop = np.broadcast_to(np.asarray(ref.stop_cov[lead], float), cov_w.shape)
    covs = np.stack([cov_w, stop], 1) + 1e-8 * np.eye(2)
    return target, obs[target], weights, means, covs


def body_mixture(ref, lead):
    """The climatology in the body frame as a function of speed, for ri.OverlapTable.

    The stopped regime's spread is fixed in world axes and nearly isotropic (axis sds within
    5%); the table uses its isotropic average.
    """
    stop = np.asarray(ref.stop_cov[lead], float)
    stop_iso = 0.5 * np.trace(stop) * np.eye(2)

    def body(v):
        mu_b, cov_b = ref._predict_moving(lead, v)
        lam = np.asarray(ref.move_weight(lead, v), float)
        w = np.column_stack([lam + 1e-12, 1.0 - lam + 1e-12])
        w /= w.sum(1, keepdims=True)
        means = np.stack([mu_b, np.zeros_like(mu_b)], 1)
        covs = np.stack([cov_b, np.broadcast_to(stop_iso, cov_b.shape)], 1) + 1e-8 * np.eye(2)
        return w, means, covs

    return body


def table_setup(ref, lead):
    """Speed grid (the climatology's knots and midpoints) and body-frame axis for one lead."""
    knots = np.asarray(ref.speed_grid[lead], float)
    v = np.sort(np.concatenate([knots, 0.5 * (knots[1:] + knots[:-1])]))
    sd_stop = float(np.sqrt(0.5 * np.trace(ref.stop_cov[lead])))
    mu_b, cov_b = ref._predict_moving(lead, v)
    reach = np.abs(mu_b).max(1) + TABLE_REACH * np.sqrt(np.linalg.eigvalsh(cov_b)[:, 1])
    core = TABLE_CORE * sd_stop
    return v, core, TABLE_STEP * sd_stop, TABLE_GROWTH, max(float(reach.max()), 2.0 * core)


def build_table(job):
    ref, lead, info, c, path, sig2 = job
    t = time.time()
    v, core, h, growth, extent = table_setup(ref, lead)
    tab = ri.OverlapTable.build(
        body_mixture(ref, lead), c, v, core, h, growth, extent, sig2=sig2, **OVERLAP
    )
    tab.save(path)
    return lead, info, time.time() - t, tab.G.shape


def frame_geometry(traj, ref, lead, target):
    """Causal speed, heading and anchor position of the climatology at each target frame."""
    obs = np.asarray(traj.positions, float)
    vw = int(getattr(ref, "velocity_window", 5))
    anchors = target - lead
    vel = (obs[anchors] - obs[anchors - vw]) / (vw * ref.dt)
    return np.hypot(vel[:, 0], vel[:, 1]), np.arctan2(vel[:, 1], vel[:, 0]), obs[anchors]


def distortion_margin(entropy, d):
    """Rate-distortion margin (nats): the information beyond a climatology of differential
    entropy `entropy` needed to localize a 2D position at RMS error d (target N(0, d^2/2 I)),
    clamped at 0. The library's eps_star, with `entropy` in place of its Gaussian
    log-determinant.

    Headline (default) margin: `entropy` is that of the Gaussian with the climatology mixture's
    total mean and covariance (ri.gaussian_entropy of the mixture moments), 1 + log 2 pi + 1/2
    log det Sigma_mix in 2D, which includes the between-component spread. The mixture's own
    differential entropy (ri.mixture_entropy, recorded as the "_H" alternative) understates the
    requirement for a bimodal climatology: a standing person with a 22% chance of walking away
    looks localized within a few cm, but the position to be reported spans the walking range."""
    return np.maximum(entropy - (1.0 + np.log(2.0 * np.pi) + np.log(d * d / 2.0)), 0.0)


def oracle_outcome(first_ahead, last):
    """Oracle B's outcome: never behind the climatology, REACT the first confirmed frame."""
    return {
        "status": "recovered" if first_ahead is not None else "censored",
        "onset": None,
        "react": first_ahead,
        "duration": first_ahead if first_ahead is not None else last,
        "relapse": False,
        "relapse_after": None,
        "relapse3": False,
        "relapse3_after": None,
    }


def estimate_noise_floor(tracks, window=51, quad=9, still=0.10):
    """Measurement-noise floor sigma (m per axis): floor "A" of docs/react-spec.md.

    The root-mean-square residual about a `quad`-frame local quadratic fit (centre-frame
    residual, rescaled by 1 / sqrt(1 - leverage) to be unbiased for white noise), taken at
    frames whose `window`-frame neighbourhood is observed stationary (every position within
    `still` m of the window mean). Uses the tracks' observations only. Returns (sigma, n
    residuals per axis, n tracks used)."""
    from numpy.lib.stride_tricks import sliding_window_view as swv

    h = quad // 2
    t = np.arange(quad) - h
    X = np.stack([np.ones(quad), t, t**2], 1)
    P = X @ np.linalg.pinv(X)
    kern = (np.eye(quad)[h] - P[h])[::-1]
    scale = 1.0 / np.sqrt(1.0 - P[h, h])
    hb = window // 2
    res, used = [], 0
    for tid in tracks.ids:
        o = np.asarray(tracks[tid].positions, float)
        if len(o) < window:
            continue
        win = swv(o, window, axis=0)  # (T - window + 1, 2, window)
        mx = win.mean(-1)
        stat = np.abs(win - mx[..., None]).max((1, 2)) < still
        if not stat.any():
            continue
        used += 1
        for a in range(2):
            r = np.convolve(o[:, a], kern, mode="valid") * scale  # centre frame i + h
            res.append(r[hb - h : len(o) - hb - h][stat])
    r = np.concatenate(res) if res else np.zeros(1)
    return float(np.sqrt(np.mean(r**2))), int(len(r) // 2), used


def event_kinematics(obs, t0, dt, gap=5, span=10):
    """Speed before and after an event and the heading change across it, from displacements
    over `span` frames ending `gap` frames before t0 and starting `gap` frames after it."""
    a0, a1, b0, b1 = t0 - gap - span, t0 - gap, t0 + gap, t0 + gap + span
    if a0 < 0 or b1 >= len(obs):
        return None, None, None
    va, vb = (obs[a1] - obs[a0]) / (span * dt), (obs[b1] - obs[b0]) / (span * dt)
    sa, sb = float(np.hypot(*va)), float(np.hypot(*vb))
    turn = None
    if sa > 0.2 and sb > 0.2:
        turn = float(np.degrees(abs(np.angle(complex(*vb) / complex(*va)))))
    return sa, sb, turn


def table_path(table_dir, lead, info):
    return os.path.join(table_dir, f"overlap_lead{lead}_info{info:g}")


def model_frames(pred, T, lead, ref):
    if pred is None or len(pred) == 0:
        return None
    try:
        target, mu, S = pred.at_lead(lead)
    except ValueError:
        return None
    vw = int(getattr(ref, "velocity_window", 5))
    ok = (target < T) & (target - lead - vw >= 0)
    if ok.sum() < MIN_FRAMES:
        return None
    return target[ok].astype(int), mu[ok], S[ok] + 1e-9 * np.eye(2)


def whiten(y, mu, S):
    return np.linalg.solve(np.linalg.cholesky(S), (y - mu)[..., None])[..., 0]


def calm_mask(target, events):
    mask = np.ones(len(target), bool)
    for ev in events:
        mask &= ~((target >= ev.t0 - MASK_PRE) & (target <= ev.t0 + MASK_POST))
    return mask


def runs_of(values, mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    return [values[r] for r in np.split(idx, breaks + 1) if len(r) >= MIN_RUN]


class MixturePredictive(Predictive):
    """Moment-matched Gaussian forecast that also keeps the network's full mixture (raw arrays)."""

    def __init__(self, mix, horizon):
        from trajectron_forecaster import TrajectronForecaster

        lp = np.log(np.clip(mix["weights"], 1e-300, None))
        mu, cov = TrajectronForecaster.collapse(lp, mix["means"], mix["covs"])
        super().__init__(mix["issue_frames"], mu, cov, horizon)
        self.mix = mix


class Trajectron:
    """One network pass per track, shared by the moment-matched ("Trajectron") and full-mixture
    ("Trajectron_mix") entries of the specs."""

    def __init__(self, **kw):
        from trajectron_forecaster import TrajectronForecaster

        self.net, self._last = TrajectronForecaster(threads=1, **kw), (None, None)

    def predict(self, traj):
        if self._last[0] is not traj:
            mix = self.net.predict_mixture(traj)
            pred = None if mix is None else MixturePredictive(mix, self.net.horizon)
            self._last = (traj, pred)
        return self._last[1]


def mixture_frames(pred, T, lead, ref):
    """MixtureForecast (all components) over the frames model_frames keeps (same mask, order)."""
    mix = getattr(pred, "mix", None)
    if mix is None:
        return None
    target = mix["issue_frames"] + (lead - 1)
    ok = (target < T) & (target - lead - int(getattr(ref, "velocity_window", 5)) >= 0)
    if ok.sum() < MIN_FRAMES:
        return None
    w, mu, cov = mix["weights"][ok], mix["means"][ok, lead - 1], mix["covs"][ok, lead - 1]
    return ri.MixtureForecast(w, mu, cov)


def frame_logpdf(y, mu, S, fc=None):
    """Realized log density: Gaussian, or the full mixture (ri.mixture_logpdf) when fc is given."""
    if fc is None:
        return ri.gaussian_logpdf(y, mu, S)
    return fc.logpdf(y)


def predict_all(models, traj):
    out = {}
    for name, f in models.items():
        try:
            out[name] = f.predict(traj)
        except Exception:
            out[name] = None
    return out


def realized_score(y, mu_m, S_m, mu_r, S_r):
    return ri.gaussian_logpdf(y, mu_m, S_m) - ri.gaussian_logpdf(y, mu_r, S_r)


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------


def build_models(specs):
    out, tpp = {}, {}
    for m, (name, kw, _) in specs.items():
        if name == "Trajectron":  # "Trajectron" and "Trajectron_mix" share one network
            key = json.dumps(kw, sort_keys=True)
            tpp.setdefault(key, Trajectron(**kw))
            out[m] = tpp[key]
        else:
            out[m] = baselines.build(name, **kw)
    return out


def scored_at(specs, m, lead):
    """Models trained for one lead only (e.g. GRU@5) are scored only there."""
    leads = specs[m][2]
    return leads is None or lead in leads


def make_detector(spec):
    """Event source: ("bayes-sg", hazard) tuned Bayes with Savitzky-Golay smoothing,
    ("bayes", hazard) the earlier Bayes settings, or ("pelt", preset)."""
    kind, arg = spec
    if kind == "bayes-sg":
        return bcp.BayesianChangepoints.tuned_sg(hazard=arg)
    if kind == "bayes":
        return bcp.BayesianChangepoints(hazard=arg)
    return KinematicChangepoints.from_preset(arg)


def calib_chunk(tracks, ref, specs, det_spec):
    det = make_detector(det_spec)
    models = build_models(specs)
    names = list(specs)
    nodes, gw = ri.gh_nodes_2d(GH_NODES)
    out = {
        lead: {
            "runs": [],
            "s_sum": dict.fromkeys(names, 0.0),
            "s_n": dict.fromkeys(names, 0),
            "scale_sum": 0.0,
            "scale_n": 0,
            "sn_sum": 0.0,
            "margin_samples": [],
        }
        for lead in LEADS
    }
    for traj in tracks.values():
        events = det.detect(traj)
        preds = predict_all(models, traj)
        T = len(traj.positions)
        for lead in LEADS:
            fr = reference_frames(traj, ref, lead)
            if fr is None:
                continue
            target, y, mu_r, S_r = fr
            calm = calm_mask(target, events)
            out[lead]["runs"].extend(runs_of(whiten(y, mu_r, S_r), calm))
            if calm.any():
                _, _, wts, mus, covs = reference_mixture(traj, ref, lead)
                h_mix = ri.mixture_entropy(wts[calm], mus[calm], covs[calm], nodes, gw)
                h_cov = ri.gaussian_entropy(S_r[calm])
                scale = np.exp(h_mix - h_cov)
                out[lead]["scale_sum"] += float(scale.sum())
                out[lead]["scale_n"] += int(calm.sum())
                # mean whitening scale of an isotropic unit noise, 1/2 tr S_r^-1 (floor A)
                out[lead]["sn_sum"] += float(
                    (0.5 * np.trace(np.linalg.inv(S_r[calm]), axis1=1, axis2=2)).sum()
                )
                out[lead]["margin_samples"].append(
                    np.column_stack([wts[calm][:, 1], h_mix, h_cov])[::5]
                )
            for m in names:
                mf = model_frames(preds[m], T, lead, ref) if scored_at(specs, m, lead) else None
                if mf is None:
                    continue
                tm, mu_m, S_m = mf
                _, ia, ib = np.intersect1d(target, tm, return_indices=True)
                keep = calm[ia]
                ia, ib = ia[keep], ib[keep]
                if len(ia) == 0:
                    continue
                s = realized_score(y[ia], mu_m[ib], S_m[ib], mu_r[ia], S_r[ia])
                out[lead]["s_sum"][m] += float(s.sum())
                out[lead]["s_n"][m] += len(s)
    return out


_WORKER = {}  # per-process cache: models and overlap tables are built once, not once per task


def _cached(key, make):
    if key not in _WORKER:
        _WORKER[key] = make()
    return _WORKER[key]


def test_chunk(tracks, ref, levels, eps, log_k, specs, table_dir=None, tests_kind="exact",
               det_spec=("bayes-sg", bcp.TUNED_SG["hazard"]), decision="posterior",
               distortions=DISTORTIONS, sig=0.0, entropy_levels=None):
    """Every event window: shared truth posterior, then each model and Oracle B are tested.
    Returns (records, clock).

    The reference enters as its real two-regime mixture (moving, stopped) through
    regime-conditioned tests (react_information.RegimeTestsExact): given the regime, the truth
    keeps a share c of that regime's spread; the advantage is a quadratic minus the mixture's
    overlap term, evaluated per truth location, and tail probabilities are integrated on a
    polar grid. Oracle B is the forecaster equal to the inferred truth,
    N(smoothed posterior mean, c s_t S_mm,t), the most any forecast can extract at the
    assumed information level.

    Only frames that can decide an event are tested (react_information.classify_lazy): onset on
    the first lead + ONSET_MARGIN frames, recovery in blocks after the onset, renewed onset only
    in the RELAPSE_WINDOW frames after a recovery. The shared per-window work (truth posterior,
    RegimeTestsExact construction) still covers the whole window.

    entropy_levels: information levels at which the "_H" (mixture-entropy margin) outcomes are
    computed (None: every level); elsewhere the _H fields are None.
    """
    det = make_detector(det_spec)
    models = _cached(("models", json.dumps(specs, sort_keys=True, default=str)),
                     lambda: build_models(specs))
    nodes, gw = ri.gh_nodes_2d(GH_NODES)
    sig2 = float(sig) ** 2
    tables = {}
    if tests_kind == "exact":
        for lead in LEADS:
            for info, *_ in levels.get(lead, ()):
                p_ = table_path(table_dir, lead, info)
                tables[(lead, info)] = _cached(("table", p_), lambda p_=p_: ri.OverlapTable.load(p_))
    clock = {"predict": 0.0, "tests shared": 0.0, "tests per model": 0.0, "off-table points": 0,
             "events": 0, "model rows": 0}
    t_chunk = time.time()
    records = []
    dd = tuple(distortions)
    nd = len(dd)
    key = "post" if decision == "posterior" else "lbf"
    for tid, traj in tracks.items():
        events = sorted(det.detect(traj), key=lambda e: e.t0)
        if not events:
            continue
        t_ = time.time()
        preds = predict_all(models, traj)
        clock["predict"] += time.time() - t_
        T = len(traj.positions)
        obs_all = np.asarray(traj.positions, float)
        kin = {ev.t0: event_kinematics(obs_all, ev.t0, traj.dt) for ev in events}
        for lead in LEADS:
            rm = reference_mixture(traj, ref, lead)
            if rm is None or not levels.get(lead):
                continue
            target, y, wts, mus, covs = rm
            speed, heading, anchor = frame_geometry(traj, ref, lead, target)
            mu_r, S_r = ri.mixture_moments(wts, mus, covs)
            w = whiten(y, mu_r, S_r)
            H_ref = ri.mixture_entropy(wts, mus, covs, nodes, gw)
            H_cov = ri.gaussian_entropy(S_r)
            scale = np.exp(H_ref - H_cov)
            # whitened variance of the isotropic measurement noise sigma^2 I (floor A)
            sn_fr = sig2 * 0.5 * np.trace(np.linalg.inv(S_r), axis1=1, axis2=2)
            # default margin: Gaussian entropy of the mixture's total covariance; "_H" alternative:
            # the mixture's own entropy
            margin = {d: distortion_margin(H_cov, d) for d in dd}
            margin_H = {d: distortion_margin(H_ref, d) for d in dd}
            ref_lp = ri.mixture_logpdf(y[:, None], ri._mix_prep(wts, mus, covs))[:, 0]
            mfr = {
                m: model_frames(preds[m], T, lead, ref) if scored_at(specs, m, lead) else None
                for m in specs
            }
            mixfc = {
                m: mixture_frames(preds[m], T, lead, ref)
                if tests_kind == "exact" and m.endswith("_mix") and mfr[m] is not None
                else None
                for m in specs
            }
            for k_ev, ev in enumerate(events):
                nxt = events[k_ev + 1].t0 if k_ev + 1 < len(events) else np.inf
                sel = (target > ev.t0) & (target < nxt) & (target <= ev.t0 + WINDOW_CAP)
                if sel.sum() < MIN_WINDOW:
                    continue
                fw, yw, mrw, Srw, ww = target[sel], y[sel], mu_r[sel], S_r[sel], w[sel]
                pre = (target >= ev.t0 - PRE_WINDOW) & (target < ev.t0)
                if k_ev > 0:
                    pre &= target > events[k_ev - 1].t0
                clock["events"] += 1
                for info, c, rho_a, rho_b in levels[lead]:
                    with_h = entropy_levels is None or round(float(info), 4) in entropy_levels
                    ct = c * scale[sel]
                    sn_t = sn_fr[sel]
                    va = ri.knowable_share(ct, sn_t)[:, None, None]
                    mf_, vf_, ms_, vs_ = ri.truth_posterior_het(fw, ww, rho_a, rho_b, ct, sn_t)
                    pmu, pC = ri.to_world(mf_, vf_, mrw, Srw)
                    smu, sC = ri.to_world(ms_, vs_, mrw, Srw)
                    t_ = time.time()
                    targs = (pmu, pC, mrw, va * Srw, wts[sel], mus[sel], covs[sel], c)
                    if tests_kind == "exact":
                        tests = ri.RegimeTestsExact(
                            *targs, **OVERLAP, table=tables[(lead, info)],
                            geom=(speed[sel], heading[sel], anchor[sel]), sig2=sig2,
                        )
                        clock["off-table points"] += tests.n_off
                    else:
                        tests = ri.RegimeTests(*targs, sig2=sig2)
                    clock["tests shared"] += time.time() - t_
                    # Oracle B: the hindsight posterior predictive under the regime truth model
                    ob_mu, ob_S = ri.regime_predictive(
                        smu, sC, mrw, va * Srw, wts[sel], mus[sel], covs[sel], c, sig2
                    )
                    mtests = (
                        ri.RegimeTestsMixture.from_exact(tests, gh=5, stride=(8, 2))
                        if any(f is not None for f in mixfc.values())
                        else None
                    )
                    rows = [("oracle_B", np.arange(len(fw)), ob_mu, ob_S, None, None)]
                    for m in specs:
                        if mfr[m] is None:
                            continue
                        tm, mu_m, S_m = mfr[m]
                        _, ia, ib = np.intersect1d(fw, tm, return_indices=True)
                        if len(ia) >= MIN_WINDOW:
                            fcm = mixfc[m]
                            rows.append(
                                (
                                    m, ia, mu_m[ib], S_m[ib], (tm, mu_m, S_m, fcm),
                                    fcm.take(ib) if fcm is not None else None,
                                )
                            )
                    for m, ia, mu_x, S_x, full, fc_x in rows:
                        clock["model rows"] += 1
                        # recovery channels: 0 the base, 1..nd one per distortion under the
                        # default (covariance) margin, then the entropy ("_H") margin for those
                        # distortions where it differs from it on this window (hch[j]: the
                        # channel behind distortion j's _H outcome)
                        mg = [None] + [margin[d][sel][ia] for d in dd]
                        hch = list(range(nd + 1))
                        if with_h:
                            for jd, d in enumerate(dd):
                                mh = margin_H[d][sel][ia]
                                if not np.allclose(mg[jd + 1], mh, atol=1e-3):
                                    hch[jd + 1] = len(mg)
                                    mg.append(mh)

                        def evaluate(idx, chs, need_on):
                            t_ = time.time()
                            base = need_on or 0 in chs
                            mch = [ch for ch in chs if ch != 0]
                            margs = [mg[ch][idx] for ch in mch]
                            if fc_x is None:
                                v = tests.test_full(ia[idx], mu_x[idx], S_x[idx], eps,
                                                    margins=margs, base=base)
                            else:
                                v = mtests.test_full_mixture(ia[idx], fc_x.take(idx), eps,
                                                             margins=margs, base=base)
                            clock["tests per model"] += time.time() - t_
                            sfx = ["" if ch == 0 else f"_m{mch.index(ch)}" for ch in chs]
                            out = {
                                "rec": np.array([v[f"{key}_rec{s}"] for s in sfx]).reshape(
                                    len(chs), len(idx)),
                                "lbf_rec": np.array([v[f"lbf_rec{s}"] for s in sfx]).reshape(
                                    len(chs), len(idx)),
                            }
                            if base:
                                out.update(on=v[f"{key}_on"], lbf_on=v["lbf_on"],
                                           mean_I=v["mean_I"])
                            return out

                        F = fw[ia]
                        oracle = m == "oracle_B"
                        outs, fas, st = ri.classify_lazy(
                            evaluate, F, ev.t0, log_k, len(mg), lead + ONSET_MARGIN, oracle=oracle
                        )
                        k_on = st["k"]  # mean I over the onset-window frames, tested for all
                        I_sum = float(np.nansum(st["mean_I"][:k_on]))
                        lp = frame_logpdf(yw[ia], mu_x, S_x, fc_x)
                        pre_sum, pre_n = 0.0, 0
                        if full is not None and pre.any():
                            tm, mu_m, S_m, fc_f = full
                            _, pa, pb = np.intersect1d(target[pre], tm, return_indices=True)
                            if len(pa):
                                pre_sum = float(
                                    (
                                        frame_logpdf(
                                            y[pre][pa],
                                            mu_m[pb],
                                            S_m[pb],
                                            fc_f.take(pb) if fc_f is not None else None,
                                        )
                                        - ref_lp[pre][pa]
                                    ).sum()
                                )
                                pre_n = int(len(pa))
                        sb, sa, turn = kin[ev.t0]
                        last = int(F[-1] - ev.t0)
                        for j, dist in enumerate((0.0,) + dd):
                            if oracle:
                                # the inferred truth is never behind the climatology; its REACT
                                # is the earliest frame its advantage can be confirmed (the floor)
                                fa, fah = fas[j], fas[hch[j]]
                                first_ahead = None if fa is None else int(F[fa] - ev.t0)
                                first_ahead_H = None if fah is None else int(F[fah] - ev.t0)
                                res = oracle_outcome(first_ahead, last)
                                res_h = oracle_outcome(first_ahead_H, last) if with_h else None
                                on_pd = None
                                rec_pd = (
                                    bool(st["lbf_rec"][j, fa] < 0) if fa is not None else None
                                )
                            else:
                                first_ahead = first_ahead_H = None  # only Oracle B needs them
                                res = outs[j]
                                res_h = outs[hch[j]] if with_h else None
                                # prior-driven verdicts: the decision fired although the
                                # trajectory's evidence pointed the other way (Bayes factor < 1)
                                o_ = st["onset"]
                                on_pd = bool(st["lbf_on"][o_] < 0) if o_ is not None else None
                                r_ = st["react"][j]
                                rec_pd = bool(st["lbf_rec"][j, r_] < 0) if r_ is not None else None
                            records.append(
                                {
                                    "lead": lead,
                                    "info": round(float(info), 4),
                                    "distortion": float(dist),
                                    "distortion_label": f"{dist:g}m" if dist else CLIM,
                                    "model": m,
                                    "track": str(tid),
                                    "t0": int(ev.t0),
                                    "event_type": ev.event_type,
                                    "speed_before": sb,
                                    "speed_after": sa,
                                    "heading_change": turn,
                                    "effect_speed": float(ev.d_speed_level),
                                    "margin_mean": (
                                        float(margin[dist][sel][ia].mean()) if dist else 0.0
                                    ),
                                    "margin_mean_H": (
                                        float(margin_H[dist][sel][ia].mean()) if dist else 0.0
                                    ),
                                    "n": int(len(ia)),
                                    **res,
                                    **{f"{k}_H": (res_h[k] if res_h else None) for k in _H_FIELDS},
                                    "s_mix_sum": float((lp - ref_lp[sel][ia]).sum()),
                                    "s_sum": float(
                                        (lp - ri.gaussian_logpdf(yw[ia], mrw[ia], Srw[ia])).sum()
                                    ),
                                    "I_sum": I_sum,
                                    "I_n": int(k_on),
                                    "first_ahead": first_ahead,
                                    "first_ahead_H": first_ahead_H,
                                    "onset_prior_driven": on_pd,
                                    "recovery_prior_driven": rec_pd,
                                    "pre_s_sum": pre_sum,
                                    "pre_n": pre_n,
                                }
                            )
    clock["chunk total"] = time.time() - t_chunk
    return records, clock


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def cap_still_runs(tset, cap=WINDOW_CAP, stop_speed=0.2, window=5):
    """Calibration tracks with every still run cut at ``cap`` frames.

    REACT never scores more than WINDOW_CAP frames after an event, so stillness beyond
    that is evidence the metric cannot use; letting it train the reference would make the
    reference tighter than anything the evaluated frames substantiate. A frame is still
    when its ``window``-frame speed is under ``stop_speed``; each still run keeps its first
    ``cap`` frames and the track is split around the removed part.
    """
    out, dropped, total = {}, 0, 0
    for tid in tset.ids:
        obs = np.asarray(tset[tid].positions, float)
        total += len(obs)
        speed = np.full(len(obs), np.inf)
        speed[window:] = np.hypot(*(obs[window:] - obs[:-window]).T) / (window * tset.dt)
        edges = np.flatnonzero(np.diff(np.r_[0, (speed < stop_speed).astype(int), 0]))
        keep = np.ones(len(obs), bool)
        for a, b in zip(edges[::2], edges[1::2]):
            keep[a + cap : b] = False
        dropped += int((~keep).sum())
        cut = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0]))
        for k, (a, b) in enumerate(zip(cut[::2], cut[1::2])):
            if b - a > 20:
                out[f"{tid}_{k}"] = obs[a:b]
    return out, dropped / max(total, 1)


def fit_reference(calib, velocity_window, seed, cap_still=True, stop_isotropic=True):
    """Kinematic climatology for the headline reference.

    velocity_window 1: the stopped/moving share is fitted by EM (stop_model="fitted") on
    calibration frames with still runs capped at WINDOW_CAP. velocity_window 5: the old
    hand-set ramp on all calibration frames, kept for comparison.

    stop_isotropic: replace the stopped regime's fitted spread by its isotropic average. The
    fitted spread is fixed in world axes and 5-8% wider along one of them; the isotropic one
    scores slightly better on held-out frames (+0.0001 to +0.0006 nats per frame on Eindhoven)
    and makes the climatology depend on the speed alone in the mover's body frame, which the
    overlap tables of the exact tests rely on.
    """
    if velocity_window == 5:
        return Climatology.fit(calib, leads=LEADS, mode="kinematic"), None
    if cap_still:
        capped, share = cap_still_runs(calib)
    else:
        capped, share = {t: np.asarray(calib[t].positions, float) for t in calib.ids}, 0.0
    ref = Climatology.fit(
        capped,
        leads=LEADS,
        mode="kinematic",
        velocity_window=velocity_window,
        stop_model="fitted",
        dt=calib.dt,
        seed=seed,
    )
    if stop_isotropic:
        for lead in LEADS:
            ref.stop_cov[lead] = 0.5 * np.trace(np.asarray(ref.stop_cov[lead], float)) * np.eye(2)
    info = {
        "stop_model": "fitted",
        "stop_isotropic": stop_isotropic,
        "still_run_cap_frames": WINDOW_CAP if cap_still else None,
        "calibration_frames_dropped": share,
        "stop_sd_cm": {
            lead: float(100 * np.sqrt(np.trace(ref.stop_cov[lead]) / 2)) for lead in LEADS
        },
        "p_moving_at_zero_speed": {lead: float(ref.move_weight_grid[lead][0]) for lead in LEADS},
    }
    return ref, info


def fit_models(calib, gru_dir, gru_at5_dir=None, imm_params=None, tpp=None, ctrv_params=None,
               tpp_gaussian=True):
    """Model specs (baseline name, kwargs, leads or None) with noise fitted on calibration.

    imm_params: optional JSON from an earlier IMM.fit with IMM_BOUNDS (the fit takes ~15 min).
    ctrv_params: optional JSON with sigma_a, sigma_yaw, measurement_noise (default: hand-set CTRV).
    tpp_gaussian: False drops the moment-matched "Trajectron" entry, keeping "Trajectron_mix".
    """
    cv = baselines.ConstantVelocity.fit(calib)
    if imm_params:
        imm_fit = json.load(open(imm_params))
        imm_kw = {k: float(imm_fit[k]) for k in baselines.IMM._FIT_PARAMS if k in imm_fit}
    else:
        imm = baselines.IMM.fit(calib, bounds=IMM_BOUNDS)
        imm_kw = {k: float(getattr(imm, k)) for k in baselines.IMM._FIT_PARAMS}
        imm_fit = {**imm_kw, **imm.fit_result_}
    specs = {
        "ConstantVelocity": (
            "ConstantVelocity",
            {"measurement_noise": cv.measurement_noise, "accel_noise": cv.accel_noise},
            None,
        ),
        "CTRV": (
            "CTRV",
            {k: float(json.load(open(ctrv_params))[k]) for k in ("sigma_a", "sigma_yaw", "measurement_noise")}
            if ctrv_params
            else {},
            None,
        ),
        "IMM": ("IMM", imm_kw, None),
        "GRUPosition": ("GRUPosition", {}, None),
        "GRUOrientation": ("GRUOrientation", {}, None),
    }
    if gru_dir:
        specs["GRUPosition"][1]["checkpoint"] = os.path.join(gru_dir, "gru_position_best.pt")
        specs["GRUOrientation"][1]["checkpoint"] = os.path.join(gru_dir, "gru_orientation_best.pt")
    if gru_at5_dir:
        for m, f in (
            ("GRUPosition", "gru_position_best.pt"),
            ("GRUOrientation", "gru_orientation_best.pt"),
        ):
            specs[f"{m}@5"] = (m, {"checkpoint": os.path.join(gru_at5_dir, f)}, (5,))
    if tpp:  # tpp = dict(tpp_dir=clone, model_dir=..., checkpoint=epoch or None)
        if tpp_gaussian:
            specs["Trajectron"] = ("Trajectron", dict(tpp), None)
        specs["Trajectron_mix"] = ("Trajectron", dict(tpp), None)
    fitted = {
        "ConstantVelocity": {
            **specs["ConstantVelocity"][1],
            **cv.fit_result_,
            "effective_memory_frames": cv.effective_memory(calib.dt),
        },
        "IMM": imm_fit,
    }
    return specs, fitted


def split_chunks(tset, n):
    ids = list(tset.ids)
    return [{t: tset[t] for t in ids[i::n]} for i in range(n) if ids[i::n]]


def small_chunks(tset, size):
    """Chunks of about `size` tracks (strided, so long and short tracks mix)."""
    return split_chunks(tset, max(1, -(-len(tset.ids) // size)))


def km_summary(rs, sfx=""):
    """Anticipation and REACT medians for one group under one outcome rule.

    Conditional: Kaplan-Meier over events with an onset (time to recovery given the
    forecast fell behind). Overall: every event, anticipated ones (no onset: confidently
    better or indistinguishable) recovering at frame 0, the atom at zero. Besides medians,
    the restricted mean (area under the Kaplan-Meier curve up to WINDOW_CAP frames; Royston
    & Parmar 2013), which stays informative when over half the events are anticipated.
    """

    def rmst(t_, s_):
        t_ = np.append(np.minimum(t_, WINDOW_CAP), WINDOW_CAP)
        return float(np.sum(np.diff(t_) * s_))

    if not rs:  # e.g. the "_H" outcomes at levels where the entropy margin was not computed
        return dict.fromkeys(
            ("recovered", "censored", "better", "indistinguishable", "anticipation", "km_react",
             "km_react_overall", "rmst", "rmst_cond"), float("nan"))
    st = np.array([r["status" + sfx] for r in rs])
    dur = np.array(
        [np.nan if r["duration" + sfx] in (None, "") else float(r["duration" + sfx]) for r in rs]
    )
    onset = np.isin(st, ("recovered", "censored"))
    out = {
        "recovered": float(np.mean(st == "recovered")),
        "censored": float(np.mean(st == "censored")),
        "better": float(np.mean(st == "better")),
        "indistinguishable": float(np.mean(st == "indistinguishable")),
        "anticipation": float(np.mean(~onset)),
        "km_react": float("nan"),
        "km_react_overall": float("nan"),
        "rmst": float("nan"),
        "rmst_cond": float("nan"),
    }
    if onset.any():
        t_, s_ = kaplan_meier(dur[onset], st[onset] == "recovered")
        out["km_react"] = float(km_median(t_, s_))
        out["rmst_cond"] = rmst(t_, s_)
    d_all = np.where(onset, dur, 0.0)
    obs_all = (st == "recovered") | ~onset
    t_, s_ = kaplan_meier(d_all, obs_all)
    out["km_react_overall"] = float(km_median(t_, s_))
    out["rmst"] = rmst(t_, s_)
    return out


def summarize(records):
    """Per (lead, information level, distortion, model) group. Groups exist only where records
    do (lead 1 may be evaluated at fewer levels); the "_H" fields are NaN where the entropy
    margin was not computed. Relapse rates are over recovered events, within ri.RELAPSE_WINDOW
    frames after the recovery; mean_I is over the onset-window frames (tested for every event)."""
    groups = {}
    for r in records:
        groups.setdefault((r["lead"], r["info"], r.get("distortion", 0.0), r["model"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        onset = [r for r in rs if r["status"] in ("recovered", "censored")]
        rec = [r for r in rs if r["status"] == "recovered"]
        n_frames = sum(r["n"] for r in rs)
        n_pre = sum(r["pre_n"] for r in rs)
        n_I = sum(r["I_n"] for r in rs)
        rs_h = [r for r in rs if r["status_H"] is not None]
        under_h = {f"{k}_H": v for k, v in km_summary(rs_h, "_H").items()}
        rec_h = [r for r in rs_h if r["status_H"] == "recovered"]
        on0 = [r["onset"] for r in onset if r["onset"] is not None]
        out[key] = {
            "events": len(rs),
            **km_summary(rs),
            **under_h,
            "median_onset": float(np.median(on0)) if on0 else float("nan"),
            "relapse": float(np.mean([r["relapse"] for r in rec])) if rec else float("nan"),
            "relapse_after_median": (
                float(np.median([r["relapse_after"] for r in rec if r["relapse"]]))
                if any(r["relapse"] for r in rec)
                else float("nan")
            ),
            "relapse3": float(np.mean([r["relapse3"] for r in rec])) if rec else float("nan"),
            "relapse3_after_median": (
                float(np.median([r["relapse3_after"] for r in rec if r["relapse3"]]))
                if any(r["relapse3"] for r in rec)
                else float("nan")
            ),
            "relapse_H": float(np.mean([r["relapse_H"] for r in rec_h])) if rec_h else float("nan"),
            "relapse3_H": (
                float(np.mean([r["relapse3_H"] for r in rec_h])) if rec_h else float("nan")
            ),
            "margin_mean": sum(r["margin_mean"] * r["n"] for r in rs) / n_frames,
            "margin_mean_H": sum(r["margin_mean_H"] * r["n"] for r in rs) / n_frames,
            "mean_s_mixture": sum(r["s_mix_sum"] for r in rs) / n_frames,
            "mean_s_gaussian": sum(r["s_sum"] for r in rs) / n_frames,
            "mean_pre_s": sum(r["pre_s_sum"] for r in rs) / n_pre if n_pre else float("nan"),
            "mean_I": sum(r["I_sum"] for r in rs) / n_I if n_I else float("nan"),
        }
    return out


def fmt(x, w=6, d=0):
    return f"{'—':>{w}}" if x is None or not np.isfinite(x) else f"{x:>{w}.{d}f}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument(
        "--dataset",
        choices=["eindhoven", "day-in-the-life"],
        default="eindhoven",
        help="day-in-the-life: Standard AI public release (waist positions), see day_in_the_life.py",
    )
    ap.add_argument("--tracks", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument(
        "--decision",
        choices=("posterior", "bayes-factor"),
        default="posterior",
        help="onset/recovery on posterior probability (default) or on Bayes factors",
    )
    ap.add_argument("--p", type=float, default=0.95, help="posterior-probability threshold")
    ap.add_argument(
        "--distortions",
        type=float,
        nargs="*",
        default=list(DISTORTIONS),
        help="target distortions d (m) for REACT@d; the base case with no target (the "
        "climatology's own spread, label 'clim', distortion 0.0) is always included",
    )
    ap.add_argument("--k", type=float, default=20.0, help="Bayes-factor threshold")
    ap.add_argument(
        "--events",
        choices=("bayes-sg", "bayes", "pelt-low"),
        default="bayes-sg",
        help="event source: tuned Bayesian changepoints on Savitzky-Golay smoothed positions "
        "(default; speed, turn, vx, vy channels, fixed tuned hazard), the earlier Bayesian "
        "changepoints (speed and turn, empirical-Bayes hazard), or the hand-set PELT preset",
    )
    ap.add_argument(
        "--hazard",
        type=float,
        default=None,
        help="changepoint hazard (default: 9e-4 for bayes-sg, empirical Bayes for bayes)",
    )
    ap.add_argument(
        "--noise-floor",
        default="A",
        help="measurement-noise floor of the truth spread, Omega = c S + sigma^2 I: 'A' (default) "
        "estimates sigma from the calibration tracks (residual about a 9-frame local quadratic "
        "fit on stationary frames), 'none' sets sigma = 0, a number sets sigma in cm",
    )
    ap.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 3.0],
        help="assumed information levels, in nats",
    )
    ap.add_argument(
        "--lead1-levels",
        type=float,
        nargs="+",
        default=None,
        help="information levels (nats) at which lead 1 is evaluated (default: all --levels); "
        "the other leads always use all of --levels",
    )
    ap.add_argument(
        "--entropy-margin-levels",
        type=float,
        nargs="+",
        default=None,
        help="levels at which the mixture-entropy margin outcomes (_H) are computed "
        "(default: all --levels)",
    )
    ap.add_argument(
        "--no-tpp-gaussian",
        action="store_true",
        help="with --tpp-dir, drop the moment-matched 'Trajectron' entry and keep 'Trajectron_mix'",
    )
    ap.add_argument(
        "--chunk-tracks",
        type=int,
        default=5,
        help="test tracks per task; tasks are dealt to the workers as they free up",
    )
    ap.add_argument(
        "--velocity-window",
        type=int,
        default=1,
        help="frames the reference's velocity is taken over (5 = the old default)",
    )
    ap.add_argument(
        "--gru-at5-dir",
        default=None,
        help="directory with lead-5-only GRU checkpoints (scored at lead 5 only)",
    )
    ap.add_argument("--imm-params", default=None, help="JSON of an earlier bounded IMM fit")
    ap.add_argument(
        "--ctrv-params",
        default=None,
        help="JSON with fitted CTRV sigma_a, sigma_yaw, measurement_noise (default: hand-set)",
    )
    add_tpp_args(ap)
    ap.add_argument(
        "--no-still-cap",
        action="store_true",
        help="fit the reference on all calibration frames (no cut of still runs at WINDOW_CAP)",
    )
    ap.add_argument(
        "--gru-dir",
        default=None,
        help="directory with gru_position_best.pt / gru_orientation_best.pt",
    )
    ap.add_argument(
        "--stop-anisotropic",
        action="store_true",
        help="keep the stopped regime's fitted world-axis spread (not usable with --tests exact)",
    )
    ap.add_argument(
        "--tests",
        choices=("exact", "regime"),
        default="exact",
        help="exact: overlap term per truth location (tables); regime: averaged (old)",
    )
    ap.add_argument(
        "--test-subset",
        type=int,
        default=None,
        help="score only this many test tracks (seeded sample); calibration and fits unchanged",
    )
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.dataset == "eindhoven" and not os.path.exists(args.csv):
        raise SystemExit(
            f"{args.csv} not found; run ../get_data.sh"
        )
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    if args.dataset == "day-in-the-life":
        from day_in_the_life import load_day_in_the_life

        full, _ = load_day_in_the_life()
    else:
        from reactmetric.datasets.eindhoven import load_eindhoven

        full = load_eindhoven(path=args.csv)
    ids = list(full.ids)
    rng = np.random.default_rng(args.seed)
    pick = [ids[i] for i in rng.choice(len(ids), size=min(args.tracks, len(ids)), replace=False)]
    data = TrajectorySet({t: full[t] for t in pick})
    del full
    calib, test = data.split(fraction=0.3, seed=args.seed)
    if args.test_subset:
        keep = np.random.default_rng(args.seed + 1).permutation(sorted(test.ids))[: args.test_subset]
        test = TrajectorySet({t: test[t] for t in keep})
    if args.stop_anisotropic and args.tests == "exact":
        raise SystemExit("--tests exact needs the isotropic stopped spread")
    ref, ref_fit = fit_reference(
        calib,
        args.velocity_window,
        args.seed,
        cap_still=not args.no_still_cap,
        stop_isotropic=not args.stop_anisotropic,
    )
    chosen = {k: v for k, v in (ref_fit or {}).items() if k != "grid"}
    print(
        f"{len(calib)} calibration / {len(test)} test tracks; climatology fitted "
        f"(velocity window {args.velocity_window}, {chosen})",
        flush=True,
    )
    tpp = (
        dict(tpp_dir=args.tpp_dir, model_dir=args.tpp_model, checkpoint=args.tpp_ckpt)
        if args.tpp_dir
        else None
    )
    specs, fitted = fit_models(
        calib, args.gru_dir, args.gru_at5_dir, args.imm_params, tpp, args.ctrv_params,
        tpp_gaussian=not args.no_tpp_gaussian,
    )
    MODEL_NAMES = list(specs) + ["oracle_B"]
    print(
        f"models fitted ({(time.time() - t_start) / 60:.1f} min): "
        + json.dumps(
            {
                m: {k: round(v, 4) if isinstance(v, float) else v for k, v in f.items()}
                for m, f in fitted.items()
            }
        ),
        flush=True,
    )

    event_fit = {"detector": args.events}
    if args.events == "bayes-sg":
        hazard = args.hazard if args.hazard is not None else bcp.TUNED_SG["hazard"]
        det_spec = ("bayes-sg", float(hazard))
        event_fit["hazard"] = float(hazard)
        event_fit["settings"] = dict(bcp.TUNED_SG, hazard=float(hazard))
    elif args.events == "bayes":
        if args.hazard is None:
            hazard, curve = bcp.fit_hazard([calib[t] for t in calib.ids], seed=args.seed, jobs=args.jobs)
            event_fit["hazard_curve"] = curve
        else:
            hazard = args.hazard
        det_spec = ("bayes", float(hazard))
        event_fit["hazard"] = float(hazard)
    else:
        det_spec = ("pelt", "low")
    print(f"event source: {event_fit} ({(time.time() - t_start) / 60:.1f} min)", flush=True)
    nf = str(args.noise_floor).strip()
    if nf.lower() == "none":
        sigma, noise_fit = 0.0, {"floor": "none"}
    elif nf.upper() == "A":
        sigma, n_res, n_tr = estimate_noise_floor(calib)
        noise_fit = {"floor": "A", "sigma_cm": 100 * sigma, "residuals_per_axis": n_res,
                     "tracks": n_tr}
    else:
        sigma, noise_fit = float(nf) / 100.0, {"floor": "given"}
    noise_fit["sigma_cm"] = 100 * sigma
    print(f"measurement-noise floor {noise_fit}", flush=True)
    sig2 = sigma**2
    log_k = float(np.log(args.p / (1.0 - args.p)) if args.decision == "posterior" else np.log(args.k))

    lead1_levels = (
        None if args.lead1_levels is None else {round(float(x), 4) for x in args.lead1_levels}
    )
    entropy_levels = (
        None
        if args.entropy_margin_levels is None
        else {round(float(x), 4) for x in args.entropy_margin_levels}
    )
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        parts = list(
            pool.map(
                calib_chunk,
                split_chunks(calib, args.jobs),
                repeat(ref),
                repeat(specs),
                repeat(det_spec),
            )
        )
    calib_summary, levels = {}, {}
    for lead in LEADS:
        runs = [r for p in parts for r in p[lead]["runs"]]
        rho_w, ms = ri.pooled_autocorr(runs, N_LAGS)
        s_mean = {
            m: sum(p[lead]["s_sum"][m] for p in parts)
            / max(sum(p[lead]["s_n"][m] for p in parts), 1)
            for m in specs
        }
        rho_b = ri.overlap_corr(lead, CORR_LEN)
        # entropic information level: the calm-frame truth fraction is c times the mean
        # entropy scale of the mixture reference
        s_bar = sum(p[lead]["scale_sum"] for p in parts) / max(
            sum(p[lead]["scale_n"] for p in parts), 1
        )
        # mean whitened variance of the isotropic measurement noise on calm frames
        sn_bar = sig2 * sum(p[lead]["sn_sum"] for p in parts) / max(
            sum(p[lead]["scale_n"] for p in parts), 1
        )
        levels[lead], fits = [], {}
        for info in args.levels:
            if lead == 1 and lead1_levels is not None and round(float(info), 4) not in lead1_levels:
                continue
            c = ri.info_to_c(info)
            scale, tau, omega = ri.fit_knowable_corr(
                rho_w, rho_b, c * s_bar, FIT_LAGS, noise=sn_bar
            )
            levels[lead].append((info, c, ri.knowable_corr(CORR_LEN, scale, tau, omega), rho_b))
            fits[f"{info:.3f}"] = {"c": c, "scale": scale, "tau": tau, "omega": omega}
        ms_ = np.concatenate([x for p in parts for x in p[lead]["margin_samples"]])
        stand = ms_[:, 0] > 0.5
        margin_med = {}
        for d in args.distortions:
            row = {}
            for name, sel in (("standing", stand), ("moving", ~stand)):
                row[name] = {
                    "n": int(sel.sum()),
                    "entropy_cov": float(np.median(distortion_margin(ms_[sel, 2], d))),
                    "entropy_mix": float(np.median(distortion_margin(ms_[sel, 1], d))),
                }
            margin_med[f"{d:g}"] = row
        calib_summary[lead] = {
            "calm_mean_score": s_mean,
            "reference_mean_square": ms,
            "rho_w": rho_w[1:11].tolist(),
            "calm_frames": int(sum(len(r) for r in runs)),
            "entropy_scale_calm": s_bar,
            "noise_white_variance": sn_bar,
            "margin_median_nats": margin_med,
            "fits": fits,
        }
    print(f"calibration done ({(time.time() - t_start) / 60:.1f} min)", flush=True)

    table_dir = os.path.join(args.out, "tables")
    if args.tests == "exact":
        os.makedirs(table_dir, exist_ok=True)
        t_tab = time.time()
        jobs = [
            (ref, lead, info, c, table_path(table_dir, lead, info), sig2)
            for lead in LEADS
            for info, c, _, _ in levels[lead]
        ]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            built = list(pool.map(build_table, jobs))
        per_level = {}
        for b in built:
            per_level[b[1]] = per_level.get(b[1], 0.0) + b[2]
        print(
            f"overlap tables built ({time.time() - t_tab:.0f} s wall; per table "
            + ", ".join(f"L{b[0]}/{b[1]:g}: {b[2]:.0f} s {b[3][2]}x{b[3][3]}" for b in built)
            + "; cpu s per level: "
            + ", ".join(f"{k:g}: {v:.0f}" for k, v in per_level.items())
            + f"; {args.jobs} jobs)",
            flush=True,
        )

    chunks = small_chunks(test, args.chunk_tracks)
    results, clock = {}, {}
    t_test = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {
            pool.submit(
                test_chunk, ch, ref, levels, args.eps, log_k, specs, table_dir, args.tests,
                det_spec, args.decision, tuple(args.distortions), sigma, entropy_levels,
            ): i
            for i, ch in enumerate(chunks)
        }
        for n_done, fut in enumerate(as_completed(futs), 1):
            recs, clk = fut.result()
            results[futs[fut]] = recs
            for k_, v_ in clk.items():
                clock[k_] = clock.get(k_, 0) + v_
            if n_done % max(1, len(chunks) // 10) == 0 or n_done == len(chunks):
                print(f"  {n_done}/{len(chunks)} tasks ({time.time() - t_test:.0f} s)", flush=True)
    records = [r for i in sorted(results) for r in results[i]]
    print(
        "test phase cpu seconds (all workers): "
        + ", ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in clock.items())
        + f"; wall {time.time() - t_test:.0f} s",
        flush=True,
    )
    summary = summarize(records)
    print(
        f"measurement done ({(time.time() - t_start) / 60:.1f} min, "
        f"{len(records)} event-model-level rows)"
    )

    tag = f"seed{args.seed}_n{args.tracks}_vw{args.velocity_window}"
    if args.no_still_cap:
        tag += "_uncapped"
    with open(os.path.join(args.out, f"records_{tag}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wr.writeheader()
        wr.writerows(records)
    with open(os.path.join(args.out, f"summary_{tag}.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "reference_fit": ref_fit,
                "model_fits": fitted,
                "event_source": event_fit,
                "noise_floor": noise_fit,
                "calibration": {str(k): v for k, v in calib_summary.items()},
                "results": [
                    {
                        "lead": k[0],
                        "info": k[1],
                        "distortion": k[2],
                        "distortion_label": f"{k[2]:g}m" if k[2] else CLIM,
                        "model": k[3],
                        **v,
                    }
                    for k, v in summary.items()
                ],
            },
            f,
            indent=1,
        )

    print("\nCALIBRATION (calibration tracks, calm stretches)")
    for lead in LEADS:
        cs = calib_summary[lead]
        print(
            f"  lead {lead:>2}: calm-stretch mean score ("
            + ", ".join(f"{m} {v:+.2f}" for m, v in cs["calm_mean_score"].items())
            + ")"
            f";  climatology residual mean square {cs['reference_mean_square']:.2f}"
            " (1 = calibrated)"
        )
        print("           rho_w lags 1-10: " + " ".join(f"{v:+.2f}" for v in cs["rho_w"]))
        for d, row in cs["margin_median_nats"].items():
            print(
                f"           median margin at d = {d} m (nats; covariance / mixture entropy): "
                + "; ".join(
                    f"{k} (n {r['n']}) {r['entropy_cov']:.2f} / {r['entropy_mix']:.2f}"
                    for k, r in row.items()
                )
            )
        for key, ft in cs["fits"].items():
            print(
                f"           I = {key} nats (c = {ft['c']:.3f}): knowable correlation "
                f"scale {ft['scale']:.2f}, tau {ft['tau']:.1f} frames, "
                f"omega {ft['omega']:.2f} rad/frame"
            )

    rule = f"P >= {args.p:g}" if args.decision == "posterior" else f"BF >= {args.k:g}"
    print(
        f"\nRESULTS (test tracks; eps = {args.eps}, {rule}; "
        f"target: {CLIM} = the climatology's own spread, no distortion target)"
    )
    hdr = (
        f"{'lead':>4} {'I (nats)':>8} {'model':>16} | {'events':>6} {'KM REACT':>8} {'onset':>5} | "
        f"{'recov':>5} {'cens':>5} {'better':>6} {'indist':>6} {'relapse':>7} {'rel3':>5} | "
        f"{'anticip':>7} {'RMST':>6} | "
        f"{'mean s':>7} {'mean I':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for lead in LEADS:
        for info, *_ in levels[lead]:
            for m in MODEL_NAMES:
                v = summary.get((lead, round(float(info), 4), 0.0, m))
                if v is None:
                    continue
                print(
                    f"{lead:>4} {info:>8.2f} {m:>16} | {v['events']:>6} {fmt(v['km_react'], 8)} "
                    f"{fmt(v['median_onset'], 5)} | {v['recovered']:>5.2f} {v['censored']:>5.2f} "
                    f"{v['better']:>6.2f} {v['indistinguishable']:>6.2f} "
                    f"{fmt(v['relapse'], 7, 2)} {fmt(v['relapse3'], 5, 2)} | "
                    f"{v['anticipation']:>7.2f} {fmt(v['rmst'], 6, 1)} | "
                    f"{v['mean_s_mixture']:>+7.2f} {v['mean_I']:>+7.2f}"
                )
        print()

    if args.distortions:
        print(
            "\nREACT@d, KM-median REACT (frames) under the covariance margin (default) | the "
            "mixture-entropy margin (_H); mean margin (nats) the same; recovered share"
        )
        for lead in LEADS:
            for info, *_ in levels[lead]:
                for m in MODEL_NAMES:
                    cells = []
                    for d in args.distortions:
                        v = summary.get((lead, round(float(info), 4), float(d), m))
                        if v is not None:
                            cells.append(
                                f"d={d:g}: {fmt(v['km_react'], 4)} | {fmt(v['km_react_H'], 4)}"
                                f" (margin {v['margin_mean']:.2f} | {v['margin_mean_H']:.2f},"
                                f" rec {v['recovered']:.2f} | {v['recovered_H']:.2f})"
                            )
                    if cells:
                        print(f"  lead {lead:>2} I {info:>4.2f} {m:>16}: " + "   ".join(cells))

    prev = os.path.join(ROOT, "outputs", "exp09", f"seed{args.seed}_n{args.tracks}.json")
    if os.path.exists(prev):
        p9 = json.load(open(prev))
        print(
            "SAME EVENTS UNDER THE EARLIER RULES (exp09): "
            "KM-median REACT, 3/3 rule | posterior rule"
        )
        for m in MODEL_NAMES:
            cells = [p9["cells"].get(f"{m}|{lead}") for lead in LEADS]
            print(
                f"  {m:>16}: "
                + "   ".join(
                    f"lead {lead}: {fmt(c['cur_km'], 4)} | {fmt(c['post_km'], 4)}"
                    for lead, c in zip(LEADS, cells)
                    if c
                )
            )
    print(f"\ntotal {(time.time() - t_start) / 60:.1f} min; outputs in {args.out}")


if __name__ == "__main__":
    main()
