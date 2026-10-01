"""Bayesian changepoint detection for REACT events (kept outside the package until validated).

Offline Bayesian changepoint model (Fearnhead 2006) on the same features as the package's
KinematicChangepoints (speed and turn rate from the velocity window), reusing its event-type
classification; only the segmentation step differs.

  * Segments: each channel is Normal with unknown mean and variance, Normal-Inverse-Gamma prior
    (kappa0, nu0) on robustly standardized features. The w-frame velocity window makes
    neighbouring feature frames strongly correlated, so the likelihood is tempered by 1/w:
    n frames count as n / w observations.
  * Durations: shifted geometric with hazard h per frame above the minimum segment length.
    h is not hand-set: fit_hazard picks it by empirical Bayes (maximum summed marginal
    likelihood over calibration tracks).
  * Posterior: exact forward-backward recursions over changepoint locations. Segment marginal
    likelihoods come from cumulative sums on the fly, so memory is linear in track length and
    time quadratic.
  * Events: greedily, the mode of each window where the posterior mass of a change within
    +-delta frames is at least `threshold` (0.5: declare a change where one is more likely than
    not). t0_std is the posterior spread of the change time within that window.

On 50 Eindhoven test tracks the hand-set PELT preset ("low", penalty 80) gave 2.4 events per
track and an event on 81% of stationary surrogate tracks; this detector with the empirical-
Bayes hazard (1e-2) gave 1.0 events per track and 0.7% false-event tracks, and kept mostly the
larger speed changes, starts and stops.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.special import gammaln, logsumexp

from reactmetric.events.kinematic import KinematicChangepoints, segmentation_config

DEFAULT_HAZARD = 1e-2
# Tuned on the grounded synthetic tracks with known changes (validation split; held-out F1 0.643
# at +-10 frames, 0.435 at +-3). The hazard is fixed at its tuned value, not refitted by empirical
# Bayes: the smoothed features are much less noisy and the evidence would pick a different one.
TUNED_SG = dict(hazard=9e-4, min_size=30, kappa0=0.139, nu0=4.74, delta=3, threshold=0.4,
                velocity_window=1, temper=3.0, velocity_channels=True,
                smooth_window=5, smooth_polyorder=2, relabel_heading=True)


def _robust_sd(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


class BayesianChangepoints(KinematicChangepoints):
    """Event source: offline Bayesian changepoints on speed and turn rate."""

    def __init__(self, hazard=DEFAULT_HAZARD, min_size=50, kappa0=1.0, nu0=1.0, threshold=0.5,
                 delta=5, base_preset="low", velocity_window=None, temper=None,
                 velocity_channels=False, smooth_window=0, smooth_polyorder=2,
                 relabel_heading=False):
        over = {} if velocity_window is None else {"velocity_window": int(velocity_window)}
        super().__init__(segmentation_config(base_preset, min_size=min_size, **over))
        self.hazard, self.kappa0, self.nu0 = float(hazard), float(kappa0), float(nu0)
        self.threshold, self.delta = float(threshold), int(delta)
        # `temper` (frames): n frames count as n / temper observations. None: the velocity window
        # w of the preset (the original rule). velocity_channels: add vx, vy as channels (a
        # heading change is a mean shift in the velocity vector). smooth_window > 0:
        # Savitzky-Golay smoothing (window, polynomial order) of the positions before any
        # feature is taken.
        w = self.config.velocity_window
        self.temper = float(w if temper is None else temper)
        self.velocity_channels = bool(velocity_channels)
        self.smooth_window, self.smooth_polyorder = int(smooth_window), int(smooth_polyorder)
        self.relabel_heading = bool(relabel_heading)

    @classmethod
    def tuned_sg(cls, hazard=None, **kw):
        """Tuned settings (speed, turn, vx, vy; Savitzky-Golay pre-smoothing), see TUNED_SG."""
        cfg = dict(TUNED_SG)
        if hazard is not None:
            cfg["hazard"] = float(hazard)
        cfg.update(kw)
        return cls(**cfg)

    def _positions(self, traj):
        """Trajectory positions, Savitzky-Golay smoothed when smooth_window is set."""
        p = np.asarray(traj.positions, float)
        k = self.smooth_window
        if k and len(p) >= k:
            p = savgol_filter(p, k, min(self.smooth_polyorder, k - 1), axis=0, mode="interp")
        return p

    # -- segment marginal likelihoods -------------------------------------------------------
    def _channels(self, speed, turn, w, extra=()):
        out = []
        for x in (speed, turn, *extra):
            z = (x - np.median(x)) / max(_robust_sd(x), 1e-3)
            d = z[w:] - z[:-w]
            s0 = _robust_sd(d) / np.sqrt(2) if len(d) > 5 else 1.0
            c1 = np.concatenate([[0.0], np.cumsum(z)])
            c2 = np.concatenate([[0.0], np.cumsum(z * z)])
            out.append((c1, c2, max(s0, 1e-3) ** 2))
        return out

    def _seg(self, chans, s, e, w):
        """log marginal likelihood of segments [s, e) (arrays broadcast)."""
        n = (e - s).astype(float)
        tot = 0.0
        k0, nu0 = self.kappa0, self.nu0
        for c1, c2, s0sq in chans:
            S1, S2 = c1[e] - c1[s], c2[e] - c2[s]
            mean = S1 / n
            ss = np.maximum(S2 - n * mean**2, 0.0)
            ne = n / self.temper
            kn = k0 + ne
            a0, b0 = nu0 / 2.0, nu0 * s0sq / 2.0
            an = a0 + ne / 2.0
            bn = b0 + 0.5 * (ss / self.temper + k0 * ne * mean**2 / kn)
            tot = tot + (
                a0 * np.log(b0) - an * np.log(bn) + gammaln(an) - gammaln(a0) + 0.5 * np.log(k0 / kn)
            )
        return tot

    def posterior(self, speed, turn, w, hazard=None, extra=()):
        """(log evidence, P(change at feature index b), b = 0..N). `extra`: further feature
        channels (vx, vy) of the same length."""
        h = self.hazard if hazard is None else float(hazard)
        N, m = len(speed), self.config.min_size
        chans = self._channels(speed, turn, w, extra)
        L = np.arange(N + 1)
        with np.errstate(divide="ignore"):
            logg = np.where(L >= m, np.log(h) + (L - m) * np.log1p(-h), -np.inf)
            logG = np.where(L >= m, (L - m) * np.log1p(-h), -np.inf)
        la = np.full(N + 1, -np.inf)
        la[0] = 0.0
        for e in range(m, N + 1):
            s = np.arange(0, e - m + 1)
            la[e] = logsumexp(la[s] + logg[e - s] + self._seg(chans, s, e, w))
        s = np.arange(0, N - m + 1)
        logZ = logsumexp(la[s] + logG[N - s] + self._seg(chans, s, N, w))
        lb = np.full(N + 1, -np.inf)
        for b in range(N - m, -1, -1):
            last = logG[N - b] + self._seg(chans, np.array(b), np.array(N), w)
            e = np.arange(b + m, N)
            if len(e):
                lb[b] = np.logaddexp(last, logsumexp(logg[e - b] + self._seg(chans, b, e, w) + lb[e]))
            else:
                lb[b] = last
        post = np.exp(la + lb - logZ)
        post[0] = post[N] = 0.0
        return float(logZ), post

    # -- events -----------------------------------------------------------------------------
    def changepoints(self, post):
        p = post.copy()
        cps = []
        while p.max() > 0:
            b = int(np.argmax(p))
            lo, hi = max(b - self.delta, 0), min(b + self.delta + 1, len(p))
            if post[lo:hi].sum() < self.threshold:
                break
            cps.append(b)
            p[max(b - self.config.min_size + 1, 0) : b + self.config.min_size] = 0.0
        return sorted(cps)

    def _features_std(self, traj):
        pos = self._positions(traj)
        f = self._features(pos, traj.dt)
        if f is None:
            return None
        speed, turn, w = f
        sp_sigma = max(np.std(np.diff(speed)) / np.sqrt(2), 1e-3)
        tr_sigma = max(np.std(np.diff(turn)) / np.sqrt(2), 1e-3)
        extra = ()
        if self.velocity_channels:  # velocity vector: a heading change is a mean shift in (vx, vy)
            vel = (pos[w:] - pos[:-w]) / (w * traj.dt)
            extra = (vel[:, 0], vel[:, 1])
        return speed, turn, w, sp_sigma, tr_sigma, extra

    def detect(self, traj):
        f = self._features_std(traj)
        if f is None:
            return []
        speed, turn, w, sp_sigma, tr_sigma, extra = f
        if len(speed) < 2 * self.config.min_size:
            return []
        _, post = self.posterior(speed, turn, w, extra=extra)
        cps = self.changepoints(post)
        bounds = [0] + cps + [len(speed)]
        events = []
        for i, cp in enumerate(cps, start=1):
            ev = self._classify(speed, turn, cp, bounds[i - 1], bounds[i + 1], sp_sigma, tr_sigma,
                                traj.dt, traj.id)
            ev.t0 = int(cp + w)
            lo, hi = max(cp - self.delta, 0), min(cp + self.delta + 1, len(post))
            idx = np.arange(lo, hi)
            pw = post[lo:hi] / max(post[lo:hi].sum(), 1e-12)
            ev.t0_std = float(np.sqrt((pw * (idx - (pw * idx).sum()) ** 2).sum()))
            events.append(ev)
        if self.relabel_heading:
            relabel_from_headings(traj.positions, events, w, traj.dt)
        return events


def relabel_from_headings(obs, events, w, dt, turn_deg=25.0, dv_min=0.2, min_speed=0.3):
    """Relabel non-stop/start events from windowed headings and speeds (in place).

    The segment-mean turn rate of the package classifier rarely flags heading changes under
    per-frame velocities. Instead, across an event at t0 - w/2, compare the displacement over
    the 15-50 frames before (not past the previous event) with that over the 15-50 frames after
    (not past the next): a heading change of at least `turn_deg` is a turn, a speed change of at
    least `dv_min` m/s a speed change, both is "both". Events with a window shorter than 12
    frames, or a speed below `min_speed` before or after, keep their label.
    """
    obs = np.asarray(obs, float)
    tc = [e.t0 - w / 2 for e in events]
    for k, e in enumerate(events):
        if e.event_type in ("stop", "start"):
            continue
        prev = tc[k - 1] if k > 0 else 0
        nxt = tc[k + 1] if k + 1 < len(tc) else len(obs) - 1
        a, b = int(max(prev, tc[k] - 50)), int(tc[k] - 15)
        c, d = int(tc[k] + 15), int(min(nxt, tc[k] + 50, len(obs) - 1))
        if b - a < 12 or d - c < 12:
            continue
        dp, dq = obs[b] - obs[a], obs[d] - obs[c]
        sp, sq = np.hypot(*dp) / ((b - a) * dt), np.hypot(*dq) / ((d - c) * dt)
        if sp < min_speed or sq < min_speed:
            continue
        d_head = np.arctan2(dq[1], dq[0]) - np.arctan2(dp[1], dp[0])
        dpsi = np.rad2deg(np.angle(np.exp(1j * d_head)))
        turn, spd = abs(dpsi) >= turn_deg, abs(sq - sp) >= dv_min
        if turn and spd:
            e.event_type = "both"
        elif turn:
            e.event_type = "turn"
        elif spd:
            e.event_type = "speed_change"
    return events


def _summed_evidence(job):
    feats, h, kw = job
    det = BayesianChangepoints(**kw)
    return sum(det.posterior(s, t, w, hazard=h, extra=x)[0] for s, t, w, x in feats)


def fit_hazard(trajectories, grid=None, max_tracks=120, max_len=3000, seed=0, jobs=1, **kw):
    """Empirical-Bayes hazard: argmax over grid of the summed log evidence of the tracks.

    Uses up to max_tracks tracks with at most max_len feature frames; grid points are
    evaluated in parallel over `jobs` processes. Returns (hazard, {hazard: summed log
    evidence})."""
    grid = np.geomspace(1e-3, 1e-1, 9) if grid is None else np.asarray(grid, float)
    det = BayesianChangepoints(**kw)
    feats = []
    for traj in trajectories:
        f = det._features_std(traj)
        if f is not None and 2 * det.config.min_size <= len(f[0]) <= max_len:
            feats.append((f[0], f[1], f[2], f[5]))
    rng = np.random.default_rng(seed)
    if len(feats) > max_tracks:
        feats = [feats[i] for i in rng.choice(len(feats), max_tracks, replace=False)]
    jobs_ = [(feats, float(h), kw) for h in grid]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            ev = list(pool.map(_summed_evidence, jobs_))
    else:
        ev = [_summed_evidence(j) for j in jobs_]
    curve = dict(zip((float(h) for h in grid), ev))
    return max(curve, key=curve.get), curve
