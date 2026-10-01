"""Information formulation of REACT: core computations (see ../docs/react-spec.md).

Kept outside the package, like react_posterior.py, until it is validated.

Truth model for the target frames of one event window at lead h, in the climatological
reference's whitened coordinates w_t = L_r,t^-1 (y_t - mu_r,t):

    w_t = a_t + b_t,   a ~ GP(0, (1-c) R_a)    knowable part: where the person is headed
                       b ~ GP(0,  c    R_b)    unknowable part: forecast-window overlap
    c = exp(-2 I_traj / d),                    I_traj: information content in nats

so the truth is q_t = N(m_t, c Sigma_r,t) with m_t = mu_r,t + L_r,t a_t. The posterior over m_t
uses observations only and is shared by every model. A model enters only through
I_t = KL(q_t || p_ref,t) - KL(q_t || p_model,t), whose tail probabilities are computed by
deterministic quadrature, and decisions use Bayes factors (posterior odds / prior odds).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates
from scipy.special import ndtr

__all__ = [
    "info_to_c",
    "overlap_corr",
    "fit_knowable_corr",
    "knowable_corr",
    "pooled_autocorr",
    "truth_posterior",
    "to_world",
    "advantage_quadratic",
    "prob_greater",
    "frame_tests",
    "classify",
    "gaussian_logpdf",
    "gh_nodes_2d",
    "mixture_logpdf",
    "mixture_moments",
    "mixture_entropy",
    "gaussian_entropy",
    "MixtureTests",
    "RegimeTests",
    "RegimeTestsExact",
    "regime_predictive",
    "OverlapTable",
    "MixtureForecast",
    "RegimeTestsMixture",
    "truth_posterior_het",
]

_EYE2 = np.eye(2)


def info_to_c(info_nats: float, dim: int = 2) -> float:
    """Truth spread as a fraction of climatology's: Omega = c Sigma_ref, c = exp(-2 I / d)."""
    return float(np.exp(-2.0 * info_nats / dim))


# ---------------------------------------------------------------------------
# Correlation structure
# ---------------------------------------------------------------------------


def overlap_corr(lead: int, n_lags: int) -> np.ndarray:
    """rho_b: correlation of the unforeseeable part between targets `lag` frames apart.

    Targets lag frames apart share lead - lag frames of motion the earlier forecast could not
    see; a velocity innovation i frames before a target moves it by i steps.
    """
    rho = np.zeros(n_lags + 1)
    rho[0] = 1.0
    den = sum(i * i for i in range(1, lead + 1))
    for lag in range(1, min(lead, n_lags + 1)):
        rho[lag] = sum(i * (i + lag) for i in range(1, lead - lag + 1)) / den
    return rho


def knowable_corr(n_lags: int, scale: float, tau: float, omega: float) -> np.ndarray:
    """rho_a: nugget plus damped cosine, rho(l) = scale * exp(-l/tau) cos(omega l) for l >= 1."""
    lags = np.arange(n_lags + 1, dtype=float)
    rho = scale * np.exp(-lags / tau) * np.cos(omega * lags)
    rho[0] = 1.0
    return rho


def fit_knowable_corr(rho_w, rho_b, c, max_lag=20, noise=0.0):
    """Fit knowable_corr to (rho_w - c rho_b) / (1 - c - noise) at lags 1..max_lag by least squares.

    noise: the share of the whitened variance that is white measurement noise (truth-spread
    floor), which has no correlation at lags >= 1 and is not knowable.

    Returns (scale, tau, omega). The scale for each (tau, omega) is solved in closed form and
    clipped to [0, 1], which keeps the correlation positive definite.
    """
    lags = np.arange(1, max_lag + 1, dtype=float)
    rho_w, rho_b = np.asarray(rho_w), np.asarray(rho_b)
    target = (rho_w[1 : max_lag + 1] - c * rho_b[1 : max_lag + 1]) / max(1.0 - c - noise, 1e-6)
    ok = np.isfinite(target)
    lags, target = lags[ok], target[ok]
    taus = np.geomspace(0.3, 300.0, 120)
    omegas = np.linspace(0.0, np.pi / 2, 91)
    basis = np.exp(-lags[None, None, :] / taus[:, None, None]) * np.cos(
        omegas[None, :, None] * lags[None, None, :]
    )
    scale = np.clip((basis * target).sum(-1) / np.maximum((basis * basis).sum(-1), 1e-12), 0.0, 1.0)
    sse = ((scale[..., None] * basis - target) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(sse), sse.shape)
    return float(scale[i, j]), float(taus[i]), float(omegas[j])


def pooled_autocorr(runs, n_lags):
    """Autocorrelation of whitened residual runs, pooled over runs and axes, without demeaning.

    Returns (rho[0..n_lags], mean square per coordinate). The mean square is 1 when the
    reference is calibrated; the residuals are not demeaned because under the reference they
    have mean zero, and demeaning short runs biases every lag downward.
    """
    tot = sum(float((r * r).sum()) for r in runs)
    cnt = sum(r.size for r in runs)
    ms = tot / max(cnt, 1)
    rho = np.full(n_lags + 1, np.nan)
    rho[0] = 1.0
    for lag in range(1, n_lags + 1):
        num, n = 0.0, 0
        for r in runs:
            if len(r) > lag:
                num += float((r[:-lag] * r[lag:]).sum())
                n += r[:-lag].size
        if n:
            rho[lag] = num / n / ms
    return rho, ms


# ---------------------------------------------------------------------------
# Truth posterior (shared by all models)
# ---------------------------------------------------------------------------


def truth_posterior(frames, w, rho_a, rho_b, c):
    """Posterior of the knowable part a_t, filtering (w up to t) and smoothing (whole window).

    Returns mean_f (n, 2), var_f (n,), mean_s (n, 2), var_s (n,). Variances are per axis and
    equal across axes. One Cholesky factor serves every t: the leading t x t block of chol(K)
    is chol of the leading block, so (L^-1 x)[:t] only depends on x[:t].
    """
    frames = np.asarray(frames)
    n = len(frames)
    lag = np.abs(frames[:, None] - frames[None, :])
    n_l = len(rho_a) - 1
    idx = np.minimum(lag, n_l)
    Ka = (1.0 - c) * np.where(lag <= n_l, rho_a[idx], 0.0)
    Kw = Ka + c * np.where(lag <= n_l, rho_b[idx], 0.0) + 1e-9 * np.eye(n)
    L = cholesky(Kw, lower=True)
    V = solve_triangular(L, w, lower=True)
    U = solve_triangular(L, Ka, lower=True)
    Ut = np.triu(U)
    mean_f = Ut.T @ V
    var_f = np.maximum((1.0 - c) - (Ut * Ut).sum(0), 1e-12)
    mean_s = U.T @ V
    var_s = np.maximum((1.0 - c) - (U * U).sum(0), 1e-12)
    return mean_f, var_f, mean_s, var_s


def to_world(mean_w, var_w, mu_r, S_r):
    """Whitened posterior of a_t -> posterior of m_t in world coordinates: (mean, cov)."""
    Lr = np.linalg.cholesky(S_r)
    return mu_r + np.einsum("nij,nj->ni", Lr, mean_w), var_w[:, None, None] * S_r


# ---------------------------------------------------------------------------
# The advantage I_t and its tail probabilities
# ---------------------------------------------------------------------------


def advantage_quadratic(mu, C, mu_m, S_m, mu_r, S_r, c, sig2=0.0):
    """Write I_t(m), m ~ N(mu, C), as z'Mz + g'z + h with z ~ N(0, I_2). Omega = c S_r + sig2 I.

    Coordinates are centred on the reference mean first; I is translation invariant and the
    raw quadratic terms in world coordinates would cancel catastrophically.
    """
    mu = mu - mu_r
    mu_m = mu_m - mu_r
    Ar = np.linalg.inv(S_r)
    Am = np.linalg.inv(S_m)
    D = 0.5 * (Ar - Am)
    d = np.einsum("nij,nj->ni", Am, mu_m)
    _, ldr = np.linalg.slogdet(S_r)
    _, ldm = np.linalg.slogdet(S_m)
    e = 0.5 * (
        c * np.einsum("nij,nji->n", Ar - Am, S_r)
        + sig2 * np.einsum("nii->n", Ar - Am)
        - np.einsum("ni,nij,nj->n", mu_m, Am, mu_m)
        + ldr
        - ldm
    )
    S = np.linalg.cholesky(C + 1e-15 * _EYE2)
    M = np.einsum("nji,njk,nkl->nil", S, D, S)
    g = np.einsum("nji,nj->ni", S, 2.0 * np.einsum("nij,nj->ni", D, mu) + d)
    h = np.einsum("ni,nij,nj->n", mu, D, mu) + np.einsum("ni,ni->n", d, mu) + e
    return M, g, h


def _quad_tail(lam, g, t, tol=1e-12):
    """P(lam u^2 + g u > t) for u ~ N(0, 1), elementwise."""
    lam, g, t = np.broadcast_arrays(lam, g, t)
    out = np.empty(t.shape)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        pos = lam >= tol
        neg = lam <= -tol
        lin = ~pos & ~neg & (np.abs(g) >= tol)
        const = ~pos & ~neg & ~lin
        disc = g * g + 4.0 * lam * t
        sq = np.sqrt(np.maximum(disc, 0.0))
        val = ndtr((-g - sq) / (2.0 * lam)) + ndtr((g - sq) / (2.0 * lam))
        out[pos] = np.where(disc[pos] <= 0.0, 1.0, val[pos])
        al = -lam
        disc2 = g * g - 4.0 * al * t
        sq2 = np.sqrt(np.maximum(disc2, 0.0))
        val2 = ndtr((g + sq2) / (2.0 * al)) - ndtr((g - sq2) / (2.0 * al))
        out[neg] = np.where(disc2[neg] <= 0.0, 0.0, val2[neg])
        out[lin] = np.where(g[lin] > 0.0, ndtr(-t[lin] / g[lin]), ndtr(t[lin] / g[lin]))
        out[const] = (t[const] < 0.0).astype(float)
    return out


def _quadratic_roots(a, b, c, tol=1e-14):
    """Real roots (r1 <= r2) of a u^2 + b u + c = 0, elementwise; NaN where absent."""
    with np.errstate(divide="ignore", invalid="ignore"):
        quad = np.abs(a) >= tol
        disc = b * b - 4.0 * a * c
        sq = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
        q1, q2 = (-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)
        lin = -c / b
        r1 = np.where(quad, np.minimum(q1, q2), np.where(np.abs(b) >= tol, lin, np.nan))
        r2 = np.where(quad, np.maximum(q1, q2), np.nan)
    return r1, r2


_TS_T = np.arange(-3.2, 3.2 + 1e-9, 0.08)
_TS_X = np.tanh(0.5 * np.pi * np.sinh(_TS_T))
_TS_W = 0.08 * 0.5 * np.pi * np.cosh(_TS_T) / np.cosh(0.5 * np.pi * np.sinh(_TS_T)) ** 2
_U_MAX = 12.0


def prob_greater(M, g, h, x, tol=1e-12):
    """P(z'Mz + g'z + h > x), z ~ N(0, I_2), per row.

    Diagonalise M. The coordinate with the larger curvature is integrated exactly through the
    normal CDF; the other numerically. As a function of the outer coordinate the inner
    probability has square-root edges where the inner quadratic's discriminant crosses zero,
    so the outer integral is split at those points (found in closed form) and each piece is
    integrated by tanh-sinh quadrature, which is insensitive to endpoint singularities.
    """
    lam, R = np.linalg.eigh(M)
    gp = np.einsum("nji,nj->ni", R, g)
    swap = np.abs(lam[:, 0]) > np.abs(lam[:, 1])
    l_out = np.where(swap, lam[:, 1], lam[:, 0])
    l_in = np.where(swap, lam[:, 0], lam[:, 1])
    g_out = np.where(swap, gp[:, 1], gp[:, 0])
    g_in = np.where(swap, gp[:, 0], gp[:, 1])
    c0 = np.asarray(x, dtype=float) - h  # t(u) = c0 - l_out u^2 - g_out u
    # kinks: inner discriminant g_in^2 + 4 l_in t(u) = 0, or t(u) = 0 when the inner term vanishes
    const = (np.abs(l_in) < tol) & (np.abs(g_in) < tol)
    k1, k2 = _quadratic_roots(
        np.where(const, -l_out, -4.0 * l_in * l_out),
        np.where(const, -g_out, -4.0 * l_in * g_out),
        np.where(const, c0, g_in * g_in + 4.0 * l_in * c0),
    )
    k1 = np.clip(np.nan_to_num(k1, nan=0.0), -_U_MAX, _U_MAX)
    k2 = np.clip(np.nan_to_num(k2, nan=0.0), -_U_MAX, _U_MAX)
    edges = np.sort(np.stack([np.full_like(k1, -_U_MAX), k1, k2, np.full_like(k1, _U_MAX)], 1), 1)
    total = np.zeros(len(c0))
    for s in range(3):
        a, b = edges[:, s], edges[:, s + 1]
        half = 0.5 * (b - a)
        u = 0.5 * (a + b)[:, None] + half[:, None] * _TS_X[None, :]
        t = c0[:, None] - l_out[:, None] * u * u - g_out[:, None] * u
        f = _quad_tail(l_in[:, None], g_in[:, None], t) * np.exp(-0.5 * u * u)
        total += half * (f * _TS_W[None, :]).sum(1)
    return np.clip(total / np.sqrt(2.0 * np.pi), 0.0, 1.0)


def _log_odds(p, floor=1e-12):
    p = np.clip(p, floor, 1.0 - floor)
    return np.log(p) - np.log1p(-p)


def frame_tests(post_mu, post_C, prior_C, mu_m, S_m, mu_r, S_r, c, eps=0.0):
    """Per-frame log Bayes factors for onset (I < 0) and recovery (I > eps), and E[I | data].

    Bayes factor = posterior odds / prior odds, the prior being m_t ~ N(mu_r, prior_C).
    Lower tails are computed as upper tails of -I so both ends keep their precision.
    """
    Mq, gq, hq = advantage_quadratic(post_mu, post_C, mu_m, S_m, mu_r, S_r, c)
    Mp, gp, hp = advantage_quadratic(mu_r, prior_C, mu_m, S_m, mu_r, S_r, c)
    post_lt0 = prob_greater(-Mq, -gq, -hq, 0.0)
    prior_lt0 = prob_greater(-Mp, -gp, -hp, 0.0)
    if eps == 0.0:
        post_gt, prior_gt = prob_greater(Mq, gq, hq, 0.0), prob_greater(Mp, gp, hp, 0.0)
    else:
        post_gt, prior_gt = prob_greater(Mq, gq, hq, eps), prob_greater(Mp, gp, hp, eps)
    lbf_on = _log_odds(post_lt0) - _log_odds(prior_lt0)
    lbf_rec = _log_odds(post_gt) - _log_odds(prior_gt)
    mean_I = np.einsum("nii->n", Mq) + hq
    return lbf_on, lbf_rec, mean_I


# ---------------------------------------------------------------------------
# Event outcome
# ---------------------------------------------------------------------------


RELAPSE_RUN = 3


def _first_run(hit, n):
    """Index of the first frame of the first run of n consecutive True values, or None."""
    if len(hit) < n:
        return None
    ok = np.convolve(hit.astype(int), np.ones(n, int), mode="valid") == n
    return int(np.argmax(ok)) if ok.any() else None


def classify(lbf_on, lbf_rec, frames, t0, log_k, onset_window=None, first_verdict=True):
    """Outcome of one event window.

    first_verdict: scanning forward from the event, whichever test reaches k first
    settles it; "ahead" first means the event was anticipated and anything later belongs
    to a new regime. Otherwise the first onset (subject to the window) counts even after
    an earlier "ahead" verdict. onset_window (frames after t0): an onset counts only
    within the frames this event can explain, e.g. lead + a few frames; later onsets are
    ignored. Without an onset the event is anticipated: confirmed ("better") if the
    "ahead" test ever reaches k, otherwise unconfirmed ("indistinguishable").

    `relapse` flags a recovered event whose onset test fires again later in the window (a single
    frame suffices); `relapse_after` is the number of frames from recovery to that first renewed
    onset. `relapse3` and `relapse3_after` apply the stricter rule: the onset test must fire on
    RELAPSE_RUN (3) consecutive frames after recovery, and the time is counted to the first frame
    of the first such run.
    """
    hit_on = lbf_on >= log_k
    if onset_window is not None:
        hit_on = hit_on & (frames - t0 <= onset_window)
    on = np.flatnonzero(hit_on)
    ahead = np.flatnonzero(lbf_rec >= log_k)
    if len(on) and (not first_verdict or len(ahead) == 0 or on[0] < ahead[0]):
        o = int(on[0])
        rec = ahead[ahead > o]
        if len(rec):
            r = int(rec[0])
            again = np.flatnonzero(lbf_on[r + 1 :] >= log_k)
            run = _first_run(lbf_on[r + 1 :] >= log_k, RELAPSE_RUN)
            return {
                "status": "recovered",
                "onset": int(frames[o] - t0),
                "react": int(frames[r] - t0),
                "duration": int(frames[r] - t0),
                "relapse": bool(len(again)),
                "relapse_after": int(frames[r + 1 + again[0]] - frames[r]) if len(again) else None,
                "relapse3": run is not None,
                "relapse3_after": int(frames[r + 1 + run] - frames[r]) if run is not None else None,
            }
        return {
            "status": "censored",
            "onset": int(frames[o] - t0),
            "react": None,
            "duration": int(frames[-1] - t0),
            "relapse": False,
            "relapse_after": None,
            "relapse3": False,
            "relapse3_after": None,
        }
    status = "better" if len(ahead) else "indistinguishable"
    return {
        "status": status,
        "onset": None,
        "react": None,
        "duration": None,
        "relapse": False,
        "relapse_after": None,
        "relapse3": False,
        "relapse3_after": None,
    }


# Lazy evaluation: test only the frames that can decide an event's outcome.
RECOVERY_BLOCK = 16  # frames tested per step while searching for recovery
RELAPSE_WINDOW = 10  # frames after a recovery in which a renewed onset counts as a relapse


def classify_lazy(evaluate, frames, t0, log_k, n_ch, onset_window, oracle=False,
                  block=RECOVERY_BLOCK, relapse_window=RELAPSE_WINDOW):
    """Outcome of one event window for n_ch recovery channels (base, each distortion margin),
    testing only the frames that matter. Same headline rule as classify(first_verdict=False,
    onset_window=...), with these differences in what is tested:

    * Onset tests run on the first frames only (frame - t0 <= onset_window). The first onset
      there is shared by every channel.
    * No onset: the event is anticipated, and "better" means a channel's recovery test is
      confidently ahead somewhere in those same onset-window frames (else "indistinguishable").
    * Onset: recovery is searched after it, in blocks of `block` frames, for the channels still
      unrecovered; the search stops when all have recovered or the window ends (censored).
    * Relapse: after a recovery at frame r, a confident onset test in frames r+1 .. r+relapse_window
      (`relapse`), or a run of RELAPSE_RUN consecutive ones starting there (it may extend
      RELAPSE_RUN - 1 frames past the range; `relapse3`), both counted per channel. Nothing later
      counts, and no later frame is tested.
    * oracle: no onset rule; per channel, `first_ahead` is the first confidently-ahead frame,
      searched from the window start (for Oracle B).

    evaluate(idx, chs, need_on) tests window rows idx (array) and returns a dict with "on",
    "lbf_on" (when need_on or channel 0 is requested; arrays over idx), "rec", "lbf_rec"
    (arrays (len(chs), len(idx)); channel 0 is the base) and optionally "mean_I".

    Returns the per-channel outcome dicts (classify's fields), per-channel first_ahead (index
    into frames, oracle only), and the statistics computed (NaN where untested).
    """
    frames = np.asarray(frames)
    n = len(frames)
    don, lon = np.full(n, np.nan), np.full(n, np.nan)
    drec, lrec = np.full((n_ch, n), np.nan), np.full((n_ch, n), np.nan)
    mean_I = np.full(n, np.nan)

    def run(idx, chs, need_on):
        v = evaluate(idx, chs, need_on)
        if "on" in v:
            don[idx], lon[idx] = v["on"], v["lbf_on"]
        for a, ch in enumerate(chs):
            drec[ch, idx], lrec[ch, idx] = v["rec"][a], v["lbf_rec"][a]
        if "mean_I" in v:
            mean_I[idx] = v["mean_I"]

    def search(chs, lo, found):
        """Blocks from row lo: record in found[ch] the first row with recovery >= log_k."""
        pos = lo
        while pos < n and any(found[ch] is None for ch in chs):
            live = [ch for ch in chs if found[ch] is None]
            blk = np.arange(pos, min(pos + block, n))
            run(blk, live, 0 in live)
            for ch in live:
                hit = np.flatnonzero(drec[ch, blk] >= log_k)
                if len(hit):
                    found[ch] = int(blk[hit[0]])
            pos += block

    chs = list(range(n_ch))
    k = int(np.searchsorted(frames - t0, onset_window, side="right"))
    if k:
        run(np.arange(k), chs, True)
    found = [None] * n_ch
    for ch in chs:
        hit = np.flatnonzero(drec[ch, :k] >= log_k)
        if len(hit):
            found[ch] = int(hit[0])
    none = {"onset": None, "react": None, "relapse": False, "relapse_after": None,
            "relapse3": False, "relapse3_after": None}
    stats = {"on": don, "lbf_on": lon, "rec": drec, "lbf_rec": lrec, "mean_I": mean_I, "k": k}
    if oracle:
        search(chs, k, found)
        return [None] * n_ch, found, stats

    on = np.flatnonzero(don[:k] >= log_k)
    if not len(on):  # anticipated: "better" only if confidently ahead within the onset window
        out = [{"status": "better" if found[ch] is not None else "indistinguishable",
                "duration": None, **none} for ch in chs]
        return out, [None] * n_ch, {**stats, "onset": None, "react": [None] * n_ch}
    o = int(on[0])
    found = [None] * n_ch
    for ch in chs:  # recovery is a frame after the onset (rows o + 1 .. k - 1 are already tested)
        hit = np.flatnonzero(drec[ch, o + 1 : k] >= log_k)
        if len(hit):
            found[ch] = o + 1 + int(hit[0])
    search(chs, max(k, o + 1), found)
    # relapse: onset tests on the rows after each recovery (those not yet tested)
    ends = {}
    need = np.zeros(n, bool)
    for ch in chs:
        if found[ch] is not None:
            r = found[ch]
            ends[ch] = (
                int(np.searchsorted(frames, frames[r] + relapse_window, side="right")),
                int(np.searchsorted(frames, frames[r] + relapse_window + RELAPSE_RUN - 1,
                                    side="right")),
            )
            need[r + 1 : ends[ch][1]] = True
    todo = np.flatnonzero(need & np.isnan(don))
    if len(todo):
        run(todo, [], True)
    out = []
    for ch in chs:
        r = found[ch]
        if r is None:
            out.append({**none, "status": "censored", "onset": int(frames[o] - t0),
                        "duration": int(frames[-1] - t0)})
            continue
        hi, hi3 = ends[ch]
        again = np.flatnonzero(don[r + 1 : hi] >= log_k)
        seq = _first_run(don[r + 1 : hi3] >= log_k, RELAPSE_RUN)
        seq = seq if seq is not None and seq < hi - (r + 1) else None
        out.append({
            "status": "recovered",
            "onset": int(frames[o] - t0),
            "react": int(frames[r] - t0),
            "duration": int(frames[r] - t0),
            "relapse": bool(len(again)),
            "relapse_after": int(frames[r + 1 + again[0]] - frames[r]) if len(again) else None,
            "relapse3": seq is not None,
            "relapse3_after": int(frames[r + 1 + seq] - frames[r]) if seq is not None else None,
        })
    return out, [None] * n_ch, {**stats, "onset": o, "react": found}


def gaussian_logpdf(y, mu, S):
    """log N(y; mu, S), rowwise."""
    L = np.linalg.cholesky(S)
    z = np.linalg.solve(L, (y - mu)[..., None])[..., 0]
    return (
        -0.5 * (z * z).sum(-1)
        - np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1)
        - 0.5 * y.shape[-1] * np.log(2.0 * np.pi)
    )


# ---------------------------------------------------------------------------
# Mixture references (docs/react-spec.md: "the real climatology")
# ---------------------------------------------------------------------------
#
# The climatology is a Gaussian mixture per frame, p_ref,t = sum_k w_tk N(mu_tk, S_tk). The
# truth-location machinery above works in the mixture's moment-matched coordinates, but:
#   * the information level is entropic: I_traj = h(p_ref,t) - h(q_t), so the truth spread is
#     Omega_t = c * s_t * S_mm,t with s_t = exp(h_mix,t - h_mm,t) <= 1 (a Gaussian with the
#     mixture's entropy, shaped like its covariance);
#   * the advantage is scored against the mixture itself,
#       I_t(m) = E_q[log p_model] - E_q[log p_ref],  q = N(m, Omega_t),
#     with E_q[log p_ref] by tensor Gauss-Hermite quadrature and the posterior and prior tail
#     probabilities of I_t by Monte Carlo over m (common random numbers, shared by all models).


def gh_nodes_2d(n=5):
    """Probabilists' Gauss-Hermite nodes (n*n, 2) and weights (n*n,) for E over N(0, I_2)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    X, Y = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w)
    return np.column_stack([X.ravel(), Y.ravel()]), W.ravel()


def _mix_prep(weights, means, covs):
    """Precompute per-frame component inverses and log normalisers. weights (n, K)."""
    covs = covs + 1e-10 * _EYE2
    inv = np.linalg.inv(covs)
    logdet = np.linalg.slogdet(covs)[1]
    lognorm = np.log(np.clip(weights, 1e-300, None)) - 0.5 * logdet - np.log(2.0 * np.pi)
    return means, inv, lognorm


def mixture_logpdf(Y, prep):
    """log p_mix at points Y (n, ..., 2), mixture parameters per frame n."""
    means, inv, lognorm = prep
    n, K = lognorm.shape
    extra = Y.ndim - 2
    out = []
    for k in range(K):
        mu = means[:, k].reshape((n,) + (1,) * extra + (2,))
        P = inv[:, k].reshape((n,) + (1,) * extra + (2, 2))
        d = Y - mu
        q = np.einsum("...i,...ij,...j->...", d, P, d)
        out.append(lognorm[:, k].reshape((n,) + (1,) * extra) - 0.5 * q)
    out = np.stack(out, 0)
    peak = out.max(0)
    return peak + np.log(np.exp(out - peak).sum(0))


def mixture_moments(weights, means, covs):
    mu = np.einsum("nk,nki->ni", weights, means)
    d = means - mu[:, None]
    S = np.einsum("nk,nkij->nij", weights, covs + np.einsum("nki,nkj->nkij", d, d))
    return mu, S


def mixture_entropy(weights, means, covs, nodes, gw):
    """Differential entropy of each frame's mixture, by Gauss-Hermite within each component."""
    prep = _mix_prep(weights, means, covs)
    L = np.linalg.cholesky(covs + 1e-10 * _EYE2)  # (n, K, 2, 2)
    Y = means[:, :, None, :] + np.einsum("nkij,gj->nkgi", L, nodes)  # (n, K, G, 2)
    lp = mixture_logpdf(Y, prep)  # (n, K, G)
    return -np.einsum("nk,nkg,g->n", weights, lp, gw)


def gaussian_entropy(S):
    return 1.0 + np.log(2.0 * np.pi) + 0.5 * np.linalg.slogdet(S)[1]


def expected_logpdf_mixture(m, Omega, prep, nodes, gw):
    """E_{y ~ N(m, Omega)} log p_mix(y). m (n, J, 2), Omega (n, 2, 2) -> (n, J)."""
    L = np.linalg.cholesky(Omega)
    Y = m[:, :, None, :] + np.einsum("nij,gj->ngi", L, nodes)[:, None]  # (n, J, G, 2)
    return np.einsum("njg,g->nj", mixture_logpdf(Y, prep), gw)


def expected_logpdf_gaussian(m, Omega, mu, S):
    """E_{y ~ N(m, Omega)} log N(y; mu, S), closed form. m (n, J, 2) -> (n, J)."""
    Si = np.linalg.inv(S)
    d = m - mu[:, None]
    quad = np.einsum("nji,nik,njk->nj", d, Si, d)
    tr = np.einsum("nij,nji->n", Si, Omega)
    return -0.5 * (quad + tr[:, None] + np.linalg.slogdet(S)[1][:, None]) - np.log(2.0 * np.pi)


class MixtureTests:
    """Shared Monte Carlo state for one event window at one information level.

    Draws posterior and prior samples of the truth mean once, evaluates the model-free
    reference term once, then scores any number of models against it.
    """

    def __init__(self, post_mu, post_C, prior_mu, prior_C, Omega, ref_prep, Zq, Zp, nodes, gw):
        self.Omega = Omega
        self.mq = post_mu[:, None] + np.einsum("nij,kj->nki", np.linalg.cholesky(post_C), Zq)
        self.mp = prior_mu[:, None] + np.einsum("nij,kj->nki", np.linalg.cholesky(prior_C), Zp)
        self.Aq = expected_logpdf_mixture(self.mq, Omega, ref_prep, nodes, gw)
        self.Ap = expected_logpdf_mixture(self.mp, Omega, ref_prep, nodes, gw)

    @staticmethod
    def _log_odds_mc(hits, J):
        p = (hits + 0.5) / (J + 1.0)
        return np.log(p) - np.log1p(-p)

    def test(self, idx, mu_m, S_m, eps=0.0):
        """lbf_on, lbf_rec, mean I for a Gaussian model on window rows idx."""
        Om = self.Omega[idx]
        Iq = expected_logpdf_gaussian(self.mq[idx], Om, mu_m, S_m) - self.Aq[idx]
        Ip = expected_logpdf_gaussian(self.mp[idx], Om, mu_m, S_m) - self.Ap[idx]
        Jq, Jp = Iq.shape[1], Ip.shape[1]
        lbf_on = self._log_odds_mc((Iq < 0).sum(1), Jq) - self._log_odds_mc((Ip < 0).sum(1), Jp)
        lbf_rec = self._log_odds_mc((Iq > eps).sum(1), Jq) - self._log_odds_mc(
            (Ip > eps).sum(1), Jp
        )
        return lbf_on, lbf_rec, Iq.mean(1)


def knowable_share(c, noise=0.0):
    """Share of the whitened variance that is knowable: 1 - c_t - noise_t (floored at 0.03 when
    there is noise). c may be per frame; noise is the whitened white-noise variance."""
    c = np.asarray(c, float)
    noise = np.broadcast_to(np.asarray(noise, float), c.shape)
    if noise.any():
        return np.clip(1.0 - c - noise, 0.03, None)
    return 1.0 - c


def truth_posterior_het(frames, w, rho_a, rho_b, c, noise=0.0):
    """truth_posterior with a per-frame truth fraction c_t (entropic information level).

    noise: per-frame variance (whitened units) of white measurement noise on the observation,
    the truth-spread floor sigma^2 I whitened; the knowable share is then 1 - c_t - noise_t
    (knowable_share). noise = 0 is the original computation exactly."""
    c = np.broadcast_to(np.asarray(c, float), (len(frames),))
    noise = np.broadcast_to(np.asarray(noise, float), (len(frames),))
    va = knowable_share(c, noise)
    frames = np.asarray(frames)
    n = len(frames)
    lag = np.abs(frames[:, None] - frames[None, :])
    n_l = len(rho_a) - 1
    idx = np.minimum(lag, n_l)
    sa, sb = np.sqrt(va), np.sqrt(c)
    Ka = np.outer(sa, sa) * np.where(lag <= n_l, rho_a[idx], 0.0)
    Kw = (Ka + np.outer(sb, sb) * np.where(lag <= n_l, rho_b[idx], 0.0) + np.diag(noise)
          + 1e-9 * np.eye(n))
    L = cholesky(Kw, lower=True)
    V = solve_triangular(L, w, lower=True)
    U = solve_triangular(L, Ka, lower=True)
    Ut = np.triu(U)
    mean_f = Ut.T @ V
    var_f = np.maximum(va - (Ut * Ut).sum(0), 1e-12)
    mean_s = U.T @ V
    var_s = np.maximum(va - (U * U).sum(0), 1e-12)
    return mean_f, var_f, mean_s, var_s


# ---------------------------------------------------------------------------
# Regime-conditioned tests: exact, no Monte Carlo
# ---------------------------------------------------------------------------
#
# A two-part climatology is two Gaussian regimes with speed-dependent weights (moving, and
# stopped: for a walking person the stopped weight is the chance the walk ends within the
# lead). Make the regime explicit in the truth model:
#
#   prior:     k ~ w_tk,  m | k ~ N(mu_tk, (1 - c) S_tk),  q | k, m = N(m, c S_tk)
#
# so the prior predictive of y is the climatology mixture exactly, and I_traj = -log c is the
# information beyond the climatology within a regime. The observations' evidence about m is
# the shared machinery's Gaussian posterior with its (moment-matched) Gaussian prior divided
# out; combined with each regime's prior it gives a Gaussian posterior per regime and the
# posterior regime probabilities. Given the regime the advantage is quadratic in m,
#
#   I_k(m) = E_q[log p_model] - E_q[log w_k N_k],
#
# scoring the climatology by the regime the truth is in (the complete-data score: exact
# wherever that regime explains the position, a slight understatement of the climatology only
# where the two overlap), so every tail probability is the exact generalised chi-square above.


class RegimeTests:
    """Exact regime-conditioned onset/recovery tests for one event window at one level.

    post_mu, post_C: the shared posterior of m (n, 2), (n, 2, 2); mm_mu, mm_prior: the
    Gaussian prior it was computed under; weights (n, K), means (n, K, 2), covs (n, K, 2, 2):
    the climatology's regimes; c: truth fraction within a regime.
    """

    def __init__(self, post_mu, post_C, mm_mu, mm_prior, weights, means, covs, c, gh=5, sig2=0.0):
        n, K = weights.shape
        self.K, self.c, self.sig2 = K, float(c), float(sig2)
        self.w, self.means, self.covs = weights, means, covs
        Ai = np.linalg.inv(post_C)
        Bi = np.linalg.inv(mm_prior)
        eta = np.einsum("nij,nj->ni", Ai, post_mu) - np.einsum("nij,nj->ni", Bi, mm_mu)
        Lam = Ai - Bi
        ev, V = np.linalg.eigh(0.5 * (Lam + np.swapaxes(Lam, 1, 2)))
        Lam = np.einsum("nij,nj,nkj->nik", V, np.clip(ev, 0.0, None), V)  # evidence precision
        self.post_mean = np.empty((n, K, 2))
        self.post_cov = np.empty((n, K, 2, 2))
        log_ev = np.empty((n, K))
        # truth spread Omega_k = c S_k + sig2 I; the regime's prior spread of m is what is left of
        # S_k: V_k = (1 - c) S_k - sig2 I, eigen-clipped to stay positive definite (sig2 = 0: as is)
        self.Om = [self.c * covs[:, k] + self.sig2 * _EYE2 for k in range(K)]
        self.V = []
        for k in range(K):
            Vk = (1.0 - self.c) * covs[:, k]
            if self.sig2 > 0.0:
                Vk = Vk - self.sig2 * _EYE2
                e_, U_ = np.linalg.eigh(Vk)
                fl = 0.02 * (1.0 - self.c) * 0.5 * np.trace(covs[:, k], axis1=1, axis2=2)
                Vk = np.einsum("nij,nj,nkj->nik", U_, np.maximum(e_, fl[:, None]), U_)
            self.V.append(Vk)
        for k in range(K):
            Vk = self.V[k]
            Qk = np.linalg.inv(Vk)
            Pk = Qk + Lam
            Pk_inv = np.linalg.inv(Pk)
            hk = np.einsum("nij,nj->ni", Qk, means[:, k]) + eta
            self.post_cov[:, k] = Pk_inv
            self.post_mean[:, k] = np.einsum("nij,nj->ni", Pk_inv, hk)
            log_ev[:, k] = (
                0.5 * np.einsum("ni,nij,nj->n", hk, Pk_inv, hk)
                - 0.5 * np.einsum("ni,nij,nj->n", means[:, k], Qk, means[:, k])
                - 0.5 * np.linalg.slogdet(Vk)[1]
                - 0.5 * np.linalg.slogdet(Pk)[1]
            )
        lp = np.log(np.clip(weights, 1e-300, None)) + log_ev
        lp -= lp.max(1, keepdims=True)
        self.post_w = np.exp(lp)
        self.post_w /= self.post_w.sum(1, keepdims=True)
        # Overlap correction: scoring the climatology by the truth's regime understates it by
        # E[-log r_k(y)], r_k the regime's share of the mixture at y. It does not depend on the
        # model, so it is computed once per frame and regime, averaged over y's predictive
        # under that regime (posterior or prior), and subtracted as a constant: the advantage
        # stays quadratic.
        nodes, gw = gh_nodes_2d(gh)
        prep = _mix_prep(weights, means, covs)
        self.corr_post = np.empty((n, K))
        self.corr_prior = np.empty((n, K))
        for k in range(K):
            for which, loc, cov in (
                ("post", self.post_mean[:, k], self.post_cov[:, k] + self.Om[k]),
                ("prior", means[:, k], covs[:, k]),
            ):
                Y = loc[:, None] + np.einsum("nij,gj->ngi", np.linalg.cholesky(cov), nodes)
                comp = _mix_prep(weights[:, k : k + 1], means[:, k : k + 1], covs[:, k : k + 1])
                gap = mixture_logpdf(Y, prep) - mixture_logpdf(Y, comp)  # -log r_k(y) >= 0
                (self.corr_post if which == "post" else self.corr_prior)[:, k] = gap @ gw

    def _tails(self, idx, mu_m, S_m, eps, which, need_lt=True):
        """Regime-weighted P(I < 0) and P(I > eps), and E[I], posterior or prior. eps may be a
        per-frame array aligned with idx."""
        p_lt = np.zeros(len(idx))
        p_gt = np.zeros(len(idx))
        mean_I = np.zeros(len(idx))
        for k in range(self.K):
            mu_k, S_k = self.means[idx, k], self.covs[idx, k]
            if which == "post":
                m, C, wk = self.post_mean[idx, k], self.post_cov[idx, k], self.post_w[idx, k]
                corr = self.corr_post[idx, k]
            else:
                m, C, wk = mu_k, self.V[k][idx], self.w[idx, k]
                corr = self.corr_prior[idx, k]
            M, g, h = advantage_quadratic(m, C, mu_m, S_m, mu_k, S_k, self.c, self.sig2)
            h = h - np.log(np.clip(self.w[idx, k], 1e-300, None)) - corr
            p_lt += wk * prob_greater(-M, -g, -h, 0.0)
            p_gt += wk * prob_greater(M, g, h, eps)
            mean_I += wk * (np.einsum("nii->n", M) + h)
        return p_lt, p_gt, mean_I

    def test_full(self, idx, mu_m, S_m, eps=0.0, margins=(), base=True):
        """Per-frame statistics for a Gaussian model on rows idx: posterior log-odds of onset
        (I < 0) and recovery (I > eps), their log Bayes factors against the climatology's
        prior, and the posterior mean advantage.

        margins: further recovery margins, each a per-frame array aligned with idx (e.g. the
        rate-distortion margin for a target distortion); for each, "post_rec_m{j}" and
        "lbf_rec_m{j}" are the recovery statistics for I > margin.
        base=False skips the onset / base-recovery statistics (and mean_I): only the margin
        recoveries are returned, for callers that need nothing else on these frames."""
        out = {}
        if base:
            q_lt, q_gt, mean_I = self._tails(idx, mu_m, S_m, eps, "post")
            p_lt, p_gt, _ = self._tails(idx, mu_m, S_m, eps, "prior")
            post_on, post_rec = _log_odds(q_lt), _log_odds(q_gt)
            out = {
                "post_on": post_on,
                "post_rec": post_rec,
                "lbf_on": post_on - _log_odds(p_lt),
                "lbf_rec": post_rec - _log_odds(p_gt),
                "mean_I": mean_I,
            }
        for j, mg in enumerate(margins):
            mg = np.broadcast_to(np.asarray(mg, float), (len(idx),))
            if base and not mg.any() and eps == 0.0:
                out[f"post_rec_m{j}"], out[f"lbf_rec_m{j}"] = out["post_rec"], out["lbf_rec"]
                continue
            _, q_m, _ = self._tails(idx, mu_m, S_m, mg, "post", need_lt=False)
            _, p_m, _ = self._tails(idx, mu_m, S_m, mg, "prior", need_lt=False)
            out[f"post_rec_m{j}"] = _log_odds(q_m)
            out[f"lbf_rec_m{j}"] = out[f"post_rec_m{j}"] - _log_odds(p_m)
        return out

    def test(self, idx, mu_m, S_m, eps=0.0, decision="posterior"):
        """Decision statistics (onset, recovery) and posterior mean advantage.

        decision "posterior": posterior log-odds, compared with log(p / (1 - p)); the
        climatology's prior counts as knowledge. "bayes-factor": log Bayes factors, compared
        with log k; only the movement from that prior counts."""
        v = self.test_full(idx, mu_m, S_m, eps)
        if decision == "posterior":
            return v["post_on"], v["post_rec"], v["mean_I"]
        return v["lbf_on"], v["lbf_rec"], v["mean_I"]


def regime_predictive(post_mu, post_C, mm_mu, mm_prior, weights, means, covs, c, sig2=0.0):
    """Posterior predictive of y under the regime truth model, moment-matched to a Gaussian.

    Per regime k: m | k, data ~ N(post_mean_k, post_cov_k) (RegimeTests' regime split of the
    shared posterior) and y | m, k ~ N(m, c S_k); the regimes are weighted by their posterior
    probabilities. With a hindsight (smoothed) posterior this is Oracle B, the most a forecast
    can extract at the assumed information level.
    """
    rt = RegimeTests(post_mu, post_C, mm_mu, mm_prior, weights, means, covs, c, sig2=sig2)
    mean = np.einsum("nk,nki->ni", rt.post_w, rt.post_mean)
    d = rt.post_mean - mean[:, None]
    cov = np.einsum(
        "nk,nkij->nij", rt.post_w,
        rt.post_cov + np.stack(rt.Om, 1) + np.einsum("nki,nkj->nkij", d, d)
    )
    return mean, cov + 1e-12 * _EYE2


# ---------------------------------------------------------------------------
# Regime tests with the exact mixture score: overlap term per truth location
# ---------------------------------------------------------------------------
#
# Given the regime k, the advantage against the real mixture is
#
#   I_k(m) = Q_k(m) - G_k(m),   Q_k(m) = E_q[log p_model] - E_q[log w_k N_k]   (quadratic in m)
#   G_k(m) = E_{y ~ N(m, c S_k)} log(1 + sum_j w_j N_j(y) / (w_k N_k(y))) >= 0,
#
# G_k does not depend on the model, so it is computed once per frame and shared by every
# model. With y = m + chol(c S_k) x_g at the Gauss-Hermite nodes x_g, each log-ratio is a
# quadratic in m, so on a fixed set of points m = loc + chol(C) z it is a quadratic in z per node
# and G_k is evaluated exactly (to quadrature accuracy) with no table and no interpolation.
#
# Tail probabilities of I_k are integrated over m's standardised coordinates on a polar grid:
# along each direction the sign changes are solved from the local quadratic and the Gaussian
# mass on each side added in closed form (P(r > a) = exp(-a^2 / 2) for a standard 2D normal).
# Both tails are summed directly, so very small probabilities keep their precision.

POLAR_ANGLES = 64
POLAR_RADII = np.linspace(0.0, 8.0, 17)


def _polar_points(n_angles=POLAR_ANGLES, radii=POLAR_RADII):
    phi = 2.0 * np.pi * (np.arange(n_angles) + 0.5) / n_angles
    d = np.column_stack([np.cos(phi), np.sin(phi)])  # (A, 2)
    return d, radii


def _polar_tails(I, thr, radii, A, B):
    """P(I > thr) and P(I < thr) from I on a polar grid (n, A, R) over a standard 2D normal.

    I = A r^2 + B r + C(r) along each ray with C linear within each radial interval; each
    crossing is solved from that quadratic. Both tails are summed directly, interval by
    interval, so each keeps its precision.
    """
    e = np.exp(-0.5 * radii * radii)
    I0, I1 = I[..., :-1], I[..., 1:]
    p0, p1 = I0 > thr, I1 > thr
    r0, r1 = radii[:-1], radii[1:]
    a, b = A[..., None], B[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        C0 = I0 - a * r0 * r0 - b * r0
        sl = (I1 - a * r1 * r1 - b * r1 - C0) / (r1 - r0)
        q1, q2 = _quadratic_roots(np.broadcast_to(a, I0.shape), b + sl, C0 - sl * r0 - thr)
        rr = r0 + (thr - I0) / (I1 - I0) * (r1 - r0)
        tol = 1e-9
        rr = np.where((q1 >= r0 - tol) & (q1 <= r1 + tol), q1,
                      np.where((q2 >= r0 - tol) & (q2 <= r1 + tol), q2, rr))
        rr = np.clip(np.nan_to_num(rr, nan=0.5 * (r0 + r1)), r0, r1)
    er = np.exp(-0.5 * rr * rr)
    lo, hi = e[:-1] - er, er - e[1:]
    seg = e[:-1] - e[1:]
    pos = np.where(p0 & p1, seg, 0.0) + np.where(p0 & ~p1, lo, 0.0) + np.where(~p0 & p1, hi, 0.0)
    neg = np.where(~p0 & ~p1, seg, 0.0) + np.where(p0 & ~p1, hi, 0.0) + np.where(~p0 & p1, lo, 0.0)
    thr = np.asarray(thr, float)
    last = I[..., -1] > (thr[..., 0] if thr.ndim == I.ndim else thr)
    gt = pos.sum(-1) + np.where(last, e[-1], 0.0)
    lt = neg.sum(-1) + np.where(last, 0.0, e[-1])
    return gt.mean(-1), lt.mean(-1)


def _quad_in_z(D0, Dz, P, feats):
    """(d)' P (d) with d = D0 + Dz z at points z: D0 (n, G, 2), Dz (n, 2, 2), P (n, 2, 2).

    feats = [z1^2, 2 z1 z2, z2^2, z1, z2] (Pz, 5). Returns (n, G, Pz)."""
    Mq = np.einsum("nji,njk,nkl->nil", Dz, P, Dz)
    quad = feats[:, :3] @ np.stack([Mq[:, 0, 0], Mq[:, 0, 1], Mq[:, 1, 1]], 0)  # (Pz, n)
    PD0 = np.einsum("nij,ngj->ngi", P, D0)
    lin = 2.0 * np.einsum("nji,ngj->ngi", Dz, PD0)
    const = np.einsum("ngi,ngi->ng", D0, PD0)
    return quad.T[:, None, :] + np.einsum("pf,ngf->ngp", feats[:, 3:], lin) + const[..., None]


def overlap_on_grid(loc, C, weights, means, covs, k, c, z, nodes, gw, blob_scale=(1.0,), sig2=0.0):
    """G_k at m = loc + chol(C) z for points z (P, 2); per-frame loc (n, 2), C (n, 2, 2).

    G_k(m) = E_{y ~ N(m, O)} log(1 + sum_j w_j N_j(y) / (w_k N_k(y))), O = c S_k + sig2 I. When another
    regime is much narrower than O (the stopped regime inside a moving truth), the integrand
    is a small blob inside a wide Gaussian that fixed nodes miss. So the expectation is taken
    with two Gauss-Hermite rules combined by the balance heuristic: one on q_0 = N(m, O), one
    per other regime on q_j proportional to N(y; m, O) N_j(y) (the blob), each node weighted by
    q_0 / sum_r q_r. Every node position is affine in z, so every log-density is a quadratic in
    z and the whole evaluation is exact algebra plus the quadrature.
    """
    n, K = weights.shape
    Lc = np.linalg.cholesky(C + 1e-15 * _EYE2)
    O = c * covs[:, k] + sig2 * _EYE2 + 1e-15 * _EYE2
    Oi = np.linalg.inv(O)
    _, inv, lognorm = _mix_prep(weights, means, covs)
    feats = np.stack([z[:, 0] ** 2, 2.0 * z[:, 0] * z[:, 1], z[:, 1] ** 2, z[:, 0], z[:, 1]], 1)
    others = [j for j in range(K) if j != k]
    # rules: (Y0 (n, G, 2), Yz (n, 2, 2), centre offset mu0 (n, 2), centre slope Mz, cov)
    rules = [(loc[:, None] + np.einsum("nij,gj->ngi", np.linalg.cholesky(O), nodes), Lc, loc, Lc, O)]
    for j in others:
        Sj = np.linalg.inv(Oi + inv[:, j])
        A = np.einsum("nij,njk->nik", Sj, Oi)
        mu0 = np.einsum("nij,nj->ni", A, loc) + np.einsum("nij,njk,nk->ni", Sj, inv[:, j], means[:, j])
        Mz = np.einsum("nij,njk->nik", A, Lc)
        for bs in blob_scale:
            Sb = bs * Sj
            rules.append((mu0[:, None] + np.einsum("nij,gj->ngi", np.linalg.cholesky(Sb), nodes), Mz, mu0, Mz, Sb))
    cov_inv = [np.linalg.inv(r[4]) for r in rules]
    cov_logdet = [np.linalg.slogdet(r[4])[1] for r in rules]
    total = 0.0
    for Y0, Yz, _, _, _ in rules:
        # log-ratios l_j = log w_j N_j(y) - log w_k N_k(y)
        Qk = _quad_in_z(Y0 - means[:, k][:, None], Yz, inv[:, k], feats)
        ls = [
            0.5 * Qk - 0.5 * _quad_in_z(Y0 - means[:, j][:, None], Yz, inv[:, j], feats)
            + (lognorm[:, j] - lognorm[:, k])[:, None, None]
            for j in others
        ]
        L = np.stack(ls, 0)
        peak = np.maximum(L.max(0), 0.0)
        gap = peak + np.log(np.exp(-peak) + np.exp(L - peak).sum(0))
        # balance-heuristic weight q_0 / sum_r q_r at these nodes
        lq = np.stack([
            -0.5 * _quad_in_z(Y0 - mu0[:, None], Yz - Mz, cov_inv[r], feats)
            - 0.5 * cov_logdet[r][:, None, None]
            for r, (_, _, mu0, Mz, _) in enumerate(rules)
        ], 0)
        wt = np.exp(lq[0] - (lq.max(0) + np.log(np.exp(lq - lq.max(0)).sum(0))))
        total = total + np.einsum("ngp,g->np", gap * wt, gw)
    return total


class OverlapTable:
    """The overlap term G_k tabulated once per lead and information level.

    In the mover's body frame (origin at the last position, x along the heading) the
    climatology depends on the speed alone, so G_k(m) is a function of (speed, m_body). The
    table stores it on a speed grid times a square body-frame grid that is uniform (spacing h)
    within +-core of the origin, where the stopped regime and the sharp part of G sit, and
    grows geometrically (ratio `growth`) out to +-extent, where G varies slowly. The position
    index is a closed-form function of m, so a lookup is arithmetic plus one C interpolation
    call (trilinear in speed and position). Points beyond the table are computed directly.

    Build with OverlapTable.build(body, c, ...), where body(v) returns the climatology's
    weights (V, K), means (V, K, 2) and covs (V, K, 2, 2) in the body frame at speeds v.
    """

    def __init__(self, v, core, h, growth, X, G, c, sig2=0.0):
        self.v, self.core, self.h, self.growth, self.X = np.asarray(v, float), core, h, growth, X
        self.G, self.c, self.sig2 = G, float(c), float(sig2)
        self.K = G.shape[0]

    def _index(self, m):
        am = np.abs(m)
        a, h, g = self.core, self.h, self.growth
        with np.errstate(invalid="ignore"):
            far = a / h + np.log1p(np.maximum(am - a, 0.0) * (g - 1.0) / h) / np.log(g)
        return np.sign(m) * np.where(am <= a, am / h, far) + self.X

    @staticmethod
    def axis(core, h, growth, extent):
        """Node positions of one axis and its half-length X in index units."""
        X = int(np.ceil(core / h + np.log1p((extent - core) * (growth - 1.0) / h) / np.log(growth)))
        x = np.arange(-X, X + 1, dtype=float)
        ax = np.abs(x)
        m = np.where(ax <= core / h, ax * h, core + h * (growth ** (ax - core / h) - 1.0) / (growth - 1.0))
        return np.sign(x) * m, X

    @classmethod
    def build(cls, body, c, v_grid, core, h, growth, extent, overlap_gh=9, blob_scale=(4.0,),
              chunk=4096, sig2=0.0):
        nodes_ax, X = cls.axis(core, h, growth, extent)
        M1, M2 = np.meshgrid(nodes_ax, nodes_ax, indexing="ij")
        pts = np.column_stack([M1.ravel(), M2.ravel()])
        w, mu, S = body(np.asarray(v_grid, float))
        V, K = w.shape
        nodes, gw = gh_nodes_2d(overlap_gh)
        G = np.empty((K, V, len(nodes_ax), len(nodes_ax)), np.float32)
        zero, eye = np.zeros((1, 2)), np.eye(2)[None]
        for k in range(K):
            for j in range(V):
                out = np.empty(len(pts))
                for s0 in range(0, len(pts), chunk):
                    out[s0 : s0 + chunk] = overlap_on_grid(
                        zero, eye, w[j : j + 1], mu[j : j + 1], S[j : j + 1], k, c,
                        pts[s0 : s0 + chunk], nodes, gw, blob_scale=blob_scale, sig2=sig2,
                    )[0]
                G[k, j] = out.reshape(len(nodes_ax), len(nodes_ax))
        return cls(v_grid, core, h, growth, X, G, c, sig2)

    def save(self, path):
        np.save(path + ".npy", self.G)
        np.savez(path + "_meta.npz", v=self.v, core=self.core, h=self.h, growth=self.growth,
                 X=self.X, c=self.c, sig2=self.sig2)

    @classmethod
    def load(cls, path, mmap=True):
        meta = np.load(path + "_meta.npz")
        G = np.load(path + ".npy", mmap_mode="r" if mmap else None)
        return cls(meta["v"], float(meta["core"]), float(meta["h"]), float(meta["growth"]),
                   int(meta["X"]), G, float(meta["c"]),
                   float(meta["sig2"]) if "sig2" in meta.files else 0.0)

    def lookup(self, k, speed, heading, anchor, m):
        """G_k at world points m (n, P, 2) given per-frame speed, heading and anchor (n,).

        Returns the values (n, P) and a mask of points beyond the table."""
        cs, sn = np.cos(heading)[:, None], np.sin(heading)[:, None]
        dx, dy = m[..., 0] - anchor[:, None, 0], m[..., 1] - anchor[:, None, 1]
        x1 = self._index(cs * dx + sn * dy)  # body frame: R' (m - anchor)
        x2 = self._index(-sn * dx + cs * dy)
        N = 2 * self.X
        off = (x1 < 0) | (x1 > N) | (x2 < 0) | (x2 > N)
        vi = np.interp(np.clip(speed, self.v[0], self.v[-1]), self.v, np.arange(len(self.v)))
        coords = np.stack([np.broadcast_to(vi[:, None], x1.shape), x1, x2]).reshape(3, -1)
        vals = map_coordinates(self.G[k], coords, order=1, mode="nearest", prefilter=False)
        return vals.reshape(x1.shape).astype(float), off


class RegimeTestsExact(RegimeTests):
    """RegimeTests scored against the real mixture: the overlap term per truth location.

    Same arguments as RegimeTests. The overlap term G_k is evaluated once on each regime's
    polar grid (posterior and prior) and shared by every model tested on this window.

    With table (an OverlapTable for this lead and level) and geom = (speed, heading, anchor)
    per frame, G_k is looked up at every grid point and only points beyond the table are
    computed directly. Without a table it is computed directly on every g_stride = (angle,
    radius) grid point and interpolated between them.
    """

    def __init__(self, post_mu, post_C, mm_mu, mm_prior, weights, means, covs, c, gh=5,
                 n_angles=POLAR_ANGLES, radii=POLAR_RADII, overlap_gh=9, blob_scale=(4.0,),
                 g_stride=(1, 1), chunk=8, min_weight=1e-12, table=None, geom=None, sig2=0.0):
        super().__init__(
            post_mu, post_C, mm_mu, mm_prior, weights, means, covs, c, gh=gh, sig2=sig2
        )
        if table is not None and abs(getattr(table, "sig2", 0.0) - self.sig2) > 1e-15:
            raise ValueError("overlap table built for a different noise floor")
        self.dirs, self.radii = _polar_points(n_angles, radii)
        nodes, gw = gh_nodes_2d(overlap_gh)
        n = len(weights)
        self.G = {}
        self.n_off = 0
        if table is not None:
            z = (self.dirs[:, None, :] * self.radii[None, :, None]).reshape(-1, 2)
            speed, heading, anchor = geom
        else:
            sa, sr = g_stride
            ia = np.arange(0, n_angles, sa)
            ir = np.unique(np.append(np.arange(0, len(radii), sr), len(radii) - 1))
            z = (self.dirs[ia][:, None, :] * self.radii[ir][None, :, None]).reshape(-1, 2)
        for k in range(self.K):
            for which in ("post", "prior"):
                if which == "post":
                    loc, C, wk = self.post_mean[:, k], self.post_cov[:, k], self.post_w[:, k]
                else:
                    loc, C, wk = means[:, k], self.V[k], weights[:, k]
                g = np.zeros((n, len(z)))
                rows = np.flatnonzero(wk > min_weight)
                if table is not None and len(rows):
                    m = loc[rows, None] + np.einsum(
                        "nij,pj->npi", np.linalg.cholesky(C[rows] + 1e-15 * _EYE2), z
                    )
                    vals, off = table.lookup(k, speed[rows], heading[rows], anchor[rows], m)
                    if off.any():
                        ri_, pi_ = np.nonzero(off)
                        fr = rows[ri_]
                        vals[ri_, pi_] = overlap_on_grid(
                            m[ri_, pi_], np.broadcast_to(1e-12 * _EYE2, (len(fr), 2, 2)),
                            weights[fr], means[fr], covs[fr], k, self.c, np.zeros((1, 2)),
                            nodes, gw, blob_scale=blob_scale, sig2=self.sig2,
                        )[:, 0]
                        self.n_off += len(fr)
                    g[rows] = vals
                    self.G[(k, which)] = g.reshape(n, n_angles, len(radii))
                    continue
                for s0 in range(0, len(rows), chunk):
                    r = rows[s0 : s0 + chunk]
                    g[r] = overlap_on_grid(
                        loc[r], C[r], weights[r], means[r], covs[r], k, self.c, z, nodes, gw,
                        blob_scale=blob_scale, sig2=self.sig2,
                    )
                if table is not None:
                    self.G[(k, which)] = g.reshape(n, n_angles, len(radii))
                    continue
                g = g.reshape(n, len(ia), len(ir))
                if len(ir) < len(radii):  # linear in radius
                    g = np.stack([np.interp(self.radii, self.radii[ir], row) for row in g.reshape(-1, len(ir))])
                    g = g.reshape(n, len(ia), len(radii))
                if sa > 1:  # periodic linear in angle
                    frac = (np.arange(n_angles) % sa) / sa
                    lo = np.arange(n_angles) // sa
                    hi = (lo + 1) % len(ia)
                    g = (1 - frac)[None, :, None] * g[:, lo] + frac[None, :, None] * g[:, hi]
                self.G[(k, which)] = g
        # E[G] over each regime's Gaussian, for the posterior mean advantage
        tw = np.gradient(self.radii) * self.radii * np.exp(-0.5 * self.radii**2)
        tw /= tw.sum()
        self.mean_G = {key: val.mean(1) @ tw for key, val in self.G.items()}

    def _tails(self, idx, mu_m, S_m, eps, which, need_lt=True):
        p_lt = np.zeros(len(idx))
        p_gt = np.zeros(len(idx))
        mean_I = np.zeros(len(idx))
        r = self.radii
        eps = np.broadcast_to(np.asarray(eps, float), (len(idx),))
        thr = eps[:, None, None]
        for k in range(self.K):
            mu_k, S_k = self.means[idx, k], self.covs[idx, k]
            if which == "post":
                m, C, wk = self.post_mean[idx, k], self.post_cov[idx, k], self.post_w[idx, k]
            else:
                m, C, wk = mu_k, self.V[k][idx], self.w[idx, k]
            corr = self.mean_G[(k, which)][idx]
            M, g, h = advantage_quadratic(m, C, mu_m, S_m, mu_k, S_k, self.c, self.sig2)
            h = h - np.log(np.clip(self.w[idx, k], 1e-300, None))
            A = np.einsum("ai,nij,aj->na", self.dirs, M, self.dirs)  # (n, A)
            B = np.einsum("ai,ni->na", self.dirs, g)
            Q = A[..., None] * (r * r) + B[..., None] * r + h[:, None, None]
            G = self.G[(k, which)][idx]
            gt, lt = _polar_tails(Q - G, thr, r, A, B)
            if need_lt and eps.any():  # onset is I < 0, recovery I > eps
                _, lt = _polar_tails(Q - G, 0.0, r, A, B)
            p_lt += wk * lt
            p_gt += wk * gt
            mean_I += wk * (np.einsum("nii->n", M) + h - corr)
        return p_lt, p_gt, mean_I


# -------------------------------------------------------------------------------------------------
# Gaussian-mixture forecasts (e.g. Trajectron++'s 25 modes): realized log scores and REACT's tests
# -------------------------------------------------------------------------------------------------
LOG2PI = np.log(2.0 * np.pi)


def interp_polar(g, sa, ir, A, radii, kind="cubic"):
    """Interpolate a polar-grid field known on angle indices 0, sa, 2 sa, ... and radius indices ir
    to the full (A, len(radii)) grid. g: (n, na, len(ir)). kind "linear": linear in radius,
    periodic linear in angle;
    "cubic": cubic spline in radius, trigonometric (FFT) interpolation in angle."""
    n, na, nr = g.shape
    R = len(radii)
    if kind == "cubic" and nr >= 4:
        g = CubicSpline(radii[ir], g, axis=2)(radii)
    elif nr < R:
        g = np.stack([np.interp(radii, radii[ir], row) for row in g.reshape(-1, nr)]).reshape(
            n, na, R
        )
    if na < A:
        if kind == "cubic" and na >= 4:
            F = np.fft.rfft(g, axis=1)
            Fp = np.zeros((n, A // 2 + 1, g.shape[2]), complex)
            k = F.shape[1]
            Fp[:, :k] = F
            if na % 2 == 0:
                Fp[:, na // 2] *= 0.5  # split the Nyquist term
            g = np.fft.irfft(Fp, n=A, axis=1) * (A / na)
        else:
            frac = (np.arange(A) % sa) / sa
            lo = np.arange(A) // sa
            hi = (lo + 1) % na
            g = (1 - frac)[None, :, None] * g[:, lo] + frac[None, :, None] * g[:, hi]
    return g


class MixtureForecast:
    """Per-frame Gaussian-mixture forecast for the frames of a track at one lead.

    weights (n, J), means (n, J, 2), covs (n, J, 2, 2); every component is kept (no pruning or
    merging). Also holds the exact moment-matched Gaussian. logpdf() is the one realized-score
    path (mixture_logpdf) shared with the climatology."""

    def __init__(self, weights, means, covs):
        w = np.asarray(weights, float)
        w = w / w.sum(1, keepdims=True)
        self.w, self.mu = w, np.asarray(means, float)
        self.S = np.asarray(covs, float) + 1e-12 * _EYE2
        self.raw = (w, self.mu, self.S)
        self.mm_mu, self.mm_S = mixture_moments(w, self.mu, self.S)
        self.mm_S = self.mm_S + 1e-12 * _EYE2
        self.P = np.linalg.inv(self.S)
        self.ld = np.linalg.slogdet(self.S)[1]
        self.logw = np.log(np.clip(w, 1e-300, None))
        self.M = w.shape[1]
        self._prep = None

    def take(self, idx):
        out = object.__new__(MixtureForecast)
        for name in ("w", "mu", "S", "mm_mu", "mm_S", "P", "ld", "logw"):
            setattr(out, name, getattr(self, name)[idx])
        out.raw = (out.w, out.mu, out.S)
        out.M, out._prep = self.M, None
        return out

    def logpdf(self, y):
        """Log density at one point per frame, y (n, 2)."""
        if self._prep is None:
            self._prep = _mix_prep(*self.raw)
        return mixture_logpdf(y[:, None], self._prep)[:, 0]


def _features(z):
    return np.stack(
        [z[:, 0] ** 2, 2.0 * z[:, 0] * z[:, 1], z[:, 1] ** 2, z[:, 0], z[:, 1], np.ones(len(z))], 1
    )


def expected_logn_on_grid(loc, Lc, Om, mu, P, ld, feats):
    """E_{y ~ N(m, Om)} log N_j(y) at m = loc + Lc z for the grid z (Pz): loc (n,2), Lc (n,2,2),
    Om (n,2,2),
    component mu (n,J,2), P (n,J,2,2), ld (n,J). Returns (n, J, Pz)."""
    d0 = loc[:, None] - mu  # (n,J,2)
    PL = np.einsum("njik,nkl->njil", P, Lc)
    Mq = np.einsum("nki,njkl->njil", Lc, PL)  # Lc' P Lc
    lin = 2.0 * np.einsum("nki,njk->nji", Lc, np.einsum("njkl,njl->njk", P, d0))
    const = np.einsum("nji,njik,njk->nj", d0, P, d0)
    tr = np.einsum("njik,nki->nj", P, Om)
    coef = np.empty(d0.shape[:2] + (6,))
    coef[..., 0] = Mq[..., 0, 0]
    coef[..., 1] = Mq[..., 0, 1]
    coef[..., 2] = Mq[..., 1, 1]
    coef[..., 3] = lin[..., 0]
    coef[..., 4] = lin[..., 1]
    coef[..., 5] = const + tr + ld
    coef = -0.5 * coef
    coef[..., 5] -= LOG2PI
    return np.einsum("njf,pf->njp", coef, feats)


class RegimeTestsMixture(RegimeTestsExact):
    """RegimeTestsExact for a Gaussian-mixture model forecast.

    The advantage of regime k at truth location m, with q = N(m, c S_k), is
    I_k(m) = E_q[log p_model] - E_q[log w_k N_k] - G_k(m). Write E_q[log p_model] = F_g(m) + Res(m),
    where F_g is the exact quadratic of the moment-matched Gaussian g (what advantage_quadratic
    returns) and Res the non-Gaussian remainder. Res is Gauss-Hermite (order gh) on q of log p at a
    coarse polar subset (stride = angle, radius steps) and interpolated; regimes carrying less than
    heavy_weight of the mass use the closed-form variational bound instead. With one component
    Res = 0 and this reduces to RegimeTestsExact. Use as

        X = RegimeTestsMixture.from_exact(exact, gh=5, stride=(8, 2))
        v = X.test_full_mixture(idx, fc.take(rows), eps, margins=margins)   # same dict as test_full
    """

    def __init__(self, *args, gh=5, stride=(8, 2), interp="cubic", heavy_weight=1e-3, **kw):
        super().__init__(*args, **kw)
        self._setup(gh, stride, interp, heavy_weight)

    def _setup(self, gh, stride, interp, heavy_weight):
        self.gh, self.stride, self.interp, self.heavy_weight = gh, stride, interp, heavy_weight
        self._terms = None
        self._feats = _features((self.dirs[:, None, :] * self.radii[None, :, None]).reshape(-1, 2))
        tw = np.gradient(self.radii) * self.radii * np.exp(-0.5 * self.radii**2)
        self._tw = tw / tw.sum()

    @classmethod
    def from_exact(cls, X, gh=5, stride=(8, 2), interp="cubic", heavy_weight=1e-3):
        """Wrap an existing RegimeTestsExact (shares its regime posteriors and overlap terms)."""
        out = object.__new__(cls)
        out.__dict__.update(X.__dict__)
        out._setup(gh, stride, interp, heavy_weight)
        return out

    def _set(self, k, which, idx):
        if which == "post":
            return self.post_mean[idx, k], self.post_cov[idx, k], self.post_w[idx, k]
        return self.means[idx, k], self.V[k][idx], self.w[idx, k]

    def model_terms(self, idx, fc, min_weight=1e-9):
        """Per (regime, posterior/prior): the Gaussian quadratic (M, g, h) of the moment-matched
        Gaussian and the mixture residual Res on the polar grid (n, A, R)."""
        n, A, R = len(idx), len(self.dirs), len(self.radii)
        terms = {}
        for k in range(self.K):
            mu_k, S_k = self.means[idx, k], self.covs[idx, k]
            Om = self.Om[k][idx]
            for which in ("post", "prior"):
                loc, C, wk = self._set(k, which, idx)
                Mq, gq, hq = advantage_quadratic(
                    loc, C, fc.mm_mu, fc.mm_S, mu_k, S_k, self.c, self.sig2
                )
                Lc = np.linalg.cholesky(C + 1e-15 * _EYE2)
                res = np.zeros((n, A * R))
                live = np.flatnonzero(wk > min_weight)
                heavy = live[wk[live] > self.heavy_weight]
                light = np.setdiff1d(live, heavy)
                if len(light) and fc.M > 1:
                    res[light] = self._lse_residual(loc, Lc, Om, fc, light)
                if len(heavy):
                    Pm = np.linalg.inv(fc.mm_S[heavy])
                    ldm = np.linalg.slogdet(fc.mm_S[heavy])[1][:, None]
                    res[heavy] = self._gh_residual(
                        loc[heavy], Lc[heavy], Om[heavy], fc.take(heavy), Pm, ldm
                    )
                res = res.reshape(n, A, R)
                terms[(k, which)] = (Mq, gq, hq, res, res.mean(1) @ self._tw)
        return terms

    def _lse_residual(self, loc, Lc, Om, fc, rows):
        """Variational lower bound minus the Gaussian's value on the grid (len(rows), A*R)."""
        r_ = rows
        Pm = np.linalg.inv(fc.mm_S[r_])
        ldm = np.linalg.slogdet(fc.mm_S[r_])[1][:, None]
        Fm = expected_logn_on_grid(
            loc[r_], Lc[r_], Om[r_], fc.mu[r_], fc.P[r_], fc.ld[r_], self._feats
        )
        lse_ = Fm + fc.logw[r_][..., None]
        mx = lse_.max(1, keepdims=True)
        Fa = (mx + np.log(np.exp(lse_ - mx).sum(1, keepdims=True)))[:, 0]
        Fg = expected_logn_on_grid(
            loc[r_], Lc[r_], Om[r_], fc.mm_mu[r_][:, None], Pm[:, None], ldm, self._feats
        )[:, 0]
        return Fa - Fg

    def _gh_residual(self, loc, Lc, Om, fc, Pm, ldm):
        """Res(m) = E_q log p_M - E_q log g at a coarse (angle, radius) subset of the polar points
        (Gauss-Hermite on q of log p_M), then interpolated. E_q log g is the exact quadratic,
        so Res is the smooth non-Gaussian remainder."""
        n, A, R = len(loc), len(self.dirs), len(self.radii)
        sa, sr = self.stride
        ia = np.arange(0, A, sa)
        ir = np.unique(np.append(np.arange(0, R, sr), R - 1))
        z = (self.dirs[ia][:, None, :] * self.radii[ir][None, :, None]).reshape(-1, 2)
        feats = _features(z)
        Fg = expected_logn_on_grid(loc, Lc, Om, fc.mm_mu[:, None], Pm[:, None], ldm, feats)[
            :, 0
        ]  # (n, Ps)
        nodes, gw = gh_nodes_2d(self.gh)
        LO = np.linalg.cholesky(Om + 1e-15 * _EYE2)
        Ys = np.einsum("nij,gj->ngi", LO, nodes)
        Ps = len(z)
        gh = np.empty((n, Ps))
        f3 = np.stack([z[:, 0] ** 2, 2.0 * z[:, 0] * z[:, 1], z[:, 1] ** 2], 1)  # (Ps, 3)
        step = max(1, int(3e6 // (Ps * len(nodes) * fc.M)))
        for s0 in range(0, n, step):
            sl = slice(s0, s0 + step)
            P_, mu_, Lc_, loc_ = fc.P[sl], fc.mu[sl], Lc[sl], loc[sl]
            # (y - mu_j)' P_j (y - mu_j), y = loc + Lc z + Ys_g, is quadratic in z
            A_ = np.einsum("nki,njkl,nlm->njim", Lc_, P_, Lc_)  # Lc' P Lc
            quad = np.einsum(
                "njf,pf->njp", np.stack([A_[..., 0, 0], A_[..., 0, 1], A_[..., 1, 1]], -1), f3
            )
            d0 = loc_[:, None, None] + Ys[sl][:, :, None] - mu_[:, None]  # (b, G, M, 2)
            Pd0 = np.einsum("njkl,ngjl->ngjk", P_, d0)
            lin = 2.0 * np.einsum("nki,ngjk->ngji", Lc_, Pd0)  # (b, G, M, 2)
            const = np.einsum("ngji,ngji->ngj", d0, Pd0)
            qq = (
                quad[:, None] + np.einsum("ngjf,pf->ngjp", lin, z) + const[..., None]
            )  # (b, G, M, Ps)
            lc = (
                fc.logw[sl][:, None, :, None]
                - 0.5 * qq
                - 0.5 * fc.ld[sl][:, None, :, None]
                - LOG2PI
            )
            mx = lc.max(2, keepdims=True)
            lp = (mx + np.log(np.exp(lc - mx).sum(2, keepdims=True)))[:, :, 0]  # (b, G, Ps)
            gh[sl] = np.einsum("ngp,g->np", lp, gw)
        g = (gh - Fg).reshape(n, len(ia), len(ir))
        g = interp_polar(g, sa, ir, A, self.radii, self.interp)
        return g.reshape(n, A * R)

    def _tails(self, idx, mu_m, S_m, eps, which, need_lt=True):
        if self._terms is None:
            return super()._tails(idx, mu_m, S_m, eps, which, need_lt)
        p_lt = np.zeros(len(idx))
        p_gt = np.zeros(len(idx))
        mean_I = np.zeros(len(idx))
        r = self.radii
        eps = np.broadcast_to(np.asarray(eps, float), (len(idx),))
        thr = eps[:, None, None]
        for k in range(self.K):
            _, _, wk = self._set(k, which, idx)
            M, g, h, res, mres = self._terms[(k, which)]
            h = h - np.log(np.clip(self.w[idx, k], 1e-300, None))
            A = np.einsum("ai,nij,aj->na", self.dirs, M, self.dirs)
            B = np.einsum("ai,ni->na", self.dirs, g)
            Q = A[..., None] * (r * r) + B[..., None] * r + h[:, None, None]
            G = self.G[(k, which)][idx]
            I = Q + res - G
            gt, lt = _polar_tails(I, thr, r, A, B)
            if need_lt and eps.any():
                _, lt = _polar_tails(I, 0.0, r, A, B)
            p_lt += wk * lt
            p_gt += wk * gt
            mean_I += wk * (np.einsum("nii->n", M) + h - self.mean_G[(k, which)][idx] + mres)
        return p_lt, p_gt, mean_I

    def test_full_mixture(self, idx, fc, eps=0.0, margins=(), base=True):
        """test_full for a mixture forecast fc (MixtureForecast aligned to rows idx)."""
        self._terms = self.model_terms(idx, fc)
        try:
            return self.test_full(idx, None, None, eps, margins, base)
        finally:
            self._terms = None
