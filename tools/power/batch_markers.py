"""The current markers the batch firmware (drifter/src/power_batch.cpp) puts in front of each test.

    ID marker:            quiet 4 s, N x (busy 1 s, Stop2 1 s), quiet 3 s.  N = test ID.
                          The test starts when the marker ends.
    Capture-start marker: Stop2 2 s, busy 3 s, Stop2 2 s (only the wave tests, WAVE_IDS).
                          The capture loop itself starts when the marker ends.

detect() finds both in a current log; batch_analysis.py and batch_live.py use it.
"""

import numpy as np

NAMES = {
    1: "baseline",
    2: "baseline_gnss_stop",
    3: "busy_spin",
    4: "imu",
    5: "imu_gps",
    6: "imu_gps_sd_idle",
    7: "imu_gps_sd_log",
    8: "busy_spin_gnss_stop",
    9: "baseline_sd_idle",
    10: "imu_sd_idle",
}
# The order in which the firmware runs (kBatch1 / kBatch2 in power_batch.cpp)
# 3: all in one run, SD-card tests last (card inserted manually before 9).
# 1 and 2: older logs, without and with SD card
BATCHES = {3: [1, 2, 3, 8, 4, 5, 9, 10, 6, 7], 1: [4, 5, 8, 2, 3, 1], 2: [6, 7]}
# Tests run with the SD card inserted
CARD_IDS = {6, 7, 9, 10}
# Wave measurements with a capture-start marker; the others are quiet tests without capture
WAVE_IDS = {4, 5, 6, 7, 10}

# Timing in the markers, as in power_batch.cpp. Measured: the busy portion looks ~1.1-1.5 s,
# the Stop2 portion ~0.6-0.9 s, but the period is 2.0 s, so we allow wide limits on pulse width.
PULSE_S = 1.0       # ID pulse (busy) and the pause after it
QUIET_LEAD_S = 4.0  # quiet before the first ID pulse
QUIET_TAIL_S = 3.0  # quiet after the last ID pulse (in addition to the pause on PULSE_S)
CAP_GAP_S = 2.0     # Stop2 before and after the capture pulse
CAP_PULSE_S = 3.0   # capture-start pulse (busy)


def rolling_median(y, n):
    """Median over n samples ending at each index (the first n-1 are nan)."""
    out = np.full(len(y), np.nan)
    if len(y) >= n:
        win = np.lib.stride_tricks.sliding_window_view(y, n)
        out[n - 1:] = np.median(win, axis=1)
    return out


def find_edges(t, i_ma, side_s, delta_ma):
    """Edge times: median over side_s after minus median over side_s before.

    Returns (rising, falling) as lists of times, one per contiguous region where the
    difference is above +delta (rising) or below -delta (falling).
    """
    dt = np.median(np.diff(t))
    n = max(3, int(round(side_s / dt)))
    before = rolling_median(i_ma, n)                       # [k-n+1, k]
    after = np.roll(rolling_median(i_ma, n), -(n - 1))     # [k, k+n-1]
    after[len(after) - (n - 1):] = np.nan
    diff = after - np.concatenate(([np.nan], before[:-1]))  # etter k minus før k

    def regions(mask, score):
        out = []
        k = 0
        while k < len(mask):
            if mask[k]:
                j = k
                while j < len(mask) and mask[j]:
                    j += 1
                out.append(t[k + int(np.argmax(score[k:j]))])
                k = j
            else:
                k += 1
        return out

    d = np.nan_to_num(diff)
    return regions(d > delta_ma, d), regions(d < -delta_ma, -d)


def is_flat(y, tol):
    """At least 90 % of the samples are within tol of the median. Tolerates isolated spikes,
    e.g. the ~0.1 s MCU awake period between two Stop2 windows in a marker."""
    return len(y) > 0 and np.mean(np.abs(y - np.median(y)) < tol) >= 0.9


def level(t, i_ma, t0, t1):
    m = (t >= t0) & (t < t1)
    return np.median(i_ma[m]) if m.any() else np.nan


def find_pulses(t, i_ma, rising, falling, w_min, w_max, delta_ma):
    """Pulses: rising edge followed by a falling edge w_min..w_max later, with a plateau on both sides."""
    pulses = []
    for r in rising:
        # The FIRST falling edge after r: otherwise the rising edge of one ID pulse and the falling
        # edge of the next one can look like one long pulse (a false capture start).
        f = next((f for f in falling if f > r), None)
        if f is None or not (w_min <= f - r <= w_max):
            continue
        # The levels are taken slightly away from the edges: the instrument samples unevenly
        # (0.1-0.4 s), and an edge time can be off by one sample.
        top = level(t, i_ma, r + 0.15, f - 0.15)
        # The low part between ID pulses is only ~0.6 s, so the sides are short; it is enough that
        # the plateau sits above the lowest side (the grouping covers the rest).
        sides = [x for x in (level(t, i_ma, r - 0.5, r - 0.1), level(t, i_ma, f + 0.1, f + 0.5))
                 if np.isfinite(x)]
        if not np.isfinite(top) or not sides or top - min(sides) >= delta_ma:
            pulses.append((r, f))
    return pulses


def find_markers(t, i_ma, rising, falling, delta_ma):
    """ID markers: groups of 1 s pulses spaced ~2 s apart with quiet before and after."""
    id_pulses = find_pulses(t, i_ma, rising, falling, 0.5 * PULSE_S, 1.8 * PULSE_S, delta_ma)
    groups, cur = [], []
    for p in id_pulses:
        if cur and 1.6 * PULSE_S <= p[0] - cur[-1][0] <= 2.6 * PULSE_S:
            cur.append(p)
        else:
            if cur:
                groups.append(cur)
            cur = [p]
    if cur:
        groups.append(cur)

    edges = sorted(rising + falling)
    markers = []
    for g in groups:
        r0, f_last = g[0][0], g[-1][1]
        # Quiet = no edges, and the same level before the first pulse as after the last: this separates
        # a real marker from a random short spike in the middle of a measurement.
        lead = (r0 - (QUIET_LEAD_S - 0.5), r0 - 0.05)
        tail = (f_last + 0.05, f_last + PULSE_S + QUIET_TAIL_S - 0.5)
        no_edges = not any(lead[0] <= e < lead[1] or tail[0] < e <= tail[1] for e in edges)
        same_level = abs(level(t, i_ma, *lead) - level(t, i_ma, *tail)) < delta_ma / 2
        # Stop2 is flat (~0.05 mA spread). GPS noise can otherwise give groups of
        # "pulses" with quiet intervals between them and thus false markers.
        # The flatness is measured away from the edges: the edge time can be off by one sample,
        # and one sample from the pulse would otherwise make a flat level look uneven.
        ml = (t >= lead[0]) & (t < lead[1] - 0.4)
        mt = (t > tail[0] + 0.4) & (t <= tail[1])
        flat = is_flat(i_ma[ml], delta_ma / 4) and is_flat(i_ma[mt], delta_ma / 4)
        if no_edges and same_level and flat:
            markers.append({
                "id": len(g),
                "marker_start": r0 - QUIET_LEAD_S,
                "test_start": f_last + PULSE_S + QUIET_TAIL_S,
            })
    # A measurement lasts at least a couple of minutes, so a "marker" immediately followed by another
    # is a short spike before the real one, e.g. the GPS startup in quiet() before the ID marker.
    return [m for k, m in enumerate(markers)
            if k + 1 == len(markers) or markers[k + 1]["marker_start"] - m["test_start"] > 10.0]


def find_capture_starts(t, i_ma, rising, falling, delta_ma):
    """Candidates for capture start: {"t", "strict", "score"}.

    GPS noise during fix waiting can look like a 3 s pulse. The real one has Stop2 for 2 s on both
    sides: strict = no edges there and the same level. With GPS on, the noise can also create edges
    in the Stop2 portions, so the score (plateau height against the sides, penalized for spread) is used
    when no candidate is strict.
    """
    edges = sorted(rising + falling)
    out = []
    for r, f in find_pulses(t, i_ma, rising, falling, 0.8 * CAP_PULSE_S, 1.4 * CAP_PULSE_S, delta_ma):
        lead, tail = (r - CAP_GAP_S + 0.2, r - 0.1), (f + 0.1, f + CAP_GAP_S - 0.2)
        ml = (t >= lead[0]) & (t < lead[1])
        mt = (t >= tail[0]) & (t < tail[1])
        if not ml.any() or not mt.any():
            continue
        lo_l, lo_t = np.median(i_ma[ml]), np.median(i_ma[mt])
        top = level(t, i_ma, r + 0.15, f - 0.15)
        no_edges = not any(lead[0] <= e < lead[1] or tail[0] < e <= tail[1] for e in edges)
        # Stop2 is flat (~0.05 mA spread); GPS noise is not, even without edges.
        # Measured some distance away from the edges, as for the ID markers.
        fl = ml & (t < lead[1] - 0.4)
        ft = mt & (t > tail[0] + 0.4)
        flat = is_flat(i_ma[fl], delta_ma / 4) and is_flat(i_ma[ft], delta_ma / 4)
        strict = no_edges and flat and abs(lo_l - lo_t) < delta_ma / 2
        score = (top - max(lo_l, lo_t)) / (1.0 + np.std(i_ma[ml]) + np.std(i_ma[mt]))
        out.append({"t": f + CAP_GAP_S, "strict": strict, "score": float(score)})
    return out


def pick_capture(caps, t0, t1, full=False):
    """Capture start in (t0, t1): the strict candidate, otherwise the one with the best score.

    full=True returns the whole candidate ({"t", "strict", "score"}), otherwise only the time.
    """
    cand = [c for c in caps if t0 < c["t"] < t1]
    strict = [c for c in cand if c["strict"]]
    best = max(strict or cand, key=lambda c: c["score"], default=None)
    if full:
        return best
    return best["t"] if best else None


def capture_problem(cap, seg_end, capture_s, min_score):
    """Why a wave measurement cannot be used, or None if it looks valid."""
    if cap is None:
        return "ingen capture-start funnet"
    if not cap["strict"] and cap["score"] < min_score:
        return f"usikker capture-start (score {cap['score']:.2f} < {min_score:g}) - trolig GPS-støy"
    if seg_end - cap["t"] < capture_s - 2.0:
        return f"bare {seg_end - cap['t']:.0f} s fra capture-start til neste markør (capture er {capture_s:g} s)"
    return None


def rate_chunks(t, block_s=20.0, ratio=3.0):
    """Split the log where the sampling rate changes (e.g. from ~5 Hz to ~50 Hz in the middle of a log).

    Edge detection counts samples, so it must be run separately on each section with a uniform rate.
    Returns (start, end) indices.
    """
    edges = np.arange(t[0], t[-1] + block_s, block_s)
    idx = np.searchsorted(t, edges)
    chunks, cur_dt = [], None
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a < 3:
            continue
        dt = np.median(np.diff(t[a:b]))
        if cur_dt is not None and max(dt, cur_dt) / min(dt, cur_dt) < ratio:
            chunks[-1][1] = b
        else:
            chunks.append([a, b])
            cur_dt = dt
    return chunks or [[0, len(t)]]


def detect(t, i_ma, delta_ma=3.0, edge_side_s=0.3):
    """ID markers and capture starts in the log so far."""
    if len(t) < 2:
        return [], []
    rising, falling = [], []
    for a, b in rate_chunks(t):
        r, f = find_edges(t[a:b], i_ma[a:b], edge_side_s, delta_ma)
        rising += r
        falling += f
    markers = find_markers(t, i_ma, rising, falling, delta_ma)
    caps = find_capture_starts(t, i_ma, rising, falling, delta_ma)
    return markers, caps


def drop_repeated(markers):
    """En test som ventet på SD-kortbytte sender markøren sin to ganger; testen starter
    etter den siste, så den første (med ventetiden etter) tas ut. group_start er starten
    på den første, så segmentet til testen foran ikke tar med ventetiden."""
    out = []
    for m in markers:
        if out and out[-1]["id"] == m["id"]:
            m = dict(m, group_start=out.pop()["group_start"])
        else:
            m = dict(m, group_start=m["marker_start"])
        out.append(m)
    return out


def distinct_ids(ids):
    """The IDs in order with repeats collapsed (a marker sent twice around an SD-card swap)."""
    return [k for n, k in enumerate(ids) if n + 1 == len(ids) or ids[n + 1] != k]


def match_batch(ids):
    """The batch whose test order is ids, or None."""
    return next((b for b, seq in BATCHES.items() if ids == seq), None)
