"""FIR lowpass + decimation, mirroring the sfy buoy's src/fir.rs.

The alternative to the firmware's bucket mean, which aliases heavily.
"""

import numpy as np

NTAP = 129                   # taps, as in fir.rs; must be odd
CUTOFF = None                # None => half the output rate (fs_out/2), as fir.rs
COMPENSATE_DELAY = True      # compensate the group delay (the firmware does not)


def firwin_lowpass(ntap, cutoff, fs):
    """Windowed sinc lowpass, as scipy.signal.firwin(ntap, cutoff, fs=fs):

      h[n] = sinc(2·(fc/fs)·(n − (N−1)/2)) · w_hamming[n],  normalised so Σh = 1

    cutoff = fs/2 gives the identity filter."""
    if ntap < 3:
        raise ValueError(f"FIR: ntap must be >= 3 (got {ntap})")
    if ntap % 2 == 0:
        # Odd length -> the group delay (N−1)/2 is a whole number of samples.
        raise ValueError(f"FIR: ntap must be odd for exact phase compensation (got {ntap})")
    if not (0.0 < cutoff <= 0.5 * fs):
        raise ValueError(f"FIR: cutoff must lie in (0, fs/2) = (0, {0.5 * fs:g}) Hz (got {cutoff})")
    n = np.arange(ntap, dtype=np.float64)
    h = np.sinc(2.0 * (cutoff / fs) * (n - 0.5 * (ntap - 1)))
    h *= 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (ntap - 1))   # Hamming (symmetric)
    return h / float(np.sum(h))


def fir_filter_centered(x, coeffs, compensate=True):
    """Run the FIR over x -> (y, first_valid, last_valid).

    With compensate, y[n] is centred on x[n] (shifted by the (N−1)/2 delay);
    without, it is causal as in the firmware. [first_valid, last_valid] is where
    no zero padding contributed."""
    c = np.asarray(coeffs, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    h = (len(c) - 1) // 2 if compensate else 0
    full = np.convolve(x, c)                 # len = len(x) + len(c) − 1
    y = full[h:h + len(x)]
    # Causal index n+h is clean only when the whole window lies inside x:
    #   n + h >= len(c) − 1  and  n + h <= len(x) − 1
    return y, max(0, len(c) - 1 - h), len(x) - 1 - h


def raw_uniform_grid(t_ms, raw_dt_ms=None):
    """Put the rows on a uniform grid with the missing rows flagged.

    dt is raw_dt_ms (cfg.csv window_ms) when it agrees with the median row
    spacing within 25 %, else the median.

    Returns (n_idx, nfull, dt_ms, gap_raw); n_idx is each row's place in nfull."""
    t = np.asarray(t_ms, dtype=np.float64)
    if len(t) < 2:
        raise ValueError("FIR: needs at least two rows")
    dt = float(np.median(np.diff(t)))
    if dt <= 0.0:
        raise ValueError(f"FIR: invalid raw dt ({dt} ms)")
    if raw_dt_ms:
        if abs(float(raw_dt_ms) - dt) <= 0.25 * dt:
            dt = float(raw_dt_ms)
        else:
            print(f"  WARNING: cfg.csv window_ms={float(raw_dt_ms):g} ms, but the "
                  f"rows show {dt:g} ms - using {dt:g} ms")
    n_idx = np.rint((t - t[0]) / dt).astype(np.int64)
    nfull = np.arange(n_idx[-1] + 1, dtype=np.int64)
    gap_raw = np.ones(len(nfull), dtype=np.float64)
    gap_raw[n_idx] = 0.0
    return n_idx, nfull, dt, gap_raw


def fir_decimate_series(t_ms, series, bucket_idx, bucket_ms, fs_out,
                        ntap=NTAP, cutoff=CUTOFF, unwrap=(),
                        compensate=COMPENSATE_DELAY, raw_dt_ms=None):
    """FIR lowpass + decimation of every series onto the bucket axis.

    t_ms       = win_start_ms per row (raw rate)
    series     = {name: raw-rate array}
    bucket_idx = the bucket indices to deliver
    bucket_ms  = the output period [ms]; fs_out the output rate [Hz]
    unwrap     = names of angle series to np.unwrap before filtering

    Each output value is taken at the middle of its bucket.

    Returns (out, gap, stats); gap is 1 where the filter support touches filled
    rows or the series edges."""
    n_idx, nfull, dt_ms, gap_raw = raw_uniform_grid(t_ms, raw_dt_ms)
    fs_raw = 1000.0 / dt_ms
    dec = int(round(bucket_ms / dt_ms))
    if dec < 1:
        raise ValueError(f"FIR: the raw rate ({fs_raw:.1f} Hz) is lower than the "
                         f"output ({fs_out:g} Hz) - lower --decimate-hz")
    if abs(bucket_ms - dec * dt_ms) > 1e-6:
        print(f"  WARNING: bucket {bucket_ms} ms does not divide the row period "
              f"{dt_ms:g} ms - the decimation (D={dec}) slides against the time axis")
    fc = float(cutoff) if cutoff is not None else 0.5 * fs_out
    coeffs = firwin_lowpass(ntap, fc, fs_raw)
    half = (ntap - 1) // 2

    xf = nfull.astype(np.float64)
    xb = n_idx.astype(np.float64)
    out = {}
    for name, v in series.items():
        v = np.asarray(v, dtype=np.float64)
        if name in unwrap:
            v = np.unwrap(v)
        # Interpolate onto the uniform grid (identity when no rows are missing).
        vu = v if len(nfull) == len(v) else np.interp(xf, xb, v)
        out[name], first_ok, last_ok = fir_filter_centered(vu, coeffs, compensate)

    # The bucket centres in raw indices: bucket b starts at time b·bucket_ms.
    b = np.asarray(bucket_idx, dtype=np.float64)
    m = np.rint((b * bucket_ms - float(t_ms[0])) / dt_ms).astype(np.int64) + dec // 2
    inside = (m >= first_ok) & (m <= last_ok)
    mc = np.clip(m, 0, len(nfull) - 1)

    # A gap smears across the filter support: dilate the flag to match.
    gap_dil = (np.convolve(gap_raw, np.ones(ntap), mode="same") > 0.0).astype(np.float64)
    gap = np.maximum(gap_dil[mc], (~inside).astype(np.float64))

    out = {k: v[mc] for k, v in out.items()}
    stats = dict(ntap=ntap, cutoff=fc, fs_raw=fs_raw, dec=dec, delay_s=half / fs_raw,
                 compensate=bool(compensate), n_raw=len(nfull),
                 n_raw_filled=int(np.count_nonzero(gap_raw)),
                 n_edge=int(np.count_nonzero(~inside)),
                 n_flagged=int(np.count_nonzero(gap > 0.0)), n_out=len(b))
    return out, gap, stats
