"""The spectrum chain shared by felles_psd, hs_cutoff and psd_comp: one session ->
its PSDs (cached), with a taper applied on top.

  untapered()  postprocess.analyse_session() -> ElevationSpectra (pipeline steps
               1-9). Independent of the taper, so cached in memory and in
               ~/.cache/orb_psd.
  tapered()    untapered() + the fixed taper and sea state (steps 10-11)
               -> (path, settings, elev, cut, sea).
  psd_frame()  the same as a DataFrame, with Hs/Tz/Tc per method in
               df.attrs["wp"].

Also the shared x axis (set_hz_axis) and save_figure.

    df = spectra.psd_frame(sess_dir, spectra.DECIMATE_HZ, (0.03, 0.05), config)
"""

import hashlib
import os
import pickle
from glob import glob

import matplotlib.ticker as mticker
import pandas as pd

import _paths
import postprocess
from freq_axis import log_hz_axis
from pipeline.ahrs import METHODS
from pipeline.cutoff import choose_cutoff
from pipeline.output import spectrum_bins
from pipeline.raw_stream import find_raw_log
from pipeline.sea_state import sea_state
from pipeline.settings import Settings, normalise
from session_source import ensure_imu

# The rate the series is decimated to before Welch [Hz].
DECIMATE_HZ = 10

# Method name -> column in spectrum_bins() output.
ETA_COL = {m.name: f"eta_{m.key}" for m in METHODS}
ETA_COL["gps"] = "eta_gps"          # GPS is not in METHODS
ACC_COL = {m.name: f"acc_{m.key}" for m in METHODS}   # no GPS acceleration

# Taper cutoff for the untapered (acc) frame: below the first bin, so no bins are
# dropped or weighted and the acc figure is the same for every taper.
ACC_TAPER_FC = 1e-6

# The PSD cache: in-process dict + pickles on disk, keyed by _cache_key().
CACHE_DIR = os.path.expanduser("~/.cache/orb_psd")
_PSD_MEMO = {}


def _source_signature():
    """(path, size, mtime) for every .py file under tools/, so any code change
    invalidates the cache."""
    out = []
    for path in sorted(glob(os.path.join(_paths.TOOLS, "**", "*.py"), recursive=True)):
        st = os.stat(path)
        out.append(f"{path}:{st.st_size}:{st.st_mtime_ns}")
    return "\n".join(out)


# The input files the analysis reads (not its outputs).
INPUT_GLOBS = ["*_imu.csv", "*_cfg.csv", "*_ses.csv", "*_gps.csv",
                "*_imu_raw_*.csv", "*_raw.bin"]


def _cache_key(imu_path, st):
    """Cache key from the analysis rate, segment length, braking rejection, skipped
    start (None = cfg.csv's warm-up, which is among the input files), the input files
    (size + mtime) and the tools source."""
    directory = imu_path if os.path.isdir(imu_path) else os.path.dirname(imu_path)
    bits = ["v4", os.path.abspath(imu_path), f"decimate_hz={st.decimation.decimate_hz}",
            f"seglen={st.welch.seglen}", f"brake_reject={st.welch.brake_reject}",
            f"skip_start={st.welch.skip_start}"]
    files = set()
    for m in INPUT_GLOBS:
        files.update(glob(os.path.join(directory, m)))
        # One level up, as find_raw_log.
        files.update(glob(os.path.join(os.path.dirname(directory), m)))
    # raw.bin where the analysis finds it (also from a config work directory).
    for imu in glob(os.path.join(directory, "*_imu.csv")):
        raw = find_raw_log(directory, os.path.basename(imu)[:-len("_imu.csv")])
        if raw:
            files.add(raw)
    for name in sorted(files):
        st = os.stat(name)
        bits.append(f"{os.path.relpath(name, directory)}:{st.st_size}:{st.st_mtime_ns}")
    bits.append(_source_signature())
    return hashlib.sha256("\n".join(bits).encode()).hexdigest()[:32]


def untapered(path, st):
    """The ElevationSpectra for the session at path (a work directory from
    ensure_imu), with normalised settings st. From the cache when possible."""
    key = _cache_key(path, st)
    if key in _PSD_MEMO:
        print("  psd:   from memory", flush=True)
        return _PSD_MEMO[key]

    file = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.isfile(file):
        try:
            with open(file, "rb") as f:
                elev = pickle.load(f)
            _PSD_MEMO[key] = elev
            print(f"  psd:   from disk ({os.path.basename(file)})", flush=True)
            return elev
        except Exception as e:
            print(f"  psd:   could not read the cache ({e}) - recomputing")

    elev = postprocess.analyse_session(path, st).elev
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        # Temporary name + rename: never half a cache file.
        tmp = f"{file}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(elev, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, file)
    except OSError as e:
        print(f"  psd:   could not write the cache ({e}) - continuing without")
    _PSD_MEMO[key] = elev
    return elev


def tapered(sess_dir, decimate_hz, taper, config="", seglen=None):
    """One session with the fixed taper (f1, f2) -> (path, st, elev, cut, sea).

    path is the work directory, st the normalised Settings, elev the untapered
    ElevationSpectra, cut the LowFreqCut and sea the SeaState. seglen=None is the
    default for the rate."""
    f1, f2 = taper
    path = ensure_imu(sess_dir, config)
    st = Settings.from_flat(decimate_hz=float(decimate_hz), taper_f1=f1, taper_f2=f2,
                            plot=False, seglen=seglen or None)
    normalise(st)

    elev = untapered(path, st)
    cut = choose_cutoff(elev, None, st.lowfreq, st.sea_state.fmax)
    return path, st, elev, cut, sea_state(elev, cut, st.sea_state.param_fmax)


def psd_frame(sess_dir, decimate_hz, taper, config=""):
    """Elevation and acceleration spectrum for one session -> DataFrame with columns
    f, T, acc_<key>, eta_<key> (+ eta_gps). df.attrs["wp"] holds Hs/Tz/Tc per method,
    computed over the whole band up to fmax."""
    _, st, elev, cut, sea = tapered(sess_dir, decimate_hz, taper, config)
    df = pd.DataFrame(spectrum_bins(elev, cut.tapers, st.sea_state.fmax))
    df.attrs["wp"] = {m.name: sea.wp[m.key] for m in METHODS}
    df.attrs["wp"]["gps"] = sea.gps
    return df


def set_hz_axis(ax, xlim):
    """freq_axis.log_hz_axis with two decimals on every tick label."""
    log_hz_axis(ax, xlim=xlim)
    two_decimals = mticker.FuncFormatter(lambda v, _p: f"{v:.2f}" if v > 0 else "")
    for which in ("major", "minor"):
        getattr(ax.xaxis, f"set_{which}_formatter")(two_decimals)


def save_figure(fig, out_file):
    """Write the figure to PNG (300 dpi). Call before any plt.show()."""
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"  wrote {os.path.basename(out_file)}")
