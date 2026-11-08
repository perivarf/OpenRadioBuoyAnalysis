"""Analysis settings: one dataclass per pipeline step, collected in Settings.

  SourceSettings      steps 1-3  where the vertical accel comes from (raw log or replay)
  AhrsSettings        step 4     orientation filters
  DecimationSettings  step 5     post-AHRS decimation to the analysis rate
  WelchSettings       steps 6-8  detrend, segmentation, window, segment rejection
  LowFreqSettings     step 10    low-frequency taper / cutoff
  SeaStateSettings    step 11    band limit for the moments
  OutputSettings      step 12    files, figure and console report

Field names are the command-line flag names (cli/arguments.py maps them 1:1).
Defaults are the constants below, mirroring settings.h where noted.

    st = Settings(lowfreq=LowFreqSettings(taper_f1=0.2, taper_f2=0.3))
    normalise(st)        # validates and fills welch.seglen / sea_state.fmax
"""

import dataclasses
import os
import sys
from dataclasses import dataclass, field

import _paths                         # noqa: F401
import fir

# --- Step 4: AHRS -------------------------------------------------------------
MADGWICK_BETA = 0.2          # kMadgwickBeta -> 0.2 gives roughly the same as Kalman/SFLP

# --- Steps 1-3: source of the vertical accel ----------------------------------
#   "csv"   the filter replayed here at the row rate
#   "raw"   the vacc_fir column (AHRS on the raw stream, on board or from raw.bin)
#   "auto"  raw when the column exists, else csv
# With a raw log the alternative filters also run on the raw stream (raw_stream.py).
VACC_SOURCE = "auto"

# --- Step 5: decimation to the analysis rate ---------------------------------
DECIMATE_HZ = 10.0           # kWelchInputOdrHz; 1000/rate must be whole ms
# "mean" = bucket mean (firmware; aliases heavily), "fir" = FIR low-pass + keep
# every D-th sample, as the sfy buoy (fir.py).
DECIMATE_MODE = "fir"
# $FIR_STAGE2 (set by plot.sh) tries another stage 2 than the capture ran:
# "average" (bucket mean) or a tap count for the FIR. Empty = the capture's FIR.
_STAGE2 = os.environ.get("FIR_STAGE2", "").strip().lower()
if _STAGE2 in ("average", "mean"):
    DECIMATE_MODE = "mean"
elif _STAGE2 and not _STAGE2.isdigit():
    sys.exit(f"FIR_STAGE2='{_STAGE2}': expected empty, 'average' or a tap count")
FIR_NTAP = fir.NTAP          # 129 taps, odd - only when cfg.csv lacks fir_ntap
# Taps of stage 2: None = the capture's own (fir_ntap in cfg.csv).
FIR_S2_NTAP = int(_STAGE2) if _STAGE2.isdigit() else None
FIR_CUTOFF = fir.CUTOFF      # None = half the analysis rate, as fir.rs
FIR_COMPENSATE_DELAY = fir.COMPENSATE_DELAY  # compensate (NTAP−1)/2 samples
# Missing rows are filled by linear interpolation so the time axis is uniform.
GAP_MAX_FILL = 50            # gaps longer than this [buckets] are reported as long

# --- Steps 6-8: detrend, segmentation, Welch ---------------------------------
OVERLAP_DIV = 4              # kWelchStepDiv (4 => 75 % overlap)
WINDOW = "hann"              # kWelchWindow: "hann" | "hamming"
# "none" (firmware) | "mean" | "linear". A linear drift leaks as f⁻² into the low
# band, where ÷ω⁴ magnifies it.
DETREND = "linear"
# Reject a segment above this braking fraction (0 = off). $BRAKE_REJECT (set by
# plot.sh) overrides it for every tool in the run.
BRAKE_REJECT = float(os.environ.get("BRAKE_REJECT") or 0.05)
OVF_REJECT = 0.0             # ... any fifo_ovf (0.0); None = off
# Leave the first N s out of the Welch input (the AHRS start transient; NXP takes
# ~20 s to lock from a tilted start). None = the capture's own warm-up,
# filter_warm_up_ms in cfg.csv (wave_capture_warmup on board; 0 if absent).
# $SKIP_START (set by plot.sh) overrides it for every tool in the run.
SKIP_START = (float(os.environ["SKIP_START"]) if os.environ.get("SKIP_START", "").strip()
              else None)
GAP_REJECT = 0.01            # ... above this fraction of filled gap; None = off


def default_seglen(fs):
    """Segment length for the rate fs: keeps segments ~80-100 s long."""
    if fs <= 10.0:
        return 1024          # 102.4 s @ 10 Hz
    if fs < 26.0:
        return 2048          # 102.4 s @ 20 Hz
    if fs <= 52.0:
        return 4096          # 81.9 s @ 50 Hz
    return 8192              # 81.9 s @ 100 Hz


# --- Step 10: low-frequency taper / cutoff -----------------------------------
TAPER_F1 = 0.03              # kTaperF1 (T=0 below this)
TAPER_F2 = 0.05              # kTaperF2 (T=1 above this)
CUTOFF_MODE = "taper"        # "taper" (fixed f1/f2) | "auto" (find_peaks) | "gps"
# auto: f_c = the first minimum above CUT_FMIN in the elevation spectrum.
CUT_FMIN = 0.05              # lower search limit (Hz)
CUT_DISTANCE = 3             # minimum distance between minima [frequency bins]
CUT_PROMINENCE = 0.05        # prominence requirement on the minimum [m²/Hz]
# gps: f_c is the first frequency where the smoothed S_eta(IMU)/S_eta(GPS) stays
# below GPS_REL_FACTOR x its GPS_REF_QUANTILE percentile over GPS_SPAN_HZ; the
# taper is then a half-cosine ramp f_c -> GPS_RAMP·f_c.
GPS_RAMP = 1.5
GPS_MAX_F1 = 0.15            # upper limit for the ramp's lower edge [Hz]; 0 = off
GPS_SEARCH_FMAX = 0.5        # upper limit of the f_c search [Hz]
GPS_REL_FACTOR = 1.5         # threshold = this x the reference level
GPS_REF_QUANTILE = 10.0      # percentile of the smoothed curve used as the reference
GPS_SMOOTH_HZ = 0.02         # running median over this band before thresholding
GPS_SPAN_HZ = 0.02           # the ratio must stay below the threshold over this span
GPS_NOISE_BAND = (1.2, 2.0)  # band for the GPS's own white-noise floor (Hz)
GPS_OWN_PROMINENCE = 0.005   # prominence for the GPS's own cutoff (validity check)
GPS_MIN_SNR = 3.0            # minimum GPS SNR at f_c for the reference to hold

# --- Step 11: sea state parameters -------------------------------------------
WAVE_FMAX = 5.0              # kWaveFMax (upper band limit, Hz)
# Upper limit for the moments alone (Hs/Tz/Tc/Tp) - the spectra still run to fmax.
# $WAVE_PARAM_FMAX (set by plot.sh) sets it for every tool in the run; unset or
# empty = no limit of its own (up to fmax).
WAVE_PARAM_FMAX = float(os.environ.get("WAVE_PARAM_FMAX") or "inf")

# --- Step 12: output ---------------------------------------------------------
PLOT = True                  # write <stamp>_spec_acc.png / _spec_eta.png
COMPARE = True               # print Hs/Tz/Tc for every cutoff variant


@dataclass
class SourceSettings:
    """Steps 1-3: where the vertical accel per method comes from."""
    vacc_source: str = VACC_SOURCE        # "auto" | "raw" | "csv"
    raw: str = None                       # raw.bin path, None = search, "off" = none
    regen_raw: bool = False               # rebuild <stamp>_imu_raw_<method>.csv


@dataclass
class AhrsSettings:
    """Step 4: the orientation filters."""
    beta: float = MADGWICK_BETA           # Madgwick gain


@dataclass
class DecimationSettings:
    """Step 5: rows -> the analysis rate."""
    decimate_hz: float = DECIMATE_HZ      # 1000/rate must be whole ms
    decimate: str = DECIMATE_MODE         # "mean" | "fir"
    fir_ntap: int = FIR_S2_NTAP           # None = the session's cfg.csv (fir_ntap)
    fir_cutoff: float = FIR_CUTOFF
    fir_compensate: bool = FIR_COMPENSATE_DELAY

    @property
    def bucket_ms(self):
        """The analysis period [ms] (one bucket)."""
        return int(round(1000.0 / self.decimate_hz))

    @property
    def fs(self):
        """The analysis rate [Hz], from the whole-ms bucket."""
        return 1000.0 / self.bucket_ms


@dataclass
class WelchSettings:
    """Steps 6-8: detrend, segmentation, windowing and segment rejection."""
    seglen: int = None                    # power of 2; None -> default_seglen()
    overlap_div: int = OVERLAP_DIV        # step = seglen/div
    window: str = WINDOW                  # "hann" | "hamming"
    detrend: str = DETREND                # "none" | "mean" | "linear"
    skip_start: float = SKIP_START        # s cut from the start; None = cfg warm-up
    capture_min: float = None             # analyse only the first N minutes; None = all
    brake_reject: float = BRAKE_REJECT    # 0 = off
    ovf_reject: float = OVF_REJECT        # None = off
    gap_reject: float = GAP_REJECT        # None = off


@dataclass
class LowFreqSettings:
    """Step 10: the low-frequency cutoff."""
    cutoff: str = CUTOFF_MODE             # "taper" | "auto" | "gps"
    taper_f1: float = TAPER_F1            # T=0 below [Hz]
    taper_f2: float = TAPER_F2            # T=1 above [Hz]
    cut_fmin: float = CUT_FMIN
    cut_distance: int = CUT_DISTANCE
    cut_prominence: float = CUT_PROMINENCE
    gps_rel: float = GPS_REL_FACTOR
    gps_smooth_hz: float = GPS_SMOOTH_HZ
    gps_span_hz: float = GPS_SPAN_HZ
    gps_ramp: float = GPS_RAMP
    gps_max_f1: float = GPS_MAX_F1
    gps_fmax: float = GPS_SEARCH_FMAX
    gps_own_prominence: float = GPS_OWN_PROMINENCE


@dataclass
class SeaStateSettings:
    """Step 11: the moments."""
    fmax: float = None                    # upper band limit; None -> WAVE_FMAX/Nyquist
    param_fmax: float = WAVE_PARAM_FMAX   # moments up to min(fmax, this)


@dataclass
class OutputSettings:
    """Step 12: files and report."""
    plot: bool = PLOT                     # write <stamp>_spec_acc/_spec_eta.png
    compare: bool = COMPARE               # print every cutoff variant
    config: str = ""                      # suffix _<config> on the PNG


@dataclass
class Settings:
    """Every analysis choice for one session, grouped by pipeline step."""
    source: SourceSettings = field(default_factory=SourceSettings)
    ahrs: AhrsSettings = field(default_factory=AhrsSettings)
    decimation: DecimationSettings = field(default_factory=DecimationSettings)
    welch: WelchSettings = field(default_factory=WelchSettings)
    lowfreq: LowFreqSettings = field(default_factory=LowFreqSettings)
    sea_state: SeaStateSettings = field(default_factory=SeaStateSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    @classmethod
    def from_flat(cls, **values):
        """Settings from flat field names (the flag names), e.g.
        Settings.from_flat(taper_f1=0.2, seglen=2048). Unknown names raise."""
        groups = {}
        for g in dataclasses.fields(cls):
            names = {f.name for f in dataclasses.fields(g.default_factory)}
            groups[g.name] = g.default_factory(
                **{k: values.pop(k) for k in list(values) if k in names})
        if values:
            raise TypeError(f"unknown settings: {', '.join(values)}")
        return cls(**groups)

    def flat(self):
        """{field name: value} over every group (the inverse of from_flat)."""
        return {k: v for g in dataclasses.fields(self)
                for k, v in dataclasses.asdict(getattr(self, g.name)).items()}


def normalise(st):
    """Validate st and fill in st.welch.seglen / st.sea_state.fmax from the analysis
    rate. Call before analysing."""
    dec = st.decimation
    if dec.decimate_hz <= 0.0:
        sys.exit(f"--decimate-hz must be positive (got {dec.decimate_hz:g})")
    bucket = 1000.0 / dec.decimate_hz
    if abs(bucket - round(bucket)) > 1e-6:
        # The bucket must be a whole number of ms (t // bucket_ms).
        near = [1000.0 / m for m in (int(bucket), int(bucket) + 1) if m > 0]
        sys.exit(f"--decimate-hz {dec.decimate_hz:g} gives a bucket of {bucket:.3f} ms - "
                 f"it must be a whole number of ms. Nearest valid: "
                 + ", ".join(f"{h:.4g} Hz" for h in sorted(set(near))))
    fs = dec.fs

    if st.welch.seglen is None:
        st.welch.seglen = default_seglen(fs)
    elif st.welch.seglen & (st.welch.seglen - 1):
        sys.exit(f"--seglen must be a power of 2 (got {st.welch.seglen})")

    # Checked here rather than minutes later in fir.py.
    if dec.decimate == "fir" and dec.fir_ntap is not None and (
            dec.fir_ntap < 3 or dec.fir_ntap % 2 == 0):
        sys.exit(f"--fir-ntap must be an odd number >= 3 (got {dec.fir_ntap})")

    # fmax = min(WAVE_FMAX, Nyquist).
    if st.sea_state.fmax is None:
        st.sea_state.fmax = min(WAVE_FMAX, 0.5 * fs)
    if st.sea_state.fmax > 0.5 * fs:
        print(f"  WARNING: --fmax {st.sea_state.fmax:g} Hz is above Nyquist "
              f"({0.5 * fs:g} Hz) - the moments stop at Nyquist regardless")
    st.sea_state.param_fmax = min(st.sea_state.param_fmax, st.sea_state.fmax)

    if 0.5 * fs <= GPS_NOISE_BAND[0]:
        print(f"  WARNING: the GPS noise band {GPS_NOISE_BAND[0]:g}-{GPS_NOISE_BAND[1]:g} Hz "
              f"is above Nyquist ({0.5 * fs:g} Hz) - gps_vup_noise_psd will be 0")
