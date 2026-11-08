"""Offline analysis of one ORB logging session: the pipeline steps in
analysis/pipeline/ chained together (see pipeline/__init__.py for the step map).

  AHRS -> vertical linear accel -> decimation (default 10 Hz)
  -> Welch PSD (Hann/Hamming, 75 % overlap, Σw² normalisation)
  -> elevation = acc/ω⁴ with a low-frequency taper (T²)
  -> m0/m2/m4 -> Hs = 4√m0, Tz = √(m0/m2), Tc = √(m2/m4), Tp at the largest bin

Mirrors the firmware's DEBUG_POSTPROCESS chain (analysis.cpp), except for optional
FIR decimation (as the sfy buoy), linear detrending per segment, gap filling with
segment rejection, the extra methods and GPS vUp as an independent reference.

Output: <stamp>_spec_python.csv and <stamp>_ana_python.csv beside the imu.csv,
optionally <stamp>_spec_acc.png and _spec_eta.png, and a console report.

    st = pipeline.settings.Settings.from_flat(taper_f1=0.2, taper_f2=0.3)
    r = process(session_dir, st)          # normalise + analyse + write + report

Run with script_postprocess.py.
"""

import copy
import sys
from dataclasses import dataclass

import _paths                         # noqa: F401
from pipeline.ahrs import vertical_accel_series
from pipeline.cutoff import choose_cutoff, gps_reference
from pipeline.decimation import decimate
from pipeline.elevation import to_elevation
from pipeline.output import spectrum_bins, write_outputs
from pipeline.raw_stream import raw_series
from pipeline.report import print_report, print_sources
from pipeline.sea_state import sea_state
from pipeline.session_input import read_session, truncate_session
from pipeline.settings import FIR_NTAP, normalise
from pipeline.spectral import welch_spectra


@dataclass
class Result:
    """Every step's output for one session."""
    session: object              # Session          steps 1-2
    raw: object                  # RawSeries        step 3
    vacc: object                 # VerticalAccel    step 4
    series: object               # AnalysisSeries   step 5
    spectra: object              # Spectra          steps 6-8
    elev: object                 # ElevationSpectra step 9
    gps_ref: object              # GpsReference     step 10 (None without GPS)
    cut: object                  # LowFreqCut       step 10
    sea: object                  # SeaState         step 11
    bins: dict                   # the spec.csv table (step 12)


def analyse_session(path, st):
    """Steps 1-11 for the session at path (directory or *_imu.csv) with normalised
    settings st. Writes nothing (except cached raw-stream CSVs). -> Result."""
    session = read_session(path)
    if st.welch.capture_min:
        session = truncate_session(session, st.welch.capture_min)
    if st.decimation.fir_ntap is None:
        # The taps the firmware ran (kFirNtap), as the reconstruction uses.
        st = copy.deepcopy(st)
        st.decimation.fir_ntap = int(float(session.cfg.get("fir_ntap") or FIR_NTAP))
    if st.lowfreq.cutoff == "gps" and session.gps is None:
        sys.exit("--cutoff gps requires <stamp>_gps.csv with rel_ms + vUp")
    raw = raw_series(session, st.source)
    vacc = vertical_accel_series(session, raw, st.ahrs, st.source)
    series = decimate(vacc, session.gps, st.decimation, session.row_dt_ms)
    skip = st.welch.skip_start
    if skip is None:
        # The capture's own warm-up, as the firmware leaves it out of the Welch input.
        skip = (session.cfg_number("filter_warm_up_ms") or 0.0) / 1000.0
    spectra = welch_spectra(series, st.welch, skip)
    elev = to_elevation(spectra)
    fmax = st.sea_state.fmax
    ref = gps_reference(elev, st.lowfreq, fmax)
    cut = choose_cutoff(elev, ref, st.lowfreq, fmax)
    sea = sea_state(elev, cut, st.sea_state.param_fmax)
    r = Result(session=session, raw=raw, vacc=vacc, series=series, spectra=spectra,
               elev=elev, gps_ref=ref, cut=cut, sea=sea,
               bins=spectrum_bins(elev, cut.tapers, fmax))
    print_sources(r)
    return r


def process(path, st):
    """The whole chain for one session: normalise, analyse, write the files and print
    the report. -> Result."""
    normalise(st)
    r = analyse_session(path, st)
    spec_path, ana_path = write_outputs(r, st)
    print_report(r, st, spec_path, ana_path)
    return r
