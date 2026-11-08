"""Command-line flags shared by the script_*.py files, and their conversion to the
arguments the libraries take.

  add_set_arguments / set_choice          SET, --sessions, --filename, --config
  add_postprocess_arguments / postprocess_settings   -> pipeline.settings.Settings
  add_map_arguments / map_options          -> mapplot.MapOptions

Flag defaults come from the dataclasses, so each default is defined once.
"""

import argparse
import dataclasses
from collections import namedtuple
from types import SimpleNamespace

import mapplot
import session_sets
from pipeline import settings as ps
from pipeline.settings import Settings

SetChoice = namedtuple("SetChoice", "sset rows dirs filename description config")


def add_set_arguments(ap, description=True):
    """SET, --sessions, --filename, --config, and --description when the figures
    have a title. Read back with set_choice()."""
    ap.add_argument("set", metavar="SET",
                    help="a name in session_sets.SETS (e.g. SKJAERHALDEN), or a "
                         "measurement directory - then the sessions are found by "
                         "scanning it")
    ap.add_argument("--sessions", nargs="+", metavar="SESSION",
                    help="only these of the set's sessions, in this order (folder "
                         "names in the set's directory). Sessions in the set keep "
                         "its labels and colours")
    ap.add_argument("--filename",
                    help="prefix on the figure files (default: the set's)")
    if description:
        ap.add_argument("--description",
                        help="goes first in the figure titles (default: the set's)")
    ap.add_argument("--config", default="",
                    help="name of this run, e.g. 0.2-0.3: everything is written "
                         "under <measurement>/<config>/ (each session in "
                         "<config>/<stamp>/) and the figures get _<config> in the "
                         "name - see configs.config_dir. Default: in place")


def set_choice(args):
    """Flags from add_set_arguments() -> SetChoice. dirs are the absolute session
    directories, checked to exist. config is passed through unvalidated."""
    sset = session_sets.get_set(args.set)
    rows = session_sets.select(sset, args.sessions)
    return SetChoice(sset=sset, rows=rows,
                     dirs=session_sets.check(sset, [row[0] for row in rows]),
                     filename=args.filename or sset.filename,
                     description=getattr(args, "description", None) or sset.description,
                     config=args.config)


def _from_args(cls, args):
    """A dataclass with every field taken from the argparse namespace."""
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


def add_postprocess_arguments(ap):
    """Every Settings field as a flag (not the session path); the flag names are the
    field names."""
    d = SimpleNamespace(**Settings().flat())
    ap.add_argument("--decimate-hz", type=float, default=d.decimate_hz,
                    help=f"The rate the IMU series is decimated to before Welch (default {ps.DECIMATE_HZ:g}). "
                         "1000/rate must give a whole number of ms - the bucket indices are "
                         "integer arithmetic. Segment length and band limit follow along.")
    ap.add_argument("--seglen", type=int, default=d.seglen,
                    help="Welch segment length (power of 2). Omitted => derived from "
                         "--decimate-hz: 1024 @ <=10 Hz, 2048 @ <26, 4096 @ <=52, "
                         "otherwise 8192 - see default_seglen()")
    ap.add_argument("--capture-min", type=float, default=d.capture_min, metavar="MIN",
                    help="analyse only the first MIN minutes of the capture, as if it had "
                         "stopped there (rows and GPS after it are dropped)")
    ap.add_argument("--skip-start", type=float, default=d.skip_start, metavar="SEC",
                    help="Ignore the first SEC seconds (e.g. 300 = 5 min of "
                         "deployment/handling): cut before Welch, so the first segment "
                         "starts there, as the firmware's warm-up. Omitted => "
                         "filter_warm_up_ms in cfg.csv (0 if absent). The AHRS filters "
                         "still run over the whole session, so they are converged")
    ap.add_argument("--overlap-div", type=int, default=d.overlap_div, help="Step = seglen/div (4 => 75%%)")
    ap.add_argument("--window", choices=["hann", "hamming"], default=d.window)
    ap.add_argument("--fmax", type=float, default=d.fmax,
                    help=f"Upper band limit (Hz). Omitted => {ps.WAVE_FMAX:g}, clamped to "
                         "Nyquist when --decimate-hz is lower than twice that")
    ap.add_argument("--param-fmax", type=float, default=d.param_fmax,
                    help="upper limit for Hs/Tz/Tc/Tp alone; the spectra still run to "
                         "--fmax. Default $WAVE_PARAM_FMAX, else none")
    ap.add_argument("--taper-f1", type=float, default=d.taper_f1, help="Low-frequency taper start (T=0 below)")
    ap.add_argument("--taper-f2", type=float, default=d.taper_f2, help="Low-frequency taper end (T=1 above)")
    ap.add_argument("--beta", type=float, default=d.beta, help="Madgwick beta")
    ap.add_argument("--cutoff", choices=["taper", "auto", "gps"], default=d.cutoff,
                    help="Low-frequency cutoff: 'taper' = fixed f1/f2, "
                         "'auto' = the first spectral minimum (find_peaks), "
                         "'gps' = where the IMU stops running away from GPS vUp")
    ap.add_argument("--gps-rel", type=float, default=d.gps_rel,
                    help="gps: threshold = this x the ratio curve's own minimum")
    ap.add_argument("--gps-smooth-hz", type=float, default=d.gps_smooth_hz,
                    help="gps: running median over this band (Hz)")
    ap.add_argument("--gps-span-hz", type=float, default=d.gps_span_hz,
                    help="gps: the ratio must stay below the threshold over this span (Hz)")
    ap.add_argument("--gps-ramp", type=float, default=d.gps_ramp,
                    help="gps: half-cosine ramp f_c -> this x f_c "
                         "(1.0 = hard cut)")
    ap.add_argument("--gps-max-f1", type=float, default=d.gps_max_f1,
                    help="gps: maximum for the ramp's lower edge (Hz). The upper "
                         "edge follows from the ramp. 0 = off")
    ap.add_argument("--gps-fmax", type=float, default=d.gps_fmax,
                    help="gps: upper limit for the SEARCH for f_c (Hz). Separate from "
                         "--fmax; a hit above the wave band is no "
                         "low-frequency cutoff")
    ap.add_argument("--gps-own-prominence", type=float, default=d.gps_own_prominence,
                    help="Prominence for the GPS's OWN cutoff (the validity limit of "
                         "the reference, not used as a taper)")
    ap.add_argument("--compare", dest="compare", action="store_true", default=d.compare,
                    help="Print all three cutoffs (untapered / fixed / GPS) "
                         "for all the methods")
    ap.add_argument("--no-compare", dest="compare", action="store_false",
                    help="Only the table for the chosen --cutoff")
    ap.add_argument("--cut-fmin", type=float, default=d.cut_fmin,
                    help="auto: lower search limit for f_c (Hz)")
    ap.add_argument("--cut-distance", type=int, default=d.cut_distance,
                    help="auto: minimum distance between minima [frequency bins]")
    ap.add_argument("--cut-prominence", type=float, default=d.cut_prominence,
                    help="auto: prominence requirement on the minimum")
    ap.add_argument("--brake-reject", type=float, default=d.brake_reject,
                    help="Reject a Welch segment if the braking fraction > this (0 = off)")
    ap.add_argument("--ovf-reject", type=float, default=d.ovf_reject,
                    help="Reject a Welch segment if the fifo_ovf fraction > this "
                         "(0 = any occurrence rejects)")
    ap.add_argument("--no-ovf-reject", dest="ovf_reject", action="store_const",
                    const=None, help="Do not reject on fifo_ovf")
    ap.add_argument("--vacc-source", choices=["auto", "raw", "csv"],
                    default=d.vacc_source,
                    help="where the vertical accel is taken from: raw = the vacc_fir "
                         "column, where the AHRS has already run on the raw stream "
                         "(on board, or offline from raw.bin). csv = replay of the "
                         "filter at the row rate, as before. auto (default) = raw "
                         "when the column exists, and EVERY alternative filter is "
                         "then run on the raw stream via raw_to_csv (cached as "
                         "<stamp>_imu_raw_<filter>.csv next to imu.csv). "
                         "csv does neither")
    ap.add_argument("--raw", default=d.raw, metavar="PATH",
                    help="<stamp>_raw.bin to run the alternative filters on. Without "
                         "it, the search goes next to imu.csv and one level up. "
                         "'off' disables the lookup")
    ap.add_argument("--regen-raw", action="store_true",
                    help="rebuild <stamp>_imu_raw_<filter>.csv even if they "
                         "exist (after a change in raw_to_csv/the filters)")
    ap.add_argument("--detrend", choices=["none", "mean", "linear"], default=d.detrend,
                    help="Detrending per Welch segment. 'none' = firmware faithful; "
                         f"default '{ps.DETREND}'")
    ap.add_argument("--decimate", choices=["mean", "fir"], default=d.decimate,
                    help="How the logged rows become the decimated series: "
                         "'bucket' = the mean over the bucket (firmware faithful), 'fir' = "
                         f"FIR low-pass + decimation as the sfy buoy. Default '{ps.DECIMATE_MODE}'")
    ap.add_argument("--fir-ntap", type=int, default=d.fir_ntap,
                    help="fir: number of taps (odd). Omitted = the session's "
                         f"cfg.csv (fir_ntap), or {ps.FIR_NTAP} without it")
    ap.add_argument("--fir-cutoff", type=float, default=d.fir_cutoff,
                    help="fir: cutoff in Hz. Omitted = half the analysis rate, as fir.rs. "
                         "Lower gives real margin against folding")
    ap.add_argument("--fir-causal", dest="fir_compensate", action="store_false",
                    default=d.fir_compensate,
                    help="fir: do not compensate the group delay - firmware faithful, "
                         "but the series then lags the GPS by (ntap-1)/2 samples")
    ap.add_argument("--gap-reject", type=float, default=d.gap_reject,
                    help="Reject a Welch segment if more than this fraction is filled "
                         "gap in the decimated series (0 = any occurrence rejects)")
    ap.add_argument("--no-gap-reject", dest="gap_reject", action="store_const",
                    const=None, help="Do not reject on gaps (but fill them anyway)")
    ap.add_argument("--plot", dest="plot", action="store_true", default=d.plot,
                    help="Plot the spectrum (default: PLOT in pipeline/settings.py)")
    ap.add_argument("--no-plot", dest="plot", action="store_false", help="Do not plot")
    ap.add_argument("--config", default=d.config,
                    help="name of the run (e.g. 0.2-0.3): appended as _<config> to "
                         "<stamp>_spec_acc.png and _spec_eta.png. WHERE the files go is decided by the "
                         "directory analysed - see session_source.ensure_imu")


def postprocess_settings(args):
    return Settings.from_flat(**{k: getattr(args, k) for k in Settings().flat()})


def add_map_arguments(ap):
    """Every mapplot.MapOptions field as a flag (not the session paths)."""
    d = mapplot.MapOptions()
    ap.add_argument("--nr", help="measurement number per session, comma separated "
                                 "(e.g. --nr 3,7,8). Default 1,2,3,...")
    ap.add_argument("--badge-colors", metavar="COLOURS",
                    help="number-badge colour per session, comma separated, any "
                         "matplotlib colour (e.g. --badge-colors '#b03000,,tab:blue'); "
                         "an empty entry keeps the default red")
    ap.add_argument("--badge-legend", metavar="TEXT",
                    help="what the badge colours mean, under the overview map after "
                         "'number = measurement (see table): ' (e.g. 'red measured, "
                         "blue derived (decimated)')")
    ap.add_argument("-o", "--out",
                    help="output file (default <session dir>/<stamp>_map.png). "
                         "Only valid with a single session.")
    ap.add_argument("--format", choices=("png", "pdf"), default=d.format,
                    help="output format; derived from --out when that is given")
    ap.add_argument("--dpi", type=int, default=d.dpi)
    ap.add_argument("--width", type=float, default=d.width,
                    help="figure width in inches (default 7.5 = A4 with margins)")
    ap.add_argument("--taper-f1", type=float, default=d.taper_f1,
                    help="low-frequency taper: T=0 below this [Hz] (default 0.1). "
                         "Passed to postprocess.py and written in the figure.")
    ap.add_argument("--taper-f2", type=float, default=d.taper_f2,
                    help="low-frequency taper: T=1 above this [Hz] (default 0.2)")
    ap.add_argument("--seglen", type=int, default=d.seglen,
                    help="passed to postprocess.py: the Welch segment length, a power "
                         "of 2. Omitted => postprocess derives it from the analysis "
                         "rate (default_seglen). It appears in the wave table, because "
                         "Hs/Tz/Tc are not comparable across the choice")
    ap.add_argument("--fir-ntap", type=int, default=d.fir_ntap,
                    help="passed to postprocess.py: taps (odd) in the FIR that "
                         "decimates the rows to the analysis rate. Omitted => "
                         "the session's cfg.csv (fir_ntap), as the firmware ran. Only "
                         "the row->analysis stage; the raw->row stage of a "
                         "reconstruction takes fir_ntap from cfg.csv itself")
    ap.add_argument("--raw", default=d.raw, metavar="PATH",
                    help="passed to postprocess.py: which <stamp>_raw.bin the "
                         "alternative filters should run on. 'off' keeps the "
                         "analysis to what imu.csv itself contains")
    ap.add_argument("--vacc-source", choices=("auto", "raw", "csv"), default=d.vacc_source,
                    help="passed to postprocess.py: raw = the vertical accel from "
                         "vacc_fir (AHRS on the raw stream), csv = replay at the row "
                         "rate, auto = raw when the column exists. It appears in the "
                         "wave table, because the numbers are not comparable across "
                         "the choice")
    ap.add_argument("--psd-fmin", type=float, default=d.psd_fmin,
                    help="lower frequency in the spectrum plot [Hz] "
                         "(default: the taper's f1 - below it everything is zeroed)")
    ap.add_argument("--psd-fmax", type=float, default=d.psd_fmax,
                    help="upper frequency in the spectrum plot [Hz] (default 2.0)")
    ap.add_argument("--skip-postprocess", action="store_true",
                    help="do not rerun postprocess.py - use _ana_python.csv as it is "
                         "(fast, but the numbers may be stale)")
    ap.add_argument("--plot-only", action="store_true",
                    help="compute nothing: --skip-postprocess, and _spec_acc/_spec_eta.png "
                         "redrawn from data/")
    ap.add_argument("--tick-sec", type=float, default=d.tick_sec,
                    help="points along the track every N seconds (default 30)")
    ap.add_argument("--label-sec", type=float, default=d.label_sec,
                    help="labelled points with a clock every N seconds (default 300)")
    ap.add_argument("--zoom", type=int, default=d.zoom, help="override the tile zoom")
    ap.add_argument("--no-map", action="store_true",
                    help="do not download tiles - draw the track on a grid")
    ap.add_argument("--no-inset", action="store_true",
                    help="drop the overview inset inside each single map")
    ap.add_argument("--psd-table", metavar="STAMPS", default=d.psd_table,
                    help="also write <stamp>_psd_table.png (the spectrum of the "
                         "single map on its own, titled with the measurement name) "
                         "for these sessions, comma separated, 'all' or '' for none "
                         f"(default {d.psd_table})")
    ap.add_argument("--no-kvs", action="store_true",
                    help="no kystvarslingssenteret.no Hs/Tp row in the wave "
                         "table (default: on, fetched once per position and cached)")
    ap.add_argument("--inset-km", type=float, default=d.inset_km,
                    help="width in km of the NEAR inset in the single maps. The same "
                         "value for every measurement, so they can be compared "
                         "(default 2)")
    ap.add_argument("--inset-far-km", type=float, default=d.inset_far_km,
                    help="width in km of the FAR inset below the near one in the "
                         "single maps (default 50)")
    ap.add_argument("--overview-inset-km", default=d.overview_inset_km,
                    help="comma-separated widths in km of the insets in the overview "
                         "map. Each inset shows where the extent below it lies "
                         "(default 10,100)")
    ap.add_argument("--prefix", default=d.prefix,
                    help="prefix on the overview filenames, e.g. "
                         "Skjærhalden_20260731 -> "
                         "Skjærhalden_20260731_overview.png (plus _map.png and "
                         "_table.png). Without it the names stand without saying "
                         "which measurement series they are")
    ap.add_argument("--config", default=d.config,
                    help="name of the run (e.g. 0.2-0.3): appended as _<config> to "
                         "every figure (_map, _overview, _table, and "
                         "postprocess's _spec). WHERE they go is the session "
                         "directories given - plot.py --config points them at "
                         "<measurement>/<config>/<stamp>/")
    ap.add_argument("--overview-out",
                    help="output file for the whole overview (default "
                         "<common directory>/[<prefix>_]overview.png). The map "
                         "and table crops follow its stem: _map.png and _table.png")
    ap.add_argument("--overview-method", default=d.overview_method,
                    help="method for Hs/Tz/Tc in the overview table (default madgwick)")
    ap.add_argument("--no-overview", action="store_true",
                    help="do not make an overview map even with several sessions")
    ap.add_argument("--only-overview", action="store_true",
                    help="make ONLY the overview map, no single figures")
    ap.add_argument("--tile-url", default=d.tile_url)
    ap.add_argument("--tile-cache", default=d.tile_cache)
    ap.add_argument("--init-note", action="store_true",
                    help="write a <stamp>_notat.txt stub and exit")


def map_options(args):
    return _from_args(mapplot.MapOptions, args)


def map_options_from_argv(argv):
    """mapplot flags given as a list of strings -> MapOptions. Unknown flags stop."""
    ap = argparse.ArgumentParser(prog="mapplot options", allow_abbrev=False)
    add_map_arguments(ap)
    return map_options(ap.parse_args(argv))
