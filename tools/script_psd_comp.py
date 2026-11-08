#!/usr/bin/env python3
"""Untapered PSD comparison, Madgwick and Kalman against GPS: psd_comp.run() from the
command line.

    python3 tools/script_psd_comp.py SKJAERHALDEN
    python3 tools/script_psd_comp.py SKJAERHALDEN --report --config 0.2-0.3
    python3 tools/script_psd_comp.py SKJAERHALDEN --only-report --report \\
        --sessions 20260731_110303 20260731_122523
"""

import argparse

import _paths                                               # noqa: F401
import arguments                                            # noqa: E402
import configs                                              # noqa: E402
import psd_comp                                             # noqa: E402
import spectra                                              # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Simple, UNTAPERED PSD comparison: Madgwick and Kalman against GPS.")
    arguments.add_set_arguments(ap, description=False)
    ap.add_argument("--decimate-hz", type=float, default=spectra.DECIMATE_HZ,
                    help=f"analysis rate [Hz] (default {spectra.DECIMATE_HZ:g}, as "
                         f"plot_felles_psd)")
    ap.add_argument("--seglen", type=int, default=None,
                    help="Welch segment length (default: postprocess's for the rate)")
    ap.add_argument("--out-dir", default=None,
                    help="where the figures are written (default: the session's "
                         "directory, or its <config>/<stamp>/ with --config)")
    ap.add_argument("--lag-t", type=float, default=0.95,
                    help="time constant [s] in the reference curve in the ratio plot "
                         "(0 = leave out; default 0.95, fitted on the Skjærhalden "
                         "sessions)")
    ap.add_argument("--report", nargs="?", const="", default=None,
                    metavar="PATH",
                    help="summary figure over ALL the sessions. With a path it is "
                         "written there; without, it becomes <out-dir or the set's "
                         "directory>/<filename>_psd_rapport.png")
    ap.add_argument("--only-report", action="store_true",
                    help="skip the per-session figures and make only the summary")
    ap.add_argument("--plot-only", action="store_true",
                    help="draw from the existing data/ CSVs, compute nothing")
    args = ap.parse_args(argv)
    choice = arguments.set_choice(args)

    psd_comp.run(choice.dirs, config=configs.check_config(choice.config),
                 decimate_hz=args.decimate_hz, seglen=args.seglen,
                 out_dir=args.out_dir, lag_t=args.lag_t, report=args.report,
                 only_report=args.only_report, report_dir=choice.sset.directory,
                 filename=choice.filename, plot_only=args.plot_only)


if __name__ == "__main__":
    main()
