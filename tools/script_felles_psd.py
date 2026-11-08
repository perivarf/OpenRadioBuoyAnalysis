#!/usr/bin/env python3
"""All sessions of a series in one common elevation and acceleration spectrum:
felles_psd.main()/sweep() from the command line.

    python3 tools/script_felles_psd.py SKJAERHALDEN             # acc + eta per taper
    python3 tools/script_felles_psd.py SKJAERHALDEN --figures acc --method kalman
    python3 tools/script_felles_psd.py SKJAERHALDEN --taper 0.03 0.05 --figures eta --show
    python3 tools/script_felles_psd.py Torrevieja --figures acc  # directory without a set
"""

import argparse

import _paths                                               # noqa: F401
import arguments                                            # noqa: E402
import felles_psd as fp                                     # noqa: E402
import configs                                              # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Common elevation and acceleration spectrum for several sessions.")
    arguments.add_set_arguments(ap)
    ap.add_argument("--method", default=fp.METHOD, choices=sorted(fp.ETA_COL),
                    help=f"which AHRS's spectrum is plotted (default {fp.METHOD})")
    ap.add_argument("--figures", default="sweep",
                    choices=("sweep", "both", "eta", "acc"),
                    help="sweep (default) = acc once + eta per --taper")
    ap.add_argument("--taper", nargs=2, type=float, action="append",
                    metavar=("F1", "F2"), dest="tapers",
                    help="taper band, repeatable. F1 == F2 gives a hard cut with no "
                         "ramp. Default: "
                         + ", ".join(f"{a:g}-{b:g}" for a, b in fp.TAPERS))
    ap.add_argument("--decimate-hz", type=float, default=fp.DECIMATE_HZ,
                    help=f"the rate the series is decimated to before Welch "
                         f"(default {fp.DECIMATE_HZ:g})")
    ap.add_argument("--f-lo", type=float, default=fp.F_LO,
                    help=f"plot window, lower [Hz] (default {fp.F_LO:g})")
    ap.add_argument("--f-hi", type=float, default=fp.F_HI,
                    help=f"plot window, upper [Hz] (default {fp.F_HI:g})")
    # None = what the set says; --no-legend-avg can override a set's True.
    ap.add_argument("--legend-avg", action=argparse.BooleanOptionalAction, default=None,
                    help="add each curve's geometric mean to its legend entry "
                         "(default: what the set says, see session_sets.legend_avg)")
    ap.add_argument("--plot-only", action="store_true",
                    help="draw from the existing data/ CSVs, compute nothing")
    ap.add_argument("--show", action="store_true",
                    help="show the figures at the end (plt.show), besides the PNGs")
    args = ap.parse_args(argv)
    choice = arguments.set_choice(args)
    sset = choice.sset

    common = dict(sessions=choice.rows, directory=sset.directory, method=args.method,
                  f_lo=args.f_lo, f_hi=args.f_hi, decimate_hz=args.decimate_hz,
                  filename=choice.filename, description=choice.description,
                  legend_avg=(sset.legend_avg if args.legend_avg is None
                              else args.legend_avg),
                  config=configs.check_config(choice.config),
                  plot_only=args.plot_only)
    tapers = args.tapers or fp.TAPERS
    if args.figures == "sweep":
        fp.sweep(tapers, **common)
    else:
        for t in tapers:
            fp.main(taper=tuple(t), figures=args.figures, **common)

    if args.show:
        fp.plt.show()


if __name__ == "__main__":
    main()
