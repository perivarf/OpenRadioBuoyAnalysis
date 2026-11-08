#!/usr/bin/env python3
"""Map figures for a measurement series: an overview map plus one map per session.

Every session runs with the same taper and segment length, so Hs/Tz/Tc are
comparable. The order of the sessions is the measurement number in the badges and
the overview table. Sessions with only a raw log are reconstructed first
(session_source.ensure_imu).

    python3 tools/script_plot.py SKJAERHALDEN
    python3 tools/script_plot.py SKJAERHALDEN --taper 0.03 0.05
    python3 tools/script_plot.py SKJAERHALDEN --skip-postprocess
    python3 tools/script_plot.py SKJAERHALDEN --plot-only    # redraw from data/ only
    python3 tools/script_plot.py SKJAERHALDEN --only-overview --format pdf --dpi 300
    python3 tools/script_plot.py SKJAERHALDEN --sessions 20260731_110303 20260731_110314
    python3 tools/script_plot.py Torrevieja              # directory without a set

Other flags are mapplot's (see script_mapplot.py --help).
"""

import argparse
import dataclasses
import sys

import _paths                                               # noqa: F401
import arguments                                            # noqa: E402
import mapplot                                              # noqa: E402
import configs                                              # noqa: E402
import session_source                                       # noqa: E402

# Fallback taper [Hz] when neither --taper nor the set gives one.
TAPER_F1, TAPER_F2 = 0.2, 0.3

# Welch segment length: 1024 at 10 Hz = 51.2 s segments, df = 0.0195 Hz.
SEGLEN = 1024


def main(argv=None):
    # allow_abbrev=False, or --sessions would swallow mapplot flags as prefixes.
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Map figures for a measurement series: overview + one per session.",
        epilog="Unknown arguments are passed on to mapplot "
               "(--skip-postprocess, --only-overview, --format, --no-map, ...).")
    arguments.add_set_arguments(ap, description=False)
    ap.add_argument("--taper", nargs=2, type=float, default=None,
                    metavar=("F1", "F2"),
                    help=f"low-frequency taper: T=0 below F1, T=1 above F2 [Hz]; "
                         f"F1 == F2 is a hard cut (default: the set's taper, else "
                         f"{TAPER_F1:g} {TAPER_F2:g})")
    ap.add_argument("--seglen", type=int, default=SEGLEN,
                    help=f"Welch segment length, power of 2 (default {SEGLEN})")
    args, rest = ap.parse_known_args(argv)
    choice = arguments.set_choice(args)
    sset = choice.sset
    config = configs.check_config(choice.config)

    # Work directory per session (under <config>/ with a config); the overview lands
    # in their common parent. --plot-only (a mapplot flag) touches no inputs.
    if "--plot-only" in rest:
        dirs = [session_source.workdir(d, config)[0] for d in choice.dirs]
    else:
        dirs = [session_source.ensure_imu(d, config) for d in choice.dirs]

    # Taper: command line, then the set, then the fallback.
    if args.taper:
        f1, f2 = args.taper
    else:
        f1, f2 = sset.taper or (TAPER_F1, TAPER_F2)
        if sset.taper:
            print(f"  taper {f1:g}-{f2:g} Hz, from the set {sset.name}", flush=True)
    if not 0.0 < f1 <= f2:
        sys.exit(f"--taper needs 0 < F1 <= F2 (got {f1:g} {f2:g})")

    opts = dataclasses.replace(arguments.map_options_from_argv(rest),
                               taper_f1=f1, taper_f2=f2, seglen=args.seglen,
                               prefix=choice.filename, config=config)
    if sset.badge_colors and not opts.badge_colors:
        opts = dataclasses.replace(opts, badge_colors=",".join(
            sset.badge_colors.get(row[0], "") for row in choice.rows))
    if sset.badge_legend and not opts.badge_legend:
        opts = dataclasses.replace(opts, badge_legend=sset.badge_legend)
    if sset.badge_nr and not opts.nr:
        # A session the set does not number gets its position.
        opts = dataclasses.replace(opts, nr=",".join(
            sset.badge_nr.get(row[0], str(i)) for i, row in enumerate(choice.rows, 1)))
    print(f"$ mapplot {' '.join(dirs)}  taper {f1:g}-{f2:g}  seglen {args.seglen}  "
          f"prefix {choice.filename}" + (f"  config {config}" if config else ""),
          flush=True)
    return mapplot.make_maps(dirs, opts)


if __name__ == "__main__":
    raise SystemExit(main())
