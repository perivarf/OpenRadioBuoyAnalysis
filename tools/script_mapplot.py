#!/usr/bin/env python3
"""Map figure (OpenStreetMap) per session, and an overview map for several:
mapplot.make_maps() from the command line.

    python3 tools/script_mapplot.py <session dir> [<session dir> ...] [flags]
    python3 tools/script_mapplot.py <session dir> --no-map -o map.pdf
    python3 tools/script_mapplot.py <session dir> --init-note

For a whole series with the set's taper, use script_plot.py.
"""

import argparse
import os

import _paths                                               # noqa: F401
import arguments                                            # noqa: E402
import mapplot                                              # noqa: E402

# Used when no path is given.
DEFAULT_SESSION = os.path.join(os.path.dirname(_paths.TOOLS), "20260731 Skjærhalden", "20260731_131527")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Map figure (OpenStreetMap) for an ORB logging session.")
    ap.add_argument("path", nargs="*", default=[DEFAULT_SESSION],
                    help="one or more session directories / *_gps.csv, in "
                         "measurement order (override the numbers with --nr)")
    arguments.add_map_arguments(ap)
    args = ap.parse_args(argv)
    mapplot.make_maps(args.path or [DEFAULT_SESSION], arguments.map_options(args))


if __name__ == "__main__":
    main()
