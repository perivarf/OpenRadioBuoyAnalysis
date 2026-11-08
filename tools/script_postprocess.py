#!/usr/bin/env python3
"""Offline analysis of one session: postprocess.process() from the command line.

Writes <stamp>_spec_python.csv and <stamp>_ana_python.csv (and <stamp>_spec_acc.png / _spec_eta.png
unless --no-plot) beside the imu.csv, and prints the report.

    python3 tools/script_postprocess.py <session dir or *_imu.csv> [flags]
    python3 tools/script_postprocess.py <path> --taper-f1 0.2 --taper-f2 0.3 --no-plot
"""

import argparse
import sys

import _paths                                               # noqa: F401
import arguments                                            # noqa: E402
import postprocess                                          # noqa: E402

# Used when no path is given (F5 in Spyder). Empty -> the path is required.
DEFAULT_PATH = ""


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Offline ORB post-processing (mirrors DEBUG_POSTPROCESS).")
    ap.add_argument("path", nargs="?", default=DEFAULT_PATH,
                    help="session directory or path to <stamp>_imu.csv")
    arguments.add_postprocess_arguments(ap)
    args = ap.parse_args(argv)
    if not args.path:
        sys.exit("No path given: script_postprocess.py <path>, or set DEFAULT_PATH.")
    postprocess.process(args.path, arguments.postprocess_settings(args))


if __name__ == "__main__":
    main()
