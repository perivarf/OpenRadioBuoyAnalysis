#!/usr/bin/env python3
"""Summary of a raw log (header, rates, lost blocks): rawlog.summary() from the
command line.

    python3 tools/script_rawlog.py <stamp>_raw.bin
"""

import argparse

import _paths                                               # noqa: F401
import rawlog                                               # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summary of a <stamp>_raw.bin.")
    ap.add_argument("path", help="<stamp>_raw.bin")
    args = ap.parse_args(argv)
    print(rawlog.summary(rawlog.read(args.path)))


if __name__ == "__main__":
    main()
