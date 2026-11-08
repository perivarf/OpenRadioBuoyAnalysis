#!/usr/bin/env python3
"""<stamp>_raw.bin -> CSV: raw_to_imu_csv.convert() from the command line.

    python3 tools/script_raw_to_imu_csv.py <session dir or *_raw.bin>            # samples
    python3 tools/script_raw_to_imu_csv.py <path> --mode imu --compare
    python3 tools/script_raw_to_imu_csv.py <path> --mode imu --ahrs nxp -o out.csv
"""

import argparse

import _paths                                               # noqa: F401
import raw_to_imu_csv as rti                                # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="<stamp>_raw.bin or the session directory")
    ap.add_argument("--mode", choices=["samples", "imu"], default="samples")
    ap.add_argument("--ahrs", choices=list(rti.AHRS_NAMES), default=None,
                    help="override the filter; without it cfg.csv orientation_name "
                         "is followed. nxp exists only here - it is an "
                         "alternative you ask for, not something a capture ran")
    ap.add_argument("-o", "--out", help="write the CSV here (default: next to the .bin)")
    ap.add_argument("--compare", action="store_true",
                    help="--mode imu: compare against the device's own imu.csv")
    ap.add_argument("--allow-damaged", action="store_true",
                    help="continue even if the raw log lost blocks (see "
                         "kRawFlagWriteFail). Everything after the first loss is "
                         "misparsed")
    args = ap.parse_args(argv)
    rti.convert(args.path, mode=args.mode, ahrs=args.ahrs, out=args.out,
                compare_device=args.compare, allow_damaged=args.allow_damaged)


if __name__ == "__main__":
    main()
