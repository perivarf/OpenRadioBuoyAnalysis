#!/usr/bin/env python3
"""Wave heights for one session under a list of processing combinations, as a CSV
and one table per taper: config_test.run() from the command line.

    python3 tools/script_test_config.py KATTEN --session 20260830_131145 \\
        --combo "1 average 1024 30" --combo "2 sinc 512 10" --taper 0.1 0.2
    python3 tools/script_test_config.py KATTEN ... --plot-only   # redraw from the CSV

A combination is "D FIR SEGLEN MINUTES": raw averaging D (480/D Hz), FIR average or
sinc (both decimation stages), Welch segment length, capture length. Output in
<measurement>/test_config/. Run from test_config.sh in the measurement directory.
"""

import argparse
import os

import _paths                                               # noqa: F401
import config_test                                          # noqa: E402
import session_sets                                         # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("set", help="set name in session_sets.py")
    ap.add_argument("--session", help="session folder (default: the set's first)")
    ap.add_argument("--combo", action="append", required=True, metavar="'D FIR SEGLEN MIN'",
                    help="one combination, repeatable, e.g. '2 average 1024 30'")
    ap.add_argument("--taper", nargs=2, type=float, action="append", required=True,
                    metavar=("F1", "F2"), help="taper band, repeatable: one table each")
    ap.add_argument("--psd-compare", action="append", default=[],
                    metavar="'D FIR SEGLEN MIN [HZ [TAPS]]'",
                    help="a combination (one of --combo) in the PSD comparison figure, "
                         "repeatable")
    ap.add_argument("--psd-taper", nargs=2, type=float, metavar=("F1", "F2"),
                    help="the taper whose Hs goes in the PSD comparison's legend "
                         "(one of --taper; default the first)")
    ap.add_argument("--fmax-compare", type=float, action="append", default=[],
                    metavar="HZ",
                    help="an upper integration limit for the Hs table of the "
                         "--psd-compare combinations, repeatable")
    ap.add_argument("--chain-table", action="append", default=[], metavar="'FIR [TAPS]'",
                    help="a chain (the FIR field of --combo, with its TAPS) as rows in "
                         "the chain table, repeatable")
    ap.add_argument("--chain-fmax", type=float, action="append", default=[], metavar="HZ",
                    help="an upper integration limit for the chain table (one table "
                         "each, from the saved PSDs), repeatable; default the CSV's")
    ap.add_argument("--decimation-table", action="append", default=[],
                    metavar="'FIR SEGLEN MIN [HZ [TAPS]]'",
                    help="a combination without D: Hs and Tp per AHRS filter at every D of "
                         "it in COMBOS, one table each, repeatable")
    ap.add_argument("--plot-only", action="store_true",
                    help="draw the tables from the existing CSV, compute nothing")
    args = ap.parse_args(argv)
    sset = session_sets.get_set(args.set)
    session = args.session or sset.sessions[0][0]
    directory = sset.directory              # relative to the repo root, as the others
    config_test.run(os.path.join(directory, session),
                    [config_test.parse_combo(c) for c in args.combo],
                    [tuple(t) for t in args.taper],
                    os.path.join(directory, "test_config"), sset.filename,
                    sset.description, plot_only=args.plot_only,
                    psd_compare=[config_test.parse_combo(c) for c in args.psd_compare],
                    psd_taper=tuple(args.psd_taper) if args.psd_taper else None,
                    fmax_compare=args.fmax_compare,
                    chain_fmax=args.chain_fmax,
                    chain_table=[(p[0].lower(), int(p[1]) if len(p) > 1 else None)
                                 for p in (c.split() for c in args.chain_table)],
                    decimation_table=[config_test.parse_combo("1 " + c)[1:]
                                      for c in args.decimation_table])


if __name__ == "__main__":
    main()
