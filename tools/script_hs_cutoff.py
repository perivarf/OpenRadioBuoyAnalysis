#!/usr/bin/env python3
"""Hs as a function of the low-frequency cutoff, one curve per session:
hs_cutoff.main() from the command line.

    python3 tools/script_hs_cutoff.py SKJAERHALDEN
    python3 tools/script_hs_cutoff.py SKJAERHALDEN --fc-hi 0.2 --no-references
    python3 tools/script_hs_cutoff.py SKJAERHALDEN --method kalman --show
"""

import argparse

import _paths                                               # noqa: F401
import arguments                                            # noqa: E402
import hs_cutoff as hc                                      # noqa: E402
import spectra                                              # noqa: E402


def _reference(text):
    """"0.03:Kohout $f_1$" -> (0.03, "Kohout $f_1$"). The label is optional."""
    f, _, label = text.partition(":")
    try:
        return float(f), label.strip()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--reference takes F[:LABEL] with F in Hz, got '{text}'")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Significant wave height as a function of the low-frequency cutoff.")
    arguments.add_set_arguments(ap)
    ap.add_argument("--method", default=hc.METHOD, choices=sorted(spectra.ETA_COL),
                    help=f"whose Hs is plotted (default {hc.METHOD})")
    ap.add_argument("--decimate-hz", type=float, default=hc.DECIMATE_HZ,
                    help=f"the rate the series is decimated to before Welch "
                         f"(default {hc.DECIMATE_HZ:g}). Must match plot_felles_psd "
                         f"if the figures are to be read together")
    ap.add_argument("--fc-lo", type=float, default=hc.FC_LO,
                    help="lower f_c in the sweep [Hz]. Omitted = the lowest bin "
                         "the analysis has")
    ap.add_argument("--fc-hi", type=float, default=hc.FC_HI,
                    help=f"upper f_c in the sweep [Hz] (default {hc.FC_HI:g})")
    ap.add_argument("--reference", type=_reference, action="append",
                    dest="references", metavar="F[:LABEL]",
                    help="vertical reference line, may be repeated. Default: "
                         + ", ".join(f"{f:g}" for f, _ in hc.REFERENCES))
    ap.add_argument("--no-references", action="store_true",
                    help="draw no reference lines")
    ap.add_argument("--plot-only", action="store_true",
                    help="draw from the existing data/ CSVs, compute nothing")
    ap.add_argument("--show", action="store_true",
                    help="show the figure at the end (plt.show), besides the PNG")
    args = ap.parse_args(argv)
    choice = arguments.set_choice(args)

    hc.main(method=args.method, decimate_hz=args.decimate_hz, fc_lo=args.fc_lo,
            fc_hi=args.fc_hi, sessions=choice.rows, directory=choice.sset.directory,
            filename=choice.filename, description=choice.description,
            references=[] if args.no_references else (args.references or hc.REFERENCES),
            config=choice.config, plot_only=args.plot_only)

    if args.show:
        hc.plt.show()


if __name__ == "__main__":
    main()
