#!/usr/bin/env python3
"""Named session sets: which measurements belong together in a figure.

One set per measurement series: directory, sessions in order, labels, styles,
filename prefix and taper. The scripts take the set name as their first argument:

    python3 tools/script_plot.py SKJAERHALDEN
    python3 tools/script_felles_psd.py SKJAERHALDEN --figures acc

A measurement directory without a set can also be given; it is then scanned for
session folders (get_set), with folder names as labels and matplotlib's colours.
"""

import os
import sys
from collections import namedtuple
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))        # the repo root

# directory    the measurement directory, relative to ROOT (or absolute)
# description  first in the figure titles
# filename     prefix on the figure files
# sessions     [[session dir, label, line style, colour], ...]; the order is the
#              measurement number (mapplot badges, overview table, legends)
# legend_avg   add each PSD curve's geometric mean to its legend entry
# badge_colors {session dir: colour} for the mapplot number badges, or None; a
#              session not in it gets mapplot's default (C_BADGE, red)
# badge_legend what the badge colours mean, in the key under the overview map, or
#              None
# badge_nr     {session dir: label} for the mapplot measurement numbers (badges,
#              overview table), or None for 1, 2, 3, ... in set order
# taper        (f1, f2) [Hz] for the low-frequency taper, or None for the tool's
#              default. Used by script_plot.py; script_felles_psd.py sweeps its own.
SessionSet = namedtuple("SessionSet",
                        "name directory description filename sessions "
                        "legend_avg taper badge_colors badge_legend badge_nr",
                        defaults=(False, None, None, None, None))


# Locations 1..3, each logged by two buoys side by side: A = 240 Hz, B = 480 Hz
# (the thesis' names). Colour = location, style = rate.
SKJAERHALDEN = SessionSet(
    name="SKJAERHALDEN",
    directory="20260731 Skjærhalden",
    description="Skjærhalden",
    filename="Skjærhalden_20260731",
    sessions=[["20260731_110303", "1A - 240 Hz", "-",  "C0"],
              ["20260731_110314", "1B - 480 Hz", "--", "C0"],
              ["20260731_122523", "2A - 240 Hz", "-",  "C1"],
              ["20260731_122652", "2B - 480 Hz", "--", "C1"],
              ["20260731_131527", "3A - 240 Hz", "-",  "C2"],
              ["20260731_131522", "3B - 480 Hz", "--", "C2"],
              ],
    # Inshore bay: periods over 5 s are drift and noise, not sea.
    taper=(0.2, 0.3),
    # The measurement names in the thesis: location 1-3, A = the 240 Hz buoy,
    # B = the 480 Hz buoy.
    badge_nr={"20260731_110303": "1A", "20260731_110314": "1B",
              "20260731_122523": "2A", "20260731_122652": "2B",
              "20260731_131527": "3A", "20260731_131522": "3B"})

# One 30-minute session drifting off Katten, Skjærhalden (Hs 0.59 m on board).
KATTEN = SessionSet(
    name="KATTEN",
    directory="20260830 Skjærhalden",
    description="Katten, Skjærhalden",
    filename="Katten_20260830",
    sessions=[["20260830_131145", "13:11 - 480 Hz", "-", "C0"]],
    # The wave hump is at 0.13-0.25 Hz, a drift blob peaks at 0.049 Hz; the taper
    # sits in the valley between them at ~0.1 Hz.
    taper=(0.07, 0.10))


SETS = {S.name: S for S in (SKJAERHALDEN, KATTEN)}


def find_sessions(directory):
    """Session folders in directory (those with a *_gps.csv) -> sessions list with
    folder names as labels and no colours."""
    return [[os.path.basename(p.rstrip("/")), os.path.basename(p.rstrip("/")),
             "-", None]
            for p in sorted(glob(os.path.join(directory, "*", "")))
            if glob(os.path.join(p, "*_gps.csv"))]


def measurement_dirs(root=ROOT):
    """The directories under root that contain session folders. Error messages only."""
    return sorted(os.path.basename(p.rstrip("/"))
                  for p in glob(os.path.join(root, "*", ""))
                  if glob(os.path.join(p, "*", "*_gps.csv")))


def get_set(name, root=ROOT, quiet=False):
    """Set name or measurement directory -> SessionSet with an absolute directory.

    A name in SETS wins over a directory with the same name. An unknown name stops
    with the list of sets and measurement directories."""
    if name and name.upper() in SETS:
        s = SETS[name.upper()]
        return s._replace(directory=os.path.join(root, s.directory))

    directory = name if os.path.isabs(name or "") else os.path.join(root, name or "")
    if os.path.isdir(directory):
        sessions = find_sessions(directory)
        if not sessions:
            sys.exit(f"{directory} holds no session folders with a *_gps.csv.")
        base = os.path.basename(os.path.abspath(directory))
        if not quiet:
            print(f"'{name}' is not a set in session_sets.SETS - found "
                  f"{len(sessions)} sessions by scanning the directory. "
                  f"Labels = folder names, colours from matplotlib.", flush=True)
        return SessionSet(name=base, directory=os.path.abspath(directory),
                          description=base, filename=base, sessions=sessions)

    sys.exit(f"Unknown set or directory '{name}'.\n"
             f"Sets in session_sets.SETS:\n  " + "\n  ".join(sorted(SETS))
             + f"\nMeasurement directories under {root}:\n  "
             + "\n  ".join(measurement_dirs(root) or ["(none)"]))


def paths(sset, sessions=None):
    """Absolute session directories for the set's sessions, or for `sessions`
    (names in the set's directory)."""
    return [os.path.join(sset.directory,
                         s[0] if isinstance(s, (list, tuple)) else s)
            for s in (sset.sessions if sessions is None else sessions)]


def check(sset, sessions=None):
    """Like paths(), but stops listing the existing sessions if any are missing."""
    out = paths(sset, sessions)
    missing = [p for p in out if not os.path.isdir(p)]
    if missing:
        existing = [s[0] for s in find_sessions(sset.directory)]
        sys.exit("Could not find the session directory/directories:\n  "
                 + "\n  ".join(missing)
                 + f"\nSessions in {sset.directory}:\n  "
                 + "\n  ".join(existing or ["(none)"]))
    return out


def select(sset, sessions=None):
    """The set's rows, or the rows for `sessions` in that order. Names in the set
    keep its label, style and colour; others get a bare row."""
    if not sessions:
        return list(sset.sessions)
    own = {row[0]: row for row in sset.sessions}
    return [own.get(s) or [s, os.path.basename(s.rstrip("/")), "-", None]
            for s in sessions]
