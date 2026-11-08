"""Puts tools/ and its library subdirectories on sys.path.

The script_*.py files in tools/ import this first; after that every module is
imported by bare name ("import postprocess", "import osm"), whatever subdirectory it
is in.

    import _paths                 # noqa: F401
"""

import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))

for _d in ("analysis", "plots", "cli", "filters", "psd", "conversion", "maps", "models", "power",
           "norce_kystverket", ""):
    _d = os.path.join(TOOLS, _d) if _d else TOOLS
    if _d not in sys.path:
        sys.path.insert(0, _d)
