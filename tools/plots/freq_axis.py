"""The logarithmic frequency axis shared by the spectrum figures."""

import math


def log_hz_axis(ax, xlim=None):
    """Logarithmic frequency axis with plain Hz labels: decades as major, 2 and 5 as
    minor ticks."""
    import matplotlib.ticker as mticker
    ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2.0, 5.0)))

    # Decimals per value (0.05 -> "0.05", 1 -> "1").
    def _hz(v, _pos):
        if v <= 0.0:
            return ""
        return f"{v:.{max(0, int(math.ceil(-math.log10(v))))}f}"

    for which in ("major", "minor"):
        getattr(ax.xaxis, f"set_{which}_formatter")(mticker.FuncFormatter(_hz))
    ax.tick_params(axis="x", which="minor", labelsize=7)
