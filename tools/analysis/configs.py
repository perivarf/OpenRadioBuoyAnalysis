"""Configs: a config names one run of the figure chain (e.g. "0.2-0.3").

Its output goes in <measurement>/<config>/ (series figures) and
<measurement>/<config>/<stamp>/ (per session, see session_source.ensure_imu), and
figures get _<config> in their name. "" = no config.
"""

import os
import sys

CONFIG_SUFFIXED = (".png", ".pdf")


def check_config(config):
    """The config name, or exit if it is not a plain directory name."""
    config = config or ""
    if config and (config != config.strip() or any(c in config for c in "/\\")
                   or config.startswith(".") or ".." in config):
        sys.exit(f"--config {config!r}: must be a plain directory name "
                 f"(no /, \\, '..', leading '.' or surrounding spaces)")
    return config


def config_dir(directory, config):
    """<directory>/<config>/, created - or directory itself when config is empty."""
    if not config:
        return directory
    out = os.path.join(directory, check_config(config))
    os.makedirs(out, exist_ok=True)
    return out


def config_suffix(filename, config):
    """name.png -> name_<config>.png (PNG and PDF only; CSV names stay fixed).
    Empty config -> unchanged."""
    stem, ext = os.path.splitext(filename)
    if not config or ext.lower() not in CONFIG_SUFFIXED:
        return filename
    return f"{stem}_{check_config(config)}{ext}"


# Figures that sweep over the taper themselves (felles_psd, hs_cutoff) do not
# depend on the config's taper, so they share one directory across configs.
SWEEP_DIR = "taper_sweep"


def sweep_dir(directory):
    """<directory>/taper_sweep/, created."""
    out = os.path.join(directory, SWEEP_DIR)
    os.makedirs(out, exist_ok=True)
    return out


# Every computed figure writes its numbers here first and is drawn from them, so a
# figure can be redrawn (--plot-only) without recomputing anything.
DATA_DIR = "data"


def data_path(png_path, config=""):
    """The CSV a figure is drawn from: <dir>/data/<stem>.csv beside the PNG, with
    the _<config> suffix taken off the stem (CSV names stay fixed). data/ is created."""
    directory, name = os.path.split(png_path)
    stem = os.path.splitext(name)[0]
    tail = f"_{config}" if config else ""
    if tail and stem.endswith(tail):
        stem = stem[:-len(tail)]
    out = os.path.join(directory, DATA_DIR)
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, f"{stem}.csv")


def write_data(df, path):
    """DataFrame -> CSV at full precision, so a figure drawn from it is identical to
    one drawn from the numbers in memory."""
    df.to_csv(path, index=False, float_format="%.17g")
    print(f"  wrote data/{os.path.basename(path)}")


def read_data(path, step):
    """CSV -> DataFrame; stop with the step that writes it when it is missing
    (--plot-only before the data has been computed)."""
    import pandas as pd
    if not os.path.isfile(path):
        sys.exit(f"--plot-only: {path} is missing - run {step} without --plot-only "
                 f"first")
    return pd.read_csv(path)
