"""Step 9, conversion to elevation: Spectra -> ElevationSpectra.

    IMU:  S_eta = S_acc / ω⁴          GPS:  S_eta = S_vUp / ω²       (ω = 2πf)

Bin 0 (f = 0) is set to 0. No taper here; that is step 10.

    elev = to_elevation(spectra)
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ElevationSpectra:
    """Step 9 output: untapered elevation PSDs [m²/Hz] on the Spectra bins."""
    spectra: object              # the Spectra they came from (acc/vel, fs, seglen)
    eta: dict                    # {key: S_acc/ω⁴}
    eta_gps: np.ndarray          # S_vUp/ω², or None

    @property
    def fs(self):
        return self.spectra.fs

    @property
    def seglen(self):
        return self.spectra.seglen

    @property
    def df(self):
        return self.spectra.df

    @property
    def f(self):
        return self.spectra.f


def divide_by_omega(psd, df, power):
    """psd[k] / (2π·k·df)^power for k >= 1, 0 at k = 0."""
    out = np.zeros(len(psd), dtype=np.float64)
    for k in range(1, len(psd)):
        w = 2.0 * math.pi * (k * df)
        out[k] = psd[k] / (w ** power)
    return out


def to_elevation(spectra):
    """Spectra -> ElevationSpectra."""
    df = spectra.df
    eta = {key: divide_by_omega(psd, df, 4) for key, psd in spectra.acc.items()}
    eta_gps = divide_by_omega(spectra.vel, df, 2) if spectra.vel is not None else None
    return ElevationSpectra(spectra=spectra, eta=eta, eta_gps=eta_gps)
