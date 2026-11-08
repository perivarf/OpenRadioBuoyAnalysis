"""NXP Sensor Fusion Kalman (12 states), the filter the SFY buoy runs, through the
compiled python module `ahrs_fusion`.

    f = KalmanNxp(fs=50.0)
    f.init_from_accel(ax, ay, az)
    f.update(gx, gy, gz, ax, ay, az, dt)   # gyro rad/s, accel m/s²
    f.q                                    # [w, x, y, z], body -> world

Units are converted here to NXP's deg/s and g; the magnetometer is zero.

Fixed rate: dt = 1/fs is set at construction, and update() ignores dt (except
skipping dt <= 0). fs must be the actual row rate from cfg.csv, and the time axis
must be uniform.

Initial orientation: ahrs_fusion is NXP's 9DOF filter, whose once-only eCompass
lock on the first update counts a zero magnetometer as calibrated (|m| <= 50 uT)
and locks to a near-identity orientation; from there the Kalman crawls to the true
tilt (~20 s from 60°, vacc -5 m/s² at t=0). init_from_accel therefore spends that
lock and then sets the quaternion from the accel, as Madgwick and Kalman start.
"""

import math

from rotation import quat_from_accel

GRAVITY = 9.80665                     # kGravity [m/s²] - same as postprocess
RADS2DPS = 180.0 / math.pi            # rad/s -> deg/s (NXP's gyro unit)


class KalmanNxp:
    """NXP 12-state Kalman. All state lives in the compiled module."""

    def __init__(self, fs, gravity=GRAVITY):
        # Imported here: ahrs_fusion is optional.
        try:
            from ahrs_fusion import NxpFusion
        except ImportError as e:
            raise ImportError(
                "kalman_nxp requires the python module 'ahrs_fusion' (the same "
                "code SFY runs). Build it from the ahrs-fusion repo with "
                "'maturin develop --features python'."
            ) from e
        if not (fs > 0.0):
            raise ValueError(f"kalman_nxp: fs must be > 0, got {fs}")
        self.fs = float(fs)
        self.gravity = float(gravity)
        self._f = NxpFusion.new(self.fs)

    @property
    def q(self):
        """Orientation as [w, x, y, z], body -> world."""
        return list(self._f.quaternion)

    def _step(self, gx, gy, gz, ax, ay, az):
        """One filter step, converted to NXP's units."""
        self._f.update(gx * RADS2DPS, gy * RADS2DPS, gz * RADS2DPS,
                       ax / self.gravity, ay / self.gravity, az / self.gravity,
                       0.0, 0.0, 0.0)          # magnetometer off, as in buf.rs

    def init_from_accel(self, ax, ay, az):
        """Initial orientation from one accel sample (yaw = 0). One step with zero
        gyro spends NXP's once-only lock first - set before it, the quaternion
        would be overwritten by the lock."""
        self._step(0.0, 0.0, 0.0, ax, ay, az)
        self._f.set_quaternion([float(v) for v in quat_from_accel(ax, ay, az)])

    def update(self, gx, gy, gz, ax, ay, az, dt):
        """One filter step. dt only skips non-monotonic rows; the step is 1/fs."""
        if dt <= 0.0:
            return self.q
        self._step(gx, gy, gz, ax, ay, az)
        return self.q
