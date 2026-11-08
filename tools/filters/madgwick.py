"""Madgwick 6-axis AHRS (gyro + accel), numerically identical to analysis.cpp.

Same interface as the other filters:

    f = Madgwick(beta=0.2)
    f.init_from_accel(ax, ay, az)          # one sample -> initial orientation
    f.update(gx, gy, gz, ax, ay, az, dt)   # gyro in rad/s, accel in any unit
    f.q                                    # [w, x, y, z], body -> world

The accel unit does not matter (the vector is normalised), gyro MUST be rad/s and
dt seconds.
"""

import math

from rotation import quat_from_accel


class Madgwick:
    """Madgwick 6-axis AHRS. The state is the quaternion alone."""

    def __init__(self, beta=0.2):
        # beta: accel correction gain; high follows accel noise, low gyro drift.
        self.beta = float(beta)
        self._q = [1.0, 0.0, 0.0, 0.0]

    @property
    def q(self):
        """The orientation as [w, x, y, z], body -> world."""
        return self._q

    def init_from_accel(self, ax, ay, az):
        """Set the initial orientation from one accel sample (yaw = 0)."""
        self._q = quat_from_accel(ax, ay, az)

    def update(self, gx, gy, gz, ax, ay, az, dt):
        """One step; returns the new quaternion. Written out as in analysis.cpp;
        s0..s3 is the analytic gradient of the gravity error."""
        q0, q1, q2, q3 = self._q
        qdot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qdot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qdot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qdot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        if not (ax == 0.0 and ay == 0.0 and az == 0.0):
            recip = 1.0 / math.sqrt(ax * ax + ay * ay + az * az)
            ax *= recip; ay *= recip; az *= recip
            _2q0, _2q1, _2q2, _2q3 = 2.0 * q0, 2.0 * q1, 2.0 * q2, 2.0 * q3
            _4q0, _4q1, _4q2 = 4.0 * q0, 4.0 * q1, 4.0 * q2
            _8q1, _8q2 = 8.0 * q1, 8.0 * q2
            q0q0, q1q1, q2q2, q3q3 = q0 * q0, q1 * q1, q2 * q2, q3 * q3
            s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
            s1 = (_4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay
                  - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az)
            s2 = (4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay
                  - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az)
            s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay
            recip = 1.0 / math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
            s0 *= recip; s1 *= recip; s2 *= recip; s3 *= recip
            qdot1 -= self.beta * s0
            qdot2 -= self.beta * s1
            qdot3 -= self.beta * s2
            qdot4 -= self.beta * s3

        q0 += qdot1 * dt; q1 += qdot2 * dt; q2 += qdot3 * dt; q3 += qdot4 * dt
        recip = 1.0 / math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        self._q = [q0 * recip, q1 * recip, q2 * recip, q3 * recip]
        return self._q
