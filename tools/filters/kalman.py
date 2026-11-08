"""EKF with orientation deviation states and gyro bias - Kok et al. (2017),
Algorithm 4 with the bias extension in Appendix C.4.

    M. Kok, J. D. Hol, T. B. Schön, "Using Inertial Sensors for Position and
    Orientation Estimation", Foundations and Trends in Signal Processing 11(1-2),
    2017. Equation numbers below refer to that text.

The buoy's Kalman filter; the firmware's KalmanAhrs (kalman.cpp) is a port of it.
Accel + gyro only (the magnetometer rows of Algorithm 4 are dropped).

    f = Kalman()
    f.init_from_accel(ax, ay, az)
    f.update(gx, gy, gz, ax, ay, az, dt)   # gyro rad/s, accel m/s²
    f.q                                    # [w, x, y, z], body -> world

State: x = [η^n, δω], the orientation deviation in the NAVIGATION frame (3) and the
gyro bias (3). The linearisation point q̃ is the attitude estimate.

    time update   q̃ <- q̃ ⊙ exp_q(T/2·(y_ω - δ̂_ω))                     (C.8)
                  P <- F P Fᵀ + G Q Gᵀ                                  (4.54b)
                  F = [I  -T·R̃; 0  I],  G = [T·R̃  0; 0  I]             (C.9)
                  Q = diag(Σ_ω, Σ_δω)                                   (C.5b)
    measurement   y_a = -R̃ᵀ gⁿ - R̃ᵀ[gⁿ×]·η + e_a                     (4.51)
                  H = [-R̃ᵀ[gⁿ×]  0],  R = Σ_a                          (Alg. 4)
                  ε = y - ŷ,  S = H P Hᵀ + R,  K = P Hᵀ S⁻¹             (4.38)
                  x̂ <- x̂ + K ε,  P <- P - K S Kᵀ                       (4.55)
    relinearise   q̃ <- exp_q(η̂/2) ⊙ q̃,  η̂ <- 0                       (4.56)

with gⁿ = [0, 0, -g] (z up) and R̃ = R(q̃) body -> world, i.e. R̃ = R^nb.

CONVENTIONS. Kok's exp_q(v) is the quaternion [cos|v|, sin|v|·v/|v|], a rotation by
2|v|. rotation.quat_exp(v) is a rotation by |v|, so exp_q(T/2·ω) = quat_exp(T·ω) and
exp_q(η/2) = quat_exp(η). Kok's ⊙ is the Hamilton product, as quat_mul.

NOISE AS DENSITIES, so the bandwidth is the same at any log rate:

    Σ_ω  = σ_g²/T      (G Q Gᵀ gives σ_g²·T on η)
    Σ_δω = σ_b²·T      (bias random walk)
    Σ_a  = σ_a²/T      (σ_a is an accel noise density [m/s²/√Hz])

Optional deviations, off by default:
    joseph=True   P update in Joseph form instead of (4.55b)
    lambda_w > 0  R scaled by (1 + λ_ω(|ω|/ω0)²)

The accel is not normalised: wave acceleration is measurement noise covered by Σ_a.
"""

import math

import numpy as np

from rotation import quat_exp, quat_from_accel, quat_matrix, quat_mul, skew

GRAVITY = 9.80665            # [m/s²]
SIGMA_G = 0.005              # gyro noise density [rad/s/√Hz]
SIGMA_B = 1.0e-5             # gyro bias random walk [rad/s²/√Hz]
# Accel noise density [m/s²/√Hz]; kKalmanParams.sigmaA in the firmware's kalman.h.
SIGMA_A = math.sqrt(2.0e-5) * GRAVITY
P0_ANGLE = 5.0               # initial orientation uncertainty [deg]
P0_BIAS = 1.0                # prior on the gyro bias [deg/s] (C.4)
LAMBDA_W = 0.0               # adaptive R weight; 0 = Kok as published
W0 = 1.0                     # normalisation for |ω| [rad/s]
DT_SEED = 0.002              # nominal dt [s] until the first time update


class Kalman:
    """Kok et al. (2017) Algorithm 4 + Appendix C.4, accel + gyro only."""

    def __init__(self, sigma_g=SIGMA_G, sigma_b=SIGMA_B, sigma_a=SIGMA_A,
                 lambda_w=LAMBDA_W, w0=W0, joseph=False, gravity=GRAVITY):
        self._q = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)                    # δ̂_ω, part of the state
        self.P = np.zeros((6, 6))
        p0a = math.radians(P0_ANGLE)
        p0b = math.radians(P0_BIAS)
        self.P[:3, :3] = np.eye(3) * p0a * p0a
        self.P[3:, 3:] = np.eye(3) * p0b * p0b
        self.sg, self.sb, self.sa = sigma_g, sigma_b, sigma_a
        self.lw, self.w0 = lambda_w, max(w0, 1e-9)
        self.joseph = joseph
        self.g_n = np.array([0.0, 0.0, -gravity])  # gⁿ, z up
        self._w_norm = 0.0
        self._dt = DT_SEED

    @property
    def q(self):
        """Attitude as [w, x, y, z], body -> world."""
        return list(self._q)

    def init_from_accel(self, ax, ay, az):
        """Algorithm 4 step 1 (§3.6): initial orientation from the accel, yaw = 0."""
        self._q = np.array(quat_from_accel(ax, ay, az), dtype=np.float64)

    def predict(self, gx, gy, gz, dt):
        """Time update, (C.8), (C.9), (4.54b)."""
        if dt <= 0.0:
            return self.q
        w = np.array([gx, gy, gz], dtype=np.float64)
        self._w_norm = float(np.linalg.norm(w))
        self._dt = dt
        # (C.8): q̃ <- q̃ ⊙ exp_q(T/2·(y_ω - δ̂_ω)) = q̃ ⊙ quat_exp(T·(y_ω - δ̂_ω))
        self._q = quat_mul(self._q, quat_exp((w - self.bias) * dt))
        self._q /= np.linalg.norm(self._q)
        Rt = quat_matrix(self._q)                  # R̃^nb_{t+1|t}
        # (C.9)
        F = np.eye(6)
        F[:3, 3:] = -dt * Rt
        G = np.zeros((6, 6))
        G[:3, :3] = dt * Rt
        G[3:, 3:] = np.eye(3)
        # (C.5b) from densities
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * (self.sg * self.sg / dt)
        Q[3:, 3:] = np.eye(3) * (self.sb * self.sb * dt)
        self.P = F @ self.P @ F.T + G @ Q @ G.T    # (4.54b)
        return self.q

    def r_accel(self):
        """Σ_a, scaled by the adaptive term when lambda_w > 0."""
        dt = self._dt if self._dt > 0.0 else DT_SEED
        r = self.sa * self.sa / dt
        if self.lw > 0.0:
            wn = self._w_norm / self.w0
            r *= 1.0 + self.lw * wn * wn
        return r

    def correct(self, ax, ay, az):
        """Measurement update (4.51), (4.38), (4.55), then relinearisation (4.56)."""
        y = np.array([ax, ay, az], dtype=np.float64)
        RtT = quat_matrix(self._q).T               # R̃^bn
        y_hat = -RtT @ self.g_n                    # ŷ = -R̃^bn gⁿ
        H = np.zeros((3, 6))
        H[:, :3] = -RtT @ skew(self.g_n)           # (4.51)
        Rm = np.eye(3) * self.r_accel()
        eps = y - y_hat                            # (4.38)
        S = H @ self.P @ H.T + Rm
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.q
        K = self.P @ H.T @ Sinv
        dx = K @ eps                               # (4.55a); η̂ before was 0
        if self.joseph:
            IKH = np.eye(6) - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ Rm @ K.T
        else:
            self.P = self.P - K @ S @ K.T          # (4.55b)
        self.bias = self.bias + dx[3:]
        # (4.56): q̃ <- exp_q(η̂/2) ⊙ q̃ (left: η is in the navigation frame).
        self._q = quat_mul(quat_exp(dx[:3]), self._q)
        self._q /= np.linalg.norm(self._q)
        return self.q

    def update(self, gx, gy, gz, ax, ay, az, dt):
        """One step: time update, then measurement update."""
        if dt <= 0.0:
            return self.q
        self.predict(gx, gy, gz, dt)
        return self.correct(ax, ay, az)
