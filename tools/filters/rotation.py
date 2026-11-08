"""Quaternion and rotation maths, no domain constants.

Conventions (as analysis.cpp):
  - a quaternion is [w, x, y, z] (Hamilton, scalar first)
  - q rotates body -> world: v_world = R(q) · v_body
  - roll about x, pitch about y, yaw = 0

Every function takes one quaternion (one sample).
"""

import math

import numpy as np


def quat_from_accel(ax, ay, az):
    """Quaternion from one accel sample (any unit), yaw = 0, as analysis.cpp."""
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    return quat_from_roll_pitch(roll, pitch)


def quat_from_roll_pitch(roll, pitch):
    """Quaternion [w,x,y,z] from roll/pitch with yaw = 0."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return [cp * cr, cp * sr, sp * cr, -sp * sr]


def quat_mul(a, b):
    """Hamilton product for [w,x,y,z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_exp(v):
    """Rotation vector (rad) -> quaternion. The small-angle branch avoids 0/0."""
    th = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if th < 1e-12:
        return np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]])
    s = math.sin(0.5 * th) / th
    return np.array([math.cos(0.5 * th), v[0] * s, v[1] * s, v[2] * s])


def quat_matrix(q):
    """R(q) body -> world, 3x3, for ONE quaternion."""
    qw, qx, qy, qz = q
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
        [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
        [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)]])


def world_z(q, ax, ay, az):
    """(R(q)·a)_z: the vertical component of a in the world frame. Gravity is not
    subtracted."""
    qw, qx, qy, qz = q
    return (2.0 * (qx * qz - qw * qy) * ax
            + 2.0 * (qy * qz + qw * qx) * ay
            + (1.0 - 2.0 * (qx * qx + qy * qy)) * az)


def skew(v):
    """Skew-symmetric matrix such that skew(a) @ b == np.cross(a, b)."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])
