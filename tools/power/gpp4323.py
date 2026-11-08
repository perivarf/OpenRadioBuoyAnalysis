"""GW Instek GPP-4323 power supply over LAN (raw TCP socket with SCPI), and sample sources.

A sample source yields (t [s], voltage [V], current [A]): sample() from the instrument,
replay() from an existing <prefix>_raw.csv.
"""

import csv
import re
import socket
import time


class GPP4323:
    def __init__(self, host="192.168.2.124", port=1026, timeout=3.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self._buf = b""

    def close(self):
        self.sock.close()

    def write(self, cmd):
        self.sock.sendall((cmd + "\n").encode("ascii"))

    def query(self, cmd):
        self.write(cmd)
        while b"\n" not in self._buf:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise ConnectionError("The connection was closed by the instrument")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("ascii").strip()

    @staticmethod
    def _to_float(s):
        # Handles both "0.1234" and "0.1234A"
        return float(re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s).group())

    def idn(self):
        return self.query("*IDN?")

    def set_voltage(self, ch, volts):
        self.write(f":SOURce{ch}:VOLTage {volts:.3f}")

    def set_current_limit(self, ch, amps):
        self.write(f":SOURce{ch}:CURRent {amps:.4f}")

    def output(self, ch, on):
        self.write(f":OUTPut{ch}:STATe {'ON' if on else 'OFF'}")

    def measure_voltage(self, ch):
        return self._to_float(self.query(f":MEASure{ch}:VOLTage?"))

    def measure_current(self, ch):
        return self._to_float(self.query(f":MEASure{ch}:CURRent?"))

    def measure_vi(self, ch):
        """Voltage [V] and current [A] in one query (response: "U,I,P"; P has only 10 mW resolution)."""
        u, i, _ = self.query(f":MEASure{ch}:ALL?").split(",")
        return self._to_float(u), self._to_float(i)


def configure(psu, ch, voltage=None, current_limit=None):
    if voltage is not None:
        psu.set_voltage(ch, voltage)
    if current_limit is not None:
        psu.set_current_limit(ch, current_limit)


def power_cycle(psu, ch, off_time):
    """Output off for off_time, then on. Returns t0 = the moment of switch-on (perf_counter)."""
    psu.output(ch, False)
    time.sleep(off_time)
    psu.output(ch, True)
    return time.perf_counter()


def sample(psu, ch, t0, interval):
    """Measure every interval seconds, forever: yields (t - t0, V, A)."""
    next_t = t0
    while True:
        v, i = psu.measure_vi(ch)
        yield time.perf_counter() - t0, v, i
        next_t += interval
        pause = next_t - time.perf_counter()
        if pause > 0:
            time.sleep(pause)
        else:
            next_t = time.perf_counter()


def replay(path):
    """The samples of a <prefix>_raw.csv, up to the first unreadable line."""
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                yield float(row["t_s"]), float(row["voltage_V"]), float(row["current_A"])
            except (TypeError, ValueError):
                return
