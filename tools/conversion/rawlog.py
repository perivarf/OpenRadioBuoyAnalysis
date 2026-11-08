"""Reads <stamp>_raw.bin - the undecimated IMU log from wave_raw_log.

Format (little-endian, see wave_raw_log in wave_config.h):

    header  32 B: magic "ORWB", version, word length, ODRs,
                  sensitivities (LSB -> mg / mdps), capture start UTC, reading_ID
    word     7 B: tag_sensor + 6 B payload, as in the FIFO
    sync    17 B: 0xFF + u32 t_us + u32 accel_n + u32 millis + u16 n_words + u16 flags
    gps     61 B: 0xFE + u8 payload length (59) + u32 millis + u32 rel_ms + u32 iTOW_ms
                  + u8 hour, minute, second, fixType, numSV + i32 lat_e7, lng_e7,
                  gSpeed_mms, velN_mms, velE_mms, velD_mms, headMot_e5
                  + u32 sAcc_mms, hAcc_mm, vAcc_mm + u16 pDOP_e2      (v3 only)

In v3 every tag above 0x1F other than the sync carries a length byte, so a record
this reader does not know is skipped. v3 replaced the firmware's gps.csv with the GPS
records; write_gps_csv() writes them back out in the gps.csv layout.

Sync flags:
    bit 0  FIFO_OVF    the sensor overwrote words: time is missing, file intact
    bit 1  WRITE_FAIL  the file lost bytes: everything after is misparsed

Words have no timestamps; the accel times are interpolated between sync records
(t_us, accel_n), and gyro/SFLP take the accel time at arrival.

    cap = rawlog.read("20260731_110314_raw.bin")
    cap.acc_t, cap.acc      # (N,) s since capture start, (N,3) mg
    cap.gyr_t, cap.gyr      # (M,) s, (M,3) mdps
    cap.quat_t, cap.quat    # (K,) s, (K,4) [w,x,y,z]
    cap.gps_t, cap.gps      # (G,) s, list of GPS_FIELDS tuples (v3)
"""

import struct
import sys

import numpy as np

MAGIC = 0x4257524F            # "ORWB"
SYNC_TAG = 0xFF
HEADER_BYTES = 32
FLAG_FIFO_OVF = 0x0001        # kRawFlagFifoOvf
FLAG_WRITE_FAIL = 0x0002      # kRawFlagWriteFail
TAG_GYR = 1
TAG_ACC = 2
TAG_SFLP = 0x13               # kSflpGameRotationTag
GPS_TAG = 0xFE                # kRawGpsTag
# Fields: millis rel_ms itow_ms hour minute second fix_type num_sv lat_e7 lng_e7
# gspeed_mms veln_mms vele_mms veld_mms headmot_e5 sacc_mms hacc_mm vacc_mm pdop_e2
GPS_PAYLOAD = struct.Struct("<IIIBBBBBiiiiiiiIIIH")   # kRawGpsPayloadBytes = 59
# gps.csv as the firmware wrote it before v3, plus t_s (accel time axis) last
GPS_COLUMNS = ("rel_ms", "utc", "itow", "lat", "lon", "gspeed", "vN", "vE", "vUp", "head",
               "sAccuracy", "hAccuracy", "vAccuracy", "pdop", "sats", "t_s")


class RawCapture:
    """A decoded raw capture. The fields are numpy arrays; times are seconds from start."""

    def __init__(self, hdr):
        self.__dict__.update(hdr)
        self.acc_t = self.gyr_t = self.quat_t = None
        self.acc = self.gyr = self.quat = None
        self.gps = []             # GPS_FIELDS tuples, one per fix (v3)
        self.gps_t = np.empty(0)
        self.sync = []            # dicts, one per drain
        self.n_overflow = 0
        self.n_write_fail = 0
        # Accel index of the first WRITE_FAIL: the cut point.
        self.first_write_fail_n = None
        self.unknown_tags = {}
        # (byte offset, reason) where a foreign tail starts, or None.
        self.tail_at = None

    def trustworthy_upto(self):
        """Number of usable accel samples. len(acc) when nothing was lost."""
        if self.first_write_fail_n is None:
            return len(self.acc)
        return min(int(self.first_write_fail_n), len(self.acc))


def read_header(buf):
    if len(buf) < HEADER_BYTES:
        sys.exit("raw: the file is shorter than the header")
    magic, ver = struct.unpack_from("<IB", buf, 0)
    if magic != MAGIC:
        sys.exit(f"raw: wrong magic 0x{magic:08X} - not a _raw.bin")

    # v1 has a u16 build_seq after word_bytes; v2 does not. Records are the same.
    if ver == 1:
        wordlen, = struct.unpack_from("<B", buf, 5)
        imu_odr, sflp_odr = struct.unpack_from("<HH", buf, 8)
        acc_sens, gyr_sens = struct.unpack_from("<ff", buf, 12)
        t0, reading_id, sync_bytes = struct.unpack_from("<IHH", buf, 20)
    elif ver in (2, 3):     # v3 added the GPS record, the header is v2's
        wordlen, imu_odr, sflp_odr = struct.unpack_from("<BHH", buf, 5)
        acc_sens, gyr_sens = struct.unpack_from("<ff", buf, 10)
        t0, reading_id, sync_bytes = struct.unpack_from("<IHH", buf, 18)
    else:
        sys.exit(f"raw: unknown format v{ver} - this rawlog.py knows v1 to v3")

    return dict(version=ver, word_bytes=wordlen,
                imu_odr_hz=imu_odr, sflp_odr_hz=sflp_odr,
                acc_sens_mg=acc_sens, gyr_sens_mdps=gyr_sens,
                t0_utc=t0, reading_id=reading_id, sync_bytes=sync_bytes)


def read(path):
    with open(path, "rb") as f:
        buf = f.read()
    cap = RawCapture(read_header(buf))
    cap.file_bytes = len(buf)
    wb, sb = cap.word_bytes, cap.sync_bytes

    acc_raw, gyr_raw, quat_raw = [], [], []
    acc_idx_at, acc_t_at = [], []      # sync points for the accel time axis
    acc_n = gyr_n = 0
    gyr_after_acc, quat_after_acc = [], []   # accel count at each gyro/quat word

    # An interrupted capture leaves the preallocated file with old data after the
    # last block. Parsing stops at a sync record that goes backwards in time or
    # accel count, or at a long run of unknown tags.
    MAX_UNKNOWN_IN_A_ROW = 64
    unknown_in_a_row = 0
    last_ms = last_an = None

    o = HEADER_BYTES
    n = len(buf)
    while o < n:
        tag = buf[o]
        if tag == SYNC_TAG:
            if o + sb > n:
                break                  # truncated tail
            t_us, a_n, ms, nw, flags = struct.unpack_from("<IIIHH", buf, o + 1)
            if last_ms is not None and (ms < last_ms or a_n < last_an):
                cap.tail_at = (o, f"sync went backwards: {ms} ms / accel {a_n} "
                                  f"after {last_ms} ms / accel {last_an}")
                break
            last_ms, last_an = ms, a_n
            unknown_in_a_row = 0
            cap.sync.append(dict(t_us=t_us, accel_n=a_n, millis=ms,
                                 n_words=nw, flags=flags,
                                 fifo_ovf=bool(flags & FLAG_FIFO_OVF),
                                 write_fail=bool(flags & FLAG_WRITE_FAIL)))
            if flags & FLAG_FIFO_OVF:
                cap.n_overflow += 1
            if flags & FLAG_WRITE_FAIL:
                cap.n_write_fail += 1
                if cap.first_write_fail_n is None:
                    cap.first_write_fail_n = a_n
            acc_idx_at.append(a_n)
            acc_t_at.append(t_us * 1e-6)
            o += sb
            continue
        if cap.version >= 3 and tag > 0x1F:
            if o + 2 > n:
                break
            ln = buf[o + 1]
            if o + 2 + ln > n:
                break                  # truncated tail
            if tag == GPS_TAG and ln >= GPS_PAYLOAD.size:
                g = GPS_PAYLOAD.unpack_from(buf, o + 2)
                if cap.gps and g[0] < cap.gps[-1][0]:
                    cap.tail_at = (o, f"gps went backwards: {g[0]} ms after "
                                      f"{cap.gps[-1][0]} ms")
                    break
                cap.gps.append(g)
            else:
                cap.unknown_tags[tag] = cap.unknown_tags.get(tag, 0) + 1
            unknown_in_a_row = 0
            o += 2 + ln
            continue
        if o + wb > n:
            break
        p = buf[o + 1:o + wb]
        if tag == TAG_ACC:
            acc_raw.append(struct.unpack("<hhh", p)); acc_n += 1
            unknown_in_a_row = 0
        elif tag == TAG_GYR:
            gyr_raw.append(struct.unpack("<hhh", p)); gyr_n += 1
            gyr_after_acc.append(acc_n)
            unknown_in_a_row = 0
        elif tag == TAG_SFLP:
            x, y, z = struct.unpack("<HHH", p)
            quat_raw.append((x, y, z))
            quat_after_acc.append(acc_n)
            unknown_in_a_row = 0
        else:
            cap.unknown_tags[tag] = cap.unknown_tags.get(tag, 0) + 1
            unknown_in_a_row += 1
            if unknown_in_a_row >= MAX_UNKNOWN_IN_A_ROW:
                cap.tail_at = (o - (unknown_in_a_row - 1) * wb,
                               f"{unknown_in_a_row} unknown tags in a row")
                break
        o += wb

    # Accel time axis: linear between the sync points (accel_n, t_us), and at the
    # nominal ODR outside them (np.interp would clamp).
    idx = np.arange(len(acc_raw), dtype=np.float64)
    nominal = 1.0 / float(cap.imu_odr_hz)
    if len(acc_idx_at) >= 2:
        xp = np.asarray(acc_idx_at, dtype=np.float64)
        fp = np.asarray(acc_t_at, dtype=np.float64)
        keep = np.diff(xp, prepend=xp[0] - 1) > 0      # sync records must increase
        xp, fp = xp[keep], fp[keep]
        acc_t = np.interp(idx, xp, fp)
        lo, hi = idx < xp[0], idx > xp[-1]
        acc_t[lo] = fp[0] + (idx[lo] - xp[0]) * nominal
        acc_t[hi] = fp[-1] + (idx[hi] - xp[-1]) * nominal
    else:
        acc_t = idx * nominal

    # GPS records onto the same axis, through the sync records' (millis, t_us);
    # 1 ms per ms outside them.
    if cap.gps and len(cap.sync) >= 2:
        ms = np.asarray([g[0] for g in cap.gps], dtype=np.float64)
        xp = np.asarray([s["millis"] for s in cap.sync], dtype=np.float64)
        fp = np.asarray([s["t_us"] for s in cap.sync], dtype=np.float64) * 1e-6
        keep = np.diff(xp, prepend=xp[0] - 1) > 0
        xp, fp = xp[keep], fp[keep]
        cap.gps_t = np.interp(ms, xp, fp)
        lo, hi = ms < xp[0], ms > xp[-1]
        cap.gps_t[lo] = fp[0] + (ms[lo] - xp[0]) * 1e-3
        cap.gps_t[hi] = fp[-1] + (ms[hi] - xp[-1]) * 1e-3
    elif cap.gps:
        cap.gps_t = np.full(len(cap.gps), np.nan)

    cap.acc = np.asarray(acc_raw, dtype=np.float64) * cap.acc_sens_mg
    cap.gyr = np.asarray(gyr_raw, dtype=np.float64) * cap.gyr_sens_mdps
    cap.acc_t = acc_t
    # Gyro and SFLP get the accel time at arrival, as the firmware pairs them.
    cap.gyr_t = (np.interp(np.asarray(gyr_after_acc, dtype=np.float64), idx, acc_t)
                 if len(gyr_after_acc) else np.empty(0))

    if quat_raw:
        q = np.asarray(quat_raw, dtype="<u2")
        xyz = q.view("<f2").astype(np.float64).reshape(-1, 3)
        # w = sqrt(1 - |xyz|²), as payloadToQuat.
        sumsq = np.clip((xyz ** 2).sum(axis=1), 0.0, 1.0)
        w = np.sqrt(1.0 - sumsq)
        cap.quat = np.column_stack([w, xyz])
        cap.quat_t = np.interp(np.asarray(quat_after_acc, dtype=np.float64), idx, acc_t)
    else:
        cap.quat = np.empty((0, 4))
        cap.quat_t = np.empty(0)
    return cap


def write_gps_csv(cap, path):
    """The GPS records as gps.csv, formatted as the firmware printed it before v3,
    with t_s (s on the accel time axis) as an extra last column. Returns the row count."""
    with open(path, "w", newline="") as f:
        f.write(",".join(GPS_COLUMNS) + "\n")
        for g, t in zip(cap.gps, cap.gps_t):
            (_, rel_ms, itow, hh, mm, ss, _fix, nsv, lat, lon, gspd, vn, ve, vd, head,
             sacc, hacc, vacc, pdop) = g
            f.write(f"{rel_ms},{hh * 1000000 + mm * 10000 + ss * 100},{itow},"
                    f"{lat * 1e-7:.6f},{lon * 1e-7:.6f},{gspd / 1000:.4f},"
                    f"{vn / 1000:.4f},{ve / 1000:.4f},{-vd / 1000:.4f},{head * 1e-5:.2f},"
                    f"{sacc / 1000:.2f},{hacc / 1000:.2f},{vacc / 1000:.2f},"
                    f"{pdop * 0.01:.2f},{nsv},{t:.6f}\n")
    return len(cap.gps)


def summary(cap):
    def rate(t):
        return len(t) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else float("nan")
    lines = [
        f"format v{cap.version}, reading {cap.reading_id}",
        f"cfg: imu_odr {cap.imu_odr_hz} Hz, sflp_odr {cap.sflp_odr_hz} Hz, "
        f"acc {cap.acc_sens_mg:g} mg/LSB, gyr {cap.gyr_sens_mdps:g} mdps/LSB",
        f"accel {len(cap.acc):7d} samples, measured {rate(cap.acc_t):7.2f} Hz",
        f"gyro  {len(cap.gyr):7d} samples, measured {rate(cap.gyr_t):7.2f} Hz",
        f"sflp  {len(cap.quat):7d} samples, measured {rate(cap.quat_t):7.2f} Hz",
        f"sync  {len(cap.sync):7d} drains, {cap.n_overflow} with FIFO overflow",
    ]
    if cap.version >= 3:
        lines.append(f"gps   {len(cap.gps):7d} records")
    if len(cap.sync) > 1:
        d = np.diff([s["millis"] for s in cap.sync])
        # FIFO headroom from the capture's word rate (accel + gyro + SFLP). The
        # FIFO holds 256 words (1536 B / 6 B, DS13476 section 6.10).
        words = 2.0 * cap.imu_odr_hz + cap.sflp_odr_hz
        headroom = 256.0 / words * 1000.0 if words > 0 else float("nan")
        over = int((d > headroom).sum())
        lines.append(f"drain: median {np.median(d):.0f} ms, max {d.max():.0f} ms "
                     f"(FIFO headroom {headroom:.0f} ms at {words:.0f} words/s)")
        if over:
            lines.append(f"       {over} drains ({100*over/len(d):.2f} %) over the "
                         f"headroom - samples may have been lost there")
    if cap.unknown_tags:
        lines.append(f"unknown tags: {cap.unknown_tags}")
    if cap.tail_at:
        o, reason = cap.tail_at
        lines.append(f"interrupted capture: read {o} B of {cap.file_bytes} - the rest "
                     f"is not this capture's ({reason}).")
        lines.append(f"                    The figures above cover the {len(cap.acc)} "
                     f"samples that were written, not the preallocated length.")
    if cap.n_write_fail:
        n = cap.trustworthy_upto()
        lines += [
            "",
            f"*** {cap.n_write_fail} blocks lost while writing (kRawFlagWriteFail).",
            f"*** The file is desynchronised from accel sample {cap.first_write_fail_n}"
            f" ({n / max(1, cap.imu_odr_hz):.1f} s in); everything after is misparsed.",
            "*** Use cap.trustworthy_upto() and discard the rest - do not trust the"
            " rates above, they are computed over the whole file.",
        ]
    return "\n".join(lines)
