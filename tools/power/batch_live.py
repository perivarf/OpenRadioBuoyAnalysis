"""Follows a batch log while it is being recorded: reports new tests, capture starts and SD-card swaps,
and decides when the batch is finished.

A marker is reported CONFIRM_S after it ends: by then the next test is underway, so a short spike in
front of the marker is not mistaken for a separate marker (as in batch_markers.find_markers).
"""

import numpy as np

from batch_markers import BATCHES, CARD_IDS, NAMES, WAVE_IDS, detect, distinct_ids, pick_capture

# Length of each test / capture in firmware (power_batch_test_ms in config.h)
TEST_S = 120.0
# Delay after a marker before it is reported
CONFIRM_S = 10.0


class BatchTracker:
    def __init__(self, batch, window, delta, auto_stop=True):
        self.batch = batch
        self.order = BATCHES.get(batch, [])
        self.window = window
        self.delta = delta
        self.auto_stop = auto_stop
        self.seen = {}          # test_start -> id, for markers that are reported
        self.caps_seen = set()
        self.stop_at = None     # time [s] the log can stop, once the last test is done

    def update(self, t, ta, ia):
        """Look for markers in the log so far (ta [s], ia [mA]) at time t, and report what is new."""
        w0, w1 = self.window
        seen = self.seen
        markers, caps = detect(ta, ia, self.delta)
        for m in markers:
            key = round(m["test_start"], 1)
            # The edge time can move a little as the log grows: the same marker is not new
            if any(abs(key - k) < 5.0 for k in seen) or t < m["test_start"] + CONFIRM_S:
                continue
            prev = list(seen.values())[-1] if seen else None
            seen[key] = m["id"]
            ids = distinct_ids(list(seen.values()))
            name = NAMES.get(m["id"], f"test{m['id']}")
            if m["id"] == prev:
                print(f"\n[{m['test_start']:7.1f} s] Kortet er byttet - test {m['id']} ({name}) starter nå")
                continue
            if prev is not None and (m["id"] in CARD_IDS) != (prev in CARD_IDS):
                print(f"\n{'=' * 60}\n  {'SETT INN' if m['id'] in CARD_IDS else 'TA UT'} "
                      f"SD-KORTET NÅ - test {m['id']} venter på det\n{'=' * 60}")
            pos = f"{len(ids)}/{len(self.order)}" if self.order else "?"
            print(f"\n[{m['test_start']:7.1f} s] Test {m['id']} ({name}) starter - batch {self.batch}, "
                  f"test {pos}")
            if m["id"] not in WAVE_IDS:
                print(f"            analysevindu {m['test_start'] + w0:.1f}-{m['test_start'] + w1:.1f} s")
            if self.order and ids != self.order[:len(ids)]:
                print(f"            ADVARSEL: rekkefølgen {ids} avviker fra {self.order}")

        cur = max(seen) if seen else None
        # Only strict candidates are reported live; the final analysis chooses among all of them.
        # Only in the wave tests: in the others, e.g. the GNSS startup before backup in test 1 can
        # look like a capture marker, but the analysis does not use capture there anyway.
        for c in (c["t"] for c in caps if c["strict"]):
            key = round(c, 1)
            # Reported after CONFIRM_S like the markers, so the marker in front is reported first
            if (cur is None or seen[cur] not in WAVE_IDS or c <= cur or t < c + CONFIRM_S
                    or any(abs(key - k) < 5.0 for k in self.caps_seen)):
                continue
            self.caps_seen.add(key)
            print(f"\n[{c:7.1f} s] Capture starter i test {seen[cur]} - "
                  f"analysevindu {c + w0:.1f}-{c + w1:.1f} s")

        # Done when the last test in the batch is over
        done = distinct_ids(list(seen.values()))
        if self.stop_at is None and self.auto_stop and self.order and len(done) == len(self.order):
            last_id, last_start = seen[cur], cur
            if last_id not in WAVE_IDS:
                self.stop_at = last_start + TEST_S + 5.0
            else:
                cap = pick_capture(caps, last_start, float("inf"), full=True)
                if cap is not None and cap["strict"]:
                    self.stop_at = cap["t"] + TEST_S + 20.0
                elif t > last_start + TEST_S + 15.0:
                    # With GPS tracking the capture marker is not visible; done when the board
                    # has settled in Stop2 for good (as in the ID markers, < 10 mA)
                    k = ta > t - 15.0
                    if np.max(ia[k]) < 10.0:
                        self.stop_at = t + 5.0
