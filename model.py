"""
ParkSense discrete-event simulation (pure Python + numpy, no SimPy needed).

One run = one day (open -> all vehicles gone) for one method and one demand sample.
Methods (see config.METHODS):
  CONV      gate + drivers search area by area (no information)
  GUIDE     gate + space-level guidance lights (delayed info, no lock) -> drivers may race for a space
  SPLIT     gate + fixed reserved zone near the mall; walk-ins use guidance lights outside the zone
  NEAREST   ParkSense hardware (gate-free, lock, space check) but always assigns the free space nearest the mall
  PARKSENSE proposed: gate-free entrance, dual-mode weighted scoring + congestion penalty (= parking.py), space check
Common random numbers: the same demand list is used for every method in one replication.
"""
import heapq
import math
from collections import deque

import numpy as np

FREE, RESERVED, OCCUPIED = 0, 1, 2


# ============================================================ layout
def build_layout(cfg, seed=0):
    rng = np.random.default_rng(10_000 + seed)
    ax, ay, nr, ns = cfg["areas_x"], cfg["areas_y"], cfg["rows_per_area"], cfg["spaces_per_row"]
    aw, ah = ns * 4.5 + 5, nr * 8 + 5           # area width / depth in metres
    X, Y, AREA, POS = [], [], [], []
    for gy in range(ay):
        for gx in range(ax):
            a = gy * ax + gx
            for j in range(nr):
                for i in range(ns):
                    X.append(gx * aw + 2.5 + i * 4.5)
                    Y.append(5 + gy * ah + 4 + j * 8)
                    AREA.append(a)
                    # easy-parking position score: row ends 9, next to a pillar 3, others 6   [ASSUMED]
                    POS.append(9 if i in (0, ns - 1) else (3 if i % 4 == 2 else 6))
    X, Y, AREA, POS = map(np.array, (X, Y, AREA, POS))
    n = len(X)
    W = ax * aw
    mall = (W / 2, 0.0)                          # mall door: middle of the top edge
    ent = (0.0, 5 + ay * ah + 5)                 # vehicle entrance: bottom-left corner
    WALK = np.abs(X - mall[0]) + np.abs(Y - mall[1])
    DRIVE = np.abs(X - ent[0]) + np.abs(Y - ent[1])
    na = ax * ay
    cen = np.array([[X[AREA == a].mean(), Y[AREA == a].mean()] for a in range(na)])
    awalk = np.array([WALK[AREA == a].mean() for a in range(na)])

    def to10(v):                                 # nearer = higher, scaled 1..10 like the prototype sheet
        return 1 + 9 * (v.max() - v) / (v.max() - v.min())
    AE = to10(awalk)[AREA]
    SE = to10(WALK)
    OK = np.ones(n, bool)
    OK[rng.choice(n, int(round(cfg["broken_ratio"] * n)), replace=False)] = False
    order = np.argsort(WALK)
    ZONE = np.zeros(n, bool)
    ok_sorted = [k for k in order if OK[k]]
    ZONE[ok_sorted[: int(round(cfg["split_zone_share"] * len(ok_sorted)))]] = True
    return dict(n=n, na=na, X=X, Y=Y, AREA=AREA, POS=POS.astype(float), WALK=WALK, DRIVE=DRIVE,
                AE=AE, SE=SE, OK=OK, ZONE=ZONE, cen=cen, awalk=awalk, ent=ent, mall=mall,
                names=[chr(65 + a) for a in range(na)])


# ============================================================ demand
def _lognorm(rng, mean, sd, size=None):
    s2 = math.log(1 + (sd / mean) ** 2)
    return rng.lognormal(math.log(mean) - s2 / 2, math.sqrt(s2), size)


def generate_demand(cfg, capacity, seed):
    rng = np.random.default_rng(seed)
    prof = np.array(cfg["profile_" + cfg["profile"]], float)
    hours = cfg["close_h"] - cfg["open_h"]
    lam_peak = cfg["demand_level"] * capacity / (cfg["dwell_mean_min"] / 60) / 3600   # veh/s at profile 1.0
    lam_max = lam_peak * prof.max()
    t, out = 0.0, []
    pb = np.array(cfg["patience_probs"], float); pb /= pb.sum()
    while True:                                   # non-homogeneous Poisson by thinning
        t += rng.exponential(1 / lam_max)
        if t >= hours * 3600:
            break
        if rng.random() > prof[int(t // 3600)] / prof.max():
            continue
        b = cfg["patience_bins_min"][rng.choice(len(pb), p=pb)]
        is_res = rng.random() < cfg["res_share"]
        v = dict(id=len(out), planned=t, is_res=is_res,
                 dwell=_lognorm(rng, cfg["dwell_mean_min"], cfg["dwell_sd_min"]) * 60,
                 patience=rng.uniform(*b) * 60,
                 gate_s=_lognorm(rng, *cfg["gate_service_s"]),
                 anpr_fail=rng.random() < cfg["anpr_fail"],
                 verify_fail=rng.random() < cfg["verify_fail"],
                 u=rng.random(8))
        if is_res:
            v["lead"] = rng.uniform(*cfg["lead_min"]) * 60
            v["no_show"] = rng.random() < cfg["no_show"]
            v["arrival"] = max(0.0, t + rng.normal(0, cfg["arrive_sd_min"] * 60))
        else:
            v["lead"], v["no_show"], v["arrival"] = 0.0, False, t
        out.append(v)
    return out


# ============================================================ simulator
class Sim:
    def __init__(self, cfg, method, layout, demand, seed=0):
        self.c, self.m, self.L = cfg, method, layout
        self.rng = np.random.default_rng(seed + 777)
        self.v_ms = cfg["speed_kmh"] / 3.6
        n = layout["n"]
        self.status = np.zeros(n, np.int8)
        self.owner = -np.ones(n, int)
        self.used = np.zeros(layout["na"], int)
        self.total = np.bincount(layout["AREA"][layout["OK"]], minlength=layout["na"]).astype(float)
        self.snap = self.status.copy()
        self.ev, self.seq, self.now = [], 0, 0.0
        self.queue, self.bay, self.busy = deque(), deque(), False
        self.assign_free_at, self.pending, self.claims = 0.0, 0, 0
        self.V = [dict(v) for v in demand]
        for v in self.V:
            v.update(state="new", space=-1, booked=-1, res_ok=False, conflicts=0, drive_m=0.0, idle_s=0.0,
                     t_q=None, t_exit=None, t_park=None, balked=False, tried=set(), hold_s=0.0)
        self.samples = []
        self.maxq, self._wake = 0, False
        o = cfg.get("outage")                              # (start hour, minutes): gate down / entrance camera down
        self.out_win = ((o[0] - cfg["open_h"]) * 3600, (o[0] - cfg["open_h"]) * 3600 + o[1] * 60) if o and o[1] > 0 else None
        self.dups = 0
        self.stolen = 0

    # ---------------- event helpers
    def at(self, t, fn, *a):
        self.seq += 1
        heapq.heappush(self.ev, (t, self.seq, fn, a))

    def run(self):
        c = self.c
        gated_res = self.m in ("SPLIT", "NEAREST", "PARKSENSE")
        for v in self.V:
            if v["is_res"] and gated_res:
                self.at(v["planned"] - v["lead"], self.book, v)      # may be before opening (negative time)
            if not v["no_show"]:
                self.at(v["arrival"], self.arrive, v)
        self.at(0.0, self.sample)
        if self.m in ("GUIDE", "SPLIT"):
            self.at(0.0, self.refresh_snap)
        while self.ev:
            t, _, fn, a = heapq.heappop(self.ev)
            self.now = t
            fn(*a)
        return self

    # ---------------- space state
    def set_status(self, s, st, vid=-1):
        a = self.L["AREA"][s]
        was = self.status[s]
        if was == FREE and st != FREE:
            self.used[a] += 1
        elif was != FREE and st == FREE:
            self.used[a] -= 1
        self.status[s] = st
        self.owner[s] = vid if st != FREE else -1

    def free_mask(self, zone=None):
        m = self.L["OK"] & (self.status == FREE)
        if zone is True:
            m &= self.L["ZONE"]
        elif zone is False:
            m &= ~self.L["ZONE"]
        return m

    def best_space(self, mode):
        """PARKSENSE: same formula as parking.py calculate_candidate_slots(). NEAREST: min walking distance."""
        L, c = self.L, self.c
        m = self.free_mask()
        if not m.any():
            return -1
        if self.m == "NEAREST":
            w = np.where(m, L["WALK"], np.inf)
            return int(np.argmin(w))
        s = np.where(m, self.scores(mode), -np.inf)
        return int(np.argmax(s))                       # first max = same tie rule as Python's stable sort

    def scores(self, mode):
        L, c = self.L, self.c
        W1, W2, W3, W4 = c["w_reservation"] if mode == "reservation" else c["w_walkin"]
        cong = self.used / np.maximum(self.total, 1)
        ca = cong[L["AREA"]]
        s = W1 * (1 - ca) + W2 * L["AE"] / 10 + W3 * L["SE"] / 10 + W4 * L["POS"] / 10
        return np.where(ca > c["penalty_threshold"], s * c["penalty_factor"], s)

    # ---------------- assignment server (ParkSense / NEAREST)
    def request(self, v, mode, cb):
        lat = self.c["assign_latency_s"]
        self.pending += 1
        if self.c["assign_lock"]:
            start = max(self.now, self.assign_free_at)
            self.assign_free_at = start + lat
            self.at(start + lat, self._assign_locked, v, mode, cb)
        else:
            s = self.best_space(mode)                   # read now, write after latency (no lock)
            self.at(self.now + lat, self._assign_write, v, mode, cb, s)

    def _assign_locked(self, v, mode, cb):
        self.pending -= 1
        s = self.best_space(mode)
        if s >= 0:
            self.set_status(s, RESERVED, v["id"])
        cb(v, s)

    def _assign_write(self, v, mode, cb, s):
        self.pending -= 1
        if s >= 0 and self.status[s] != FREE:          # someone else wrote it first -> duplicate assignment
            self.dups += 1
            v["conflicts"] += 1
            self.at(self.now + self.c["conflict_extra_s"], self.request, v, mode, cb)
            return
        if s >= 0:
            self.set_status(s, RESERVED, v["id"])
        cb(v, s)

    # ---------------- reservation
    def book(self, v):
        if self.m == "SPLIT":
            m = self.free_mask(zone=True)
            s = int(np.argmin(np.where(m, self.L["WALK"], np.inf))) if m.any() else -1
            if s >= 0:
                self.set_status(s, RESERVED, v["id"])
            self.booked(v, s)
        else:
            self.request(v, "reservation", self.booked)

    def booked(self, v, s):
        v["booked"] = s
        v["t_book"] = self.now
        if s >= 0:
            self.at(v["planned"] + self.c["grace_min"] * 60, self.release_hold, v)

    def release_hold(self, v):
        s = v["booked"]
        if s >= 0 and self.owner[s] == v["id"] and self.status[s] == RESERVED and v["state"] in ("new", "queue", "bay"):
            v["hold_s"] += self.now - v["t_book"]
            v["booked"] = -1
            self.set_status(s, FREE)
            self.space_freed()

    # ---------------- entrance
    def arrive(self, v):
        v["t_q"] = self.now
        v["state"] = "queue"
        self.queue.append(v)
        self.maxq = max(self.maxq, len(self.queue) + len(self.bay))
        self.at(self.now + v["patience"], self.renege, v)
        self.serve()

    def has_booking(self, v):
        s = v["booked"]
        return s >= 0 and self.owner[s] == v["id"]

    def outage_now(self):
        return self.out_win is not None and self.out_win[0] <= self.now < self.out_win[1]

    def _wake_up(self):
        self._wake = False
        self.space_freed()

    def serve(self):
        m = self.m
        while not self.busy and self.queue:
            v = self.queue[0]
            if self.outage_now():
                if m in ("CONV", "GUIDE", "SPLIT"):          # gate broken: nobody can enter
                    if not self._wake:
                        self._wake = True
                        self.at(self.out_win[1], self._wake_up)
                    return
                if not self.has_booking(v) and not self.c.get("outage_fallback", False):
                    # as built: no plate read -> walk-ins cannot be assigned; they wait (bay first, then the lane)
                    if len(self.bay) < self.c["bay_size"]:
                        self.queue.popleft()
                        v["state"] = "bay"
                        v["idle_s"] += self.now - v["t_q"]
                        v["t_bay"] = self.now
                        self.bay.append(v)
                        continue
                    if not self._wake:
                        self._wake = True
                        self.at(self.out_win[1], self._wake_up)
                    return
                # reserved drivers (space number from the app; the space camera still checks the plate),
                # or everyone when a fallback mode lets vehicles in to search
                self.queue.popleft()
                self.busy = True
                v["state"] = "entering"
                v["fallback"] = True
                v["idle_s"] += self.now - v["t_q"]
                self.at(self.now + self.c["gatefree_headway_s"], self.end_fallback, v)
                continue
            if m == "CONV":
                ok = True
            elif m == "GUIDE":
                ok = self.free_mask().any()
            elif m == "SPLIT":
                ok = self.has_booking(v) or self.free_mask(zone=False).any()
            else:                                           # gate-free methods
                free_n = int(self.free_mask().sum()) - self.pending - self.claims - len(self.bay)
                ok = self.has_booking(v) or free_n > 0
                if not ok and len(self.bay) < self.c["bay_size"]:
                    self.queue.popleft()                    # walk-in waits in the bay, lane stays open
                    v["state"] = "bay"
                    v["idle_s"] += self.now - v["t_q"]
                    v["t_bay"] = self.now
                    self.bay.append(v)
                    continue
            if not ok:
                qf = self.c.get("queue_fallback", 0)
                if qf and m in ("NEAREST", "PARKSENSE") and len(self.queue) >= qf:
                    # v3: queue too long -> admit the head vehicle to search freely (fallback mode)
                    self.queue.popleft()
                    self.busy = True
                    v["state"] = "entering"
                    v["fallback"] = True
                    v["idle_s"] += self.now - v["t_q"]
                    self.at(self.now + self.c["gatefree_headway_s"], self.end_fallback, v)
                    continue
                return                                      # head of lane blocked until a space frees
            self.queue.popleft()
            self.busy = True
            v["state"] = "entering"
            v["idle_s"] += self.now - v["t_q"]
            if m in ("NEAREST", "PARKSENSE"):
                svc = self.c["gatefree_headway_s"] + (self.c["anpr_fail_stop_s"] if v["anpr_fail"] else 0)
                v["idle_s"] += self.c["anpr_fail_stop_s"] if v["anpr_fail"] else 0
                self.gatefree_pass(v)
            else:
                svc = v["gate_s"]
                v["idle_s"] += svc
            self.at(self.now + svc, self.end_service, v)

    def end_service(self, v):
        self.busy = False
        if self.m not in ("NEAREST", "PARKSENSE"):
            v["t_exit"] = self.now
            v["pos"] = self.L["ent"]
            if self.m == "SPLIT" and self.has_booking(v):
                self.drive_to_booked(v, v["booked"])
            elif self.m == "CONV":
                v["area_order"] = list(np.argsort(self.L["awalk"]))
                v["k"] = 0
                self.conv_next_area(v)
            else:
                self.guide_choose(v)
        self.serve()

    def end_fallback(self, v):
        self.busy = False
        v["t_exit"] = self.now
        v["pos"] = self.L["ent"]
        if self.has_booking(v):                              # reserved driver knows the space number from the app
            self.drive_to_booked(v, v["booked"])
        else:
            v["area_order"] = list(np.argsort(self.L["awalk"]))
            v["k"] = 0
            self.conv_next_area(v)
        self.serve()

    def renege(self, v):
        if v["state"] == "queue":
            self.queue.remove(v)
        elif v["state"] == "bay":
            self.bay.remove(v)
        else:
            return
        v["state"] = "left"
        v["balked"] = True
        v["idle_s"] += self.now - v.get("t_bay", v["t_q"])
        if self.has_booking(v):
            self.release_hold(v)
        self.serve()

    def space_freed(self):
        # bay first (gate-free methods), then the lane
        while self.bay and int(self.free_mask().sum()) - self.pending - self.claims > 0 and not self.outage_now():
            v = self.bay.popleft()
            v["state"] = "entering"
            v["idle_s"] += self.now - v["t_bay"]
            v["t_q_bay_done"] = self.now
            self.gatefree_pass(v)
        self.serve()

    # ---------------- gate-free entrance (NEAREST / PARKSENSE)
    def gatefree_pass(self, v):
        v["t_bump"] = self.now
        lane_t = self.c["lane_m"] / self.v_ms
        if self.has_booking(v):
            v["res_ok"] = True
            v["hold_s"] += self.now - v["t_book"]
            self.at(self.now + lane_t, self.leave_led, v, v["booked"])
            return
        v["booked"] = -1
        self.claims += 1                                    # plate seen, request not sent yet
        self.at(self.now + self.c["anpr_latency_s"], self._send_walkin, v)

    def _send_walkin(self, v):
        self.claims -= 1
        self.request(v, "walkin", self.walkin_assigned)

    def walkin_assigned(self, v, s):
        if s < 0:                                            # lot became full meanwhile -> wait in the bay
            v["state"] = "bay"
            v["t_bay"] = self.now
            self.bay.append(v)
            return
        lane_t = self.c["lane_m"] / self.v_ms
        t_led = v["t_bump"] + lane_t
        if self.now > t_led:                                 # result late -> driver stops at the LED screen
            v["idle_s"] += self.now - t_led
            v["led_stop"] = self.now - t_led
        self.at(max(self.now, t_led), self.leave_led, v, s)

    def leave_led(self, v, s):
        v["t_exit"] = self.now
        v["state"] = "driving"
        L = self.L
        p = self.c.get("noncomply", 0.0)
        if p > 0 and self.rng.random() < p:
            # v3: a non-compliant driver ignores the assigned space and heads for the empty-looking space
            # nearest the mall (FREE, or RESERVED for someone who has not parked yet)
            m = L["OK"] & (self.status != OCCUPIED)
            m[s] = False
            m &= L["WALK"] < L["WALK"][s]
            if m.any():
                t = int(np.argmin(np.where(m, L["WALK"], np.inf)))
                v["noncomp"] = True
                d = L["DRIVE"][t]
                v["drive_m"] += d
                self.at(self.now + d / self.v_ms, self.wrong_space, v, s, t)
                return
        d = L["DRIVE"][s]
        v["drive_m"] += d
        self.at(self.now + d / self.v_ms, self.at_space_locked, v, s)

    def wrong_space(self, v, s, t):
        """v3 ablation: what happens when a driver tries a space that is not the assigned one."""
        c, L = self.c, self.L
        back = abs(L["X"][t] - L["X"][s]) + abs(L["Y"][t] - L["Y"][s])
        if c.get("barrier", True) or self.status[t] == OCCUPIED:
            # barrier stays up (plate not bound to t), or t already taken: drive on to the assigned space
            wait = c["verify_s"] if c.get("barrier", True) else 0.0
            v["idle_s"] += wait
            v["refused"] = v.get("refused", 0) + (1 if c.get("barrier", True) else 0)
            v["drive_m"] += back
            self.at(self.now + wait + back / self.v_ms, self.at_space_locked, v, s)
            return
        # no barrier: the driver simply parks in t; the space bound to v stays reserved until the grace time ends
        if self.status[t] == RESERVED and self.owner[t] >= 0:
            self.stolen += 1
        self.park(v, t, c["maneuver_s"])
        self.at(self.now + c["grace_min"] * 60, self.release_orphan, v, s)

    def release_orphan(self, v, s):
        if self.status[s] == RESERVED and self.owner[s] == v["id"]:
            self.set_status(s, FREE)
            self.space_freed()

    def at_space_locked(self, v, s):
        c = self.c
        if self.m in ("NEAREST", "PARKSENSE") and self.status[s] == OCCUPIED and self.owner[s] != v["id"]:
            # v3: the bound space was taken by a non-compliant driver (possible only without a barrier)
            v["conflicts"] += 1
            v["res_ok"] = False
            v["pos"] = (self.L["X"][s], self.L["Y"][s])
            self.at(self.now + c["conflict_extra_s"], self.request, v, "walkin", self.reassigned)
            return
        t = c["verify_s"] + (c["verify_retry_s"] if v["verify_fail"] else 0)
        if self.m == "SPLIT" or not c.get("barrier", True):
            t = 0.0
        self.park(v, s, t + c["maneuver_s"])

    def reassigned(self, v, s):
        if s < 0:                                           # lot full: try again later
            self.at(self.now + self.c["cruise_retry_s"], self.request, v, "walkin", self.reassigned)
            return
        L = self.L
        d = abs(L["X"][s] - v["pos"][0]) + abs(L["Y"][s] - v["pos"][1])
        v["drive_m"] += d
        self.at(self.now + d / self.v_ms, self.at_space_locked, v, s)

    def drive_to_booked(self, v, s):                         # SPLIT reserved vehicle
        v["res_ok"] = True
        v["hold_s"] += self.now - v["t_book"]
        v["state"] = "driving"
        d = abs(self.L["X"][s] - v["pos"][0]) + abs(self.L["Y"][s] - v["pos"][1])
        v["drive_m"] += d
        self.at(self.now + d / self.v_ms, self.at_space_locked, v, s)

    # ---------------- parking / leaving
    def park(self, v, s, delay):
        self.set_status(s, OCCUPIED, v["id"])
        v["space"] = s
        v["state"] = "parked"
        v["t_park"] = self.now + delay
        self.at(v["t_park"] + v["dwell"], self.depart, v)

    def depart(self, v):
        self.set_status(v["space"], FREE)
        v["state"] = "gone"
        self.space_freed()

    def give_up_inside(self, v):
        v["state"] = "left"
        v["balked"] = True

    # ---------------- CONV: search area by area
    def conv_next_area(self, v):
        if self.now - v["t_q"] > v["patience"] and v["k"] >= self.L["na"]:
            return self.give_up_inside(v)
        a = v["area_order"][v["k"] % self.L["na"]]
        v["k"] += 1
        cx, cy = self.L["cen"][a]
        d = abs(cx - v["pos"][0]) + abs(cy - v["pos"][1])
        v["drive_m"] += d
        v["pos"] = (cx, cy)
        extra = self.c["cruise_retry_s"] if v["k"] > self.L["na"] else 0.0
        self.at(self.now + d / self.v_ms + extra, self.conv_scan, v, a)

    def conv_scan(self, v, a):
        idx = np.flatnonzero(self.L["AREA"] == a)
        idx = idx[self.rng.permutation(len(idx))]             # random scanning order
        free = self.L["OK"][idx] & (self.status[idx] == FREE)
        pitch = self.c["space_pitch_m"]
        if free.any():
            k = int(np.argmax(free))
            d = (k + 1) * pitch
            v["drive_m"] += d
            self.at(self.now + d / self.v_ms, self.conv_claim, v, a, int(idx[k]))
        else:
            d = len(idx) * pitch
            v["drive_m"] += d
            self.at(self.now + d / self.v_ms, self.conv_next_area, v)

    def conv_claim(self, v, a, s):
        if self.status[s] == FREE:
            self.park(v, s, self.c["maneuver_s"])
        else:
            v["conflicts"] += 1
            self.conv_scan(v, a)

    # ---------------- GUIDE / SPLIT walk-ins: delayed lights, no lock
    def refresh_snap(self):
        self.snap = self.status.copy()
        if self.c["info_period_s"] <= 0:                     # real-time lights: guide_choose reads live status
            return
        if self.ev:
            self.at(self.now + self.c["info_period_s"], self.refresh_snap)

    def guide_choose(self, v):
        L = self.L
        snap = self.status if self.c["info_period_s"] <= 0 else self.snap
        m = L["OK"] & (snap == FREE)
        if self.m == "SPLIT":
            m &= ~L["ZONE"]
        if v["tried"]:
            m[list(v["tried"])] = False
        if not m.any():
            if self.now - v["t_q"] > v["patience"]:
                return self.give_up_inside(v)
            v["tried"].clear()
            v["drive_m"] += 100.0                             # one loop while waiting   [ASSUMED]
            self.at(self.now + self.c["cruise_retry_s"], self.guide_choose, v)
            return
        cand = np.flatnonzero(m)
        k = min(self.c["guide_topk"], len(cand))
        top = cand[np.argsort(L["WALK"][cand])[:k]]
        s = int(top[self.rng.integers(k)])
        d = abs(L["X"][s] - v["pos"][0]) + abs(L["Y"][s] - v["pos"][1])
        v["drive_m"] += d
        self.at(self.now + d / self.v_ms, self.guide_arrive, v, s)

    def guide_arrive(self, v, s):
        v["pos"] = (self.L["X"][s], self.L["Y"][s])
        if self.status[s] == FREE:
            self.park(v, s, self.c["maneuver_s"])
        else:
            v["conflicts"] += 1
            v["tried"].add(s)
            self.guide_choose(v)

    # ---------------- sampling
    def sample(self):
        occ = self.used / np.maximum(self.total, 1)
        self.samples.append((self.now, occ.copy(), self.used.sum() / self.total.sum()))
        if self.now < (self.c["close_h"] - self.c["open_h"] + 4) * 3600:
            self.at(self.now + self.c["sample_s"], self.sample)

    # ---------------- results
    def kpis(self):
        c, L = self.c, self.L
        arrived = [v for v in self.V if not v["no_show"]]
        parked = [v for v in arrived if v["t_park"] is not None]
        ent = np.array([v["t_exit"] - v["t_q"] for v in parked])
        srch = np.array([v["t_park"] - v["t_exit"] for v in parked])
        walk = np.array([L["WALK"][v["space"]] for v in parked])
        drive_km = sum(v["drive_m"] for v in arrived) / 1000
        idle = sum(v["idle_s"] for v in arrived)
        res_req = [v for v in arrived if v["is_res"]]
        w0, w1 = (h - c["open_h"] for h in c["kpi_window_h"])
        win = [(occ, tot) for t, occ, tot in self.samples if w0 * 3600 <= t <= w1 * 3600]
        imb = float(np.mean([occ.std() for occ, _ in win])) if win else float("nan")
        maxarea = float(np.mean([occ.max() for occ, _ in win])) if win else float("nan")
        peak = max(tot for _, _, tot in self.samples)
        hot_h = sum(int((occ >= 0.9).sum()) for _, occ, _ in self.samples) * c["sample_s"] / 3600   # area-hours >= 90% full
        res_walk = [L["WALK"][v["space"]] for v in parked if v["is_res"]]
        wi_walk = [L["WALK"][v["space"]] for v in parked if not v["is_res"]]
        n = len(arrived)
        gated_res = self.m in ("SPLIT", "NEAREST", "PARKSENSE")
        return {
            "method": self.m,
            "vehicles": n,
            "parked_pct": 100 * len(parked) / n,
            "balk_pct": 100 * sum(v["balked"] for v in arrived) / n,
            "entrance_s_mean": ent.mean(), "entrance_s_p95": np.percentile(ent, 95),
            "search_s_mean": srch.mean(), "search_s_p95": np.percentile(srch, 95),
            "time_to_park_s_mean": (ent + srch).mean(),
            "walk_m_mean": walk.mean(),
            "drive_m_per_veh": 1000 * drive_km / n,
            "conflicts_per_100": 100 * sum(v["conflicts"] for v in arrived) / n,
            "stolen_per_100": 100 * self.stolen / n,
            "refused_per_100": 100 * sum(v.get("refused", 0) for v in arrived) / n,
            "noncomp_pct": 100 * sum(1 for v in arrived if v.get("noncomp")) / n,
            "fallback_pct": 100 * sum(1 for v in arrived if v.get("fallback")) / n,
            "dup_assign_per_100": 100 * self.dups / n,
            "res_success_pct": (100 * sum(v["res_ok"] for v in res_req) / len(res_req)) if (gated_res and res_req) else float("nan"),
            "hold_space_h": sum(v["hold_s"] for v in self.V) / 3600 if gated_res else 0.0,
            "area_imbalance_sd": imb,
            "max_area_occ": maxarea,
            "hot_area_h": hot_h,
            "walk_res_m": float(np.mean(res_walk)) if res_walk else float("nan"),
            "walk_walkin_m": float(np.mean(wi_walk)) if wi_walk else float("nan"),
            "peak_occ": peak,
            "co2_kg": drive_km * c["ef_kg_per_km"] + idle * c["idle_kg_per_s"],
            "led_stop_pct": 100 * sum(1 for v in arrived if v.get("led_stop", 0) > 0) / n,
            "max_queue": self.maxq,
            **self._window_kpis(arrived),
        }


def _window_kpis_impl(self, arrived):
    w = self.c.get("outage_kpi")                           # (start hour, minutes) window of arrivals to report
    if not w:
        return {}
    a = (w[0] - self.c["open_h"]) * 3600
    vs = [v for v in arrived if a <= v["t_q"] < a + w[1] * 60]
    pk = [v for v in vs if v["t_park"] is not None]
    return {"win_n": len(vs),
            "win_time_to_park_s": float(np.mean([v["t_park"] - v["t_q"] for v in pk])) if pk else float("nan"),
            "win_p95_time_to_park_s": float(np.percentile([v["t_park"] - v["t_q"] for v in pk], 95)) if pk else float("nan"),
            "win_balk_pct": 100 * sum(v["balked"] for v in vs) / max(len(vs), 1),
            "win_drive_m": float(np.mean([v["drive_m"] for v in vs])) if vs else float("nan")}


Sim._window_kpis = _window_kpis_impl


def run_one(cfg, method, rep):
    layout = build_layout(cfg, seed=0)
    cap = int(layout["OK"].sum())
    demand = generate_demand(cfg, cap, seed=rep)
    return Sim(cfg, method, layout, demand, seed=rep).run()
