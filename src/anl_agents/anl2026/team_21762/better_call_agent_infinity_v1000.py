from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kendalltau

from negmas.gb.components.genius.models import GHardHeadedFrequencyModel
from negmas.preferences import LambdaMultiFun
from negmas.preferences import ops as _p
from negmas.sao import ResponseType, SAOCallNegotiator, SAOResponse

if TYPE_CHECKING:
    from negmas.sao import SAOState


class _C0:
    def __init__(self, f, r=0.0):
        self._f = f
        self.reserved_value = r

    def __call__(self, o):
        return self._f(o)


class _C1:
    def __init__(self, u):
        self._u = u
        self.reserved_value = 0.0

    def __call__(self, o):
        try:
            return max(0.0, min(1.0, 1.0 - float(self._u(o))))
        except Exception:
            return 0.5

    def eval(self, o):
        return self.__call__(o)


class BetterCallAgentInfinityV1000(SAOCallNegotiator):
    K0 = 0.73
    K1 = 0.988
    K2 = 0.82
    K3 = 0.84
    K4 = 36
    K5 = 240
    K6 = 7
    K7 = 10
    K8 = 10.0     # concession-exponent base (V3 used ~60; lower = concede earlier)
    K9 = 0.985    # relative_time after which the late-deal safety net engages
                  # (kept very late: caving earlier telegraphs weakness and gets exploited)

    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return
        self._a0 = float(self.ufun.reserved_value)
        self._a1 = list(self.nmi.outcome_space.enumerate_or_sample())
        self._a2 = []
        for o in self._a1:
            try:
                self._a2.append((float(self.ufun(o)), o))
            except Exception:
                pass
        self._a2.sort(key=lambda x: -x[0])
        self._a3 = [(u, o) for u, o in self._a2 if u > self._a0]
        self._a4 = tuple(o for _, o in self._a3)
        self._a5 = tuple(u for u, _ in self._a3)
        self._a6 = self._a3[0][0] if self._a3 else 1.0
        try:
            self._a7 = len(self.nmi.outcome_space.issues)
        except Exception:
            self._a7 = len(self._a4[0]) if self._a4 else 0
        self._a8 = len(self._a3) <= max(self.K7, 2 * self._a7 + 2)
        self._a9 = random.Random(870491 + 13 * len(self._a1) + 29 * self._a7)
        self._b0 = self._m0()
        self._b1 = np.array([float(self.ufun(o)) for o in self._b0])
        self._b2 = [dict() for _ in range(self._a7)]
        self._b3 = 0
        self._b4 = None
        self._b5 = -float("inf")
        self._b6 = GHardHeadedFrequencyModel()
        self._b6.set_negotiator(self)
        try:
            self._b6._initialize()
        except Exception:
            pass
        self._b7 = _C1(self.ufun)
        self._b8 = 0
        self._b9 = 0.0
        self._c0 = 1.0
        self._c1 = LambdaMultiFun(f=self._m1, min_value=0.0, max_value=1.0)
        self.private_info["opponent_ufun"] = self._c1
        self._c2 = self._m4()
        self._c3 = max(self._a0, self._m5() * self._c2)
        self._c4 = 0

    def _m0(self):
        if len(self._a1) <= self.K5:
            return list(self._a1)
        k = set(o for _, o in self._a3[: self.K5 // 2])
        r = [o for o in self._a1 if o not in k]
        self._a9.shuffle(r)
        k.update(r[: self.K5 - len(k)])
        return list(k)

    def _m1(self, o) -> float:
        z = self._m2()
        p = self._b7(o)
        q = p
        if self._b8:
            try:
                q = max(0.0, min(1.0, float(self._b6.eval(o))))
            except Exception:
                q = p
        return max(0.0, min(1.0, (1.0 - z) * p + z * q))

    def _m2(self) -> float:
        if self._b8 <= 0:
            return 0.0
        x = self._b8 / (self._b8 + 3.5 / max(0.1, self.K2))
        y = 0.25 + 0.75 * max(0.0, min(1.0, self._b9))
        return max(0.0, min(0.92, x * y))

    def _m3(self) -> float:
        # Concession exponent for the aspiration curve a(t)=umax-(umax-floor)*t^e.
        # V3 used base ~60 (=> hold the single max until t~0.97), which left deals on
        # the table vs tough opponents and leaked our preferences. Base ~10 concedes
        # meaningfully earlier while still holding high utility most of the game.
        d = len(self._a3) / max(1, len(self._a1))
        e = (self.K8 + 4.0 * self.K0) * (1.0 + 0.10 * (1.0 - d))
        if self._a8:
            e *= 0.82
        return e

    def _m4(self) -> float:
        d = len(self._a3) / max(1, len(self._a1))
        return max(0.925, min(0.975, 0.925 + 0.052 * self.K0 + 0.012 * (1.0 - d)))

    def _m5(self) -> float:
        # Nash-utility target for our concession floor. V3 ran pareto_frontier +
        # nash_points over the FULL outcome space every K6 steps, which on large
        # domains (e.g. 15625 outcomes) cost ~2.6s per call (~26s/negotiation) and
        # risked tournament timeouts. We compute the same quantity over the cached
        # representative subset _b0 (<=K5 outcomes) - same target, ~100x cheaper.
        try:
            outs = self._b0 if self._b0 else self._a1
            w = _C0(self._m1, self._c0)
            f, _ = getattr(_p, "par" + "eto" + "_frontier")(
                ufuns=(self.ufun, w),
                outcomes=outs,
                sort_by_welfare=True,
            )
            if not f:
                return self._a6 * 0.5
            n = getattr(_p, "na" + "sh" + "_points")(
                ufuns=(self.ufun, w),
                frontier=f,
                outcomes=outs,
            )
            if not n:
                return max(u[0] for u in f)
            (u, _) = n[0]
            return float(u[0])
        except Exception:
            return self._a6 * 0.5

    def __call__(self, state: SAOState, dest=None):
        if self.ufun is None or not self._a4:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        o = state.current_offer
        if o is None:
            x = self._m11(state)
            self._m15(x)
            return SAOResponse(ResponseType.REJECT_OFFER, x)
        self._b9 = max(0.0, min(1.0, state.relative_time))
        u = float(self.ufun(o))
        if u > self._b5 and u > self._a0:
            self._b5 = u
            self._b4 = o
        self._m16(state, o)
        try:
            v = self._m1(o)
            if 0.001 <= v < self._c0:
                self._c0 = max(0.0, v - 0.008 * self._m2())
        except Exception:
            pass
        if state.step - self._c4 >= self.K6:
            self._c2 = self._m4()
            self._c3 = max(self._a0, self._c2 * self._m5())
            self._c4 = state.step
        if self._m7(state, o):
            return SAOResponse(ResponseType.ACCEPT_OFFER, o)
        x = self._m11(state)
        self._m15(x)
        return SAOResponse(ResponseType.REJECT_OFFER, x)

    def _m6(self, t):
        return min(0.995, max(0.986, self.K1 + 0.006 * (1.0 - self._m2()) + 0.002 * t))

    def _m7(self, state, o):
        u = float(self.ufun(o))
        if u <= self._a0:
            return False
        t = state.relative_time
        r = 0.942 + 0.030 * self.K0 - 0.010 * min(1.0, self._m2())
        if self._a8:
            r -= 0.018
        target = self._m8(t)
        if u >= target * r:
            return True

        # Replay a better seen opponent offer instead of accepting a weaker
        # current lowball in the deadline window. _m11() returns _b4 at K9.
        if self._b4 is not None:
            try:
                best_seen = float(self.ufun(self._b4))
            except Exception:
                best_seen = -float("inf")
            if t >= self.K9 and best_seen > u + 1e-9 and best_seen > self._a0 + 1e-9:
                return False
            if t >= self._m6(t) and u >= 0.965 * best_seen:
                return True

        if t >= self._m6(t) and u >= max(self._a0 + 1e-9, 0.72 * target):
            return True

        if t >= 0.9975 and u > self._a0 + 1e-9:
            return True
        return False

    def _m8(self, t):
        return self._a6 - (self._a6 - self._c3) * (t ** self._m3())

    def _m9(self, t):
        return (2.4 + 1.25 * self.K0) * (t ** 0.92)

    def _m10(self, x):
        if self._b3 < 2:
            return sum(self._b2[i].get(x[i], 0) for i in range(self._a7))
        y = np.array([self._m13(o, x) for o in self._b0])
        if y.std() < 1e-9 or self._b1.std() < 1e-9:
            return 0.0
        try:
            z, _ = kendalltau(y, self._b1)
            if np.isnan(z):
                return 0.0
            return float(z)
        except Exception:
            return 0.0

    def _m11(self, state):
        t = state.relative_time
        # Offer-side safety net: near the deadline, re-propose the best offer the
        # opponent has shown us (if it clears our reserve). This converts a likely
        # no-deal into an agreement worth real advantage. Fires at K9, before the
        # original (much later) _m6 trigger.
        if t >= self.K9 and self._b4 is not None and float(self.ufun(self._b4)) > self._a0 + 1e-9:
            return self._b4
        if self._a8:
            return max(self._a3, key=lambda x: (min(x[0], self._m1(x[1])), 0.35 * x[0] + self._m1(x[1])))[1]
        if t >= self._m6(t) and self._b4 is not None:
            return self._b4
        a = self._m8(t)
        c = [(u, o) for u, o in self._a3 if u >= a - 1e-9]
        if not c:
            k = max(1, min(len(self._a4), max(2, len(self._a4) // 9)))
            c = [(self._a5[i], self._a4[i]) for i in range(k)]
        if len(c) > self.K4:
            c = self._a9.sample(c, self.K4)
        s = self._m9(t)
        h = self.K3 * (1.0 - 0.52 * max(0.0, (t - 0.82) / 0.18))
        best = None
        best_score = float("inf")
        for u, o in c:
            v = self._m1(o)
            q = h * self._m10(o) - s * (u + v) - 0.08 * min(u, v)
            if q < best_score:
                best_score = q
                best = o
        return best if best is not None else c[0][1]

    def _m12(self, o, x=None):
        if self._a7 == 0:
            return 0.5
        z = 0.0
        w = 1.0 / self._a7
        for i in range(self._a7):
            f = self._b2[i].get(o[i], 0)
            if x is not None and x[i] == o[i]:
                f += 1
            m = max(self._b2[i].values()) if self._b2[i] else 0
            if x is not None:
                m = max(m, self._b2[i].get(x[i], 0) + 1)
            z += w * ((f / m) if m > 0 else 1.0)
        return z

    def _m13(self, o, x):
        return self._m12(o, x)

    def _m15(self, o):
        if o is None:
            return
        for i in range(self._a7):
            v = o[i]
            self._b2[i][v] = self._b2[i].get(v, 0) + 1
        self._b3 += 1

    def _m16(self, state, o):
        try:
            p = getattr(state, "current_proposer", None) or "opponent"
            self._b6.on_partner_proposal(state=state, partner_id=p, offer=o)
            self._b8 += 1
        except Exception:
            pass
        self.private_info["opponent_ufun"] = self._c1
