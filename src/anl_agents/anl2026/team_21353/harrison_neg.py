import math
import random
from collections import defaultdict
from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun


class HarrisonNeg(SAOCallNegotiator):
    """
    ANL 2026 agent optimising Score = Advantage + Concealing.

    Advantage: time-based linear concession with hard floor — never accept
      below MIN_FRAC of our utility range above reserved value.

    Concealing: issue-value frequency balancing across the full eligible set.
      We track which values we've offered per issue and prefer outcomes that
      use underrepresented values — but only when the opponent model suggests
      the opponent is unlikely to accept (safe decoys). This keeps the
      opponent's frequency model flat (low Kendall-tau with our true utility).

    Opponent model: frequency-based with inverse-entropy issue weighting.
      Issues where the opponent shows concentrated value choices (low entropy)
      are presumed more important and weighted higher.
    """

    rational_outcomes: tuple = ()
    _eligible: tuple = ()
    _MIN_FRAC: float = 0.20

    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return

        rv = float(self.ufun.reserved_value)
        max_u = float(self.ufun.max())
        floor = rv + self._MIN_FRAC * max(0.0, max_u - rv)

        scored = [
            (float(self.ufun(o)), o)
            for o in self.nmi.outcome_space.enumerate_or_sample()
            if float(self.ufun(o)) > rv
        ]
        scored.sort(reverse=True)
        self.rational_outcomes = tuple(o for _, o in scored)

        self._eligible = tuple(o for u, o in scored if u >= floor)
        if not self._eligible:
            self._eligible = self.rational_outcomes

        self._issues = list(self.nmi.outcome_space.issues)
        self._n_issues = len(self._issues)
        self._issue_names = [iss.name for iss in self._issues]
        self._issue_values = {iss.name: list(iss.values) for iss in self._issues}

        # Opponent offer tracking
        self._opp_counts: dict = defaultdict(lambda: defaultdict(int))
        self._opp_total: int = 0

        # Our own offer tracking (for frequency balancing)
        self._my_counts: dict = defaultdict(lambda: defaultdict(int))
        self._my_total: int = 0

        self.private_info["opponent_ufun"] = LambdaMultiFun(f=lambda _x: 0.5)

    # ------------------------------------------------------------------ #

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer

        if offer is None:
            bid = self._bid(state)
            self._track_my(bid)
            return SAOResponse(ResponseType.REJECT_OFFER, bid)

        self.update_opponent_model(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        bid = self._bid(state)
        self._track_my(bid)
        return SAOResponse(ResponseType.REJECT_OFFER, bid)

    # ------------------------------------------------------------------ #

    def _track_my(self, offer) -> None:
        if offer is None:
            return
        for idx, name in enumerate(self._issue_names):
            self._my_counts[name][offer[idx]] += 1
        self._my_total += 1

    # ------------------------------------------------------------------ #

    def acceptance_strategy(self, state: SAOState) -> bool:
        assert self.ufun
        offer = state.current_offer
        if offer is None:
            return False

        u = float(self.ufun(offer))
        rv = float(self.ufun.reserved_value)
        max_u = float(self.ufun.max())
        gap = max(0.0, max_u - rv)
        if gap == 0:
            return u >= rv

        t = state.relative_time
        frac = max(self._MIN_FRAC, 0.9 - (0.9 - self._MIN_FRAC) * t)
        return u >= rv + frac * gap

    # ------------------------------------------------------------------ #

    def _bid(self, state: SAOState) -> Outcome | None:
        n = len(self._eligible)
        if n == 0:
            return None

        t = state.relative_time

        # Open with best outcome to anchor high.
        if t < 0.02:
            return self._eligible[0]

        # Phase 1 (0.02–0.1): small top pool, light frequency balancing.
        if t < 0.1:
            pool_size = max(1, n // 10)
            return self._balanced_pick(self._eligible[:pool_size], balance_weight=0.25)

        # Phase 2 (0.1–0.65): try safe-decoy concealing first, then balanced pick.
        if t < 0.65:
            # Expand aspirational pool gradually for eventual concession.
            pool_frac = 0.10 + 0.25 * (t - 0.1) / 0.55
            pool_size = max(1, int(n * pool_frac))

            # Try a safe concealing bid from the full eligible set (~40% of rounds).
            if self._my_total >= 5 and random.random() < 0.40:
                decoy = self._safe_concealing_bid(pool_size)
                if decoy is not None:
                    return decoy

            return self._balanced_pick(self._eligible[:pool_size], balance_weight=0.70)

        # Phase 3 (≥0.65): concede and Nash-seek to close a deal.
        frac = 0.40 + 0.60 * min(1.0, (t - 0.65) / 0.35)
        pool_size = max(1, int(n * frac))
        candidates = list(self._eligible[:pool_size])

        if self._opp_total >= 10 and random.random() < 0.50:
            opp_ufun = self.private_info.get("opponent_ufun")
            if opp_ufun is not None:
                rv = float(self.ufun.reserved_value)
                best = max(
                    candidates,
                    key=lambda o: (float(self.ufun(o)) - rv) * float(opp_ufun(o)),
                )
                return best

        return self._balanced_pick(candidates, balance_weight=0.30)

    # ------------------------------------------------------------------ #

    def _balanced_pick(self, pool, balance_weight: float = 0.5) -> Outcome | None:
        """
        Pick from pool weighting utility rank and issue-value frequency balance.

        balance_weight=0 → pure utility (top of pool)
        balance_weight=1 → pure frequency balancing

        balance_score = sum over issues of (ideal_freq - current_freq) for the
        value used by the candidate in that issue. Positive when a value is
        underrepresented — we prefer it to confuse opponent frequency models.
        """
        if not pool:
            return None

        candidates = list(pool)[:min(80, len(pool))]
        n_cands = len(candidates)

        if balance_weight == 0 or self._my_total == 0:
            top = max(1, n_cands // 3)
            return random.choice(candidates[:top])

        ideal: dict[str, float] = {
            name: 1.0 / max(1, len(self._issue_values[name]))
            for name in self._issue_names
        }
        denom = self._my_total + 1

        def balance_score(o) -> float:
            s = 0.0
            for idx, name in enumerate(self._issue_names):
                n_vals = len(self._issue_values[name])
                if n_vals <= 1:
                    continue
                curr = self._my_counts[name][o[idx]] / denom
                s += ideal[name] - curr
            return s

        max_idx = max(n_cands - 1, 1)

        best = max(
            enumerate(candidates),
            key=lambda t: (
                (1 - balance_weight) * (1.0 - t[0] / max_idx)
                + balance_weight * balance_score(t[1])
            ),
        )[1]
        return best

    # ------------------------------------------------------------------ #

    def _safe_concealing_bid(self, min_pool_size: int) -> Outcome | None:
        """
        Find an eligible outcome from the FULL eligible set that:
          1. Uses issue values that are underrepresented in our offer history
             (frequency balancing score is positive/high), AND
          2. The opponent is unlikely to accept (low estimated opponent utility)
             so we don't accidentally concede value.

        Searches up to 200 eligible outcomes (balanced sample across utility range).
        """
        n = len(self._eligible)
        if n == 0:
            return None

        opp_ufun = self.private_info.get("opponent_ufun")
        ideal: dict[str, float] = {
            name: 1.0 / max(1, len(self._issue_values[name]))
            for name in self._issue_names
        }
        denom = self._my_total + 1

        # Sample across full eligible range (not just top).
        if n <= 200:
            candidates = list(self._eligible)
        else:
            # Always include the aspirational top pool, then sample remainder.
            top = list(self._eligible[:min_pool_size])
            rest_indices = random.sample(range(min_pool_size, n), min(150, n - min_pool_size))
            rest = [self._eligible[i] for i in rest_indices]
            candidates = top + rest

        # Compute opponent utility threshold: only offer below median opp utility.
        if opp_ufun is not None and self._opp_total >= 5:
            sample_utils = sorted(float(opp_ufun(o)) for o in candidates[:min(50, len(candidates))])
            opp_threshold = sample_utils[len(sample_utils) // 2]  # median
        else:
            opp_threshold = 1.0  # no filtering before opponent offers

        best_score = -float("inf")
        best = None

        for o in candidates:
            # Frequency balance score.
            bal = sum(
                ideal[name] - self._my_counts[name][o[idx]] / denom
                for idx, name in enumerate(self._issue_names)
                if len(self._issue_values[name]) > 1
            )

            # Safety: skip outcomes the opponent is likely to accept
            # (opp utility above threshold when we have a model).
            if opp_ufun is not None and self._opp_total >= 5:
                opp_u = float(opp_ufun(o))
                if opp_u > opp_threshold:
                    continue

            if bal > best_score:
                best_score = bal
                best = o

        return best

    # ------------------------------------------------------------------ #

    def update_opponent_model(self, state: SAOState) -> None:
        assert self.ufun and self.opponent_ufun

        offer = state.current_offer
        if offer is None:
            return

        for idx, name in enumerate(self._issue_names):
            self._opp_counts[name][offer[idx]] += 1
        self._opp_total += 1

        total = self._opp_total
        counts = {k: dict(v) for k, v in self._opp_counts.items()}
        issue_names = self._issue_names
        n_issues = self._n_issues

        # Issue importance weight: inverse normalised entropy.
        # Low-entropy (concentrated) offers → issue is likely important.
        issue_weights: dict[str, float] = {}
        for name in issue_names:
            val_counts = counts.get(name, {})
            if not val_counts:
                issue_weights[name] = 1.0
                continue
            t_cnt = sum(val_counts.values())
            freqs = [c / t_cnt for c in val_counts.values()]
            entropy = -sum(f * math.log(f) for f in freqs if f > 0)
            n_vals = max(len(self._issue_values.get(name, ["x"])), 2)
            max_entropy = math.log(n_vals)
            issue_weights[name] = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0)

        total_w = sum(issue_weights.values()) or 1.0
        norm_w = {k: v / total_w for k, v in issue_weights.items()}

        def weighted_freq_ufun(outcome,
                               _c=counts, _names=issue_names,
                               _t=total, _n=n_issues, _w=norm_w):
            if _n == 0 or _t == 0:
                return 0.5
            score = 0.0
            for i in range(_n):
                name = _names[i]
                freq = _c.get(name, {}).get(outcome[i], 0) / _t
                score += _w.get(name, 1.0 / _n) * freq
            return score

        self.private_info["opponent_ufun"] = LambdaMultiFun(f=weighted_freq_ufun)
