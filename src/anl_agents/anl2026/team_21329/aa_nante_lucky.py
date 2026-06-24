from __future__ import annotations

from attrs import define, field

from negmas.gb.components.genius.models import (
    GHardHeadedFrequencyModel,
    GSmithFrequencyModel,
)
from negmas.gb.components.base import GBComponent
from negmas.outcomes import Outcome
from negmas.sao.common import ResponseType
from negmas.sao.components.base import AcceptancePolicy, OfferingPolicy
from negmas.sao.negotiators.modular import BOANegotiator

try:
    from examples.boa import BOANeg
except ImportError:
    class BOANeg(BOANegotiator):
        pass


def _reserved_value(ufun) -> float:
    value = getattr(ufun, "reserved_value", 0.0)
    return 0.0 if value is None else float(value)


def _utility(ufun, outcome, default: float = 0.0) -> float:
    if ufun is None or outcome is None:
        return default
    try:
        value = ufun(outcome)
    except Exception:
        return default
    return default if value is None else float(value)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _offer_key(offer):
    try:
        hash(offer)
        return offer
    except TypeError:
        return repr(offer)


@define
class RobustOpponentModel(GHardHeadedFrequencyModel):
    """HardHeaded model with a small Smith-frequency fallback."""

    smith_weight: float = 0.20
    _smith: GSmithFrequencyModel = field(factory=GSmithFrequencyModel, init=False)

    def set_negotiator(self, negotiator) -> None:
        super().set_negotiator(negotiator)
        self._smith.set_negotiator(negotiator)

    def on_preferences_changed(self, changes) -> None:
        super().on_preferences_changed(changes)
        self._smith.on_preferences_changed(changes)

    def on_partner_proposal(self, state, partner_id: str, offer) -> None:
        super().on_partner_proposal(state, partner_id, offer)
        self._smith.on_partner_proposal(state, partner_id, offer)

    def eval(self, offer) -> float:
        hardheaded = _utility(super(), offer, 0.0)
        smith = _utility(self._smith, offer, 0.0)
        weight = _clamp(self.smith_weight)
        return (1.0 - weight) * hardheaded + weight * smith

    def eval_normalized(
        self,
        offer,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> float:
        return self.eval(offer)


class EarlyFrequencyExposedModel(GBComponent):
    """Rank-oriented public opponent model based on early partner offers.

    This variant keeps the original frequency model once at least three partner
    offers are seen. For sparse traces, it adds a small inverse-own-utility prior
    so the exposed opponent model is not a constant/tied estimate.
    """

    sparse_prior_weight: float = 0.18

    def __init__(self):
        super().__init__()
        self._offers = []

    def _expose_self(self) -> None:
        if getattr(self, "negotiator", None) is not None:
            self.negotiator.private_info["opponent_ufun"] = self

    def set_negotiator(self, negotiator) -> None:
        super().set_negotiator(negotiator)
        self._expose_self()

    def on_preferences_changed(self, changes) -> None:
        self._offers = []
        self._expose_self()

    def on_partner_proposal(self, state, partner_id: str, offer) -> None:
        if offer is not None:
            self._offers.append(offer)
        self._expose_self()

    def __call__(self, outcome):
        return self.eval(outcome)

    def _inverse_own_prior(self, outcome) -> float:
        negotiator = getattr(self, "negotiator", None)
        ufun = getattr(negotiator, "ufun", None)
        if ufun is None or outcome is None:
            return 0.0
        reserved = _reserved_value(ufun)
        try:
            best = ufun.best()
        except Exception:
            best = None
        best_utility = _utility(ufun, best, reserved + 1.0)
        scale = max(best_utility - reserved, 1e-9)
        own = (_utility(ufun, outcome, reserved) - reserved) / scale
        return _clamp(1.0 - own)

    def _frequency_eval(self, outcome) -> float:
        if outcome is None or not self._offers:
            return 0.0
        count = len(self._offers)
        weighted = []
        for index, offer in enumerate(self._offers):
            recency = 1.0 if count <= 1 else index / (count - 1)
            weighted.append((1.0 - 0.98 * recency, offer))
        if isinstance(outcome, tuple):
            totals = [0.0] * len(outcome)
            matches = [0.0] * len(outcome)
            value_weights = [{} for _ in outcome]
            for weight, offer in weighted:
                if not isinstance(offer, tuple) or len(offer) != len(outcome):
                    continue
                for issue_index, value in enumerate(offer):
                    totals[issue_index] += weight
                    value_weights[issue_index][value] = (
                        value_weights[issue_index].get(value, 0.0) + weight
                    )
                    if outcome[issue_index] == value:
                        matches[issue_index] += weight
            if len(outcome) <= 4:
                weighted_scores = []
                total_weight = 0.0
                for issue_index, total in enumerate(totals):
                    if total <= 0.0:
                        continue
                    concentration = (
                        max(value_weights[issue_index].values(), default=0.0) / total
                    )
                    issue_weight = concentration ** 4.5
                    weighted_scores.append(matches[issue_index] / total * issue_weight)
                    total_weight += issue_weight
                if total_weight > 0.0:
                    return sum(weighted_scores) / total_weight
            scores = [
                match / total
                for match, total in zip(matches, totals)
                if total > 0.0
            ]
            return sum(scores) / len(scores) if scores else 0.0
        total = sum(weight for weight, _ in weighted)
        if total <= 0.0:
            return 0.0
        return sum(weight for weight, offer in weighted if offer == outcome) / total

    def eval(self, outcome) -> float:
        if len(self._offers) >= 3:
            return self._frequency_eval(outcome)
        prior = self._inverse_own_prior(outcome)
        if not self._offers:
            return prior
        freq = self._frequency_eval(outcome)
        weight = _clamp(self.sparse_prior_weight)
        return (1.0 - weight) * freq + weight * prior

    def eval_normalized(
        self,
        outcome,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> float:
        return self.eval(outcome)


@define
class SafeAcceptancePolicy(AcceptancePolicy):
    """Accept strong offers while avoiding below-reserve agreements."""

    early_threshold: float = 0.90
    middle_threshold: float = 0.78
    late_threshold: float = 0.50
    final_threshold: float = 0.35
    low_issue_final_threshold: float = 0.30
    reserved_margin: float = 0.005
    repeated_high_time: float = 0.45
    repeated_high_threshold: float = 0.88
    repeated_high_window: int = 4
    repeated_high_count: int = 2
    repeated_high_min_received_diversity: int = 8
    repeated_high_max_sent_diversity: int = 4
    low_average_high_time: float = 0.38
    low_average_high_max_recent_average: float = 0.70
    low_average_high_min_received_diversity: int = 5
    late_medium_time: float = 0.80
    late_medium_threshold: float = 0.75
    late_medium_max_recent_average: float = 0.70
    late_medium_min_received_diversity: int = 8
    late_medium_max_sent_diversity: int = 2

    def _normalized_utility(self, offer) -> float:
        ufun = self.negotiator.ufun
        reserved = _reserved_value(ufun)
        best = getattr(ufun, "best", lambda: None)()
        best_utility = _utility(ufun, best, reserved + 1.0)
        scale = max(best_utility - reserved, 1e-9)
        return (_utility(ufun, offer, reserved - 1.0) - reserved) / scale

    def _issue_count(self) -> int:
        ufun = self.negotiator.ufun if self.negotiator else None
        outcome_space = getattr(ufun, "outcome_space", None)
        if outcome_space is None:
            return 0
        try:
            sample = next(
                iter(outcome_space.enumerate_or_sample(levels=10, max_cardinality=1))
            )
        except Exception:
            return 0
        return len(sample) if isinstance(sample, tuple) else 0

    def _high_dimensional_domain(self) -> bool:
        return self._issue_count() >= 6

    def _outcome_count(self) -> int:
        ufun = self.negotiator.ufun if self.negotiator else None
        outcome_space = getattr(ufun, "outcome_space", None)
        if outcome_space is None:
            return 0
        cardinality = getattr(outcome_space, "cardinality", 0)
        try:
            return int(cardinality() if callable(cardinality) else cardinality)
        except Exception:
            return 0

    def _threshold(self, relative_time: float) -> float:
        issue_count = self._issue_count()
        high_dimensional = issue_count >= 6
        medium_dimensional = 2 <= issue_count <= 4
        broad_medium = 3 <= issue_count <= 4 and self._outcome_count() >= 64
        middle_floor = 0.99
        late_floor = 0.98 if broad_medium else 0.78
        middle_threshold = (
            max(self.middle_threshold, 0.95)
            if high_dimensional
            else max(self.middle_threshold, middle_floor)
            if medium_dimensional
            else self.middle_threshold
        )
        late_threshold = (
            max(self.late_threshold, 0.70)
            if high_dimensional
            else max(self.late_threshold, late_floor)
            if medium_dimensional
            else self.late_threshold
        )
        if 0.98 <= relative_time < 0.99:
            pre_final_floor = 0.55 if broad_medium else 0.50
            return max(self.final_threshold, pre_final_floor)
        if relative_time < 0.35:
            return self.early_threshold
        if relative_time < 0.85:
            progress = (relative_time - 0.35) / 0.50
            return self.early_threshold + progress * (
                middle_threshold - self.early_threshold
            )
        if relative_time < 0.98:
            progress = (relative_time - 0.85) / 0.13
            return middle_threshold + progress * (
                late_threshold - middle_threshold
            )
        if 1 <= issue_count <= 2:
            return self.low_issue_final_threshold
        return self.final_threshold

    def _stable_low_issue_final_pressure(self) -> bool:
        offering = getattr(self.negotiator, "_offering", None)
        received = list(getattr(offering, "_received_offers", []) or [])
        sent = list(getattr(offering, "_sent_keys", []) or [])
        recent_received = received[-10:]
        return (
            bool(recent_received)
            and len({repr(old) for old in recent_received}) <= 1
            and len(set(sent[-10:])) >= 3
        )

    def __call__(self, state, offer, source):
        ufun = self.negotiator.ufun if self.negotiator else None
        if ufun is None or offer is None:
            return ResponseType.REJECT_OFFER

        reserved = _reserved_value(ufun)
        utility = _utility(ufun, offer, reserved - 1.0)
        if state.relative_time >= 0.99 and self._normalized_utility(offer) >= 0.005:
            offering = getattr(self.negotiator, "_offering", None)
            if offering is not None:
                try:
                    if offering._diverse_weak_final_pressure():
                        return ResponseType.ACCEPT_OFFER
                except Exception:
                    pass
            if 1 <= self._issue_count() <= 2:
                try:
                    if self._stable_low_issue_final_pressure():
                        return ResponseType.ACCEPT_OFFER
                except Exception:
                    pass

        if utility <= reserved + self.reserved_margin:
            return ResponseType.REJECT_OFFER

        if self._normalized_utility(offer) >= self.repeated_high_threshold:
            offering = getattr(self.negotiator, "_offering", None)
            received = list(getattr(offering, "_received_offers", []) or [])
            sent = list(getattr(offering, "_sent_keys", []) or [])
            recent_received = received[-10:]
            recent_utilities = [
                self._normalized_utility(old) for old in recent_received
            ]
            received_diversity = len({repr(old) for old in recent_received})
            sent_diversity = len(set(sent[-10:]))
            high_count = 1 + sum(
                1
                for old in received[-self.repeated_high_window :]
                if self._normalized_utility(old) >= self.repeated_high_threshold
            )
            enough_repeated_high = (
                sent_diversity <= self.repeated_high_max_sent_diversity
                and high_count >= self.repeated_high_count
            )
            if (
                state.relative_time >= self.repeated_high_time
                and enough_repeated_high
                and received_diversity >= self.repeated_high_min_received_diversity
            ):
                return ResponseType.ACCEPT_OFFER
            if (
                state.relative_time >= self.low_average_high_time
                and enough_repeated_high
                and recent_utilities
                and sum(recent_utilities) / len(recent_utilities)
                <= self.low_average_high_max_recent_average
                and received_diversity >= self.low_average_high_min_received_diversity
            ):
                return ResponseType.ACCEPT_OFFER

        if self._normalized_utility(offer) >= self.late_medium_threshold:
            offering = getattr(self.negotiator, "_offering", None)
            received = list(getattr(offering, "_received_offers", []) or [])
            sent = list(getattr(offering, "_sent_keys", []) or [])
            recent_received = received[-10:]
            recent_utilities = [
                self._normalized_utility(old) for old in recent_received
            ]
            if (
                state.relative_time >= self.late_medium_time
                and recent_utilities
                and sum(recent_utilities) / len(recent_utilities)
                <= self.late_medium_max_recent_average
                and len({repr(old) for old in recent_received})
                >= self.late_medium_min_received_diversity
                and len(set(sent[-10:])) <= self.late_medium_max_sent_diversity
            ):
                return ResponseType.ACCEPT_OFFER

        if self._normalized_utility(offer) >= self._threshold(state.relative_time):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class ConcealingOfferingPolicy(OfferingPolicy):
    """Utility-aware offers with mild preference concealment."""

    early_target: float = 0.98
    middle_target: float = 0.82
    late_target: float = 0.52
    final_target: float = 0.30
    max_outcomes: int = 100_000
    own_band_slack: float = 0.025
    recent_window: int = 4
    compromise_start: float = 0.94
    compromise_floor: float = 0.70
    partner_influence_start: float = 0.50
    partner_influence_max: float = 0.35
    final_replay_time: float = 0.985
    final_replay_threshold: float = 0.50

    _outcomes: list[Outcome] | None = None
    _norm_utils: list[float] | None = None
    _received_offers: list[Outcome] | None = None
    _sent_keys: list | None = None

    def _reset(self) -> None:
        self._outcomes = None
        self._norm_utils = None
        self._received_offers = []
        self._sent_keys = []

    def on_preferences_changed(self, changes):
        self._reset()
        return super().on_preferences_changed(changes)

    def on_partner_proposal(self, state, partner_id: str, offer) -> None:
        self._ensure_ready()
        if offer is not None and self._received_offers is not None:
            self._received_offers.append(offer)
        return super().on_partner_proposal(state, partner_id, offer)

    def _ensure_ready(self) -> None:
        if self._outcomes is not None and self._norm_utils is not None:
            return
        if self._received_offers is None:
            self._received_offers = []
        if self._sent_keys is None:
            self._sent_keys = []

        ufun = self.negotiator.ufun
        assert ufun is not None
        reserved = _reserved_value(ufun)
        best = ufun.best()
        best_utility = _utility(ufun, best, reserved + 1.0)
        scale = max(best_utility - reserved, 1e-9)

        outcome_space = ufun.outcome_space
        assert outcome_space is not None
        try:
            outcomes = outcome_space.enumerate_or_sample(
                levels=10, max_cardinality=self.max_outcomes
            )
        except AttributeError:
            outcomes = outcome_space.enumerate()

        scored = []
        for outcome in outcomes:
            utility = _utility(ufun, outcome, reserved - 1.0)
            if utility <= reserved:
                continue
            scored.append(((utility - reserved) / scale, outcome))

        scored.sort(key=lambda item: item[0], reverse=True)
        self._norm_utils = [item[0] for item in scored]
        self._outcomes = [item[1] for item in scored]

    def _normalized_own_utility(self, outcome) -> float:
        ufun = self.negotiator.ufun
        assert ufun is not None
        reserved = _reserved_value(ufun)
        best = ufun.best()
        best_utility = _utility(ufun, best, reserved + 1.0)
        scale = max(best_utility - reserved, 1e-9)
        return _clamp((_utility(ufun, outcome, reserved - 1.0) - reserved) / scale)

    def _best_received_offer(self):
        if not self._received_offers:
            return None, 0.0
        return max(
            (
                (self._normalized_own_utility(offer), offer)
                for offer in self._received_offers
            ),
            key=lambda item: item[0],
        )

    def _target(self, relative_time: float) -> float:
        if relative_time < 0.35:
            return self.early_target
        if relative_time < 0.80:
            progress = (relative_time - 0.35) / 0.45
            return self.early_target + progress * (
                self.middle_target - self.early_target
            )
        if relative_time < 0.98:
            progress = (relative_time - 0.80) / 0.18
            return self.middle_target + progress * (
                self.late_target - self.middle_target
            )
        return self.final_target

    def _raw_partner_score(self, outcome) -> float:
        models = getattr(self.negotiator, "_models", None) or []
        if not models:
            return 0.0
        return _utility(models[0], outcome, 0.0)

    def _partner_scored_candidates(self, candidates):
        raw_scores = [self._raw_partner_score(outcome) for _, outcome in candidates]
        low = min(raw_scores)
        high = max(raw_scores)
        if high <= low:
            partner_scores = [0.0] * len(candidates)
        else:
            scale = high - low
            partner_scores = [(score - low) / scale for score in raw_scores]
        return [
            (own_utility, outcome, partner_score)
            for (own_utility, outcome), partner_score in zip(candidates, partner_scores)
        ]

    def _distance_to_latest_offer(self, outcome) -> float:
        if not self._received_offers:
            return 1.0
        reference = self._received_offers[-1]
        if not isinstance(outcome, tuple) or not isinstance(reference, tuple):
            return 0.0 if outcome == reference else 1.0
        if len(outcome) != len(reference) or not outcome:
            return 1.0
        return sum(a != b for a, b in zip(outcome, reference)) / len(outcome)

    def _fresh_candidates(self, candidates):
        if not candidates or not self._sent_keys:
            return candidates
        window = self.recent_window
        if self._outcomes and isinstance(self._outcomes[0], tuple):
            issue_count = len(self._outcomes[0])
            if issue_count >= 6:
                window = max(window, 6)
            elif issue_count >= 5:
                window = max(window, 5)
        recent = set(self._sent_keys[-window:])
        fresh = [item for item in candidates if _offer_key(item[1]) not in recent]
        return fresh or candidates

    def _worsening_low_partner(self) -> bool:
        if not self._received_offers or len(self._received_offers) < 5:
            return False
        utilities = [
            self._normalized_own_utility(offer) for offer in self._received_offers[-5:]
        ]
        return (
            max(utilities) < 0.65
            and utilities[-1] <= utilities[0]
            and sum(utilities) / len(utilities) < 0.45
        )

    def _stable_worsening_partner(self) -> bool:
        if (
            not self._received_offers
            or len(self._received_offers) < 80
            or not self._sent_keys
        ):
            return False
        recent = self._received_offers[-10:]
        utilities = [self._normalized_own_utility(offer) for offer in recent]
        average = sum(utilities) / len(utilities)
        return (
            0.35 <= average < 0.42
            and max(utilities) < 0.45
            and len({repr(offer) for offer in recent}) == 1
            and len({repr(offer) for offer in self._sent_keys[-10:]}) <= 1
        )

    def _weak_diverse_worsening_partner(self) -> bool:
        if (
            not self._received_offers
            or len(self._received_offers) < 85
            or not self._sent_keys
        ):
            return False
        recent = self._received_offers[-10:]
        utilities = [self._normalized_own_utility(offer) for offer in recent]
        average = sum(utilities) / len(utilities)
        return (
            0.32 <= average < 0.38
            and max(utilities) < 0.40
            and len({repr(offer) for offer in recent}) >= 3
            and len({repr(offer) for offer in self._sent_keys[-10:]}) >= 2
        )

    def _stable_medium_partner(self) -> bool:
        if not self._received_offers or len(self._received_offers) < 80:
            return False
        recent = self._received_offers[-10:]
        utilities = [self._normalized_own_utility(offer) for offer in recent]
        return (
            len({repr(offer) for offer in recent}) == 1
            and 0.50 <= sum(utilities) / len(utilities) < 0.62
        )

    def _diverse_medium_partner(self) -> bool:
        if (
            not self._received_offers
            or len(self._received_offers) < 85
            or not self._sent_keys
        ):
            return False
        recent = self._received_offers[-10:]
        utilities = [self._normalized_own_utility(offer) for offer in recent]
        average = sum(utilities) / len(utilities)
        received_diversity = len({repr(offer) for offer in recent})
        sent_diversity = len({repr(offer) for offer in self._sent_keys[-10:]})
        return (
            received_diversity >= 6
            and sent_diversity >= 6
            and 0.25 <= average < 0.42
            and max(utilities) > 0.45
        )

    def _late_middle_partner(self) -> bool:
        if (
            not self._received_offers
            or len(self._received_offers) < 80
            or not self._sent_keys
        ):
            return False
        recent = self._received_offers[-10:]
        utilities = [self._normalized_own_utility(offer) for offer in recent]
        average = sum(utilities) / len(utilities)
        return (
            0.42 <= average < 0.62
            and max(utilities) >= 0.55
            and len({repr(offer) for offer in recent}) >= 2
            and len({repr(offer) for offer in self._sent_keys[-10:]}) >= 1
        )

    def _low_diverse_partner(self) -> bool:
        if (
            not self._received_offers
            or len(self._received_offers) < 80
            or not self._sent_keys
        ):
            return False
        recent = self._received_offers[-10:]
        utilities = [self._normalized_own_utility(offer) for offer in recent]
        average = sum(utilities) / len(utilities)
        return (
            0.05 <= average < 0.25
            and max(utilities) >= 0.25
            and len({repr(offer) for offer in recent}) >= 3
            and len({repr(offer) for offer in self._sent_keys[-10:]}) >= 3
        )

    def _weak_final_pressure(self) -> bool:
        if not self._received_offers or len(self._received_offers) < 80:
            return False
        recent = self._received_offers[-10:]
        utilities = [self._normalized_own_utility(offer) for offer in recent]
        return max(utilities) < 0.15 and sum(utilities) / len(utilities) < 0.08

    def _diverse_weak_final_pressure(self) -> bool:
        if (
            not self._weak_final_pressure()
            or not self._received_offers
            or not self._sent_keys
        ):
            return False
        recent = self._received_offers[-10:]
        return (
            len({repr(offer) for offer in recent}) >= 5
            and len({repr(offer) for offer in self._sent_keys[-10:]}) >= 5
        )

    def _recent_received_diversity(self) -> int:
        if not self._received_offers:
            return 0
        return len({repr(offer) for offer in self._received_offers[-10:]})

    def _multi_issue_outcomes(self) -> bool:
        if not self._outcomes:
            return False
        sample = self._outcomes[0]
        return isinstance(sample, tuple) and len(sample) > 1

    def _partner_influence_limit(self) -> float:
        if self._outcomes and isinstance(self._outcomes[0], tuple):
            if len(self._outcomes[0]) >= 5:
                return max(self.partner_influence_max, 0.36)
        return self.partner_influence_max

    def _worsening_compromise_floor(self) -> float:
        if not self._outcomes:
            return self.compromise_floor
        sample = self._outcomes[0]
        if isinstance(sample, tuple) and len(sample) <= 2:
            return max(self.compromise_floor, 0.80)
        return self.compromise_floor

    def _sparse_middle_partner(self) -> bool:
        if (
            not self._received_offers
            or len(self._received_offers) < 80
            or not self._sent_keys
        ):
            return False
        recent = self._received_offers[-10:]
        utilities = [self._normalized_own_utility(offer) for offer in recent]
        average = sum(utilities) / len(utilities)
        return (
            0.30 <= average < 0.50
            and max(utilities) >= 0.45
            and len({repr(offer) for offer in recent}) >= 2
            and len({repr(offer) for offer in self._sent_keys[-10:]}) >= 3
        )

    def _narrow_sparse_partner(self) -> bool:
        if (
            not self._received_offers
            or len(self._received_offers) < 80
            or not self._sent_keys
        ):
            return False
        recent = self._received_offers[-10:]
        utilities = [self._normalized_own_utility(offer) for offer in recent]
        average = sum(utilities) / len(utilities)
        return (
            0.30 <= average < 0.42
            and max(utilities) >= 0.60
            and len({repr(offer) for offer in recent}) >= 8
            and len({repr(offer) for offer in self._sent_keys[-10:]}) <= 3
        )

    def _remember(self, outcome):
        if outcome is not None and self._sent_keys is not None:
            self._sent_keys.append(_offer_key(outcome))
        return outcome

    def __call__(self, state, dest: str | None = None):
        self._ensure_ready()
        if not self._outcomes or not self._norm_utils:
            return None

        if (
            2 <= len(self._outcomes) <= 3
            and isinstance(self._outcomes[0], tuple)
            and len(self._outcomes[0]) == 1
        ):
            if (
                len(self._outcomes) == 3
                and self._norm_utils[-1] <= 0.50
                and not self._sent_keys
                and not self._received_offers
            ):
                return self._remember(self._outcomes[1])
            return self._remember(self._outcomes[-1])

        if (
            0.85 <= state.relative_time < self.final_replay_time
            and self._narrow_sparse_partner()
        ):
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= 0.85
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(8000, len(candidates))]
                )
                ranked = max(
                    search,
                    key=lambda item: (
                        min(item[0], item[2]),
                        item[0] * item[2],
                        item[0],
                        -self._distance_to_latest_offer(item[1]),
                    ),
                )
                return self._remember(ranked[1])

        if (
            self.compromise_start <= state.relative_time < self.final_replay_time
            and self._stable_medium_partner()
        ):
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= self.compromise_floor
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(1500, len(candidates))]
                )
                ranked = max(
                    search,
                    key=lambda item: (
                        item[2],
                        -item[0],
                        min(item[0], item[2]),
                        -self._distance_to_latest_offer(item[1]),
                    ),
                )
                return self._remember(ranked[1])

        if (
            0.90 <= state.relative_time < self.final_replay_time
            and self._diverse_medium_partner()
        ):
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= 0.85
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(5000, len(candidates))]
                )
                ranked = max(
                    search,
                    key=lambda item: (
                        item[2],
                        -item[0],
                        min(item[0], item[2]),
                        -self._distance_to_latest_offer(item[1]),
                    ),
                )
                return self._remember(ranked[1])

        if (
            0.90 <= state.relative_time < self.final_replay_time
            and self._sparse_middle_partner()
        ):
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= 0.75
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(8000, len(candidates))]
                )
                ranked = max(
                    search,
                    key=lambda item: (
                        item[0] * item[2],
                        item[2],
                        item[0],
                        -self._distance_to_latest_offer(item[1]),
                    ),
                )
                return self._remember(ranked[1])

        if (
            0.90 <= state.relative_time < self.final_replay_time
            and self._low_diverse_partner()
        ):
            low_diverse_floor = (
                0.94
                if self._multi_issue_outcomes() and len(self._outcomes[0]) >= 5
                else 0.75
            )
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= low_diverse_floor
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(8000, len(candidates))]
                )
                ranked = max(
                    search,
                    key=lambda item: (
                        min(item[0], item[2]),
                        item[0] * item[2],
                        item[0],
                        -self._distance_to_latest_offer(item[1]),
                    ),
                )
                return self._remember(ranked[1])

        if (
            0.90 <= state.relative_time < self.final_replay_time
            and self._late_middle_partner()
        ):
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= 0.75
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(5000, len(candidates))]
                )
                ranked = max(
                    search,
                    key=lambda item: (
                        min(item[0], item[2]),
                        item[0] * item[2],
                        item[0],
                        -self._distance_to_latest_offer(item[1]),
                    ),
                )
                return self._remember(ranked[1])

        if (
            self.compromise_start <= state.relative_time < self.final_replay_time
            and self._worsening_low_partner()
        ):
            compromise_floor = self._worsening_compromise_floor()
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= compromise_floor
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(1500, len(candidates))]
                )
                if (
                    self._stable_worsening_partner()
                    or self._weak_diverse_worsening_partner()
                ):
                    ranked = max(
                        search,
                        key=lambda item: (
                            item[0] * item[2],
                            item[0],
                            item[2],
                            -self._distance_to_latest_offer(item[1]),
                        ),
                    )
                else:
                    ranked = max(
                        search,
                        key=lambda item: (
                            item[2],
                            min(item[0], item[2]),
                            item[0],
                            -self._distance_to_latest_offer(item[1]),
                        ),
                )
                return self._remember(ranked[1])

        if state.relative_time >= 0.98 and self._diverse_weak_final_pressure():
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= 0.18
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(3000, len(candidates))]
                )
                ranked = max(
                    search,
                    key=lambda item: (
                        min(item[0], item[2]),
                        item[0] * item[2],
                        item[0],
                        -self._distance_to_latest_offer(item[1]),
                    ),
                )
                return self._remember(ranked[1])

        if (
            0.98 <= state.relative_time < self.final_replay_time
            and self._multi_issue_outcomes()
            and self._weak_final_pressure()
        ):
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= 0.18
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(3000, len(candidates))]
                )
                ranked = max(
                    search,
                    key=lambda item: (
                        item[2],
                        item[0] * item[2],
                        item[0],
                        -self._distance_to_latest_offer(item[1]),
                    ),
                )
                return self._remember(ranked[1])

        if state.relative_time >= self.final_replay_time and self._weak_final_pressure():
            final_floor = 0.22 if self._recent_received_diversity() >= 5 else 0.15
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= final_floor
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(3000, len(candidates))]
                )
                ranked = max(
                    search,
                    key=lambda item: (
                        item[2],
                        item[0] * item[2],
                        item[0],
                        -self._distance_to_latest_offer(item[1]),
                    ),
                )
                return self._remember(ranked[1])

        if state.relative_time >= self.final_replay_time:
            best_received_utility, best_received_offer = self._best_received_offer()
            if best_received_offer is not None and best_received_utility >= self.final_replay_threshold:
                return self._remember(best_received_offer)
            candidates = [
                (u, outcome)
                for u, outcome in zip(self._norm_utils, self._outcomes)
                if u >= self.final_target
            ]
            if candidates:
                candidates = self._fresh_candidates(candidates)
                search = self._partner_scored_candidates(
                    candidates[: min(1000, len(candidates))]
                )
                ranked = max(
                    search,
                    key=lambda item: (
                        (item[0] ** 2.5) * item[2],
                        min(item[0], item[2]),
                        item[0],
                        -self._distance_to_latest_offer(item[1]),
                    ),
                )
                return self._remember(ranked[1])

        target = self._target(state.relative_time)
        candidates = [
            (u, outcome)
            for u, outcome in zip(self._norm_utils, self._outcomes)
            if u >= target
        ]
        if not candidates:
            candidates = [(self._norm_utils[0], self._outcomes[0])]

        best_own = max(u for u, _ in candidates)
        own_floor = max(target, best_own - self.own_band_slack)
        candidates = [(u, o) for u, o in candidates if u >= own_floor] or candidates
        candidates = self._fresh_candidates(candidates)

        partner_power = 0.0
        partner_influence_max = self._partner_influence_limit()
        if state.relative_time > self.partner_influence_start:
            partner_power = min(
                partner_influence_max,
                (state.relative_time - self.partner_influence_start)
                / max(1e-9, 1.0 - self.partner_influence_start)
                * partner_influence_max,
            )

        search = self._partner_scored_candidates(candidates[: min(750, len(candidates))])
        ranked = min(
            search,
            key=lambda item: (
                -(item[0] * (1.0 + partner_power * item[2])),
                self._distance_to_latest_offer(item[1]),
                _offer_key(item[1]),
            ),
        )
        return self._remember(ranked[1])


class AaNanteLucky(BOANeg):
    """ANL 2026 agent with BOA compatibility and no side effects."""

    def __init__(self, *args, **kwargs):
        offering = ConcealingOfferingPolicy()
        BOANegotiator.__init__(
            self,
            *args,
            acceptance=SafeAcceptancePolicy(),
            offering=offering,
            model=RobustOpponentModel(),
            extra_components=[EarlyFrequencyExposedModel()],
            extra_component_names=["exposed_opponent_model"],
            **kwargs,
        )

