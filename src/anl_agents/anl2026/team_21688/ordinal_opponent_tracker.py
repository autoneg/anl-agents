import numpy as np
from negmas.preferences import LinearAdditiveUtilityFunction, TableFun

try:
    from .opponent_tracker import OpponentTracker
except ImportError:
    from opponent_tracker import OpponentTracker


class OrdinalOpponentTracker(OpponentTracker):
    """
    Stable opponent tracker with a small ordinal correction layer.

    The base `OpponentTracker` remains the main model. After each normal update,
    this class applies a conservative pairwise-ranking step so the predicted
    utility function better matches ANL's Kendall-style scoring target.
    """

    def __init__(
        self,
        *args,
        ordinal_lr=0.025,
        ordinal_margin=0.02,
        ordinal_max_delta=0.025,
        max_constraints_per_update=40,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ordinal_lr = ordinal_lr
        self.ordinal_margin = ordinal_margin
        self.ordinal_max_delta = ordinal_max_delta
        self.max_constraints_per_update = max_constraints_per_update
        self.value_lists = [list(issue.values) for issue in self.oppo_issues]
        self.value_index = [
            {value: idx for idx, value in enumerate(values)}
            for values in self.value_lists
        ]
        self.slices = []
        start = 0
        for values in self.value_lists:
            stop = start + len(values)
            self.slices.append(slice(start, stop))
            start = stop

    def update(self, sigma=0.1):
        """
        Run the original stable model first, then gently nudge it using
        high-confidence ordinal constraints.
        """
        super().update(sigma=sigma)

        constraints = self._build_ordinal_constraints()
        if constraints:
            self._apply_ordinal_correction(constraints)
            self._rebuild_predicted_ufun_from_theta()

    def _theta_from_current_ufun(self):
        theta = np.zeros(sum(len(values) for values in self.value_lists), dtype=float)
        weights = [float(w) for w in self.oppo_weights]
        total = sum(weights)
        if total <= 1e-12:
            weights = [1.0 / max(1, len(weights))] * len(weights)
        else:
            weights = [w / total for w in weights]

        for issue_idx, values in enumerate(self.value_lists):
            mapping = self.oppo_values[issue_idx].mapping
            for value_idx, value in enumerate(values):
                theta[self.slices[issue_idx].start + value_idx] = (
                    weights[issue_idx] * float(mapping.get(value, 0.0))
                )
        return theta

    def _features(self, offer):
        x = np.zeros(sum(len(values) for values in self.value_lists), dtype=float)
        if offer is None:
            return x
        for issue_idx, value in enumerate(offer):
            idx = self.value_index[issue_idx].get(value)
            if idx is not None:
                x[self.slices[issue_idx].start + idx] = 1.0
        return x

    def _add_constraint(self, constraints, better, worse, weight):
        if better is None or worse is None or better == worse or weight <= 0:
            return
        constraints.append((better, worse, float(weight)))

    def _build_ordinal_constraints(self):
        constraints = []
        n_opponent = len(self.opponent_offers)
        if n_opponent == 0:
            return constraints

        newest = self.opponent_offers[-1]

        # If our offer was rejected and the opponent countered, their counter
        # should usually be better for them than our last offer.
        if self.self_offers:
            self._add_constraint(
                constraints,
                newest,
                self.self_offers[-1],
                1.0 + 0.5 * float(self.relative_time),
            )

        # Opponent offers often move from their preferred region toward
        # concession. This is useful but noisy, so keep it weak.
        if n_opponent >= 2:
            previous = self.opponent_offers[-2]
            self._add_constraint(constraints, previous, newest, 0.18)

        for idx, old_offer in enumerate(self.opponent_offers[:-1]):
            age = n_opponent - idx - 1
            if age <= 8:
                self._add_constraint(constraints, old_offer, newest, 0.12 / np.sqrt(age))

        # Frequently repeated values are likely preferred. Convert this to local
        # same-context pair constraints with a very small weight.
        recent = self.opponent_offers[-12:]
        for issue_idx, values in enumerate(self.value_lists):
            counts = {value: 0 for value in values}
            for offer in recent:
                if offer[issue_idx] in counts:
                    counts[offer[issue_idx]] += 1

            best_value, best_count = max(counts.items(), key=lambda item: item[1])
            if best_count < 2:
                continue

            for offer in recent:
                if offer[issue_idx] == best_value:
                    continue
                preferred = list(offer)
                preferred[issue_idx] = best_value
                self._add_constraint(
                    constraints,
                    tuple(preferred),
                    offer,
                    0.08 * best_count / max(1, len(recent)),
                )

        return constraints[-self.max_constraints_per_update :]

    def _apply_ordinal_correction(self, constraints):
        theta = self._theta_from_current_ufun()
        base_theta = theta.copy()
        grad = np.zeros_like(theta)

        for better, worse, weight in constraints:
            diff = self._features(better) - self._features(worse)
            score = float(np.dot(theta, diff))
            z = np.clip(score - self.ordinal_margin, -40.0, 40.0)
            grad += weight * (-1.0 / (1.0 + np.exp(z))) * diff

        grad = grad / max(1, len(constraints))

        # Let ordinal information matter a bit more as history accumulates, but
        # keep it a correction layer rather than a replacement model.
        history_factor = min(1.0, len(self.opponent_offers) / 20.0)
        time_factor = 0.5 + 0.5 * float(self.relative_time)
        step = self.ordinal_lr * history_factor * time_factor

        theta = theta - step * grad
        max_delta = self.ordinal_max_delta * max(0.25, history_factor)
        theta = np.clip(theta, base_theta - max_delta, base_theta + max_delta)
        theta = np.maximum(theta, 1e-6)
        self._ordinal_theta = self._normalize_theta(theta)

    def _normalize_theta(self, theta):
        issue_masses = np.array(
            [float(np.sum(theta[s])) for s in self.slices], dtype=float
        )
        total = float(np.sum(issue_masses))
        if total <= 1e-12:
            issue_masses = np.full(len(self.slices), 1.0 / max(1, len(self.slices)))
        else:
            issue_masses = issue_masses / total

        # Prevent ordinal correction from zeroing an issue weight.
        issue_masses = np.maximum(issue_masses, 0.03)
        issue_masses = issue_masses / float(np.sum(issue_masses))

        normalized = np.zeros_like(theta)
        for issue_idx, s in enumerate(self.slices):
            local = theta[s]
            local_total = float(np.sum(local))
            if local_total <= 1e-12:
                local = np.full(len(local), 1.0 / max(1, len(local)))
            else:
                local = local / local_total
            normalized[s] = issue_masses[issue_idx] * local
        return normalized

    def _rebuild_predicted_ufun_from_theta(self):
        theta = getattr(self, "_ordinal_theta", None)
        if theta is None:
            return

        weights = []
        values = []
        for issue_idx, issue_values in enumerate(self.value_lists):
            local = np.asarray(theta[self.slices[issue_idx]], dtype=float)
            weight = float(np.sum(local))
            if weight <= 1e-12:
                normalized = np.full(len(issue_values), 1.0 / max(1, len(issue_values)))
            else:
                normalized = local / weight

            weights.append(weight)
            values.append(
                TableFun(
                    {
                        value: float(np.clip(normalized[value_idx], 0.0, 1.0))
                        for value_idx, value in enumerate(issue_values)
                    }
                )
            )

        total = sum(weights)
        if total <= 1e-12:
            weights = [1.0 / max(1, len(weights))] * len(weights)
        else:
            weights = [float(weight / total) for weight in weights]

        self.oppo_weights = weights
        self.oppo_values = values
        self.predicted_oppo_ufun = LinearAdditiveUtilityFunction(
            values=self.oppo_values,
            weights=self.oppo_weights,
            issues=self.oppo_issues,
        )
