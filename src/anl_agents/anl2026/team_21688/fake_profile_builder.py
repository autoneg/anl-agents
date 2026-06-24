"""
Fake profile builder for deceptive negotiation.

Generates fake cross-issue weights and within-issue option values
that are uncorrelated with true preferences while maintaining
tier-coverage constraints:
  - True tier-1 outcomes must be in fake tier-1 or tier-2.
  - True tier-2 outcomes must be in fake tier-1, tier-2, or tier-3.

Three generation methods are mixed:
  1. Noisy permutations  — shuffled true values + Gaussian noise
  2. Random samples       — Dirichlet (weights) or independent (values)
  3. Pinched profiles     — one element dominates

Polarity inversion: if the true distribution is polarised we
generate more uniform fake candidates, and vice versa.
"""

import random
import itertools
import numpy as np


class FakeProfileBuilder:

    def __init__(self, true_weights, issue_options_sorted, ufun_values,
                 rational_outcomes, true_tier1, true_tier2, true_tier3):
        self.true_weights = list(true_weights)
        self.issue_options_sorted = issue_options_sorted
        self.ufun_values = ufun_values  # list of {option: utility} per issue
        self.rational_outcomes = rational_outcomes
        self._true_tier1 = true_tier1
        self._true_tier2 = true_tier2
        self._true_tier3 = true_tier3
        self.n = len(true_weights)

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def build(self):
        """
        Build and return a fake preference profile.

        Returns:
            (fake_weights, fake_option_utils, blacklist)
        """
        if self.n <= 1:
            result = self._build_single_issue()
            self._print_stats(result[0], result[1], result[2])
            return result

        rng = random.Random()

        # Part 1 — Fake cross-issue weights
        top_weight_candidates = self._generate_weight_candidates(rng)

        # Part 2 — Fake within-issue option values
        top_value_candidates = self._generate_value_candidates(rng)

        # Part 3 — Coverage-aware selection
        fake_weights, fake_opt_utils, blacklist = self._select_by_coverage(
            top_weight_candidates, top_value_candidates
        )
        self._print_stats(fake_weights, fake_opt_utils, blacklist)
        return fake_weights, fake_opt_utils, blacklist

    def _print_stats(self, fake_weights, fake_opt_utils, blacklist):
        """Print tier coverage statistics for the built fake profile."""
        fake_tier1, fake_tier2, fake_tier3 = self._classify_tiers_for_profile(
            fake_weights, fake_opt_utils
        )

        total = len(self.rational_outcomes)
        n_t1 = len(self._true_tier1)
        n_t2 = len(self._true_tier2)

        # Count how many true-tier1 outcomes are in fake-tier1 or fake-tier2
        t1_satisfied = sum(1 for o in self._true_tier1
                           if o in fake_tier1 or o in fake_tier2)
        # Count how many true-tier2 outcomes are in fake-tier1/2/3
        t2_satisfied = sum(1 for o in self._true_tier2
                           if o in fake_tier1 or o in fake_tier2 or o in fake_tier3)

        # print(f"[FakeProfile] rational_outcomes={total}  "
        #       f"tier1_covered={t1_satisfied}/{n_t1}  "
        #       f"tier2_covered={t2_satisfied}/{n_t2}  "
        #       f"blacklist={len(blacklist)}")

    # ═══════════════════════════════════════════════════════════════
    #  Edge case: single issue
    # ═══════════════════════════════════════════════════════════════

    def _build_single_issue(self):
        fake_weights = self.true_weights[:]
        fake_option_utils = []
        for i in range(self.n):
            options = self.issue_options_sorted[i]
            n_opts = len(options)
            fake_option_utils.append(
                {opt: 1.0 - r / max(n_opts - 1, 1)
                 for r, opt in enumerate(options)}
            )
        return fake_weights, fake_option_utils, set()

    # ═══════════════════════════════════════════════════════════════
    #  Part 1 — Fake cross-issue weights
    # ═══════════════════════════════════════════════════════════════

    def _generate_weight_candidates(self, rng):
        weight_std = float(np.std(self.true_weights))
        weights_polarized = self._is_polarized(self.true_weights)
        candidates = []

        # Method 1: Noisy permutations
        for _ in range(max(80, 15 * self.n)):
            perm = rng.sample(self.true_weights, self.n)
            noise = [rng.gauss(0, weight_std * 0.35) for _ in range(self.n)]
            raw = [max(0.005, p + n) for p, n in zip(perm, noise)]
            total = sum(raw)
            candidates.append(tuple(w / total for w in raw))

        # Method 2: Dirichlet samples (polarity-aware)
        n_dir = 60 if weights_polarized else 20
        for _ in range(n_dir):
            if weights_polarized:
                alpha = [rng.uniform(2.0, 5.0) for _ in range(self.n)]
            else:
                alpha = [rng.uniform(0.2, 2.0) for _ in range(self.n)]
            raw = list(np.random.RandomState(rng.randint(0, 9999)).dirichlet(alpha))
            candidates.append(tuple(raw))

        # Method 3: Pinched profiles (polarity-aware)
        n_pin = 8 if weights_polarized else 40
        for _ in range(n_pin):
            pinch_idx = rng.randrange(self.n)
            if weights_polarized:
                raw = [rng.uniform(0.08, 0.20) for _ in range(self.n)]
                raw[pinch_idx] = rng.uniform(0.20, 0.35)
            else:
                raw = [rng.uniform(0.02, 0.10) for _ in range(self.n)]
                raw[pinch_idx] = rng.uniform(0.40, 0.72)
            total = sum(raw)
            candidates.append(tuple(w / total for w in raw))

        # Pure permutations (small n)
        if self.n <= 5:
            candidates.extend(itertools.permutations(self.true_weights))

        return self._pick_top_k_uncorrelated(
            candidates, self.true_weights, rng, k=5
        )

    # ═══════════════════════════════════════════════════════════════
    #  Part 2 — Fake within-issue option values
    # ═══════════════════════════════════════════════════════════════

    def _generate_value_candidates(self, rng):
        top_value_candidates = []

        for i in range(self.n):
            options = self.issue_options_sorted[i]
            true_vals = [self.ufun_values[i][opt] for opt in options]
            n_opts = len(options)

            if n_opts <= 1:
                single_map = {options[0]: 0.5} if n_opts == 1 else {}
                single_vals = [single_map.get(o, 0.0) for o in options]
                top_value_candidates.append([(single_vals, 0.0, 1.0)])
                continue

            is_pol = self._is_polarized(true_vals)
            val_std = float(np.std(true_vals))
            opt_candidates = []

            # Method 1: Noisy permutations
            for _ in range(max(40, 10 * n_opts)):
                perm = rng.sample(true_vals, n_opts)
                ns = max(val_std * 0.35, 0.03)
                noise = [rng.gauss(0, ns) for _ in range(n_opts)]
                raw = [max(0.01, min(0.99, p + n))
                       for p, n in zip(perm, noise)]
                opt_candidates.append(tuple(raw))

            # Method 2: Independent random values (polarity-aware)
            n_rand = 30 if is_pol else 12
            for _ in range(n_rand):
                if is_pol:
                    centre = rng.uniform(0.30, 0.70)
                    raw = [centre + rng.gauss(0, val_std * 0.15)
                           for _ in range(n_opts)]
                else:
                    raw = [rng.uniform(0.05, 0.95) for _ in range(n_opts)]
                raw = [max(0.01, min(0.99, v)) for v in raw]
                opt_candidates.append(tuple(raw))

            # Method 3: Pinched option values (polarity-aware)
            n_pin_o = 6 if is_pol else 25
            for _ in range(n_pin_o):
                pinch_idx = rng.randrange(n_opts)
                if is_pol:
                    raw = [rng.uniform(0.12, 0.28) for _ in range(n_opts)]
                    raw[pinch_idx] = rng.uniform(0.30, 0.50)
                else:
                    raw = [rng.uniform(0.01, 0.12) for _ in range(n_opts)]
                    raw[pinch_idx] = rng.uniform(0.55, 0.92)
                opt_candidates.append(tuple(raw))

            # Reversed order as an extra candidate
            opt_candidates.append(tuple(reversed(true_vals)))

            # If polarized and n_opts small, add explicit uniform
            if is_pol and n_opts <= 5:
                for _ in range(5):
                    u = [rng.uniform(0.25, 0.65) for _ in range(n_opts)]
                    opt_candidates.append(tuple(u))

            top_vals = self._pick_top_k_uncorrelated(
                opt_candidates, true_vals, rng, k=3
            )
            top_value_candidates.append(top_vals)

        return top_value_candidates

    # ═══════════════════════════════════════════════════════════════
    #  Part 3 — Coverage-aware selection
    # ═══════════════════════════════════════════════════════════════

    def _select_by_coverage(self, top_weight_candidates, top_value_candidates):
        best_combined = -1.0
        best_weights = None
        best_opt_utils = None
        best_blacklist = set()

        def _build_opt_utils_from_vals(top_vc):
            ou = []
            for j, issue_cands in enumerate(top_vc):
                v_vals = issue_cands[0][0]
                j_options = self.issue_options_sorted[j]
                ou.append({opt: float(val) for opt, val in zip(j_options, v_vals)})
            return ou

        # Try each top weight candidate with best values per issue
        for w_vals, w_rho, w_score in top_weight_candidates:
            w_total = sum(w_vals)
            w_norm = [v / w_total for v in w_vals]

            opt_utils = _build_opt_utils_from_vals(top_value_candidates)
            coverage, violations = self._evaluate_coverage(w_norm, opt_utils)
            combined_score = w_score * 0.3 + coverage * 0.7

            if combined_score > best_combined:
                best_combined = combined_score
                best_weights = w_norm
                best_opt_utils = opt_utils
                best_blacklist = violations

        # Also try alternative value candidates per issue
        for issue_idx in range(self.n):
            for v_idx in range(1, len(top_value_candidates[issue_idx])):
                v_vals = top_value_candidates[issue_idx][v_idx][0]
                v_score = top_value_candidates[issue_idx][v_idx][2]

                opt_utils = _build_opt_utils_from_vals(top_value_candidates)
                options = self.issue_options_sorted[issue_idx]
                opt_utils[issue_idx] = {opt: float(val) for opt, val in zip(options, v_vals)}

                coverage, violations = self._evaluate_coverage(
                    best_weights, opt_utils
                )
                combined_score = v_score * 0.1 + coverage * 0.9

                if combined_score > best_combined:
                    best_combined = combined_score
                    best_opt_utils = opt_utils
                    best_blacklist = violations

        return best_weights, best_opt_utils, best_blacklist

    # ═══════════════════════════════════════════════════════════════
    #  Tier classification
    # ═══════════════════════════════════════════════════════════════

    def _classify_tiers_for_profile(self, weights, option_utils):
        """Classify outcomes into three tiers given a utility profile."""
        n_out = len(self.rational_outcomes)
        ni = len(weights)

        utils = []
        for o in self.rational_outcomes:
            u = sum(weights[i] * option_utils[i].get(o[i], 0.0) for i in range(ni))
            utils.append(u)

        max_u = max(utils) if utils else 1.0

        tier1_abs = {o for o, u in zip(self.rational_outcomes, utils)
                     if u >= max_u - 0.07}
        tier2_abs = {o for o, u in zip(self.rational_outcomes, utils)
                     if u >= max_u - 0.19}
        tier3_abs = {o for o, u in zip(self.rational_outcomes, utils)
                     if u >= max_u - 0.33}

        top13 = max(1, int(n_out * 0.13))
        top28 = max(1, int(n_out * 0.28))
        top43 = max(1, int(n_out * 0.43))

        tier1_pct = set(self.rational_outcomes[:top13])
        tier2_pct = set(self.rational_outcomes[:top28])
        tier3_pct = set(self.rational_outcomes[:top43])

        tier1 = tier1_abs | tier1_pct
        tier2 = (tier2_abs | tier2_pct) - tier1
        tier3 = (tier3_abs | tier3_pct) - tier1 - tier2

        # tier1 = tier1_abs & tier1_pct
        # tier2 = (tier2_abs & tier2_pct) - tier1
        # tier3 = (tier3_abs & tier3_pct) - tier1 - tier2

        return tier1, tier2, tier3

    def _evaluate_coverage(self, fake_weights, fake_option_utils):
        """
        Evaluate tier coverage of a fake profile.
        Returns (coverage_rate, blacklist_set).
        """
        fake_tier1, fake_tier2, fake_tier3 = self._classify_tiers_for_profile(
            fake_weights, fake_option_utils
        )

        violations = set()

        for o in self._true_tier1:
            if o not in fake_tier1 and o not in fake_tier2:
                violations.add(o)

        for o in self._true_tier2:
            if o not in fake_tier1 and o not in fake_tier2 and o not in fake_tier3:
                violations.add(o)

        total_check = len(self._true_tier1) + len(self._true_tier2)
        if total_check == 0:
            return 1.0, violations

        covered = total_check - len(violations)
        return covered / total_check, violations

    # ═══════════════════════════════════════════════════════════════
    #  Static helpers
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _is_polarized(values):
        """
        Detect whether a distribution is polarized (high variance)
        or uniform (low variance).
        Uses max/min ratio; returns True if ratio > 3.0 or min < 0.001.
        """
        arr = np.array(values, dtype=float)
        if len(arr) <= 1:
            return False
        vmin, vmax = np.min(arr), np.max(arr)
        if vmin < 0.001:
            return True
        return (vmax / vmin) > 3.0

    @staticmethod
    def _pick_top_k_uncorrelated(candidates, true_values, rng, k=5, target=0.0):
        """
        From a list of candidate tuples, return top K by Spearman ρ
        closest to `target`. Returns [(values_list, rho, score), ...].

        target=0   → prefer uncorrelated (ρ ≈ 0, for option values)
        target=-1  → prefer inversely correlated (ρ ≈ -1, for weights)
        """
        true_ranks = np.argsort(np.argsort(true_values))
        scored = []
        for cand in candidates:
            cand_ranks = np.argsort(np.argsort(cand))
            rho = float(np.corrcoef(true_ranks, cand_ranks)[0, 1])
            if np.isnan(rho):
                rho = 1.0
            score = 1.0 - abs(rho - target) / 2.0
            scored.append((list(cand), rho, score))
        scored.sort(key=lambda x: -x[2])
        return scored[:k]
