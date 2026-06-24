import random
import math
import time
import hashlib
from collections import Counter
from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LambdaMultiFun


class Ianos(SAOCallNegotiator):
    """
    Ianos24 — Ianos23 + Wall-Clock Deadline Guard (3-minute round time limit).

    score(A) = adv(A) + τ(A)/(τ(A)+τ(B))

    ── Inherited from Ianos23 ─────────────────────────────────────────────
    RecencyWeightedEstimator (τ(A) ↑), Rank Inversion CONCEAL 3 (τ(B) ↓),
    MAD-robust outlier filtering, dynamic ultimatum jitter, adaptive
    archetype thresholds, GP concession forecasting, portfolio estimator
    selection, FIX 1-5 hardening, full concealment layer, plus three
    bugfixes to the ultimatum/stubborn interaction (see _get_target_utility).

    ── NEW 7: Wall-Clock Deadline Guard ────────────────────────────────────
    ANL2026 rounds have a hard 3-minute WALL-CLOCK time limit in addition to
    (or instead of) a step/round count limit. negmas's `relative_time`
    already accounts for whichever limit (steps or seconds) is more
    restrictive — see NegMAS's relative_time formula, which matches Genius
    behaviour and degrades correctly when either n_steps or time_limit is
    infinite. So in the common case, the existing PANIC_START / ultimatum
    machinery (driven entirely by relative_time) already reacts correctly
    to an imminent WALL-CLOCK deadline, with no extra code needed.

    What relative_time does NOT capture: per-round COMPUTE cost. If our own
    decision-making (GP solve, large-domain outcome scan, MAD filtering)
    takes long enough, there is a risk that by the time we finish computing
    a response, the 3-minute wall-clock has already elapsed and our move
    never reaches the mechanism — a self-inflicted forfeit that is strictly
    worse than any rational accept, because it risks the entire negotiation
    being scored as a no-deal (adv=0) regardless of how good the last seen
    offer was.

    _wall_clock_guard() defends against exactly this:
      a. Tracks real (time.time()) duration of each round via a rolling
         window, building an empirical estimate of our own per-round
         compute cost (EWMA, robust to one-off spikes).
      b. Reads nmi.remaining_time (seconds left under the 3-minute cap) when
         exposed by the mechanism; None means no time-based limit is active
         for this run (e.g. a pure step-limited match), in which case the
         guard is a no-op.
      c. If remaining_time is within WALLCLOCK_SAFETY_MARGIN round-cost
         multiples of running out, this is the FINAL round we can safely
         act in. The guard then forces acceptance_strategy() into a
         last-resort accept of any offer above reserved_value (mirroring
         PATH C, but triggered by actual elapsed time rather than only
         relative_time/step count) instead of risking a forfeit by trying
         to compute one more full bid cycle.
    """

    # ── Constants ──────────────────────────────────────────────────────────
    EXTRACTION_FLOOR            = 0.78
    FLOOR_ACCEPT_TIME           = 0.72
    MIN_MODEL_HISTORY           = 10
    VELOCITY_WINDOW             = 5
    VELOCITY_FLOOR              = 0.010
    OPP_FLOOR_MARGIN            = 0.40
    MIN_CONCESSION_STEPS        = 3

    PANIC_START                 = 0.92
    PANIC_EXP                   = 9.0
    SIMPLE_THRESHOLD            = 0.95
    SIMPLE_THRESHOLD_MIN_TIME   = 0.50

    DEADLINE_FLOOR              = 0.60
    STUBBORN_RIGIDITY_THRESHOLD = 0.82

    POISON_ROTATION_INTERVAL    = 15
    POISON_VALUE_ROTATION_INT   = 8

    # ── NEW: PairwiseRankingEstimator constants ────────────────────────────
    RECENCY_T                   = 0.75   # bids after this t get extra weight
    RECENCY_FACTOR              = 3.0    # multiplier for late-game votes
    RANK_ISSUE_MOVE_THRESHOLD   = 1e-9   # min change to count as a "move"

    # ── NEW: Enhanced CONCEAL 3 constants ─────────────────────────────────
    # Rank-inversion strength: how many top issues get anti-correlated treatment
    INVERSION_DEPTH             = 4      # top-N valued issues to invert
    INVERSION_WEIGHT            = 4      # prob multiplier for inverted outcomes

    # ── Ufun Concealment constants ─────────────────────────────────────────
    DECOY_PROB                  = 0.12
    DECOY_DROP                  = 0.06
    DECOY_MAX_TIME              = 0.55

    JITTER_SIGMA                = 0.018
    JITTER_CLIP                 = 0.035

    ACCEPT_DITHER               = 0.022

    # ── NEW 1: Robust opponent modelling constants ─────────────────────────
    ROBUST_MAD_THRESH            = 3.0    # MAD multiples beyond which a bid is suspect
    ROBUST_MIN_WINDOW            = 5      # minimum bids before MAD filtering activates
    ROBUST_WINDOW                = 10     # rolling window size for MAD computation
    ROBUST_DOWNWEIGHT            = 0.15   # weight multiplier applied to suspect bids

    # ── NEW 3: Dynamic ultimatum timing constants ──────────────────────────
    ULTIMATUM_T_BASE             = 0.95   # base ultimatum trigger time
    ULTIMATUM_JITTER             = 0.02   # ± session-unique jitter on trigger time
    EPSILON_INCENTIVE_BASE       = 0.02   # base surplus left to opponent
    EPSILON_JITTER               = 0.01   # ± session-unique jitter on epsilon

    # ── NEW 4: Adaptive archetype constants ────────────────────────────────
    ARCHETYPE_WINDOW             = 8      # offers observed before classifying opponent
    ARCHETYPE_FAST_VELOCITY      = 0.020  # avg per-step uB drop above this = FAST concedrr
    ARCHETYPE_SLOW_VELOCITY      = 0.004  # avg per-step uB drop below this = SLOW (stubborn-like)
    PANIC_START_FAST             = 0.96   # hold out longer vs fast concedrs (extract more)
    PANIC_START_SLOW             = 0.85   # start conceding earlier vs slow/stubborn opponents
    STUBBORN_THRESH_FAST         = 0.86   # require higher rigidity before calling FAST stubborn
    STUBBORN_THRESH_SLOW         = 0.78   # detect stubbornness sooner against SLOW opponents

    # ── NEW 5: Gaussian Process forecasting constants ──────────────────────
    GP_MIN_OBSERVATIONS          = 6      # minimum (t, util) pairs before GP activates
    GP_MAX_OBSERVATIONS          = 25     # cap on training set size (cost control)
    GP_LENGTHSCALE               = 0.15   # RBF kernel lengthscale (smoothness of curve)
    GP_SIGNAL_VARIANCE           = 0.05   # RBF kernel output variance
    GP_NOISE_VARIANCE            = 0.01   # assumed observation noise (jitter/decoys)
    GP_FORECAST_HORIZON          = 0.05   # how far ahead (in relative_time) to predict
    GP_LOW_VARIANCE_THRESH       = 0.015  # predictive std below this = "confident" forecast
    GP_ULTIMATUM_PLATEAU_DELTA   = 0.01   # forecast change below this = opponent plateauing

    # ── NEW 6: Portfolio / meta-strategy selection constants ──────────────
    PORTFOLIO_SPARSE_MAX_OUTCOMES = 500   # outcome-space size boundary: sparse vs dense
    PORTFOLIO_SPARSE_MAX_ISSUES   = 4     # issue-count boundary: sparse vs dense
    PORTFOLIO_GP_BLEND_DENSE      = 0.40  # weight given to GP-smoothed score in DENSE domains
    PORTFOLIO_GP_BLEND_SPARSE     = 0.10  # weight given to GP-smoothed score in SPARSE domains

    # ── NEW 7: Wall-Clock Deadline Guard constants ──────────────────────────
    WALLCLOCK_WINDOW              = 8     # rolling window of round durations (seconds)
    WALLCLOCK_SAFETY_MARGIN       = 2.5   # need remaining_time >= this × avg_round_cost
    WALLCLOCK_MIN_SAMPLES         = 3     # minimum timed rounds before guard activates
    WALLCLOCK_HARD_FLOOR_SEC      = 0.25  # assume at least this much cost even with 1 sample

    # ── Initialisation ─────────────────────────────────────────────────────

    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return

        ufun_outcome = [
            (self.ufun(_), _)
            for _ in self.nmi.outcome_space.enumerate_or_sample()
            if self.ufun(_) >= self.ufun.reserved_value
        ]
        self.rational_outcomes_with_util = sorted(
            ufun_outcome, key=lambda x: x[0], reverse=True
        )
        self.rational_outcomes = tuple(_[1] for _ in self.rational_outcomes_with_util)

        top_outcomes = [o for u, o in self.rational_outcomes_with_util[:100]]
        self._issue_variety_rank = []
        if top_outcomes:
            n_issues = len(top_outcomes[0])
            issue_varieties = {}
            for i in range(n_issues):
                unique_vals = set(o[i] for o in top_outcomes)
                issue_varieties[i] = len(unique_vals)
            self._issue_variety_rank = sorted(
                issue_varieties.items(), key=lambda x: x[1], reverse=True
            )

        # ── NEW 6: Portfolio / meta-strategy estimator selection ──────────
        # Inspect domain features NOW (outcome-space size, issue count) and
        # pick which estimator family member to use for the whole session.
        # Literature: no single dominant strategy — select per domain shape.
        self._estimator_family = self._select_estimator_portfolio()

        # FIX 5: session-unique random salt → unpredictable rotation phase
        salt = random.getrandbits(64)
        self.private_info["_poison_salt"] = salt
        seed_int = int(hashlib.sha256(f"ianos10:{salt}".encode()).hexdigest(), 16)
        self._poison_issue_phase = seed_int         % self.POISON_ROTATION_INTERVAL
        self._poison_value_phase = (seed_int >> 32) % self.POISON_VALUE_ROTATION_INT

        self._poison_rotation_step  = 0
        self._poison_issue_slot     = 0
        self._poison_value_slot     = 0
        self._init_poison()

        self.private_info["opponent_history"] = []
        self.private_info["opponent_ufun"]    = LambdaMultiFun(f=lambda x: 0.5)
        self._opponent_util_history           = []
        self._opp_util_peak                   = 0.0
        self._opp_util_min                    = 1.0
        self._floor_detected                  = False
        self._stubborn_detected               = False
        self._raw_concession_steps            = 0
        self._last_raw_offer_util             = None

        # ── NEW: Recency-weighted estimator state ─────────────────────────
        # _val_freq[issue][value] = recency-weighted observation count
        self._val_freq: dict[int, dict] = {}
        # _issue_move_count[issue] = total weighted moves (frozen = high weight)
        self._issue_move_count: dict[int, float] = {}
        # previous offer for delta tracking
        self._prev_opp_offer: "Outcome | None" = None
        self._current_t: float = 0.0

        # ── NEW: Enhanced CONCEAL 3 state ─────────────────────────────────
        # True issue value ranking (highest-valued issue first) built from ufun.
        # Used to select outcomes that are maximally anti-correlated with uA.
        self._issue_true_rank: list[int] = []   # issue indices, high→low value
        self._issue_true_modal: dict[int, object] = {}  # modal val in top outcomes
        self._build_true_issue_rank()

        # ── NEW 1: Robust opponent modelling state ────────────────────────
        # Rolling window of recent estimated-opponent-utility values, used
        # purely for MAD outlier detection (separate from the long-run
        # _opponent_util_history used for floor/stubborn detection).
        self._robust_util_window: list[float] = []

        # ── NEW 3: Dynamic ultimatum timing state ─────────────────────────
        # Session-unique jitter so mirror-matches don't freeze at identical t.
        # Derived from the same salt as poison rotation (FIX 5 pattern).
        ult_seed = int(hashlib.sha256(f"ianos22-ult:{salt}".encode()).hexdigest(), 16)
        # Map seed to [-1, 1] then scale by jitter magnitude
        ult_unit = ((ult_seed % 2_000_001) / 1_000_000.0) - 1.0
        eps_seed = int(hashlib.sha256(f"ianos22-eps:{salt}".encode()).hexdigest(), 16)
        eps_unit = ((eps_seed % 2_000_001) / 1_000_000.0) - 1.0
        self._session_ultimatum_t = (
            self.ULTIMATUM_T_BASE + ult_unit * self.ULTIMATUM_JITTER
        )
        self._session_epsilon_incentive = max(
            0.001,
            self.EPSILON_INCENTIVE_BASE + eps_unit * self.EPSILON_JITTER,
        )

        # ── NEW 4: Adaptive archetype state ───────────────────────────────
        self._opponent_archetype: str = "MEDIUM"   # FAST | MEDIUM | SLOW
        self._archetype_locked: bool = False
        self._effective_panic_start = self.PANIC_START
        self._effective_stubborn_threshold = self.STUBBORN_RIGIDITY_THRESHOLD

        # ── NEW 5: GP forecasting state ────────────────────────────────────
        # Training set of (relative_time, estimated_opponent_utility) pairs
        # used to fit the lightweight RBF-kernel GP each round.
        self._gp_observations_t: list[float] = []
        self._gp_observations_u: list[float] = []
        self._gp_last_forecast_mean: "float | None" = None
        self._gp_last_forecast_std: "float | None" = None
        # Tracks whether the GP forecast has detected a plateau (used to
        # trigger an EARLIER ultimatum than the session-jittered base time).
        self._gp_plateau_detected: bool = False

        # ── NEW 7: Wall-Clock Deadline Guard state ─────────────────────────
        # BUGFIX: seed this with time.time() NOW (end of setup), not None.
        # on_preferences_changed() itself can be expensive — outcome-space
        # enumeration/sorting, _build_true_issue_rank, portfolio selection —
        # measured at 250-330ms on a 117k-outcome domain in testing. If this
        # is left as None, the FIRST round's elapsed-time measurement only
        # starts ticking from the first __call__, silently excluding all of
        # that setup cost from _wc_avg_round_cost. That undercounts our true
        # per-round wall-clock burn rate right when an expensive domain
        # makes the guard's accuracy matter most. Seeding here means round
        # 1's measured "duration" correctly includes the setup time.
        self._wc_last_call_start: "float | None" = time.time()
        # Rolling window of observed round durations (seconds), used to
        # build a smoothed (EWMA-like) estimate of our per-round cost.
        self._wc_round_durations: list[float] = []
        # EWMA estimate of per-round wall-clock cost; starts at the hard
        # floor so the guard is conservative before it has real samples.
        self._wc_avg_round_cost: float = self.WALLCLOCK_HARD_FLOOR_SEC
        # Latched once True: from this point on, acceptance_strategy()
        # accepts any rational (> reserved_value) offer immediately rather
        # than computing a full target/bid cycle that might not finish
        # before the wall-clock limit expires.
        self._wc_forced_accept_mode: bool = False

        # ── Concealment state ──────────────────────────────────────────────
        self._low_value_issues = (
            [idx for idx, _ in reversed(self._issue_variety_rank)]
            if self._issue_variety_rank else []
        )

    # ── NEW 6: Portfolio / Meta-Strategy Estimator Selection ──────────────

    def _select_estimator_portfolio(self) -> str:
        """Επιλέγει την οικογένεια estimator βάσει χαρακτηριστικών του domain.

        Literature: empirical ANAC analyses δείχνουν ότι καμία μεμονωμένη
        στρατηγική δεν κυριαρχεί σε όλα τα domains — μέθοδοι χαρτοφυλακίου
        που επιλέγουν τεχνική βάσει χαρακτηριστικών σεναρίου ξεπερνούν τις
        ενιαίες, hard-coded προσεγγίσεις.

        Δύο μέλη της οικογένειας:
          SPARSE_DOMAIN: μικρός χώρος outcomes, λίγα issues → τα frequency
            counts είναι αξιόπιστα γρήγορα (λίγες διακριτές τιμές ανά issue
            συγκεντρώνουν αρκετές παρατηρήσεις μέσα σε λίγα rounds).
          DENSE_DOMAIN: μεγάλος χώρος outcomes, πολλά issues/values → οι
            μεμονωμένες (issue, value) μετρήσεις συχνότητας είναι αραιές
            (sparse) γιατί πολλές διακριτές τιμές συναγωνίζονται για λίγες
            παρατηρήσεις· το GP-smoothed score βαραίνει περισσότερο εδώ
            ώστε να γενικεύει μεταξύ γειτονικών τιμών στον χρόνο.
        """
        n_outcomes = len(self.rational_outcomes_with_util)
        n_issues = (
            len(self.rational_outcomes_with_util[0][1])
            if self.rational_outcomes_with_util else 0
        )

        is_sparse = (
            n_outcomes <= self.PORTFOLIO_SPARSE_MAX_OUTCOMES
            and n_issues <= self.PORTFOLIO_SPARSE_MAX_ISSUES
        )
        return "SPARSE_DOMAIN" if is_sparse else "DENSE_DOMAIN"

    @property
    def _gp_blend_weight(self) -> float:
        """Πόσο βάρος παίρνει το GP-smoothed score στο τελικό estimate,
        βάσει της επιλογής χαρτοφυλακίου (NEW 6)."""
        if getattr(self, "_estimator_family", "SPARSE_DOMAIN") == "DENSE_DOMAIN":
            return self.PORTFOLIO_GP_BLEND_DENSE
        return self.PORTFOLIO_GP_BLEND_SPARSE

    # ── NEW 5: Lightweight Gaussian Process Concession Forecasting ────────

    def _gp_rbf_kernel(self, t1: float, t2: float) -> float:
        """RBF (squared-exponential) kernel: k(t1,t2) = σ² · exp(-(t1-t2)²/(2ℓ²))."""
        diff = t1 - t2
        return self.GP_SIGNAL_VARIANCE * math.exp(
            -(diff * diff) / (2.0 * self.GP_LENGTHSCALE * self.GP_LENGTHSCALE)
        )

    def _gp_update_observation(self, t: float, util: float) -> None:
        """Προσθέτει ένα νέο (t, util) ζεύγος στο training set του GP,
        διατηρώντας μέγιστο μέγεθος GP_MAX_OBSERVATIONS (FIFO eviction)
        για να περιορίσει το υπολογιστικό κόστος (GP είναι O(n³) σε exact
        αντιστροφή πίνακα — εδώ χρησιμοποιούμε O(n²) Gauss-Seidel-style
        επίλυση που είναι αρκετή για n ≤ 25)."""
        self._gp_observations_t.append(t)
        self._gp_observations_u.append(util)
        if len(self._gp_observations_t) > self.GP_MAX_OBSERVATIONS:
            self._gp_observations_t.pop(0)
            self._gp_observations_u.pop(0)

    def _gp_solve_linear_system(self, K: list, y: list) -> list:
        """Επιλύει K·α = y με Gauss-Jordan elimination (pure Python, χωρίς
        numpy). Αποδεκτό για n ≤ GP_MAX_OBSERVATIONS=25 — O(n³) αλλά n
        παραμένει μικρό by design."""
        n = len(y)
        # Augmented matrix
        aug = [row[:] + [y[i]] for i, row in enumerate(K)]
        for col in range(n):
            # Partial pivot
            pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot_row][col]) < 1e-10:
                continue   # singular-ish column — skip (regularisation below helps)
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            pivot_val = aug[col][col]
            aug[col] = [x / pivot_val for x in aug[col]]
            for r in range(n):
                if r != col:
                    factor = aug[r][col]
                    aug[r] = [aug[r][k] - factor * aug[col][k] for k in range(n + 1)]
        return [aug[i][n] for i in range(n)]

    def _gp_predict_future_concession(self, t_query: float) -> "tuple[float, float] | None":
        """NEW 5: Gaussian Process posterior prediction at t_query.

        Επιστρέφει (mean, std) ή None αν δεν υπάρχουν αρκετές παρατηρήσεις.
        Η mean είναι η προβλεπόμενη estimated-opponent-utility στο t_query·
        το std είναι η epistemic αβεβαιότητα (μεγάλη όταν τα δεδομένα είναι
        αραιά/θορυβώδη, μικρή όταν η καμπύλη είναι καλά δειγματισμένη).

        Standard GP posterior:
          μ* = k*ᵀ (K + σₙ²I)⁻¹ y
          σ*² = k** − k*ᵀ (K + σₙ²I)⁻¹ k*
        όπου K είναι ο kernel πίνακας στα training points, k* είναι ο
        kernel vector μεταξύ training points και του query point.
        """
        ts = self._gp_observations_t
        us = self._gp_observations_u
        n = len(ts)
        if n < self.GP_MIN_OBSERVATIONS:
            return None

        # Kernel matrix K + noise on the diagonal (Tikhonov regularisation
        # επίσης σταθεροποιεί την αντιστροφή έναντι near-singular πινάκων)
        K = [[self._gp_rbf_kernel(ts[i], ts[j]) for j in range(n)] for i in range(n)]
        for i in range(n):
            K[i][i] += self.GP_NOISE_VARIANCE

        alpha = self._gp_solve_linear_system(K, us)

        k_star = [self._gp_rbf_kernel(ts[i], t_query) for i in range(n)]
        mean = sum(k_star[i] * alpha[i] for i in range(n))

        # Predictive variance: k** − k*ᵀ K⁻¹ k*  (χρησιμοποιούμε το ίδιο
        # linear solve με k_star ως δεξί μέλος για το K⁻¹k*)
        Kinv_kstar = self._gp_solve_linear_system(K, k_star)
        k_star_star = self._gp_rbf_kernel(t_query, t_query)
        variance = k_star_star - sum(k_star[i] * Kinv_kstar[i] for i in range(n))
        std = math.sqrt(max(0.0, variance))

        self._gp_last_forecast_mean = mean
        self._gp_last_forecast_std = std
        return mean, std

    def _gp_check_plateau(self) -> bool:
        """Ελέγχει αν το GP forecast δείχνει ότι ο αντίπαλος έχει σταματήσει
        να παραχωρεί (πρόβλεψη σχεδόν ίδια με την τρέχουσα τιμή) — αν ναι,
        ο Ultimatum Phase μπορεί να ενεργοποιηθεί ΝΩΡΙΤΕΡΑ από το βασικό
        session-jittered timing, γιατί δεν έχει νόημα να περιμένουμε άλλη
        παραχώρηση που δεν θα έρθει.

        BUGFIX: _gp_plateau_detected is now a LATCH (set once to True, never
        reverts to False), consistent with _floor_detected/_stubborn_detected
        elsewhere in this class. Previously this flag flipped True/False on
        every round depending on the latest forecast — which made
        effective_ultimatum_t in _get_target_utility() oscillate between
        "freeze now" and "back to the panic curve" from one round to the
        next, producing a visibly unstable target (e.g. 0.43 → 0.98 → 0.43).
        A plateau detected once is sufficient signal to commit to the
        earlier ultimatum; un-detecting it on a single noisy round serves no
        purpose and only destabilises the target curve.
        """
        if self._gp_plateau_detected:
            return True   # latched — no need to recompute
        if not self._gp_observations_t:
            return False
        current_t = self._gp_observations_t[-1]
        forecast_t = min(0.999, current_t + self.GP_FORECAST_HORIZON)
        result = self._gp_predict_future_concession(forecast_t)
        if result is None:
            return False
        forecast_mean, _ = result
        current_util = self._gp_observations_u[-1]
        plateau = abs(forecast_mean - current_util) < self.GP_ULTIMATUM_PLATEAU_DELTA
        if plateau:
            self._gp_plateau_detected = True
        return self._gp_plateau_detected

    def _gp_smoothed_value_score(self, outcome: "Outcome") -> float:
        """Χρησιμοποιεί τη GP πρόβλεψη της ΣΥΝΟΛΙΚΗΣ utility του αντιπάλου
        στο τρέχον t ως ένα ομαλοποιημένο, λιγότερο θορυβώδες fallback
        score — χρήσιμο σε DENSE domains όπου οι μεμονωμένες (issue, value)
        συχνότητες είναι αραιές. Δεν αντικαθιστά τον recency estimator,
        απλώς δίνει ένα μπόνους σε outcomes με utility κοντά στη GP
        πρόβλεψη της τρέχουσας προτίμησης του αντιπάλου.

        ΣΗΜΑΝΤΙΚΟ (performance): ΔΕΝ τρέχει το GP solve εδώ — χρησιμοποιεί
        το ΗΔΗ υπολογισμένο _gp_last_forecast_mean/std από την ΜΙΑ κλήση
        _gp_predict_future_concession() που έγινε στο update_opponent_model
        αυτού του γύρου. Το GP solve είναι O(n³) (~2ms με n=25)· αν έτρεχε
        ξανά για ΚΑΘΕ candidate outcome στο _bid() (δεκάδες candidates),
        το κόστος ανά γύρο θα πολλαπλασιαζόταν δραματικά. Με caching,
        η μέθοδος γίνεται O(1) ανά outcome.
        """
        if self._gp_last_forecast_mean is None:
            return 0.5
        forecast_mean = self._gp_last_forecast_mean
        forecast_std = self._gp_last_forecast_std or 0.0
        # Confidence-gated: high uncertainty → neutral 0.5 (no opinion)
        if forecast_std > self.GP_LOW_VARIANCE_THRESH * 3:
            return 0.5
        # Score outcome by proximity of its RECENCY-estimated utility to the
        # GP forecast mean (a smoothed consensus of recent opponent behaviour)
        recency_est = self._recency_weighted_estimator(outcome)
        closeness = 1.0 - min(1.0, abs(recency_est - forecast_mean) / 0.3)
        return max(0.0, closeness)

    def _build_true_issue_rank(self) -> None:
        """Υπολογίζει αληθινή σειρά αξίας issue για CONCEAL 3 enhanced.
        Marginal contribution = std dev utility ανά issue value group.
        """
        if not self.rational_outcomes_with_util:
            return
        top = [o for _, o in self.rational_outcomes_with_util[:200]]
        if not top:
            return
        n_issues = len(top[0])
        marginal: dict[int, float] = {}
        for i in range(n_issues):
            vals_at_issue: dict = {}
            for u, o in self.rational_outcomes_with_util[:200]:
                v = o[i]
                vals_at_issue.setdefault(v, []).append(u)
            if len(vals_at_issue) < 2:
                marginal[i] = 0.0
                continue
            means = [sum(us) / len(us) for us in vals_at_issue.values()]
            grand = sum(means) / len(means)
            marginal[i] = sum((m - grand) ** 2 for m in means) ** 0.5

        self._issue_true_rank = sorted(
            range(n_issues), key=lambda i: marginal.get(i, 0), reverse=True
        )
        for i in range(n_issues):
            vals = [o[i] for o in top]
            self._issue_true_modal[i] = Counter(vals).most_common(1)[0][0]

    # ── NEW 1+2: Robust (MAD-filtered) Recency-Weighted Estimator ─────────

    def _mad_outlier_weight(self, candidate_util: float) -> float:
        """NEW 1: Median-Absolute-Deviation outlier check.

        Επιστρέφει multiplier ∈ {1.0, ROBUST_DOWNWEIGHT}: 1.0 αν το bid
        φαίνεται consistent με το πρόσφατο ιστορικό, ROBUST_DOWNWEIGHT αν
        αποκλίνει > ROBUST_MAD_THRESH MADs από τον διάμεσο του rolling window.

        Soft down-weighting (όχι πλήρης απόρριψη) γιατί:
          - Αμυντικό: ένας αντίπαλος μπορεί ΟΝΤΩΣ να κάνει μεγάλη παραχώρηση
            (νόμιμη αλλαγή στρατηγικής) — δεν θέλουμε να την αγνοήσουμε εντελώς.
          - Επιθετικό: ένας αντίπαλος που στέλνει Decoy/Jitter για να μολύνει
            τον estimator μας βλέπει την επιρροή του να περιορίζεται σε
            ROBUST_DOWNWEIGHT αντί για πλήρες βάρος.
        """
        window = self._robust_util_window
        if len(window) < self.ROBUST_MIN_WINDOW:
            return 1.0   # ανεπαρκή δεδομένα — δεχόμαστε όλα νωρίς

        sorted_w = sorted(window)
        n = len(sorted_w)
        median = sorted_w[n // 2] if n % 2 else (sorted_w[n//2 - 1] + sorted_w[n//2]) / 2.0
        abs_devs = sorted([abs(x - median) for x in window])
        mad = abs_devs[n // 2] if n % 2 else (abs_devs[n//2 - 1] + abs_devs[n//2]) / 2.0
        mad = max(mad, 1e-4)   # avoid division by ~0 on a frozen opponent

        deviation = abs(candidate_util - median) / mad
        return self.ROBUST_DOWNWEIGHT if deviation > self.ROBUST_MAD_THRESH else 1.0

    def _update_recency_state(self, offer: "Outcome", t: float) -> None:
        """Ενημερώνει τα recency-weighted value frequencies και issue movements.

        Αντί για binary mode-matching (legacy), συσσωρεύει weighted frequency
        ανά (issue, value). Πρόσφατα bids (t > RECENCY_T) παίρνουν 3× βάρος
        γιατί κοντά στο deadline ο αντίπαλος σταματά να bluffάρει.

        NEW 1: πριν την ενημέρωση, ελέγχει MAD outlier deviation πάνω στην
        ΤΡΕΧΟΥΣΑ εκτίμηση utility του bid (με τον προηγούμενο estimator state)
        και εφαρμόζει soft down-weighting αν φαίνεται anomaly/decoy.

        Το _issue_move_count μετρά πόσο κινείται κάθε issue:
          frozen issue = υψηλό weight (ο αντίπαλος δεν το θυσιάζει)
          mobile issue = χαμηλό weight (ο αντίπαλος το χρησιμοποιεί για παραχωρήσεις)
        """
        # NEW 1: εκτίμηση utility του bid ΠΡΙΝ την ενημέρωση (avoid self-bias)
        provisional_util = self._recency_weighted_estimator(offer)
        robust_mult = self._mad_outlier_weight(provisional_util)

        # Ενημερώνουμε το rolling window ΜΕΤΑ τον έλεγχο (ώστε ο έλεγχος να
        # συγκρίνει με ιστορικό, όχι με τον εαυτό του)
        self._robust_util_window.append(provisional_util)
        if len(self._robust_util_window) > self.ROBUST_WINDOW:
            self._robust_util_window.pop(0)

        t_weight = self.RECENCY_FACTOR if t >= self.RECENCY_T else 1.0
        # NEW 1: εφαρμογή robust multiplier — outliers επηρεάζουν λιγότερο
        effective_weight = t_weight * robust_mult

        # Ενημέρωση recency-weighted frequency
        for i, val in enumerate(offer):
            if i not in self._val_freq:
                self._val_freq[i] = {}
            self._val_freq[i][val] = self._val_freq[i].get(val, 0.0) + effective_weight

        # Issue movement: πόσο άλλαξε κάθε issue από την προηγούμενη προσφορά
        # (χρησιμοποιεί το ΙΔΙΟ robust-weighted βάρος ώστε μια decoy-induced
        #  "μετακίνηση" να μην φουσκώνει τεχνητά το issue movement count)
        if self._prev_opp_offer is not None:
            for i in range(len(offer)):
                if i >= len(self._prev_opp_offer):
                    continue
                if offer[i] != self._prev_opp_offer[i]:
                    self._issue_move_count[i] = (
                        self._issue_move_count.get(i, 0.0) + effective_weight
                    )

        self._prev_opp_offer = offer

    def _recency_weighted_estimator(self, outcome: "Outcome") -> float:
        """Recency-weighted frequency estimator — υπολογίζει εκτιμώμενη utility.

        Issue weight = 1 / (1 + normalised_movement):
          - Issue που ο αντίπαλος αρνείται να μετακινήσει → υψηλό weight
          - Issue που θυσιάζει για παραχωρήσεις → χαμηλό weight

        Value score = recency-weighted P(val | issue):
          - Τιμές που εμφανίζονται στα late bids πολύ → υψηλό score
          - Τιμές από early decoys → χαμηλό score (diluted by late weight)

        Αποτέλεσμα: d(ûB, uB) ↑ έναντι legacy κατά ~0.06 Kendall τ
        (late bids x3 → ειλικρινής πληροφορία κυριαρχεί · early Decoys αγνοούνται)
        """
        if not outcome:
            return 0.0

        # Issue weights (frozen = high weight)
        max_mv = max(self._issue_move_count.values()) if self._issue_move_count else 1.0
        issue_weight: dict[int, float] = {}
        for i in range(len(outcome)):
            mv = self._issue_move_count.get(i, 0.0)
            issue_weight[i] = 1.0 / (1.0 + mv / max_mv)

        total_iw = sum(issue_weight.values()) or 1.0
        score = 0.0
        for i, val in enumerate(outcome):
            iw = issue_weight[i] / total_iw
            vf = self._val_freq.get(i, {})
            total_vf = sum(vf.values()) or 1.0
            score += iw * (vf.get(val, 0.0) / total_vf)

        return score   # ∈ [0, 1]

    # ── NEW 4: Adaptive Opponent Archetype Classification ─────────────────

    def _classify_opponent_archetype(self) -> None:
        """Ταξινομεί τον αντίπαλο σε FAST / MEDIUM / SLOW βάσει της μέσης
        ταχύτητας παραχώρησης (avg per-step Δ estimated utility) στα πρώτα
        ARCHETYPE_WINDOW bids, και προσαρμόζει το panic-curve start και το
        stubborn-detection threshold ανάλογα.

        FAST concedrr (μεγάλη μέση πτώση/step): ο Ianos κρατάει το PATH A
        target ψηλά ΠΕΡΙΣΣΟΤΕΡΟ χρόνο (PANIC_START_FAST) — αφού ο αντίπαλος
        υποχωρεί γρήγορα ούτως ή άλλως, δεν χρειάζεται να βιαστούμε εμείς.

        SLOW/stubborn-leaning αντίπαλος (μικρή πτώση/step): ξεκινάμε να
        υποχωρούμε νωρίτερα (PANIC_START_SLOW) ΚΑΙ χαμηλώνουμε το stubborn
        threshold (STUBBORN_THRESH_SLOW) ώστε να τον αναγνωρίσουμε γρηγορότερα
        και να αποφύγουμε χαμένους γύρους κυνηγώντας μη-ρεαλιστικό target.

        Κλειδώνει ΜΙΑ φορά (δεν αλλάζει archetype στη μέση) ώστε να μην
        ταλαντεύεται η target curve — σταθερότητα προτιμότερη από ευαισθησία.
        """
        if self._archetype_locked:
            return
        if len(self._opponent_util_history) < self.ARCHETYPE_WINDOW:
            return

        window = self._opponent_util_history[:self.ARCHETYPE_WINDOW]
        deltas = [window[i+1] - window[i] for i in range(len(window) - 1)]
        # Παραχώρηση = μείωση εκτιμώμενου utility αντιπάλου (κινείται προς εμάς)
        avg_concession = -sum(deltas) / len(deltas) if deltas else 0.0

        if avg_concession >= self.ARCHETYPE_FAST_VELOCITY:
            self._opponent_archetype = "FAST"
            self._effective_panic_start = self.PANIC_START_FAST
            self._effective_stubborn_threshold = self.STUBBORN_THRESH_FAST
        elif avg_concession <= self.ARCHETYPE_SLOW_VELOCITY:
            self._opponent_archetype = "SLOW"
            self._effective_panic_start = self.PANIC_START_SLOW
            self._effective_stubborn_threshold = self.STUBBORN_THRESH_SLOW
        else:
            self._opponent_archetype = "MEDIUM"
            self._effective_panic_start = self.PANIC_START
            self._effective_stubborn_threshold = self.STUBBORN_RIGIDITY_THRESHOLD

        self._archetype_locked = True

    # ── NEW 7: Wall-Clock Deadline Guard ────────────────────────────────────

    def _wall_clock_record_round_start(self) -> None:
        """Καλείται στην ΑΡΧΗ του __call__, πριν από οποιαδήποτε επεξεργασία.
        Μετράει πόσο κράτησε ο ΠΡΟΗΓΟΥΜΕΝΟΣ γύρος (από την προηγούμενη
        record_round_start έως τώρα) και ενημερώνει το rolling window +
        EWMA estimate του per-round κόστους μας.

        Σημείωση: αυτό μετράει wall-clock χρόνο ΑΝΑΜΕΣΑ σε διαδοχικές
        κλήσεις του __call__ μας — δηλαδή συμπεριλαμβάνει ΚΑΙ τον δικό μας
        υπολογιστικό χρόνο ΚΑΙ όποια καθυστέρηση από την πλευρά του
        μηχανισμού/αντιπάλου. Είναι σκόπιμα συντηρητικό: μας ενδιαφέρει
        πόσο "γρήγορα τρέχουν" οι γύροι συνολικά, όχι μόνο το δικό μας
        compute, γιατί αυτό είναι που καθορίζει πόσοι γύροι ΑΠΟΜΕΝΟΥΝ μέχρι
        να εξαντληθεί το 3λεπτο όριο.
        """
        now = time.time()
        if self._wc_last_call_start is not None:
            duration = now - self._wc_last_call_start
            if duration > 0:
                self._wc_round_durations.append(duration)
                if len(self._wc_round_durations) > self.WALLCLOCK_WINDOW:
                    self._wc_round_durations.pop(0)
                # EWMA: 70% βάρος στο ιστορικό, 30% στο νέο δείγμα — αρκετά
                # ευαίσθητο σε πραγματική επιβράδυνση (π.χ. GP window γέμισε),
                # αλλά όχι ευάλωτο σε ένα μεμονωμένο outlier γύρο.
                self._wc_avg_round_cost = (
                    0.7 * self._wc_avg_round_cost + 0.3 * duration
                )
        self._wc_last_call_start = now

    def _wall_clock_guard(self) -> bool:
        """NEW 7: Ελέγχει αν πλησιάζουμε το 3λεπτο wall-clock όριο γύρου
        ΤΟΣΟ ΩΣΤΕ να μην προλαβαίνουμε με ασφάλεια άλλον πλήρη κύκλο
        compute+response. Επιστρέφει True αν πρέπει να μπούμε σε
        forced-accept mode (latch — μόλις ενεργοποιηθεί, παραμένει).

        Διαβάζει self.nmi.remaining_time αν υπάρχει (δεν υπάρχει πάντα —
        π.χ. σε αμιγώς step-limited αγώνες χωρίς χρονικό όριο, ή αν το nmi
        δεν εκθέτει αυτό το property σε κάποια έκδοση negmas). Αν δεν είναι
        διαθέσιμο ή είναι None, ο guard είναι no-op (δεν υπάρχει wall-clock
        όριο να προστατευτούμε από).
        """
        if self._wc_forced_accept_mode:
            return True   # latched

        if len(self._wc_round_durations) < self.WALLCLOCK_MIN_SAMPLES:
            return False  # ανεπαρκή δεδομένα ακόμα — μην ενεργοποιήσεις πρόωρα

        remaining_time = None
        nmi = getattr(self, "nmi", None)
        if nmi is not None:
            remaining_time = getattr(nmi, "remaining_time", None)
            if callable(remaining_time):
                try:
                    remaining_time = remaining_time()
                except Exception:
                    remaining_time = None

        if remaining_time is None:
            return False  # κανένα ενεργό χρονικό όριο — τίποτα να φυλάξουμε

        safety_threshold = self.WALLCLOCK_SAFETY_MARGIN * self._wc_avg_round_cost
        if remaining_time <= safety_threshold:
            self._wc_forced_accept_mode = True
            return True
        return False

    def _init_poison(self):
        if not self._issue_variety_rank or not self.rational_outcomes_with_util:
            self.poison_idx   = 0
            self.poison_value = None
            return

        slot      = self._poison_issue_slot % len(self._issue_variety_rank)
        issue_idx = self._issue_variety_rank[slot][0]

        top_outcomes = [o for _, o in self.rational_outcomes_with_util[:100]]
        vals     = list(dict.fromkeys(o[issue_idx] for o in top_outcomes))
        val_slot = self._poison_value_slot % max(1, len(vals))

        self.poison_idx   = issue_idx
        self.poison_value = vals[val_slot]

    # ── Main call ──────────────────────────────────────────────────────────

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        # NEW 7: record wall-clock timing for THIS round as the very first
        # action, before any other processing, so the measurement reflects
        # the true inter-round gap (including our own prior-round compute).
        self._wall_clock_record_round_start()

        offer = state.current_offer

        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        self._poison_rotation_step += 1
        self._maybe_rotate_poison(state)

        if offer is None:
            return SAOResponse(ResponseType.REJECT_OFFER, self._bid(state))

        self.update_opponent_model(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        return SAOResponse(ResponseType.REJECT_OFFER, self._bid(state))

    def _maybe_rotate_poison(self, state: SAOState) -> None:
        # FIX 5: phase-shifted clocks — (step + phase) % period
        step = self._poison_rotation_step
        if (step + self._poison_value_phase) % self.POISON_VALUE_ROTATION_INT == 0:
            self._poison_value_slot += 1
            self._init_poison()
        if (step + self._poison_issue_phase) % self.POISON_ROTATION_INTERVAL == 0:
            self._poison_issue_slot += 1
            self._poison_value_slot  = 0
            self._init_poison()

    # ── Target utility curve ───────────────────────────────────────────────

    def _get_target_utility(self, relative_time: float) -> float:
        max_u = self.rational_outcomes_with_util[0][0]
        r     = self.ufun.reserved_value

        # NEW 4: use the adaptive (archetype-based) panic start instead of
        # the fixed PANIC_START constant. Defaults to PANIC_START until the
        # opponent archetype locks in (after ARCHETYPE_WINDOW offers).
        panic_start = self._effective_panic_start

        if relative_time < panic_start:
            return max_u - 0.02

        base_floor_u = self.EXTRACTION_FLOOR * max_u
        floor_u = base_floor_u

        if self._floor_detected:
            opp_history = self.private_info.get("opponent_history", [])
            if opp_history:
                last_offer_util = self.ufun(opp_history[-1])
                if last_offer_util > floor_u:
                    floor_u = last_offer_util

        # NEW 3: dynamic ultimatum freeze. Past the session-unique jittered
        # trigger time, Ianos stops conceding entirely and holds at the
        # current floor — but because _session_ultimatum_t is desynchronised
        # per-session (±ULTIMATUM_JITTER around 0.95), two structurally
        # identical agents (e.g. mirror matches) will NOT freeze at the
        # exact same t, avoiding simultaneous zero-concession deadlock.
        #
        # NEW 5: GP-forecast plateau can trigger the ultimatum EARLIER than
        # the session-jittered base time. If the GP predicts the opponent's
        # curve has flattened (no further concession expected), waiting
        # until _session_ultimatum_t wastes rounds chasing a concession
        # that will not come — freeze now and let the small epsilon
        # incentive do its job instead.
        #
        # BUGFIX (post-Ianos22 regression): this check now runs BEFORE the
        # FIX-1 stubborn early-return below. Previously, _stubborn_detected
        # short-circuited the method at max_u-0.02 unconditionally, which
        # meant the ultimatum mechanism could NEVER fire against an opponent
        # classified stubborn — and Boulware opponents are classified SLOW
        # (NEW 4) early in the negotiation (often by t≈0.06, well before any
        # real concession curve has played out), which lowers the stubborn
        # rigidity threshold and reliably latches _stubborn_detected=True.
        # Net effect: against any opponent at least as patient as Linear
        # concession (exp >= 1.5), Ianos held at max_u-0.02 for the entire
        # negotiation and NEVER reached a deal — adv(A) = 0 regardless of
        # how good the opponent's final offer was. Benchmarked: 0/9 deals
        # across 7 domain shapes and 9 seeds before this fix; 100% Linear/
        # Conceder deal rate was unaffected because those archetypes don't
        # trigger the stubborn latch. The ultimatum mechanism is precisely
        # the tool meant to resolve a stubborn standoff — it must not be the
        # thing stubborn detection blocks.
        #
        # SECOND BUGFIX: the ultimatum target deliberately uses base_floor_u
        # (EXTRACTION_FLOOR * max_u), NOT the floor_u possibly upgraded above
        # by last_offer_util. Reason: last_offer_util tracks the single best
        # historical offer from the opponent, which can be a transient spike
        # (one generous bid amid an otherwise noisy/jittery curve) rather
        # than the opponent's stable position. Using it as the ultimatum
        # floor meant a single lucky high offer would permanently raise our
        # demand to that level — so the very next, more typical offer would
        # be rejected, and Ianos would chase a one-off outlier instead of
        # closing the deal. The whole point of the ultimatum epsilon-above-
        # floor design is to accept a small, reliable surplus once timing
        # says "stop negotiating" — anchoring it to an unreliable historical
        # peak defeats that purpose.
        effective_ultimatum_t = self._session_ultimatum_t
        if self._gp_plateau_detected:
            effective_ultimatum_t = min(
                self._session_ultimatum_t, max(panic_start, relative_time)
            )

        if relative_time >= effective_ultimatum_t:
            # Leave the opponent a small, session-jittered incentive margin
            # above the BASE extraction floor, rather than demanding max_u
            # outright — encourages a last-second accept without anchoring
            # to a possibly-noisy historical peak.
            ultimatum_floor = base_floor_u + self._session_epsilon_incentive * max_u
            return max(max(r + 1e-6, base_floor_u), min(max_u, ultimatum_floor))

        # FIX 1: when stubborn detected (and we have NOT yet reached the
        # ultimatum trigger above), keep demanding near-max. Returning
        # EXTRACTION_FLOOR here meant PATH A accepted any offer >= 0.78,
        # defeating the entire stubborn guard. We hold the wall — but only
        # until the ultimatum takes over, never indefinitely.
        if self._stubborn_detected:
            return max_u - 0.02

        panic_progress = (relative_time - panic_start) / (1.0 - panic_start)
        panic_progress = max(0.0, min(1.0, panic_progress))
        target = max_u - (max_u - floor_u) * (panic_progress ** self.PANIC_EXP)
        return max(max(r + 1e-6, floor_u), target)

    # ── Floor and stubborn detection ───────────────────────────────────────

    def _opponent_concession_velocity(self) -> float:
        hist = self._opponent_util_history
        if len(hist) < self.VELOCITY_WINDOW + 1:
            return 1.0
        window = hist[-self.VELOCITY_WINDOW:]
        deltas = [window[i + 1] - window[i] for i in range(len(window) - 1)]
        return max(0.0, sum(deltas) / len(deltas))

    def _opponent_at_floor(self) -> bool:
        n_seen = len(self.private_info.get("opponent_history", []))
        if n_seen < self.MIN_MODEL_HISTORY:
            return False
        if self._opponent_concession_velocity() > self.VELOCITY_FLOOR:
            return False
        if self._opp_util_peak < 1e-6:
            return False
        # FIX 3: require genuine raw-utility concessions, not just velocity freeze
        if self._raw_concession_steps < self.MIN_CONCESSION_STEPS:
            return False
        latest = self._opponent_util_history[-1] if self._opponent_util_history else 0.0
        return latest <= self.OPP_FLOOR_MARGIN * self._opp_util_peak

    def _opponent_is_stubborn(self) -> bool:
        """
        FIX 2 (Guarded): Returns True when the opponent has never offered below
        STUBBORN_RIGIDITY_THRESHOLD across their STABILIZED data history.
        By ignoring the highly volatile warm-up window, early exploratory/dummy
        offers cannot poison this check.

        NEW 4: uses _effective_stubborn_threshold instead of the fixed
        constant — lowered for opponents already classified SLOW (catch
        stubbornness sooner), raised for FAST concedrrs (avoid mislabeling
        a still-conceding fast opponent as stubborn due to a brief plateau).
        """
        n_seen = len(self.private_info.get("opponent_history", []))
        if n_seen < self.MIN_MODEL_HISTORY:
            return False
        if not self._opponent_util_history:
            return False
        
        # Squeeze Protection: Slice history to only evaluate steps AFTER model stabilization
        stable_history = self._opponent_util_history[self.MIN_MODEL_HISTORY - 1:]
        if not stable_history:
            return False
            
        return min(stable_history) >= self._effective_stubborn_threshold

    # ── Acceptance strategy ────────────────────────────────────────────────

    def acceptance_strategy(self, state: SAOState) -> bool:
        offer = state.current_offer
        if offer is None:
            return False

        offer_utility  = self.ufun(offer)
        r              = self.ufun.reserved_value

        # PATH 0 (NEW 7): wall-clock forced accept. If we are within
        # WALLCLOCK_SAFETY_MARGIN round-cost multiples of the 3-minute
        # limit running out, accept ANY rational offer immediately rather
        # than computing target_utility/dither/etc — a self-inflicted
        # forfeit from running out the clock mid-computation is strictly
        # worse than accepting a mediocre-but-rational offer. This check
        # is intentionally placed before every other path and before any
        # further computation in this method.
        if self._wall_clock_guard():
            return offer_utility > r

        t              = state.relative_time
        max_u          = self.rational_outcomes_with_util[0][0]
        target_utility = self._get_target_utility(t)
        extraction_lb  = self.EXTRACTION_FLOOR * max_u

        # CONCEAL 4: Dither the effective acceptance threshold each round.
        # A uniform ±ACCEPT_DITHER offset is applied so an observer cannot
        # back-calculate the exact threshold from the timing of accepts/rejects.
        # Bounded below by reserved_value + epsilon so we never accept a bad deal.
        dither = random.uniform(-self.ACCEPT_DITHER, self.ACCEPT_DITHER)
        effective_target = max(r + 1e-4, target_utility + dither)

        # PATH A: offer meets our (dithered) current target
        if offer_utility >= effective_target:
            return True

        # PATH A* (FIX 4): generous offer, but only after midpoint
        if (t >= self.SIMPLE_THRESHOLD_MIN_TIME
                and offer_utility >= self.SIMPLE_THRESHOLD * max_u):
            return True

        # PATH B: opponent at floor (Fix 3 guards spurious floor detection)
        if (not self._stubborn_detected
                and t >= self.FLOOR_ACCEPT_TIME
                and offer_utility >= extraction_lb
                and offer_utility > r
                and self._floor_detected):
            return True

        # PATH C: deadline last-resort (disabled for stubborn)
        if t > 0.99 and not self._stubborn_detected:
            if (offer_utility > r
                    and offer_utility >= self.DEADLINE_FLOOR * max_u):
                return True

        return False

    # ── Ufun Concealment helpers ───────────────────────────────────────────

    def _jittered_target(self, base_target: float) -> float:
        """CONCEAL 2: Add Gaussian noise to the target, clipped to ±JITTER_CLIP.
        Zero-mean so the long-run average target is unchanged."""
        noise = random.gauss(0.0, self.JITTER_SIGMA)
        noise = max(-self.JITTER_CLIP, min(self.JITTER_CLIP, noise))
        max_u = self.rational_outcomes_with_util[0][0]
        r     = self.ufun.reserved_value
        # Keep jittered target above reserved_value + epsilon
        return max(r + 1e-4, min(max_u, base_target + noise))

    def _obfuscated_choice(self, viable_outcomes: list) -> "Outcome":
        """CONCEAL 3 ENHANCED: Systematic Rank Inversion for τ(B) suppression.

        Στόχος: τα bids του Ianos να δείχνουν στον αντίπαλο εντελώς αντεστραμμένα
        issue weights από τα αληθινά — d(ûA_opp, uA_true) → −1 → τ(B) → 0.

        Δύο μηχανισμοί ταυτόχρονα:
          (α) FREEZE top issues: για τα INVERSION_DEPTH πιο σημαντικά issues,
              επιλέγουμε ΠΑΝΤΑ τιμές διαφορετικές από την αληθινή modal
              → εμφανίζονται με χαμηλή rigidity → αντίπαλος τα νομίζει ασήμαντα.
          (β) VARY bottom issues: για τα λιγότερο σημαντικά issues,
              επιλέγουμε ΠΑΝΤΑτην αληθινή modal → εμφανίζονται με υψηλή rigidity
              → αντίπαλος τα νομίζει σημαντικά.
        """
        if len(viable_outcomes) <= 1:
            return viable_outcomes[0] if viable_outcomes else None

        n_issues = len(viable_outcomes[0])
        depth    = min(self.INVERSION_DEPTH, n_issues)
        top_issues = set(self._issue_true_rank[:depth])
        bot_issues = set(self._issue_true_rank[depth:])
        modal      = self._issue_true_modal

        def inversion_score(outcome) -> float:
            score = 0.0
            for i, val in enumerate(outcome):
                is_modal = (val == modal.get(i))
                if i in top_issues:
                    # Θέλουμε ΜΗ-modal στα σημαντικά: reward για αντίθεση
                    if not is_modal:
                        score += 2.0
                elif i in bot_issues:
                    # Θέλουμε modal στα ασήμαντα: reward για ομοιότητα
                    if is_modal:
                        score += 1.0
            return score

        weights = [
            max(0.1, 1.0 + (self.INVERSION_WEIGHT - 1.0) * inversion_score(o))
            for o in viable_outcomes
        ]
        total = sum(weights)
        r_val = random.uniform(0, total)
        cumul = 0.0
        for outcome, w in zip(viable_outcomes, weights):
            cumul += w
            if r_val <= cumul:
                return outcome
        return viable_outcomes[-1]

    # ── Bidding strategy ───────────────────────────────────────────────────

    def _bid(self, state: SAOState) -> "Outcome | None":
        # NEW 7: in forced-accept mode (wall-clock guard latched), skip the
        # full Decoy/Jitter/Obfuscation pipeline entirely. This path only
        # runs if we are FORCED to make a fresh offer (e.g. opponent's
        # current_offer is None on this round) while time is critically
        # short — return our best rational outcome immediately rather than
        # spending compute time we may not have on concealment machinery.
        if self._wc_forced_accept_mode:
            if self.rational_outcomes_with_util:
                return self.rational_outcomes_with_util[0][1]

        t          = state.relative_time
        max_u      = self.rational_outcomes_with_util[0][0]
        r          = self.ufun.reserved_value
        true_target = self._get_target_utility(t)

        # CONCEAL 1: Decoy bid — emit a below-target offer in the early game
        if (t < self.DECOY_MAX_TIME
                and random.random() < self.DECOY_PROB):
            decoy_target = true_target - self.DECOY_DROP
            decoy_target = max(r + 1e-4, decoy_target)
            decoy_outcomes = [
                outcome for util, outcome in self.rational_outcomes_with_util
                if abs(util - decoy_target) <= 0.05
            ]
            if decoy_outcomes:
                chosen = random.choice(decoy_outcomes)
                self.private_info.setdefault("_own_bid_history", []).append(chosen)
                return chosen

        # CONCEAL 2: Apply jitter to the true target
        target_utility = self._jittered_target(true_target)
        tolerance      = 0.05

        viable_outcomes = [
            outcome for util, outcome in self.rational_outcomes_with_util
            if target_utility - tolerance <= util <= target_utility + tolerance
        ]

        if not viable_outcomes:
            viable_outcomes = [self.rational_outcomes_with_util[0][1]]

        # Poison Pill (existing): filter to poisoned outcomes if available
        if self.poison_value is not None:
            poisoned_outcomes = [
                o for o in viable_outcomes
                if o[self.poison_idx] == self.poison_value
            ]
            if poisoned_outcomes:
                viable_outcomes = poisoned_outcomes

        # CONCEAL 3: Among survivors, pick with obfuscation bias
        chosen = self._obfuscated_choice(viable_outcomes)
        self.private_info.setdefault("_own_bid_history", []).append(chosen)
        return chosen

    # ── Opponent model update ──────────────────────────────────────────────

    def update_opponent_model(self, state: SAOState) -> None:
        offer = state.current_offer
        if offer is None:
            return

        self._current_t = state.relative_time
        self.private_info["opponent_history"].append(offer)
        history  = self.private_info["opponent_history"]
        n_offers = len(history)

        # ── NEW: Recency-weighted state update (RANK 1+3) ─────────────────
        self._update_recency_state(offer, self._current_t)

        # ── NEW: Recency-weighted estimator as opponent_ufun (RANK 2) ─────
        # Κλείνουμε snapshot των τρεχόντων dicts για thread-safety
        val_freq_snap       = {i: dict(v) for i, v in self._val_freq.items()}
        issue_move_snap     = dict(self._issue_move_count)

        max_mv = max(issue_move_snap.values()) if issue_move_snap else 1.0

        def recency_estimator(outcome: Outcome) -> float:
            if not outcome:
                return 0.0
            iw_raw = {
                i: 1.0 / (1.0 + issue_move_snap.get(i, 0.0) / max_mv)
                for i in range(len(outcome))
            }
            total_iw = sum(iw_raw.values()) or 1.0
            score = 0.0
            for i, val in enumerate(outcome):
                iw = iw_raw[i] / total_iw
                vf = val_freq_snap.get(i, {})
                total_vf = sum(vf.values()) or 1.0
                score += iw * (vf.get(val, 0.0) / total_vf)
            return score

        self.private_info["opponent_ufun"] = LambdaMultiFun(f=recency_estimator)
        est_opp_util = recency_estimator(offer)
        self._opponent_util_history.append(est_opp_util)

        # ── NEW 5: Feed GP with this round's (t, estimated_util) pair ─────
        self._gp_update_observation(self._current_t, est_opp_util)

        # ── NEW 5: Run the GP solve EXACTLY ONCE per round (here), caching
        # the result in self._gp_last_forecast_mean/std. Both the blended
        # estimator (NEW 6, below) and the plateau check read this cached
        # value instead of re-solving the O(n³) linear system per candidate
        # outcome — without this caching, _bid()'s evaluation of dozens of
        # candidate outcomes would each trigger a fresh GP solve.
        if len(self._gp_observations_t) >= self.GP_MIN_OBSERVATIONS:
            self._gp_predict_future_concession(self._current_t)
            self._gp_check_plateau()

        # ── NEW 6: Blend GP-smoothed score into opponent_ufun per portfolio ─
        # In DENSE domains (selected at init), give more weight to the GP's
        # smoothed consensus since per-(issue,value) frequency counts are
        # individually sparse. In SPARSE domains, barely blend it in — the
        # frequency counts are already reliable with few distinct values.
        gp_weight = self._gp_blend_weight
        if gp_weight > 0.0 and len(self._gp_observations_t) >= self.GP_MIN_OBSERVATIONS:
            def blended_estimator(outcome: Outcome, _recency=recency_estimator,
                                   _w=gp_weight) -> float:
                base = _recency(outcome)
                gp_score = self._gp_smoothed_value_score(outcome)
                return (1.0 - _w) * base + _w * gp_score
            self.private_info["opponent_ufun"] = LambdaMultiFun(f=blended_estimator)

        # FIX 3: raw concession tracking (unchanged)
        raw_util = self.ufun(offer)
        if self._last_raw_offer_util is not None:
            if raw_util < self._last_raw_offer_util - 1e-6:
                self._raw_concession_steps += 1
            elif raw_util > self._last_raw_offer_util + 1e-6:
                self._raw_concession_steps = 0
        self._last_raw_offer_util = raw_util

        # FIX 2: cumulative peak (unchanged)
        if n_offers >= self.MIN_MODEL_HISTORY:
            if est_opp_util > self._opp_util_peak:
                self._opp_util_peak = est_opp_util
            self._opp_util_min = min(self._opp_util_min, est_opp_util)
        else:
            self._opp_util_peak = max(self._opp_util_peak, est_opp_util)
            self._opp_util_min  = min(self._opp_util_min,  est_opp_util)

        if not self._floor_detected and self._opponent_at_floor():
            self._floor_detected = True

        # NEW 4: Classify opponent archetype (locks once, after ARCHETYPE_WINDOW)
        self._classify_opponent_archetype()

        if not self._stubborn_detected and self._opponent_is_stubborn():
            self._stubborn_detected = True