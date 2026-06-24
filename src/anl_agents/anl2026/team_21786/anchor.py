"""ANL 2026 negotiation agent.

A single-class BOA agent. The three negotiation responsibilities are kept as
separate, self-contained methods so each maps one-to-one onto a section of the
report:

    update_opponent_model      -> report: "Opponent Model"
    concealing_bidding_strategy -> report: "Concealing Bidding Strategy"
    acceptance_strategy        -> report: "Acceptance Strategy"

Design summary ("model hard, deceive lightly"):
  * Scoring is `Advantage + Concealing`. Advantage (how far the deal beats our
    reservation value) dominates; Concealing is a small shared term split by how
    well each side models the other. So we maximise Advantage first and treat
    concealment as a near-free add-on.
  * Bidding decouples *how much* to concede (a time-based Boulware utility
    target) from *which* bid to make (among bids in our utility band, offer the
    one our opponent model likes most -- palatable at zero cost to us).
  * The opponent model is a simple, interpretable frequency + stability additive
    estimate. It is emitted every round into `private_info["opponent_ufun"]`,
    which the competition scorer reads -- emitting nothing forfeits the whole
    Concealing point.

Anti-exploitation (the key to not being walked over by a firm opponent):
  A naive Boulware that concedes all the way to its reservation value lets a firm
  opponent simply wait and collect its own ideal outcome. We therefore floor our
  concession at a *fair share* of our utility range, and even the end-game rescue
  only releases that floor partway (RESCUE_FLOOR_FRACTION). We never hand the deal
  away; instead a firm opponent must yield to our offers or accept a no-deal.

  End-game close (SECURE_BEST + SECURE_BEST_FAIR=False): a no-deal scores only the
  ~0.5 Concealing term, while ANY positive-advantage deal beats it. So once inside
  the (deadline-adaptive) end-game window we bank the best offer the opponent has
  shown us as long as it clears our reservation value -- we do NOT require it to be
  "fair" by our own model. MEASURED: this is the single biggest fix relative to the
  agent that placed mid-pack in the live tournament (whose median negotiation was a
  no-deal): against hold-and-wait opponents it roughly doubles our deal rate
  (e.g. vs a pure hardliner 0.03 -> 0.61) at the cost of only a small concealing
  swing. The reservation hard floor -- not a fairness test -- is what still
  guarantees we never accept worse than walking away.

  Deadline-adaptive end-game: the rescue and secure-accept windows trigger at the
  EARLIER of a relative-time threshold or a fixed number of absolute rounds before
  the deadline (MIN_RESCUE_ROUNDS / MIN_SECURE_ROUNDS). At short deadlines a fixed
  5% window is too few rounds to actually close; this pulls the end-game earlier so
  we still reach agreement (MEASURED: n_steps=20 score 0.98 -> 1.04).

  First-mover close (LAST_OFFER_GRAB): on our final proposal -- when the opponent
  gets no turn to accept a fresh bid -- we table the opponent's own best-shown
  offer, which is provably acceptable to them, rather than time out.

Adaptivity (how we win against opposite opponent styles):
  A firm opponent yields by *accepting* our firm offers, so against it we hold the
  floor and extract. A conceding opponent yields by *improving its offers*, so we
  capture its best offer as soon as it clears CAPTURE_FRACTION of our range rather
  than greedily holding for our maximum and losing it. The same firm floor handles
  both -- no explicit opponent-type switch.

We optimise for the tournament metric -- our ABSOLUTE mean score (Advantage +
Concealing), not pairwise margin: a balanced deal scores ~1.0, a no-deal only the
~0.5 Concealing term, so we close deals even when the opponent also gains.

Measured behaviour (local harness: ~20 NegMAS opponents incl. self-play and firm
GSmith-modelling rivals, over the 7 provided + generated domains, both move
orders). We score positively against every opponent type and close deals at a high
rate (~0.85 vs ~0.63 for the firmer earlier version). We extract hard from
conceders (~1.6), close and bank a positive deal against hold-and-wait firms
(Tough/MiCRO etc., where the earlier version timed out), and contest the
model-capable agents. Opponent-model accuracy (the free half of Concealing):
tau_me ~0.57 with a slight early-offer/opening boost, giving a Concealing split at
parity (~0.50) rather than losing it.

Three deliberate-deception variants were implemented, MEASURED, and left OFF:
offer diversification (DECEPTION) raises the opponent's tau of us (it reveals more
of our high-utility region); a Bayesian issue-weight model (MODEL_KIND) is
tau-equivalent to the simple model; decoy-issue freezing (DECOY_FREEZE) does lower
a weight-learner's tau of us but costs more Advantage than it returns. All confirm
the project's "deceive only when nearly free" thesis -- our concealment comes from
holding firm and modelling them well, not from noise. The flags are kept for the
report's ablation.
"""

from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import TableFun


class AnchorNegotiator(SAOCallNegotiator):
    """Advantage-first Boulware agent with anti-exploitation concession floor."""

    # --- Tunable constants (tuned empirically against a roster of opponents) --
    # Concession exponent of the Boulware target curve. e < 1 => firm early,
    # concede late. Smaller e == firmer (holds near our maximum longer).
    CONCESSION_EXPONENT = 0.10

    # We never let our utility target fall below this fraction of our own utility
    # range (reservation .. maximum) until the end-game rescue. This is the
    # anti-exploitation floor: a firm opponent cannot drag us down to its ideal.
    FAIR_FLOOR_FRACTION = 0.85

    # Fraction of time after which the end-game rescue engages: the floor is
    # released down toward the reservation value so we still close *some*
    # positive-Advantage deal rather than time out into a zero. Kept late so the
    # rescue is hard to exploit.
    RESCUE_TIME = 0.95

    # In the end-game the rescue concedes toward (but not to) the reservation
    # value to CLOSE a deal. The tournament ranks by our absolute score
    # (Advantage + Concealing): a balanced deal scores ~1.0 while a no-deal scores
    # only the ~0.5 Concealing term, so closing almost any positive deal beats
    # timing out -- even against a firm opponent who also gains. We keep a small
    # floor so we never accept a near-worthless deal. MEASURED: lowering this from
    # 0.9 lifted our mean score vs firm opponents (1.07 -> 1.19) at no cost to the
    # extraction we get from conceders (they concede before the rescue engages).
    RESCUE_FLOOR_FRACTION = 0.20

    # We accept the opponent's best-yet offer as soon as it clears this fraction
    # of our range, instead of holding out for our absolute maximum (a conceding
    # opponent's strong offers oscillate and may not return). Lower == grab deals
    # sooner (more robust to firm opponents, less extraction from yielders).
    CAPTURE_FRACTION = 0.85

    # How to pick a bid within our utility band:
    #   "opponent"  -> the one the opponent likes most (maximises agreement prob,
    #                  but most generous to a firm opponent)
    #   "greedy"    -> the one best for us (gives the opponent the least)
    #   "nash"      -> maximise our_utility * opponent_utility (balanced)
    SELECTION_MODE = "nash"

    # End-game (rescue) bid selector -- see _select_bid. "fair" (most balanced)
    # is the measured default; "rvtarget"/"kalai"/"nash" are extraction-oriented
    # variants tried for deal QUALITY against responsive rivals.
    RESCUE_SELECT = "fair"
    # Slack applied to the estimated opponent reservation for "rvtarget": we
    # assume they will accept a little below the worst they have themselves
    # offered (their offers lower-bound, not pin, their true reservation).
    RV_SLACK = 0.9

    # Deception (Phase 2): flatten the frequency signal we leak. Because our firm
    # floor makes us repeatedly offer our few best outcomes -- which is exactly
    # what a frequency-based opponent model needs to pin down our preferences --
    # we instead diversify across our high-utility band, preferring values we have
    # revealed least often. This lowers the opponent's Kendall-tau of us (raising
    # our share of the Concealing point) at ~no Advantage cost, since every bid we
    # pick still clears the fair floor. Flag-gated for the Phase 2 ablation.
    # MEASURED: against a capable opponent model this BACKFIRES -- diversifying
    # across our top band reveals MORE of our high-utility region, raising the
    # opponent's tau of us. Kept for the ablation but disabled by default.
    DECEPTION = False

    # How to time-weight opponent offers when modelling their PREFERENCES:
    #   "uniform" -> every offer counts equally
    #   "recency" -> later offers count more (good for "what they'll accept",
    #                but late offers are concessions, not true preferences)
    #   "early"   -> earlier offers count more (their opening is near their ideal,
    #                so early offers are the honest preference signal)
    MODEL_WEIGHTING = "early"

    # Extra weight given to the opponent's OPENING offer, which is typically their
    # ideal outcome and thus the single strongest preference signal. MEASURED: a
    # moderate boost raises our model accuracy (tau_me 0.55 -> 0.57) and lifts our
    # Concealing split from losing (0.496) to parity (0.500) at zero advantage cost
    # -- the free half of Concealing. Above ~8 the gain saturates.
    OPENING_BOOST = 3.0

    # How to estimate the opponent's ISSUE WEIGHTS:
    #   "stability" -> issues they change less are weighted more (cheap heuristic)
    #   "bayesian"  -> posterior over rank-weight hypotheses, favouring the
    #                  weighting under which their (early) offers look high-utility
    # MEASURED: the two are within noise on tau_me (0.561 vs 0.559) -- Kendall-tau
    # only needs the ranking, which the simple model already recovers, and ~100
    # offers is not the sparse regime where Bayesian helps. We keep "stability"
    # (simpler, interpretable) and retain the Bayesian path for the report.
    MODEL_KIND = "pairwise"   # v5: pairwise beats plain frequency (see below)
    BAYES_TEMP = 6.0  # softmax sharpness over weight hypotheses

    # "optim": COMPUTE-HEAVY opponent model. Instead of cheap frequency counting,
    # fit the opponent's full additive utility (per-issue value scores AND issue
    # weights) by max-entropy inverse-RL: assume their offers are Boltzmann-rational
    # (drawn ~ exp(u_opp(o))), early offers weighted more, and run OPTIM_ITERS of
    # gradient ascent on the offered-vs-all-outcomes contrast each update. This is
    # the "spend seconds to model better" lever -- the one thing that lifts BOTH
    # scoring terms (tau_me / Concealing share, and deal-finding / Advantage).
    # Tested head-to-head vs the frequency model; see report/EXPERIMENTS.md.
    OPTIM_ITERS = 60
    OPTIM_LR = 0.5

    # "pairwise": frequency value scores PENALISED by how often WE offered each
    # value (values we keep proposing and they keep rejecting are likely bad for
    # them -- information the pure frequency model throws away). GAMMA is the
    # penalty weight. Targets tau_me (the parity-pinned Concealing term) with a
    # genuinely additive signal. Flag-gated; tested on the live-matching ANL dist.
    PAIRWISE_GAMMA = 0.3
    # N6: issue-weight estimator for the pairwise model. "stability" (default, the
    # v6 shipped behaviour) or "divergence" (weight by our-vs-their value-distribution
    # total variation). Flag-gated experiment on top of pairwise values.
    PAIRWISE_WEIGHT = "stability"
    # N9: earliness-weight the our-offer penalty (early near-ideal rejected offers
    # weigh more). Flag-gated experiment; expectation marginal (model side maxed).
    PAIRWISE_TIMEW = False

    # DECOUPLE_BID: the scorer reads our EMITTED estimate (tau_me) but what we
    # REVEAL drives the opponent's tau of us (tau_opp) -- these are INDEPENDENT.
    # So emit the accurate pairwise model (raises tau_me) while BIDDING on the
    # plain frequency model (so we reveal no more than v4 does, keeping tau_opp at
    # baseline). Net: the Concealing SHARE rises without any Advantage cost. Only
    # active with MODEL_KIND="pairwise". Flag-gated; tested on the ANL dist.
    DECOUPLE_BID = True   # v5: emit pairwise (high tau_me), bid plain (tau_opp flat)
    DECOUPLE_PHASED = False  # experiment: bid plain early, sharp model in end-game

    # Decoy-freeze concealment: hold a FIXED value on our least-important issue
    # across all offers. Opponent models that learn issue weights from which
    # issues we leave unchanged (HardHeaded/AgentX-style) then over-rate this
    # cheap issue and mis-rank our true weights -- lowering their tau of us
    # (raising our Concealing share) at ~zero Advantage cost, since we hold our
    # own preferred value on an issue we barely care about. Does nothing against
    # uniform-weight frequency models (e.g. GSmith). Flag-gated for the ablation.
    DECOY_FREEZE = False

    # End-game safety: rather than risk a no-deal (which can LOSE on the
    # Concealing term if the opponent models us better than we model them), in the
    # rescue window we secure the best offer the opponent has actually shown us.
    # Against a yielding opponent this rarely triggers (they accept our firm
    # offers first); against a firm one it guarantees we bank their best concession
    # instead of timing out.
    SECURE_BEST = True

    # Whether the end-game SECURE_BEST capture requires the offer to be "fair" by
    # our own model (u_to_us >= our_estimate_of_their_utility). MEASURED: the live
    # field is dominated by firm rivals whose best-shown offer is their near-ideal
    # -- fair by reality but "lopsided" by our model -- so the fairness test blocks
    # the very deals we need to close and we time out into a ~0.5 no-deal. The
    # tournament scores absolute mean, so ANY positive-Advantage deal beats a
    # no-deal; with this False we bank any rescue-window offer above our
    # reservation. The reservation hard floor (not this test) is what still
    # prevents us being walked below the value of walking away.
    SECURE_BEST_FAIR = False

    # When the SECURE_BEST_FAIR blanket-accept (above) activates, as a fraction of
    # time. Kept slightly LATER than RESCUE_TIME so that "accept-only-high" rivals
    # (e.g. AcceptTop-style) still get a window to accept OUR offer first -- we
    # only cave to their lopsided best at the very end. MEASURED below.
    SECURE_BEST_TIME = 0.97

    # Deadline-adaptive end-game (robustness across round counts). The rescue and
    # secure-accept windows are defined by RELATIVE time, but at short deadlines a
    # 5% window is only ~1 round -- too few to actually exchange and close, which
    # MEASURABLY hurt us at n_steps=20. So we also require a MINIMUM number of
    # absolute rounds for each window: the effective trigger time is the EARLIER of
    # the relative threshold and "this many rounds before the end". At long
    # deadlines the relative thresholds dominate (unchanged); at short ones the
    # round floors pull the end-game earlier so we still close.
    MIN_RESCUE_ROUNDS = 4
    MIN_SECURE_ROUNDS = 2

    # Concession-conditional secure-accept timing. A conceding opponent will keep
    # improving its offers, so we can afford to HOLD (delay the blanket-accept) and
    # extract more; a firm hold-and-wait opponent never improves, so we should cave
    # at the normal time to still close. We push SECURE_BEST_TIME later in
    # proportion to the opponent's revealed concession (the round floor still
    # guarantees we accept before timing out). Off by default; measured.
    ADAPTIVE_SECURE = False
    ADAPT_SECURE_K = 0.025

    # Minimum advantage (as a fraction of our range) required for the end-game
    # blanket-accept to fire. A near-zero-advantage deal barely beats a no-deal on
    # the Advantage term, but closing early means we observe FEWER opponent offers
    # -> a weaker model -> lower Concealing; holding for more offers can raise
    # tau_A. A small floor refuses worthless deals to protect Concealing on hard
    # (competitive) domains. 0.0 = accept any positive deal (the deal-rate-max
    # default). Measured.
    SECURE_ACCEPT_FLOOR = 0.05

    # AC_combi-style end-game acceptance. Our secure path historically required the
    # opponent's CURRENT offer to be its best-ever (MAX_T), which the literature
    # finds "too strict -- loses deals": an opponent that oscillates and offers a
    # good-but-not-best deal late is rejected, risking a no-deal (a bottom-quartile
    # leak). With SECURE_ANY=True we instead accept ANY end-game offer that clears
    # our acceptance floor (reservation + SECURE_ACCEPT_FLOOR), regardless of
    # whether it is their all-time best -- closer to the measured-best AC_combi
    # rule. The reservation+floor still guards against underselling. Measured.
    SECURE_ANY = False

    # "Never undersell": never propose a bid worth less to us than the best the
    # opponent has already offered (anti-self-sabotage). Measured.
    NEVER_UNDERSELL = False

    # Calibrated P(accept) bid search: among band candidates, offer the one
    # maximising our_utility * P(accept), where P(accept) rises monotonically with
    # the bid's estimated utility to the opponent between their estimated
    # reservation and their ideal. A SOFT version of reservation-targeting (which,
    # as a hard filter, dropped deal rate); the probabilistic weighting trades
    # extraction against acceptance smoothly. Measured.
    PACCEPT_SEARCH = False

    # LOOKAHEAD (the "spend more compute like the slow leaders do" experiment).
    # Instead of picking the Nash bid greedily, score each candidate bid by its
    # EXPECTED final utility: EV(o) = P_accept(o,t)*u(o) + (1-P_accept)*continuation,
    # where continuation = what we can still secure later (max of our reservation
    # and the best the opponent has shown us). P_accept is a time-decaying
    # behavioural model of the opponent (they demand near-ideal early, concede
    # toward their estimated reservation late). LOOKAHEAD_MC>0 averages EV over that
    # many Monte-Carlo samples of the opponent's (uncertain) reservation -- the
    # genuine compute-heavy path. Flag-gated; see report/EXPERIMENTS.md for whether
    # it beats greedy Nash (prior: no -- the 1-step PACCEPT version already lost,
    # because acting harder on a NOISY model amplifies its errors).
    LOOKAHEAD = False
    LOOKAHEAD_SOFT = 0.20
    LOOKAHEAD_MC = 0

    # --- Offline-trained concession policy (Phase 3) ---------------------------
    # A state-conditioned policy whose weights are optimised OFFLINE
    # (eval/train_policy.py) against the opponent roster and FROZEN here. At
    # runtime it maps negotiation-state features to a concession target,
    # replacing the fixed Boulware curve in the normal phase. The end-game rescue
    # and secure-accept safety net still apply on top, so a bad policy can never
    # time out into a no-deal -- it can only affect how much we concede. Inference
    # is pure-python (a 7-dim dot product + sigmoid); no runtime memory, no deps.
    # Off unless a trained weight vector is shipped.
    USE_LEARNED_POLICY = False
    # Residual MLP policy: target = Boulware_fraction(t) + correction(state), where
    # correction is a small tanh-MLP (9 features -> 6 tanh hidden -> 1), bounded by
    # POLICY_CORR_SCALE. With all-zero weights the correction is 0 and the policy IS
    # the heuristic Boulware curve -- so training starts exactly at the proven bar
    # and can only learn to deviate where it helps (non-regressing by construction).
    POLICY_NFEAT = 9
    POLICY_NHID = 6
    POLICY_CORR_SCALE = 0.25
    POLICY_WEIGHTS = [0.0] * (6 * 9 + 6 + 6 + 1)  # W1(6x9)+b1(6)+W2(6)+b2(1) = 67

    # First-mover end-game: if we are the LAST to propose (the opponent will not
    # get a turn to accept a fresh bid), table the opponent's own best-shown offer
    # instead -- it is provably acceptable to them, guaranteeing a close where a
    # fresh rescue bid would time out. Off by default; measured.
    LAST_OFFER_GRAB = True
    LAST_OFFER_ROUNDS = 1

    # Behaviour-adaptive concession: speed up our clock in proportion to how much
    # the opponent has actually conceded (reciprocity), to break semi-firm
    # deadlocks. A firm opponent shows ~0 concession so we stay firm. Off by
    # default; BETA scales the effect. Measured.
    ADAPTIVE_CONCESSION = False
    ADAPT_BETA = 0.5

    # Scenario conditioning (the ANL2024-winner "Shochan" idea): adapt our firmness
    # to the DOMAIN STRUCTURE rather than to opponent behaviour. competitiveness =
    # correlation between our utility and our estimate of theirs over the outcome
    # space. On a COOPERATIVE domain (corr>0, our high-utility outcomes are also
    # good for them) we can hold a HIGHER floor and extract more -- they will still
    # accept. On a COMPETITIVE/zero-sum domain (corr<0) we keep the standard floor.
    # COND_K scales the floor shift; sign of COND_K flips the direction for the
    # ablation. Flag-gated, default off. Tested on the live-matching ANL dist.
    SCENARIO_CONDITION = False
    COND_K = 0.15

    # BID_STRATEGY: the BIDDING ARCHITECTURE (not a tuning knob -- a different
    # algorithm for "which outcome to propose").
    #   "boulware" (default): time-based floored Boulware concession (v4).
    #   "micro"   : MiCRO (Minimal Concession + reciprocity). Walk our outcomes
    #               best-first; concede the next one ONLY when we have not offered
    #               more distinct outcomes than the opponent has (so we never out-
    #               concede them); otherwise repeat. Time-agnostic, behaviour-driven.
    #               A proven strong+fast ANAC architecture; tested here head-to-head
    #               vs Boulware. The end-game safety nets still apply so short
    #               deadlines stay safe. See report/EXPERIMENTS.md for the verdict.
    BID_STRATEGY = "boulware"

    # ------------------------------------------------------------------ init --
    def on_preferences_changed(self, changes):
        """Initialise per-negotiation state.

        Called once our utility function is known. All state here is
        instance-local and rebuilt every negotiation -- we keep NO memory across
        negotiations (a competition rule).
        """
        if self.ufun is None:
            return

        os_ = self.ufun.outcome_space
        self._issues = list(os_.issues)
        self._n_issues = len(self._issues)
        # Candidate values per issue, in issue order (outcomes are tuples in this
        # same order).
        self._issue_values = [list(issue.all) for issue in self._issues]

        self._reserve = float(self.ufun.reserved_value)
        self._u_max = float(self.ufun.max())
        # The anti-exploitation floor in absolute utility terms.
        self._fair_floor = self._reserve + self.FAIR_FLOOR_FRACTION * (
            self._u_max - self._reserve
        )

        # All outcomes sorted by OUR utility, descending. We keep only rational
        # outcomes (utility strictly above the reservation value) -- we would
        # never propose or accept anything below the value of walking away.
        outcomes = list(os_.enumerate_or_sample(max_cardinality=100_000))
        scored = [(float(self.ufun(o)), o) for o in outcomes]
        scored = [su for su in scored if su[0] > self._reserve]
        scored.sort(key=lambda su: su[0], reverse=True)
        if not scored:  # degenerate: nothing beats reservation -> keep the best
            scored = sorted(
                ((float(self.ufun(o)), o) for o in outcomes),
                key=lambda su: su[0],
                reverse=True,
            )[:1]
        self._sorted_utils = [su[0] for su in scored]
        self._sorted_outcomes = [su[1] for su in scored]

        # --- MiCRO bidding state (used when BID_STRATEGY == "micro") ---
        self._micro_i = 0                  # highest sorted-outcome index offered
        self._my_unique: set = set()       # distinct outcomes WE have proposed
        self._opp_unique: set = set()      # distinct outcomes the opponent proposed

        # --- Decoy-freeze: pick our least-important issue (smallest utility swing)
        # and hold our preferred value on it across all offers (see DECOY_FREEZE).
        self._decoy_issue = None
        self._decoy_value = None
        if self.DECOY_FREEZE and self._n_issues >= 2:
            best = self._sorted_outcomes[0]
            sens = []
            for i in range(self._n_issues):
                us = []
                for v in self._issue_values[i]:
                    o = list(best)
                    o[i] = v
                    us.append(float(self.ufun(tuple(o))))
                sens.append(max(us) - min(us))
            self._decoy_issue = min(range(self._n_issues), key=lambda i: sens[i])
            self._decoy_value = best[self._decoy_issue]

        # --- Opponent-model accumulators (see update_opponent_model) ---
        self._value_counts = [dict() for _ in range(self._n_issues)]
        self._last_opp_value = [None] * self._n_issues
        self._issue_changes = [0] * self._n_issues
        self._opp_offers_seen = 0
        self._optim_idx = None  # cached all-outcome matrix for MODEL_KIND="optim"
        self._comp_cache = None  # cached competitiveness (SCENARIO_CONDITION)
        self._comp_step = -1
        self._bid_model = None  # decoupled bidding model (set each rebuild)

        # --- Deception accumulators: how often WE have revealed each value, so we
        # can prefer under-revealed values and keep our offer distribution flat.
        self._my_value_counts = [dict() for _ in range(self._n_issues)]
        self._my_value_weighted = [dict() for _ in range(self._n_issues)]  # earliness-wtd
        self._my_offer_count = 0

        # Opponent offer history (offer, early-weight) for the Bayesian model.
        self._opp_offer_hist: list[tuple] = []
        # Pre-enumerate rank-weight hypotheses for the Bayesian model (only when it
        # is enabled and the issue count keeps the permutation set small).
        self._weight_hypotheses = None
        if self.MODEL_KIND == "bayesian" and 1 <= self._n_issues <= 6:
            import itertools

            self._weight_hypotheses = []
            for perm in itertools.permutations(range(self._n_issues)):
                # perm[i] = rank of issue i (0 = most important).
                raw = [float(self._n_issues - perm[i]) for i in range(self._n_issues)]
                s = sum(raw)
                self._weight_hypotheses.append([r / s for r in raw])

        # Best offer (to us) the opponent has made -- a provably attainable
        # utility/outcome we can fall back on in the end-game.
        self._best_opp_util = self._reserve
        self._best_opp_offer = None
        self._first_opp_util = self._reserve
        self._opp_concession = 0.0

        # Small memo so a single round computes its planned bid once.
        self._cached_bid_step = -1
        self._cached_bid = None

        # Emit an initial (uniform) estimate so the scoring contract holds even
        # before the opponent has made any offer.
        self._rebuild_opponent_model()

    # ------------------------------------------------------- opponent model --
    def _rebuild_opponent_model(self) -> None:
        """Rebuild our estimate of the opponent's utility as an additive ufun.

        Two interpretable signals, both derived only from the opponent's own
        offers (their rejections are weak evidence and are ignored):

          * Value scores: within an issue, values the opponent offers more often
            are assumed better for them -> normalised frequency.
          * Issue weights: issues whose value the opponent rarely changes are
            assumed more important to them -> stability. Until we have a couple
            of offers we fall back to uniform weights.

        We only claim to recover the opponent's *ranking* of outcomes (which is
        exactly what the Kendall-tau scoring rewards) -- not cardinal utilities
        and not their reservation value.
        """
        value_funs = []
        for i in range(self._n_issues):
            counts = self._value_counts[i]
            max_count = max(counts.values()) if counts else 1.0
            # The tiny index gradient guarantees the value function is never
            # perfectly flat. A constant model makes Kendall-tau undefined, which
            # the scorer reads as -1 and which would forfeit the ENTIRE Concealing
            # point if we are scored before folding in any opponent offer (e.g. we
            # open and the opponent accepts immediately). The gradient is
            # negligible (1e-6) once real counts accumulate.
            # NOTE: a real [0,1] ramp here was tested as a forfeit-tail fix and
            # REJECTED -- it regressed tau_me (0.574->0.533) without raising the
            # min, because the con=0 cases are 3-4 outcome stress-mode artifacts
            # whose sparse data is simply wrong; they do not occur on the ~1000-
            # outcome live domains. See report/EXPERIMENTS.md.
            if self.MODEL_KIND == "pairwise":
                # ADDITIVE signal the frequency model discards: values WE keep
                # offering (and they keep rejecting) are probably bad for them.
                # score(v) = their_freq(v) - GAMMA * our_freq(v). Uses _my_value_
                # counts (the offers they rejected) -- genuinely new information,
                # not a re-processing of their sparse offers. GAMMA small so it
                # cannot dominate when our/their good values legitimately overlap
                # (cooperative domains).
                my = (self._my_value_weighted[i] if self.PAIRWISE_TIMEW
                      else self._my_value_counts[i])
                max_my = max(my.values()) if my else 1.0
                mapping = {
                    v: (counts.get(v, 0) / max_count if max_count > 0 else 0.0)
                    - self.PAIRWISE_GAMMA * (my.get(v, 0) / max_my if max_my > 0 else 0.0)
                    + 1e-6 * idx
                    for idx, v in enumerate(self._issue_values[i])
                }
            else:
                mapping = {
                    v: (counts.get(v, 0) / max_count if max_count > 0 else 0.0)
                    + 1e-6 * idx
                    for idx, v in enumerate(self._issue_values[i])
                }
            value_funs.append(TableFun(mapping=mapping))

        if self.MODEL_KIND == "optim" and self._opp_offers_seen >= 2:
            fitted = self._optim_model()
            if fitted is not None:
                self.private_info["opponent_ufun"] = fitted
                self._bid_model = None  # defense-in-depth: no stale bid model
                return

        if self.MODEL_KIND == "entropy" and self._opp_offers_seen >= 2:
            # Issue weight from the CONCENTRATION of the opponent's offered-value
            # distribution: an issue on which they keep offering the same few
            # values (low entropy) is important; one they vary freely (high
            # entropy) is not. This uses the full distribution rather than only
            # consecutive-change counts, and is a sharper estimator of the
            # issue-WEIGHT ranking, which dominates the full-space Kendall-tau.
            import math
            raw = []
            for i in range(self._n_issues):
                counts = self._value_counts[i]
                tot = sum(counts.values())
                k = len(self._issue_values[i])
                if tot <= 0 or k <= 1:
                    raw.append(1.0)
                    continue
                ent = -sum((c / tot) * math.log(c / tot) for c in counts.values() if c > 0)
                raw.append(max(0.0, 1.0 - ent / math.log(k)))  # 1 = concentrated
            total = sum(raw)
            weights = ([r / total for r in raw] if total > 0
                       else [1.0 / self._n_issues] * self._n_issues)
        elif (self.MODEL_KIND == "pairwise" and self.PAIRWISE_WEIGHT == "divergence"
              and self._opp_offers_seen >= 2):
            # N6: weight an issue by how much OUR offered value-distribution diverges
            # from THEIRS (total variation). High divergence = a contested issue we
            # keep pushing and they keep resisting = important to them. Uses the
            # same discarded our-offer signal as the pairwise value penalty.
            raw = []
            for i in range(self._n_issues):
                their = self._value_counts[i]
                tt = sum(their.values()) or 1.0
                mine = self._my_value_counts[i]
                mt = sum(mine.values()) or 1.0
                tv = 0.5 * sum(
                    abs(their.get(v, 0) / tt - mine.get(v, 0) / mt)
                    for v in self._issue_values[i]
                )
                raw.append(tv)
            total = sum(raw)
            weights = ([r / total for r in raw] if total > 0
                       else [1.0 / self._n_issues] * self._n_issues)
        elif self.MODEL_KIND == "bayesian" and self._weight_hypotheses:
            weights = self._bayesian_weights(value_funs)
        elif self._opp_offers_seen >= 2:
            # changes are counted across transitions => denominator is one less
            # than the number of offers seen.
            denom = max(1, self._opp_offers_seen - 1)
            raw = [
                max(0.0, 1.0 - (self._issue_changes[i] / denom))
                for i in range(self._n_issues)
            ]
            total = sum(raw)
            weights = (
                [r / total for r in raw]
                if total > 0
                else [1.0 / self._n_issues] * self._n_issues
            )
        else:
            weights = [1.0 / self._n_issues] * self._n_issues

        self.private_info["opponent_ufun"] = LinearAdditiveUtilityFunction(
            values=value_funs,
            weights=weights,
            outcome_space=self.ufun.outcome_space,
        )

        # DECOUPLE: bid on a PLAIN frequency model (no pairwise penalty) so our
        # revealed behaviour -- and thus the opponent's tau of us -- matches v4,
        # while the EMITTED (scored) model above stays the accurate pairwise one.
        self._bid_model = None
        if self.DECOUPLE_BID and self.MODEL_KIND == "pairwise":
            plain = []
            for i in range(self._n_issues):
                counts = self._value_counts[i]
                mx = max(counts.values()) if counts else 1.0
                plain.append(TableFun(mapping={
                    v: (counts.get(v, 0) / mx if mx > 0 else 0.0) + 1e-6 * idx
                    for idx, v in enumerate(self._issue_values[i])
                }))
            self._bid_model = LinearAdditiveUtilityFunction(
                values=plain, weights=weights,
                outcome_space=self.ufun.outcome_space,
            )

    def _bidding_ufun(self):
        """The opponent model used for BIDDING/strategy (may differ from the
        EMITTED, scored model when DECOUPLE_BID is on). Defaults to the emitted.

        DECOUPLE_PHASED (experiment): bid on the PLAIN model early (the phase where
        our offers teach the opponent's model of us -> keep tau_opp low), then switch
        to the sharp EMITTED (pairwise) model in the end-game (where deals close ->
        pick outcomes the opponent actually likes -> close better deals). Aims to add
        ADVANTAGE without the tau_opp cost of bidding sharp throughout."""
        bm = getattr(self, "_bid_model", None)
        if bm is None:
            return self.opponent_ufun
        if self.DECOUPLE_PHASED and getattr(self, "_last_t", 0.0) >= self._effective_times()[0]:
            return self.opponent_ufun  # sharp (emitted) model in the end-game
        return bm

    def _optim_model(self):
        """Max-entropy inverse-RL fit of the opponent's additive utility.

        Model: P(opponent offers o) ~ exp(u(o)), u(o)=sum_i w_i * s_i(o_i), with
        early offers up-weighted. We maximise (early-weighted offered utility) minus
        log-sum-exp over ALL outcomes -- i.e. make the outcomes they actually
        proposed score high relative to the whole space -- by OPTIM_ITERS steps of
        gradient ascent on the value scores s and (softmax-parameterised) weights w.
        Compute-heavy by design. Returns a LinearAdditiveUtilityFunction or None.
        """
        import numpy as np

        ni = self._n_issues
        if ni == 0 or not self._opp_offer_hist:
            return None
        # Cache the all-outcomes value-index matrix once per negotiation.
        if getattr(self, "_optim_idx", None) is None:
            os_ = self.ufun.outcome_space
            allo = list(os_.enumerate_or_sample(max_cardinality=100_000))
            self._optim_vmap = [
                {v: j for j, v in enumerate(self._issue_values[i])} for i in range(ni)
            ]
            self._optim_idx = np.array(
                [[self._optim_vmap[i][o[i]] for i in range(ni)] for o in allo]
            )
        idx = self._optim_idx                      # (N, ni)
        ks = [len(self._issue_values[i]) for i in range(ni)]
        orow = np.array(
            [[self._optim_vmap[i][o[i]] for i in range(ni)] for o, _ in self._opp_offer_hist]
        )                                          # (T, ni)
        ow = np.array([w for _, w in self._opp_offer_hist], dtype=float)
        ow = ow / max(1e-9, ow.sum())

        def softmax(z):
            z = z - z.max()
            e = np.exp(z)
            return e / e.sum()

        s = [np.zeros(ks[i]) for i in range(ni)]
        theta = np.zeros(ni)
        for _ in range(int(self.OPTIM_ITERS)):
            w = softmax(theta)
            S = np.stack([s[i][idx[:, i]] for i in range(ni)], axis=1)   # (N, ni)
            u_all = S @ w
            p = softmax(u_all)                                           # (N,)
            So = np.stack([s[i][orow[:, i]] for i in range(ni)], axis=1)  # (T, ni)
            for i in range(ni):
                g = np.zeros(ks[i])
                np.add.at(g, orow[:, i], ow * w[i])
                np.add.at(g, idx[:, i], -w[i] * p)
                s[i] += self.OPTIM_LR * g
            gw = (ow[:, None] * So).sum(0) - (p[:, None] * S).sum(0)     # (ni,)
            theta += self.OPTIM_LR * (w * (gw - (w * gw).sum()))

        # Normalise value scores to [0,1] (+tiny gradient so never constant).
        value_funs = []
        for i in range(ni):
            si = s[i]
            lo, hi = float(si.min()), float(si.max())
            rng = hi - lo if hi > lo else 1.0
            mapping = {
                v: (float(si[j]) - lo) / rng + 1e-6 * j
                for j, v in enumerate(self._issue_values[i])
            }
            value_funs.append(TableFun(mapping=mapping))
        w = softmax(theta)
        return LinearAdditiveUtilityFunction(
            values=value_funs, weights=[float(x) for x in w],
            outcome_space=self.ufun.outcome_space,
        )

    def _bayesian_weights(self, value_funs) -> list[float]:
        """Posterior-mean issue weights over rank-weight hypotheses.

        Likelihood model: a rational opponent offers outcomes that are high on
        *their* utility, especially early. For each candidate weight vector h we
        score how high the opponent's (early-weighted) offers look under
        u_h(o) = sum_i h_i * value_i(o_i); the posterior is a softmax over those
        scores, and we return the posterior-mean weight vector. Value functions
        are the frequency estimates (passed in) -- only the weights are Bayesian.
        """
        import math

        if not self._opp_offer_hist:
            return [1.0 / self._n_issues] * self._n_issues
        # Per-issue value score of each historical offer, cached once.
        vals = [
            [float(value_funs[i](o[i])) for i in range(self._n_issues)]
            for o, _ in self._opp_offer_hist
        ]
        wts = [w for _, w in self._opp_offer_hist]
        scores = []
        for h in self._weight_hypotheses:
            s = sum(
                wt * sum(h[i] * vals[t][i] for i in range(self._n_issues))
                for t, wt in enumerate(wts)
            )
            scores.append(s / max(1e-9, sum(wts)))
        mx = max(scores)
        post = [math.exp(self.BAYES_TEMP * (s - mx)) for s in scores]
        z = sum(post)
        post = [p / z for p in post]
        weights = [
            sum(post[k] * self._weight_hypotheses[k][i] for k in range(len(post)))
            for i in range(self._n_issues)
        ]
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights

    def update_opponent_model(self, state: SAOState) -> None:
        """Fold the opponent's latest offer into the frequency/stability model."""
        offer = state.current_offer
        if offer is None:
            return
        self._opp_offers_seen += 1
        self._opp_unique.add(offer)  # for MiCRO reciprocity counting
        n = self._opp_offers_seen
        u_to_us = float(self.ufun(offer))
        if self._opp_offers_seen == 1:
            self._first_opp_util = u_to_us
        if u_to_us > self._best_opp_util:
            self._best_opp_util = u_to_us
            self._best_opp_offer = offer
        # Opponent concession in OUR utility terms (how far their offers have
        # improved for us since their opening), normalised to our range.
        rng = max(1e-9, self._u_max - self._reserve)
        self._opp_concession = min(
            1.0, max(0.0, (self._best_opp_util - self._first_opp_util) / rng)
        )
        # Time-weight this offer for PREFERENCE modelling (see MODEL_WEIGHTING).
        if self.MODEL_WEIGHTING == "recency":
            w = float(n)
        elif self.MODEL_WEIGHTING == "early":
            w = 1.0 / n
        else:  # uniform
            w = 1.0
        if n == 1:  # the opening offer is the strongest preference signal
            w += self.OPENING_BOOST
        self._opp_offer_hist.append((offer, w))
        for i in range(self._n_issues):
            v = offer[i]
            self._value_counts[i][v] = self._value_counts[i].get(v, 0) + w
            if self._last_opp_value[i] is not None and self._last_opp_value[i] != v:
                self._issue_changes[i] += 1
            self._last_opp_value[i] = v
        self._rebuild_opponent_model()

    # ----------------------------------------------------- bidding strategy --
    def _effective_times(self) -> tuple[float, float]:
        """(rescue_time, secure_time) adjusted for the deadline length.

        At short deadlines a fixed relative window is too few rounds to close, so
        we pull each trigger earlier to leave at least MIN_*_ROUNDS rounds. At long
        deadlines the relative thresholds are unchanged."""
        n = None
        nmi = getattr(self, "nmi", None)
        if nmi is not None:
            n = getattr(nmi, "n_steps", None)
        secure_base = self.SECURE_BEST_TIME
        if self.ADAPTIVE_SECURE:
            # Hold longer (cave later) the more the opponent has conceded.
            secure_base = min(
                0.995, self.SECURE_BEST_TIME + self.ADAPT_SECURE_K * self._opp_concession
            )
        if not n or n <= 0:
            return self.RESCUE_TIME, secure_base
        rt = min(self.RESCUE_TIME, 1.0 - self.MIN_RESCUE_ROUNDS / n)
        # The absolute-round floor still guarantees we accept before timing out.
        st = min(secure_base, 1.0 - self.MIN_SECURE_ROUNDS / n)
        return max(0.0, rt), max(0.0, st)

    def _policy_target(self, t: float, rescue_time: float) -> float:
        """Concession target from the offline-trained RESIDUAL policy.

        target_fraction = Boulware_fraction(t) + correction(state), where the
        correction is a tanh-MLP over negotiation-state features, bounded by
        POLICY_CORR_SCALE. All-zero weights => correction 0 => the heuristic curve.
        Pure-numpy, deterministic, no runtime memory. The caller still applies the
        end-game rescue and secure-accept safety net.
        """
        import numpy as np
        rng = self._u_max - self._reserve
        if rng <= 0:
            return self._u_max
        # Heuristic Boulware fraction (the residual base).
        tt = t / rescue_time if rescue_time > 0 else 1.0
        bff = self.FAIR_FLOOR_FRACTION + (1.0 - self.FAIR_FLOOR_FRACTION) * (
            1.0 - tt ** (1.0 / self.CONCESSION_EXPONENT))
        # State features.
        opp_best = min(1.0, max(0.0, (self._best_opp_util - self._reserve) / rng))
        reserve_norm = self._reserve / self._u_max if self._u_max > 0 else 0.0
        conc = self._opp_concession
        n_norm = min(1.0, self._opp_offers_seen / 30.0)
        f = np.array([t, t * t, t * t * t, conc, opp_best, reserve_norm, n_norm,
                      opp_best * t, conc * t], dtype=float)
        # Parse flat weights into the MLP (W1,b1,W2,b2).
        nf, nh = self.POLICY_NFEAT, self.POLICY_NHID
        w = np.asarray(self.POLICY_WEIGHTS, dtype=float)
        i = nh * nf
        W1 = w[:i].reshape(nh, nf)
        b1 = w[i:i + nh]; i += nh
        W2 = w[i:i + nh]; i += nh
        b2 = w[i] if i < w.size else 0.0
        corr = self.POLICY_CORR_SCALE * float(np.tanh(W2 @ np.tanh(W1 @ f + b1) + b2))
        frac = min(1.0, max(self.RESCUE_FLOOR_FRACTION, bff + corr))
        return self._reserve + frac * rng

    def _target_utility(self, relative_time: float) -> float:
        """Utility target for the current time.

        Two regimes:
          * Normal (t < rescue_time): firm Boulware from our maximum down to the
            fair floor -- we refuse to concede below a fair share of our range,
            which is what stops a firm opponent from extracting its ideal.
          * Rescue (t >= rescue_time): release the floor, decaying from the fair
            floor down to the reservation value, so we still close a positive
            deal in the final rounds instead of timing out into a zero.
        rescue_time is deadline-adaptive (see _effective_times)."""
        rescue_time = self._effective_times()[0]
        t = min(max(relative_time, 0.0), 1.0)
        if self.ADAPTIVE_CONCESSION and self._opp_offers_seen >= 5:
            # Reciprocity: advance our clock in proportion to the opponent's
            # revealed concession (firm opponent -> ~0 -> unchanged).
            t = min(1.0, t * (1.0 + self.ADAPT_BETA * self._opp_concession))
        fair_floor = self._fair_floor
        if self.SCENARIO_CONDITION:
            # Shift the floor by domain competitiveness (cooperative => hold higher).
            comp = self._competitiveness()
            fair_floor = min(
                self._u_max,
                max(self._reserve, self._fair_floor
                    + self.COND_K * comp * (self._u_max - self._fair_floor)),
            )
        if t < rescue_time:
            if self.USE_LEARNED_POLICY:
                return self._policy_target(t, rescue_time)
            tt = t / rescue_time if rescue_time > 0 else 1.0  # renormalise firm window
            concession = 1.0 - tt ** (1.0 / self.CONCESSION_EXPONENT)
            return fair_floor + (self._u_max - fair_floor) * concession
        frac = (t - rescue_time) / max(1e-9, 1.0 - rescue_time)  # 0 -> 1
        rescue_floor = self._reserve + self.RESCUE_FLOOR_FRACTION * (
            self._u_max - self._reserve
        )
        return rescue_floor + (fair_floor - rescue_floor) * (1.0 - frac)

    def _competitiveness(self) -> float:
        """Domain competitiveness in [-1,1]: correlation between OUR utility and our
        ESTIMATE of the opponent's utility over a sample of outcomes. >0 cooperative
        (aligned preferences), <0 competitive/zero-sum. Cached per step. Returns 0
        (neutral) until we have a usable model."""
        if getattr(self, "_comp_cache", None) is not None and self._comp_step == self._opp_offers_seen:
            return self._comp_cache
        opp = self._bidding_ufun()
        if opp is None or self._opp_offers_seen < 2 or not self._sorted_outcomes:
            return 0.0
        import numpy as np
        outs = self._sorted_outcomes
        if len(outs) > 300:
            outs = outs[:: max(1, len(outs) // 300)]
        a = np.array([float(self.ufun(o)) for o in outs])
        b = np.array([float(opp(o)) for o in outs])
        comp = 0.0 if a.std() < 1e-9 or b.std() < 1e-9 else float(np.corrcoef(a, b)[0, 1])
        self._comp_cache = comp
        self._comp_step = self._opp_offers_seen
        return comp

    def _opp_reservation_est(self) -> float:
        """Estimate the opponent's acceptance floor (in OUR model's units).

        A rational opponent never proposes an outcome it would itself reject, so
        the lowest utility-to-them (under our model) among the offers they have
        made lower-bounds their reservation value. We relax it by RV_SLACK because
        their offers only lower-bound, not pin, that floor. Used by the "rvtarget"
        rescue selector to keep only bids they are likely to accept.
        """
        opp = self._bidding_ufun()
        if opp is None or not self._opp_offer_hist:
            return 0.0
        worst = min(float(opp(o)) for o, _ in self._opp_offer_hist)
        return self.RV_SLACK * worst

    def _band_candidates(self, target: float) -> list[Outcome]:
        """Outcomes whose utility (to us) is at least `target`.

        `_sorted_utils` is descending, so the band is a prefix. We always return
        at least our single best remaining outcome.
        """
        hi = 0
        n = len(self._sorted_outcomes)
        while hi < n and self._sorted_utils[hi] >= target:
            hi += 1
        return self._sorted_outcomes[: max(1, hi)]

    def _select_bid(self, candidates: list[Outcome], rescue: bool) -> Outcome:
        """Pick which bid to offer from our utility band.

        Among the band we favour the outcomes our opponent model rates highest
        (raises agreement probability at no cost to our own utility). Tie-break:
          * Normal: prefer the one best for *us* (protect Advantage).
          * Rescue: prefer the *fairest* (smallest gap between our and their
            estimated utility). This lets us table a genuine compromise in the
            end-game even when the opponent never revealed it -- e.g. the only
            closeable deal on a pure-conflict domain.
        """
        opponent = self._bidding_ufun()
        if opponent is None:
            return candidates[0]

        if self.LOOKAHEAD and self._opp_offer_hist:
            return self._lookahead_select(candidates, opponent)

        if self.PACCEPT_SEARCH and self._opp_offer_hist:
            # P(accept) rises from the opponent's estimated reservation (lo) to
            # their ideal (hi, ~their best self-offer). Pick argmax our_u * P.
            opp_utils = [float(opponent(o)) for o, _ in self._opp_offer_hist]
            lo = self.RV_SLACK * min(opp_utils)
            hi = max(opp_utils)
            span = max(1e-6, hi - lo)

            def p_accept(o):
                return min(1.0, max(0.0, (float(opponent(o)) - lo) / span))

            return max(candidates, key=lambda o: float(self.ufun(o)) * p_accept(o))

        if rescue:
            # End-game bid selection. The default ("fair") tables the most
            # balanced outcome in our band. RESCUE_SELECT switches in extraction-
            # oriented selectors that try to close at a HIGHER-for-us point that
            # the opponent is still likely to accept (deal-quality frontier):
            #   "rvtarget" : among band outcomes the opponent is predicted to
            #                accept (estimated from the worst they have offered),
            #                pick the BEST for us. Degenerates to caving vs a pure
            #                hardliner (only its ideal is "acceptable") and
            #                extracts vs accept-high / responsive rivals.
            #   "kalai"    : maximise min(our utility, their estimated utility).
            #   "nash"     : maximise our utility * their estimated utility.
            mode = self.RESCUE_SELECT
            if mode == "rvtarget":
                r_opp = self._opp_reservation_est()
                accept_pred = [o for o in candidates if float(opponent(o)) >= r_opp]
                if accept_pred:
                    return max(accept_pred, key=lambda o: float(self.ufun(o)))
            elif mode == "kalai":
                return max(
                    candidates,
                    key=lambda o: min(float(self.ufun(o)), float(opponent(o))),
                )
            elif mode == "nash":
                return max(
                    candidates,
                    key=lambda o: float(self.ufun(o)) * float(opponent(o)),
                )
            # "fair" (default) or rvtarget fallback: most balanced outcome.
            return min(
                candidates,
                key=lambda o: abs(float(self.ufun(o)) - float(opponent(o))),
            )

        if self.DECEPTION:
            # Diversify: prefer the band outcome whose values we have revealed
            # least so far (flattens our frequency signal), tie-broken toward the
            # outcome best for us. Every candidate already clears the fair floor,
            # so this costs ~no Advantage.
            def novelty(o):
                shown = sum(
                    self._my_value_counts[i].get(o[i], 0)
                    for i in range(self._n_issues)
                )
                return (shown, -float(self.ufun(o)))

            return min(candidates, key=novelty)

        if self.SELECTION_MODE == "greedy":
            return candidates[0]  # already sorted by our utility, descending
        if self.SELECTION_MODE == "nash":
            return max(
                candidates,
                key=lambda o: float(self.ufun(o)) * float(opponent(o)),
            )
        # "opponent": most opponent-friendly, ties broken toward our utility
        # (candidate order is our-utility-descending).
        return max(candidates, key=lambda o: float(opponent(o)))

    def _lookahead_select(self, candidates: list[Outcome], opponent) -> Outcome:
        """Expected-final-utility bid choice (the compute-heavy lookahead path).

        For each candidate o, EV(o) = P_accept(o)*u(o) + (1-P_accept)*continuation.
          * P_accept models a time-conceding opponent: it demands a share of its own
            range that falls from ~1 (its ideal) early to ~0 (its reservation) by the
            deadline; an offer above that demand is likely accepted, softened by
            LOOKAHEAD_SOFT so the choice is smooth, not a hard cliff.
          * continuation = max(our reservation, best the opponent has shown us): not
            closing now is not catastrophic, so EV correctly values holding high
            early and only conceding when continuation stops covering it.
          * LOOKAHEAD_MC>0 integrates EV over that many samples of the opponent's
            uncertain reservation (the genuinely time-spending Monte-Carlo path).
        """
        t = float(getattr(self, "_last_t", 0.0))
        opp_utils = [float(opponent(o)) for o, _ in self._opp_offer_hist]
        ideal = max(float(opponent(o)) for o in self._sorted_outcomes)
        rv_base = self.RV_SLACK * min(opp_utils)
        span = max(1e-6, ideal - rv_base)
        cont = max(self._reserve, self._best_opp_util)
        # MC samples of the opponent reservation (deterministic if MC==0).
        if self.LOOKAHEAD_MC > 0:
            m = int(self.LOOKAHEAD_MC)
            rvs = [rv_base + (ideal - rv_base) * (j / (2 * m)) for j in range(m)]
        else:
            rvs = [rv_base]

        def ev(o):
            u_o = float(self.ufun(o))
            v_o = float(opponent(o))
            acc = 0.0
            for rv in rvs:
                sp = max(1e-6, ideal - rv)
                share = (v_o - rv) / sp          # how good o is for them (0..1)
                demand = (1.0 - t)               # they demand near-ideal early
                p = (share - demand) / self.LOOKAHEAD_SOFT + 0.5
                acc += min(1.0, max(0.0, p))
            p_accept = acc / len(rvs)
            return p_accept * u_o + (1.0 - p_accept) * cont

        return max(candidates, key=ev)

    def _record_my_offer(self, outcome: Outcome) -> None:
        """Remember which values we have revealed (for deception diversification)."""
        if outcome is None:
            return
        self._my_unique.add(outcome)  # for MiCRO reciprocity counting
        self._my_offer_count += 1
        # Earliness weight: our EARLY offers sit near our ideal; when the opponent
        # rejects them they are the strongest signal about what is bad for them.
        ew = 1.0 / self._my_offer_count
        for i in range(self._n_issues):
            self._my_value_counts[i][outcome[i]] = (
                self._my_value_counts[i].get(outcome[i], 0) + 1
            )
            self._my_value_weighted[i][outcome[i]] = (
                self._my_value_weighted[i].get(outcome[i], 0.0) + ew
            )

    def _micro_bid(self, state: SAOState, t: float) -> Outcome:
        """MiCRO bid: minimal concession with reciprocity (the alternative
        bidding ARCHITECTURE). We walk our rational outcomes best-first and are
        entitled to have proposed at most (distinct opponent offers + 1) of them,
        so we never concede faster than the opponent does. A firm opponent (few
        distinct offers) => we hold near our ideal; a conceding one => we match its
        pace. Monotone (never un-concede). The deadline-adaptive rescue still forces
        full concession late, and LAST_OFFER_GRAB still banks the opponent's best on
        our final turn, so short deadlines cannot deadlock into a zero."""
        last = len(self._sorted_outcomes) - 1
        # Reciprocity entitlement: concede to index = number of DISTINCT offers the
        # opponent has made (idx 0 = our ideal at the opening, before they offer).
        idx = min(len(self._opp_unique), last)
        rescue_time = self._effective_times()[0]
        if t >= rescue_time:
            # Deadline pressure: blend toward full concession so a firm opponent
            # cannot deadlock us past the deadline into a no-deal.
            frac = (t - rescue_time) / max(1e-9, 1.0 - rescue_time)
            idx = max(idx, int(frac * last))
        self._micro_i = max(self._micro_i, idx)  # monotone concession
        # Final-turn safety: bank the opponent's best-shown (provably acceptable)
        # offer rather than table our near-reservation bid into a timeout.
        if self.LAST_OFFER_GRAB and self._best_opp_offer is not None:
            nmi = getattr(self, "nmi", None)
            n = getattr(nmi, "n_steps", None) if nmi else None
            step = getattr(state, "step", -1)
            if (n and (n - step) <= self.LAST_OFFER_ROUNDS
                    and self._best_opp_util > self._reserve):
                return self._best_opp_offer
        return self._sorted_outcomes[self._micro_i]

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        """Choose our counter-offer.

        Decouples *how much* to concede (the floored Boulware target) from
        *which* bid to make (the most opponent-friendly bid in our band).
        """
        step = getattr(state, "step", -1)
        if step == self._cached_bid_step and self._cached_bid is not None:
            return self._cached_bid

        t = state.relative_time
        self._last_t = t  # used by the LOOKAHEAD selector (P_accept time decay)

        if self.BID_STRATEGY == "micro":
            bid = self._micro_bid(state, t)
            self._cached_bid_step = step
            self._cached_bid = bid
            return bid
        # "Never undersell" (ChargingBoul): never let our concession target fall
        # below the best utility the opponent has already offered us -- proposing
        # worse than a deal we could already get is pure self-sabotage. The
        # end-game safety net still closes; this just stops us giving away value.
        if self.NEVER_UNDERSELL and self._best_opp_util > self._reserve:
            floor_target = self._best_opp_util - 1e-9
            t_target = self._target_utility(t)
            if t_target < floor_target:
                cands = self._band_candidates(floor_target)
                bid = self._select_bid(cands, rescue=t >= self._effective_times()[0])
                self._cached_bid_step = step
                self._cached_bid = bid
                return bid
        # First-mover end-game: on our final proposal (opponent gets no turn to
        # accept a fresh bid), table their own best-shown offer -- provably
        # acceptable to them -- rather than time out. Only when it beats walking
        # away; the rescue logic still governs ordinary rounds.
        if self.LAST_OFFER_GRAB and self._best_opp_offer is not None:
            nmi = getattr(self, "nmi", None)
            n = getattr(nmi, "n_steps", None) if nmi else None
            if n and (n - step) <= self.LAST_OFFER_ROUNDS and self._best_opp_util > self._reserve:
                self._cached_bid_step = step
                self._cached_bid = self._best_opp_offer
                return self._best_opp_offer
        target = self._target_utility(t)
        candidates = self._band_candidates(target)
        if self._decoy_issue is not None:
            # Hold our fixed value on the decoy issue so weight-learning opponents
            # over-rate it. Only narrow the band when some candidate still matches.
            frozen = [o for o in candidates if o[self._decoy_issue] == self._decoy_value]
            if frozen:
                candidates = frozen
        bid = self._select_bid(candidates, rescue=t >= self._effective_times()[0])

        self._cached_bid_step = step
        self._cached_bid = bid
        return bid

    # -------------------------------------------------- acceptance strategy --
    def acceptance_strategy(self, state: SAOState) -> bool:
        """Decide whether to accept the opponent's current offer.

        Rules, in order:
          * Hard floor -- never accept below our reservation value.
          * ACNext -- accept if the offer is at least as good for us as the bid
            we are about to make. Because our planned bid respects the fair floor,
            before the end-game we only accept genuinely good offers (we are not
            talked down to the opponent's ideal).
          * Rescue relaxation -- once in the end-game the planned bid (and hence
            this threshold) decays toward the reservation value, so we accept any
            safely-positive offer rather than time out into a zero.
        """
        offer = state.current_offer
        if offer is None:
            return False

        u_offer = float(self.ufun(offer))
        if u_offer < self._reserve:  # hard floor
            return False

        next_bid = self.concealing_bidding_strategy(state)
        if next_bid is not None and u_offer >= float(self.ufun(next_bid)):
            return True

        # Don't be so greedy we lose a near-best offer. If the opponent just made
        # us its best offer yet and it already clears our fair floor, take it --
        # holding out for our absolute maximum risks the offer never returning
        # (a conceding opponent's offers oscillate). Never fires against a firm
        # opponent, whose offers sit far below our floor.
        capture = self._reserve + self.CAPTURE_FRACTION * (self._u_max - self._reserve)
        if u_offer >= capture and u_offer >= self._best_opp_util - 1e-9:
            return True

        # End-game: secure the best the opponent has shown us rather than risk a
        # no-deal that may lose us the Concealing term -- but only if that offer
        # is not lopsided against us (by our model). A firm opponent's "best" is
        # its near-ideal, which fails this fairness test, so we stay firm and let
        # it yield; a genuinely conceding opponent's best is fair, so we bank it.
        if (
            self.SECURE_BEST
            and state.relative_time >= self._effective_times()[1]
            and (self.SECURE_ANY or u_offer >= self._best_opp_util - 1e-9)
            and u_offer >= self._reserve
                + self.SECURE_ACCEPT_FLOOR * (self._u_max - self._reserve)
            and u_offer > self._reserve
        ):
            if not self.SECURE_BEST_FAIR:
                # Field is mostly firm rivals whose best offer is fair-by-reality
                # but lopsided-by-our-model; banking it (it is already > our
                # reservation) closes a positive deal that beats a ~0.5 no-deal.
                return True
            opp = self._bidding_ufun()
            if opp is None or u_offer >= float(opp(offer)):
                return True
        return False

    # --------------------------------------------------------- main entry ----
    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """Respond to the opponent: accept, or reject with a counter-offer."""
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer

        # First move of the negotiation: open with a counter-offer.
        if offer is None:
            bid = self.concealing_bidding_strategy(state)
            self._record_my_offer(bid)
            return SAOResponse(ResponseType.REJECT_OFFER, bid)

        # Update our model with their offer *before* deciding, so acceptance and
        # bidding both use the freshest estimate.
        self.update_opponent_model(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        bid = self.concealing_bidding_strategy(state)
        self._record_my_offer(bid)
        return SAOResponse(ResponseType.REJECT_OFFER, bid)
