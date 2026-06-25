"""
Phantom8Negotiator — ANL 2026 agent.

Score = advantage + kopt_us/(kopt_us + kopt_them): modeling the opponent well is
worth exactly as much as concealing from them. Phantom8 layers four packages on
the board-validated Phantom2 core (Boulware + concession-ceiling curve-fit
hold-out at CONCEDE_BETA=0.04, single consistent lose-lose decoy):

0. DEADLINE-PARITY ULTIMATUM. Under AOP with a known fixed deadline, each
   mechanism step runs the first mover then the second, so the FIRST MOVER
   makes the final proposal of the whole negotiation and the second mover's
   last counter dies unseen. As first mover: refuse mediocre offers in the
   last two steps (only when the ultimatum candidate beats them by a 0.15
   premium — insurance against bar-keepers that reject ultimatums) and make
   the final proposal max-for-us subject to clearing the opponent's estimated
   acceptance bar. As second mover the power does not exist; behavior is
   unchanged. Paired: +0.069 vs tough holders, +0.033 vs deceivers, zero
   delta vs the honest field.

1. TWO DECOUPLED OPPONENT MODELS. Internal decisions (bid selection, decoy
   choice, endgame) use a battle-tested unique-weighted frequency model. The
   EXPOSED `private_info["opponent_ufun"]` — the only thing the grader
   Kendall-scores — is a RANKFIT estimate: ridge regression on one-hot
   issue-value features, offered outcomes targeted by their declining
   aspiration over time, unoffered outcomes anchored low, plus a robust pass
   that detects and discounts decoy offers (they contradict the fit). RankFit
   matches frequency counting against honest opponents and beats it by
   +0.05-0.06 kopt against decoy-led ones. A fast-fading anti-prior (additive
   projection of our negated utility) covers the low-data regime — a uniform
   model is all ties and scores kopt 0.
2. CONCEALMENT SAFETY. The decoy runs while the opponent has not conceded much
   (AGREEABLE_TH=0.80) but is skipped whenever the model says it clears their
   current acceptance bar (DECOY_SAFE — time-based conceders accept junk once
   their aspiration sinks), and is chosen to avoid values THEY offered (a model
   fooled by their decoy would otherwise pick a decoy they pretend to love).
3. STONEWALLED-REGIME RECIPROCITY. If the opponent has given us ~nothing by
   midgame: concede one step each time their best-to-us improves (TFT-family
   opponents mirror our movement; stonewallers get nothing), interleave the
   decoy with fresh band offers (visible movement for TFT/MiCRO without
   surrendering the frequency-corruption of the decoy), cycle fresh outcomes
   in-band (MiCRO concedes per unique offer seen), and in tiny domains accept
   the rational compromise late. Plus LAST-CALL: with <=2 proposals left,
   re-propose THEIR best offer rather than time out at advantage 0.

Paired vs the Phantom5 predecessor (n=1386, grader-faithful metric, field incl.
deceptive + reciprocal opponents): +0.0094 overall [CI +0.006,+0.013], with
+0.0495 vs decoy-deceivers, +0.032 vs tit-for-tat, +0.024 vs MiCRO, ~0 vs
honest conceders/modelers. Self-contained (no cross-file imports).
"""

import random
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import pareto_frontier, nash_points
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import TableFun


class Phantom8Negotiator(SAOCallNegotiator):
    # ---- deception knobs (inherited behavior from Veil) ----
    DECEPTION_MODE = "off"            # balanced | aggressive | off
    DECEPT_END = 0.75
    AGREEABLE_TH = 0.80              # decoy stays on unless they concede a LOT (0.65
                                     # turned it off vs ordinary conceders, handing
                                     # them an accurate read of us for nothing). 0.80
                                     # captures most of the measured share gain
                                     # (0.561->0.575, 0.90 only adds +0.003) with less
                                     # junk-bid exposure vs behavior-reciprocal real
                                     # agents the local proxies can't represent
    DECEPT_FLOOR = 0.45
    MODEL_DECAY = 0.85
    HONESTY_FILTER = False           # down-weight offers they don't value (measured:
                                     # slightly HURTS — self-reinforcing; kept off)
    CONCESSION_FILTER = False        # trust offers that ADVANCE the concession frontier;
                                     # down-weight regressions. MEASURED: hurts (TauB
                                     # vs Garret 0.674->0.575) — a single early probe
                                     # ratchets the frontier, then genuine ideal-holding
                                     # offers get mislabeled as regressions. Kept off.
    CONCESSION_TOL = 0.05            # slack before an offer counts as a regression
    OFF_PATH_FLOOR = 0.15            # weight kept for off-concession (deceptive) offers
    EARLY_ACCEPT_BOOST = False       # fold an early acceptance of OUR offer into the
                                     # model as a high-for-them point (measured: helps
                                     # vs real-threshold modelers e.g. BOA, but HURTS
                                     # vs push-overs like Nice/Random where "early
                                     # accept" ≠ high-for-them; net negative, kept off)
    HONESTY_FLOOR = 0.2              # min weight kept for a "junk"-looking offer
    ENDGAME_T = 0.95
    ENDGAME_OPP_FLOOR = 0.25
    MAX_OUTCOMES = 20000
    # ---- RUFL concession-prediction knobs ----
    REJECTION_SIGNAL = False         # use their rejections of OUR generous offers as a
                                     # toughness signal to close earlier. MEASURED: no
                                     # effect — the remaining no-deals are STRUCTURAL
                                     # (no rational deal existed), not impatience; conceding
                                     # more just lowers advantage on closable deals. Off.
    REJ_RELAX = 0.20                 # max downward target adjustment from that signal
    NASH_ANCHOR = False              # concede toward our NASH utility (vs estimated opp
                                     # ufun) instead of reservation — the ANL2024 winners'
                                     # technique. MEASURED: net-negative unconditional
                                     # (helps tough opponents e.g. InverseConceder +0.036,
                                     # hurts exploitable conceders), neutral when gated to
                                     # tough-only. Doesn't transfer: 2024 agents use the TRUE
                                     # opp ufun (given by the framework); our noisy frequency
                                     # estimate miscalibrates the Nash point. Off; toggleable.
    NASH_RECOMPUTE = 10              # recompute the Nash floor every N opponent offers
    NASH_MARGIN = 0.03               # concede very slightly below Nash to help close
    NASH_TOUGH_TH = 0.60             # only apply the Nash floor when the opponent is tough
                                     # (best offer to us below this) — conceders get exploited
    RV_PUSH = False                  # estimate opponent reserved value from their concession
                                     # trajectory and hold out for the best-for-US outcome
                                     # that still clears it. MEASURED: null (1.510 vs 1.513) —
                                     # mapping RV->outcomes needs our noisy opp-ufun estimate,
                                     # same failure as Nash. Off; toggleable. (The RV scalar is
                                     # observable-ish but the outcome mapping is not.)
    RV_MARGIN = 0.05                 # give them a little above the estimated RV so they accept
    BOULWARE_BETA = 0.25             # Veil's robust floor curve (open 1.0 -> reserve)
    CONCEDE_BETA = 0.04              # REAL-BOARD EVIDENCE: Phantom2 (0.04) ranks ~#8 while
                                     # Phantom3/4 (0.02) rank >10 — the tighter hold-out
                                     # over-holds against the real field even though every
                                     # LOCAL proxy field says 0.02 is better (local stand-in
                                     # opponents concede more than real submissions). Trust
                                     # the board, not the proxy.
    E_INIT = 17.0                    # initial concession exponent (very Boulware)
    E_EMA = 0.7                      # weight on previous e estimate when smoothing
    MIN_POINTS = 4                   # min villain offers before trusting the fit
    MIN_UNIQUE = 3                   # min distinct villain offers before trusting it
    # ---- Phantom "mirage" deception knobs (the real, validated deception lever) ----
    PHANTOM = True                   # present a CONSISTENT lose-lose decoy early: it
                                     # misrepresents our preferences (low-for-us values,
                                     # offered repeatedly so their model converges to it ->
                                     # lowers a_them -> raises Deception) AND is bad-for-them
                                     # so they reject it -> ZERO advantage cost.
    DECOY_END = 0.80                 # present the decoy until this fraction of time
    DECOY_BOTTOM = 0.75              # draw the decoy from our bottom (1-this) utility band
    # ---- COHERENT false-profile decoy (defeats decoy-RESISTANT smart modelers) ----
    COHERENT = False                 # REAL-BOARD EVIDENCE: Phantom4 (coherent decoy) ranks
                                     # below Phantom2 (single decoy) on the live board; the
                                     # SmartModeler upside was hypothetical. Reverted to the
                                     # single reliably-rejected decoy. Toggleable.
                                     # Original rationale: against a TOUGH opponent, present a VARIED lose-lose
                                     # sequence (a believable fake concession path) instead
                                     # of one repeated outcome — a sophisticated modeler that
                                     # discounts static/repeated junk still converges to the
                                     # FALSE profile. Validated vs a unique-weighting
                                     # SmartModeler: kopt_them 0.63->0.40, score +0.06..0.11,
                                     # advantage held. Single decoy kept vs conceders (safe).
    TOUGH_DECOY_TH = 0.45            # only use the varied coherent decoy when the opponent is
                                     # this-tough (best offer to us below TH -> they REJECT our
                                     # junk -> no advantage cost). Conceders -> single decoy.
    # ---- Phantom8: decoy-resistant opponent model + structural prior ----
    MODEL_MODE = "uniq"              # "uniq": each DISTINCT opponent offer counts once,
                                     # so a repeated decoy cannot dominate the model
                                     # (the real field is full of concealers — vs phantom3
                                     # self-play our kopt collapses 0.85 -> 0.57 with the
                                     # legacy model); "freq": legacy recency frequency
    RANK_DECAY = 0.97                # soft decay over distinct offers in first-appearance
                                     # order (honest conceders lead with their best)
    MIX = 0.5                        # weight of the unique table in "mix" mode
    PRIOR_K = 1.0                    # prior->data blend half-life (in distinct offers);
                                     # MUST fade fast — at K=4 the prior drags an accurate
                                     # data model down (measured -0.02 score)
    PRIOR_EPS = 0.01                 # always-on prior share: breaks ties (an all-ties
                                     # model scores kendall tau NaN -> kopt 0); keep tiny
    LAST_CALL = True                 # with <=2 of our proposals left, re-propose the best
                                     # offer THEY made (a near-sure accept) if it clears
                                     # our reserve — converts late stalemates into deals
    TINY_N = 6                       # tiny-domain safety: with <= this many rational
                                     # outcomes the compromise IS the deal — accept any
                                     # above-reserve offer after TINY_T_ACC instead of
                                     # gambling on capitulation (bidding stays default:
                                     # changing tiny-domain bids measured WORSE vs Garret)
    TINY_T_ACC = 0.85                # tiny mode: accept anything > reserve after this
    RECIP = True                     # reciprocity for the STONEWALLED regime: TFT-family
                                     # opponents mirror OUR concessions; decoy + Boulware
                                     # hold gives them no signal -> mutual deadlock (adv
                                     # 0.61, 14% no-deals vs NaiveTitForTat). Once the
                                     # regime triggers (ADAPT_T/ADAPT_OPP_TH), concede ONE
                                     # step each time their best-to-us IMPROVES in response:
                                     # a TFT spirals to the middle with us; a stonewaller
                                     # gets nothing (no response -> we hold).
    RECIP_STEP = 0.05                # our-utility step per reciprocated concession
    RECIP_EPS = 0.01                 # their improvement needed to count as a response
    ADAPT_DECOY = False              # ending the decoy early costs more concealment than
                                     # reciprocity wins back (kThem 0.54->0.63 measured);
                                     # instead the stonewalled regime INTERLEAVES decoy
                                     # (odd steps, consistency preserved) with fresh band
                                     # offers (even steps, visible movement for TFT/MiCRO)
    ADAPT_T = 0.50                   # stalemate check time
    ADAPT_OPP_TH = 0.35              # ...best offer to us still below this (normalized)
    ADAPT_DECOY_END = 0.50           # effective DECOY_END in that regime
    DECOY_SAFE = True                # never present a decoy that our model says clears
                                     # the opponent's CURRENT acceptance bar (their own
                                     # latest offer's value-to-themselves minus margin).
                                     # Time-based conceders accept lose-lose junk once
                                     # their aspiration sinks (45% bottom-band deals vs
                                     # ConcederTB unguarded!); modelers/Boulware keep a
                                     # high bar so decoys continue against them
    SAFE_MARGIN = 0.10               # skip decoy if its their-value >= bar - this
    SAFE_FROM_T = 0.40               # decoys are always safe before this time: threshold
                                     # agents hold a high bar early (all measured junk
                                     # acceptances are late), and early the model is
                                     # poisoned by THEIR decoy (their junk scores high ->
                                     # our whole bottom band looks falsely acceptable)
    DECOY_SHAPED = False             # multi-decoy with SHAPED frequencies (3:2:1 cycle
                                     # teaching an inverted value ordering): the kThem
                                     # mechanism WORKS (BOA 0.529->0.501, OraclePlain
                                     # 0.585->0.525) but varied junk corrupts opponents'
                                     # concession-prediction of US — they hold harder,
                                     # adv -0.04 vs holders, net -0.026. Third independent
                                     # confirmation (after Phantom4 board + curtain) that
                                     # the SINGLE consistent decoy is special. Keep off.
    SHAPED_K = 3                     # number of decoy outcomes in the cycle
    GREEDY_BAND = False              # conceder squeeze: vs a clearly-conceding opponent
                                     # (their bar keeps dropping with time), pick the
                                     # best-for-US outcome in the bid band until SQUEEZE_T
                                     # instead of the most-acceptable one — we close at
                                     # t~0.87 with 13% of their concession clock unused
    SQUEEZE_T = 0.93                 # revert to acceptable picks after this (to close)
    SQUEEZE_OPP = 0.55               # gate: opponent best offer to us at least this

    # ------------------------------------------------------------------ setup
    def on_preferences_changed(self, changes):
        if self.ufun is None:
            return
        self.reservation_value = float(self.ufun.reserved_value or 0.0)
        # defensive: if a runner ever uses time-limited (n_steps=None)
        # negotiations, every deadline comparison must still work
        self.total_steps = self.nmi.n_steps or (1 << 30)
        self.issues = self.nmi.outcome_space.issues
        self.n_issues = len(self.issues)

        outcomes = list(self.nmi.outcome_space.enumerate_or_sample())
        if len(outcomes) > self.MAX_OUTCOMES:
            outcomes = random.sample(outcomes, self.MAX_OUTCOMES)
        utils = [(o, float(self.ufun(o))) for o in outcomes]
        self.min_util = min(u for _, u in utils)
        self.max_util = max(u for _, u in utils)

        self.sorted_outcomes = sorted(
            ((o, u, self._norm(u)) for o, u in utils if u > self.reservation_value),
            key=lambda x: x[1], reverse=True)
        if not self.sorted_outcomes:
            o, u = max(utils, key=lambda x: x[1])
            self.sorted_outcomes = [(o, u, 1.0)]

        rational = [(o, n) for o, _, n in self.sorted_outcomes if n >= self.DECEPT_FLOOR]
        if len(rational) < 10:
            rational = [(o, n) for o, _, n in self.sorted_outcomes]
        self._decep_pool = (random.sample(rational, 1500)
                            if len(rational) > 1500 else rational)

        # opponent ufun model (recency- + honesty-weighted frequency) — for Deception
        self._opp_offers = []            # every offer they made, in order
        self._bonus = []                 # ground-truth (outcome, weight) points, e.g.
                                         # an early acceptance reveals a high-for-them deal
        self._last_t = 0.0
        self._n_opp_offers = 0
        self.best_opp_offer = None
        self.best_opp_offer_util = float("-inf")
        self._our_counts = [defaultdict(int) for _ in range(self.n_issues)]
        self._our_offered = set()        # outcomes we proposed (to detect THEY accepted ours)
        # rejection-bound signal: when they reject an offer of ours, it was below their
        # threshold. Track how generous (by our model, normalized) our rejected offers
        # were — a high value late = a tough opponent we should close with.
        self._last_our_offer = None
        self._max_rej_their = 0.0
        self._opp_lo = float("inf")
        self._opp_hi = float("-inf")

        # RUFL concession-prediction state — for Advantage
        self._villain_ut = []            # list of (normalized_util, relative_time)
        self._unique_villain = set()
        self._e = self.E_INIT
        self._maxu = 0.99

        # Nash-anchor state (normalized our-utility at the Nash point), recomputed lazily
        self._nash_norm = None
        self._nash_last = -999
        self._rv_norm = None
        self._rv_last = -999
        # sample spanning the WHOLE rational space (both corners) for Pareto/Nash
        _rat = [o for o, _, _ in self.sorted_outcomes]
        self._nash_sample = random.sample(_rat, 1500) if len(_rat) > 1500 else _rat
        self._decoy = None               # cached consistent lose-lose decoy offer
        # reciprocity state (stonewalled-regime TFT mirroring)
        self._recip_on = False
        self._recip_level = 1.0          # our standing demand in that regime
        self._recip_best = 0.0           # their best-to-us when we last stepped
        self._tiny = len(self.sorted_outcomes) <= self.TINY_N
        # coherent false-profile pool: our worst outcomes, ordered worst-for-us first
        # (= the FAKE ideal) so progressing through them traces a believable fake concession
        _cut = int(len(self.sorted_outcomes) * self.DECOY_BOTTOM)
        _pool = [o for o, _, _ in self.sorted_outcomes[_cut:]] or [self.sorted_outcomes[-1][0]]
        self._decoy_pool = _pool[::-1]

        # Structural anti-prior: an additive projection of our NEGATED own utility.
        # Bargaining domains are mostly competitive (preferences opposed), so with
        # little or no data "they like what we don't" beats a uniform model — which
        # is all ties and scores kendall tau NaN -> kopt 0 with the grader.
        self._anti_vals, self._anti_w = self._build_anti_prior(utils)
        self._set_model_prior()

    # ------------------------------------------------------------ small utils
    def _norm(self, u: float) -> float:
        if self.max_util > self.min_util:
            return (u - self.min_util) / (self.max_util - self.min_util)
        return 1.0

    def _build_anti_prior(self, utils):
        """Per-issue value tables + issue weights approximating -ufun(ours) as a
        linear-additive function (mean our-utility per value, inverted per issue;
        issue weight = the spread our ufun puts on that issue)."""
        sums = [defaultdict(float) for _ in range(self.n_issues)]
        cnts = [defaultdict(int) for _ in range(self.n_issues)]
        for o, u in utils:
            n = self._norm(u)
            for i, v in enumerate(o):
                sums[i][v] += n
                cnts[i][v] += 1
        tables, spreads = [], []
        for i, issue in enumerate(self.issues):
            means = {v: (sums[i][v] / cnts[i][v]) if cnts[i][v] else 0.5
                     for v in issue.values}
            lo, hi = min(means.values()), max(means.values())
            spread = hi - lo
            if spread > 1e-9:
                tables.append({v: (hi - m) / spread for v, m in means.items()})
            else:
                nv = max(len(issue.values) - 1, 1)
                tables.append({v: j / nv for j, v in enumerate(issue.values)})
            spreads.append(max(spread, 1e-6))
        tot = sum(spreads)
        return tables, [s / tot for s in spreads]

    def _set_model_prior(self):
        self.private_info["opponent_ufun"] = LinearAdditiveUtilityFunction(
            values=[TableFun({v: self._anti_w[i] * s
                              for v, s in self._anti_vals[i].items()})
                    for i in range(self.n_issues)],
            outcome_space=self.nmi.outcome_space)

    # ----------------------------------------------------------------- driver
    # ---- deadline-parity ultimatum (Phantom8) ----
    ULTIMATUM = True                 # AOP with a KNOWN fixed deadline: within each step
                                     # the first mover proposes, then the second responds —
                                     # so the FIRST MOVER makes the final proposal of the
                                     # whole negotiation, and at the last step the second
                                     # mover's only real choice is accept-or-nothing. When
                                     # WE move first, the final proposal is take-it-or-
                                     # leave-it power: don't accept mediocre late offers,
                                     # and make the last offer max-for-us subject to
                                     # clearing their estimated acceptance bar.
    ULT_ACCEPT = 0.60                # as first mover in the last 2 steps, still accept
                                     # their offer if it clears this (bounds the downside:
                                     # the ultimatum only replaces BAD endings)
    ULT_BAR_MARGIN = 0.05            # ultimatum must clear their estimated bar by this
    ULT_RV_FLOOR = 0.15              # ...and never sit below this their-value percentile
                                     # (clears unknown reservation values)
    ULT_PREMIUM = 0.15               # insurance: only reject their late offer for the
                                     # ultimatum when the ultimatum is worth at least
                                     # this much more — bounds the bar-keeper downside
                                     # (MiCRO-likes reject ultimatums: 12% no-deals
                                     # without this guard)
    ULT_BARKEEPER = False            # detect MiCRO-like bar-keepers (their unique-offer
                                     # count mirrors OURS — MiCRO's defining mechanic)
                                     # and UNCAP the ultimatum bar for them: clear their
                                     # real threshold instead of gambling they deadline-
                                     # accept (they don't: 12% no-deals)
    ULT_BK_MIN_UNIQ = 10             # bar-keeper gate: at least this many uniques
    ULT_BK_RATIO = 0.5               # ...and >= this fraction of our unique count
    RECIP_MIN_OUT = 100              # reciprocity only on domains with at least this
                                     # many rational outcomes: its TFT/MiCRO gains come
                                     # from rich domains; on coarse small grids the
                                     # 0.05-steps concede whole outcomes at a time
                                     # (measured -0.027 on generated 1-4 issue domains)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        self._last_t = state.relative_time
        if not hasattr(self, "_first_mover"):
            self._first_mover = state.current_offer is None
        offer = state.current_offer
        if offer is not None:
            # They countered instead of accepting -> our last offer was rejected.
            if self._last_our_offer is not None:
                self._note_rejection(self._last_our_offer)
            self._update_opponent_model(offer, state)
        target = self._target(state)
        if offer is not None and self._accept(state, offer, target):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)
        bid = self._choose_bid(state, target)
        self._last_our_offer = bid
        return SAOResponse(ResponseType.REJECT_OFFER, bid)

    @property
    def _opp_model(self):
        """Internal (frequency) opponent model for decisions; the EXPOSED
        private_info["opponent_ufun"] is the RankFit estimate the grader scores."""
        m = getattr(self, "_internal_opp", None)
        return m if m is not None else self.private_info["opponent_ufun"]

    def _note_rejection(self, our_offer: Outcome) -> None:
        """Record that the opponent rejected our offer: their_util(our_offer) was
        below their threshold. Track the most generous (by our model) rejected offer,
        normalized against the their-utility range we've observed."""
        opp = self._opp_model
        u = float(opp(our_offer))
        self._opp_lo = min(self._opp_lo, u)
        self._opp_hi = max(self._opp_hi, u)
        if self._opp_hi > self._opp_lo:
            self._max_rej_their = max(self._max_rej_their,
                                      (u - self._opp_lo) / (self._opp_hi - self._opp_lo))

    # --------------------------------------------- RUFL concession prediction
    def _predicted_ceiling(self) -> float | None:
        """Fit the opponent's normalized utility-vs-time and return how high we
        predict their offers will reach (their concession ceiling). None until we
        have enough data to trust the fit."""
        pts = self._villain_ut
        if len(pts) < self.MIN_POINTS or len(self._unique_villain) < self.MIN_UNIQUE:
            return None
        times = np.array([t for _, t in pts], dtype=float)
        utils = np.array([u for u, _ in pts], dtype=float)
        base = min(max(pts[0][0], self._norm(self.reservation_value)), 0.99)

        def model(t, e, mx):
            return np.minimum(np.power(t, e) + base, mx)

        try:
            popt, _ = curve_fit(
                model, times, utils, p0=[self._e, self._maxu],
                bounds=([1.0, max(base, 0.5)], [50.0, 1.0]), maxfev=3000)
            self._e = self.E_EMA * self._e + (1 - self.E_EMA) * float(popt[0])
            self._maxu = float(popt[1])
        except Exception:
            pass
        # Never predict below what they have already offered us, nor below 0.5.
        return min(max(self._maxu, self._norm(self.best_opp_offer_util), 0.5), 1.0)

    def _nash_floor(self) -> float:
        """Normalized our-utility at the Nash bargaining point, computed against our
        ESTIMATED opponent ufun. We never voluntarily concede below this — the core
        technique of the strong ANL2024 bargainers. Recomputed lazily as the model
        improves; falls back to reservation on any failure."""
        if not self.NASH_ANCHOR:
            return self._norm(self.reservation_value)
        if self._nash_norm is not None and self._n_opp_offers - self._nash_last < self.NASH_RECOMPUTE:
            return self._nash_norm
        self._nash_last = self._n_opp_offers
        try:
            opp = self._opp_model
            fu, _ = pareto_frontier([self.ufun, opp], self._nash_sample)
            nps = nash_points([self.ufun, opp], fu, outcome_space=self.nmi.outcome_space)
            nash_my_u = float(nps[0][0][0]) if nps else self.reservation_value
            self._nash_norm = max(0.0, self._norm(nash_my_u) - self.NASH_MARGIN)
        except Exception:
            self._nash_norm = self._norm(self.reservation_value)
        return self._nash_norm

    def _rv_floor(self) -> float:
        """Estimate the opponent's reserved value from their concession trajectory
        (their utility-to-THEMSELVES, via our model, declining over time -> asymptote),
        then return the normalized OUR-utility of the best-for-us outcome that still
        gives them >= RV. Holding out for that pushes us right up to their walk-away
        point. No true opp ufun needed — only the concession shape + the RV<=min bound."""
        if not self.RV_PUSH:
            return 0.0
        if self._rv_norm is not None and self._n_opp_offers - self._rv_last < self.NASH_RECOMPUTE:
            return self._rv_norm
        self._rv_last = self._n_opp_offers
        try:
            opp = self._opp_model
            svals = [float(opp(o)) for o in self._nash_sample]
            lo, hi = min(svals), max(svals)
            rng = (hi - lo) or 1.0
            # opponent utility-to-themselves of each of their offers, normalized, vs time
            tt = np.array([t for _, t in self._villain_ut], dtype=float)
            uu = np.array([(float(opp(o)) - lo) / rng for o in self._opp_offers], dtype=float)
            if len(uu) < self.MIN_POINTS or len(set(np.round(uu, 3))) < self.MIN_UNIQUE:
                return self._rv_norm or 0.0
            u0 = max(uu[0], float(uu.max()))
            minu = float(uu.min())

            def model(t, rv, e):
                return u0 - (u0 - rv) * np.power(t, e)
            try:
                popt, _ = curve_fit(model, tt, uu, p0=[min(minu, 0.3), 2.0],
                                    bounds=([0.0, 0.2], [max(minu, 1e-3), 10.0]), maxfev=3000)
                rv = float(popt[0])
            except Exception:
                rv = minu
            if rv > 0.6:        # anti-bluff: a "hard-head" spoofing a high RV
                rv = 0.25
            rv = min(rv, minu) + self.RV_MARGIN
            # best-for-us (normalized) outcome that still gives them >= rv
            best = 0.0
            for o, sv in zip(self._nash_sample, svals):
                if (sv - lo) / rng >= rv:
                    best = max(best, self._norm(float(self.ufun(o))))
            self._rv_norm = best
        except Exception:
            self._rv_norm = self._rv_norm or 0.0
        return self._rv_norm

    def _target(self, state: SAOState) -> float:
        """Adaptive normalized target: hold the predicted ceiling / Nash floor early,
        relax toward the Nash floor near the deadline so we still close."""
        t = state.relative_time
        res_n = self._norm(self.reservation_value)
        # Floor at the Nash utility (vs estimated opp) ONLY against tough opponents
        # (those keeping us starved). Against conceders giving us a lot, skip it and
        # keep exploiting — holding above the "fair" Nash split extracts more.
        floor = res_n
        if (self._n_opp_offers >= 3
                and self._norm(self.best_opp_offer_util) < self.NASH_TOUGH_TH):
            floor = max(res_n, self._nash_floor())
        # Boulware: open at 1.0, concede toward the Nash floor (not reservation).
        boulware = floor + (1.0 - floor) * (1.0 - t ** (1.0 / self.BOULWARE_BETA))
        target = boulware
        # Held-long curve (slow CONCEDE_BETA): open at 1.0, concede toward `x`.
        held = lambda x: res_n + (x - res_n) * (1.0 - t ** (1.0 / self.CONCEDE_BETA))
        # Predicted concession ceiling (what they'll give us) — only ever RAISES demand.
        ceil = self._predicted_ceiling()
        if ceil is not None:
            target = max(target, held(ceil))
        # Reserved-value push: hold out for the best-for-us outcome that still clears
        # their estimated RV — extract right up to their walk-away point.
        rv_floor = self._rv_floor()
        if rv_floor > 0.0:
            target = max(target, held(rv_floor))

        # Rejection-bound toughness: if late in the game we've offered them generous
        # deals (high their-util) and they STILL rejected, they're tough -> lower our
        # target to close instead of timing out at Advantage 0. Junk offers we make
        # during the deception phase are low-their-util, so they don't trigger this.
        if self.REJECTION_SIGNAL and t > 0.6 and self._max_rej_their > 0.5:
            tough = (self._max_rej_their - 0.5) * 2.0 * (t - 0.6) / 0.4
            target -= self.REJ_RELAX * max(0.0, min(1.0, tough))

        # Reciprocity (stonewalled regime only): TFT opponents move only when WE
        # move. Enter the regime when they have given us ~nothing by midgame;
        # then step our demand down by RECIP_STEP each time their best-to-us
        # improves by RECIP_EPS since our last step. No response -> no free
        # concession (stonewallers unchanged); conceders never enter (gate).
        if (self.RECIP and self._n_opp_offers >= 3
                and len(self.sorted_outcomes) >= self.RECIP_MIN_OUT):
            best_n = self._norm(self.best_opp_offer_util)
            if (not self._recip_on and t >= self.ADAPT_T
                    and best_n < self.ADAPT_OPP_TH):
                self._recip_on = True
                # one unconditional probe step: a TFT will not move until WE do
                self._recip_level = min(target, 1.0) - self.RECIP_STEP
                self._recip_best = best_n
            if self._recip_on:
                if best_n >= self._recip_best + self.RECIP_EPS:
                    self._recip_level -= self.RECIP_STEP
                    self._recip_best = best_n
                # never below the time floor (boulware) or what they already give
                lvl = max(self._recip_level, boulware, best_n)
                target = min(target, lvl)
        return max(res_n, target)

    # ------------------------------------------------------------- acceptance
    def _accept(self, state: SAOState, offer: Outcome, target: float) -> bool:
        u = float(self.ufun(offer))
        if u <= self.reservation_value:
            return False
        if state.step >= self.total_steps - 2:
            # First mover holds the FINAL proposal: don't take a mediocre late
            # offer — the coming ultimatum is worth more if they are a rational
            # deadline-accepter. Insurance: only hold out when the ultimatum
            # candidate beats their offer by a real premium (bar-keepers like
            # MiCRO reject ultimatums; a marginal edge isn't worth the gamble).
            if (self.ULTIMATUM and getattr(self, "_first_mover", False)
                    and self._norm(u) < self.ULT_ACCEPT):
                cand = self._ultimatum_offer()
                if (cand is not None
                        and self._norm(float(self.ufun(cand)))
                        >= self._norm(u) + self.ULT_PREMIUM):
                    return False
            return True
        # Tiny domains: the rational compromise IS the deal — take it late
        # rather than gamble on capitulation.
        if (self._tiny and state.relative_time >= self.TINY_T_ACC
                and not (self.ULTIMATUM and getattr(self, "_first_mover", False))):
            return True
        return self._norm(u) >= target - 1e-9

    # ----------------------------------------------------- opponent modelling
    def _update_opponent_model(self, offer: Outcome, state: SAOState) -> None:
        u = float(self.ufun(offer))
        if u > self.best_opp_offer_util:
            self.best_opp_offer_util = u
            self.best_opp_offer = offer
        # concession-prediction data (normalized utility + time)
        self._villain_ut.append((self._norm(u), max(state.relative_time, 1e-3)))
        self._unique_villain.add(offer)
        self._opp_offers.append(offer)
        self._n_opp_offers += 1
        self._rebuild_model()
        # widen the their-utility range with their own (high-for-them) offers
        ou = float(self._opp_model(offer))
        self._opp_lo = min(self._opp_lo, ou)
        self._opp_hi = max(self._opp_hi, ou)

    def on_negotiation_end(self, state) -> None:
        """If the opponent ACCEPTED early, that agreement is a ground-truth point
        they value highly (their acceptance bar is still high early). Fold it into
        the model with a weight that grows the earlier they accepted, then rebuild
        — this sharpens the model right before it is scored. A late acceptance
        (they conceded to us) carries ~no weight, so it can't mislead us."""
        agr = getattr(state, "agreement", None)
        # Only when THEY accepted OUR offer: an early accept ⇒ high-for-them. If we
        # accepted their offer, it's already in their offer history (correctly
        # weighted by when they proposed it), so boosting would double-count/mislead.
        if (not self.EARLY_ACCEPT_BOOST or agr is None or self.ufun is None
                or self._n_opp_offers == 0 or agr not in self._our_offered):
            return
        t = getattr(state, "relative_time", self._last_t) or self._last_t
        earliness = max(0.0, 1.0 - t)
        if earliness > 0.05:
            # scale relative to the total recency mass (~1/(1-decay)) so an early
            # acceptance is a strong but not overwhelming data point.
            mass = 1.0 / (1.0 - self.MODEL_DECAY)
            self._bonus.append((agr, 1.5 * earliness * mass))
            self._rebuild_model()

    def _model_counts_uniq(self, offers):
        # Decoy resistance: each DISTINCT outcome counts once, weighted softly
        # by first-appearance order. A decoy repeated 30 times early carries the
        # same weight as any genuine offer, so it cannot dominate the model.
        wcounts = [defaultdict(float) for _ in range(self.n_issues)]
        first_rank = {}
        for o in offers:
            if o not in first_rank:
                first_rank[o] = len(first_rank)
        for o, rank in first_rank.items():
            w = self.RANK_DECAY ** rank
            for i, v in enumerate(o):
                wcounts[i][v] += w
        return wcounts

    def _model_counts_freq(self, offers):
        wcounts = [defaultdict(float) for _ in range(self.n_issues)]
        for idx, o in enumerate(offers):
            w = self.MODEL_DECAY ** idx
            for i, v in enumerate(o):
                wcounts[i][v] += w
        return wcounts

    # -------------------------------------------------- RankFit opponent model
    RANKFIT = True                   # ridge regression on one-hot issue-value features:
                                     # offered outcomes get targets descending by first-
                                     # appearance rank, unoffered sample gets a weak low
                                     # anchor, solved JOINTLY (attributes credit across
                                     # co-occurring values, unlike counting) + a second
                                     # pass dropping offered rows the fit calls junk
                                     # (decoys violate concession monotonicity). Offline
                                     # (real-scenario sequences): parity vs freq on honest
                                     # (0.852/0.858), +0.06 decoy-led, +0.05 coherent.
    RF_W_OFF = 4.0                   # weight per offered row
    RF_W_UN = 2.0                    # weight per unseen anchor row
    RF_Y_UN = 0.1                    # unseen anchor target
    RF_LAM = 0.05                    # ridge strength
    RF_MAX_UNSEEN = 600              # unseen anchor sample size

    def _rankfit_tables(self):
        """Per-issue value-score tables fit from the dedup'd offer sequence."""
        # dedup with first-offer TIME: targets follow their declining aspiration
        # over time, not sequence rank — robust to band-pickers whose offers
        # within one aspiration band are unordered in their own utility
        dedup, times, seen = [], [], set()
        for idx, o in enumerate(self._opp_offers):
            if o not in seen:
                seen.add(o)
                dedup.append(o)
                times.append(self._villain_ut[idx][1] if idx < len(self._villain_ut)
                             else 1.0)
        if len(dedup) < 2:
            return None
        if not hasattr(self, "_rf_enc"):
            vidx, offs, off = [], [], 0
            for issue in self.issues:
                vidx.append({v: j for j, v in enumerate(issue.values)})
                offs.append(off)
                off += len(issue.values)
            self._rf_vidx, self._rf_offs, self._rf_P = vidx, offs, off
            pool = [o for o, _, _ in self.sorted_outcomes]
            self._rf_unseen_pool = (random.sample(pool, self.RF_MAX_UNSEEN)
                                    if len(pool) > self.RF_MAX_UNSEEN else list(pool))
            self._rf_enc = lambda o: [self._rf_offs[i] + self._rf_vidx[i][v]
                                      for i, v in enumerate(o)]
        unseen = [o for o in self._rf_unseen_pool if o not in seen]
        n = len(dedup) + len(unseen)
        A = np.zeros((n, self._rf_P))
        y = np.empty(n)
        w = np.empty(n)
        for k, o in enumerate(dedup):
            A[k, self._rf_enc(o)] = 1.0
            y[k] = 1.0 - 0.5 * times[k]
            w[k] = self.RF_W_OFF
        for j, o in enumerate(unseen):
            r = len(dedup) + j
            A[r, self._rf_enc(o)] = 1.0
            y[r] = self.RF_Y_UN
            w[r] = self.RF_W_UN

        def solve(wv):
            sw = np.sqrt(wv)
            Aw = A * sw[:, None]
            return np.linalg.lstsq(Aw.T @ Aw + self.RF_LAM * np.eye(self._rf_P),
                                   Aw.T @ (y * sw), rcond=None)[0]

        try:
            theta = solve(w)
            self._rf_deceiver = False
            if len(dedup) > 4:
                resid = y[: len(dedup)] - A[: len(dedup)] @ theta
                cut = np.percentile(resid, 85)
                keep = resid <= max(cut, 0.35)
                # decoy evidence: offered rows the fit flatly contradicts
                self._rf_deceiver = bool((resid > 0.35).any())
                if keep.sum() >= 3 and not keep.all():
                    w2 = w.copy()
                    w2[: len(dedup)] = np.where(keep, self.RF_W_OFF, 0.2)
                    theta = solve(w2)
        except Exception:
            return None
        return [{v: float(theta[self._rf_offs[i] + self._rf_vidx[i][v]])
                 for v in issue.values} for i, issue in enumerate(self.issues)]

    def _rebuild_model(self) -> None:
        """Two models, decoupled: the battle-tested FREQUENCY model drives every
        internal decision (bids, decoys, endgame — advantage behavior identical
        to Phantom6), while the EXPOSED private_info["opponent_ufun"] — the only
        thing the grader Kendall-scores — is the RankFit estimate, which matches
        frequency on honest opponents and beats it against deceivers."""
        self._rebuild_model_freq()
        self._internal_opp = self.private_info["opponent_ufun"]
        if not self.RANKFIT:
            return
        tabs = self._rankfit_tables()
        if tabs is None:
            return  # exposed = freq model until rankfit has data
        alpha = 1.0 - self.PRIOR_EPS
        # shift each issue to 0 (constants don't affect ranking) but scale
        # GLOBALLY — per-issue theta spread IS the learned issue weight
        los = [min(t.values()) for t in tabs]
        gmax = max(max(t.values()) - lo for t, lo in zip(tabs, los)) or 1.0
        vals = []
        for i, t in enumerate(tabs):
            vals.append(TableFun(
                {v: alpha * (t[v] - los[i]) / gmax
                    + (1 - alpha) * self._anti_w[i] * self._anti_vals[i][v]
                 for v in t}))
        self.private_info["opponent_ufun"] = LinearAdditiveUtilityFunction(
            values=vals, outcome_space=self.nmi.outcome_space)

    def _rebuild_model_freq(self) -> None:
        offers = self._opp_offers
        if self.MODEL_MODE == "uniq":
            wcounts = self._model_counts_uniq(offers)
        elif self.MODEL_MODE == "mix":
            # Honest agents repeat their genuinely-best offers (repetition IS
            # signal); deceivers repeat decoys. Blend the per-issue-normalized
            # frequency and unique tables to hedge both regimes.
            wu = self._model_counts_uniq(offers)
            wf = self._model_counts_freq(offers)
            wcounts = []
            for i in range(self.n_issues):
                tu, tf = sum(wu[i].values()) or 1.0, sum(wf[i].values()) or 1.0
                d = defaultdict(float)
                for v, w in wu[i].items():
                    d[v] += self.MIX * w / tu
                for v, w in wf[i].items():
                    d[v] += (1.0 - self.MIX) * w / tf
                wcounts.append(d)
        else:
            wcounts = self._model_counts_freq(offers)
        # Ground-truth bonus points (e.g. an early acceptance).
        for o, bw in self._bonus:
            for i, v in enumerate(o):
                wcounts[i][v] += bw

        n_distinct = len(set(offers))
        alpha = n_distinct / (n_distinct + self.PRIOR_K) if n_distinct else 0.0
        alpha *= 1.0 - self.PRIOR_EPS
        if alpha <= 0.0:
            self._set_model_prior()
            return

        value_funcs, weights = [], []
        for i, issue in enumerate(self.issues):
            counts = wcounts[i]
            total = sum(counts.values())
            if total <= 0:
                value_funcs.append({v: 0.5 for v in issue.values})
                weights.append(1.0)
                continue
            probs = {v: counts.get(v, 0.0) / total for v in issue.values}
            scores = {v: (0.5 + 0.5 * probs[v]) if counts.get(v, 0) else 0.0
                      for v in issue.values}
            n = len(issue.values)
            if n > 1:
                herf = sum(p * p for p in probs.values())
                weights.append(0.5 + 0.5 * max(0.0, (herf - 1.0 / n) / (1.0 - 1.0 / n)))
            else:
                weights.append(1.0)
            value_funcs.append(scores)
        wsum = sum(weights) or 1.0
        weights = [w_ / wsum for w_ in weights]
        # Blend the data model with the structural anti-prior: dominates at low
        # data, fades to PRIOR_EPS as evidence accumulates, and guarantees the
        # exposed model is never an all-ties (kopt-0) function.
        self.private_info["opponent_ufun"] = LinearAdditiveUtilityFunction(
            values=[TableFun({v: alpha * weights[i] * s
                              + (1.0 - alpha) * self._anti_w[i] * self._anti_vals[i][v]
                              for v, s in vf.items()})
                    for i, vf in enumerate(value_funcs)],
            outcome_space=self.nmi.outcome_space)

    # ----------------------------------------------------------- bid selection
    def _ultimatum_offer(self) -> "Outcome | None":
        """Max-for-us outcome clearing the opponent's estimated acceptance bar
        (capped at 0.5 — most final responders accept anything above reserve)
        AND their reservation value. Their rv is unknown, but every offer THEY
        made clears it by definition, so matching their lowest self-valued own
        offer is provably rv-safe (within model error). Ranges computed FRESH
        under the current model — values accumulated across rebuilds are stale."""
        opp = self._opp_model
        mv = [float(opp(o)) for o in self._nash_sample[:500]]
        lo, hi = min(mv), max(mv)
        rng = (hi - lo) if hi > lo else 0.0
        bar, rv_floor = 0.0, self.ULT_RV_FLOOR
        if self._opp_offers and rng > 0:
            selfvals = [(float(opp(o)) - lo) / rng for o in self._opp_offers]
            bar = selfvals[-1]
            rv_floor = max(rv_floor, min(selfvals))
        cap = 0.5
        if (self.ULT_BARKEEPER
                and len(self._unique_villain) >= self.ULT_BK_MIN_UNIQ
                and len(self._unique_villain)
                >= self.ULT_BK_RATIO * max(len(self._our_offered), 1)):
            cap = 1.0  # bar-keeper: must clear their REAL threshold
        floor_v = max(min(bar - self.ULT_BAR_MARGIN, cap), rv_floor)
        for o, _, n in self.sorted_outcomes:
            ov = ((float(opp(o)) - lo) / rng) if rng > 0 else 1.0
            if ov >= floor_v:
                return o
        return None

    def _choose_bid(self, state: SAOState, target: float) -> Outcome:
        t = state.relative_time
        ult_power = (self.ULTIMATUM and self.total_steps
                     and getattr(self, "_first_mover", False))
        # Final-proposal ultimatum: as FIRST mover our last offer is take-it-or-
        # leave-it (their last counter dies unseen). Offer max-for-us subject to
        # clearing their estimated acceptance bar (capped: most final responders
        # accept anything above reservation).
        if ult_power and state.step >= self.total_steps - 1:
            cand = self._ultimatum_offer()
            if cand is not None:
                self._record(cand)
                return cand
            # fallback MUST clear our reservation — re-proposing their best
            # offer unguarded closed below-rv deals on rational_fraction<1
            # scenarios (worse than no deal; invisible when rv=0)
            if (self.best_opp_offer is not None
                    and self.best_opp_offer_util > self.reservation_value):
                self._record(self.best_opp_offer)
                return self.best_opp_offer

        if (t >= self.DECEPT_END and self.best_opp_offer is not None
                and self.best_opp_offer_util > self.reservation_value):
            if self._norm(self.best_opp_offer_util) >= target - 0.02:
                self._record(self.best_opp_offer)
                return self.best_opp_offer

        # Last-call closing: with at most 2 of our proposals left and the target
        # still above anything they ever offered, re-propose THEIR best offer —
        # a near-sure accept — instead of timing out at advantage 0. As first
        # mover we hold the final proposal instead (ultimatum dominates).
        if (self.LAST_CALL and self.total_steps and not ult_power
                and state.step >= self.total_steps - 2
                and self.best_opp_offer is not None
                and self.best_opp_offer_util > self.reservation_value):
            self._record(self.best_opp_offer)
            return self.best_opp_offer

        agreeable = (self._n_opp_offers >= 3
                     and self._norm(self.best_opp_offer_util) >= self.AGREEABLE_TH)

        # Phantom decoy: while the opponent is still stingy, repeatedly offer ONE fixed
        # lose-lose outcome — low for us (misrepresents our preferences) and worst for
        # them (so they reject -> no advantage cost). Consistency makes their frequency
        # model converge to this false profile, lowering their accuracy of us (a_them).
        decoy_end = self.DECOY_END
        if (self.ADAPT_DECOY and t >= self.ADAPT_T and self._n_opp_offers >= 5
                and self._norm(self.best_opp_offer_util) < self.ADAPT_OPP_TH):
            decoy_end = self.ADAPT_DECOY_END
        # Stonewalled-regime interleave: even steps break out of the decoy to show
        # a genuine (fresh) band offer — TFT/MiCRO need visible movement — while
        # odd steps keep hammering the decoy so their frequency model stays fooled.
        interleave_out = self._recip_on and state.step % 2 == 0
        if (self.PHANTOM and t < decoy_end and not agreeable
                and self._n_opp_offers >= 2 and not interleave_out):
            opp = self._opp_model

            def decoy_safe(o) -> bool:
                """A decoy is only safe while it sits clearly BELOW the opponent's
                current acceptance bar (else they may accept the lose-lose junk).
                Bar = their-value of their own latest offer, normalized over the
                CURRENT model's value range (rankfit rescales every refit, so
                ranges accumulated across refits are invalid)."""
                if not self.DECOY_SAFE:
                    return True
                if t < self.SAFE_FROM_T:
                    return True
                lo, hi = getattr(self, "_model_rng", (self._opp_lo, self._opp_hi))
                if hi <= lo:
                    return True
                rng = hi - lo
                bar = (float(opp(self._opp_offers[-1])) - lo) / rng
                val = (float(opp(o)) - lo) / rng
                return val < bar - self.SAFE_MARGIN
            # Adaptive concealment: a TOUGH opponent (gives us little -> high self-threshold)
            # REJECTS our junk, so present a VARIED coherent fake-profile that fools even a
            # decoy-resistant modeler. A conceder might ACCEPT a varied decoy -> use the
            # single reliably-rejected decoy (zero advantage cost).
            tough = (self.COHERENT
                     and self._norm(self.best_opp_offer_util) < self.TOUGH_DECOY_TH)
            if tough:
                pool = self._decoy_pool
                K = max(8, len(pool) // 4)
                lose_lose = sorted(pool, key=lambda o: float(opp(o)))[:K]
                lose_lose.sort(key=lambda o: self._norm(float(self.ufun(o))))  # fake-ideal first
                frac = min(1.0, t / max(self.DECOY_END, 1e-6))
                bid = lose_lose[int(frac * (len(lose_lose) - 1))]
                if decoy_safe(bid):
                    self._record(bid)
                    return bid
            else:
                if self._decoy is None:
                    cut = int(len(self.sorted_outcomes) * self.DECOY_BOTTOM)
                    bottom = self.sorted_outcomes[cut:] or self.sorted_outcomes
                    # avoid THEIR decoy region: prefer outcomes sharing no values
                    # with anything they offered (a model fooled by their junk
                    # would otherwise pick a decoy they pretend to love)
                    their_vals = [set() for _ in range(self.n_issues)]
                    for o in self._opp_offers:
                        for i, v in enumerate(o):
                            their_vals[i].add(v)
                    overlap = lambda o: sum(1 for i, v in enumerate(o)
                                            if v in their_vals[i])
                    ranked = sorted(bottom,
                                    key=lambda x: (overlap(x[0]), float(opp(x[0]))))
                    self._decoy = ranked[0][0]
                    # shaped multi-decoy: 2 more junk outcomes, value-disjoint
                    # from the first where possible, ordered by OUR utility
                    # ascending (worst-for-us = most-repeated = fake favorite)
                    seq = [self._decoy]
                    if self.DECOY_SHAPED:
                        # B/C must stay LOSE-LOSE: cap their-value at the 20th
                        # percentile of the bottom band (unconstrained disjoint
                        # picks drift up in their-utility and get ACCEPTED —
                        # measured adv -0.05)
                        tv = sorted(float(opp(x[0])) for x in bottom)
                        cap = tv[max(0, len(tv) // 5 - 1)]
                        for o, _, _ in ranked[1:]:
                            if len(seq) >= self.SHAPED_K:
                                break
                            if float(opp(o)) > cap:
                                continue
                            if all(sum(1 for i in range(self.n_issues)
                                       if o[i] == p[i]) <= self.n_issues // 3
                                   for p in seq):
                                seq.append(o)
                        seq.sort(key=lambda o: float(self.ufun(o)))
                    # frequency shape: A,A,A,B,B,C — inverted 3-level ordering
                    self._decoy_cycle = ([seq[0]] * 3 + [seq[1]] * 2 + [seq[2]]
                                         if len(seq) >= 3 else
                                         [seq[0]] * 3 + [seq[-1]] * 2
                                         if len(seq) == 2 else seq)
                    self._decoy_i = 0
                bid = self._decoy_cycle[self._decoy_i % len(self._decoy_cycle)]
                if not decoy_safe(bid):
                    bid = self._decoy  # fall back to the safest (min-their-value)
                if decoy_safe(bid):
                    self._decoy_i += 1
                    self._record(bid)
                    return bid
            # unsafe decoy (their bar is low enough to accept junk) -> genuine bid

        if self.DECEPTION_MODE != "off" and t < self.DECEPT_END and not agreeable:
            o = self._deceptive_offer()
            if o is not None:
                self._record(o)
                return o

        if t >= self.ENDGAME_T:
            o = self._endgame_offer()
            if o is not None:
                self._record(o)
                return o

        band_lo, band_hi = target - 1e-9, target + 0.12
        candidates = [o for o, _, n in self.sorted_outcomes if band_lo <= n <= band_hi]
        if not candidates:
            candidates = [o for o, _, n in self.sorted_outcomes if n >= band_lo]
        if not candidates:
            candidates = [self.sorted_outcomes[0][0]]
        opp = self._opp_model
        if (self.GREEDY_BAND and t < self.SQUEEZE_T
                and self._norm(self.best_opp_offer_util) >= self.SQUEEZE_OPP):
            chosen = max(candidates, key=lambda o: float(self.ufun(o)))
        elif self._recip_on:
            # Stonewalled regime: prefer a FRESH outcome from the band — MiCRO-like
            # opponents concede one step per unique outcome we show, and TFT-likes
            # need visible movement; repeating one pick stalls both.
            fresh = [o for o in candidates if o not in self._our_offered]
            chosen = max(fresh or candidates, key=lambda o: float(opp(o)))
        else:
            chosen = max(candidates, key=lambda o: float(opp(o)))
        self._record(chosen)
        return chosen

    def _endgame_offer(self) -> "Outcome | None":
        opp = self._opp_model
        ovals = [(o, float(opp(o))) for o, _, _ in self.sorted_outcomes]
        omin = min(v for _, v in ovals)
        omax = max(v for _, v in ovals)
        rng = (omax - omin) or 1.0
        for o, v in ovals:
            if (v - omin) / rng >= self.ENDGAME_OPP_FLOOR:
                return o
        return self.sorted_outcomes[0][0]

    def _deceptive_offer(self) -> "Outcome | None":
        opp = self._opp_model
        scored = sorted(self._decep_pool, key=lambda on: float(opp(on[0])))
        keep = scored[: max(1, int(len(scored) * 0.4))]

        def key(on):
            o, n = on
            rep = sum(self._our_counts[i].get(v, 0) for i, v in enumerate(o))
            if self.DECEPTION_MODE == "aggressive":
                return rep + 3.0 * n
            return rep

        return min(keep, key=key)[0]

    def _record(self, o: Outcome) -> None:
        self._our_offered.add(o)
        for i, v in enumerate(o):
            self._our_counts[i][v] += 1
