import random
from bisect import bisect_right

from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import TableFun


class RAINeg(SAOCallNegotiator):
    """
    Your negotiator code. This is the ONLY class you need to implement.
    """
    rational_outcomes = tuple()
    PROFILE_SAMPLE = 150 # check stride of 150 outcomes
    E_CONCESSION = 0.18
    PROFILE_EVERY = 4 # identify profile every 4 rounds
    ENDGAME = 0.95
    WINDOW_FRAC = 0.15
    _TOP_K_BLUFFING = 1 # randomly choose one decepting offers out of top 5
    _P_PRESSURE =  2 # convex: (late-heavy) recency weight
    _ESP_PRIOR = 0.05
    _LAMBDA_SURVIVAL = 0.3
    _DIVERGENCE = 0.3
    _DELTA_LABEL = 0.15

    """ Auxiliary functions """
    def _ufun_tilde(self, outcome:Outcome)->float:
        return sum([self._wtilde[i]*self._eval_fun[i](outcome[i]) for i in range(self._n_issues)])

    def _build_uniform_opponent(self) -> LinearAdditiveUtilityFunction:
        values = [
            TableFun({v:1.0 for v in self._issues_values[i]}) for i in range(self._n_issues)
        ]
        return LinearAdditiveUtilityFunction(
            values=values,
            weights=[1.0/self._n_issues] * self._n_issues,
            outcome_space=self.nmi.outcome_space
        )

    def _aspiration(self, relative_time:float) -> float:
        # Time-dependent Boulware aspiration
        relative_time = min(max(relative_time,0.0),1.0)
        return self._r + self._span * (1.0-relative_time)**self.E_CONCESSION

    def _acceptable_indices(self, theta: float) -> range:
        count = max(bisect_right(self._utilde, -theta),1)
        return range(count)

    @staticmethod
    def _rank_divergence(vec_a:list[float|int], vec_b:list[float|int]) -> float:
        try:
            from scipy.stats import kendalltau
            Ktau = kendalltau(vec_a, vec_b).statistic
        except Exception:
            Ktau = None
        if Ktau is None or Ktau != Ktau:
            return 1.0
        return 0.5 * (1.0 - float(Ktau))
    ###############################################################################################
    def on_preferences_changed(self, changes):
        """
        Called when preferences change. In ANL 2026, this is equivalent with initializing the agent.

        Remarks:
            - Can optionally be used for initializing your agent.
            - We use it to save a list of all rational outcomes.

        """

        # If there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return

        self._issues = list(self.nmi.outcome_space.issues)
        self._n_issues = len(self._issues)
        # List of values per issue
        self._issues_values = [list(issue.all) for issue in self._issues]

        self._true_issue_weights = list(getattr(self.ufun, "weight", [1.0]*self._n_issues))
        self._w = [w/sum(self._true_issue_weights) for w in self._true_issue_weights]
        self._eval_fun = list(getattr(self.ufun, "values", []))

        self._r = float(self.ufun.reserved_value)
        self._umax = float(self.ufun.max())

        self._span = self._umax - self._r

        # Descepting polict
        self._beta = max(0.0, min(1.1, 1.0-self._r)) # Regulates deceiving policy: prioritise good deals to deception when reservation value is high

        self._scored_outcomes = sorted(
            ((float(self.ufun(_)), _) for _ in self.nmi.outcome_space.enumerate_or_sample() if self.ufun(_) > self._r),
            key=lambda t:t[0], reverse=True
        )
        self._best_outcome = self._scored_outcomes[0][1] if self._scored_outcomes else None

        n =len(self._scored_outcomes)

        inv_weight = [1.0/max(w,1e-6) for w in self._w]
        self._wtilde = [w/sum(inv_weight) for w in inv_weight]

        # self.reordering =[self._scored_outcomes[n-i] if i%2==0 else self._scored_outcomes[i] for i in range(n)]

        # self._utilde = [self.ufun(o) for _,o in self.reordering]
        self._utilde = [self._ufun_tilde(o) for _,o in self._scored_outcomes]

        if n<=self.PROFILE_SAMPLE:
            self._sample = [o for _,o in self._scored_outcomes]
        else:
            stride = n/self.PROFILE_SAMPLE
            self._sample = [self._scored_outcomes[int(i*stride)][1] for i in range(self.PROFILE_SAMPLE)]

        # Opponent modeling
        self._opp_history : list[tuple[float,Outcome]] = [] # (relative_time, offer)
        self._b_star = 1 # trusted boundary
        self._profile = 0 # 'unknown profile' --> 1: early/stationary, 2: late pivot, 3: mid pivot
        self._max_surprise = 0.0


        self._cached_step = -1
        self._cached_proposal: Outcome | None = None
        print(f"N. rational: {len(self._scored_outcomes)}")

        self.private_info["opponent_ufun"] = self._build_uniform_opponent()

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        offer = state.current_offer

        # If there are no outcomes (should in theory never happen)
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        # If there is no offer yet (first call), make a counter offer
        if offer is None:
            return SAOResponse(
                ResponseType.REJECT_OFFER, self.concealing_bidding_strategy(state)
            )

        self.update_opponent_model(state)
        response, res_offer = self.acceptance_strategy(state)
        return SAOResponse(response, res_offer)

    def acceptance_strategy(self, state: SAOState):
        assert self.ufun
        offer = state.current_offer
        alternative_offer = self.concealing_bidding_strategy(state)

        # Cannot accept a non-existent offer
        if offer is None:
            return ResponseType.REJECT_OFFER, alternative_offer

        u_offer = float(self.ufun(offer))
        tau = state.relative_time
        acceptance_threshold = max(self._r, self._aspiration(tau))
        u_alt_offer = float(self.ufun(alternative_offer)) if alternative_offer is not None else self._r

        if u_offer > max(acceptance_threshold, u_alt_offer):
            return ResponseType.ACCEPT_OFFER, offer

        if tau>= self.ENDGAME and u_offer >= self._r:
            return ResponseType.ACCEPT_OFFER, offer

        return ResponseType.REJECT_OFFER, alternative_offer



    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        if state.step == self._cached_step and self._cached_proposal is not None:
            return self._cached_proposal

        if not self._scored_outcomes:
            return None

        tau = min(max(state.relative_time, 0.0), 1.0)
        theta = self._aspiration(tau)
        indices = self._acceptable_indices(theta)
        acceptable_outcomes = [self._scored_outcomes[idx] for idx in indices]
        deception_strength = (1-tau) * self._beta

        decepting_outcomes = sorted(
            ((deception_strength*self._ufun_tilde(o) + (1-deception_strength) * u, o) for u,o in acceptable_outcomes),
            key=lambda t: t[0], reverse=True
        )

        idx = random.choice(range(min(len(decepting_outcomes),self._TOP_K_BLUFFING)))
        counter_offer = decepting_outcomes[idx][1]

        self._cached_step = state.step
        self._cached_proposal = counter_offer
        return counter_offer


    def update_opponent_model(self, state: SAOState) -> None:
        assert self.ufun and self.opponent_ufun
        offer = state.current_offer
        if offer is None:
            return

        self._opp_history.append((state.relative_time, offer))

        if len(self._opp_history) % self.PROFILE_EVERY == 0:
            self._identify_opponent_profile()

        weights, value_tables = self._fit_model(
            self._opp_history, b_start=self._b_star
        )
        self.private_info["opponent_ufun"] = LinearAdditiveUtilityFunction(
            values = [TableFun(t) for t in value_tables],
            weights = weights,
            outcomes = self.nmi.outcome_space
        )

    ###############################################################################################
    # Opponent Modeling Auxiliary functions                                                       #
    ###############################################################################################
    def _fit_model(self,
                   history: list[tuple[float, Outcome]],
                   b_start: int = 1,
                   adaptive: bool = True
    ) -> tuple[list[float], list[dict]]:
        n = self._n_issues
        freq = [dict() for _ in range(n)]
        stability = [0.0 for _ in range(n)]
        last_change_tau = [0.0 for _ in range(n)]
        seen = [set() for _ in range(n)]

        prev_outcome = None
        for idx, (tau, offer) in enumerate(history):
            gamma = tau ** self._P_PRESSURE
            if adaptive:
                gamma *= self._ESP_PRIOR # discard pre-pivot bluff window
            gamma = max(gamma,1e-10)

            for i in range(n):
                val = offer[i]
                freq[i][val] = freq[i].get(val, 0.0) + gamma
                seen[i].add(val)

            if prev_outcome is not None:
                for i in range(n):
                    if offer[i] == prev_outcome[i]:
                        stability[i] += gamma
                    else:
                        last_change_tau[i] = tau
            prev_outcome = offer

        value_tables : list[dict] = []
        for i in range(n):
            fi = freq[i]
            max_f = max(fi.values()) if fi else 0.0
            diversity = max(1,len(seen[i]))
            kappa = 1.0 / (1.0 + diversity)
            n_val = max(1.0, len(self._issues_values[i]))
            uniform_weight = 1.0 / n_val
            table = {}

            for val in self._issues_values[i]:
                e = fi.get(val,0.0) / max_f if max_f>0 else uniform_weight
                table[val] = (1.0-kappa)*e + kappa*uniform_weight
            value_tables.append(table)

        normalised_stability = [x/sum(stability) if sum(stability)>0 else 0.0 for x in stability]
        issue_weights = [
            (1.0-self._LAMBDA_SURVIVAL)*normalised_stability[i] +
            self._LAMBDA_SURVIVAL*last_change_tau[i] for i in range(n)
        ]
        s = sum(issue_weights)
        if s >0:
            issue_weights = [x/s for x in issue_weights]
        else:
            issue_weights = [1/n for _ in issue_weights] # uniform weights until seeing stability

        return issue_weights, value_tables

    def _model_vector(self, weights:list[float], value_tables:list[dict]) -> list[float]:
        out = []
        for o in self._sample:
            total = sum([
                weights[i]*value_tables[i].get(o[i], 0.0) for i in range(self._n_issues)
            ])
            out.append(total)
        return out

    def _identify_opponent_profile(self) -> None:
        T = len(self._opp_history)
        W = max(2, int(self.WINDOW_FRAC * T) +1)

        if T<2*W:
            self._b_star = 1 # too little data --> trust everything
            return

        early = self._fit_model(self._opp_history[:T-W], adaptive=False)
        recent =self._fit_model(self._opp_history[T-W:], adaptive=False)
        sigma = self._rank_divergence(
            self._model_vector(*early), self._model_vector(*recent)
        )

        self._max_surprise = max(self._max_surprise, sigma)

        end_vec = self._model_vector(
            *self._fit_model(self._opp_history[T-W:], adaptive=False)
        )
        b_star = 1
        n_candidates = min(T, 12)
        step = max(1, T//n_candidates)
        for b in range(1,T+1,step):
            seg_vec = self._model_vector(
                *self._fit_model(self._opp_history[b-1:], adaptive=False)
            )
            if self._rank_divergence(seg_vec, end_vec) < self._DIVERGENCE:
                b_star = b
                break
            else:
                b_star = max(1,T-W+1)

        self._b_star = b_star
        phi = (T-b_star)/ T

        if self._max_surprise< self._DIVERGENCE or phi>= 1.0-self._DELTA_LABEL:
            self._profile = 1
        elif phi<=self._DELTA_LABEL:
            self._profile = 2
        else:
            self._profile = 3

