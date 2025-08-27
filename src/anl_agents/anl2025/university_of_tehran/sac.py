# from anl2025.negotiator import Shochan2025, AgentRenting2025
from anl2025 import ANL2025Negotiator
from negmas import Outcome, ResponseType, SAOState
from negmas import PresortingInverseUtilityFunction
from anl2025.ufun import CenterUFun

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import os

device = torch.device("cpu")

__all__ = ["SacAgent"]


class Policy(nn.Module):
    def __init__(self, observation_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(observation_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_size)
        self.logvar = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        # logvar = torch.tanh(self.logvar(x))
        # logvar = -5 + 3.5 * (logvar + 1) # Bounded logvar
        logvar = torch.clamp(self.logvar(x), -5, 2)  # Bounded logvar
        return self.mean(x), logvar.exp()


observation_size = 4
action_size = 2
policy_net = Policy(observation_size, action_size).to(device)

# policy_net.load_state_dict(torch.load('saves2/p_model_2600.pt'))
policy_net.load_state_dict(
    torch.load(
        f"{os.path.dirname(os.path.realpath(__file__))}/saves/p_model_110.pt",
        map_location=device,
    )
)


class SacAgent(ANL2025Negotiator):
    def _vectorize(self, steps):
        data = [
            self._num_negotiations / len(self.negotiators),
            len(self.negotiators),
            steps,
            isinstance(self.ufun, CenterUFun),
        ]
        tensor = torch.tensor(data, device=device, dtype=torch.float32)
        return tensor

    def _sample_action(self, state):
        with torch.no_grad():
            mu, sigma = policy_net(state)
            # dist = Normal(mu, sigma)
            # action = dist.sample()
            action = mu
        return action

    def _ufun(self, offer, negotiator_id, normalized=False):
        _, cntxt = self.negotiators[negotiator_id]
        ufun: SideUFun = cntxt["ufun"]
        if normalized:
            return float(ufun.eval_normalized(offer))
        else:
            return float(ufun.eval(offer))

    def _create_inverter(self, negotiator_id):
        if negotiator_id not in self._inverters:
            _, cntxt = self.negotiators[negotiator_id]
            ufun = cntxt["ufun"]
            inverter = PresortingInverseUtilityFunction(ufun, rational_only=True)
            inverter.init()
            self._inverters[negotiator_id] = inverter
            self._best = inverter.best()
            self._mx, self._mn = inverter.max(), max(inverter.min(), ufun(None))
            if self._mx == 0:
                self._mx = 1.0

    def _proposal_in_range(self, u_min, u_max, negotiator_id):
        self._create_inverter(negotiator_id)
        return self._inverters[negotiator_id].one_in((u_min, u_max), normalized=True)

    def init(self):
        self._replay = []
        self._num_negotiations = 0
        self._prev_util = 0

    def thread_init(self, negotiator_id: str, state: SAOState) -> None:
        self._inverters = dict()
        self._gstate = self._vectorize(1 / state.relative_time)
        self._action = self._sample_action(self._gstate.unsqueeze(0))[0]
        _action1, _action2 = self._action.cpu().tolist()
        if _action2 < 0:
            _action2 = 0
        self._exp_min = math.e ** (_action1 - _action2)
        self._exp_max = math.e**_action1

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        u_min = 1 - state.relative_time**self._exp_min
        u_max = 1 - state.relative_time**self._exp_max
        proposal = self._proposal_in_range(u_min, u_max, negotiator_id)
        if proposal is None:
            return self._best
        return proposal

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        self._create_inverter(negotiator_id)
        util_none = self._ufun(None, negotiator_id)
        if self._mx < util_none:
            return ResponseType.END_NEGOTIATION

        u_reject = 1 - state.relative_time**self._exp_min
        if (
            self._ufun(state.current_offer, negotiator_id) / self._mx < u_reject
            or self._ufun(state.current_offer, negotiator_id) < self.ufun(None)
            or self._ufun(state.current_offer, negotiator_id) < self._prev_util
        ):
            return ResponseType.REJECT_OFFER
        return ResponseType.ACCEPT_OFFER

    def thread_finalize(self, negotiator_id: str, state: SAOState) -> None:
        pass
