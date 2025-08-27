from typing import Literal
from random import random, choice
from anl2025.ufun import SideUFun
from negmas import SAONMI, InverseUFun, PolyAspiration, PresortingInverseUtilityFunction
import numpy as np

from negmas.sao.controllers import SAOState
from negmas import (
    ResponseType,
)
from negmas.outcomes import Outcome
from anl2025.negotiator import ANL2025Negotiator
from .NN_model import MyLSTMpro
import torch
import torch.nn as nn
import pathlib

__all__ = ["UfunATAgent"]


class UfunATAgent(ANL2025Negotiator):
    """
    A time-based conceding agent
    """

    def __init__(
        self,
        *args,
        aspiration_type: Literal["boulware"]
        | Literal["conceder"]
        | Literal["linear"]
        | Literal["hardheaded"]
        | float = "boulware",
        deltas: tuple[float, ...] = (1e-3, 1e-1, 2e-1, 4e-1, 8e-1, 1.0),
        reject_exactly_as_reserved: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._curve = PolyAspiration(1.0, aspiration_type)
        self._inverter: dict[str, InverseUFun] = dict()
        self._best: list[Outcome] = None  # type: ignore
        self._mx: float = 1.0
        self._mn: float = 0.0
        self.alpha: float = 0.05
        self._deltas = deltas
        self._best_margin = 1e-8
        self._prediction: dict[str, SideUFun] = dict()
        self.reject_exactly_as_reserved = reject_exactly_as_reserved
        self.model = MyLSTMpro()
        self.path_2_myfile = pathlib.Path(__file__).parent / "agentdata" / "model.pth"
        self.model.load_state_dict(torch.load(self.path_2_myfile))
        self.train = False
        self.hidden = None
        self.notrange: bool = False
        self.compromise: bool = False
        self.loss1 = 0
        self.out = None
        self.outcome = None
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.fiveUfun = torch.zeros((1, 10, 4))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def ensure_inverter(self, negotiator_id) -> InverseUFun:
        """Ensures that utility inverter is available"""
        if self._inverter.get(negotiator_id, None) is None:
            _, cntxt = self.negotiators[negotiator_id]
            ufun = cntxt["ufun"]
            inverter = PresortingInverseUtilityFunction(ufun, rational_only=True)
            inverter.init()
            # breakpoint()
            self._mx, self._mn = inverter.max(), inverter.min()
            self._mn = max(self._mn, ufun(None))
            self._best = inverter.some(
                (
                    max(0.0, self._mn, ufun(None), self._mx - self._best_margin),
                    self._mx,
                ),
                normalized=True,
            )
            if not self._best:
                self._best = [inverter.best()]  # type: ignore
            self._inverter[negotiator_id] = inverter

        return self._inverter[negotiator_id]

    def make_trainingPro(self, nmi: SAONMI, state: SAOState, ufun, flag, cent):
        # 学習したデータの再利用ができる
        x = torch.tensor(
            (flag, ufun(state.current_offer), state.relative_time, cent),
            dtype=torch.float32,
        ).view(-1, 4)
        for i_index, row in enumerate(self.fiveUfun):
            for j_index, value in enumerate(row):
                if j_index == len(row) - 1:
                    self.fiveUfun[i_index][j_index] = x
                else:
                    self.fiveUfun[i_index][j_index] = self.fiveUfun[i_index][
                        j_index + 1
                    ]
        self.model.train()
        if self.hidden is not None:
            self.hidden = tuple(h.detach() for h in self.hidden)
        self.out, self.hidden = self.model(self.fiveUfun, self.hidden)
        if np.isnan(self.out[0][0].item()):
            self.out[0][0] = 0.7
        return self.out[0][0].item()

    def make_searchingPro(self, nmi: SAONMI, ufun, state: SAOState, flag, cent):
        self.model.eval()
        x = torch.tensor(
            (flag, ufun(state.current_offer), state.relative_time, cent),
            dtype=torch.float32,
        ).view(-1, 4)
        for i_index, row in enumerate(self.fiveUfun):
            for j_index, value in enumerate(row):
                if j_index == len(row) - 1:
                    self.fiveUfun[i_index][j_index] = x
                else:
                    self.fiveUfun[i_index][j_index] = self.fiveUfun[i_index][
                        j_index + 1
                    ]
        with torch.no_grad():
            if self.hidden is not None:
                self.hidden = tuple(h.detach() for h in self.hidden)
            out, self.hidden = self.model(self.fiveUfun, self.hidden)
        return out[0][0].item()

    def makerange(self, val):
        return (val - self.alpha, val + self.alpha)

    def propose(
        self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        """Proposes to the given partner (dest) using the side negotiator (negotiator_id).

        Remarks:
        """
        assert self.ufun
        inverter = self.ensure_inverter(negotiator_id)
        negotiator, cntxt = self.negotiators[negotiator_id]
        nmi = negotiator.nmi
        ufun: SideUFun = cntxt["ufun"]
        cent = cntxt["center"]
        self.outcome = None
        if self.train == True:
            out = self.make_trainingPro(nmi, state, ufun, 1, cent)
        else:
            out = self.make_searchingPro(nmi, ufun, state, 1, cent)
        out = self.makerange(out)
        self.outcome = inverter.one_in(out, normalized=True, fallback_to_best=True)
        if self.outcome is None:
            self.outcome = choice(self._best)
            return self.outcome
        return self.outcome

    def respond(
        self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        """Responds to the given partner (source) using the side negotiator (negotiator_id).

        Remarks:
            - source: is the ID of the partner.
            - the mapping from negotiator_id to source is stable within a negotiation.

        """
        assert self.ufun
        inverter = self.ensure_inverter(negotiator_id)
        _, cntxt = self.negotiators[negotiator_id]
        ufun: SideUFun = cntxt["ufun"]
        cent = cntxt["center"]
        nmi = self.negotiators[negotiator_id][0].nmi
        if self.train == True:
            out = self.make_trainingPro(nmi, state, ufun, 2, cent)
        else:
            out = self.make_searchingPro(nmi, ufun, state, 2, cent)
        out = self.makerange(out)
        mx = out[1] * self._mx
        mn = out[0] * self._mx
        if (
            mn <= ufun(state.current_offer)
            and mx >= ufun(state.current_offer)
            and random() < 0.8
        ):
            return ResponseType.ACCEPT_OFFER
        elif mx < ufun(state.current_offer):
            return ResponseType.ACCEPT_OFFER
        if state.relative_time > 0.75 and ufun(state.current_offer) > ufun(None):
            return ResponseType.ACCEPT_OFFER
        if ufun(state.current_offer) > 0.85 * self._mx:
            return ResponseType.ACCEPT_OFFER
        else:
            return ResponseType.REJECT_OFFER

    def thread_init(self, negotiator_id, state):
        self.hidden = None
        return super().thread_init(negotiator_id, state)

    def interval_cover_loss(self, lower_pred, upper_pred, y_true):
        """
        lower_pred: 下限の予測値（Tensor）
        upper_pred: 上限の予測値（Tensor）
        y_true: 目的値（Tensor）
        """
        lower_violation = torch.clamp(y_true - upper_pred, min=0.0)  # 上限超過
        upper_violation = torch.clamp(lower_pred - y_true, min=0.0)  # 下限未満
        return (lower_violation**2 + upper_violation**2).mean()

    def interval_cover_loss_with_penalty(
        self, lower_pred, upper_pred, y_true, alpha=0.1
    ):
        coverage_loss = self.interval_cover_loss(lower_pred, upper_pred, y_true)
        width_penalty = ((upper_pred - lower_pred) ** 2).mean()
        return coverage_loss + alpha * width_penalty  # αで範囲の狭さをコントロール

    def thread_finalize(self, negotiator_id: str, state: SAOState) -> None:
        if (
            self.out is not None
            and self.train == True
            and negotiator_id != state.last_negotiator
        ):
            assert self.ufun
            self.model.train()

            self.optimizer.zero_grad()
            _, cntxt = self.negotiators[negotiator_id]
            ufun: SideUFun = cntxt["ufun"]
            self.loss1 = -self.out[0][0].mean()
            self.loss1.backward()
            self.optimizer.step()
            torch.save(self.model.state_dict(), self.path_2_myfile)

        if (
            self.out is not None
            and self.train == True
            and negotiator_id == state.last_negotiator
        ):
            assert self.ufun
            self.optimizer.zero_grad()
            _, cntxt = self.negotiators[negotiator_id]
            ufun: SideUFun = cntxt["ufun"]
            self.compromise = state.timedout
            self.loss1 = -self.out[0][0].mean()
            self.loss1.backward()
            self.optimizer.step()
            torch.save(self.model.state_dict(), self.path_2_myfile)
        for side in self.negotiators.keys():
            if side == negotiator_id:
                continue
            if side in self._inverter:
                del self._inverter[side]
        super.on_negotiation_end(negotiator_id, state)
