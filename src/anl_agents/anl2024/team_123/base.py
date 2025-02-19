from abc import ABC, abstractmethod
from typing import final

from negmas import Outcome
from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.sao import ResponseType, SAONegotiator
from negmas.sao.common import SAOResponse, SAOState

__all__ = ["BaseAgent"]


class BaseAgent(SAONegotiator, ABC):
    @abstractmethod
    def should_accept(self, state: SAOState) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_first_offer(self) -> Outcome:
        raise NotImplementedError()

    @abstractmethod
    def get_offer(self, state: SAOState) -> Outcome:
        raise NotImplementedError()

    @final
    @property
    def my_ufun_typesafe(self) -> BaseUtilityFunction:
        if self.ufun is None:
            raise ValueError("Utility function is not set")
        return self.ufun

    @final
    @property
    def opponent_ufun_typesafe(self) -> BaseUtilityFunction:
        if self.opponent_ufun is None:
            raise ValueError("Utility function is not set")
        return self.opponent_ufun

    @final
    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        if state.current_offer is None:
            # start negotiation
            outcome = self.get_first_offer()
            return SAOResponse(ResponseType.REJECT_OFFER, outcome)

        if self.should_accept(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

        outcome = self.get_offer(state)
        return SAOResponse(ResponseType.REJECT_OFFER, outcome)
