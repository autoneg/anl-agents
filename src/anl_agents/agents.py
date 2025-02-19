from __future__ import annotations

from inspect import ismodule
from typing import Literal, overload

from negmas import SAONegotiator
from negmas.helpers import get_class, get_full_type_name

import anl_agents.anl2024 as anl2024
from anl_agents.anl2024.antiagents.antiagent import AntiAgent
from anl_agents.anl2024.carc.carcagent import CARCAgent
from anl_agents.anl2024.susumu.nayesian2 import Nayesian2
from anl_agents.anl2024.takafam.shochan import Shochan
from anl_agents.anl2024.team_123.nyan import AgentNyan
from anl_agents.anl2024.team_182.missg import MissG
from anl_agents.anl2024.team_186.ilan import Ilan
from anl_agents.anl2024.team_191.inegotiator import INegotiator
from anl_agents.anl2024.team_199.hard_chaos import HardChaosNegotiator
from anl_agents.anl2024.team_205.kosagent import KosAgent
from anl_agents.anl2024.team_209.bargain_bot import BargainBot
from anl_agents.anl2024.team_232.tak import TAKAgent
from anl_agents.anl2024.team_234.ardabot import Ardabot
from anl_agents.anl2024.team_235.group6 import Group6
from anl_agents.anl2024.team_236.group7 import Group7
from anl_agents.anl2024.team_240.bidbot import BidBot
from anl_agents.anl2024.team_moto.uoagent import UOAgent
from anl_agents.anl2024.team_renting.renting2024 import AgentRenting2024
from anl_agents.anl2024.team_twistin.group5 import Group5
from anl_agents.anl2024.teamkb.agentkb import AgentKB
from anl_agents.anl2024.tipsonly.katla_nir_aent import KatlaNirAgent
from anl_agents.anl2024.tulsa_eulers.goldie import Goldie


__all__ = ["get_agents", "FAILING_AGENTS"]

FAILING_AGENTS = {
    # get_full_type_name(
    #     anl2021.YIYAgent
    # ): "Needs scikit-learn<=1.3.* and is tested on python 3.10 only",
    # get_full_type_name(
    #     anl2021.QlAgent
    # ): "Needs scikit-learn<=1.3.* and is tested on python 3.10 only",
    # get_full_type_name(
    #     anl2022.AdaptiveQlAgent
    # ): "Needs scikit-learn<=1.3.* and is tested on python 3.10 only",
}
"""Maps agents known to fail to the failure reason."""


@overload
def get_agents(
    version: str | int,
    *,
    track: str = "advantage",
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
    top_only: int | float | None = None,
    ignore_failing=False,
    as_class: Literal[False] = False,
) -> tuple[str, ...]: ...


@overload
def get_agents(
    version: str | int,
    *,
    track: str = "advantage",
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
    top_only: int | float | None = None,
    ignore_failing=False,
    as_class: Literal[True] = True,
) -> tuple[type[SAONegotiator], ...]: ...


def get_agents(
    version: str | int,
    *,
    track: str = "advantage",
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
    top_only: int | float | None = None,
    ignore_failing=False,
    as_class: bool = True,
) -> tuple[type[SAONegotiator] | str, ...]:
    """
    Gets agent classes/full class names for a version which can either be a competition year (int) or "contrib".

    Args:
        version: Either a competition year (2019, 2020, 2021, ....) or the following special values:

                 - "contrib" for agents contributed directly to the repository not through ANAC's anl Competition
                 - "all"/"any" for all agents

        track: The track ("welfare", "utility", "advantage", "nash", "kalai", "kalai-smorodonisky")
        qualified_only: If true, only agents that were submitted to anl and ran in the qualifications round will be
                        returned
        finalists_only: If true, only agents that were submitted to anl and passed qualifications will be
                        returned
        winners_only: If true, only winners of anl (the given version) will be returned.
        top_only: Either a fraction of finalists or the top n finalists with highest scores in the finals of
                  anl
        as_class: If true, the agent classes will be returned otherwise their full class names.
    """
    if version in ("all", "any"):
        results = []
        for v in (2024, "contrib"):
            results += list(
                get_agents(  # type: ignore
                    v,
                    track=track,
                    qualified_only=qualified_only,
                    finalists_only=finalists_only,
                    winners_only=winners_only,
                    top_only=top_only,
                    as_class=as_class,  # type: ignore
                )
            )
        if ignore_failing:
            results = [
                _ for _ in results if get_full_type_name(_) not in FAILING_AGENTS.keys()
            ]
        return tuple(results)
    classes: tuple[str | type[SAONegotiator], ...] = tuple()  # type: ignore
    track = track.lower()
    if isinstance(version, int) and version == 2024:
        if track in ("advantage",) and winners_only:
            classes = (Shochan, UOAgent, AgentRenting2024)
        elif track in ("nash",) and winners_only:
            classes = (Shochan,)
        elif track in ("advantage",) and finalists_only:
            classes = (
                Shochan,
                UOAgent,
                AgentRenting2024,
                AntiAgent,
                HardChaosNegotiator,
                KosAgent,
                Nayesian2,
                CARCAgent,
                BidBot,
                AgentNyan,
            )
        elif track in ("nash",) and finalists_only:
            classes = (
                Shochan,
                CARCAgent,
                Nayesian2,
                AntiAgent,
                UOAgent,
                AgentRenting2024,
                KosAgent,
                BidBot,
                HardChaosNegotiator,
                AgentNyan,
            )
        elif track in ("kalai",) and finalists_only:
            classes = (
                Nayesian2,
                Shochan,
                CARCAgent,
                UOAgent,
                KosAgent,
                HardChaosNegotiator,
                BidBot,
                AntiAgent,
                AgentRenting2024,
                AgentNyan,
            )
        elif track in ("kalai-smorodonisky",) and finalists_only:
            classes = (
                Nayesian2,
                CARCAgent,
                KosAgent,
                Shochan,
                UOAgent,
                AntiAgent,
                AgentRenting2024,
                BidBot,
                HardChaosNegotiator,
                AgentNyan,
            )
        elif track in ("utility", "welfare") and finalists_only:
            classes = (
                UOAgent,
                Shochan,
                AgentRenting2024,
                AntiAgent,
                KosAgent,
                HardChaosNegotiator,
                Nayesian2,
                CARCAgent,
                BidBot,
                AgentNyan,
            )
        elif track in ("welfare",) and qualified_only:
            classes = (
                Nayesian2,
                INegotiator,
                Ardabot,
                Shochan,
                BidBot,
                AgentRenting2024,
                Group6,
                AntiAgent,
                KosAgent,
                UOAgent,
                Group5,
                CARCAgent,
                Goldie,
                Group7,
                Ilan,
                AgentNyan,
                KatlaNirAgent,
                BargainBot,
                HardChaosNegotiator,
                MissG,
                TAKAgent,
                AgentKB,
            )
        elif track in ("nash", "kalai", "kalai-smorodonisky") and qualified_only:
            classes = (
                CARCAgent,
                Nayesian2,
                Shochan,
                Group6,
                BidBot,
                Goldie,
                AntiAgent,
                Ilan,
                Group7,
                INegotiator,
                AgentRenting2024,
                KosAgent,
                AgentNyan,
                BargainBot,
                UOAgent,
                Group5,
                HardChaosNegotiator,
                MissG,
                TAKAgent,
                Ardabot,
                AgentKB,
                KatlaNirAgent,
            )
        elif (
            track
            in (
                "advantage",
                "welfare",
                "utility",
            )
            and qualified_only
        ):
            classes = (
                AgentRenting2024,
                AntiAgent,
                Shochan,
                UOAgent,
                KosAgent,
                HardChaosNegotiator,
                Nayesian2,
                CARCAgent,
                MissG,
                AgentNyan,
                BidBot,
                Group6,
                Group7,
                INegotiator,
                BargainBot,
                Group5,
                Ilan,
                Goldie,
                TAKAgent,
                AgentKB,
                Ardabot,
                KatlaNirAgent,
            )
        else:
            classes = tuple(
                sum(
                    (
                        [eval(f"anl2024.{_}.{a}") for a in eval(f"anl2024.{_}").__all__]
                        for _ in dir(anl2024)
                        if ismodule(eval(f"anl2024.{_}"))
                    ),
                    [],
                )
            )
    elif isinstance(version, str) and version == "contrib":
        classes = tuple()
    else:
        raise ValueError(
            f"The version {version} is unknown. Valid versions are 2019, 2020 (as ints), 'contrib' as a string"
        )
    classes: tuple[type[SAONegotiator] | str, ...]
    if as_class:
        classes = tuple(get_class(_) for _ in classes)
    else:
        classes = tuple(get_full_type_name(_) for _ in classes)  # type: ignore

    if ignore_failing:
        classes = tuple(
            [_ for _ in classes if get_full_type_name(_) not in FAILING_AGENTS.keys()]
        )
    if top_only is not None:
        n = int(top_only) if top_only >= 1 else (top_only * len(classes))
        if n > 0:
            return tuple(classes[: min(n, len(classes))])

    return classes  # type: ignore
