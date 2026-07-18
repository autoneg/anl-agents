"""Generated registry of participants per (year, track).

WRITTEN by scmlweb/python/update_agents_repo.py and friends. Edits
made by hand will be overwritten on the next run -- use
set_finalists.py / set_winners.py to flip the per-entry metadata
flags.

`get_participants(year, track=None, qualified_only=False,
finalists_only=False, winners_only=False)` returns the participants (or
a filtered subset) for a given year / track. `track` is required for
SCML and unused for ANL/HAN. `qualified_only` drops disqualified
entries (a qualified agent is any non-disqualified participant).
"""
from __future__ import annotations

import json
from typing import Optional


# (year, track-or-None) -> list of {class_path, metadata}.
# Stored as JSON (loaded at import) so the booleans/None serialise correctly —
# a raw Python-literal paste would emit JSON `false`/`true`/`null` and break.
_REGISTRY: dict = json.loads(r"""
{
    "2026|": [
        {
            "class_path": "anl_agents.anl2026.team_20462.overlap_conceder.OverlapConceder",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "20462",
                "name": "OverlapConceder"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_20829.snake_agent.Snake",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "20829",
                "name": "Snake"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_20963.anl2026.ianos.ianos.Ianos",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "20963",
                "name": "Ianos"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21043.agent360.Agent360",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21043",
                "name": "Agent360"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21044.emef_agent.EmEfAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21044",
                "name": "EmEfAgent"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21102.negotiatorx_anl.agent.NegotiatorXAnl",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21102",
                "name": "NegotiatorX_ANL"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21124.iscas.IscasAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "21124",
                "name": "IscasAgent"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21140.tjqzagent.TjAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21140",
                "name": "tjqzagent"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21328.miya_dreambelief_agent.agent.MiyaDreamBeliefNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21328",
                "name": "MiyaDreamBelief"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21329.aa_nante_lucky.AaNanteLucky",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21329",
                "name": "AaNanteLucky"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21350.whale.WhaleNegotiator",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21350",
                "name": "WhaleV0.2"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21353.harrison_neg.HarrisonNeg",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": false,
                "has_description": true,
                "team_id": "21353",
                "name": "LoonyGryphon"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21369.lionelwei_agent.LionelWei",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21369",
                "name": "LIonel"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21381.chang_agent.ChangAgent",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21381",
                "name": "ChangAgent"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21413.ozu_negotiator.OzuNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21413",
                "name": "OzuNegotiator"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21565.badiron.Badiron",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21565",
                "name": "BadIronV4"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21569.group_n.GroupN",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21569",
                "name": "GroupN"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21573.decepton.DecepTon",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21573",
                "name": "DecepTor"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21584.examples.myway.MajiKayo",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21584",
                "name": "MajiKayo"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21589.cunningmerchant.CunningMerchant",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21589",
                "name": "Cunning Merchant"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21600.agent_tokyo_v11.AgentTokyoV11",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": false,
                "disqualified": true,
                "uses_llm": false,
                "has_report": false,
                "has_description": false,
                "team_id": "21600",
                "name": "TokyoV11"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21621.aoa.AOA008",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21621",
                "name": "AOA007"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21639.agentnexus.AgentNexus",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21639",
                "name": "AgentNexus_2.0"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21676.hallucinators.Hallucinators",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21676",
                "name": "HalucinatorAgent2026"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21688.balanceok.BalanceOKNegotiator",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21688",
                "name": "BalanceAgent"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21697.phantom8_negotiator.Phantom8Negotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": false,
                "disqualified": true,
                "uses_llm": false,
                "has_report": false,
                "has_description": false,
                "team_id": "21697",
                "name": "phantom_etneg"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21728.xgnegotiator.XGAgent",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21728",
                "name": "XGAgent"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21759.ceanl.CeanlNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21759",
                "name": "AdaptiveBathNegotiator"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21762.better_call_agent_infinity_v1000.BetterCallAgentInfinityV1000",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21762",
                "name": "BetterCallAgentInfinityV1000"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21786.anchor.AnchorNegotiator",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21786",
                "name": "Anchor"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21795.nashty.NashtyNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21795",
                "name": "Nashty Negotiator 12"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21802.anl_agent.entry.AnlOmegaNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21802",
                "name": "CodexAgentAnl"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_21810.phantom8_negotiator.Phantom8Negotiator",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "21810",
                "name": "Phantom8"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_22139.erman_conceal_negotiator.ErmanConcealNegotiator",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": false,
                "team_id": "22139",
                "name": "Erman Conceal Negotiator"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_22146.staborn_negotiator.StaBornNeg",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "22146",
                "name": "Staborn Negotiator"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_22257.hi.anl.Anl",
            "metadata": {
                "finalist": false,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": false,
                "has_description": true,
                "team_id": "22257",
                "name": "ohanl"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_22271.perikos_v3.PerikosV3",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "22271",
                "name": "PerikosV3"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_22289.mirage_v145.MirageV145",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "22289",
                "name": "MirageV145"
            }
        },
        {
            "class_path": "anl_agents.anl2026.team_22290.sbdanl.SBDANL",
            "metadata": {
                "finalist": true,
                "winner": false,
                "qualified": true,
                "disqualified": false,
                "uses_llm": false,
                "has_report": true,
                "has_description": true,
                "team_id": "22290",
                "name": "SBDANL"
            }
        }
    ]
}
""")


def get_participants(
    year: int,
    track: Optional[str] = None,
    *,
    qualified_only: bool = False,
    finalists_only: bool = False,
    winners_only: bool = False,
) -> tuple[str, ...]:
    """Return the dotted Python paths of registered participants."""
    # _REGISTRY keys are the JSON-serialised "year|track" strings (see
    # rewrite_registry); build the same form rather than a tuple.
    key = f"{int(year)}|{track.lower() if track else ''}"
    entries = _REGISTRY.get(key, [])
    out = []
    for e in entries:
        meta = e.get("metadata", {})
        if qualified_only and meta.get("disqualified"):
            continue
        if finalists_only and not meta.get("finalist"):
            continue
        if winners_only and not meta.get("winner"):
            continue
        out.append(e["class_path"])
    return tuple(out)
