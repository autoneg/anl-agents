from .the_memorizer import *
from .dinners_agent import *
from .itay_agent import *
from .itay_jhn_agent import *
from .job_dinner_agent import *
from .job_henter_agent import *
from .myagent import *

MAIN_AGENT = TheMemorizer
__all__ = (
    the_memorizer.__all__
    + dinners_agent.__all__
    + itay_agent.__all__
    + itay_jhn_agent.__all__
    + job_dinner_agent.__all__
    + job_henter_agent.__all__
    + myagent.__all__
)

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20715
NAME = 'TheMemorizer'
CLASS_NAME = 'TheMemorizer'
VERSION = '3.11.8'
TEAM = 'Team 307'
AUTHOR = 'Ido'
MEMBERS = [{'name': 'Ido', 'institution': 'Bar-Ilan University', 'country': 'Israel'}, {'name': 'Gabriele Guetta', 'institution': 'Bar-Ilan University', 'country': 'Israel'}, {'name': 'Itay Elyashiv', 'institution': 'Bar-Ilan University', 'country': 'Israel'}]
COUNTRY = 'Israel'
INSTITUTION = 'Bar-Ilan University'
TAGS = []
USES_LLM = False
DESC = 'TheMemorizer Agent Strategy Description\r\nTheMemorizer is an adaptive negotiation agent that employs a memory-based learning approach with dynamic strategy adjustment based on agent position and negotiation context.\r\nCore Strategy Components:\r\nMemory System: The agent maintains detailed records of all offers, rejections, and outcomes across negotiations, building a comprehensive understanding of opponent behavior patterns and negotiation dynamics.\r\nAdaptive Positioning: The strategy differentiates between edge agents (first/last in sequence) and center agents, with edge agents focusing on individual utility maximization while center agents consider the full negotiation chain context.\r\nDynamic Acceptance Threshold: Uses statistical analysis of utility distributions to set acceptance thresholds, incorporating negotiation progress, variance in outcomes, and agent position. The threshold becomes more lenient as time progresses and adjusts based on the standard deviation of available utilities.\r\nLeverage-Based Adjustment: Implements a leverage system where later negotiations in the sequence carry more weight, allowing the agent to hold stronger positions when it has more negotiating power.\r\nMCUF Optimization: For Maximum Center Utility Function scenarios, employs specialized algorithms to compute optimal outcomes when the search space is manageable, switching to heuristic approaches for larger spaces.\r\nThe agent balances exploration of new outcomes with exploitation of learned patterns, making it particularly effective in multi-round tournaments where opponent modeling becomes crucial for success.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
