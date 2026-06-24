from .carcagent import *

MAIN_AGENT = CARCAgent
__all__ = carcagent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20344
NAME = 'CARCAgent'
CLASS_NAME = 'CARCAgent'
VERSION = '3.11.8'
TEAM = 'CARC'
AUTHOR = 'Tianzi Ma'
MEMBERS = [{'name': 'Tianzi Ma', 'institution': 'Harbin Institute of Technology, Shenzhen', 'country': 'China'}]
COUNTRY = 'China'
INSTITUTION = 'Harbin Institute of Technology, Shenzhen'
TAGS = ['Bayesian Methods', 'Game Theory', 'Optimization']
USES_LLM = False
DESC = "Our team has developed a highly efficient and effective automated negotiation agent for handling bilateral negotiations. Utilizing the alternating offer protocol for this tournament, we have devised a structured workflow divided into three key components: opponent modeling, acceptance strategy, and offer strategy. This approach enables us to estimate the opponent's undisclosed reserved value through curve fitting. Our acceptance strategy involves evaluating the current offer against our anticipated offer to determine whether to accept it. Lastly, our offer strategy aims to maximize our benefits throughout the negotiation process by generating offers in descending order relative to the negotiation's timeline."
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
