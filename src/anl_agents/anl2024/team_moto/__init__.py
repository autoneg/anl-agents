from .uoagent import *

MAIN_AGENT = UOAgent
__all__ = uoagent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20277
NAME = 'UOAgent'
CLASS_NAME = 'UOAgent'
VERSION = '3.11.8'
TEAM = 'Team moto'
AUTHOR = 'Hirotada Matsumoto'
MEMBERS = [{'name': 'Hirotada Matsumoto', 'institution': 'Tokyo University of Agriculture and Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Tokyo University of Agriculture and Technology'
TAGS = []
USES_LLM = False
DESC = "Greedy agents while aiming for a final agreement.\r\nEstimate the partner's reservation value and repeat the bidding at the very edge of that value.\r\nThe acceptance strategy is very stubborn and rarely accepts anything but the last step."
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
