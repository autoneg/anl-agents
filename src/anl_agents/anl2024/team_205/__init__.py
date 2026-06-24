from .kosagent import *

MAIN_AGENT = KosAgent
__all__ = kosagent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20269
NAME = 'KosAgent'
CLASS_NAME = 'KosAgent'
VERSION = '3.11.8'
TEAM = 'Team 205'
AUTHOR = 'Kosuke Nakata'
MEMBERS = [{'name': 'Kosuke Nakata', 'institution': 'Tokyo University of Agriculture and Technology', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'Tokyo University of Agriculture and Technology'
TAGS = []
USES_LLM = False
DESC = 'My agent is basically bullish and aim to reach an agreement with the other party as it approaches the end. In addition, it can choose the best option for the other person from a range of utility value choices I have determined. To implement it, I introduced some functional models. My agent is based on time dependent model. It is a model that decreases its own desired utility value over time. Also, I implemented to make suggestions so that the other party could not read my model. My agent is used different functional models for the acceptance and proposal strategies. However, I didn’t implement opponent model. That’s because I can find opponent’s utility in this time.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
