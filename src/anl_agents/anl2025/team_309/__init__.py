from .kagent import *

MAIN_AGENT = KAgent
__all__ = kagent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20744
NAME = 'MyAgent2000'
CLASS_NAME = 'MyAgent2000'
VERSION = '3.13.3'
TEAM = 'Team 309'
AUTHOR = 'kon'
MEMBERS = [{'name': 'kon', 'institution': 'colleagues', 'country': 'Japan'}]
COUNTRY = 'Japan'
INSTITUTION = 'colleagues'
TAGS = []
USES_LLM = False
DESC = 'My agent is designed to concede in the final stages when acting as a center agent, and to remain aggressive until the end when acting as an edge agent. Additionally, it observes the two most recent offers from the opponent and adjusts its level of aggressiveness based on the degree of the opponent’s concessions.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
