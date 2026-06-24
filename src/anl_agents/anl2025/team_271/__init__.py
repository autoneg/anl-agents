from .agents import *

MAIN_AGENT = RUFL
__all__ = agents.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20826
NAME = 'NewUtilityFitLookAheadAgent'
CLASS_NAME = 'NewUtilityFitLookAheadAgent'
VERSION = '3.11.11'
TEAM = 'Team 271'
AUTHOR = 'Garrett Seo'
MEMBERS = [{'name': 'Garrett Seo', 'institution': 'Rutgers University', 'country': 'United States'}, {'name': 'Xintong Wang', 'institution': 'Rutgers University', 'country': 'United States'}, {'name': 'Tri-an Nguyen', 'institution': 'Rutgers University', 'country': 'United States'}]
COUNTRY = 'United States'
INSTITUTION = 'Rutgers University'
TAGS = ['Bayesian Methods', 'Heuristic / Rule-based', 'Optimization']
USES_LLM = False
DESC = 'The agent first plans by doing a lookahead. Because future subnegotiations affect current negotiations, the value of an outcome is dependent on the actions made in later subnegotiatons. We aim to do this expressing the negotiation as a game tree, where each level represents one subnegotiation and the children represents the possible outcomes at each subnegotiation. We assign a probability distribution over the outcomes at each node, depending on its expected utility. We backpropagate expected utilities all the way to root of the tree.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
