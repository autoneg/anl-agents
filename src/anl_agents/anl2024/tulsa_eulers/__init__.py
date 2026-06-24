from .goldie import *

MAIN_AGENT = Goldie
__all__ = goldie.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20282
NAME = 'Goldie'
CLASS_NAME = 'Goldie'
VERSION = '3.11.4'
TEAM = 'Tulsa Eulers'
AUTHOR = 'Ethan Beaird'
MEMBERS = [{'name': 'Ethan Beaird', 'institution': 'University of Tulsa', 'country': 'United States'}]
COUNTRY = 'United States'
INSTITUTION = 'University of Tulsa'
TAGS = ['Optimization']
USES_LLM = False
DESC = 'Guided Optimization Learning-Driven Intermediary Entity (GOLDIE) is a negotiation agent that combines the MiCRO strategy with a modified threshold-based conceder approach. It begins as a conceder based on a procedurally-chosen threshold determined by its reservation value. Once this threshold is passed, GOLDIE switches to behaving like a standard MiCRO agent. To enable opponent modeling, GOLDIE uses simplified RV fitting and adds stochastic noise to offers during the initial conceder phase to prevent adversarial modeling. It assumes a boulware-like opponent and accepts offers based on this model initially. Goldie also introduces randomness by occasionally giving worse offers, a tactic designed to disrupt opponent modeling strategies.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
