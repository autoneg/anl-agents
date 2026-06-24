from .inegotiator import *
from .others import *

MAIN_AGENT = INegotiator
__all__ = inegotiator.__all__ + others.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = None
NAME = 'INegotiator'
CLASS_NAME = 'INegotiator'
VERSION = ''
TEAM = 'Team 191'
AUTHOR = 'ingoari'
MEMBERS = [{'name': 'ingoari', 'institution': 'Utrecht University', 'country': 'Netherlands'}]
COUNTRY = 'Netherlands'
INSTITUTION = 'Utrecht University'
TAGS = []
USES_LLM = False
DESC = ''
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
