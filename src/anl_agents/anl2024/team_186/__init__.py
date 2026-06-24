from .ilan import *
from .myagent import *

MAIN_AGENT = Ilan
__all__ = ilan.__all__ + myagent.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20230
NAME = 'Ilan'
CLASS_NAME = 'Ilan'
VERSION = '3.12.2'
TEAM = 'Team 186'
AUTHOR = 'Ilan Brilovitch'
MEMBERS = [{'name': 'Ilan Brilovitch', 'institution': 'College of Management Academic Studies', 'country': 'Israel'}, {'name': 'Yehuda Daniel', 'institution': 'College of Management Academic Studies', 'country': 'Israel'}, {'name': 'Chen Shalev', 'institution': 'College of Management Academic Studies', 'country': 'Israel'}, {'name': 'Tal Teri', 'institution': 'College of Management Academic Studies', 'country': 'Israel'}]
COUNTRY = 'Israel'
INSTITUTION = 'College of Management Academic Studies'
TAGS = ['Optimization']
USES_LLM = False
DESC = 'Changed to more aggressive and smarter negotiations'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
