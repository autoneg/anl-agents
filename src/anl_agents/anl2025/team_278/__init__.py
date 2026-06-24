from .smart import *

MAIN_AGENT = SmartNegotiator
__all__ = smart.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20442
NAME = 'Smart Negotiator'
CLASS_NAME = 'SmartNegotiator'
VERSION = '3.13.1'
TEAM = 'Dream Team'
AUTHOR = 'Jonathan Mandl'
MEMBERS = [{'name': 'Jonathan Mandl', 'institution': 'Bar-Ilan University', 'country': 'Israel'}, {'name': 'Valeriia Ivanova', 'institution': 'Bar-Ilan University', 'country': 'Israel'}]
COUNTRY = 'Israel'
INSTITUTION = 'Bar-Ilan University'
TAGS = ['Reinforcement Learning']
USES_LLM = False
DESC = "Our agent combines Boulware-style concession with opponent modeling to dynamically adjust its utility acceptance thresholds over time. It computes a local threshold based on the negotiation step and a global threshold based on the negotiation's position in the overall sequence, averaging the two for center agents. Using an exponentially weighted moving average, it estimates the opponent’s concession rate and adapts both acceptance and proposal strategies accordingly. Proposals are drawn probabilistically from bids exceeding the adjusted threshold, ensuring a balance between utility optimization and responsiveness to opponent behavior."
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
