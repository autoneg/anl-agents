from .group6 import *

MAIN_AGENT = Group6
__all__ = group6.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20357
NAME = 'Group6'
CLASS_NAME = 'Group6'
VERSION = '3.11.8'
TEAM = 'Team 235'
AUTHOR = 'Asim Ozturk'
MEMBERS = [{'name': 'Asim Ozturk', 'institution': 'Özyeğin University', 'country': 'Türkiye'}, {'name': 'Selin Yilmaz', 'institution': 'Özyeğin University', 'country': 'Türkiye'}, {'name': 'Ege Ertugrul Eyiler', 'institution': 'Özyeğin University', 'country': 'Türkiye'}]
COUNTRY = 'Türkiye'
INSTITUTION = 'Özyeğin University'
TAGS = ['Game Theory', 'Psychology', 'Optimization']
USES_LLM = False
DESC = 'Our agent, named Group6, employs a dynamic negotiation strategy tailored to adapt its decisions over time. It bases its behavior on an evolving aspiration level, computed through a custom function that models utility expectations from maximum to minimum, adjusting for negotiation progression. This strategic flexibility is informed by historical offer utilities and timings, which the agent continuously tracks. During negotiations, Group6 calculates the aspiration level at each step to decide whether to accept the current offer or propose a counter based on a selection of potential outcomes that meet or exceed the current aspiration threshold. This approach allows the agent to strategically align with the negotiation phase and opponent behavior, aiming for optimal outcomes throughout the negotiation process.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
