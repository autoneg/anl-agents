from .astrat3m import *

MAIN_AGENT = Astrat3m
__all__ = astrat3m.__all__

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"

# --- BEGIN generated metadata (scmlweb update_agents_repo.py) ---
ID = 20407
NAME = 'Astrat3m'
CLASS_NAME = 'Astrat3m'
VERSION = '3.11.11'
TEAM = 'ChongqingAgent'
AUTHOR = 'Yunfei Wang'
MEMBERS = [{'name': 'Yunfei Wang', 'institution': 'Chongqing Jiaotong University', 'country': 'China'}, {'name': 'Siqi Chen', 'institution': 'Chongqing Jiaotong University', 'country': 'China'}]
COUNTRY = 'China'
INSTITUTION = 'Chongqing Jiaotong University'
TAGS = ['Reinforcement Learning']
USES_LLM = False
DESC = 'Astrat3m adopts a central-peripheral architecture. The central agent handles multi-round negotiations through dynamic programming, leveraging an opponent-checking mechanism and utility evaluation mechanism to build a negotiation combination optimization model for global optimal control. It can also flexibly adjust strategies based on the current situation, address the growth of combinatorial space, and monitor opponent strategies to update its own strategies. The peripheral agents use the MCQ (TD3) algorithm, which integrates TD3 and MCQ, to generate bidding strategies. Through offline reinforcement learning and inverse utility mapping technology, combined with a high-adversarial environment checking mechanism (using an SVM model to identify high-adversarial environments and activate the corresponding model), they determine the optimal negotiation combination according to the current utility function and propose solutions. When receiving a proposal, they decide whether to accept it by judging the utility ratio and other factors.'
METADATA = {
    'id': ID, 'name': NAME, 'class_name': CLASS_NAME,
    'version': VERSION, 'team': TEAM, 'author': AUTHOR,
    'members': MEMBERS, 'country': COUNTRY,
    'institution': INSTITUTION, 'tags': TAGS,
    'uses_llm': USES_LLM, 'description': DESC,
}
# --- END generated metadata ---
