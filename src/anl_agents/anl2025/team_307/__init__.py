from .the_memorizer import *
from .dinners_agent import *
from .itay_agent import *
from .itay_jhn_agent import *
from .job_dinner_agent import *
from .job_henter_agent import *
from .myagent import *

MAIN_AGENT = TheMemorizer
__all__ = (
    the_memorizer.__all__
    + dinners_agent.__all__
    + itay_agent.__all__
    + itay_jhn_agent.__all__
    + job_dinner_agent.__all__
    + job_henter_agent.__all__
    + myagent.__all__
)

__author__ = ""
__team__ = ""
__email__ = ""
__version__ = "0.1.0"
