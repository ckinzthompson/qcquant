"""
qcquant - Quantitative Chemotaxis Quantification
"""

__title__ = "qcquant"
__version__ = "0.4.0"

__description__ = "qcquant is for analyzing bacterial chemotaxis plate assays"

__license__ = "GPLv3"
__url__ = "https://github.com/ckinzthompson/qcquant"

__author__ = "ckinzthompson"

from .qcquant import run_app
from . import qcquant_fitting as fitting