"""
qcquant - Quantitative Chemotaxis Quantification
"""

__title__ = "qcquant"
__version__ = "0.5.1"

__description__ = "qcquant is for analyzing bacterial chemotaxis plate assays"

__license__ = "GPLv3"
__url__ = "https://github.com/ckinzthompson/qcquant"

__author__ = "ckinzthompson"

from . import qcquant_fitting as fitting
from . import qcquant_calibrate as calibrate
from . import qcquant_process as process
from . import qcquant_app as app