"""
FCA-based Concept Drift Detection System
"""
__version__ = "0.1.0"
__author__ = "Maksym Kozlov"

from . import core
from . import fca
from . import detection
from . import evaluation
from . import visualization
from . import utils

__all__ = [
    'core',
    'fca',
    'detection',
    'evaluation',
    'visualization',
    'utils',
]
