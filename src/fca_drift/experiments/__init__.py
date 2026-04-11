"""
Experiments module - Runners and comparators
"""
from .runner import GridSearchRunner
from .comparator import DetectorComparison

__all__ = [
    'GridSearchRunner',
    'DetectorComparison',
]
