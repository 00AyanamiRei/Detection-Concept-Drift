"""
Drift detection modules
"""
from .base import BaseDriftDetector, DriftEvent
from .fca_detector import FCADriftDetector
from .ddm_detector import DDM
from .eddm_detector import EDDM

try:
    from .adwin_detector import ADWINDetector
    ADWIN_AVAILABLE = True
except:
    ADWIN_AVAILABLE = False

__all__ = [
    'BaseDriftDetector',
    'DriftEvent',
    'FCADriftDetector',
    'DDM',
    'EDDM',
]

if ADWIN_AVAILABLE:
    __all__.append('ADWINDetector')
