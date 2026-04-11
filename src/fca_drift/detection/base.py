"""
Base class for drift detectors
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class DriftEvent:
    """Represents a detected drift event"""
    instance_id: int
    signal_idx: Optional[int]
    drift_type: str
    confidence: float
    metadata: dict

    def __repr__(self):
        if self.signal_idx is None:
            return (
                f"DriftEvent(instance_id={self.instance_id}, "
                f"type={self.drift_type}, conf={self.confidence:.3f})"
            )
        return (
            f"DriftEvent(instance_id={self.instance_id}, signal_idx={self.signal_idx}, "
            f"type={self.drift_type}, conf={self.confidence:.3f})"
        )


class BaseDriftDetector(ABC):
    """Abstract base class for drift detectors"""

    def __init__(self, name: str = "BaseDetector"):
        self.name = name
        self.drift_indices = []
        self.drift_events = []

    @abstractmethod
    def update(self, value: Any, instance_id: int) -> Optional[DriftEvent]:
        """
        Update detector with new value

        Returns:
            DriftEvent if drift detected, None otherwise
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset detector state"""
        pass

    def get_drift_count(self) -> int:
        """Get total number of detected drifts"""
        return len(self.drift_indices)
