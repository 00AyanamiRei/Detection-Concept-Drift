"""
Early Drift Detection Method (EDDM)
Baena-García et al., 2006
"""
import numpy as np
from typing import Optional
from .base import BaseDriftDetector, DriftEvent


class EDDM(BaseDriftDetector):
    """
    Early Drift Detection Method
    Monitors distance between errors
    """

    def __init__(self,
                 min_instances: int = 30,
                 warning_level: float = 0.95,
                 drift_level: float = 0.90):
        super().__init__(name="EDDM")

        self.min_instances = min_instances
        self.warning_level = warning_level
        self.drift_level = drift_level

        self.reset()

    def update(self, error: bool, instance_id: int) -> Optional[DriftEvent]:
        """Update with prediction error"""
        self.instance_count += 1

        if error:
            if self.last_error_instance is not None:
                distance = self.instance_count - self.last_error_instance
                self.distances.append(distance)
            self.last_error_instance = self.instance_count

        if len(self.distances) < self.min_instances:
            return None

        # Compute mean and std of distances
        mean_dist = np.mean(self.distances)
        std_dist = np.std(self.distances)

        # Update max
        if mean_dist + 2 * std_dist > self.max_mean + 2 * self.max_std:
            self.max_mean = mean_dist
            self.max_std = std_dist

        # Check for drift
        current = mean_dist + 2 * std_dist
        max_val = self.max_mean + 2 * self.max_std

        if max_val > 0 and current / max_val < self.drift_level:
            self.drift_indices.append(instance_id)

            event = DriftEvent(
                instance_id=instance_id,
                signal_idx=None,
                drift_type="gradual",
                confidence=1 - (current / max_val),
                metadata={
                    'mean_distance': mean_dist,
                    'std_distance': std_dist
                }
            )

            self.drift_events.append(event)
            self.reset()
            return event

        return None

    def reset(self):
        """Reset detector"""
        self.instance_count = 0
        self.last_error_instance = None
        self.distances = []
        self.max_mean = 0.0
        self.max_std = 0.0


if __name__ == "__main__":
    print("EDDM Detector loaded")
