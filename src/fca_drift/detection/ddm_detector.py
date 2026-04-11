"""
Drift Detection Method (DDM)
Gama et al., 2004
"""
import numpy as np
from typing import Optional
from .base import BaseDriftDetector, DriftEvent


class DDM(BaseDriftDetector):
    """
    Drift Detection Method
    Monitors error rate and standard deviation
    """

    def __init__(self,
                 min_instances: int = 30,
                 warning_level: float = 2.0,
                 drift_level: float = 3.0):
        super().__init__(name="DDM")

        self.min_instances = min_instances
        self.warning_level = warning_level
        self.drift_level = drift_level

        self.reset()

    def update(self, error: bool, instance_id: int) -> Optional[DriftEvent]:
        """
        Update with prediction error

        Args:
            error: True if prediction was wrong
            instance_id: Current instance ID
        """
        self.instance_count += 1
        if error:
            self.error_count += 1

        if self.instance_count < self.min_instances:
            return None

        # Compute error rate and std
        p = self.error_count / self.instance_count
        s = np.sqrt(p * (1 - p) / self.instance_count)

        # Update minimum
        if p + s < self.min_error_rate + self.min_std:
            self.min_error_rate = p
            self.min_std = s

        # Check for drift
        if p + s > self.min_error_rate + self.drift_level * self.min_std:
            self.drift_indices.append(instance_id)

            # Guard against zero/invalid denominator in early or degenerate states.
            denom = self.min_error_rate + self.drift_level * self.min_std
            if not np.isfinite(denom) or denom <= 0:
                confidence = 1.0
            else:
                confidence = (p + s) / denom

            event = DriftEvent(
                instance_id=instance_id,
                signal_idx=None,
                drift_type="sudden",
                confidence=confidence,
                metadata={
                    'error_rate': p,
                    'std': s,
                    'min_error_rate': self.min_error_rate
                }
            )

            self.drift_events.append(event)
            self.reset()
            return event

        # Check for warning
        if p + s > self.min_error_rate + self.warning_level * self.min_std:
            self.in_warning = True
        else:
            self.in_warning = False

        return None

    def reset(self):
        """Reset detector"""
        self.error_count = 0
        self.instance_count = 0
        self.min_error_rate = float('inf')
        self.min_std = float('inf')
        self.in_warning = False


if __name__ == "__main__":
    print("DDM Detector loaded")
