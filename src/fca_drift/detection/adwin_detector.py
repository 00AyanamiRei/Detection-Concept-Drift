"""
ADWIN Detector - River wrapper
"""
from typing import Optional
from .base import BaseDriftDetector, DriftEvent

try:
    from river.drift import ADWIN as RiverADWIN
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    print("[WARNING] River not available for ADWIN")


class ADWINDetector(BaseDriftDetector):
    """
    Wrapper for River's ADWIN
    """

    def __init__(self, delta: float = 0.002):
        super().__init__(name="ADWIN")

        if not RIVER_AVAILABLE:
            raise RuntimeError("River not available. Install: pip install river")

        self.delta = delta
        self.adwin = RiverADWIN(delta=delta)

    def update(self, value: float, instance_id: int) -> Optional[DriftEvent]:
        """
        Update with new value

        Args:
            value: Numeric value (e.g., error rate, ΔL_t)
            instance_id: Current instance ID
        """
        self.adwin.update(value)

        if self.adwin.drift_detected:
            self.drift_indices.append(instance_id)

            event = DriftEvent(
                instance_id=instance_id,
                signal_idx=None,
                drift_type="sudden",
                confidence=1.0,
                metadata={'value': value}
            )

            self.drift_events.append(event)
            return event

        return None

    def reset(self):
        """Reset detector"""
        self.adwin = RiverADWIN(delta=self.delta)


if __name__ == "__main__":
    print("ADWIN Detector loaded")
