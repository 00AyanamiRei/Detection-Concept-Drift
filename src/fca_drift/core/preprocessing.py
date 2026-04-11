"""Frozen-Median Binarization preprocessing for FCA context construction.

Design goals:
- Robust per-attribute thresholding for mixed-scale numeric streams.
- Stream-friendly adaptation using EMA of window medians.
- Threshold freeze after sufficient calibration to stabilize binarization.

Rationale:
Calibrating thresholds from too few early windows can lock in unstable values.
This implementation learns thresholds during a warm calibration phase and then
freezes them so overlapping windows remain consistent in long streams.
"""

from typing import Any, Dict, List

import numpy as np


class DataPreprocessor:
    """Transform numerical stream windows into binary FCA contexts.

    Behavior:
    - If adaptive_threshold=True:
      - Track per-attribute EMA of window medians.
      - Freeze thresholds after freeze_after windows.
    - If adaptive_threshold=False:
      - Use a fixed global threshold.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        adaptive_threshold: bool = True,
        freeze_after: int = 2500,
        ema_alpha: float = 0.1,
    ):
        self.threshold = float(threshold)
        self.adaptive_threshold = bool(adaptive_threshold)
        self.freeze_after = max(1, int(freeze_after))
        self.ema_alpha = float(min(max(ema_alpha, 0.001), 1.0))

        self._ema_thresholds: Dict[str, float] = {}
        self._frozen_thresholds: Dict[str, float] = {}
        self._window_counter = 0
        self._is_frozen = False

    @staticmethod
    def _to_float(value: Any) -> float:
        """Convert value to float; map non-numeric values deterministically."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(hash(str(value)) % 2)

    @staticmethod
    def _is_binary_column(col: np.ndarray) -> bool:
        unique = set(np.unique(col).tolist())
        return unique.issubset({0.0, 1.0})

    def _update_adaptive_thresholds(self, attr_names: List[str], matrix: np.ndarray) -> Dict[str, float]:
        medians = np.median(matrix, axis=0)
        current: Dict[str, float] = {}

        for i, attr_name in enumerate(attr_names):
            col = matrix[:, i]

            if attr_name == "__target__" and self._is_binary_column(col):
                current[attr_name] = 0.5
                continue

            med = float(medians[i])
            if attr_name not in self._ema_thresholds:
                self._ema_thresholds[attr_name] = med
            else:
                prev = self._ema_thresholds[attr_name]
                self._ema_thresholds[attr_name] = self.ema_alpha * med + (1.0 - self.ema_alpha) * prev

            current[attr_name] = float(self._ema_thresholds[attr_name])

        return current

    def preprocess_window(self, window_data: List[Dict[str, Any]]) -> np.ndarray:
        """Convert one sliding window into a binary matrix."""
        if not window_data:
            return np.empty((0, 0), dtype=int)

        attr_names = sorted(window_data[0].keys(), key=str)
        matrix = np.array(
            [[self._to_float(sample.get(attr, 0.0)) for attr in attr_names] for sample in window_data],
            dtype=float,
        )

        if self.adaptive_threshold:
            if self._is_frozen:
                for i, attr_name in enumerate(attr_names):
                    if attr_name not in self._frozen_thresholds:
                        col = matrix[:, i]
                        if attr_name == "__target__" and self._is_binary_column(col):
                            self._frozen_thresholds[attr_name] = 0.5
                        else:
                            self._frozen_thresholds[attr_name] = float(np.median(col))
                active_thresholds = self._frozen_thresholds
            else:
                active_thresholds = self._update_adaptive_thresholds(attr_names, matrix)
                self._window_counter += 1

                if self._window_counter >= self.freeze_after:
                    self._frozen_thresholds = dict(active_thresholds)
                    self._is_frozen = True
                    active_thresholds = self._frozen_thresholds
        else:
            for attr_name in attr_names:
                if attr_name not in self._frozen_thresholds:
                    self._frozen_thresholds[attr_name] = self.threshold
            self._is_frozen = True
            active_thresholds = self._frozen_thresholds

        thresholds = np.array([active_thresholds[attr] for attr in attr_names], dtype=float)
        return (matrix > thresholds).astype(int)

    def get_frozen_thresholds(self) -> Dict[str, float]:
        """Return frozen thresholds, or latest EMA thresholds before freeze."""
        if self._is_frozen:
            return dict(self._frozen_thresholds)
        return dict(self._ema_thresholds)

    def is_frozen(self) -> bool:
        return self._is_frozen

    def reset(self) -> None:
        self._ema_thresholds = {}
        self._frozen_thresholds = {}
        self._window_counter = 0
        self._is_frozen = False


if __name__ == "__main__":
    pre = DataPreprocessor(adaptive_threshold=True, freeze_after=3, ema_alpha=0.2)
    w = [
        {"x0": 1.0, "x1": 10.0},
        {"x0": 2.0, "x1": 20.0},
        {"x0": 3.0, "x1": 30.0},
    ]
    for i in range(5):
        b = pre.preprocess_window(w)
        print(i, "frozen=", pre.is_frozen(), "thr=", pre.get_frozen_thresholds(), "bin=", b.tolist())
