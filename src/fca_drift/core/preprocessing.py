"""Multi-Level Quantile Binarization for FCA context construction.

WHY THIS REPLACES MEDIAN BINARIZATION:
  The original single-threshold (median) binarization encodes each numeric
  attribute as ONE bit. Two windows that differ in distribution but share the
  same median produce identical binary rows → identical FCA lattices → delta_L ≈ 0
  even when a real drift has occurred.

  Multi-level quantile binarization uses Q quantile cut-points per attribute
  (default Q=3: the 25th, 50th and 75th percentiles), producing Q binary
  attributes per original attribute. This gives the FCA lattice 3× more
  structural information to work with. After a sudden drift the cut-points
  shift, creating clearly different lattice structures and a visible spike in
  delta_L.

STREAM-SAFE DESIGN:
  - Cut-points are learned from the first `freeze_after` windows via EMA.
  - After freeze the thresholds are fixed so lattices remain comparable.
  - Binary columns (0/1 only) and the __target__ column are handled specially.

BACKWARD COMPATIBILITY:
  The class keeps the same public API as the old DataPreprocessor so no other
  file needs to change.
"""

from typing import Any, Dict, List, Tuple

import numpy as np


# Default quantile levels. Three levels give a good balance between
# expressiveness and FCA lattice size (more levels → exponentially more concepts).
DEFAULT_QUANTILE_LEVELS: Tuple[float, ...] = (0.25, 0.50, 0.75)


class DataPreprocessor:
    """Transform numerical stream windows into binary FCA contexts.

    Each numeric attribute A is expanded into Q binary attributes:
        A_q = 1  if  A > quantile_q(A),  else 0      for q in quantile_levels

    Binary columns (values in {0, 1} only) and __target__ keep a single
    threshold at 0.5 to avoid inflating the attribute space needlessly.

    Parameters
    ----------
    threshold : float
        Fallback global threshold used only when adaptive_threshold=False.
    adaptive_threshold : bool
        If True, learn per-attribute quantile thresholds from the stream.
        If False, use `threshold` for every attribute (single-level).
    freeze_after : int
        Number of windows after which learned thresholds are frozen.
    ema_alpha : float
        EMA smoothing factor for threshold updates (0 < ema_alpha ≤ 1).
    quantile_levels : tuple of float
        Quantile cut-points in (0, 1).  Default: (0.25, 0.50, 0.75).
    """

    def __init__(
        self,
        threshold: float = 0.5,
        adaptive_threshold: bool = True,
        freeze_after: int = 2500,
        ema_alpha: float = 0.1,
        quantile_levels: Tuple[float, ...] = DEFAULT_QUANTILE_LEVELS,
    ):
        self.threshold = float(threshold)
        self.adaptive_threshold = bool(adaptive_threshold)
        self.freeze_after = max(1, int(freeze_after))
        self.ema_alpha = float(min(max(ema_alpha, 0.001), 1.0))
        self.quantile_levels: Tuple[float, ...] = tuple(sorted(quantile_levels))

        # ema_thresholds[attr_name][q_idx] = current EMA value for quantile q
        self._ema_thresholds: Dict[str, List[float]] = {}
        self._frozen_thresholds: Dict[str, List[float]] = {}
        self._window_counter = 0
        self._is_frozen = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(hash(str(value)) % 2)

    @staticmethod
    def _is_binary_column(col: np.ndarray) -> bool:
        unique = set(np.unique(col).tolist())
        return unique.issubset({0.0, 1.0})

    # ------------------------------------------------------------------
    # Threshold learning
    # ------------------------------------------------------------------

    def _update_adaptive_thresholds(
        self, attr_names: List[str], matrix: np.ndarray
    ) -> Dict[str, List[float]]:
        """Update EMA quantile thresholds from current window."""
        current: Dict[str, List[float]] = {}

        for i, attr_name in enumerate(attr_names):
            col = matrix[:, i]

            # Binary columns and __target__: single threshold at 0.5
            if self._is_binary_column(col) or attr_name == "__target__":
                current[attr_name] = [0.5]
                self._ema_thresholds[attr_name] = [0.5]
                continue

            # Compute quantiles from current window
            q_values = [float(np.quantile(col, q)) for q in self.quantile_levels]

            if attr_name not in self._ema_thresholds:
                self._ema_thresholds[attr_name] = list(q_values)
            else:
                prev = self._ema_thresholds[attr_name]
                # EMA update per quantile level
                updated = [
                    self.ema_alpha * qv + (1.0 - self.ema_alpha) * pv
                    for qv, pv in zip(q_values, prev)
                ]
                self._ema_thresholds[attr_name] = updated

            current[attr_name] = list(self._ema_thresholds[attr_name])

        return current

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def preprocess_window(self, window_data: List[Dict[str, Any]]) -> np.ndarray:
        """Convert one sliding window into a multi-level binary matrix.

        Output shape: (n_objects, n_expanded_attributes)
        where n_expanded_attributes = sum of quantile levels per attribute
        (binary/target columns contribute 1; numeric columns contribute Q).
        """
        if not window_data:
            return np.empty((0, 0), dtype=int)

        attr_names = sorted(window_data[0].keys(), key=str)
        matrix = np.array(
            [
                [self._to_float(sample.get(attr, 0.0)) for attr in attr_names]
                for sample in window_data
            ],
            dtype=float,
        )

        # ── Determine active thresholds ──────────────────────────────────
        if self.adaptive_threshold:
            if self._is_frozen:
                # Fill any unseen attributes with per-window quantiles (fallback)
                for i, attr_name in enumerate(attr_names):
                    if attr_name not in self._frozen_thresholds:
                        col = matrix[:, i]
                        if self._is_binary_column(col) or attr_name == "__target__":
                            self._frozen_thresholds[attr_name] = [0.5]
                        else:
                            self._frozen_thresholds[attr_name] = [
                                float(np.quantile(col, q)) for q in self.quantile_levels
                            ]
                active_thresholds = self._frozen_thresholds
            else:
                active_thresholds = self._update_adaptive_thresholds(attr_names, matrix)
                self._window_counter += 1
                if self._window_counter >= self.freeze_after:
                    self._frozen_thresholds = {k: list(v) for k, v in active_thresholds.items()}
                    self._is_frozen = True
                    active_thresholds = self._frozen_thresholds
        else:
            # Non-adaptive: single fixed threshold for every attribute
            for attr_name in attr_names:
                if attr_name not in self._frozen_thresholds:
                    self._frozen_thresholds[attr_name] = [self.threshold]
            self._is_frozen = True
            active_thresholds = self._frozen_thresholds

        # ── Build expanded binary matrix ─────────────────────────────────
        # Each attribute produces len(thresholds_for_attr) binary columns.
        binary_cols: List[np.ndarray] = []
        for i, attr_name in enumerate(attr_names):
            col = matrix[:, i]
            thresholds_for_attr = active_thresholds.get(attr_name, [self.threshold])
            for thr in thresholds_for_attr:
                binary_cols.append((col > thr).astype(int))

        if not binary_cols:
            return np.empty((len(window_data), 0), dtype=int)

        return np.column_stack(binary_cols)

    # ------------------------------------------------------------------
    # Introspection helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def get_frozen_thresholds(self) -> Dict[str, List[float]]:
        """Return frozen thresholds (list per attribute), or latest EMA values."""
        if self._is_frozen:
            return {k: list(v) for k, v in self._frozen_thresholds.items()}
        return {k: list(v) for k, v in self._ema_thresholds.items()}

    def is_frozen(self) -> bool:
        return self._is_frozen

    def reset(self) -> None:
        self._ema_thresholds = {}
        self._frozen_thresholds = {}
        self._window_counter = 0
        self._is_frozen = False


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    pre = DataPreprocessor(adaptive_threshold=True, freeze_after=3, ema_alpha=0.3)
    w = [
        {"salary": 30000.0, "age": 25.0, "label": 0.0},
        {"salary": 60000.0, "age": 40.0, "label": 1.0},
        {"salary": 90000.0, "age": 55.0, "label": 0.0},
        {"salary": 45000.0, "age": 33.0, "label": 1.0},
    ]
    for i in range(5):
        b = pre.preprocess_window(w)
        print(
            f"step={i} frozen={pre.is_frozen()} "
            f"output_shape={b.shape}  "
            f"(original 3 attrs → {b.shape[1]} binary cols)"
        )
        if i == 0:
            print("  binary matrix:\n", b)
