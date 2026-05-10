"""
FCA-based drift detector.

Key guarantees:
- Dual-index model:
  - instance_id: original stream position (reporting only)
  - signal_idx: position in delta_L_history (all computations)
- Stable signal extraction:
  - support-filtered similarity from lattice similarity module
  - EMA smoothing over similarity before delta computation
- Detection remains point-based, but final drift type should be inferred at
  episode level by DriftAggregatorV2.

FIXES (supervisor feedback):
- Window size 300-500: cooldown and z_window scaled correctly
- EMA alpha raised to 0.6 so signal reacts faster to sudden drift
- spike_multiplier lowered so sudden spikes are not suppressed
- soft_floor kept tiny so adaptive threshold does NOT become huge
- similarity_ema_alpha set conservatively: 0.6 means faster response
"""

from typing import Dict, List, Optional

import numpy as np

from .base import BaseDriftDetector, DriftEvent
from ..fca import ConceptLattice, LatticeSimilarityCalculator
from ..history import DriftStatusEnum, LatticeFrame, LatticeHistoryManager
from ..visualization.lattice_snapshot_collector import LatticeSnapshotCollector


class FCADriftDetector(BaseDriftDetector):
    """Detect drift candidates from FCA lattice change signal."""

    # Unified sudden threshold used across detector + aggregator + report text.
    SUDDEN_Z_THRESHOLD = 2.5

    def __init__(
        self,
        theta: float = 0.4,
        alpha: float = 2.0,
        window_size: int = 300,
        adaptive_window: int = 10,
        recurring_lookback: int = 5,
        recurring_similarity_threshold: float = 0.85,
        min_persistence_windows: int = 2,
        cooldown_windows: Optional[int] = None,
        spike_multiplier: float = 1.1,      # FIX: was 1.25, lowered so spikes are not missed
        noise_baseline_k: float = 0.25,
        spike_min_z: float = 1.8,
        similarity_ema_alpha: float = 0.6,  # FIX: was 0.35, raised for faster response to sudden drift
        z_window: Optional[int] = None,
    ):
        super().__init__(name="FCA")

        self.theta = float(theta)
        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.adaptive_window = int(adaptive_window)
        self.recurring_lookback = int(recurring_lookback)
        self.recurring_similarity_threshold = float(recurring_similarity_threshold)
        self.min_persistence_windows = max(1, int(min_persistence_windows))

        # FIX: cooldown scaled to window_size but capped at reasonable value.
        # Old formula window_size // 4 gave cooldown=75 for W=300, which was
        # too large and suppressed the signal after the first alarm.
        # New formula: small fixed cooldown regardless of window size.
        if cooldown_windows is not None:
            self.cooldown_windows = max(1, int(cooldown_windows))
        else:
            self.cooldown_windows = max(3, min(15, self.window_size // 20))

        self.spike_multiplier = max(1.0, float(spike_multiplier))
        self.noise_baseline_k = max(0.0, float(noise_baseline_k))
        self.spike_min_z = max(0.0, float(spike_min_z))
        self.similarity_ema_alpha = min(max(float(similarity_ema_alpha), 0.01), 1.0)

        # FIX: z_window should be proportional to adaptive_window, NOT window_size.
        # Old: max(30, adaptive_window * 3) was fine. Keep that.
        self.z_window = int(z_window) if z_window is not None else max(30, self.adaptive_window * 3)

        # FIX: delta_smooth_window kept small (3) so sudden spikes are not
        # blurred into gradual-looking signal.
        self.delta_smooth_window = 3

        self.previous_lattice: Optional[ConceptLattice] = None
        self.delta_L_history: List[float] = []
        self.delta_L_filtered_history: List[float] = []
        self.similarity_history: List[float] = []
        self.raw_similarity_history: List[float] = []

        # signal_idx -> instance_id mapping (authoritative index bridge)
        self.signal_to_instance_id: List[int] = []
        self.instance_to_signal_idx: Dict[int, int] = {}
        self.drift_signal_indices: List[int] = []

        self._smoothed_similarity: Optional[float] = None

        # Stable-window buffer for recurring detection
        self._stable_lattice_buffer: List[tuple] = []
        self._stable_buffer_size = max(self.recurring_lookback * 3, 15)

        # Detection stabilizers
        self._over_threshold_run = 0
        self._cooldown_left = 0

        self.history_manager = LatticeHistoryManager()
        self.lattice_history = []
        self.concepts_count_history = []
        self.lattice_objects = []
        self.similarity_calc = LatticeSimilarityCalculator(min_support_ratio=0.1)
        self.prev_concepts_count = 0
        self.prev_levels = 0
        self.intent_snapshots = {}
        self.intent_diffs = {}
        self.snapshot_collector = LatticeSnapshotCollector()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, current_lattice: ConceptLattice, instance_id: int) -> Optional[DriftEvent]:
        raw_similarity = 1.0
        similarity = 1.0
        delta_L = 0.0
        adaptive_theta = self.theta
        drift_detected = False
        drift_type = "none"
        confidence = 0.0
        signal_idx: Optional[int] = None

        concepts_count = (
            len(current_lattice.concepts)
            if hasattr(current_lattice, "concepts") and current_lattice.concepts
            else 0
        )
        levels = (
            len(current_lattice.levels)
            if hasattr(current_lattice, "levels") and current_lattice.levels
            else 0
        )

        if self.previous_lattice is not None:
            raw_similarity = self.similarity_calc.compute_similarity(self.previous_lattice, current_lattice)
            self._smoothed_similarity = (
                raw_similarity
                if self._smoothed_similarity is None
                else self.similarity_ema_alpha * raw_similarity
                + (1.0 - self.similarity_ema_alpha) * self._smoothed_similarity
            )
            similarity = float(self._smoothed_similarity)
            delta_L = 1.0 - similarity

            self.raw_similarity_history.append(float(raw_similarity))
            self.similarity_history.append(float(similarity))
            self.delta_L_history.append(float(delta_L))

            signal_idx = len(self.delta_L_history) - 1
            self.signal_to_instance_id.append(instance_id)
            self.instance_to_signal_idx[instance_id] = signal_idx

            filtered_delta = self._compute_smoothed_delta(signal_idx)
            self.delta_L_filtered_history.append(float(filtered_delta))

            adaptive_theta = self._compute_adaptive_threshold()
            adaptive_baseline = self._compute_noise_baseline(signal_idx)
            point_stats = self._compute_point_statistics(signal_idx)

            is_above = filtered_delta > adaptive_theta
            above_baseline = filtered_delta > adaptive_baseline

            if is_above:
                self._over_threshold_run += 1
            else:
                self._over_threshold_run = 0

            # Spike: signal clearly above threshold
            strong_spike = filtered_delta > (adaptive_theta * self.spike_multiplier)

            # Persistent: signal above threshold for min_persistence_windows in a row
            persistent_signal = (
                self._over_threshold_run >= self.min_persistence_windows
                and above_baseline
            )

            in_cooldown = self._cooldown_left > 0
            if in_cooldown:
                self._cooldown_left -= 1

            drift_detected = (
                (strong_spike or persistent_signal)
                and (not in_cooldown or strong_spike)
            )

            if drift_detected:
                self._cooldown_left = self.cooldown_windows
                self._over_threshold_run = 0

                drift_type = self._classify_point_signal(signal_idx, point_stats, current_lattice)
                confidence = min(1.0, filtered_delta / max(adaptive_theta, 1e-6))
            elif not is_above:
                self._add_to_stable_buffer(instance_id, current_lattice)

        frame = LatticeFrame(
            instance_id=instance_id,
            window_number=len(self.history_manager),
            concepts_count=concepts_count,
            levels=levels,
            similarity=float(similarity),
            delta_L=float(delta_L),
            drift_detected=drift_detected,
            drift_type=drift_type,
            confidence=confidence,
            threshold_adaptive=float(adaptive_theta),
            concepts_diff=concepts_count - self.prev_concepts_count,
            levels_diff=levels - self.prev_levels,
            lattice_visual_html=self._generate_lattice_visual(current_lattice, drift_detected),
        )

        if drift_detected:
            frame.drift_status = (
                DriftStatusEnum.HARD_DRIFT
                if confidence > 0.8
                else DriftStatusEnum.SOFT_DRIFT
                if confidence > 0.5
                else DriftStatusEnum.VIRTUAL_DRIFT
            )

        self.history_manager.add_frame(frame)
        self.lattice_history.append(
            {
                "instance_id": instance_id,
                "concepts_count": concepts_count,
                "levels": levels,
                "similarity": float(similarity),
                "delta_L": float(delta_L),
                "is_drift": int(drift_detected),
                "lattice_visual": frame.lattice_visual_html,
                "signal_idx": signal_idx,
            }
        )
        self.lattice_objects.append(current_lattice)

        if self.previous_lattice is None:
            self.previous_lattice = current_lattice
            self.prev_concepts_count = concepts_count
            self.prev_levels = levels
            self._add_to_stable_buffer(instance_id, current_lattice)
            return None

        if drift_detected and signal_idx is not None:
            # Reporting index (instance_id)
            self.drift_indices.append(instance_id)
            # Computation index (signal_idx)
            self.drift_signal_indices.append(signal_idx)

            prev_intents = self.previous_lattice.get_intents()
            curr_intents = current_lattice.get_intents()

            self.intent_snapshots[signal_idx] = {
                "instance_id": instance_id,
                "before": prev_intents,
                "after": curr_intents,
            }
            self.intent_diffs[signal_idx] = {
                "instance_id": instance_id,
                "lost": prev_intents - curr_intents,
                "gained": curr_intents - prev_intents,
                "stable": prev_intents & curr_intents,
                "concept_count_before": len(self.previous_lattice.concepts),
                "concept_count_after": len(current_lattice.concepts),
            }

            self.snapshot_collector.record(
                instance_id=instance_id,
                intents_before=prev_intents,
                intents_after=curr_intents,
                delta_lt=float(delta_L),
                similarity=float(similarity),
            )

            point_stats = self._compute_point_statistics(signal_idx)
            event = DriftEvent(
                instance_id=instance_id,
                signal_idx=signal_idx,
                drift_type=drift_type,
                confidence=confidence,
                metadata={
                    "signal_idx": signal_idx,
                    "instance_id": instance_id,
                    "delta_L": float(delta_L),
                    "delta_L_filtered": float(filtered_delta),
                    "threshold": float(adaptive_theta),
                    "noise_baseline": float(adaptive_baseline),
                    "similarity": float(similarity),
                    "similarity_raw": float(raw_similarity),
                    "z_score": float(point_stats["z_current"]),
                },
            )
            self.drift_events.append(event)

            self.previous_lattice = current_lattice
            self.prev_concepts_count = concepts_count
            self.prev_levels = levels
            return event

        if delta_L <= adaptive_theta:
            self._add_to_stable_buffer(instance_id, current_lattice)

        self.previous_lattice = current_lattice
        self.prev_concepts_count = concepts_count
        self.prev_levels = levels
        return None

    # ------------------------------------------------------------------
    # Adaptive threshold and local signal stats
    # ------------------------------------------------------------------

    def _compute_adaptive_threshold(self) -> float:
        """Robust adaptive threshold with a soft floor.

        FIX: soft_floor is now tiny (0.01 * theta max 0.02) so that the
        adaptive threshold stays close to the actual signal baseline.
        A large soft_floor (old: 0.05 * theta which could be ~0.02) was
        fine, but after user edits it became too large and suppressed all
        detections. Keep it minimal.
        """
        source = self.delta_L_filtered_history if self.delta_L_filtered_history else self.delta_L_history
        soft_floor = max(1e-4, 0.01 * self.theta)
        if len(source) < self.adaptive_window:
            if not source:
                return soft_floor
            recent = np.array(source, dtype=float)
            mu = float(np.median(recent))
            mad = float(np.median(np.abs(recent - mu)))
            sigma = max(1.4826 * mad, float(np.std(recent)), 1e-6)
            return max(soft_floor, mu + self.alpha * sigma)

        recent = np.array(source[-self.adaptive_window:], dtype=float)
        mu = float(np.median(recent))
        mad = float(np.median(np.abs(recent - mu)))
        sigma = max(1.4826 * mad, float(np.std(recent)), 1e-6)
        return max(soft_floor, mu + self.alpha * sigma)

    def _compute_smoothed_delta(self, signal_idx: int) -> float:
        """Causal moving average of recent ΔL values to suppress micro-noise."""
        if signal_idx < 0 or signal_idx >= len(self.delta_L_history):
            return 0.0

        start = max(0, signal_idx - self.delta_smooth_window + 1)
        local = self.delta_L_history[start: signal_idx + 1]
        if not local:
            return 0.0
        return float(np.mean(local))

    def _compute_noise_baseline(self, signal_idx: int) -> float:
        """Adaptive local baseline used to suppress weak fluctuations."""
        if signal_idx < 0:
            return 0.0

        source = self.delta_L_filtered_history if self.delta_L_filtered_history else self.delta_L_history
        if not source:
            return 0.0

        end = min(signal_idx + 1, len(source))
        start = max(0, end - max(self.adaptive_window * 2, 10))
        local = np.array(source[start:end], dtype=float)

        if local.size == 0:
            return 0.0

        mu = float(np.median(local))
        mad = float(np.median(np.abs(local - mu)))
        sigma = max(1.4826 * mad, float(np.std(local)), 1e-6)
        return float(mu + self.noise_baseline_k * sigma)

    def _compute_point_statistics(self, signal_idx: int) -> Dict[str, float]:
        """Compute robust local statistics for a signal index."""
        source = self.delta_L_filtered_history if self.delta_L_filtered_history else self.delta_L_history
        if signal_idx < 0 or signal_idx >= len(source):
            return {
                "z_current": 0.0,
                "z_prev": 0.0,
                "sigma": 1.0,
                "rise": 0.0,
                "rise_over_sigma": 0.0,
            }

        start = max(0, signal_idx - self.z_window + 1)
        local = np.array(source[start: signal_idx + 1], dtype=float)

        mu = float(np.median(local))
        mad = float(np.median(np.abs(local - mu)))
        sigma = max(1.4826 * mad, float(np.std(local)), 1e-6)

        current = float(source[signal_idx])
        prev = float(source[signal_idx - 1]) if signal_idx > 0 else current
        z_current = (current - mu) / sigma
        z_prev = (prev - mu) / sigma
        rise = current - prev

        return {
            "z_current": float(z_current),
            "z_prev": float(z_prev),
            "sigma": float(sigma),
            "rise": float(rise),
            "rise_over_sigma": float(rise / sigma),
        }

    # ------------------------------------------------------------------
    # Stable buffer for recurring pattern check
    # ------------------------------------------------------------------

    def _add_to_stable_buffer(self, instance_id: int, lattice: ConceptLattice) -> None:
        self._stable_lattice_buffer.append((instance_id, lattice))
        if len(self._stable_lattice_buffer) > self._stable_buffer_size:
            self._stable_lattice_buffer.pop(0)

    def _check_recurring(self, current_lattice: ConceptLattice) -> bool:
        """Return True if current lattice resembles an older stable lattice."""
        if len(self._stable_lattice_buffer) < self.recurring_lookback + 2:
            return False

        older = self._stable_lattice_buffer[: -self.recurring_lookback]
        if not older:
            return False

        max_sim = max(
            self.similarity_calc.compute_similarity(stable_lattice, current_lattice)
            for _, stable_lattice in older
        )
        return max_sim >= self.recurring_similarity_threshold

    # ------------------------------------------------------------------
    # Point-level provisional type (final type is episode-level in aggregator)
    # ------------------------------------------------------------------

    def _classify_point_signal(
        self,
        signal_idx: int,
        point_stats: Dict[str, float],
        current_lattice: Optional[ConceptLattice] = None,
    ) -> str:
        """Classify a candidate point with conservative rules."""
        z_current = point_stats.get("z_current", 0.0)
        z_prev = point_stats.get("z_prev", 0.0)
        rise_over_sigma = point_stats.get("rise_over_sigma", 0.0)

        if current_lattice is not None and z_current < 1.0 and self._check_recurring(current_lattice):
            return "recurring"

        # Unified sudden criterion (2.5 sigma) with rise requirement.
        if z_current >= self.SUDDEN_Z_THRESHOLD and (z_current - z_prev) > 0.8 and rise_over_sigma > 0.6:
            return "sudden"

        if z_current >= 1.2:
            return "gradual"
        if z_current >= 0.8 and self._is_local_monotonic(signal_idx, span=min(10, self.z_window // 2)):
            return "incremental"
        return "unknown"

    def _is_local_monotonic(self, signal_idx: int, span: int = 10) -> bool:
        """Check monotonic increase tendency in local neighborhood."""
        start = max(0, signal_idx - span + 1)
        source = self.delta_L_filtered_history if self.delta_L_filtered_history else self.delta_L_history
        series = np.array(source[start: signal_idx + 1], dtype=float)
        if len(series) < 4:
            return False
        diffs = np.diff(series)
        non_negative_ratio = float(np.mean(diffs >= 0.0))
        slope = float(np.polyfit(np.arange(len(series)), series, 1)[0])
        return non_negative_ratio >= 0.75 and slope > 0.0

    # Backward-compatible wrapper name used in older code paths.
    def _classify_drift_type(self, current_lattice: Optional[ConceptLattice] = None) -> str:
        if not self.delta_L_history:
            return "unknown"
        idx = len(self.delta_L_history) - 1
        return self._classify_point_signal(idx, self._compute_point_statistics(idx), current_lattice)

    # ------------------------------------------------------------------
    # Visual helper
    # ------------------------------------------------------------------

    def _generate_lattice_visual(self, lattice: ConceptLattice, is_drift: bool) -> str:
        n = len(lattice.concepts) if hasattr(lattice, "concepts") and lattice.concepts else 0
        border = "2px solid #d32f2f;" if is_drift else "1px solid #ccc;"
        color = "#d32f2f" if is_drift else "#1976d2"
        html = (
            f'<div class="lattice-grid" style="display:grid;grid-template-columns:'
            f'repeat(4,1fr);gap:5px;padding:10px;background:#f5f5f5;'
            f'border-radius:4px;border:{border}">'
        )
        for i in range(16):
            cell = ("#" if is_drift else "*") if i < n else "-"
            html += (
                f'<div style="width:30px;height:30px;display:flex;align-items:center;'
                f'justify-content:center;background:white;border:1px solid {color};'
                f'border-radius:3px;font-weight:bold;color:{color};font-size:16px;">{cell}</div>'
            )
        html += "</div>"
        return html

    def reset(self):
        self.previous_lattice = None
        self.delta_L_history = []
        self.delta_L_filtered_history = []
        self.similarity_history = []
        self.raw_similarity_history = []
        self.signal_to_instance_id = []
        self.instance_to_signal_idx = {}

        self.lattice_history = []
        self.concepts_count_history = []
        self.lattice_objects = []
        self.drift_indices = []
        self.drift_signal_indices = []
        self.drift_events = []
        self.history_manager = LatticeHistoryManager()

        self.prev_concepts_count = 0
        self.prev_levels = 0
        self.intent_snapshots = {}
        self.intent_diffs = {}
        self.snapshot_collector = LatticeSnapshotCollector()
        self._stable_lattice_buffer = []
        self._over_threshold_run = 0
        self._cooldown_left = 0
        self._smoothed_similarity = None


if __name__ == "__main__":
    print("FCA Drift Detector loaded")
