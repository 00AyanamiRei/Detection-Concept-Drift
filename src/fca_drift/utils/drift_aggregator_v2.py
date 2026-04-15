"""
Comprehensive Drift Aggregation & Classification System (V2).

Core principles:
- Internal computations are always performed on signal_idx.
- Reporting uses instance_id mapped from signal_idx.
- Episodes are formed first, then classified by episode shape.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DriftEpisode:
    """Represents one merged drift episode."""

    # Signal coordinates (authoritative for computations)
    start: int
    end: int
    center: int

    # Stream coordinates (for reporting only)
    start_instance_id: int
    end_instance_id: int
    center_instance_id: int

    dlt_max: float
    dlt_mean: float
    sim_min: float
    sim_mean: float
    raw_hits_count: int
    dominant_type: str
    type_votes: Dict[str, int]
    z_scores: List[float]

    lost_intents: Optional[set] = None
    gained_intents: Optional[set] = None
    stable_intents: Optional[set] = None
    concept_count_before: int = 0
    concept_count_after: int = 0

    def to_tuple(self) -> Tuple[int, int, str]:
        """Signal-coordinate tuple used by plotting code."""
        return (self.start, self.end, self.dominant_type)

    def episode_length(self) -> int:
        return self.end - self.start + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "center": self.center,
            "start_instance_id": self.start_instance_id,
            "end_instance_id": self.end_instance_id,
            "center_instance_id": self.center_instance_id,
            "length": self.episode_length(),
            "dlt_max": float(self.dlt_max),
            "dlt_mean": float(self.dlt_mean),
            "sim_min": float(self.sim_min),
            "sim_mean": float(self.sim_mean),
            "raw_hits": self.raw_hits_count,
            "dominant_type": self.dominant_type,
            "type_votes": self.type_votes,
            "lost_intents": len(self.lost_intents) if self.lost_intents else 0,
            "gained_intents": len(self.gained_intents) if self.gained_intents else 0,
            "stable_intents": len(self.stable_intents) if self.stable_intents else 0,
            "concept_count_before": self.concept_count_before,
            "concept_count_after": self.concept_count_after,
        }


_TYPE_PRIORITY = {
    "sudden": 5,
    "recurring": 4,
    "gradual": 3,
    "incremental": 2,
    "unknown": 1,
}


class DriftAggregatorV2:
    """Single source of truth for episode-level drift analysis."""

    SUDDEN_Z_THRESHOLD = 2.5
    SIGNAL_SMOOTH_WINDOW = 5
    LOCALITY_MIN_POINTS = 3
    LATE_GAP_FACTOR = 6

    def __init__(
        self,
        delta_L_history: List[float],
        similarity_history: List[float],
        window_size: int = 50,
        warm_up_windows: int = 5,
        alpha: float = 2.0,
        n_adapt: int = 10,
        merge_gap: Optional[int] = None,
        cooldown: Optional[int] = None,
        intent_diffs: Optional[Dict[int, Dict]] = None,
        drift_types_raw: Optional[Dict[int, str]] = None,
        signal_to_instance_id: Optional[List[int]] = None,
        drift_signal_indices: Optional[List[int]] = None,
        # Backward-compatible alias: interpreted as signal indices.
        drift_indices: Optional[List[int]] = None,
        dominant_type_mode: str = "weighted",
        signal_smoothing_window: Optional[int] = None,
    ):
        self.delta_L = np.array(delta_L_history, dtype=float)
        self.similarity = np.array(similarity_history, dtype=float)
        self.n_total = len(self.delta_L)

        self.window_size = int(window_size)
        self.warm_up_windows = int(warm_up_windows)
        self.warm_up_threshold = self.warm_up_windows * self.window_size
        self.alpha = float(alpha)
        self.n_adapt = int(n_adapt)
        self.z_window = max(self.window_size * 2, self.n_adapt * 3, 30)

        mode = str(dominant_type_mode).strip().lower()
        if mode not in {"weighted", "count"}:
            mode = "weighted"
        self.dominant_type_mode = mode

        if signal_smoothing_window is None:
            self.signal_smooth_window = max(3, min(self.SIGNAL_SMOOTH_WINDOW, max(self.window_size, 3)))
        else:
            self.signal_smooth_window = max(1, int(signal_smoothing_window))

        raw_indices = drift_signal_indices if drift_signal_indices is not None else drift_indices
        self.drift_signal_indices = sorted({int(i) for i in (raw_indices or [])})

        self.drift_types_raw = {
            int(k): str(v).strip().lower() for k, v in (drift_types_raw or {}).items()
        }

        self.intent_diffs = intent_diffs or {}

        default_merge_gap = max(self.window_size * 2, 1)
        merge_gap_value = int(merge_gap) if merge_gap is not None else default_merge_gap
        self.merge_gap = max(merge_gap_value, 1)
        cooldown_value = int(cooldown) if cooldown is not None else 0
        # Enforce cooldown < merge_gap as requested.
        self.cooldown = max(0, min(cooldown_value, self.merge_gap - 1))

        self.signal_to_instance_id = list(signal_to_instance_id or [])
        if len(self.signal_to_instance_id) < self.n_total:
            self.signal_to_instance_id.extend(list(range(len(self.signal_to_instance_id), self.n_total)))

        self._merged_episodes: Optional[List[DriftEpisode]] = None
        self._diagnostics: Optional[Dict[str, Any]] = None
        self._actual_warm_up_threshold: Optional[int] = None

        self.delta_L_smooth = self._causal_moving_average(self.delta_L, self.signal_smooth_window)
        self._global_mu = float(np.median(self.delta_L_smooth)) if self.n_total > 0 else 0.0
        mad = float(np.median(np.abs(self.delta_L_smooth - self._global_mu))) if self.n_total > 0 else 0.0
        self._global_sigma = max(
            1.4826 * mad,
            float(np.std(self.delta_L_smooth)) if self.n_total > 0 else 0.0,
            1e-6,
        )
        self._high_peak_threshold = (
            float(np.quantile(self.delta_L_smooth, 0.90)) if self.n_total > 0 else 0.0
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_merged_episodes(self) -> List[DriftEpisode]:
        if self._merged_episodes is None:
            self._compute_merged_episodes()
        return self._merged_episodes

    def get_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ep in self.get_merged_episodes():
            counts[ep.dominant_type] = counts.get(ep.dominant_type, 0) + 1
        return counts

    def get_weighted_type(self) -> str:
        episodes = self.get_merged_episodes()
        if not episodes:
            return "none"

        weighted: Dict[str, float] = {}
        for ep in episodes:
            weight = max(ep.dlt_max, 0.01)
            boost = _TYPE_PRIORITY.get(ep.dominant_type, 1)
            weighted[ep.dominant_type] = weighted.get(ep.dominant_type, 0.0) + weight * boost

        return max(weighted.items(), key=lambda x: x[1])[0]

    def get_majority_type(self) -> str:
        counts = self.get_type_counts()
        if not counts:
            return "none"
        return max(counts.items(), key=lambda x: x[1])[0]

    def get_dominant_type(self) -> str:
        if self.dominant_type_mode == "count":
            return self.get_majority_type()
        return self.get_weighted_type()

    def get_drift_rate(self) -> float:
        episodes = self.get_merged_episodes()
        return 100.0 * len(episodes) / self.n_total if self.n_total > 0 else 0.0

    def get_diagnostics(self) -> Dict[str, Any]:
        episodes = self.get_merged_episodes()
        actual = self._actual_warm_up_threshold if self._actual_warm_up_threshold is not None else self.warm_up_threshold
        raw_filtered = sum(1 for i in self.drift_signal_indices if i >= actual)

        diag = {
            "total_instances": self.n_total,
            "warm_up_instances": actual,
            "warm_up_percent": 100.0 * actual / max(self.n_total, 1),
            "raw_drift_count": raw_filtered,
            "merged_episodes_count": len(episodes),
            "drift_rate_percent": self.get_drift_rate(),
            "counts_by_type": self.get_type_counts(),
            "dominant_type_mode": self.dominant_type_mode,
            "dominant_type_majority": self.get_majority_type(),
            "dominant_type_weighted": self.get_weighted_type(),
            "dominant_type": self.get_dominant_type(),
            "parameters": {
                "window_size": self.window_size,
                "warm_up_windows": self.warm_up_windows,
                "alpha": self.alpha,
                "n_adapt": self.n_adapt,
                "smoothing_window": self.signal_smooth_window,
                "merge_gap": self.merge_gap,
                "cooldown": self.cooldown,
                "locality_min_points": self.LOCALITY_MIN_POINTS,
                "late_gap_factor": self.LATE_GAP_FACTOR,
                "sudden_z_threshold": self.SUDDEN_Z_THRESHOLD,
            },
            "episodes": [ep.to_dict() for ep in episodes],
        }
        self._diagnostics = diag
        return diag

    def save_diagnostics(self, output_path: Path, filename: str = "report_debug.json") -> Path:
        diag = self.get_diagnostics()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        out = output_path / filename
        with open(out, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, ensure_ascii=False)
        return out

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _compute_merged_episodes(self) -> None:
        min_ratio = max(1, int(0.1 * self.n_total))
        adaptive_wu = min(self.warm_up_threshold, min_ratio)
        self._actual_warm_up_threshold = adaptive_wu

        filtered = [
            idx
            for idx in self.drift_signal_indices
            if adaptive_wu <= idx < self.n_total
        ]

        filtered = self._filter_micro_noise_indices(filtered)

        if not filtered:
            self._merged_episodes = []
            return

        # 1) Merge by temporal distance first (required).
        groups = self._merge_by_distance(filtered)
        # 2) Keep only dense/strong local clusters, suppress isolated late noise.
        groups = self._filter_clusters_by_locality(groups)
        if not groups:
            self._merged_episodes = []
            return
        # 2) Apply cooldown suppression only after merge (required).
        groups = self._apply_cooldown_after_merge(groups)

        self._merged_episodes = [self._build_episode(group) for group in groups]

    def _merge_by_distance(self, indices: List[int], gap: Optional[int] = None) -> List[List[int]]:
        if not indices:
            return []

        gap_value = max(int(gap) if gap is not None else self.merge_gap, 1)

        sorted_idx = sorted(indices)
        groups: List[List[int]] = [[sorted_idx[0]]]

        for idx in sorted_idx[1:]:
            if idx - groups[-1][-1] <= gap_value:
                groups[-1].append(idx)
            else:
                groups.append([idx])

        return groups

    def _filter_micro_noise_indices(self, indices: List[int]) -> List[int]:
        """Filter raw drift hits using smoothed ΔL against adaptive local baseline."""
        if not indices:
            return []

        filtered: List[int] = []
        baseline_window = max(self.window_size, self.signal_smooth_window * 3)
        sigma_floor = max(self._global_sigma * 0.15, 1e-6)

        for idx in indices:
            start = max(0, idx - baseline_window + 1)
            local = self.delta_L_smooth[start : idx + 1]
            if len(local) == 0:
                continue

            mu = float(np.median(local))
            mad = float(np.median(np.abs(local - mu)))
            sigma = max(1.4826 * mad, float(np.std(local)), sigma_floor)
            adaptive_baseline = mu + 0.35 * sigma

            if float(self.delta_L_smooth[idx]) >= adaptive_baseline:
                filtered.append(idx)

        return filtered

    def _cluster_stats(self, indices: List[int]) -> Dict[str, Any]:
        idx_sorted = sorted(indices)
        start = idx_sorted[0]
        end = idx_sorted[-1]
        duration = max(end - start + 1, 1)
        count = len(idx_sorted)

        segment = self.delta_L_smooth[start : end + 1]
        if len(segment) == 0:
            segment = np.array([0.0], dtype=float)

        peak = float(np.max(segment))
        peak_offset = int(np.argmax(segment))
        peak_idx = start + peak_offset
        peak_z = self._compute_robust_z(peak_idx)

        density = float(count / duration)
        compactness = float(duration / max(count, 1))
        score = max(peak, 1e-6) * max(peak_z, 0.1) * max(density, 0.05)

        return {
            "indices": idx_sorted,
            "start": start,
            "end": end,
            "center": int(round(float(np.mean(idx_sorted)))),
            "count": count,
            "duration": duration,
            "density": density,
            "compactness": compactness,
            "peak": peak,
            "peak_z": peak_z,
            "score": score,
        }

    def _filter_clusters_by_locality(self, groups: List[List[int]]) -> List[List[int]]:
        """Retain only meaningful local drift clusters and penalize very late isolated groups."""
        if not groups:
            return []

        stats = [self._cluster_stats(g) for g in groups if g]
        if not stats:
            return []

        strong_peak_floor = max(self._high_peak_threshold, self._global_mu + 1.5 * self._global_sigma)

        candidates: List[Dict[str, Any]] = []
        for cluster in stats:
            enough_points = cluster["count"] >= self.LOCALITY_MIN_POINTS
            strong_peak = (
                cluster["peak"] >= strong_peak_floor
                or cluster["peak_z"] >= (self.SUDDEN_Z_THRESHOLD - 0.2)
            )

            if enough_points or strong_peak:
                cluster["strong_peak"] = strong_peak
                candidates.append(cluster)

        if not candidates:
            best = max(stats, key=lambda c: c["score"])
            return [best["indices"]]

        top_score = max(cluster["score"] for cluster in candidates)
        score_floor = max(0.20 * top_score, 1e-9)
        candidates = [
            cluster
            for cluster in candidates
            if cluster["score"] >= score_floor or cluster.get("strong_peak", False)
        ]

        accepted: List[Dict[str, Any]] = []
        previous_center: Optional[int] = None
        late_gap = max(self.window_size * self.LATE_GAP_FACTOR, self.merge_gap * 3)

        for cluster in sorted(candidates, key=lambda c: c["center"]):
            is_late = previous_center is not None and (cluster["center"] - previous_center) > late_gap
            if is_late:
                late_peak_ok = (
                    cluster.get("strong_peak", False)
                    and cluster["peak_z"] >= (self.SUDDEN_Z_THRESHOLD + 0.35)
                )
                late_dense_ok = (
                    cluster["count"] >= (self.LOCALITY_MIN_POINTS + 1)
                    and cluster["density"] >= 0.40
                )
                if not (late_peak_ok or late_dense_ok):
                    continue

            accepted.append(cluster)
            previous_center = cluster["center"]

        if not accepted:
            best = max(candidates, key=lambda c: c["score"])
            accepted = [best]

        return [cluster["indices"] for cluster in accepted]

    def _apply_cooldown_after_merge(self, groups: List[List[int]]) -> List[List[int]]:
        if not groups or self.cooldown <= 0:
            return groups

        merged: List[List[int]] = [list(groups[0])]
        for group in groups[1:]:
            prev_end = merged[-1][-1]
            curr_start = group[0]
            if (curr_start - prev_end) <= self.cooldown:
                merged[-1].extend(group)
            else:
                merged.append(list(group))

        return merged

    def _build_episode(self, signal_indices: List[int]) -> DriftEpisode:
        signal_indices = sorted(signal_indices)
        start = signal_indices[0]
        end = signal_indices[-1]

        dlt_vals = self.delta_L[start : end + 1] if start <= end else np.array([0.0], dtype=float)
        sim_vals = self.similarity[start : end + 1] if start <= end else np.array([1.0], dtype=float)

        if len(dlt_vals) == 0:
            dlt_vals = np.array([0.0], dtype=float)
        if len(sim_vals) == 0:
            sim_vals = np.array([1.0], dtype=float)

        dlt_sum = float(np.sum(dlt_vals))
        if dlt_sum > 1e-12:
            center = int(np.average(np.arange(start, end + 1), weights=dlt_vals / dlt_sum))
        else:
            center = (start + end) // 2

        z_scores = [self._compute_robust_z(i) for i in signal_indices]

        vote_map: Dict[str, int] = {}
        for idx in signal_indices:
            raw_t = self.drift_types_raw.get(idx, "unknown")
            vote_map[raw_t] = vote_map.get(raw_t, 0) + 1

        dominant = self._classify_episode(signal_indices, z_scores, vote_map)

        lost_i, gained_i, stable_i = set(), set(), set()
        cb_before = cb_after = 0
        for idx in signal_indices:
            if idx in self.intent_diffs:
                diff = self.intent_diffs[idx]
                lost_i.update(diff.get("lost", set()))
                gained_i.update(diff.get("gained", set()))
                stable_i.update(diff.get("stable", set()))
                cb_before = max(cb_before, int(diff.get("concept_count_before", 0)))
                cb_after = max(cb_after, int(diff.get("concept_count_after", 0)))

        return DriftEpisode(
            start=start,
            end=end,
            center=center,
            start_instance_id=self._to_instance_id(start),
            end_instance_id=self._to_instance_id(end),
            center_instance_id=self._to_instance_id(center),
            dlt_max=float(np.max(dlt_vals)),
            dlt_mean=float(np.mean(dlt_vals)),
            sim_min=float(np.min(sim_vals)),
            sim_mean=float(np.mean(sim_vals)),
            raw_hits_count=len(signal_indices),
            dominant_type=dominant,
            type_votes=vote_map,
            z_scores=z_scores,
            lost_intents=lost_i or None,
            gained_intents=gained_i or None,
            stable_intents=stable_i or None,
            concept_count_before=cb_before,
            concept_count_after=cb_after,
        )

    # ------------------------------------------------------------------
    # Episode classification
    # ------------------------------------------------------------------

    def _classify_episode(self, indices: List[int], z_scores: List[float], vote_map: Dict[str, int]) -> str:
        """Classify an already merged episode by signal shape."""
        if not indices:
            return "unknown"

        start, end = indices[0], indices[-1]
        segment = self.delta_L_smooth[start : end + 1]
        if len(segment) == 0:
            return "unknown"

        peak = float(np.max(segment))
        peak_pos = int(np.argmax(segment))
        peak_idx = start + peak_pos
        peak_z = float(max(z_scores) if z_scores else self._compute_robust_z(peak_idx))
        duration = end - start + 1
        hit_count = len(indices)
        compactness = float(duration / max(hit_count, 1))
        hit_density = float(hit_count / max(duration, 1))

        if len(segment) > 1:
            diffs = np.diff(segment)
            max_rise = float(np.max(diffs))
            max_drop = float(np.min(diffs))
            slope = float(np.polyfit(np.arange(len(segment)), segment, 1)[0])
            monotonic_ratio = float(np.mean(diffs >= 0.0))
            rise_sigma = max(float(np.std(segment)), self._global_sigma * 0.25, 1e-6)
            rise_over_sigma = float(max_rise / rise_sigma)
        else:
            max_rise = 0.0
            max_drop = 0.0
            slope = 0.0
            monotonic_ratio = 0.0
            rise_over_sigma = 0.0

        # Plateau behavior: substantial fraction near the local peak.
        if peak > 0:
            plateau_ratio = float(np.mean(segment >= (0.75 * peak)))
        else:
            plateau_ratio = 0.0

        high_level = max(self._global_mu + 0.75 * self._global_sigma, 0.6 * peak)
        active_duration = int(np.sum(segment >= high_level))
        active_duration = max(active_duration, 1)
        effective_duration = min(duration, active_duration + self.signal_smooth_window)

        recurring_votes = vote_map.get("recurring", 0)
        if recurring_votes > 0 and recurring_votes / max(len(indices), 1) >= 0.6 and peak_z < 1.5:
            return "recurring"

        sudden_peak_floor = max(self._high_peak_threshold, self._global_mu + 1.2 * self._global_sigma)
        sudden_duration_limit = max(self.window_size, 6)

        if (
            peak_z >= (self.SUDDEN_Z_THRESHOLD - 0.1)
            and peak >= sudden_peak_floor
            and effective_duration <= sudden_duration_limit
            and rise_over_sigma >= 0.8
            and (compactness <= 12.0 or hit_density >= 0.12)
        ):
            return "sudden"

        # Incremental: long smooth trend with sustained monotonic rise and no sharp spike.
        if (
            duration >= max(self.window_size, 8)
            and monotonic_ratio >= 0.75
            and slope > 0
            and rise_over_sigma < 1.2
            and peak_z < (self.SUDDEN_Z_THRESHOLD - 0.2)
        ):
            return "incremental"

        # Gradual: sustained elevated phase / plateau without abrupt jump.
        if (
            duration >= max(self.window_size, 6)
            and (plateau_ratio >= 0.30 or active_duration >= max(self.window_size // 3, 4))
            and peak_z >= 1.0
            and rise_over_sigma < 1.8
        ):
            return "gradual"

        # Ambiguous cases should stay unknown rather than forced incremental.
        if peak_z >= 1.0 and max_drop > -0.5 * max(peak, 1e-6):
            return "gradual"

        return "unknown"

    @staticmethod
    def _causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
        """Causal moving average that preserves stream order and output length."""
        if values.size == 0:
            return values.astype(float)

        w = max(int(window), 1)
        kernel = np.ones(w, dtype=float)
        smoothed = np.convolve(values, kernel, mode="full")[: values.size]
        counts = np.minimum(np.arange(1, values.size + 1), w).astype(float)
        return smoothed / counts

    def _compute_robust_z(self, idx: int) -> float:
        if idx < 0 or idx >= self.n_total:
            return 0.0

        start = max(0, idx - self.z_window)
        end = min(self.n_total, idx + self.z_window + 1)
        window = self.delta_L_smooth[start:end]

        if len(window) < 3:
            return float((self.delta_L_smooth[idx] - self._global_mu) / self._global_sigma)

        mu = float(np.median(window))
        mad = float(np.median(np.abs(window - mu)))
        sigma = max(1.4826 * mad, self._global_sigma * 0.25, 1e-6)
        return float((self.delta_L_smooth[idx] - mu) / sigma)

    def _to_instance_id(self, signal_idx: int) -> int:
        if 0 <= signal_idx < len(self.signal_to_instance_id):
            return int(self.signal_to_instance_id[signal_idx])
        return int(signal_idx)
