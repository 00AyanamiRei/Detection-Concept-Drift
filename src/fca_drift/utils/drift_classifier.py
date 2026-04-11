# """
# Advanced Drift Type Classification System
# Classifies drifts based on Z-score amplitude and run-length (temporal pattern)
# Ensures consistency across all visualizations and reports
# """

# import numpy as np
# from typing import Dict, List, Tuple, Optional
# from scipy import stats


# class DriftTypeClassifier:
#     """
#     Robust drift type classification using:
#     - Z-score amplitude classification (sudden vs gradual)
#     - Run-length analysis (single peak vs sustained trend)
#     - Context-aware thresholding
#     """

#     # Z-score thresholds (robust using median absolute deviation)
#     Z_THRESHOLD_SUDDEN = 2.5      # Very high amplitude
#     Z_THRESHOLD_GRADUAL = 1.5     # Moderate amplitude
#     Z_THRESHOLD_INCREMENTAL = 0.8 # Low amplitude

#     @staticmethod
#     def classify_drift(
#         delta_L_history: List[float],
#         drift_indices: List[int],
#         drift_types_raw: Dict[int, str],
#         window_size: int = 50,
#         merge_gap: int = 5
#     ) -> Tuple[Dict[int, str], List[Tuple[int, int, str]]]:
#         """
#         Classify drifts and merge nearby events.

#         Args:
#             delta_L_history: Full ΔL time series
#             drift_indices: Original drift detection indices
#             drift_types_raw: Raw types from detector
#             window_size: Size of sliding window
#             merge_gap: Maximum gap between instances to consider as same event

#         Returns:
#             - Tuple (drift_types_classified, merged_episodes)
#             - merged_episodes: List of (start, end, dominant_type)
#         """

#         if not drift_indices:
#             return {}, []

#         # Step 1: Compute Z-scores with robust statistics
#         z_scores = DriftTypeClassifier._compute_robust_z_scores(delta_L_history)

#         # Step 2: Classify each drift by amplitude and context
#         drift_types_classified = {}
#         for idx in drift_indices:
#             z_score = z_scores[idx] if idx < len(z_scores) else 0
#             amplitude = delta_L_history[idx] if idx < len(delta_L_history) else 0

#             # Classify based on Z-score
#             if z_score >= DriftTypeClassifier.Z_THRESHOLD_SUDDEN:
#                 classified_type = 'sudden'
#             elif z_score >= DriftTypeClassifier.Z_THRESHOLD_GRADUAL:
#                 classified_type = 'gradual'
#             elif z_score >= DriftTypeClassifier.Z_THRESHOLD_INCREMENTAL:
#                 classified_type = 'incremental'
#             else:
#                 classified_type = 'unknown'

#             drift_types_classified[idx] = classified_type

#         # Step 3: Merge nearby drifts into episodes
#         merged_episodes = DriftTypeClassifier._merge_drift_episodes(
#             drift_indices,
#             drift_types_classified,
#             merge_gap=merge_gap
#         )

#         # Step 4: Refine types based on temporal pattern within episodes
#         for start, end, drift_type in merged_episodes:
#             episode_indices = [i for i in drift_indices if start <= i <= end]
#             if len(episode_indices) > 1:
#                 # Multiple points → more likely gradual/incremental
#                 refined_type = DriftTypeClassifier._refine_type_by_pattern(
#                     delta_L_history,
#                     episode_indices,
#                     current_type=drift_type
#                 )
#                 # Update all indices in episode
#                 for idx in episode_indices:
#                     drift_types_classified[idx] = refined_type

#         return drift_types_classified, merged_episodes

#     @staticmethod
#     def _compute_robust_z_scores(data: List[float]) -> List[float]:
#         """
#         Compute robust Z-scores using median absolute deviation (MAD).
#         Less sensitive to outliers than standard deviation.
#         """
#         data = np.array(data)
#         median = np.median(data)
#         mad = np.median(np.abs(data - median))

#         # Avoid division by zero
#         if mad < 1e-10:
#             mad = np.std(data) if np.std(data) > 0 else 1.0

#         # Compute modified Z-scores
#         z_scores = 0.6745 * (data - median) / mad
#         return z_scores.tolist()

#     @staticmethod
#     def _merge_drift_episodes(
#         drift_indices: List[int],
#         drift_types: Dict[int, str],
#         merge_gap: int = 5
#     ) -> List[Tuple[int, int, str]]:
#         """
#         Merge drifts that are within merge_gap of each other.
#         Returns list of episodes: (start_idx, end_idx, dominant_type)
#         """
#         if not drift_indices:
#             return []

#         episodes = []
#         sorted_indices = sorted(drift_indices)

#         current_start = sorted_indices[0]
#         current_end = sorted_indices[0]
#         episode_types = [drift_types.get(sorted_indices[0], 'unknown')]

#         for idx in sorted_indices[1:]:
#             if idx - current_end <= merge_gap:
#                 # Extend current episode
#                 current_end = idx
#                 episode_types.append(drift_types.get(idx, 'unknown'))
#             else:
#                 # End current episode, start new
#                 dominant_type = DriftTypeClassifier._get_dominant_type(episode_types)
#                 episodes.append((current_start, current_end, dominant_type))

#                 current_start = idx
#                 current_end = idx
#                 episode_types = [drift_types.get(idx, 'unknown')]

#         # Add last episode
#         if episode_types:
#             dominant_type = DriftTypeClassifier._get_dominant_type(episode_types)
#             episodes.append((current_start, current_end, dominant_type))

#         return episodes

#     @staticmethod
#     def _get_dominant_type(types: List[str]) -> str:
#         """Get dominant drift type from a list of types."""
#         type_counts = {}
#         for t in types:
#             type_counts[t] = type_counts.get(t, 0) + 1

#         # Priority: sudden > gradual > incremental > unknown
#         priority = {'sudden': 3, 'gradual': 2, 'incremental': 1, 'unknown': 0}

#         dominant = max(
#             type_counts.items(),
#             key=lambda x: (priority.get(x[0], 0), x[1])
#         )
#         return dominant[0]

#     @staticmethod
#     def _refine_type_by_pattern(
#         delta_L_history: List[float],
#         episode_indices: List[int],
#         current_type: str
#     ) -> str:
#         """
#         Refine drift type based on temporal pattern.
#         - Single sharp peak → sudden
#         - Multiple peaks with trend → gradual
#         - Monotonic low change → incremental
#         """
#         if len(episode_indices) == 1:
#             return 'sudden'

#         # Analyze values and differences
#         values = [delta_L_history[i] for i in sorted(episode_indices)]

#         # Check for monotonic trend
#         is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
#         is_decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
#         is_monotonic = is_increasing or is_decreasing

#         # Compute coefficient of variation
#         mean_val = np.mean(values)
#         std_val = np.std(values)
#         cv = std_val / mean_val if mean_val > 0 else 0

#         peak_value = max(values)
#         avg_value = np.mean(values)

#         # Classification logic
#         if is_monotonic and cv < 0.3:
#             # Stable trend with small variance
#             return 'incremental'
#         elif len(episode_indices) >= 3 and cv >= 0.3:
#             # Multiple points with significant variation
#             return 'gradual'
#         elif peak_value > 2 * avg_value and len(episode_indices) <= 2:
#             # Sharp isolated peak(s)
#             return 'sudden'
#         else:
#             return current_type

#     @staticmethod
#     def get_statistics(
#         merged_episodes: List[Tuple[int, int, str]],
#         total_instances: int
#     ) -> Dict:
#         """Compute statistics from merged episodes."""
#         if not merged_episodes:
#             return {
#                 'total_episodes': 0,
#                 'drift_rate': 0.0,
#                 'episode_lengths': [],
#                 'type_distribution': {},
#                 'dominant_type': 'no_drift'
#             }

#         episode_lengths = [end - start + 1 for start, end, _ in merged_episodes]
#         type_dist = {}

#         for _, _, dtype in merged_episodes:
#             type_dist[dtype] = type_dist.get(dtype, 0) + 1

#         # Dominant type by frequency
#         dominant = max(type_dist.items(), key=lambda x: x[1])[0] if type_dist else 'unknown'

#         drift_rate = len(merged_episodes) / max(total_instances, 1) * 100

#         return {
#             'total_episodes': len(merged_episodes),
#             'drift_rate': drift_rate,
#             'episode_lengths': episode_lengths,
#             'type_distribution': type_dist,
#             'dominant_type': dominant,
#             'avg_episode_length': np.mean(episode_lengths) if episode_lengths else 0
#         }

"""
Advanced Drift Type Classification System
Classifies drifts based on Z-score amplitude, run-length (temporal pattern),
and recurring pattern detection.

FIXED v2:
- Added 'recurring' type support
- Weighted priority: sudden > recurring > gradual > incremental > unknown
- _refine_type_by_pattern: no longer collapses everything to incremental
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# Consistent priority weights used across the whole module
_TYPE_PRIORITY = {
    'sudden':      5,
    'recurring':   4,
    'gradual':     3,
    'incremental': 2,
    'unknown':     1,
}


class DriftTypeClassifier:
    """
    Robust drift type classification using:
    - Z-score amplitude (sudden vs gradual vs incremental)
    - Run-length analysis (single peak vs sustained trend)
    - Recurring pattern detection (passed in from detector)
    - Context-aware thresholding
    """

    Z_THRESHOLD_SUDDEN      = 2.5
    Z_THRESHOLD_GRADUAL     = 1.5
    Z_THRESHOLD_INCREMENTAL = 0.8

    @staticmethod
    def classify_drift(
        delta_L_history: List[float],
        drift_indices: List[int],
        drift_types_raw: Dict[int, str],
        window_size: int = 50,
        merge_gap: int = 5,
    ) -> Tuple[Dict[int, str], List[Tuple[int, int, str]]]:
        """
        Classify drifts and merge nearby events.

        Args:
            delta_L_history:  Full ΔL time series
            drift_indices:    Original drift detection indices
            drift_types_raw:  Raw types from detector (may include 'recurring')
            window_size:      Size of sliding window
            merge_gap:        Max gap to consider same event

        Returns:
            (drift_types_classified, merged_episodes)
            merged_episodes: [(start, end, dominant_type), ...]
        """
        if not drift_indices:
            return {}, []

        # Step 1: Robust Z-scores over full series
        z_scores = DriftTypeClassifier._compute_robust_z_scores(delta_L_history)

        # Step 2: Classify each index
        drift_types_classified: Dict[int, str] = {}
        for idx in drift_indices:
            # Honour 'recurring' from the detector — it has lattice access, we don't
            if drift_types_raw.get(idx) == 'recurring':
                drift_types_classified[idx] = 'recurring'
                continue

            z = z_scores[idx] if idx < len(z_scores) else 0.0

            if z >= DriftTypeClassifier.Z_THRESHOLD_SUDDEN:
                classified = 'sudden'
            elif z >= DriftTypeClassifier.Z_THRESHOLD_GRADUAL:
                classified = 'gradual'
            elif z >= DriftTypeClassifier.Z_THRESHOLD_INCREMENTAL:
                classified = 'incremental'
            else:
                classified = 'unknown'

            drift_types_classified[idx] = classified

        # Step 3: Merge into episodes
        merged_episodes = DriftTypeClassifier._merge_drift_episodes(
            drift_indices, drift_types_classified, merge_gap=merge_gap
        )

        # Step 4: Refine multi-point episodes
        for start, end, _ in merged_episodes:
            episode_indices = [i for i in drift_indices if start <= i <= end]
            if len(episode_indices) > 1:
                # Don't override a recurring episode
                current_types = [drift_types_classified.get(i, 'unknown') for i in episode_indices]
                if 'recurring' not in current_types:
                    refined = DriftTypeClassifier._refine_type_by_pattern(
                        delta_L_history, episode_indices,
                        current_type=DriftTypeClassifier._get_dominant_type(current_types)
                    )
                    for i in episode_indices:
                        drift_types_classified[i] = refined

        # Rebuild episodes after refinement
        merged_episodes = DriftTypeClassifier._merge_drift_episodes(
            drift_indices, drift_types_classified, merge_gap=merge_gap
        )

        return drift_types_classified, merged_episodes

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_robust_z_scores(data: List[float]) -> List[float]:
        """MAD-based robust Z-scores."""
        arr = np.array(data)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad < 1e-10:
            mad = np.std(arr) if np.std(arr) > 0 else 1.0
        return (0.6745 * (arr - median) / mad).tolist()

    @staticmethod
    def _merge_drift_episodes(
        drift_indices: List[int],
        drift_types: Dict[int, str],
        merge_gap: int = 5,
    ) -> List[Tuple[int, int, str]]:
        if not drift_indices:
            return []

        episodes: List[Tuple[int, int, str]] = []
        sorted_idx = sorted(drift_indices)
        cur_start = cur_end = sorted_idx[0]
        ep_types = [drift_types.get(sorted_idx[0], 'unknown')]

        for idx in sorted_idx[1:]:
            if idx - cur_end <= merge_gap:
                cur_end = idx
                ep_types.append(drift_types.get(idx, 'unknown'))
            else:
                episodes.append((cur_start, cur_end,
                                  DriftTypeClassifier._get_dominant_type(ep_types)))
                cur_start = cur_end = idx
                ep_types = [drift_types.get(idx, 'unknown')]

        if ep_types:
            episodes.append((cur_start, cur_end,
                              DriftTypeClassifier._get_dominant_type(ep_types)))

        return episodes

    @staticmethod
    def _get_dominant_type(types: List[str]) -> str:
        counts: Dict[str, int] = {}
        for t in types:
            counts[t] = counts.get(t, 0) + 1
        # Score = count * priority
        scored = {t: cnt * _TYPE_PRIORITY.get(t, 1) for t, cnt in counts.items()}
        return max(scored.items(), key=lambda x: x[1])[0]

    @staticmethod
    def _refine_type_by_pattern(
        delta_L_history: List[float],
        episode_indices: List[int],
        current_type: str,
    ) -> str:
        """
        Refine drift type based on temporal pattern within an episode.
        Never changes 'recurring' — that's set by the detector.
        """
        if current_type == 'recurring':
            return 'recurring'

        if len(episode_indices) == 1:
            # Single isolated point → sudden
            return 'sudden'

        values = [delta_L_history[i] for i in sorted(episode_indices)]
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / mean_val if mean_val > 0 else 0.0
        peak = max(values)

        is_increasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
        is_decreasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
        is_monotonic = is_increasing or is_decreasing

        # Incremental: monotonic, low variance
        if is_monotonic and cv < 0.3:
            return 'incremental'

        # Sudden: sharp isolated peak(s)
        if peak > 2.5 * mean_val and len(episode_indices) <= 2:
            return 'sudden'

        # Gradual: multiple points with significant variation
        if len(episode_indices) >= 3 and cv >= 0.3:
            return 'gradual'

        return current_type

    @staticmethod
    def get_statistics(
        merged_episodes: List[Tuple[int, int, str]],
        total_instances: int,
    ) -> Dict:
        if not merged_episodes:
            return {
                'total_episodes': 0,
                'drift_rate': 0.0,
                'episode_lengths': [],
                'type_distribution': {},
                'dominant_type': 'no_drift',
            }

        lengths = [end - start + 1 for start, end, _ in merged_episodes]
        type_dist: Dict[str, int] = {}
        for _, _, t in merged_episodes:
            type_dist[t] = type_dist.get(t, 0) + 1

        # Dominant: weighted priority
        scored = {t: cnt * _TYPE_PRIORITY.get(t, 1) for t, cnt in type_dist.items()}
        dominant = max(scored.items(), key=lambda x: x[1])[0] if scored else 'unknown'

        return {
            'total_episodes': len(merged_episodes),
            'drift_rate': len(merged_episodes) / max(total_instances, 1) * 100,
            'episode_lengths': lengths,
            'type_distribution': type_dist,
            'dominant_type': dominant,
            'avg_episode_length': float(np.mean(lengths)) if lengths else 0.0,
        }
