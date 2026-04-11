"""
Comprehensive Lattice History Management System
Stores, manages, and visualizes concept lattice evolution over time
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime


class DriftStatusEnum(Enum):
    """Drift status classification"""
    NORMAL = 0
    SOFT_DRIFT = 1
    HARD_DRIFT = 2
    VIRTUAL_DRIFT = 3


@dataclass
class LatticeFrame:
    """Represents single lattice state at a point in time"""
    # Core identification
    instance_id: int
    window_number: int
    timestamp: str = None

    # Lattice structure
    concepts_count: int = 0
    levels: int = 0
    lattice_visual_html: str = ""

    # Drift detection metrics
    similarity: float = 1.0
    delta_L: float = 0.0
    drift_detected: bool = False
    drift_type: str = "none"  # "sudden", "gradual", "virtual", "unknown"
    drift_status: DriftStatusEnum = DriftStatusEnum.NORMAL
    confidence: float = 0.0

    # Additional metadata
    threshold_adaptive: float = 0.0
    concepts_diff: int = 0  # Difference from previous lattice
    levels_diff: int = 0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_json_dict(self) -> Dict:
        """Convert to JSON-serializable dictionary"""
        data = asdict(self)
        data['drift_status'] = self.drift_status.value
        data['drift_detected'] = int(self.drift_detected)  # Convert bool to int
        return data

    def get_summary(self) -> str:
        """Get human-readable summary"""
        return (f"Frame #{self.instance_id} | "
                f"Concepts: {self.concepts_count} | "
                f"Levels: {self.levels} | "
                f"Δ_L: {self.delta_L:.4f} | "
                f"Drift: {'⚠️' if self.drift_detected else '✓'}")


class LatticeHistoryManager:
    """Manages complete lattice history with advanced querying and comparison"""

    def __init__(self):
        self.frames: List[LatticeFrame] = []
        self.drift_frames: List[int] = []  # Indices of drift frames
        self.window_number = 0

    def add_frame(self, frame: LatticeFrame) -> None:
        """Add new lattice frame to history"""
        frame.window_number = len(self.frames)
        self.frames.append(frame)

        if frame.drift_detected:
            self.drift_frames.append(len(self.frames) - 1)

    def get_frame(self, index: int) -> Optional[LatticeFrame]:
        """Get frame by index"""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None

    def get_frames_by_range(self, start: int, end: int) -> List[LatticeFrame]:
        """Get frames in index range"""
        return self.frames[max(0, start):min(len(self.frames), end)]

    def get_drift_frames(self) -> List[Tuple[int, LatticeFrame]]:
        """Get all drift frames with their indices"""
        return [(idx, self.frames[idx]) for idx in self.drift_frames]

    def get_frames_before_drift(self, drift_index: int, window_size: int = 5) -> List[LatticeFrame]:
        """Get frames before a drift event"""
        start = max(0, drift_index - window_size)
        return self.frames[start:drift_index]

    def get_frames_after_drift(self, drift_index: int, window_size: int = 5) -> List[LatticeFrame]:
        """Get frames after a drift event"""
        end = min(len(self.frames), drift_index + window_size + 1)
        return self.frames[drift_index + 1:end]

    def compute_structural_changes(self) -> Dict[int, Dict]:
        """Compute structural changes between consecutive frames"""
        changes = {}

        for i in range(1, len(self.frames)):
            prev_frame = self.frames[i - 1]
            curr_frame = self.frames[i]

            changes[i] = {
                'concepts_increase': curr_frame.concepts_count - prev_frame.concepts_count,
                'levels_change': curr_frame.levels - prev_frame.levels,
                'similarity': curr_frame.similarity,
                'delta_L': curr_frame.delta_L,
                'is_significant': abs(curr_frame.delta_L) > 0.1
            }

        return changes

    def get_lattice_evolution_stats(self) -> Dict:
        """Get statistics about lattice evolution"""
        if not self.frames:
            return {
                'total_frames': 0,
                'avg_concepts': 0,
                'avg_levels': 0,
                'drift_count': 0
            }

        concepts_counts = [f.concepts_count for f in self.frames]
        levels_counts = [f.levels for f in self.frames]

        return {
            'total_frames': len(self.frames),
            'avg_concepts': sum(concepts_counts) / len(concepts_counts),
            'max_concepts': max(concepts_counts),
            'min_concepts': min(concepts_counts),
            'avg_levels': sum(levels_counts) / len(levels_counts),
            'max_levels': max(levels_counts),
            'min_levels': min(levels_counts),
            'drift_count': len(self.drift_frames),
            'drift_rate': len(self.drift_frames) / len(self.frames) if self.frames else 0
        }

    def to_json(self) -> str:
        """Serialize entire history to JSON"""
        frames_json = [frame.to_json_dict() for frame in self.frames]
        return json.dumps(frames_json)

    def to_json_compact(self) -> str:
        """Serialize history without HTML visuals (more compact)"""
        frames_json = []

        for frame in self.frames:
            compact = {
                'id': frame.instance_id,
                'w': frame.window_number,
                'c': frame.concepts_count,
                'l': frame.levels,
                's': round(frame.similarity, 3),
                'd': round(frame.delta_L, 4),
                'drift': int(frame.drift_detected),
                'type': frame.drift_type
            }
            frames_json.append(compact)

        return json.dumps(frames_json)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> LatticeFrame:
        return self.frames[index]


class LatticeComparator:
    """Compare lattices between different time points (before/after drift)"""

    @staticmethod
    def compare_frames(frame1: LatticeFrame, frame2: LatticeFrame) -> Dict:
        """Compare two lattice frames"""
        return {
            'instances_between': frame2.instance_id - frame1.instance_id,
            'concepts_change': frame2.concepts_count - frame1.concepts_count,
            'levels_change': frame2.levels - frame1.levels,
            'similarity_change': frame2.similarity - frame1.similarity,
            'delta_L_change': frame2.delta_L - frame1.delta_L,
            'drift_transition': (
                'drift_detected' if not frame1.drift_detected and frame2.drift_detected
                else 'drift_resolved' if frame1.drift_detected and not frame2.drift_detected
                else 'no_change'
            )
        }

    @staticmethod
    def compare_drift_windows(
        history: LatticeHistoryManager,
        drift_index: int,
        before_window: int = 5,
        after_window: int = 5
    ) -> Dict:
        """Compare lattice before and after a drift event"""

        frames_before = history.get_frames_before_drift(drift_index, before_window)
        frames_after = history.get_frames_after_drift(drift_index, after_window)

        if not frames_before or not frames_after:
            return {}

        last_before = frames_before[-1]
        first_after = frames_after[0]

        return {
            'drift_at': history.frames[drift_index].instance_id,
            'before': {
                'frame': last_before.to_json_dict(),
                'avg_concepts': sum(f.concepts_count for f in frames_before) / len(frames_before),
                'avg_levels': sum(f.levels for f in frames_before) / len(frames_before),
                'stability': 1.0 - (sum(abs(f.delta_L) for f in frames_before) / len(frames_before))
            },
            'after': {
                'frame': first_after.to_json_dict(),
                'avg_concepts': sum(f.concepts_count for f in frames_after) / len(frames_after),
                'avg_levels': sum(f.levels for f in frames_after) / len(frames_after),
                'stability': 1.0 - (sum(abs(f.delta_L) for f in frames_after) / len(frames_after))
            },
            'change_summary': LatticeComparator.compare_frames(last_before, first_after)
        }


class LatticeHistorySerializer:
    """Handle serialization of lattice history to various formats"""

    @staticmethod
    def to_animation_json(history: LatticeHistoryManager) -> str:
        """Convert history to JSON suitable for HTML animation"""
        frames = []

        for i, frame in enumerate(history.frames):
            frames.append({
                'frameIndex': i,
                'instanceId': frame.instance_id,
                'conceptsCount': frame.concepts_count,
                'levels': frame.levels,
                'similarity': round(frame.similarity, 4),
                'deltaL': round(frame.delta_L, 4),
                'isDrift': int(frame.drift_detected),
                'driftType': frame.drift_type,
                'latticeVisual': frame.lattice_visual_html,
                'confidence': round(frame.confidence, 2)
            })

        return json.dumps(frames)

    @staticmethod
    def to_comparison_json(comparison_data: Dict) -> str:
        """Convert comparison data to JSON for visualization"""
        return json.dumps(comparison_data, default=str)

    @staticmethod
    def export_csv_timeline(history: LatticeHistoryManager, filepath: str) -> None:
        """Export lattice history as CSV timeline"""
        import csv

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Frame', 'Instance_ID', 'Window', 'Concepts', 'Levels',
                'Similarity', 'Delta_L', 'Is_Drift', 'Drift_Type', 'Confidence'
            ])

            # Rows
            for i, frame in enumerate(history.frames):
                writer.writerow([
                    i,
                    frame.instance_id,
                    frame.window_number,
                    frame.concepts_count,
                    frame.levels,
                    f"{frame.similarity:.4f}",
                    f"{frame.delta_L:.4f}",
                    int(frame.drift_detected),
                    frame.drift_type,
                    f"{frame.confidence:.2f}"
                ])

        print(f"✓ Exported lattice history to {filepath}")
