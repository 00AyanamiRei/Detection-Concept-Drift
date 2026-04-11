# """
# lattice_snapshot_collector.py
# ─────────────────────────────
# Колектор знімків решіток — додається в FCADriftDetector.
# """

# from typing import Dict, Set, FrozenSet, Optional
# from dataclasses import dataclass


# @dataclass
# class LatticeSnapshot:
#     instance_id:    int
#     intents_before: Set[FrozenSet]
#     intents_after:  Set[FrozenSet]
#     delta_lt:       float
#     similarity:     float

#     @property
#     def lost(self) -> Set[FrozenSet]:
#         return self.intents_before - self.intents_after

#     @property
#     def gained(self) -> Set[FrozenSet]:
#         return self.intents_after - self.intents_before

#     @property
#     def stable(self) -> Set[FrozenSet]:
#         return self.intents_before & self.intents_after

#     def summary(self) -> str:
#         return (
#             f"t={self.instance_id}  ΔL={self.delta_lt:.4f}  "
#             f"sim={self.similarity:.4f}  "
#             f"lost={len(self.lost)}  gained={len(self.gained)}  "
#             f"stable={len(self.stable)}"
#         )


# class LatticeSnapshotCollector:
#     """
#     Зберігає пари решіток (до/після) для кожного виявленого дрифту.
#     """

#     def __init__(self):
#         self._snapshots: Dict[int, LatticeSnapshot] = {}

#     def record(
#         self,
#         instance_id:    int,
#         intents_before: Set[FrozenSet],
#         intents_after:  Set[FrozenSet],
#         delta_lt:       float,
#         similarity:     float,
#     ) -> None:
#         self._snapshots[instance_id] = LatticeSnapshot(
#             instance_id    = instance_id,
#             intents_before = set(intents_before),
#             intents_after  = set(intents_after),
#             delta_lt       = delta_lt,
#             similarity     = similarity,
#         )

#     def get(self, instance_id: int) -> Optional[LatticeSnapshot]:
#         return self._snapshots.get(instance_id)

#     def get_nearest(self, instance_id: int, window: int = 5) -> Optional[LatticeSnapshot]:
#         """Знаходить найближчий знімок в межах ±window від instance_id."""
#         for delta in range(0, window + 1):
#             for candidate in [instance_id - delta, instance_id + delta]:
#                 if candidate in self._snapshots:
#                     return self._snapshots[candidate]
#         return None

#     def all(self):
#         return list(self._snapshots.values())

#     def __len__(self):
#         return len(self._snapshots)
"""
lattice_snapshot_collector.py
─────────────────────────────────────────────────
Колектор знімків решіток — додається в FCADriftDetector.

ІНТЕГРАЦІЯ:
    Замість того щоб змінювати весь детектор,
    просто передай LatticeSnapshotCollector в детектор
    і викликай .record() на кожному кроці.
"""

from typing import Dict, Set, FrozenSet, Optional
from dataclasses import dataclass, field


@dataclass
class LatticeSnapshot:
    instance_id:    int
    intents_before: Set[FrozenSet]
    intents_after:  Set[FrozenSet]
    delta_lt:       float
    similarity:     float

    @property
    def lost(self) -> Set[FrozenSet]:
        return self.intents_before - self.intents_after

    @property
    def gained(self) -> Set[FrozenSet]:
        return self.intents_after - self.intents_before

    @property
    def stable(self) -> Set[FrozenSet]:
        return self.intents_before & self.intents_after

    def summary(self) -> str:
        return (
            f"t={self.instance_id}  ΔL={self.delta_lt:.4f}  "
            f"sim={self.similarity:.4f}  "
            f"lost={len(self.lost)}  gained={len(self.gained)}  "
            f"stable={len(self.stable)}"
        )


class LatticeSnapshotCollector:
    """
    Зберігає пари решіток (до/після) для кожного виявленого дрифту.

    ПІДКЛЮЧЕННЯ до FCADriftDetector:

        # В __init__ детектора:
        self.snapshot_collector = LatticeSnapshotCollector()

        # В методі update() — одразу після виявлення дрифту:
        if drift_detected:
            self.snapshot_collector.record(
                instance_id    = instance_id,
                intents_before = prev_intents,
                intents_after  = curr_intents,
                delta_lt       = delta_lt,
                similarity     = similarity,
            )
    """

    def __init__(self):
        self._snapshots: Dict[int, LatticeSnapshot] = {}

    def record(
        self,
        instance_id:    int,
        intents_before: Set[FrozenSet],
        intents_after:  Set[FrozenSet],
        delta_lt:       float,
        similarity:     float,
    ) -> None:
        self._snapshots[instance_id] = LatticeSnapshot(
            instance_id    = instance_id,
            intents_before = set(intents_before),
            intents_after  = set(intents_after),
            delta_lt       = delta_lt,
            similarity     = similarity,
        )

    def get(self, instance_id: int) -> Optional[LatticeSnapshot]:
        return self._snapshots.get(instance_id)

    def get_nearest(self, instance_id: int, window: int = 5) -> Optional[LatticeSnapshot]:
        """Знаходить найближчий знімок в межах ±window від instance_id."""
        for delta in range(0, window + 1):
            for candidate in [instance_id - delta, instance_id + delta]:
                if candidate in self._snapshots:
                    return self._snapshots[candidate]
        return None

    def all(self):
        return list(self._snapshots.values())

    def __len__(self):
        return len(self._snapshots)
