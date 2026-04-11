"""
drift_results_ssot.py — єдине джерело правди для всіх результатів.

ВИКОРИСТАННЯ:
  Замість того щоб передавати detector.drift_indices, merged_episodes,
  drift_types окремо — створи один об'єкт DriftResults і передавай його.

  results = DriftResults.from_detector(detector, aggregator, dataset_name, params)

  # Тепер скрізь одне число:
  print(results.display_count)        # → 6 (merged episodes)
  print(results.raw_count)            # → 12 (raw detections)
  print(results.drift_rate_pct)       # → 1.33%

  # Передавай в звіт:
  generate_html_report(results=results, ...)
  export_graphs(results=results, ...)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class EpisodeInfo:
    index: int
    start: int
    end: int
    duration: int
    dlt_max: float
    sim_min: float
    dominant_type: str
    raw_hits: int


@dataclass
class DriftResults:
    """
    SSOT — єдиний об'єкт який містить всі числа.
    Передавай ТІЛЬКИ його в репортер і в графіки.
    """
    dataset_name: str

    # ── Що показувати користувачу (MERGED episodes) ──────────────────
    display_count: int          # ← ЦЕ число йде в заголовок звіту
    display_episodes: List[EpisodeInfo] = field(default_factory=list)

    # ── Технічні деталі (RAW detections) ─────────────────────────────
    raw_count: int = 0          # скільки разів ΔL перевищив поріг
    total_instances: int = 0
    warm_up_end: int = 0

    # ── Метрики ───────────────────────────────────────────────────────
    drift_rate_pct: float = 0.0
    dominant_type: str = "unknown"
    delta_l_mean: float = 0.0
    delta_l_max: float = 0.0
    similarity_mean: float = 0.0

    # ── Параметри запуску ─────────────────────────────────────────────
    params: Dict = field(default_factory=dict)

    # ── Часові ряди для графіків ──────────────────────────────────────
    delta_l_history: List[float] = field(default_factory=list)
    similarity_history: List[float] = field(default_factory=list)
    adaptive_threshold_history: List[float] = field(default_factory=list)

    # ── Карта типів для графіків (index → type) ───────────────────────
    drift_type_map: Dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_detector(
        cls,
        detector,       # FCADriftDetector instance
        aggregator,     # DriftAggregatorV2 instance
        dataset_name: str,
        params: dict,
    ) -> "DriftResults":
        """
        Єдиний спосіб створити DriftResults.
        Всі числа беруться з ОДНОГО джерела.
        """
        merged = aggregator.get_merged_episodes()

        # Будуємо карту типів ВЖЕ з merged episodes (не з raw)
        drift_type_map = {}
        for ep in merged:
            for idx in range(ep.start, ep.end + 1):
                drift_type_map[idx] = ep.dominant_type

        # Підраховуємо dominant type за голосуванням
        type_votes: Dict[str, int] = {}
        for ep in merged:
            t = ep.dominant_type
            type_votes[t] = type_votes.get(t, 0) + 1
        dominant = max(type_votes, key=type_votes.get) if type_votes else "unknown"

        dl = detector.delta_L_history
        sim = detector.similarity_history
        total = len(dl) + params.get("window_size", 50)  # приблизно

        episodes_info = [
            EpisodeInfo(
                index=i + 1,
                start=ep.start,
                end=ep.end,
                duration=ep.end - ep.start + 1,
                dlt_max=round(ep.dlt_max, 4),
                sim_min=round(ep.sim_min, 4),
                dominant_type=ep.dominant_type,
                raw_hits=getattr(ep, "raw_hits", 1),
            )
            for i, ep in enumerate(merged)
        ]

        return cls(
            dataset_name=dataset_name,
            display_count=len(merged),          # ← MERGED count
            display_episodes=episodes_info,
            raw_count=len(detector.drift_indices),
            total_instances=total,
            warm_up_end=getattr(aggregator, "warm_up_end", 0),
            drift_rate_pct=round(len(merged) / max(total, 1) * 100, 2),
            dominant_type=dominant,
            delta_l_mean=round(sum(dl) / len(dl), 4) if dl else 0.0,
            delta_l_max=round(max(dl), 4) if dl else 0.0,
            similarity_mean=round(sum(sim) / len(sim), 4) if sim else 0.0,
            params=params,
            delta_l_history=dl,
            similarity_history=sim,
            adaptive_threshold_history=getattr(detector, "adaptive_threshold_history", []),
            drift_type_map=drift_type_map,
        )

    def to_json(self, path: str) -> None:
        """Зберігає результати як JSON для діагностики."""
        data = {
            "dataset": self.dataset_name,
            "display_count": self.display_count,
            "raw_count": self.raw_count,
            "total_instances": self.total_instances,
            "drift_rate_pct": self.drift_rate_pct,
            "dominant_type": self.dominant_type,
            "delta_l_mean": self.delta_l_mean,
            "delta_l_max": self.delta_l_max,
            "similarity_mean": self.similarity_mean,
            "params": self.params,
            "episodes": [
                {
                    "index": ep.index,
                    "range": [ep.start, ep.end],
                    "duration": ep.duration,
                    "dlt_max": ep.dlt_max,
                    "sim_min": ep.sim_min,
                    "type": ep.dominant_type,
                    "raw_hits": ep.raw_hits,
                }
                for ep in self.display_episodes
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def summary_line(self) -> str:
        return (
            f"[RESULTS] {self.dataset_name}: "
            f"{self.display_count} episodes "
            f"({self.raw_count} raw detections), "
            f"drift rate={self.drift_rate_pct}%, "
            f"dominant={self.dominant_type}"
        )
