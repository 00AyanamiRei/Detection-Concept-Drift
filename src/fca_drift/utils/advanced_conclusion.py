"""
Advanced Conclusion Builder
Generates scientifically-grounded conclusions based on merged drift episodes
Ensures consistency across all report components
"""

from typing import Dict, List, Tuple
import json
from pathlib import Path


class AdvancedConclusionBuilder:
    """
    Builds conclusions from analyzed drift episodes.
    Generates consistent conclusions, icons, and recommendations.
    """

    RECOMMENDATIONS = {
        'no_drift': {
            'icon': '🟢',
            'en': 'System is stable. Continue monitoring.',
            'sk': 'Systém je stabilný. Pokračovať v monitoringu.',
            'uk': 'Система стабільна. Продовжувати моніторинг.'
        },
        'incremental': {
            'icon': '🟡',
            'en': 'Incremental drift detected. Regular retraining recommended every {period} instances.',
            'sk': 'Zistený inkrementálny drift. Odporúčaná pravidelná obnova každých {period} inštancií.',
            'uk': 'Виявлено накопичуванський дрейф. Рекомендується регулярне переучування кожні {period} екземплярів.'
        },
        'gradual': {
            'icon': '🟠',
            'en': 'Gradual drift detected in {count} episodes. Monitoring and periodic retraining ({period} instances) recommended.',
            'sk': 'Pozvolný drift zistený v {count} epizódach. Odporúčaný monitoring a periodická obnova ({period} inštancií).',
            'uk': 'Виявлено поступовий дрейф у {count} епізодах. Рекомендується моніторинг та періодичне переучування ({period} екземплярів).'
        },
        'sudden': {
            'icon': '🔴',
            'en': 'Sudden drift detected in {count} episodes. Immediate retraining and model inspection required.',
            'sk': 'Prudký drift zistený v {count} epizódach. Vyžaduje sa okamžitá obnova a kontrola modelu.',
            'uk': 'Виявлено раптовий дрейф у {count} епізодах. Потрібне негайне переучування та перевірка моделі.'
        }
    }

    @staticmethod
    def build_from_episodes(
        merged_episodes: List[Tuple[int, int, str]],
        total_instances: int,
        window_size: int,
        theta: float,
        alpha: float,
        dataset_id: str = 'unknown',
        language: str = 'en'
    ) -> Dict:
        """
        Build comprehensive conclusion from merged episodes.

        Returns:
            {
                'conclusion_type': 'gradual|sudden|incremental|no_drift',
                'icon': '🟡|🔴|🟡|🟢',
                'episodes_count': int,
                'drift_rate': float,
                'dominant_type': str,
                'text': str,
                'recommendation': str,
                'details': {
                    'episode_distribution': list,
                    'affected_regions': list,
                }
            }
        """

        if not merged_episodes:
            conclusion_type = 'no_drift'
            episodes_count = 0
            drift_rate = 0.0
            dominant_type = 'no_drift'
        else:
            # Determine dominant type from episodes
            type_counts = {}
            for _, _, dtype in merged_episodes:
                type_counts[dtype] = type_counts.get(dtype, 0) + 1

            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]

            # Priority for conclusion type
            if 'sudden' in type_counts and type_counts['sudden'] > 0:
                conclusion_type = 'sudden'
            elif 'gradual' in type_counts and type_counts['gradual'] > 0:
                conclusion_type = 'gradual'
            elif 'incremental' in type_counts and type_counts['incremental'] > 0:
                conclusion_type = 'incremental'
            else:
                conclusion_type = 'unknown'

            episodes_count = len(merged_episodes)
            drift_rate = episodes_count / max(total_instances, 1) * 100

        # Get icon and base text
        rec = AdvancedConclusionBuilder.RECOMMENDATIONS.get(
            conclusion_type,
            AdvancedConclusionBuilder.RECOMMENDATIONS['no_drift']
        )
        icon = rec['icon']

        # Generate recommendation with parameters
        retraining_period = max(window_size * 20, int(total_instances * 0.05))
        recommendation = rec[language].format(
            count=episodes_count,
            period=retraining_period
        )

        # Build detailed text
        text = AdvancedConclusionBuilder._build_detailed_text(
            conclusion_type,
            episodes_count,
            drift_rate,
            dominant_type,
            merged_episodes,
            total_instances,
            window_size,
            language
        )

        # Determine affected regions
        affected_regions = AdvancedConclusionBuilder._identify_affected_regions(
            merged_episodes,
            total_instances,
            num_regions=5
        )

        return {
            'conclusion_type': conclusion_type,
            'icon': icon,
            'episodes_count': episodes_count,
            'drift_rate': drift_rate,
            'dominant_type': dominant_type,
            'text': text,
            'recommendation': recommendation,
            'details': {
                'episode_distribution': merged_episodes,
                'affected_regions': affected_regions,
                'retraining_period': retraining_period,
                'parameters': {
                    'window_size': window_size,
                    'theta': theta,
                    'alpha': alpha,
                    'dataset': dataset_id
                }
            }
        }

    @staticmethod
    def _build_detailed_text(
        conclusion_type: str,
        episodes_count: int,
        drift_rate: float,
        dominant_type: str,
        merged_episodes: List[Tuple[int, int, str]],
        total_instances: int,
        window_size: int,
        language: str
    ) -> str:
        """Build detailed text explanation."""

        if language == 'sk':
            if conclusion_type == 'no_drift':
                return f"Žiadny drift nebol detegovaný. Všetky {total_instances} inštancií vykazujú stabilnú správu."
            elif conclusion_type == 'sudden':
                return (
                    f"{episodes_count} prudkých driftov detegovaných ({drift_rate:.1f}% inštancií). "
                    f"Dominantný typ: {dominant_type}. "
                    f"Epizódy sa vyskytujú v pozíciách: {', '.join(str(s) for s, _, _ in merged_episodes[:5])}. "
                    f"Tieto sú väčšinou izolované zmeny s velkou amplitúdou."
                )
            elif conclusion_type == 'gradual':
                return (
                    f"{episodes_count} pozvolných driftov detegovaných ({drift_rate:.1f}% inštancií). "
                    f"Zmeny sú postupné a trvalé. Priemer dĺžky epizódy: {np.mean([e[1]-e[0] for e in merged_episodes]):.0f} okien. "
                    f"Táto zmena naznačuje postupný posun v distribúcii dát."
                )
            elif conclusion_type == 'incremental':
                return (
                    f"{episodes_count} inkrementálnych driftov detegovaných ({drift_rate:.1f}% inštancií). "
                    f"Tieto sú malé, ale výrazné zmeny, ktoré sa postupne hromadia."
                )
            else:
                return f"{episodes_count} neznámych typov driftov."

        elif language == 'uk':
            if conclusion_type == 'no_drift':
                return f"Дрейф не виявлено. Усі {total_instances} екземплярів показують стабільну поведінку."
            elif conclusion_type == 'sudden':
                return (
                    f"{episodes_count} раптових дрейфів виявлено ({drift_rate:.1f}% екземплярів). "
                    f"Переважний тип: {dominant_type}. "
                    f"Епізоди виникають в позиціях: {', '.join(str(s) for s, _, _ in merged_episodes[:5])}. "
                    f"Це переважно ізольовані зміни з великою амплітудою."
                )
            elif conclusion_type == 'gradual':
                return (
                    f"{episodes_count} поступових дрейфів виявлено ({drift_rate:.1f}% екземплярів). "
                    f"Зміни є поступовими та тривалими. Середня довжина епізоду: {np.mean([e[1]-e[0] for e in merged_episodes]):.0f} вікон. "
                    f"Це свідчить про поступовий зсув у розподілі даних."
                )
            elif conclusion_type == 'incremental':
                return (
                    f"{episodes_count} накопичувальних дрейфів виявлено ({drift_rate:.1f}% екземплярів). "
                    f"Це малі, але значні зміни, які поступово накопичуються."
                )
            else:
                return f"{episodes_count} невідомих типів дрейфів."

        else:  # English
            if conclusion_type == 'no_drift':
                return f"No drift detected. All {total_instances} instances exhibit stable behavior."
            elif conclusion_type == 'sudden':
                return (
                    f"{episodes_count} sudden drifts detected ({drift_rate:.1f}% of instances). "
                    f"Dominant type: {dominant_type}. "
                    f"Episodes occur at positions: {', '.join(str(s) for s, _, _ in merged_episodes[:5])}. "
                    f"These are mostly isolated changes with high amplitude."
                )
            elif conclusion_type == 'gradual':
                return (
                    f"{episodes_count} gradual drifts detected ({drift_rate:.1f}% of instances). "
                    f"Changes are progressive and sustained. Mean episode length: {np.mean([e[1]-e[0] for e in merged_episodes]):.0f} windows. "
                    f"This indicates a gradual shift in data distribution."
                )
            elif conclusion_type == 'incremental':
                return (
                    f"{episodes_count} incremental drifts detected ({drift_rate:.1f}% of instances). "
                    f"These are small but significant changes that accumulate progressively."
                )
            else:
                return f"{episodes_count} unknown drift types detected."

    @staticmethod
    def _identify_affected_regions(
        merged_episodes: List[Tuple[int, int, str]],
        total_instances: int,
        num_regions: int = 5
    ) -> List[Dict]:
        """Identify which regions of the stream are affected by drift."""

        regions = []
        region_size = total_instances / num_regions

        for region_idx in range(num_regions):
            region_start = int(region_idx * region_size)
            region_end = int((region_idx + 1) * region_size)

            drifts_in_region = [
                (s, e, t) for s, e, t in merged_episodes
                if not (e < region_start or s >= region_end)
            ]

            regions.append({
                'region': f'R{region_idx + 1}',
                'range': (region_start, region_end),
                'drift_count': len(drifts_in_region),
                'types': [t for _, _, t in drifts_in_region]
            })

        return regions

    @staticmethod
    def save_detailed_report(
        conclusion: Dict,
        output_path: Path,
        run_id: str = 'unknown'
    ):
        """Save detailed conclusion and metadata to JSON."""

        report_data = {
            'run_id': run_id,
            'conclusion': conclusion['conclusion_type'],
            'icon': conclusion['icon'],
            'episodes_count': conclusion['episodes_count'],
            'drift_rate': conclusion['drift_rate'],
            'recommendation': conclusion['recommendation'],
            'text': conclusion['text'],
            'details': conclusion['details']
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path / 'conclusion.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)


import numpy as np
