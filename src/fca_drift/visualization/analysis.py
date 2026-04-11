"""
Detailed Drift Analysis and Explanations
Детальний аналіз дрейфів та пояснення типів дрейфу
"""
from typing import List, Dict, Any
import numpy as np
from pathlib import Path


class DriftAnalyzer:
    """
    Analyzes detected drifts and provides detailed explanations
    """

    # Explanations for drift types in Ukrainian
    DRIFT_EXPLANATIONS = {
        'sudden': {
            'title': '⚡ РАПТОВИЙ ДРЕЙФ (Sudden Drift)',
            'description': 'Різкий скачок у структурі даних без попередження',
            'characteristics': [
                '• ΔL_t раптово зростає на великий значень (>2x попереднього)',
                '• Концепти гратки кардинально змінюються',
                '• Часто спричинено зовнішньою подією або помилкою в даних',
                '• Вимагає негайного реагування алгоритму'
            ],
            'example': 'Раптова зміна правил у системі, відмова датчика'
        },
        'gradual': {
            'title': '📈 ПОСТУПОВИЙ ДРЕЙФ (Gradual Drift)',
            'description': 'Повільне та плавне змінення розподілу даних',
            'characteristics': [
                '• ΔL_t поступово зростає на протязі кількох вікон',
                '• Концепти еволюціонують поступово',
                '• Похідна ΔL_t > 0 (монотонне зростання)',
                '• Більш складно виявити, але більш поширений'
            ],
            'example': 'Повільна деградація оборудування, зміни в часових умовах'
        },
        'incremental': {
            'title': '➕ НАКОПИЧУВАЛЬНИЙ ДРЕЙФ (Incremental Drift)',
            'description': 'Накопичення малих змін, які призводять до великого зсуву',
            'characteristics': [
                '• Багато невеликих змін, які складаються',
                '• ΔL_t повільно але стійко зростає',
                '• Кумулятивний ефект з часом',
                '• Потребує довгострокового моніторингу'
            ],
            'example': 'Поступове забруднення даних, сезонні зміни'
        },
        'unknown': {
            'title': '❓ НЕВИЗНАЧЕНИЙ ДРЕЙФ (Unknown/Uncertain Drift)',
            'description': 'Дрейф виявлено, але його тип не можна точно класифікувати',
            'characteristics': [
                '• ΔL_t перевищує поріг, але шаблон не помітний',
                '• Недостатньо історичних даних для класифікації',
                '• Може бути комбінація різних типів',
                '• Потребує додаткового дослідження'
            ],
            'example': 'Аномальні шуми в даних, невідомі процеси'
        }
    }

    @staticmethod
    def classify_drift_type(current_delta_L: float,
                           delta_L_history: List[float],
                           window_size: int = 10) -> str:
        """
        Classify drift type based on ΔL_t pattern

        Args:
            current_delta_L: Current ΔL value
            delta_L_history: History of ΔL values
            window_size: Window for trend analysis

        Returns:
            Drift type: 'sudden', 'gradual', 'incremental', or 'unknown'
        """
        if len(delta_L_history) < window_size:
            return 'unknown'

        recent = delta_L_history[-window_size:]

        # Check for sudden drift: sharp increase
        if len(recent) >= 2:
            if current_delta_L > 2 * recent[-2]:
                return 'sudden'

        # Check for gradual drift: monotonic increase
        increases = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        if increases >= len(recent) * 0.7:  # 70% of values increasing
            return 'gradual'

        # Check for incremental drift: slow but steady increase
        mean_recent = np.mean(recent)
        if current_delta_L > mean_recent * 1.5:
            return 'incremental'

        return 'unknown'

    @staticmethod
    def analyze_drift_period(drift_indices: List[int],
                            delta_L_history: List[float]) -> Dict[str, Any]:
        """
        Analyze the period and intensity of drifts
        """
        if not drift_indices:
            return {
                'total_drifts': 0,
                'drift_rate': 0.0,
                'avg_intensity': 0.0,
                'max_intensity': 0.0,
                'periods': []
            }

        drift_indices = sorted(drift_indices)
        intensities = [delta_L_history[i] for i in drift_indices if i < len(delta_L_history)]

        # Find periods between drifts
        periods = []
        if len(drift_indices) > 1:
            for i in range(len(drift_indices) - 1):
                period_length = drift_indices[i+1] - drift_indices[i]
                periods.append(period_length)

        return {
            'total_drifts': len(drift_indices),
            'drift_rate': len(drift_indices) / len(delta_L_history) * 100 if delta_L_history else 0,
            'avg_intensity': np.mean(intensities) if intensities else 0.0,
            'max_intensity': np.max(intensities) if intensities else 0.0,
            'min_intensity': np.min(intensities) if intensities else 0.0,
            'avg_period_between_drifts': np.mean(periods) if periods else 0,
            'periods': periods
        }

    @staticmethod
    def generate_drift_report(drift_indices: List[int],
                             drift_types: Dict[int, str],
                             delta_L_history: List[float],
                             similarity_history: List[float],
                             output_path: Path = None) -> str:
        """
        Generate detailed drift analysis report
        """
        analysis = DriftAnalyzer.analyze_drift_period(drift_indices, delta_L_history)

        report = []
        report.append("\n" + "="*80)
        report.append("ДЕТАЛЬНИЙ АНАЛІЗ ДРЕЙФУ КОНЦЕПТІВ")
        report.append("Detailed Drift Analysis Report")
        report.append("="*80)

        # Summary statistics
        report.append("\n📊 СТАТИСТИКА ДРЕЙФУ / Drift Statistics:")
        report.append(f"   Всього дрейфів виявлено: {analysis['total_drifts']}")
        report.append(f"   Частота дрейфу: {analysis['drift_rate']:.2f}%")
        report.append(f"   Середня інтенсивність ΔL: {analysis['avg_intensity']:.4f}")
        report.append(f"   Максимальна інтенсивність ΔL: {analysis['max_intensity']:.4f}")
        report.append(f"   Мінімальна інтенсивність ΔL: {analysis['min_intensity']:.4f}")

        if analysis['avg_period_between_drifts'] > 0:
            report.append(f"   Середня відстань між дрейфами: {analysis['avg_period_between_drifts']:.0f} екземплярів")

        # Drift type distribution
        report.append("\n🎯 РОЗПОДІЛ ТИПІВ ДРЕЙФУ / Drift Type Distribution:")
        type_counts = {}
        for drift_idx in drift_indices:
            dtype = drift_types.get(drift_idx, 'unknown')
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        for dtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(drift_indices) * 100) if drift_indices else 0
            report.append(f"   {dtype:15} : {count:3} ({percentage:5.1f}%)")

        # Detailed drift explanations
        report.append("\n📖 ПОЯСНЕННЯ ТИПІВ ДРЕЙФУ / Drift Type Explanations:")
        report.append("-" * 80)

        for dtype in sorted(set(drift_types.values())):
            if dtype in DriftAnalyzer.DRIFT_EXPLANATIONS:
                info = DriftAnalyzer.DRIFT_EXPLANATIONS[dtype]
                report.append(f"\n{info['title']}")
                report.append(f"{info['description']}")
                report.append(f"\n   Характеристики / Characteristics:")
                for char in info['characteristics']:
                    report.append(f"   {char}")
                report.append(f"\n   Приклад: {info['example']}")

        # Top 5 most intense drifts
        if drift_indices:
            report.append("\n\n⚠️  ТОП-5 НАЙІНТЕНСИВНІШИХ ДРЕЙФІВ / Top 5 Most Intense Drifts:")
            report.append("-" * 80)

            ranked_drifts = sorted(
                [(idx, delta_L_history[idx], drift_types.get(idx, 'unknown'))
                 for idx in drift_indices if idx < len(delta_L_history)],
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for i, (idx, intensity, dtype) in enumerate(ranked_drifts, 1):
                report.append(f"\n   {i}. Instance #{idx}: ΔL={intensity:.4f} ({dtype})")
                if idx < len(similarity_history):
                    sim = similarity_history[idx]
                    report.append(f"      Similarity: {sim:.4f}")

                    if intensity > 0.6:
                        interpretation = "Дуже сильна зміна структури гратки"
                    elif intensity > 0.4:
                        interpretation = "Помірна зміна структури гратки"
                    else:
                        interpretation = "Слабка зміна структури гратки"

                    report.append(f"      Interpretation: {interpretation}")

        # Recommendations
        report.append("\n\n💡 РЕКОМЕНДАЦІЇ / Recommendations:")
        report.append("-" * 80)

        if analysis['drift_rate'] > 10:
            report.append("   ⚠️  Висока частота дрейфу (>10%) - система нестабільна")
            report.append("       Потрібно дослідити причини часту зміни концептів")
        elif analysis['drift_rate'] > 5:
            report.append("   ⚠️  Помірна частота дрейфу (5-10%) - звичайна для деяких систем")
            report.append("       Продовжити моніторинг та аналіз")
        else:
            report.append("   ✓ Низька частота дрейфу (<5%) - система стабільна")
            report.append("       Дрейфи, що виявлені, потребують особливої уваги")

        if type_counts.get('sudden', 0) > 0:
            report.append(f"   • Виявлено {type_counts['sudden']} раптових дрейфів - потребують невідкладного реагування")

        if type_counts.get('gradual', 0) > len(drift_indices) * 0.5:
            report.append("   • Переважно поступові дрейфи - система еволюціонує плавно")

        report.append("\n" + "="*80)

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"[OK] Detailed report saved to {output_path}")

        return report_text


def print_drift_analysis(drift_indices: List[int],
                        drift_types: Dict[int, str],
                        delta_L_history: List[float],
                        similarity_history: List[float]):
    """
    Print drift analysis to console
    """
    report = DriftAnalyzer.generate_drift_report(
        drift_indices, drift_types, delta_L_history, similarity_history
    )
    print(report)
