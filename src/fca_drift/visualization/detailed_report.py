"""
Detailed Analysis Report Generator
Generates comprehensive HTML and text reports with detailed drift analysis
"""

from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime
import numpy as np


class DetailedReportGenerator:
    """Generate comprehensive detailed analysis reports"""

    # Drift type Ukrainian/English names and descriptions
    DRIFT_DESCRIPTIONS = {
        'sudden': {
            'uk': 'Раптовий дрейф',
            'en': 'Sudden Drift',
            'description_uk': 'Різкий стрибок у структурі гратки',
            'description_en': 'Sharp jump in lattice structure',
            'icon': '⚡'
        },
        'gradual': {
            'uk': 'Поступовий дрейф',
            'en': 'Gradual Drift',
            'description_uk': 'Повільна еволюція структури',
            'description_en': 'Slow evolution of structure',
            'icon': '📈'
        },
        'incremental': {
            'uk': 'Накопичувальний дрейф',
            'en': 'Incremental Drift',
            'description_uk': 'Накопичення малих змін',
            'description_en': 'Accumulation of small changes',
            'icon': '🔄'
        },
        'unknown': {
            'uk': 'Невизначений дрейф',
            'en': 'Unknown Drift',
            'description_uk': 'Не можна точно класифікувати',
            'description_en': 'Cannot be exactly classified',
            'icon': '❓'
        }
    }

    @staticmethod
    def generate_html_report(
        delta_L_history: List[float],
        similarity_history: List[float],
        drift_indices: List[int],
        drift_types: Dict[int, str],
        output_path: Path,
        language: str = 'uk'
    ) -> str:
        """
        Generate comprehensive HTML report

        Args:
            delta_L_history: History of lattice changes (ΔL)
            similarity_history: History of lattice similarities
            drift_indices: Indices where drifts were detected
            drift_types: Mapping of drift index to type
            output_path: Path to save HTML report
            language: 'uk' for Ukrainian or 'en' for English
        """

        # Title
        title = 'Детальний звіт аналізу дрейфу' if language == 'uk' else 'Detailed Drift Analysis Report'

        # Generate statistics
        stats = DetailedReportGenerator._calculate_statistics(
            delta_L_history, similarity_history, drift_indices, drift_types, language
        )

        # Generate drift sections
        drift_sections = DetailedReportGenerator._generate_drift_sections(
            drift_indices, drift_types, delta_L_history, similarity_history, language
        )

        # Generate HTML
        html_content = f"""<!DOCTYPE html>
<html lang="{('uk' if language == 'uk' else 'en')}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 50px;
        }}

        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .stat-card h3 {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-card p {{
            font-size: 0.85em;
            opacity: 0.8;
        }}

        .drift-card {{
            background: #f8f9fa;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .drift-card:hover {{
            transform: translateX(5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .drift-card.sudden {{
            border-left-color: #ff6b6b;
        }}

        .drift-card.gradual {{
            border-left-color: #ffd93d;
        }}

        .drift-card.incremental {{
            border-left-color: #6bcf7f;
        }}

        .drift-card.unknown {{
            border-left-color: #a0aec0;
        }}

        .drift-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}

        .drift-type {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }}

        .drift-type.sudden {{
            background: #ffe0e0;
            color: #ff6b6b;
        }}

        .drift-type.gradual {{
            background: #fff4e0;
            color: #ffd93d;
        }}

        .drift-type.incremental {{
            background: #e0f0e6;
            color: #6bcf7f;
        }}

        .drift-type.unknown {{
            background: #e8eef5;
            color: #667eea;
        }}

        .drift-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
        }}

        .metric {{
            display: flex;
            flex-direction: column;
        }}

        .metric-label {{
            font-size: 0.8em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}

        .metric-value {{
            font-size: 1.4em;
            font-weight: bold;
            color: #667eea;
        }}

        .interpretation {{
            background: #e8eef5;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
            font-style: italic;
            color: #555;
            border-left: 3px solid #667eea;
        }}

        .explanation {{
            background: #f0f4f8;
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
        }}

        .explanation h4 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}

        .explanation ul {{
            margin-left: 20px;
            color: #666;
        }}

        .explanation li {{
            margin-bottom: 8px;
        }}

        footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #999;
            font-size: 0.9em;
            border-top: 1px solid #ddd;
        }}

        .highlight {{
            background: #fff4e0;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 600;
        }}

        @media (max-width: 768px) {{
            .content {{
                padding: 20px;
            }}

            header h1 {{
                font-size: 1.8em;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p>{'Аналіз дрейфу концептів в потоку даних' if language == 'uk' else 'Analysis of Concept Drifts in Data Stream'}</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Згенеровано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <div class="content">
            {stats}
            {drift_sections}
        </div>

        <footer>
            <p>{'FCA-based Concept Drift Detection System' if language == 'uk' else 'FCA-based Concept Drift Detection System'}</p>
            <p>© 2026 All rights reserved</p>
        </footer>
    </div>
</body>
</html>"""

        # Save HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(output_path)

    @staticmethod
    def _calculate_statistics(
        delta_L_history: List[float],
        similarity_history: List[float],
        drift_indices: List[int],
        drift_types: Dict[int, str],
        language: str
    ) -> str:
        """Generate statistics section"""

        # Calculate stats
        total_drifts = len(drift_indices)
        total_instances = len(delta_L_history)
        drift_rate = (total_drifts / total_instances * 100) if total_instances > 0 else 0

        # Type distribution
        type_counts = {}
        for idx in drift_indices:
            dtype = drift_types.get(idx, 'unknown')
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        # Delta L stats
        drift_delta_L = [delta_L_history[i] for i in drift_indices if i < len(delta_L_history)]
        avg_delta_L = np.mean(drift_delta_L) if drift_delta_L else 0
        max_delta_L = np.max(drift_delta_L) if drift_delta_L else 0

        # Similarity stats
        avg_sim = np.mean(similarity_history) if similarity_history else 0
        min_sim = np.min(similarity_history) if similarity_history else 0

        # Text
        if language == 'uk':
            section_title = "📊 Статистика дрейфу"
            total_instances_label = "Всього екземплярів"
            drifts_detected = "Виявлено дрейфів"
            drift_rate_label = "Частота дрейфу"
            avg_delta_label = "Середня ΔL"
            max_delta_label = "Максимальна ΔL"
            avg_sim_label = "Середня подібність"
        else:
            section_title = "📊 Drift Statistics"
            total_instances_label = "Total Instances"
            drifts_detected = "Drifts Detected"
            drift_rate_label = "Drift Rate"
            avg_delta_label = "Average ΔL"
            max_delta_label = "Maximum ΔL"
            avg_sim_label = "Average Similarity"

        html = f"""
        <div class="section">
            <h2 class="section-title">{section_title}</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>{total_instances_label}</h3>
                    <div class="value">{total_instances}</div>
                    <p>{'приклади' if language == 'uk' else 'instances'}</p>
                </div>
                <div class="stat-card">
                    <h3>{drifts_detected}</h3>
                    <div class="value">{total_drifts}</div>
                    <p>{f'{drift_rate:.2f}%' if language == 'uk' else f'{drift_rate:.2f}%'}</p>
                </div>
                <div class="stat-card">
                    <h3>{avg_delta_label}</h3>
                    <div class="value">{avg_delta_L:.4f}</div>
                    <p>({max_delta_label}: {max_delta_L:.4f})</p>
                </div>
                <div class="stat-card">
                    <h3>{avg_sim_label}</h3>
                    <div class="value">{avg_sim:.4f}</div>
                    <p>({('мін.' if language == 'uk' else 'min.')}: {min_sim:.4f})</p>
                </div>
            </div>
        </div>
        """

        return html

    @staticmethod
    def _generate_drift_sections(
        drift_indices: List[int],
        drift_types: Dict[int, str],
        delta_L_history: List[float],
        similarity_history: List[float],
        language: str
    ) -> str:
        """Generate detailed drift analysis sections"""

        section_title = "🎯 Детальний аналіз дрейфів" if language == 'uk' else "🎯 Detailed Drift Analysis"

        html = f'<div class="section"><h2 class="section-title">{section_title}</h2>'

        # Get top drifts by intensity
        top_indices = sorted(
            drift_indices,
            key=lambda i: delta_L_history[i] if i < len(delta_L_history) else 0,
            reverse=True
        )[:10]

        for rank, idx in enumerate(top_indices, 1):
            if idx >= len(delta_L_history):
                continue

            drift_type = drift_types.get(idx, 'unknown')
            delta_L = delta_L_history[idx]
            sim = similarity_history[idx] if idx < len(similarity_history) else 0

            type_info = DetailedReportGenerator.DRIFT_DESCRIPTIONS.get(drift_type, {})
            type_name = type_info.get('uk' if language == 'uk' else 'en', drift_type)
            type_desc = type_info.get('description_uk' if language == 'uk' else 'description_en', '')
            icon = type_info.get('icon', '•')

            # Intensity interpretation
            if language == 'uk':
                if delta_L > 0.6:
                    intensity = 'Дуже сильна'
                elif delta_L > 0.55:
                    intensity = 'Сильна'
                elif delta_L > 0.5:
                    intensity = 'Помірна'
                else:
                    intensity = 'Слабка'
            else:
                if delta_L > 0.6:
                    intensity = 'Very Strong'
                elif delta_L > 0.55:
                    intensity = 'Strong'
                elif delta_L > 0.5:
                    intensity = 'Moderate'
                else:
                    intensity = 'Weak'

            rank_label = "Позиція" if language == 'uk' else "Rank"
            instance_label = "Екземпляр" if language == 'uk' else "Instance"
            delta_label = "Зміна гратки (ΔL)" if language == 'uk' else "Lattice Change (ΔL)"
            sim_label = "Подібність" if language == 'uk' else "Similarity"
            intensity_label = "Інтенсивність" if language == 'uk' else "Intensity"
            type_label = "Тип дрейфу" if language == 'uk' else "Drift Type"
            interpretation_label = "Інтерпретація" if language == 'uk' else "Interpretation"

            html += f"""
            <div class="drift-card {drift_type}">
                <div class="drift-header">
                    <h3>{icon} #{rank}. {instance_label} #{idx}</h3>
                    <span class="drift-type {drift_type}">{type_name}</span>
                </div>
                <div class="drift-metrics">
                    <div class="metric">
                        <span class="metric-label">{delta_label}</span>
                        <span class="metric-value">{delta_L:.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">{sim_label}</span>
                        <span class="metric-value">{sim:.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">{intensity_label}</span>
                        <span class="metric-value">{intensity}</span>
                    </div>
                </div>
                <div class="interpretation">
                    {interpretation_label}: {type_desc}
                </div>
            </div>
            """

        html += '</div>'
        return html
