"""
Enhanced Detailed Analysis Report Generator with Embedded Images
Supports Slovak, Ukrainian, and English with embedded PNG graphs
Includes lattice history visualization and comparison modes
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import base64
from datetime import datetime
import numpy as np
import json
from ..history import LatticeHistoryManager, LatticeComparator
from ..utils.drift_aggregator import DriftTypeAggregator, DriftTypeNormalizer, ConclusionBuilder
from .lattice_integration_patch import LATTICE_CSS, LATTICE_JS, generate_episode_row


class DetailedReportGeneratorV2:
    """Generate comprehensive HTML reports with embedded images and Slovak support"""

    @staticmethod
    def _convert_booleans_to_ints(obj):
        """Recursively convert boolean values to integers for JSON serialization"""
        if isinstance(obj, dict):
            return {k: DetailedReportGeneratorV2._convert_booleans_to_ints(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DetailedReportGeneratorV2._convert_booleans_to_ints(item) for item in obj]
        elif isinstance(obj, bool):
            return int(obj)
        else:
            return obj

    @staticmethod
    def embed_image(image_path: Path) -> str:
        """Convert image file to base64 embedded data URL"""
        if not image_path.exists():
            return ""

        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{image_data}"

    @staticmethod
    def _prepare_lattice_history_for_json(lattice_history: Optional[List[Dict]]) -> List[Dict]:
        """Convert lattice_history to JSON-serializable format with camelCase keys for JavaScript"""
        if not lattice_history:
            return []

        prepared = []
        for frame in lattice_history:
            prepared_frame = {
                'instanceId': int(frame.get('instance_id', 0)),
                'conceptsCount': int(frame.get('concepts_count', 0)),
                'levels': int(frame.get('levels', 0)),
                'similarity': float(frame.get('similarity', 0.0)),
                'deltaL': float(frame.get('delta_L', 0.0)),
                'isDrift': int(frame.get('is_drift', False)),  # Convert bool to int
                'latticeVisual': frame.get('lattice_visual', '')
            }
            prepared.append(prepared_frame)

        return prepared

    @staticmethod
    def prepare_history_manager_for_json(history_manager: Optional[LatticeHistoryManager]) -> Dict:
        """Convert LatticeHistoryManager to comprehensive JSON for visualization"""
        if not history_manager or len(history_manager) == 0:
            return {
                'frames': [],
                'stats': {},
                'comparisons': []
            }

        frames = []
        for i, frame in enumerate(history_manager.frames):
            frames.append({
                'index': i,
                'instanceId': frame.instance_id,
                'conceptsCount': frame.concepts_count,
                'levels': frame.levels,
                'similarity': round(frame.similarity, 4),
                'deltaL': round(frame.delta_L, 4),
                'isDrift': int(frame.drift_detected),
                'driftType': frame.drift_type,
                'confidence': round(frame.confidence, 2),
                'latticeVisual': frame.lattice_visual_html,
                'conceptsDiff': frame.concepts_diff,
                'levelsDiff': frame.levels_diff
            })

        # Generate drift comparisons
        comparisons = []
        for drift_idx, drift_frame in history_manager.get_drift_frames():
            comparison = LatticeComparator.compare_drift_windows(
                history_manager,
                drift_idx,
                before_window=5,
                after_window=5
            )
            if comparison:
                # Convert all booleans to integers for JSON serialization
                comparison = DetailedReportGeneratorV2._convert_booleans_to_ints(comparison)
                comparisons.append({
                    'driftIndex': drift_idx,
                    'driftInstanceId': drift_frame.instance_id,
                    'comparison': comparison
                })

        # Get statistics
        stats = history_manager.get_lattice_evolution_stats()

        return {
            'frames': frames,
            'stats': stats,
            'driftComparisons': comparisons,
            'totalFrames': len(history_manager),
            'driftCount': len(history_manager.drift_frames)
        }

    @staticmethod
    def generate_html_report(
        delta_L_history: List[float],
        similarity_history: List[float],
        drift_indices: List[int],
        drift_types: Dict[int, str],
        output_path: Path,
        language: str = 'sk',
        image_dir: Optional[Path] = None,
        lattice_history: Optional[List[Dict]] = None,
        history_manager: Optional[LatticeHistoryManager] = None,
        lattice_objects: Optional[List] = None,
        window_size: int = 100,
        theta: float = 0.05,
        alpha: float = 0.95,
        dataset_id: str = 'unknown',
        run_id: str = 'unknown',
        merged_episodes: Optional[List] = None,
        conclusion: Optional[Dict] = None,
        aggregator: Optional[Any] = None,  # DriftAggregatorV2 for unified statistics
        snapshot_collector: Optional[Any] = None  # LatticeSnapshotCollector for PNG graphs
    ) -> str:
        """
        Generate comprehensive HTML report with embedded images.
        Uses DriftAggregatorV2 (aggregator) as Single Source of Truth for statistics.

        If aggregator is provided, all statistics (counts, drift rate, types) come from it.
        Otherwise, computed from drift_indices (legacy mode).
        """

        if image_dir is None:
            image_dir = output_path.parent
        else:
            # Ensure image_dir is a Path object
            image_dir = Path(image_dir) if not isinstance(image_dir, Path) else image_dir

        # Note: Lattice visualization UI removed - will be reimplemented later
        lattice_graphs_json = "[]"
        lattice_history_json = "[]"
        drift_comparisons_json = "[]"

        # Embed thesis-quality images - ensure they exist
        img_delta_L = DetailedReportGeneratorV2.embed_image(image_dir / "01_delta_L_thesis.png")
        img_similarity = DetailedReportGeneratorV2.embed_image(image_dir / "02_similarity_thesis.png")
        img_distribution = DetailedReportGeneratorV2.embed_image(image_dir / "03_drift_distribution_thesis.png")
        img_synchronized = DetailedReportGeneratorV2.embed_image(image_dir / "04_synchronized_view_thesis.png")

        # Debug: Log which images were found
        import sys
        has_images = bool(img_delta_L) or bool(img_similarity) or bool(img_distribution) or bool(img_synchronized)
        if not has_images:
            print(f"[WARNING] No thesis images found in {image_dir}", file=sys.stderr)

        # Get localized content (for legacy text templates)
        content, old_aggregator = DetailedReportGeneratorV2._get_localized_content(
            delta_L_history, drift_indices, drift_types, language,
            aggregator_v2=aggregator)

        # Calculate drift rate - use new aggregator (SSOT) if available, else fallback
        if aggregator is not None:
            # Use DriftAggregatorV2 as Single Source of Truth
            drift_rate = aggregator.get_drift_rate()
            episode_count = len(aggregator.get_merged_episodes())
            type_counts = aggregator.get_type_counts()
            dominant_type = aggregator.get_dominant_type()
            merge_gap_display = getattr(aggregator, 'merge_gap', max(window_size, 1))
            cooldown_display = getattr(aggregator, 'cooldown', max(int(0.3 * window_size), 1))
            sudden_z_display = getattr(aggregator, 'SUDDEN_Z_THRESHOLD', 2.5)
        else:
            # Legacy mode: compute from raw drift_indices
            drift_rate = (len(drift_indices) / len(delta_L_history) * 100) if delta_L_history and len(delta_L_history) > 0 else 0.0
            episode_count = len(drift_indices)
            type_counts = {}
            dominant_type = 'unknown'
            merge_gap_display = max(window_size, 1)
            cooldown_display = max(int(0.3 * window_size), 1)
            sudden_z_display = 2.5

        # Prepare conclusion section HTML if conclusion data is provided
        conclusion_section_html = ""
        if conclusion:
            episodes_html = ""
            if conclusion.get('details', {}).get('episode_distribution'):
                episodes_html = "<ul style='margin-left: 20px; margin-top: 10px;'>"
                for episode in conclusion['details']['episode_distribution']:
                    start, end, ep_type = episode
                    episodes_html += f"<li>Episode {start}-{end}: <strong>{ep_type}</strong></li>"
                episodes_html += "</ul>"

            regions_html = ""
            if conclusion.get('details', {}).get('affected_regions'):
                regions_html = "<table style='width: 100%; border-collapse: collapse; margin-top: 10px;'>"
                regions_html += "<tr style='background: #f5f5f5; border-bottom: 1px solid #ddd;'>"
                regions_html += "<th style='padding: 10px; text-align: left;'>Region</th>"
                regions_html += "<th style='padding: 10px; text-align: left;'>Range</th>"
                regions_html += "<th style='padding: 10px; text-align: left;'>Drift Count</th>"
                regions_html += "<th style='padding: 10px; text-align: left;'>Types</th></tr>"
                for region in conclusion['details']['affected_regions']:
                    types_str = ", ".join(region.get('types', []))
                    regions_html += f"<tr style='border-bottom: 1px solid #eee;'>"
                    regions_html += f"<td style='padding: 10px;'>{region['region']}</td>"
                    regions_html += f"<td style='padding: 10px;'>[{region['range'][0]}, {region['range'][1]}]</td>"
                    regions_html += f"<td style='padding: 10px; text-align: center;'>{region['drift_count']}</td>"
                    regions_html += f"<td style='padding: 10px;'>{types_str if types_str else 'N/A'}</td>"
                    regions_html += "</tr>"
                regions_html += "</table>"

            retraining_period = conclusion.get('details', {}).get('retraining_period', 'N/A')
            conclusion_section_html = f"""
            <div class="section">
                <h2 class="section-title">Advanced Conclusion & Recommendations</h2>
                <div style="background: #f0f7ff; padding: 15px; border-left: 4px solid #667eea; margin-bottom: 20px;">
                    <h3>{conclusion.get('icon', '')} {conclusion.get('conclusion_type', 'Unknown').capitalize()}</h3>
                    <p style="margin: 10px 0; font-size: 1.05em;">{conclusion.get('text', '')}</p>
                    <p style="margin: 10px 0; font-weight: bold; color: #d9534f;">📋 Recommendation: {conclusion.get('recommendation', '')}</p>
                    <p style="margin: 10px 0;">Suggested retraining period: <strong>{retraining_period} instances</strong></p>
                </div>

                <h3 style="margin-top: 20px; color: #667eea;">Merged Episodes:</h3>
                {episodes_html}

                <h3 style="margin-top: 20px; color: #667eea;">Affected Stream Regions:</h3>
                {regions_html}
            </div>
            """

        # Build episode details HTML from aggregator (SSOT)
        # ✨ NEW: Use new lattice visualization with PNG graphs
        episode_details_html = ""
        if aggregator is not None:
            episodes = aggregator.get_merged_episodes()
            if episodes:
                # OPTIMIZATION: Skip PNG rendering for large episode counts (>50 episodes)
                # PNG rendering is extremely slow for large datasets and may cause timeouts
                use_png = len(episodes) <= 50 and snapshot_collector is not None
                if len(episodes) > 50:
                    print(f"[INFO] Skipping PNG rendering for {len(episodes)} episodes (too many). HTML will still include all tables and statistics.", file=__import__('sys').stderr)
                    snapshot_collector = None  # Disable PNG rendering

                episode_details_html = """<table style='width: 100%; border-collapse: collapse; margin-top: 10px;'>
                <tr style='background: #f5f5f5; border-bottom: 2px solid #667eea;'>
                    <th style='padding: 12px; text-align: left;'>Episode</th>
                    <th style='padding: 12px; text-align: center;'>Time Range</th>
                    <th style='padding: 12px; text-align: center;'>Duration</th>
                    <th style='padding: 12px; text-align: right;'>DeltaL Max</th>
                    <th style='padding: 12px; text-align: right;'>Sim Min</th>
                    <th style='padding: 12px; text-align: center;'>Type</th>
                    <th style='padding: 12px; text-align: center;'>Lattice</th>
                </tr>"""

                for ep_idx, episode in enumerate(episodes, 1):
                    # Присваиваем индекс для кнопки
                    episode.index = ep_idx
                    # Генерируем рядок с кнопкой и графом решетки
                    row_html = generate_episode_row(
                        episode,
                        snapshot_collector=snapshot_collector,
                        output_dir=str(output_path.parent) if output_path else None,
                        language=language,
                    )
                    episode_details_html += row_html
                episode_details_html += "</table>"
            else:
                episode_details_html = '<p style="color: #999;">No drift episodes detected after aggregation.</p>'

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{content['title']}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}

        .container {{
            max-width: 1400px;
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

        header h1 {{ font-size: 2.5em; margin-bottom: 10px; font-weight: 700; }}
        header p {{ font-size: 1.1em; opacity: 0.9; }}

        .dataset-info {{
            font-size: 0.9em;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid rgba(255,255,255,0.3);
        }}

        .content {{ padding: 40px; }}
        .section {{ margin-bottom: 50px; }}

        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin-bottom: 25px;
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
        }}

        .stat-card .value {{ font-size: 2.5em; font-weight: bold; }}

        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .image-card {{
            background: #f8f9fa;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .image-card h4 {{
            background: #667eea;
            color: white;
            padding: 15px;
            margin: 0;
        }}

        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
            transition: transform 0.2s;
        }}

        .image-card img:hover {{ transform: scale(1.02); }}

        /* Modal Styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
            animation: fadeIn 0.3s;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        .modal-content {{
            position: relative;
            margin: auto;
            padding: 0;
            top: 50%;
            transform: translateY(-50%);
            max-width: 90%;
            max-height: 90vh;
        }}

        .modal-content img {{
            width: 100%;
            height: auto;
            max-height: 85vh;
            object-fit: contain;
            border-radius: 10px;
        }}

        .modal-title {{
            background: rgba(102, 126, 234, 0.95);
            color: white;
            padding: 15px 20px;
            text-align: left;
            border-radius: 10px 10px 0 0;
            font-weight: bold;
            font-size: 1.1em;
        }}

        .close-btn {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            z-index: 1001;
            transition: color 0.2s;
        }}

        .close-btn:hover {{ color: #ccc; }}

        .modal-description {{
            background: white;
            padding: 20px;
            border-radius: 0 0 10px 10px;
            text-align: left;
            max-height: 150px;
            overflow-y: auto;
            font-size: 0.95em;
            line-height: 1.6;
            color: #333;
        }}

        .image-card-description {{
            background: #f0f0f0;
            padding: 12px;
            border-top: 1px solid #ddd;
            font-size: 0.85em;
            color: #555;
            line-height: 1.4;
            text-align: left;
        }}

        /* ✨ NEW: FCA Intent Details Styles */
        .details-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 500;
            transition: all 0.3s ease;
        }}

        .details-btn:hover {{
            background: #5568d3;
            transform: scale(1.05);
        }}

        .details-btn.active {{
            background: #e74c3c;
        }}

        .intent-details-row {{
            transition: all 0.3s ease;
        }}

        .intent-details-row h4 {{
            font-size: 0.95em;
            font-weight: 600;
            margin-bottom: 8px;
        }}

        .conclusion {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
            font-size: 1.2em;
            font-weight: bold;
            text-align: center;
        }}

        footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #999;
            font-size: 0.9em;
            border-top: 1px solid #ddd;
        }}

        @media (max-width: 768px) {{
            .content {{ padding: 20px; }}
            header h1 {{ font-size: 1.8em; }}
            .stats-grid {{ grid-template-columns: 1fr; }}
            .image-gallery {{ grid-template-columns: 1fr; }}
        }}

        /* Modal Styles */
        {LATTICE_CSS}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{content['title']}</h1>
            <p>{content['subtitle']}</p>
            <div class="dataset-info">
                <p style="margin: 5px 0;">Dataset: <strong>{dataset_id.upper()}</strong> | Window Size: <strong>{window_size}</strong></p>
                <p style="font-size: 0.85em; margin-top: 5px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </header>

        <div class="content">
            <!-- Statistics Section -->
            <div class="section">
                <h2 class="section-title">{content['stats_title']}</h2>

                <!-- Row 1: main counters -->
                <div class="stats-grid">
                    <div class="stat-card">
                        <h3>{content['total_instances_label']}</h3>
                        <div class="value">{len(delta_L_history) if delta_L_history else 0}</div>
                        <p style="color:#888;font-size:0.85em;">Windows processed by FCA detector</p>
                    </div>
                    <div class="stat-card" style="border-left:4px solid #E69F00;">
                        <h3>Raw Detections</h3>
                        <div class="value">{len(drift_indices)}</div>
                        <p>{len(drift_indices)/max(len(delta_L_history) if delta_L_history else 1,1)*100:.4f}%</p>
                        <p style="color:#888;font-size:0.8em;margin-top:6px;">
                            Individual windows where &Delta;L<sub>t</sub> exceeded the adaptive
                            threshold &theta;&nbsp;=&nbsp;&mu;&nbsp;+&nbsp;&alpha;&sdot;&sigma;.
                            May include multiple alarms for the same drift event.
                        </p>
                    </div>
                    <div class="stat-card" style="border-left:4px solid #667eea;">
                        <h3>Merged Episodes</h3>
                        <div class="value">{episode_count}</div>
                        <p>{drift_rate:.2f}%</p>
                        <p style="color:#888;font-size:0.8em;margin-top:6px;">
                            Nearby raw points grouped into distinct drift events
                            (merge&nbsp;gap&nbsp;=&nbsp;{merge_gap_display}&nbsp;instances).
                            This is the true count of separate drift events.
                        </p>
                    </div>
                    <div class="stat-card">
                        <h3>{content['conclusion_label']}</h3>
                        <div class="value">{content['conclusion_icon']}</div>
                        <p>{content['conclusion_text']}</p>
                        <p style="color:#888;font-size:0.8em;margin-top:6px;">
                            Dominant type determined by weighted priority scoring
                            across all merged episodes.
                        </p>
                    </div>
                </div>

                <!-- Explanation box -->
                <div style="background:#f8f9ff;border-left:4px solid #667eea;padding:16px 20px;
                            border-radius:0 8px 8px 0;margin-top:20px;font-size:0.9em;color:#444;">
                    <strong>&#9432; How to read these numbers</strong><br><br>
                    <b>Raw Detections</b> = every window where the FCA lattice similarity
                    dropped enough to trigger an alarm. One real drift event typically causes
                    several consecutive alarms (the stream needs a few windows to stabilise
                    after the change).<br><br>
                    <b>Merged Episodes</b> = alarms that are within
                    <em>merge_gap = {merge_gap_display} instances</em> of each other are
                    collapsed into a single episode. This gives the true count of distinct
                    concept-drift events in the stream.<br><br>
                    <b>Drift type</b> is classified per episode using Z-score analysis of
                    &Delta;L<sub>t</sub> relative to the local signal baseline:
                    <em>Sudden</em> (&ge;{sudden_z_display}&sigma; spike) &bull;
                    <em>Gradual</em> (sustained moderate elevation) &bull;
                    <em>Incremental</em> (slow monotonic build-up) &bull;
                    <em>Recurring</em> (similarity to past stable state &ge; threshold).
                </div>
            </div>

            <!-- Image Gallery -->
            <div class="section">
                <h2 class="section-title">{content['graphs_title']}</h2>
                <div class="image-gallery">
                    <div class="image-card">
                        <h4>{content["delta_L_title"]}</h4>
                        <img src="{img_delta_L if img_delta_L else 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22300%22%3E%3Crect fill=%22%23ddd%22 width=%22400%22 height=%22300%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22%3EImage not found%3C/text%3E%3C/svg%3E'}" alt="Delta L" onclick="openModal(this, '{content["delta_L_description"]}', '{content["delta_L_title"]}')">
                        <div class="image-card-description">{content["delta_L_description"]}</div>
                    </div>
                    <div class="image-card">
                        <h4>{content["similarity_title"]}</h4>
                        <img src="{img_similarity if img_similarity else 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22300%22%3E%3Crect fill=%22%23ddd%22 width=%22400%22 height=%22300%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22%3EImage not found%3C/text%3E%3C/svg%3E'}" alt="Similarity" onclick="openModal(this, '{content["similarity_description"]}', '{content["similarity_title"]}')">
                        <div class="image-card-description">{content["similarity_description"]}</div>
                    </div>
                    <div class="image-card">
                        <h4>{content["distribution_title"]}</h4>
                        <img src="{img_distribution if img_distribution else 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22300%22%3E%3Crect fill=%22%23ddd%22 width=%22400%22 height=%22300%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22%3EImage not found%3C/text%3E%3C/svg%3E'}" alt="Distribution" onclick="openModal(this, '{content["distribution_description"]}', '{content["distribution_title"]}')">
                        <div class="image-card-description">{content["distribution_description"]}</div>
                    </div>
                    <div class="image-card">
                        <h4>{content["synchronized_title"]}</h4>
                        <img src="{img_synchronized if img_synchronized else 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22300%22%3E%3Crect fill=%22%23ddd%22 width=%22400%22 height=%22300%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22%3EImage not found%3C/text%3E%3C/svg%3E'}" alt="Synchronized View" onclick="openModal(this, '{content["synchronized_description"]}', '{content["synchronized_title"]}')">
                        <div class="image-card-description">{content["synchronized_description"]}</div>
                    </div>
                </div>
            </div>

            <!-- Merged Episodes Details (from Single Source of Truth) -->
            <div class="section">
                <h2 class="section-title">Merged Episodes Analysis (SSOT)</h2>
                <p style="color: #666; margin-bottom: 15px;">Episodes aggregated from raw detections using DriftAggregatorV2 with merge_gap={merge_gap_display} and cooldown={cooldown_display}.</p>

{episode_details_html if aggregator is not None else '<p style="color: #999;">No aggregator data available.</p>'}
            </div>

            <!-- Conclusion -->
            {conclusion_section_html}
            <div class="section">
                <div class="conclusion">
                    {content['final_conclusion']}
                </div>
            </div>
        </div>

        <footer>
            <p>FCA-based Concept Drift Detection System | 2026</p>
        </footer>
    </div>

    <!-- Modal for Full-size Images -->
    <div id="imageModal" class="modal">
        <div class="modal-title" id="modalTitle">Image</div>
        <span class="close-btn" onclick="closeModal()">&times;</span>
        <div class="modal-content">
            <img id="modalImage" src="" alt="Full size image">
            <div class="modal-description" id="modalDescription"></div>
        </div>
    </div>

    <script>
        function openModal(imgElement, description, title) {{
            const modal = document.getElementById("imageModal");
            const modalImg = document.getElementById("modalImage");
            const modalTitle = document.getElementById("modalTitle");
            const modalDesc = document.getElementById("modalDescription");

            modalImg.src = imgElement.src;
            modalTitle.textContent = title;
            modalDesc.innerHTML = description;
            modal.style.display = "block";
        }}

        function closeModal() {{
            const modal = document.getElementById("imageModal");
            modal.style.display = "none";
        }}

        window.onclick = function(event) {{
            const modal = document.getElementById("imageModal");
            if (event.target == modal) {{
                modal.style.display = "none";
            }}
        }}

        // Close modal on Escape key
        document.addEventListener('keydown', function(event) {{
            if (event.key === "Escape") {{
                closeModal();
            }}
        }});

        // ✨ NEW: Toggle FCA intent details panel
        function toggleIntentDetails(btn) {{
            const row = btn.closest('tr');
            const nextRow = row.nextElementSibling;

            if (nextRow && nextRow.classList && nextRow.classList.contains('intent-details-row')) {{
                const isVisible = nextRow.style.display !== 'none';
                nextRow.style.display = isVisible ? 'none' : 'table-row';
                btn.textContent = isVisible ? 'Show Details' : 'Hide Details';
                btn.classList.toggle('active');
            }}
        }}
    </script>
    {LATTICE_JS}
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        # NOTE: Diagnostic logs already saved by DriftAggregatorV2.save_diagnostics() in main.py
        # Skipping duplicate save to avoid compatibility issues with different aggregator versions.

        return str(output_path)

    @staticmethod
    def _get_localized_content(delta_L_history, drift_indices, drift_types,
                               language='sk', aggregator_v2=None):
        """Get localized content. Uses aggregator_v2 (DriftAggregatorV2) as SSOT if provided."""
        _BADGES = {
            'sudden': '🔴', 'gradual': '🟠', 'incremental': '🟡',
            'recurring': '🔵', 'unknown': '⚪', 'no_drift': '🟢',
        }
        _NAMES = {
            'sudden':      {'sk': 'Prudky drift',       'en': 'Sudden Drift',      'uk': 'Різкий дрейф'},
            'gradual':     {'sk': 'Pozvolny drift',     'en': 'Gradual Drift',     'uk': 'Поступовий дрейф'},
            'incremental': {'sk': 'Narastajuci drift',  'en': 'Incremental Drift', 'uk': 'Інкрементальний дрейф'},
            'recurring':   {'sk': 'Opakujuci sa drift', 'en': 'Recurring Drift',   'uk': 'Повторюваний дрейф'},
            'unknown':     {'sk': 'Neznamy drift',      'en': 'Unknown Drift',     'uk': 'Невідомий дрейф'},
            'no_drift':    {'sk': 'Zaden drift',        'en': 'No Drift',          'uk': 'Дрейф відсутній'},
        }

        if aggregator_v2 is not None:
            dominant_type = aggregator_v2.get_dominant_type()
            type_counts   = aggregator_v2.get_type_counts()
            lang          = language if language in ('sk', 'en', 'uk') else 'en'
            badge         = _BADGES.get(dominant_type, '⚪')
            type_name     = _NAMES.get(dominant_type, _NAMES['unknown'])[lang]
            total         = sum(type_counts.values()) or 1
            breakdown     = ', '.join(
                f"{t.capitalize()} {c} ({100*c/total:.1f}%)"
                for t, c in sorted(type_counts.items(), key=lambda x: -x[1])
            )
            conclusion_text  = f"{type_name} ({breakdown})"
            final_conclusion = f"{badge} {type_name}"
        else:
            aggregator = DriftTypeAggregator(drift_indices, drift_types)
            badge, conclusion_text, final_conclusion = ConclusionBuilder.build_conclusion(
                aggregator, language=language
            )

        if language == 'sk':
            return {
                'title': 'Detailný report detekcie driftu',
                'subtitle': 'Analýza driftu konceptov v dátovom toku',
                'stats_title': 'Štatistika driftu',
                'drifts_detected_label': 'Detegované drifty',
                'total_instances_label': 'Celkový počet inštancií',
                'conclusion_label': 'Záver',
                'conclusion_text': conclusion_text,
                'conclusion_icon': badge,
                'graphs_title': 'Grafy analýzy',
                'delta_L_title': 'Zmena mriežky (ΔL) — Pokročilý pohľad',
                'delta_L_description': 'Hlavný signál driftu s adaptívnym prahom (oranžová čiara). Zobrazuje zmenu formálnych konceptov medzi po sebe idúcimi oknami s aplikovaným kĺzavým priemerom (tmavo modrá čiara) pre redukciu šumu. Zlúčené driftové epizódy sú farebne zvýraznené podľa typu. Sivá šrafovaná zóna na začiatku grafu predstavuje warm-up obdobie vylúčené z analýzy. Adaptívny prah μ±Nσ prispôsobuje citlivosť na základe disperzie posledných okien.',
                'distribution_title': 'Rozdelenie typov driftu — Analýza',
                'distribution_description': 'Komplexná distribučná analýza s koláčovým grafom, histogramom + KDE krivkou, stĺpcovým grafom podľa regiónov toku a panelom štatistík. Zobrazuje celkový počet driftov, ich hustotu, priestorové rozdelenie a rozpad podľa typov. Identifikuje, ktoré typy zmien sa v toku dát vyskytujú najčastejšie a ako sa vyvíjajú v čase.',
                'similarity_title': 'Podobnosť mriežok — Trend',
                'similarity_description': 'Porovnanie podobnosti medzi po sebe idúcimi mriežkami s pásom priemeru ±σ (zelená výplň). Nižšia podobnosť = väčší drift. Kĺzavý priemer eliminuje oscilácie pre jasnú identifikáciu trendov. Červené X markery označujú detegované driftové body. Os Y ohraničená [0,1] pre normalizovanú interpretáciu.',
                'synchronized_title': 'Synchronizovaný pohľad — ΔL + Similarity',
                'synchronized_description': 'Vizualizácia s duálnou osou kombinujúca ΔL (spodný panel, červená) a Similarity (horný panel, modrá) na spoločnej osi X. Umožňuje priame vizuálne porovnanie dvoch metrík a detekciu korelácií v čase. Každá metrika používa vlastnú škálu osi Y pre optimálnu čitateľnosť. Poskytuje komplexný pohľad na dynamiku driftu.',
                'lattice_title': 'Animácia mriežky',
                'lattice_graph_title': 'Grafová štruktúra mriežky',
                'drift_comparison_title': 'Porovnanie driftu (Pred/Po)',
                'play_button': 'HRAŤ',
                'pause_button': 'PAUZA',
                'frame_label': 'Krok',
                'concepts_label': 'Koncepty',
                'levels_label': 'Úrovne',
                'final_conclusion': final_conclusion
            }, aggregator_v2

        elif language == 'uk':
            return {
                'title': 'Детальний звіт аналізу дрейфу',
                'subtitle': 'Аналіз дрейфу концептів у потоці даних',
                'stats_title': 'Статистика дрейфу',
                'drifts_detected_label': 'Виявлено дрейфів',
                'total_instances_label': 'Всього екземплярів',
                'conclusion_label': 'Висновок',
                'conclusion_text': conclusion_text,
                'conclusion_icon': badge,
                'graphs_title': 'Графіки аналізу',
                'delta_L_title': 'Зміна гратки (ΔL) — Розширений вигляд',
                'delta_L_description': 'Основний сигнал дрейфу з адаптивним порогом (помаранчева лінія). Показує зміни гратки між послідовними вікнами з ковзним середнім (темно-синя лінія) для зменшення шуму. Об\'єднані епізоди дрейфу виділені кольором за типом. Сіра штрихована зона — це warm-up період, виключений з аналізу. Адаптивний поріг μ±Nσ коригує чутливість на основі дисперсії останніх вікон.',
                'distribution_title': 'Розподіл типів дрейфу — Аналіз',
                'distribution_description': 'Комплексний аналіз розподілу з кругловою діаграмою, гістограмою + KDE-кривою, стовпчастою діаграмою за регіонами потоку та панеллю статистики. Показує загальну кількість дрейфів, їхню густину, просторовий розподіл та розбивку за типами. Визначає, які типи змін найчастіше трапляються у потоці даних і як вони розвиваються з часом.',
                'similarity_title': 'Схожість граток — Тренд',
                'similarity_description': 'Порівняння схожості між послідовними гратками з діапазоном середнього ±σ (зелена заливка). Менша схожість = більший дрейф. Ковзне середнє згладжує коливання для чіткої ідентифікації трендів. Червоні X-маркери позначають виявлені точки дрейфу. Вісь Y обмежена [0,1] для нормалізованої інтерпретації.',
                'synchronized_title': 'Синхронізований огляд — ΔL + Схожість',
                'synchronized_description': 'Візуалізація з подвійною віссю: ΔL (нижня панель, червона) та Схожість (верхня панель, синя) на спільній осі X. Дозволяє пряме візуальне порівняння двох метрик і виявлення кореляцій у часі. Кожна метрика має власну шкалу осі Y для оптимальної читабельності. Забезпечує комплексне розуміння динаміки дрейфу.',
                'lattice_title': 'Анімація гратки',
                'lattice_graph_title': 'Графова структура гратки',
                'drift_comparison_title': 'Порівняння дрейфу (До/Після)',
                'play_button': 'ГРАТИ',
                'pause_button': 'ПАУЗА',
                'frame_label': 'Крок',
                'concepts_label': 'Концепти',
                'levels_label': 'Рівні',
                'final_conclusion': final_conclusion
            }, aggregator_v2

        else:
            # English (default)

            return {
                'title': 'Detailed Drift Analysis Report',
                'subtitle': 'Analysis of Concept Drifts in Data Stream',
                'stats_title': 'Drift Statistics',
                'drifts_detected_label': 'Drifts Detected',
                'total_instances_label': 'Total Instances',
                'conclusion_label': 'Conclusion',
                'conclusion_text': conclusion_text,
                'conclusion_icon': badge,
                'graphs_title': 'Analysis Graphs',
                'delta_L_title': 'Lattice Change (ΔL) — Advanced View',
                'delta_L_description': 'Primary drift signal with adaptive threshold (orange line). Shows changes in the formal concept lattice between consecutive sliding windows, with Moving Average smoothing (dark blue line) for noise reduction. Merged drift episodes are color-coded by type. The gray hatched zone marks the warm-up period excluded from analysis. The adaptive threshold μ±Nσ dynamically adjusts sensitivity based on recent signal variance.',
                'distribution_title': 'Drift Type Distribution — Analysis',
                'distribution_description': 'Comprehensive distribution analysis with pie chart, histogram + KDE curve, stacked bar chart by stream regions, and a statistics panel. Shows total drift count, drift density, spatial distribution, and type breakdown. Identifies which types of concept changes occur most frequently in the data stream and how they evolve over time.',
                'similarity_title': 'Lattice Similarity — Trend',
                'similarity_description': 'Similarity comparison between consecutive concept lattices with mean ±σ band (green fill). Lower similarity = greater structural drift. Moving Average smoothing reduces oscillations for clear trend identification. Red X markers indicate detected drift points. Y-axis bounded [0,1] for normalized interpretation.',
                'synchronized_title': 'Synchronized View — ΔL + Similarity',
                'synchronized_description': 'Dual-axis visualization combining ΔL (lower panel, red) and Similarity (upper panel, blue) on a shared x-axis. Enables direct visual comparison of both metrics and detection of temporal correlations. Each metric uses its own y-axis scale for optimal readability. Provides a holistic view of drift dynamics.',
                'lattice_title': 'Lattice Animation',
                'lattice_graph_title': 'Lattice Graph Structure',
                'drift_comparison_title': 'Drift Comparison (Before/After)',
                'play_button': 'PLAY',
                'pause_button': 'PAUSE',
                'frame_label': 'Frame',
                'concepts_label': 'Concepts',
                'levels_label': 'Levels',
                'final_conclusion': final_conclusion
            }, aggregator_v2
