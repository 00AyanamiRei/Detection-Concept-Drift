#!/usr/bin/env python3
"""
FCA-Based Concept Drift Detection System - Main Entry Point
Unified command-line interface with parameter control
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import random

# CRITICAL: Set global random seeds for reproducibility
# This must be done BEFORE any random operations
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from fca_drift.core import StreamReader, StreamWrapper, SlidingWindow, DataPreprocessor, get_available_datasets
from fca_drift.fca import build_formal_context, ConceptLattice
from fca_drift.detection import FCADriftDetector
from fca_drift.visualization import DriftAnalyzer, create_analysis_summary
from fca_drift.visualization.detailed_report_v2 import DetailedReportGeneratorV2
from fca_drift.visualization.thesis_graphics import ThesisGraphicsExporter
from fca_drift.utils.drift_aggregator_v2 import DriftAggregatorV2



class DriftDetectionRunner:
    """Main runner for drift detection with unified output"""

    # Drift type mapping in Slovak
    DRIFT_NAMES_SK = {
        'sudden':      'Prudky drift (Sudden)',
        'gradual':     'Pozvolny drift (Gradual)',
        'incremental': 'Nahromadjujuci drift (Incremental)',
        'recurring':   'Opakujuci sa drift (Recurring)',
        'unknown':     'Neziarny drift (Unknown)',
        'no_drift':    'Zaden drift (No Drift)',
    }

    def __init__(self, args):
        self.args = args
        self.detector = None
        self._target_label_map = {}
        self._next_target_label_id = 0
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'dataset': args.dataset,
                'max_instances': args.max_instances,
                'window_size': args.window_size,
                'theta': args.theta,
                'alpha': args.alpha,
                'preprocess_freeze_after': args.preprocess_freeze_after,
                'preprocess_ema_alpha': args.preprocess_ema_alpha,
                'context_mode': args.context_mode
            },
            'detection_results': {}
        }

    def _is_supervised_mode(self) -> bool:
        """Whether target label should be appended to FCA context."""
        return self.args.context_mode == 'supervised_mode'

    def _encode_target_for_context(self, y) -> float:
        """Encode arbitrary label into numeric value for preprocessing."""
        if y is None:
            return 0.0
        if isinstance(y, (bool, int, np.integer, float, np.floating)):
            return float(y)

        key = str(y)
        if key not in self._target_label_map:
            self._target_label_map[key] = self._next_target_label_id
            self._next_target_label_id += 1
        return float(self._target_label_map[key])

    def load_data(self):
        """Load data from dataset or custom file"""
        print("[LOADING] Loading dataset...")

        if self.args.custom_file:
            # Load custom CSV file
            import pandas as pd
            df = pd.read_csv(self.args.custom_file)
            print(f"[OK] Loaded custom dataset from: {self.args.custom_file}")
            print(f"[OK] Dataset shape: {df.shape}")

            # Extract features (all but last column) and label (last column)
            X = df.iloc[:, :-1]  # All columns except last
            y = df.iloc[:, -1].values   # Last column

            # Create iterator that yields dictionaries (matching StreamWrapper format)
            # Features as dict with keys x0, x1, x2, etc.
            def custom_stream():
                for idx, row in X.iterrows():
                    x_dict = {f'x{i}': float(val) for i, val in enumerate(row)}
                    y_val = int(y[idx])
                    yield x_dict, y_val

            # Wrap in StreamWrapper to handle max_instances if needed
            return StreamWrapper(custom_stream(), max_instances=self.args.max_instances)
        else:
            # Load from generator
            reader = StreamReader(self.args.dataset, seed=42)
            stream = StreamWrapper(reader.stream, max_instances=self.args.max_instances)
            print(f"[OK] Loaded {self.args.dataset} dataset ({self.args.max_instances} instances)")
            return stream

    def run_detection(self):
        """Run drift detection with given parameters"""
        print("\n[DETECTION] Starting drift detection...")
        print(f"[PARAMS] Window size: {self.args.window_size}, Theta: {self.args.theta}, Alpha: {self.args.alpha}")

        # Load data
        stream = self.load_data()

        # Initialize components
        window = SlidingWindow(size=self.args.window_size)
        preprocessor = DataPreprocessor(
            threshold=0.5,
            adaptive_threshold=True,
            freeze_after=self.args.preprocess_freeze_after,
            ema_alpha=self.args.preprocess_ema_alpha,
        )

        # STABILITY: Disable recurring detection if requested (for reproducibility testing)
        # Otherwise use the provided threshold
        if self.args.disable_recurring:
            recurring_threshold = 10.0  # Impossible to reach (disables recurring)
        else:
            recurring_threshold = self.args.recurring_threshold

        detector = FCADriftDetector(
            theta=self.args.theta,
            alpha=self.args.alpha,
            window_size=self.args.window_size,
            adaptive_window=self.args.adaptive_window,
            recurring_similarity_threshold=recurring_threshold,
            # Anti-noise parameters:
            # Require longer persistence to reduce repetitive micro-alerts.
            min_persistence_windows=max(3, self.args.window_size // 25),
            # Short cooldown suppresses immediate retriggers after one alarm.
            cooldown_windows=max(self.args.window_size // 5, 3),
            # More conservative spike path for abrupt-change alarms.
            spike_multiplier=1.8,
            noise_baseline_k=0.25,
            spike_min_z=1.8,
        )

        # Process stream
        drift_types_map = {}
        drift_types_signal_map = {}
        instance_count = 0

        for idx, (x, y) in enumerate(stream):
            x_for_window = dict(x)
            if self._is_supervised_mode():
                x_for_window['__target__'] = self._encode_target_for_context(y)

            window.append(x_for_window)
            instance_count += 1

            if window.is_full():
                # Preprocess window
                binary_data = preprocessor.preprocess_window(window.get_data())

                # FCA lattice complexity is exponential in n_attributes.
                # Cap at 15 attributes to keep runtime manageable for wide datasets
                # (e.g. credit_card has 30 features → too slow without capping).
                MAX_ATTRS = 15
                if binary_data.ndim == 2 and binary_data.shape[1] > MAX_ATTRS:
                    # Keep columns with highest variance (most informative for FCA)
                    col_var = binary_data.var(axis=0)
                    top_cols = col_var.argsort()[-MAX_ATTRS:]
                    binary_data = binary_data[:, top_cols]

                # Build lattice
                context = build_formal_context(binary_data)
                lattice = ConceptLattice()
                lattice.build_from_context(context)

                # Update detector
                event = detector.update(lattice, idx)

                # Store drift type from detector (already classified correctly)
                if event is not None:
                    drift_types_map[event.instance_id] = event.drift_type
                    if event.signal_idx is not None:
                        drift_types_signal_map[event.signal_idx] = event.drift_type

        self.detector = detector
        self.results['detection_results'] = {
            'total_instances': instance_count,
            'drifts_detected': len(detector.drift_indices),
            'drift_rate': f"{len(detector.drift_indices)/instance_count*100:.2f}%",
            'drift_indices': detector.drift_indices,
            'drift_signal_indices': detector.drift_signal_indices,
            'drift_types': drift_types_map,
            'drift_types_signal': drift_types_signal_map,
            'avg_delta_L': f"{sum(detector.delta_L_history)/len(detector.delta_L_history):.4f}" if detector.delta_L_history else 0,
            'max_delta_L': f"{max(detector.delta_L_history):.4f}" if detector.delta_L_history else 0,
        }

        print(f"[OK] Detection complete. Drifts detected: {len(detector.drift_indices)}")
        return detector, drift_types_map

    def determine_overall_drift_type(self, drift_types_map):
        """
        Determine overall drift type based on ACTUAL classified types,
        not on drift frequency percentages.

        Priority weights: sudden=5, recurring=4, gradual=3, incremental=2, unknown=1
        Winner = type with highest weighted score (count * priority).
        """
        if not drift_types_map:
            return 'no_drift', self.DRIFT_NAMES_SK['no_drift']

        # Count occurrences of each type
        type_counts: dict = {}
        for dtype in drift_types_map.values():
            clean = str(dtype).strip().lower()
            type_counts[clean] = type_counts.get(clean, 0) + 1

        # Weighted priority scoring
        priority = {
            'sudden':      5,
            'recurring':   4,
            'gradual':     3,
            'incremental': 2,
            'unknown':     1,
        }
        scores = {t: cnt * priority.get(t, 1) for t, cnt in type_counts.items()}
        dominant = max(scores, key=scores.__getitem__)

        name = self.DRIFT_NAMES_SK.get(dominant, dominant.capitalize())

        # Build breakdown string for display, e.g. "Sudden 35 (76.1%), Incremental 11 (23.9%)"
        total = sum(type_counts.values())
        breakdown = ', '.join(
            f"{t.capitalize()} {cnt} ({cnt/total*100:.1f}%)"
            for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1])
        )

        return dominant, f"{name} ({breakdown})"

    def _get_merge_gap(self) -> int:
        """Episode merge distance used consistently across reports and plots."""
        return max(self.args.window_size // 2, 5)

    def _build_ssot_aggregator(self):
        """Create DriftAggregatorV2 as single source of truth for drift episodes."""
        if not self.detector:
            return None

        raw_type_map = {
            int(event.signal_idx): str(event.drift_type).strip().lower()
            for event in getattr(self.detector, 'drift_events', [])
            if getattr(event, 'signal_idx', None) is not None
        }

        return DriftAggregatorV2(
            delta_L_history=self.detector.delta_L_history,
            similarity_history=self.detector.similarity_history,
            drift_signal_indices=self.detector.drift_signal_indices,
            signal_to_instance_id=self.detector.signal_to_instance_id,
            drift_types_raw=raw_type_map,
            window_size=self.args.window_size,
            warm_up_windows=5,
            alpha=self.args.alpha,
            n_adapt=10,
            merge_gap=self._get_merge_gap(),
            cooldown=0,
            intent_diffs=self.detector.intent_diffs if hasattr(self.detector, 'intent_diffs') else None,
        )

    def print_unified_report(self, detector, drift_types_map):
        """Print unified report in Slovak"""
        print("\n" + "="*80)
        print("JEDNOTNY REPORT DETEKCIE DRIFTU")
        print("Unified Drift Detection Report")
        print("="*80 + "\n")

        # Determine overall drift from SSOT episodes (fallback to raw points if unavailable).
        aggregator = self._build_ssot_aggregator()
        episode_count = 0
        episode_rate = 0.0
        merged_episodes = []

        if aggregator is not None:
            merged_episodes = aggregator.get_merged_episodes()
            episode_count = len(merged_episodes)
            episode_rate = aggregator.get_drift_rate()

            dominant_type = aggregator.get_dominant_type()
            if dominant_type == 'none':
                overall_type = 'no_drift'
                overall_name = self.DRIFT_NAMES_SK['no_drift']
            else:
                overall_type = dominant_type
                counts = aggregator.get_type_counts()
                total = max(sum(counts.values()), 1)
                breakdown = ', '.join(
                    f"{t.capitalize()} {cnt} ({cnt/total*100:.1f}%)"
                    for t, cnt in sorted(counts.items(), key=lambda x: -x[1])
                )
                localized = self.DRIFT_NAMES_SK.get(overall_type, overall_type.capitalize())
                overall_name = f"{localized} ({breakdown})" if breakdown else localized
        else:
            overall_type, overall_name = self.determine_overall_drift_type(drift_types_map)
            episode_count = self.results['detection_results']['drifts_detected']
            episode_rate = float(self.results['detection_results']['drift_rate'].strip('%'))

        print("[VYSLEDKY] RESULTS:")
        print("-" * 80)
        print(f"Dataset: {self.args.dataset if not self.args.custom_file else Path(self.args.custom_file).name}")
        print(f"Pocet instancii: {self.results['detection_results']['total_instances']}")
        print(f"Pocet surovych drift alarmov: {self.results['detection_results']['drifts_detected']}")
        print(f"Pocet zlucenych drift epizod (SSOT): {episode_count}")
        print(f"Frekvencia driftu (epizody): {episode_rate:.2f}%")
        print()

        # Overall result
        print("[ZAVER] CONCLUSION:")
        print("-" * 80)
        print(f"Typ driftu: {overall_name}")
        print(f"Specialne zakoncenie: {self._get_conclusion_sk(overall_type, episode_count)}")
        print()

        # Details (episode-level SSOT preferred, fallback to raw points).
        if merged_episodes:
            print("[DETAILY] DETAILS:")
            print("-" * 80)

            top_episodes = sorted(
                merged_episodes,
                key=lambda ep: getattr(ep, 'dlt_max', 0.0),
                reverse=True,
            )[:5]

            for rank, ep in enumerate(top_episodes, 1):
                print(
                    f"  {rank}. Episode #{ep.start_instance_id}-{ep.end_instance_id}: "
                    f"maxΔL={ep.dlt_max:.4f}, Typ: {ep.dominant_type}, raw_hits={ep.raw_hits_count}"
                )
        elif detector.drift_events:
            print("[DETAILY] DETAILS:")
            print("-" * 80)
            top_drifts = sorted(
                detector.drift_events,
                key=lambda e: e.metadata.get('delta_L', 0),
                reverse=True
            )[:5]

            for rank, event in enumerate(top_drifts, 1):
                delta_L = event.metadata.get('delta_L', 0)
                drift_type = event.drift_type
                print(f"  {rank}. Instance #{event.instance_id}: deltaL={delta_L:.4f}, Typ: {drift_type}")

        print("\n" + "="*80)

    def _get_conclusion_sk(self, drift_type, drift_count):
        """Get Slovak conclusion message"""
        if drift_type == 'no_drift':
            return "Zaden drift bol vykryty v toku dat."
        elif drift_type == 'sudden':
            return f"Zisteny prudky drift v {drift_count} bodoch. Pozadovana okamzita intervencia."
        elif drift_type == 'gradual':
            return f"Zisteny pozvolny drift v {drift_count} bodoch. Odporucuje sa monitoring."
        elif drift_type == 'incremental':
            return f"Zisteny nahromadjujuci drift v {drift_count} bodoch. Postupna zmena konceptu."
        elif drift_type == 'recurring':
            return f"Zisteny opakujuci sa drift v {drift_count} bodoch. System sa vraca k predchadzajucemu stavu."
        else:
            return f"Zisteny drift v {drift_count} bodoch. Typ: {drift_type}."

    def generate_reports(self):
        """
        Generate all reports using DriftAggregatorV2 as Single Source of Truth.

        Pipeline:
        1. Initialize aggregator (SSOT for all drift data)
        2. Get merged episodes from aggregator
        3. Generate diagnostics
        4. Export graphics (using merged episodes)
        5. Generate HTML report (using merged episodes)
        6. Save JSON results
        """
        if not self.detector:
            print("[ERROR] No detector results to report")
            return

        # Create output directory
        output_dir = Path(self.args.output) / self._get_run_name()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[AGGREGATION] Aggregating and classifying drifts (SSOT)...")

        # ========== SINGLE SOURCE OF TRUTH: DriftAggregatorV2 ==========
        aggregator = self._build_ssot_aggregator()
        if aggregator is None:
            print("[ERROR] Failed to initialize aggregator")
            return

        # Get merged episodes (authoritative data source)
        merged_episodes = aggregator.get_merged_episodes()
        episode_tuples = [ep.to_tuple() for ep in merged_episodes]
        diagnostics = aggregator.get_diagnostics()

        warm_up_instances = int(diagnostics.get('warm_up_instances', 0))
        drift_indices_for_reports = [
            idx for idx in self.detector.drift_signal_indices
            if idx >= warm_up_instances
        ]
        raw_drift_count = int(diagnostics.get('raw_drift_count', len(drift_indices_for_reports)))

        print(f"[OK] Aggregated {raw_drift_count} raw drifts into {len(merged_episodes)} episodes")
        print(f"[OK] Dominant type: {aggregator.get_dominant_type().upper()}")
        print(f"[OK] Drift rate: {aggregator.get_drift_rate():.2f}%")

        # ========== SAVE DIAGNOSTICS (must-have for traceability) ==========
        print("[DIAGNOSTICS] Saving diagnostic report...")
        try:
            diag_file = aggregator.save_diagnostics(output_dir, 'report_debug.json')
            print(f"[OK] Diagnostics saved to {diag_file.name}")
        except Exception as e:
            print(f"[WARNING] Diagnostics save failed: {e}")

        # ========== BUILD DRIFT TYPES MAP FROM MERGED EPISODES (SSOT) ==========
        # This ensures consistent data across all outputs (graphics, HTML, JSON)
        drift_types_for_viz = {}
        for ep in merged_episodes:
            for idx in range(ep.start, ep.end + 1):
                drift_types_for_viz[idx] = ep.dominant_type

        # ========== EXPORT THESIS GRAPHICS (using merged episode tuples) ==========
        print("\n[THESIS GRAPHICS] Exporting publication-ready graphs...")
        thesis_exporter = ThesisGraphicsExporter(output_dir, dpi=300)

        try:
            thesis_results = thesis_exporter.export_all(
                delta_L_history=self.detector.delta_L_history,
                similarity_history=self.detector.similarity_history,
                drift_indices=drift_indices_for_reports,
                drift_types=drift_types_for_viz,  # FIXED: Use SSOT type map
                merged_episodes=episode_tuples,  # Use tuple format
                window_size=self.args.window_size,
                theta=self.args.theta,
                alpha=self.args.alpha,
                adaptive_window=self.args.adaptive_window,
                warm_up_windows=5,
                merge_gap=self._get_merge_gap(),
                seed=42
            )
            print(f"[OK] Thesis graphics exported")
        except Exception as e:
            print(f"[WARNING] Graphics export failed: {e}")

        # ========== HTML REPORT GENERATION ==========
        print("[HTML REPORT] Generating detailed report...")
        lang_suffix = self.args.language
        html_path = output_dir / f"report_{lang_suffix}.html"
        dataset_id = self.args.dataset if not self.args.custom_file else "custom"

        try:
            DetailedReportGeneratorV2.generate_html_report(
                delta_L_history=self.detector.delta_L_history,
                similarity_history=self.detector.similarity_history,
                drift_indices=drift_indices_for_reports,
                drift_types=drift_types_for_viz,  # Use SSOT type map
                merged_episodes=episode_tuples,
                aggregator=aggregator,  # Pass aggregator for unified statistics
                snapshot_collector=self.detector.snapshot_collector if hasattr(self.detector, 'snapshot_collector') else None,
                output_path=html_path,
                language=self.args.language,
                image_dir=output_dir,
                lattice_history=self.detector.lattice_history,
                history_manager=self.detector.history_manager,
                lattice_objects=self.detector.lattice_objects,
                window_size=self.args.window_size,
                theta=self.args.theta,
                alpha=self.args.alpha,
                dataset_id=dataset_id,
                run_id=self._get_run_name()
            )
            print(f"[OK] HTML report generated: {html_path.name}")
        except Exception as e:
            import traceback
            print(f"[WARNING] HTML report generation failed: {e}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")

        # ========== SAVE JSON RESULTS ==========
        json_path = output_dir / "results.json"
        self.results['aggregated_episodes'] = len(merged_episodes)
        self.results['dominant_type'] = aggregator.get_dominant_type()
        self.results['drift_rate'] = aggregator.get_drift_rate()
        self.results['type_counts'] = aggregator.get_type_counts()

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n[OK] Reports generated successfully")
        print(f"[FILES]")
        print(f"  - Thesis graphics: {list(output_dir.glob('*_thesis.png'))}")
        print(f"  - HTML report: {html_path}")
        print(f"  - Diagnostics: {output_dir / 'report_debug.json'}")
        print(f"  - JSON results: {json_path}")

    def _get_run_name(self):
        """Generate run name from parameters"""
        parts = [
            self.args.dataset if not self.args.custom_file else "custom",
            f"w{self.args.window_size}",
            f"t{self.args.theta:.1f}",
            f"a{self.args.alpha:.1f}"
        ]
        return "_".join(parts)

    def _get_drift_types_map(self):
        """Get drift types from results"""
        return self.results['detection_results'].get('drift_types', {})


def _validate_arguments(args, parser):
    """Validate command-line arguments"""
    errors = []

    # Validate window size
    if args.window_size < 10:
        errors.append("--window-size must be at least 10 (got: {})".format(args.window_size))
    if args.window_size > 500:
        errors.append("--window-size must not exceed 500 (got: {})".format(args.window_size))

    # Validate theta
    if args.theta <= 0 or args.theta >= 1:
        errors.append("--theta must be between 0 and 1 (got: {})".format(args.theta))
    if args.theta < 0.1:
        errors.append("--theta recommended minimum is 0.1 (got: {})".format(args.theta))
    if args.theta > 0.9:
        errors.append("--theta recommended maximum is 0.9 (got: {})".format(args.theta))

    # Validate alpha
    if args.alpha <= 0:
        errors.append("--alpha must be positive (got: {})".format(args.alpha))
    if args.alpha < 0.5:
        errors.append("--alpha recommended minimum is 0.5 (got: {})".format(args.alpha))
    if args.alpha > 10:
        errors.append("--alpha recommended maximum is 10 (got: {})".format(args.alpha))

    # Validate preprocessing freeze/EMA settings
    if args.preprocess_freeze_after < 10:
        errors.append("--preprocess-freeze-after must be at least 10 (got: {})".format(args.preprocess_freeze_after))
    if args.preprocess_freeze_after > 50000:
        errors.append("--preprocess-freeze-after should not exceed 50000 (got: {})".format(args.preprocess_freeze_after))

    if args.preprocess_ema_alpha <= 0 or args.preprocess_ema_alpha > 1:
        errors.append("--preprocess-ema-alpha must be in (0, 1] (got: {})".format(args.preprocess_ema_alpha))

    # Validate max_instances
    if args.max_instances < 10:
        errors.append("--max-instances must be at least 10 (got: {})".format(args.max_instances))
    if args.max_instances > 100000:
        errors.append("--max-instances should not exceed 100000 (got: {})".format(args.max_instances))

    # Validate custom file if provided
    if args.custom_file:
        custom_path = Path(args.custom_file)
        if not custom_path.exists():
            errors.append("Custom file not found: {}".format(args.custom_file))
        if not custom_path.suffix.lower() == '.csv':
            errors.append("Custom file must be CSV format (got: {})".format(custom_path.suffix))

    # Validate language
    if args.language not in ['sk', 'uk', 'en']:
        errors.append("--language must be one of: sk, uk, en (got: {})".format(args.language))

    # Validate dataset choice when no custom file
    if not args.custom_file:
        available_datasets = get_available_datasets()
        if args.dataset.lower() not in [d.lower() for d in available_datasets]:
            errors.append("--dataset '{}' not available. Choose from: {}".format(
                args.dataset, ', '.join(available_datasets)))


    # Check if both custom file and dataset are specified
    if args.custom_file and args.dataset != 'agrawal':
        print("[WARNING] Both --custom-file and --dataset specified. Using --custom-file (ignoring --dataset)")

    if errors:
        print("\n[ERROR] Validation failed:")
        for error in errors:
            print(f"  - {error}")
        print()
        parser.print_help()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='FCA-Based Concept Drift Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset agrawal --max-instances 2000
  python main.py --dataset agrawal --window-size 75 --theta 0.3 --alpha 1.5
  python main.py --custom-file mydata.csv --window-size 50
        """
    )

    # Dataset options
    dataset_group = parser.add_argument_group('Dataset Options')

    # Get available datasets dynamically
    available_datasets = get_available_datasets()

    dataset_group.add_argument(
        '--dataset',
        type=str,
        default='agrawal',
        choices=available_datasets,
        help=f'Dataset name (default: agrawal). Available: {", ".join(available_datasets)}'
    )
    dataset_group.add_argument(
        '--custom-file',
        type=str,
        help='Path to custom CSV dataset'
    )
    dataset_group.add_argument(
        '--max-instances',
        type=int,
        default=500,
        help='Number of instances to process (default: 500)'
    )

    # Detection parameters
    param_group = parser.add_argument_group('Detection Parameters')
    param_group.add_argument(
        '--window-size',
        type=int,
        default=50,
        help='Sliding window size (default: 50)'
    )
    param_group.add_argument(
        '--theta',
        type=float,
        default=0.5,
        help='Drift threshold theta (default: 0.5) - Increased from 0.4 to reduce false positives'
    )
    param_group.add_argument(
        '--alpha',
        type=float,
        default=1.5,
        help='Adaptive multiplier alpha (default: 1.5) - Reduced from 2.0 for better precision'
    )
    param_group.add_argument(
        '--adaptive-window',
        type=int,
        default=10,
        help='Window size for rolling adaptive threshold (default: 10 windows)'
    )
    param_group.add_argument(
        '--preprocess-freeze-after',
        type=int,
        default=2500,
        help='Number of windows used to calibrate preprocessing thresholds before freezing (default: 2500)'
    )
    param_group.add_argument(
        '--preprocess-ema-alpha',
        type=float,
        default=0.05,
        help='EMA smoothing factor for preprocessing threshold calibration (default: 0.05)'
    )
    param_group.add_argument(
        '--recurring-threshold',
        type=float,
        default=0.85,
        help='Similarity threshold for recurring drift detection (0.0-1.0, default: 0.85). Higher = more strict, fewer recurring detected'
    )
    param_group.add_argument(
        '--disable-recurring',
        action='store_true',
        default=False,
        help='Disable recurring drift detection for reproducibility (default: False)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output',
        type=str,
        default='experiments/results',
        help='Output directory (default: experiments/results)'
    )
    output_group.add_argument(
        '--language',
        type=str,
        choices=['sk', 'uk', 'en'],
        default='sk',
        help='Output language (default: sk for Slovak)'
    )
    output_group.add_argument(
        '--context-mode',
        type=str,
        choices=['unsupervised_mode', 'supervised_mode'],
        default='unsupervised_mode',
        help='Context construction mode: unsupervised_mode (features only) or supervised_mode (features + target)'
    )

    args = parser.parse_args()

    # Validate arguments
    _validate_arguments(args, parser)

    # Run detection
    runner = DriftDetectionRunner(args)

    print("\n" + "="*80)
    print("FCA-BASED CONCEPT DRIFT DETECTION SYSTEM")
    print("Jednotny bod vstupu / Unified Entry Point")
    print("="*80 + "\n")

    try:
        # Run detection
        detector, drift_types_map = runner.run_detection()

        # Print unified report
        runner.print_unified_report(detector, drift_types_map)

        # Generate reports
        runner.generate_reports()

        print("\n[HOTOVO] Drift detection completed successfully!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
