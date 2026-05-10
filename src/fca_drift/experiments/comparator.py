"""
Detector Comparison - Compare FCA vs baselines (DDM, EDDM, ADWIN)
"""
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from fca_drift.core import StreamReader, StreamWrapper, SlidingWindow, DataPreprocessor
from fca_drift.fca import build_formal_context, ConceptLattice
from fca_drift.detection import FCADriftDetector, DDM, EDDM
from fca_drift.evaluation import DriftEvaluator, print_evaluation_report

try:
    from river import naive_bayes
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


class DetectorComparison:
    """
    Compare multiple drift detectors
    """

    def __init__(self, output_dir: Path = Path('experiments/results/comparison')):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run_comparison(self,
                      dataset_name: str,
                      max_instances: int = 10000,
                      window_size: int = 300,
                      true_drifts: List[int] = None):
        """
        Run all detectors on same stream
        """
        print(f"\n{'='*70}")
        print(f"  Comparison: {dataset_name}")
        print(f"{'='*70}\n")

        # Initialize stream and preprocessing
        reader_kwargs = {}
        if dataset_name.lower() in {"agrawal", "sea"}:
            reader_kwargs["drift_positions"] = [max_instances // 2]
        reader = StreamReader(dataset_name, seed=42, **reader_kwargs)
        stream = StreamWrapper(reader.stream, max_instances=max_instances)

        # Initialize detectors
        fca_detector = FCADriftDetector(theta=0.5, alpha=1.5, window_size=window_size)
        ddm_detector = DDM(min_instances=30, warning_level=2.0, drift_level=3.0)
        eddm_detector = EDDM(min_instances=30, warning_level=0.95, drift_level=0.90)

        # Initialize preprocessing
        window = SlidingWindow(size=window_size)
        preprocessor = DataPreprocessor()

        # Process stream
        print("Processing stream...")
        if RIVER_AVAILABLE:
            model = naive_bayes.GaussianNB()
        else:
            model = None
            running_mean = None

        for idx, (x, y) in enumerate(stream):
            window.append(x)

            # FCA detector (needs full window)
            if window.is_full():
                binary_data = preprocessor.preprocess_window(window.get_data())
                context = build_formal_context(binary_data)
                lattice = ConceptLattice()
                lattice.build_from_context(context)
                fca_detector.update(lattice, idx)

            # DDM and EDDM (need error signal)
            if model is not None and y is not None:
                y_pred = model.predict_one(x)
                if y_pred is None:
                    y_pred = 0
                error = (y_pred != y)
                model.learn_one(x, y)
            else:
                feature_sum = float(sum(x.values()))
                if running_mean is None:
                    running_mean = feature_sum
                error = (feature_sum > running_mean) != bool(y) if y is not None else False
                running_mean = 0.9 * running_mean + 0.1 * feature_sum
            ddm_detector.update(error, idx)
            eddm_detector.update(error, idx)

        # Evaluate
        results_dict = {}

        if true_drifts:
            print("\n--- FCA Detector ---")
            evaluator = DriftEvaluator(true_drifts, tolerance=50)
            fca_metrics = evaluator.evaluate(fca_detector.drift_indices)
            print_evaluation_report(fca_metrics, method_name="FCA")
            results_dict['FCA'] = fca_metrics

            print("\n--- DDM Detector ---")
            ddm_metrics = evaluator.evaluate(ddm_detector.drift_indices)
            print_evaluation_report(ddm_metrics, method_name="DDM")
            results_dict['DDM'] = ddm_metrics

            print("\n--- EDDM Detector ---")
            eddm_metrics = evaluator.evaluate(eddm_detector.drift_indices)
            print_evaluation_report(eddm_metrics, method_name="EDDM")
            results_dict['EDDM'] = eddm_metrics
        else:
            print("\n--- Drift Detection Results (No Ground Truth) ---")
            print(f"FCA:  {len(fca_detector.drift_indices)} drifts detected")
            print(f"DDM:  {len(ddm_detector.drift_indices)} drifts detected")
            print(f"EDDM: {len(eddm_detector.drift_indices)} drifts detected")

            results_dict['FCA'] = {'drifts_detected': len(fca_detector.drift_indices)}
            results_dict['DDM'] = {'drifts_detected': len(ddm_detector.drift_indices)}
            results_dict['EDDM'] = {'drifts_detected': len(eddm_detector.drift_indices)}

        self.results[dataset_name] = results_dict
        return results_dict

    def save_results(self, filename: str = 'comparison_results.json'):
        """Save comparison results"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    def generate_report(self) -> str:
        """Generate comparison report"""
        report = "\n" + "="*70 + "\n"
        report += "DETECTOR COMPARISON REPORT\n"
        report += "="*70 + "\n\n"

        for dataset, detectors in self.results.items():
            report += f"\nDataset: {dataset}\n"
            report += "-" * 70 + "\n"

            for detector_name, metrics in detectors.items():
                report += f"\n{detector_name}:\n"

                if 'f1_score' in metrics:
                    report += f"  Precision:       {metrics['precision']:.4f}\n"
                    report += f"  Recall:          {metrics['recall']:.4f}\n"
                    report += f"  F1-Score:        {metrics['f1_score']:.4f}\n"
                    report += f"  Detection Delay: {metrics['detection_delay']:.2f}\n"
                    report += f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}\n"
                else:
                    report += f"  Drifts Detected: {metrics.get('drifts_detected', 'N/A')}\n"

        report += "\n" + "="*70 + "\n"
        return report

    def save_report(self, filename: str = 'comparison_report.txt'):
        """Save comparison report to file"""
        report = self.generate_report()
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
        print(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare Drift Detectors')
    parser.add_argument('--dataset', type=str, default='agrawal')
    parser.add_argument('--max-instances', type=int, default=10000)
    parser.add_argument('--window-size', type=int, default=300)
    parser.add_argument('--output-dir', type=str, default='experiments/results/comparison')

    args = parser.parse_args()

    comparator = DetectorComparison(output_dir=Path(args.output_dir))

    # Run comparison
    comparator.run_comparison(
        dataset_name=args.dataset,
        max_instances=args.max_instances,
        window_size=args.window_size,
        true_drifts=None
    )

    # Save results
    comparator.save_results()
    comparator.save_report()
