"""
Grid Search Runner - Hyperparameter optimization
"""
import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from fca_drift.core import StreamReader, StreamWrapper, SlidingWindow, DataPreprocessor
from fca_drift.fca import build_formal_context, ConceptLattice
from fca_drift.detection import FCADriftDetector
from fca_drift.evaluation import DriftEvaluator
from fca_drift.utils import ExperimentLogger


class GridSearchRunner:
    """
    Grid search for optimal hyperparameters
    """

    def __init__(self, output_dir: Path = Path('experiments/results/grid_search')):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def run_experiment(self,
                      dataset_name: str,
                      window_size: int,
                      theta: float,
                      alpha: float,
                      max_instances: int = 1000,
                      true_drifts: List[int] = None) -> Dict:
        """
        Run single experiment with given parameters
        """
        # Initialize
        reader = StreamReader(dataset_name, seed=42)
        stream = StreamWrapper(reader.stream, max_instances=max_instances)
        window = SlidingWindow(size=window_size)
        preprocessor = DataPreprocessor()
        detector = FCADriftDetector(theta=theta, alpha=alpha, window_size=window_size)

        # Process stream
        for idx, (x, y) in enumerate(stream):
            window.append(x)

            if window.is_full():
                # Preprocess
                binary_data = preprocessor.preprocess_window(window.get_data())

                # Build lattice
                context = build_formal_context(binary_data)
                lattice = ConceptLattice()
                lattice.build_from_context(context)

                # Detect drift
                detector.update(lattice, idx)

        # Evaluate
        if true_drifts:
            evaluator = DriftEvaluator(true_drifts, tolerance=50)
            metrics = evaluator.evaluate(detector.drift_indices)
        else:
            # No ground truth - report detection count
            metrics = {
                'precision': -1.0,
                'recall': -1.0,
                'f1_score': -1.0,
                'true_positives': -1,
                'false_positives': -1,
                'false_negatives': -1,
                'detection_delay': -1.0,
                'drifts_detected': len(detector.drift_indices)
            }

        # Compile results
        result = {
            'dataset': dataset_name,
            'window_size': window_size,
            'theta': theta,
            'alpha': alpha,
            'max_instances': max_instances,
            **metrics
        }

        self.results.append(result)
        return result

    def run_grid_search(self,
                       dataset_name: str,
                       window_sizes: List[int],
                       thetas: List[float],
                       alphas: List[float],
                       max_instances: int = 1000,
                       true_drifts: List[int] = None):
        """
        Run grid search over parameter space
        """
        print(f"\nStarting Grid Search: {dataset_name}")
        print(f"Window sizes: {window_sizes}")
        print(f"Thetas: {thetas}")
        print(f"Alphas: {alphas}\n")

        combinations = itertools.product(window_sizes, thetas, alphas)
        total = len(window_sizes) * len(thetas) * len(alphas)

        for i, (w, t, a) in enumerate(combinations, 1):
            print(f"[{i}/{total}] Running: W={w}, theta={t}, alpha={a}...")

            try:
                result = self.run_experiment(
                    dataset_name=dataset_name,
                    window_size=w,
                    theta=t,
                    alpha=a,
                    max_instances=max_instances,
                    true_drifts=true_drifts
                )
                print(f"  -> F1: {result['f1_score']:.4f}, Drifts: {result.get('drifts_detected', len(result.get('delays', [])))}")
            except Exception as e:
                print(f"  ERROR: {e}")

    def save_results(self, filename: str = 'grid_search_results.csv'):
        """Save grid search results to CSV"""
        df = pd.DataFrame(self.results)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Also save as JSON
        json_path = self.output_dir / filename.replace('.csv', '.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def get_best_params(self) -> Dict:
        """Get parameters with best F1 score"""
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)
        # Filter out invalid results (no ground truth)
        df_valid = df[df['f1_score'] >= 0]

        if len(df_valid) == 0:
            return self.results[0]

        best_idx = df_valid['f1_score'].idxmax()
        return self.results[best_idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Grid Search for FCA Drift Detector')
    parser.add_argument('--dataset', type=str, default='agrawal')
    parser.add_argument('--max-instances', type=int, default=2000)
    parser.add_argument('--output-dir', type=str, default='experiments/results/grid_search')

    args = parser.parse_args()

    runner = GridSearchRunner(output_dir=Path(args.output_dir))

    # Define parameter grid
    window_sizes = [50, 100, 150]
    thetas = [0.3, 0.4, 0.5]
    alphas = [1.5, 2.0, 2.5]

    # Run grid search
    runner.run_grid_search(
        dataset_name=args.dataset,
        window_sizes=window_sizes,
        thetas=thetas,
        alphas=alphas,
        max_instances=args.max_instances,
        true_drifts=None  # No ground truth for agrawal by default
    )

    # Save results
    runner.save_results()

    # Print best params
    best = runner.get_best_params()
    print(f"\nBest parameters: {best}")
