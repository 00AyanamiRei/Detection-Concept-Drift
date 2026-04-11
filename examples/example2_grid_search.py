"""
Example 2: Grid Search for Optimal Parameters
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fca_drift.experiments import GridSearchRunner


def main():
    print("\n" + "="*70)
    print("Example 2: Grid Search for Optimal Parameters")
    print("="*70 + "\n")

    # Create grid search runner
    output_dir = Path(__file__).parent.parent / 'experiments/results/example_grid_search'
    runner = GridSearchRunner(output_dir=output_dir)

    # Define parameter grid
    window_sizes = [50, 75]
    thetas = [0.3, 0.4]
    alphas = [1.5, 2.0]

    print("Running grid search with parameters:")
    print(f"  Window sizes: {window_sizes}")
    print(f"  Thetas (threshold): {thetas}")
    print(f"  Alphas (adaptive multiplier): {alphas}\n")

    # Run grid search
    runner.run_grid_search(
        dataset_name='agrawal',
        window_sizes=window_sizes,
        thetas=thetas,
        alphas=alphas,
        max_instances=1000
    )

    # Save and display results
    runner.save_results()

    best = runner.get_best_params()
    if 'f1_score' in best and best['f1_score'] >= 0:
        print("\n" + "="*70)
        print("Best Parameters Found:")
        print("="*70)
        print(f"  Window Size: {best['window_size']}")
        print(f"  Theta: {best['theta']}")
        print(f"  Alpha: {best['alpha']}")
        print(f"  F1-Score: {best['f1_score']:.4f}")
        print("="*70 + "\n")
    else:
        print("\nNote: No ground truth drifts available for agrawal dataset")
        print("Results show drift detection counts")

    print("Example 2 Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
