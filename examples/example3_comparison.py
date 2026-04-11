"""
Example 3: Compare Multiple Detectors with Detailed Analysis
"""
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fca_drift.experiments import DetectorComparison
from fca_drift.visualization import create_analysis_summary, DriftAnalyzer
import matplotlib.pyplot as plt
import numpy as np


def plot_detector_comparison(results: dict, output_path: Path):
    """
    Create comparative visualization of detectors
    """
    detectors = list(results.keys())
    drift_counts = [results[d]['drifts_detected'] for d in detectors]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(detectors, drift_counts, color=colors, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, count in zip(bars, drift_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(count)}',
               ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel('Number of Drifts Detected', fontsize=12, fontweight='bold')
    ax.set_title('Drift Detection Comparison:\nFCA vs DDM vs EDDM',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Comparison plot saved to {output_path}")
    plt.close()


def main():
    print("\n" + "="*80)
    print("Example 3: Comparative Drift Detection Analysis")
    print("FCA vs DDM vs EDDM")
    print("="*80 + "\n")

    # Create comparator
    output_dir = Path(__file__).parent.parent / 'experiments/results/example3_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    print("1. Running comparative drift detection...\n")
    comparator = DetectorComparison(output_dir=output_dir)
    comparator.run_comparison(
        dataset_name='agrawal',
        max_instances=1000,
        window_size=50,
        true_drifts=None
    )

    # Get FCA detector data for visualization
    print("\n2. Generating visualizations and analysis...\n")

    # Access FCA detector results
    fca_results = comparator.results.get('FCA', {})
    if fca_results:
        delta_L_history = fca_results.get('delta_L_history', [])
        similarity_history = fca_results.get('similarity_history', [])
        drift_indices = fca_results.get('drift_indices', [])
        drift_types = fca_results.get('drift_types', {})

        # Create visualizations
        if delta_L_history and drift_indices:
            create_analysis_summary(
                delta_L_history,
                similarity_history,
                drift_indices,
                drift_types,
                output_dir / "fca_analysis"
            )

            # Generate detailed report
            report_path = output_dir / "fca_detailed_report.txt"
            DriftAnalyzer.generate_drift_report(
                drift_indices,
                drift_types,
                delta_L_history,
                similarity_history,
                report_path
            )

    # Save results and report
    comparator.save_results('detector_comparison.json')
    comparator.save_report('detector_comparison_report.txt')

    # Create comparison visualization
    # Get agrawal dataset results
    results_data = comparator.results.get('agrawal', {})
    plot_detector_comparison(results_data, output_dir / "detector_comparison_chart.png")

    # Print summary comparison
    print("\n3. Summary of Detector Performance:\n")
    print("   " + "-"*76)

    for detector_name, data in sorted(results_data.items()):
        drifts = data.get('drifts_detected', 0)
        print(f"   {detector_name:10} : {drifts:3} drifts detected")

    print("   " + "-"*76)

    # Analysis and explanation
    print("\n4. Analysis Summary:\n")

    fca_drifts = results_data.get('FCA', {}).get('drifts_detected', 0)
    ddm_drifts = results_data.get('DDM', {}).get('drifts_detected', 0)
    eddm_drifts = results_data.get('EDDM', {}).get('drifts_detected', 0)

    print("   📊 Key Findings:")
    print(f"   • FCA detected {fca_drifts} drifts - using full concept lattice comparison")
    print(f"   • DDM detected {ddm_drifts} drifts - error rate based detection")
    print(f"   • EDDM detected {eddm_drifts} drifts - distance between errors")

    if fca_drifts > ddm_drifts and fca_drifts > eddm_drifts:
        print("\n   💡 Interpretation:")
        print("   FCA detects more drifts because it analyzes the complete concept")
        print("   lattice structure, not just error rates. This includes:")
        print("   • Changes in object-attribute relationships")
        print("   • Modifications in concept hierarchy")
        print("   • Structural transformations in data organization")

    print("\n" + "="*80)
    print("[COMPLETE] Example 3 Complete!")
    print(f"📁 All results saved to: {output_dir}")
    print("\n   Generated files:")
    print("   • detector_comparison.json - Structured results")
    print("   • detector_comparison_report.txt - Text report")
    print("   • detector_comparison_chart.png - Comparison visualization")
    if fca_results:
        print("   • fca_analysis/ - FCA detailed plots and analysis")
        print("   • fca_detailed_report.txt - FCA comprehensive report")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
