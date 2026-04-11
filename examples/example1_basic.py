"""
Example 1: Basic FCA Drift Detection with Detailed Analysis
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fca_drift.core import StreamReader, StreamWrapper, SlidingWindow, DataPreprocessor
from fca_drift.fca import build_formal_context, ConceptLattice
from fca_drift.detection import FCADriftDetector
from fca_drift.visualization import (
    DriftAnalyzer, print_drift_analysis, create_analysis_summary
)
from fca_drift.visualization.detailed_report import DetailedReportGenerator
import numpy as np


def main():
    print("\n" + "="*80)
    print("Example 1: FCA Drift Detection with Detailed Analysis")
    print("="*80 + "\n")

    # 1. Create data stream
    print("1. Loading data stream (Agrawal)...")
    reader = StreamReader('agrawal', seed=42)
    stream = StreamWrapper(reader.stream, max_instances=500)
    print("   [OK]\n")

    # 2. Create window
    print("2. Creating sliding window (size=50)...")
    window = SlidingWindow(size=50)
    print("   [OK]\n")

    # 3. Create preprocessor
    print("3. Creating data preprocessor...")
    preprocessor = DataPreprocessor(threshold=0.5, adaptive_threshold=True)
    print("   [OK]\n")

    # 4. Create detector
    print("4. Creating FCA drift detector...")
    detector = FCADriftDetector(theta=0.4, alpha=2.0)
    print("   [OK]\n")

    # 5. Process stream
    print("5. Processing stream with detailed tracking...\n")

    drift_count = 0
    drift_types_map = {}

    for idx, (x, y) in enumerate(stream):
        window.append(x)

        if window.is_full():
            # Preprocess window to binary matrix
            binary_data = preprocessor.preprocess_window(window.get_data())

            # Build formal context and concept lattice
            context = build_formal_context(binary_data)
            lattice = ConceptLattice()
            lattice.build_from_context(context)

            # Detect drift
            drift_event = detector.update(lattice, idx)

            if drift_event:
                drift_count += 1
                if drift_event.signal_idx is None:
                    continue
                drift_types_map[drift_event.signal_idx] = drift_event.drift_type

                # Use analyzer to classify drift type
                classified_type = DriftAnalyzer.classify_drift_type(
                    drift_event.metadata['delta_L'],
                    detector.delta_L_history
                )
                drift_types_map[drift_event.signal_idx] = classified_type

                print(f"   [DRIFT {drift_count:2d}] Instance {idx:3d} (signal={drift_event.signal_idx:3d}): "
                      f"ΔL={drift_event.metadata['delta_L']:.4f} "
                      f"Type: {classified_type:12s} "
                      f"Sim={drift_event.metadata['similarity']:.4f}")

    # 6. Report results
    print(f"\n6. Summary Statistics:")
    print(f"   " + "-"*76)
    print(f"   Total instances processed: {stream.count}")
    print(f"   Drifts detected: {drift_count}")
    print(f"   Drift rate: {drift_count/stream.count*100:.2f}%")
    print(f"   Avg similarity: {sum(detector.similarity_history)/len(detector.similarity_history):.4f}")
    print(f"   Drift instance IDs: {detector.drift_indices}")
    print(f"   Drift signal indices: {detector.drift_signal_indices}")

    # 7. Detailed analysis
    print(f"\n7. Detailed Drift Analysis:")
    print_drift_analysis(
        detector.drift_signal_indices,
        drift_types_map,
        detector.delta_L_history,
        detector.similarity_history
    )

    # 8. Generate visualizations
    print(f"\n8. Generating visualizations...")
    output_dir = Path(__file__).parent.parent / "experiments" / "results" / "example1_detailed"
    output_dir.mkdir(parents=True, exist_ok=True)

    create_analysis_summary(
        detector.delta_L_history,
        detector.similarity_history,
        detector.drift_signal_indices,
        drift_types_map,
        output_dir
    )

    # Save detailed report
    report_path = output_dir / "detailed_analysis_report.txt"
    DriftAnalyzer.generate_drift_report(
        detector.drift_signal_indices,
        drift_types_map,
        detector.delta_L_history,
        detector.similarity_history,
        report_path
    )

    # Generate HTML report
    html_path = output_dir / "detailed_analysis_report.html"
    DetailedReportGenerator.generate_html_report(
        detector.delta_L_history,
        detector.similarity_history,
        detector.drift_signal_indices,
        drift_types_map,
        html_path,
        language='uk'
    )

    print(f"\n9. Output Files:")
    print(f"   " + "-"*76)
    print(f"   📊 All visualizations saved to: {output_dir}")
    print(f"   📄 Report saved to: {report_path}")
    print(f"   🌐 HTML Report saved to: {html_path}")
    print(f"\n   Generated files:")
    print(f"      • 01_delta_L_analysis.png - ΔL time series with drift markers")
    print(f"      • 02_drift_distribution.png - Drift type distribution")
    print(f"      • 03_similarity_vs_delta.png - Lattice similarity analysis")
    print(f"      • detailed_analysis_report.txt - Comprehensive text report")
    print(f"      • detailed_analysis_report.html - Interactive HTML report")

    print("\n" + "="*80)
    print("[COMPLETE] Example 1 Complete! Check output files for detailed analysis.")
    print("[INFO] Open detailed_analysis_report.html in browser for interactive view")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
