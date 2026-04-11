"""
Evaluation Metrics for Drift Detection
"""
import numpy as np
from typing import List, Tuple, Dict


class DriftEvaluator:
    """
    Evaluates drift detection performance
    """

    def __init__(self, true_drift_points: List[int], tolerance: int = 50):
        """
        Args:
            true_drift_points: Ground truth drift locations
            tolerance: Acceptable delay (instances)
        """
        self.true_drift_points = sorted(true_drift_points)
        self.tolerance = tolerance

    def evaluate(self, detected_drifts: List[int]) -> Dict:
        """
        Compute all metrics

        Returns:
            dict with precision, recall, F1, detection_delay
        """
        detected_drifts = sorted(detected_drifts)

        TP, FP, FN, delays = self._compute_confusion_matrix(detected_drifts)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        avg_delay = np.mean(delays) if delays else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': TP,
            'false_positives': FP,
            'false_negatives': FN,
            'detection_delay': avg_delay,
            'delays': delays
        }

    def _compute_confusion_matrix(self, detected: List[int]) -> Tuple:
        """
        TP: Detection within tolerance of true drift
        FP: Detection with no true drift nearby
        FN: Missed true drift
        """
        TP = 0
        FP = 0
        FN = 0
        delays = []

        matched_true = set()
        matched_detected = set()

        # Find TPs
        for true_point in self.true_drift_points:
            closest = self._find_closest(true_point, detected)

            if closest is not None:
                distance = abs(closest - true_point)
                if distance <= self.tolerance:
                    TP += 1
                    delays.append(closest - true_point)
                    matched_true.add(true_point)
                    matched_detected.add(closest)
                else:
                    FN += 1
            else:
                FN += 1

        # Count FPs
        for det in detected:
            if det not in matched_detected:
                FP += 1

        return TP, FP, FN, delays

    @staticmethod
    def _find_closest(point: int, candidates: List[int]):
        """Find closest candidate to point"""
        if not candidates:
            return None
        return min(candidates, key=lambda x: abs(x - point))


def print_evaluation_report(metrics: Dict, method_name: str = "FCA"):
    """Pretty-print evaluation report"""
    print(f"\n{'='*60}")
    print(f"  Evaluation Report: {method_name}")
    print(f"{'='*60}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  F1-Score:         {metrics['f1_score']:.4f}")
    print(f"  Detection Delay:  {metrics['detection_delay']:.2f} instances")
    print(f"{'='*60}")
    print(f"  True Positives:   {metrics['true_positives']}")
    print(f"  False Positives:  {metrics['false_positives']}")
    print(f"  False Negatives:  {metrics['false_negatives']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test
    evaluator = DriftEvaluator(true_drift_points=[100, 500, 900])
    detected = [105, 495, 910, 300]  # 300 is FP

    metrics = evaluator.evaluate(detected)
    print_evaluation_report(metrics)
