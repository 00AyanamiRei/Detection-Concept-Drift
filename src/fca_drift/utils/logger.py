"""
Experiment Logger - CSV logging
"""
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List


class ExperimentLogger:
    """
    Logs experiment data to CSV and JSON
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.step_log = []
        self.drift_events = []

    def log_step(self, instance_id: int, similarity: float,
                 delta_L: float, drift_detected: bool):
        """Log each processing step"""
        self.step_log.append({
            'instance': instance_id,
            'similarity': similarity,
            'delta_L': delta_L,
            'drift': 1 if drift_detected else 0
        })

    def log_drift_event(self, drift_event):
        """Log drift detection event"""
        self.drift_events.append({
            'instance': drift_event.instance_id,
            'type': drift_event.drift_type,
            'confidence': drift_event.confidence,
            **drift_event.metadata
        })

    def save_to_csv(self):
        """Save logs to CSV files"""
        # Step log
        df_steps = pd.DataFrame(self.step_log)
        df_steps.to_csv(self.output_dir / 'steps.csv', index=False)

        # Drift events
        if self.drift_events:
            df_drifts = pd.DataFrame(self.drift_events)
            df_drifts.to_csv(self.output_dir / 'drifts.csv', index=False)

    def save_metrics(self, metrics: Dict):
        """Save evaluation metrics to JSON"""
        # Convert numpy types to Python types
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, (list, tuple)):
                metrics_serializable[key] = [float(v) for v in value]
            else:
                try:
                    metrics_serializable[key] = float(value)
                except:
                    metrics_serializable[key] = value

        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=2)


if __name__ == "__main__":
    print("Logger loaded")
