"""
Stream Reader - River integration + fallback generators

Fixes applied:
- Agrawal/SEA: Manual drift streams instead of ConceptDriftStream (math overflow fix)
- SEA: variant 0->3 for clearer structural drift in concept lattice
- SPAM: UTF-8 encoding wrapper (fixes Windows cp1251 error)
- INSECTS: removed (HTTP 404 on river server)
- MV: removed (does not exist in river)
- Mixed: removed unsupported n_features parameter
"""
from typing import Optional
from collections.abc import Iterable
import re

try:
    from river import datasets
    try:
        from river.datasets import synth
        SYNTH_AVAILABLE = True
    except ImportError:
        SYNTH_AVAILABLE = False
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    SYNTH_AVAILABLE = False
    print("[WARNING] River not available - using fallback generators")

import numpy as np


# ---------------------------------------------------------------------------
# Encoding-safe wrapper for SMSSpam (Windows cp1251 -> UTF-8 fix)
# ---------------------------------------------------------------------------
class _UTF8SMSSpam:
    """
    Wraps river SMSSpam and reads CSV with UTF-8 encoding
    to avoid cp1251 decode errors on Windows.
    """
    def __iter__(self):
        import csv
        import pathlib
        spam = datasets.SMSSpam()
        try:
            spam.download(verbose=False)
        except Exception:
            pass
        csv_path = pathlib.Path(spam.path)
        if not csv_path.exists():
            import river
            cache = pathlib.Path(river.__file__).parent / 'datasets' / 'data'
            for p in cache.rglob('*.csv'):
                if 'spam' in p.name.lower() or 'sms' in p.name.lower():
                    csv_path = p
                    break
        with open(csv_path, encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get('label', row.get('v1', '0'))
                text  = row.get('text',  row.get('v2', ''))
                x = {'text_length': len(text), 'word_count': len(text.split())}
                y = 1 if str(label).strip().lower() in ('spam', '1') else 0
                yield x, y


# ---------------------------------------------------------------------------
# Manual drift streams (avoid ConceptDriftStream math.exp overflow)
# ---------------------------------------------------------------------------
def _normalize_drift_positions(raw_positions, fallback_position: int) -> list[int]:
    if raw_positions is None:
        return [int(fallback_position)]
    if isinstance(raw_positions, str):
        parts = re.split(r"[,;\s]+", raw_positions.strip())
        positions = [int(p) for p in parts if p]
    else:
        positions = [int(p) for p in raw_positions]
    return sorted({p for p in positions if p > 0})


class _ManualDriftAgrawal:
    """Agrawal with one or more sudden drifts.

    FIX: changed from function 0<->2 to function 0<->7.
    Functions 0 and 7 have maximally different decision rules in Agrawal:
    - Function 0: salary > 50000 AND commission > 0
    - Function 7: age > 60 OR salary > 75000
    This produces a large structural change in the FCA concept lattice,
    making the drift clearly visible as a spike in delta_L.

    perturbation reduced to 0.0 so the boundary is sharp (sudden drift).
    """
    def __init__(self, seed: int, drift_positions: list[int]):
        self.seed = seed
        self.drift_positions = sorted(drift_positions)

    def __iter__(self):
        functions = [0, 7]          # FIX: was [0, 2]
        positions = list(self.drift_positions)
        current = 0
        segments = positions + [None]

        for seg_idx, stop in enumerate(segments):
            func = functions[seg_idx % len(functions)]
            generator = synth.Agrawal(
                classification_function=func,
                seed=self.seed + seg_idx,
                perturbation=0.0,   # FIX: was 0.05 — sharp boundary for sudden drift
            )

            if stop is None:
                for x, y in generator:
                    yield x, y
                break

            length = max(stop - current, 0)
            if length == 0:
                continue

            for i, (x, y) in enumerate(generator):
                if i >= length:
                    break
                yield x, y
                current += 1


class _ManualDriftSEA:
    """
    SEA with one or more sudden drifts (variant 0 <-> 3).
    Variants 0 and 3 have the most different decision boundaries,
    producing a clearer structural change in the concept lattice.
    """
    def __init__(self, seed: int, drift_positions: list[int]):
        self.seed = seed
        self.drift_positions = sorted(drift_positions)

    def __iter__(self):
        variants = [0, 3]
        positions = list(self.drift_positions)
        current = 0
        segments = positions + [None]

        for seg_idx, stop in enumerate(segments):
            variant = variants[seg_idx % len(variants)]
            generator = synth.SEA(seed=self.seed + seg_idx, variant=variant)

            if stop is None:
                for x, y in generator:
                    yield x, y
                break

            length = max(stop - current, 0)
            if length == 0:
                continue

            for i, (x, y) in enumerate(generator):
                if i >= length:
                    break
                yield x, y
                current += 1


# ---------------------------------------------------------------------------
# StreamReader
# ---------------------------------------------------------------------------
class StreamReader:
    """Unified interface for data streams."""

    def __init__(self, dataset_name: str, seed: int = 42, **kwargs):
        self.dataset_name = dataset_name
        self.seed         = seed
        self.kwargs       = kwargs
        self.stream       = self._load_stream()

    def _load_stream(self):
        if RIVER_AVAILABLE:
            return self._load_river_stream()
        return self._load_fallback_stream()

    def _load_river_stream(self):
        dataset_lower  = self.dataset_name.lower()
        drift_position = self.kwargs.get('drift_position', 5000)
        drift_positions = _normalize_drift_positions(
            self.kwargs.get('drift_positions', None),
            drift_position,
        )

        # ── Real datasets ────────────────────────────────────────────────────
        real_datasets = {
            'elec2':       datasets.Elec2,
            'bikes':       datasets.Bikes,
            'credit_card': datasets.CreditCard,
            'credit-card': datasets.CreditCard,
            'http':        datasets.HTTP,
            'keystroke':   datasets.Keystroke,
            'phishing':    datasets.Phishing,
            'restaurants': datasets.Restaurants,
            'taxis':       datasets.Taxis,
            'bananas':     datasets.Bananas,
            'water_flow':  datasets.WaterFlow,
            'water-flow':  datasets.WaterFlow,
        }

        if dataset_lower in real_datasets:
            return real_datasets[dataset_lower]()

        # SPAM — UTF-8 safe wrapper
        if dataset_lower in ('spam', 'sms_spam', 'sms-spam'):
            return _UTF8SMSSpam()

        # ── Synthetic datasets ───────────────────────────────────────────────
        if SYNTH_AVAILABLE:
            if dataset_lower == 'agrawal':
                return _ManualDriftAgrawal(
                    seed=self.seed, drift_positions=drift_positions)

            if dataset_lower == 'sea':
                return _ManualDriftSEA(
                    seed=self.seed, drift_positions=drift_positions)

            if dataset_lower == 'hyperplane':
                return synth.Hyperplane(
                    seed=self.seed,
                    n_features=self.kwargs.get('n_features', 10))

            if dataset_lower == 'mixed':
                return synth.Mixed(seed=self.seed)

            if dataset_lower == 'waveform':
                return synth.Waveform(seed=self.seed)

        raise ValueError(
            f"Unknown dataset: '{self.dataset_name}'. "
            f"Available: {', '.join(get_available_datasets())}"
        )

    def _load_fallback_stream(self):
        if self.dataset_name == 'agrawal':
            return SyntheticAgrawalGenerator(seed=self.seed)
        if self.dataset_name == 'elec2':
            return SyntheticElec2Generator(seed=self.seed)
        raise ValueError(
            f"Dataset '{self.dataset_name}' not available without River. "
            "Install with: pip install river"
        )


# ---------------------------------------------------------------------------
# StreamWrapper
# ---------------------------------------------------------------------------
class StreamWrapper:
    """Iterator wrapper with instance limit."""

    def __init__(self, stream: Iterable, max_instances: Optional[int] = None):
        self.stream        = stream
        self.max_instances = max_instances
        self.count         = 0

    def __iter__(self):
        for x, y in self.stream:
            if self.max_instances and self.count >= self.max_instances:
                break
            yield x, y
            self.count += 1


# ---------------------------------------------------------------------------
# Fallback generators (used when river is not installed)
# ---------------------------------------------------------------------------
class SyntheticAgrawalGenerator:
    def __init__(self, seed: int = 42, n_features: int = 9):
        self.rng        = np.random.RandomState(seed)
        self.n_features = n_features
        self.count      = 0

    def __iter__(self):
        while True:
            x = {f'x{i}': self.rng.rand() for i in range(self.n_features)}
            y = int(sum(x.values()) > self.n_features / 2)
            yield x, y
            self.count += 1


class SyntheticElec2Generator:
    def __init__(self, seed: int = 42, n_features: int = 8):
        self.rng        = np.random.RandomState(seed)
        self.n_features = n_features
        self.count      = 0

    def __iter__(self):
        while True:
            x = {
                f'feature{i}': self.rng.rand() + 0.1 * np.sin(self.count / 100)
                for i in range(self.n_features)
            }
            y = int(sum(x.values()) > self.n_features / 2)
            yield x, y
            self.count += 1


# ---------------------------------------------------------------------------
# Available datasets list (used by argparse choices)
# ---------------------------------------------------------------------------
def get_available_datasets() -> list:
    """Return list of all available dataset names."""
    if RIVER_AVAILABLE:
        real = [
            'elec2',       'bikes',    'credit_card', 'http',
            'keystroke',   'phishing', 'restaurants', 'spam',
            'taxis',       'bananas',  'water_flow',
        ]
        synth_list = [
            'agrawal', 'sea', 'hyperplane', 'mixed', 'waveform',
        ] if SYNTH_AVAILABLE else []
        return real + synth_list
    return ['agrawal', 'elec2']


if __name__ == "__main__":
    reader  = StreamReader('agrawal', seed=42)
    wrapper = StreamWrapper(reader.stream, max_instances=5)
    for i, (x, y) in enumerate(wrapper):
        print(f"Instance {i}: y={y}, features={list(x.keys())[:3]}...")
