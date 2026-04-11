"""
Stream Reader - River integration + fallback generators

Fixes applied:
- Agrawal/SEA: ManualDriftStream instead of ConceptDriftStream (math overflow fix)
- SEA: variant 0->3 for clearer structural drift in concept lattice
- SPAM: UTF-8 encoding wrapper (fixes Windows cp1251 error)
- INSECTS: removed (HTTP 404 on river server)
- MV: removed (does not exist in river)
- Mixed: removed unsupported n_features parameter
"""
from typing import Optional
from collections.abc import Iterable

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
class _ManualDriftAgrawal:
    """Agrawal with sudden drift at drift_position: function 0 -> 2."""
    def __init__(self, seed: int, drift_position: int = 5000):
        self.seed           = seed
        self.drift_position = drift_position

    def __iter__(self):
        s1 = synth.Agrawal(classification_function=0,
                           seed=self.seed,     perturbation=0.05)
        s2 = synth.Agrawal(classification_function=2,
                           seed=self.seed + 1, perturbation=0.05)
        for i, (x, y) in enumerate(s1):
            if i >= self.drift_position:
                break
            yield x, y
        for x, y in s2:
            yield x, y


class _ManualDriftSEA:
    """
    SEA with sudden drift at drift_position: variant 0 -> 3.
    Variants 0 and 3 have the most different decision boundaries,
    producing a clearer structural change in the concept lattice.
    """
    def __init__(self, seed: int, drift_position: int = 5000):
        self.seed           = seed
        self.drift_position = drift_position

    def __iter__(self):
        s1 = synth.SEA(seed=self.seed,     variant=0)
        s2 = synth.SEA(seed=self.seed + 1, variant=3)
        for i, (x, y) in enumerate(s1):
            if i >= self.drift_position:
                break
            yield x, y
        for x, y in s2:
            yield x, y


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
                    seed=self.seed, drift_position=drift_position)

            if dataset_lower == 'sea':
                return _ManualDriftSEA(
                    seed=self.seed, drift_position=drift_position)

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
