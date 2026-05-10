# debug_elec2.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))

import numpy as np
from fca_drift.core import StreamReader, StreamWrapper, SlidingWindow, DataPreprocessor
from fca_drift.fca import build_formal_context, ConceptLattice, LatticeSimilarityCalculator

def test_elec2(freeze_after, ema_alpha, label):
    reader = StreamReader('elec2', seed=42)
    stream = StreamWrapper(reader.stream, max_instances=5000)
    window = SlidingWindow(size=300)
    pre = DataPreprocessor(threshold=0.5, adaptive_threshold=True,
                           freeze_after=freeze_after, ema_alpha=ema_alpha)
    sim_calc = LatticeSimilarityCalculator()
    prev_lattice = None
    deltas = []

    for idx, (x, y) in enumerate(stream):
        window.append(dict(x))
        if not window.is_full():
            continue
        binary = pre.preprocess_window(window.get_data())
        if binary.shape[1] > 15:
            cols = binary.var(axis=0).argsort()[-15:]
            binary = binary[:, cols]
        ctx = build_formal_context(binary)
        lat = ConceptLattice()
        lat.build_from_context(ctx, max_objects=50)
        if prev_lattice is not None:
            sim = sim_calc.compute_similarity(prev_lattice, lat)
            deltas.append((idx, 1.0 - sim))
        prev_lattice = lat

    vals = [d for _, d in deltas]
    above_03 = sum(1 for v in vals if v > 0.3)
    above_05 = sum(1 for v in vals if v > 0.5)
    print(f"{label}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
          f"max={max(vals):.4f}, >0.3: {above_03}, >0.5: {above_05}")

print("Testing Elec2 with different preprocessing settings:")
test_elec2(freeze_after=9999999, ema_alpha=0.3,  label="no_freeze  ema=0.3")
test_elec2(freeze_after=9999999, ema_alpha=0.05, label="no_freeze  ema=0.05")
test_elec2(freeze_after=500,     ema_alpha=0.3,  label="freeze=500 ema=0.3")
test_elec2(freeze_after=2500,    ema_alpha=0.05, label="freeze=2500 ema=0.05 (current)")
