import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))

import numpy as np
from fca_drift.core import StreamReader, StreamWrapper, SlidingWindow, DataPreprocessor
from fca_drift.fca import build_formal_context, ConceptLattice, LatticeSimilarityCalculator

reader = StreamReader('agrawal', seed=42, drift_positions=[5000])
stream = StreamWrapper(reader.stream, max_instances=10000)

window = SlidingWindow(size=300)
preprocessor = DataPreprocessor(threshold=0.5, adaptive_threshold=True,
                                 freeze_after=2500, ema_alpha=0.05)
sim_calc = LatticeSimilarityCalculator()

prev_lattice = None
deltas = []

for idx, (x, y) in enumerate(stream):
    window.append(dict(x))
    if not window.is_full():
        continue

    binary = preprocessor.preprocess_window(window.get_data())
    if binary.shape[1] > 15:
        cols = binary.var(axis=0).argsort()[-15:]
        binary = binary[:, cols]

    ctx = build_formal_context(binary)
    lat = ConceptLattice()
    lat.build_from_context(ctx, max_objects=50)

    if prev_lattice is not None:
        sim = sim_calc.compute_similarity(prev_lattice, lat)
        delta = 1.0 - sim
        deltas.append((idx, delta))

    prev_lattice = lat

vals = [d for _, d in deltas]
print(f"delta_L: min={min(vals):.4f}, max={max(vals):.4f}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

# Show around drift
near = [(i,d) for i,d in deltas if 4700 <= i <= 5300]
print(f"\nNear drift (4700-5300), every 20th:")
for i, d in near[::20]:
    print(f"  idx={i}: delta={d:.4f}")

# Before vs after
before = [d for i,d in deltas if 2000 <= i <= 4500]
after  = [d for i,d in deltas if 5500 <= i <= 8000]
print(f"\nBefore drift (2000-4500): mean={np.mean(before):.4f}")
print(f"After  drift (5500-8000): mean={np.mean(after):.4f}")
