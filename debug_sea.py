# save as debug_sea.py and run: python debug_sea.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))

import numpy as np
from fca_drift.core import StreamReader, StreamWrapper, SlidingWindow, DataPreprocessor
from fca_drift.fca import build_formal_context, ConceptLattice
from fca_drift.fca import LatticeSimilarityCalculator

reader = StreamReader('sea', seed=42, drift_positions=[5000])
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
    # cap attrs
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

    if idx % 1000 == 0:
        print(f"idx={idx}, concepts={len(lat.concepts)}, "
              f"delta={deltas[-1][1]:.4f}" if deltas else f"idx={idx} (warmup)")

print(f"\nTotal signal points: {len(deltas)}")
if deltas:
    vals = [d for _, d in deltas]
    print(f"delta_L: min={min(vals):.4f}, max={max(vals):.4f}, mean={np.mean(vals):.4f}")
    # show around drift point
    near = [(i,d) for i,d in deltas if 4800 <= i <= 5200]
    print(f"\nNear drift (4800-5200):")
    for i, d in near[::10]:
        print(f"  idx={i}: delta={d:.4f}")
