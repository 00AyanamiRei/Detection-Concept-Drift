import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))

import numpy as np
from fca_drift.core import StreamReader, StreamWrapper, SlidingWindow, DataPreprocessor
from fca_drift.fca import build_formal_context, ConceptLattice, LatticeSimilarityCalculator

def test_dataset(name, drift_pos=5000, n=10000, w=300):
    reader = StreamReader(name, seed=42, drift_positions=[drift_pos])
    stream = StreamWrapper(reader.stream, max_instances=n)
    window = SlidingWindow(size=w)
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
            deltas.append((idx, 1.0 - sim))
        prev_lattice = lat

    vals = [d for _, d in deltas]
    before = [d for i,d in deltas if int(drift_pos*0.3) <= i <= int(drift_pos*0.8)]
    after  = [d for i,d in deltas if int(drift_pos*1.2) <= i <= int(drift_pos*1.8)]
    print(f"\n{name}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
          f"max={max(vals):.4f}")
    if before and after:
        print(f"  before_mean={np.mean(before):.4f}, after_mean={np.mean(after):.4f}, "
              f"diff={abs(np.mean(after)-np.mean(before)):.4f}")

# Test synthetic
test_dataset('agrawal', drift_pos=5000)
test_dataset('sea',     drift_pos=5000)

# Test real (no drift position needed)
reader = StreamReader('elec2', seed=42)
stream = StreamWrapper(reader.stream, max_instances=5000)
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
        deltas.append((idx, 1.0 - sim))
    prev_lattice = lat
vals = [d for _, d in deltas]
print(f"\nelec2: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, max={max(vals):.4f}")
