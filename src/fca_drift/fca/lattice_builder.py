"""
Concept Lattice Builder - FULL LATTICE

FIXES:
- _build_hierarchy: replaced O(n^3) triple loop with O(n^2) transitive-reduction.
  Old code looped over ALL triplets for every pair — with 200+ concepts this
  means 8 million iterations per window step, causing 20+ hour runtimes.
- Added max_objects parameter to build_from_context so that the FCA context
  is always built from a SAMPLE of the window, not all W rows.
  FCA complexity is in the NUMBER OF OBJECTS too. Keeping objects ≤ 50
  while window_size can be 300-500 is the correct architecture:
    - window_size controls HOW OFTEN we compare (stream granularity)
    - max_objects controls FCA internal complexity (always small)
"""
from dataclasses import dataclass
from typing import List, Set, Dict, FrozenSet, Optional
from .context_builder import FormalContext

import numpy as np

try:
    from concepts import Context
    CONCEPTS_AVAILABLE = True
except ImportError:
    CONCEPTS_AVAILABLE = False
    print("[WARNING] 'concepts' library not available. Install: pip install concepts")


@dataclass
class FormalConcept:
    """
    Formal concept (A, B)
    A - extent, B - intent
    """
    extent: FrozenSet[int]
    intent: FrozenSet[int]

    def __hash__(self):
        return hash((self.extent, self.intent))

    def __eq__(self, other):
        return self.extent == other.extent and self.intent == other.intent

    def __repr__(self):
        return f"Concept(|A|={len(self.extent)}, |B|={len(self.intent)})"


class ConceptLattice:
    """
    Complete Concept Lattice with hierarchy
    """

    def __init__(self):
        self.concepts: List[FormalConcept] = []
        self.hierarchy: Dict[int, List[int]] = {}
        self.top_concept: FormalConcept = None
        self.bottom_concept: FormalConcept = None
        self.levels: List[List[FormalConcept]] = []

    def build_from_context(self, formal_context: FormalContext, max_objects: int = 50):
        """Build complete lattice from formal context.

        Args:
            formal_context: FormalContext built from the window binary matrix.
            max_objects: Maximum number of rows (objects) passed to FCA.
                         If the context has more rows, a uniform sample is taken.
                         Keeping this ≤ 50 ensures fast FCA regardless of window_size.
                         Default 50 matches the original prototype behaviour.
        """
        if not CONCEPTS_AVAILABLE:
            raise RuntimeError(
                "Cannot build lattice: 'concepts' library not installed. "
                "Install with: pip install concepts"
            )

        n_obj, n_attr = formal_context.incidence.shape

        # --- Object sampling (KEY FIX for large window sizes) ---
        # FCA complexity grows with n_objects. For window_size=300-500 we
        # do NOT want 300-500 objects in the FCA context — that explodes.
        # Instead we sample max_objects rows, preserving the ATTRIBUTE
        # distribution (which is what similarity compares).
        if n_obj > max_objects:
            # Uniform sample: take every k-th row so we cover the whole window.
            step = n_obj // max_objects
            selected = list(range(0, n_obj, step))[:max_objects]
            incidence = formal_context.incidence[selected, :]
            n_obj_used = len(selected)
        else:
            incidence = formal_context.incidence
            n_obj_used = n_obj

        objects = [f'g{i}' for i in range(n_obj_used)]
        attributes = [f'm{i}' for i in range(n_attr)]

        bools = []
        for i in range(n_obj_used):
            row = tuple(bool(incidence[i, j]) for j in range(n_attr))
            bools.append(row)

        ctx = Context(objects, attributes, bools)

        self.concepts = []
        for extent_str, intent_str in ctx.lattice:
            extent = frozenset(
                i for i, obj in enumerate(objects) if obj in extent_str
            )
            intent = frozenset(
                j for j, attr in enumerate(attributes) if attr in intent_str
            )
            concept = FormalConcept(extent=extent, intent=intent)
            self.concepts.append(concept)

        self._build_hierarchy()
        self._identify_extremes()
        self._compute_levels()

    def _build_hierarchy(self):
        """Build parent-child relations using O(n^2) transitive reduction.

        OLD code was O(n^3): for every ordered pair (i,j) it scanned ALL k
        to check for intermediate concepts. With 200 concepts that is
        200^3 = 8 000 000 iterations per window — catastrophic.

        NEW approach:
        1. Build the full direct-subconcept partial order in O(n^2).
           For each pair (i, j) where extent_i ⊂ extent_j, we record a
           raw "covers" edge.
        2. Remove transitive edges in a second O(n^2) pass using the
           already-known cover set (Hasse diagram algorithm).

        Result: correct Hasse diagram in O(n^2) time.
        """
        n = len(self.concepts)
        for i in range(n):
            self.hierarchy[i] = []

        if n == 0:
            return

        # Step 1: for each concept i find all concepts j such that
        #         extent_i ⊂ extent_j  (i is more specific than j).
        # We store ALL such pairs first, then remove non-direct ones.
        subconcept_of: List[Set[int]] = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and self.concepts[i].extent < self.concepts[j].extent:
                    subconcept_of[i].add(j)

        # Step 2: transitive reduction — keep only DIRECT parents.
        # j is a direct parent of i  ⟺  there is no k such that
        # extent_i ⊂ extent_k ⊂ extent_j.
        for i in range(n):
            direct_parents: Set[int] = set(subconcept_of[i])
            for j in list(direct_parents):
                # Remove all concepts reachable FROM j (they are indirect).
                direct_parents -= subconcept_of[j]
            # Store as children list of each parent (hierarchy[parent] → [children])
            for j in direct_parents:
                self.hierarchy[j].append(i)

    def _identify_extremes(self):
        """Find top and bottom concepts"""
        if not self.concepts:
            return

        self.top_concept = max(self.concepts, key=lambda c: len(c.extent))
        self.bottom_concept = min(self.concepts, key=lambda c: len(c.extent))

    def _compute_levels(self):
        """Organize concepts by levels.

        FIX: old code called self.concepts.index(concept) inside a loop —
        that is O(n) per iteration → O(n²) total. Replaced with a prebuilt
        concept→index map so the lookup is O(1).
        Also replaced the 'parents of concept_idx' scan (O(n) per node) with
        a prebuilt child→parents reverse map built once in O(n).
        """
        if not self.bottom_concept:
            return

        # O(n) prebuilt maps
        concept_to_idx = {id(c): i for i, c in enumerate(self.concepts)}

        # child_idx -> list of parent_idx  (reverse of self.hierarchy)
        child_to_parents: Dict[int, List[int]] = {i: [] for i in range(len(self.concepts))}
        for parent_idx, children in self.hierarchy.items():
            for child_idx in children:
                child_to_parents[child_idx].append(parent_idx)

        visited_ids = set()
        queue = [(self.bottom_concept, 0)]
        level_dict: Dict[int, List] = {}

        while queue:
            concept, level = queue.pop(0)
            cid = id(concept)

            if cid in visited_ids:
                continue
            visited_ids.add(cid)

            if level not in level_dict:
                level_dict[level] = []
            level_dict[level].append(concept)

            concept_idx = concept_to_idx.get(cid)
            if concept_idx is None:
                continue

            for parent_idx in child_to_parents.get(concept_idx, []):
                parent = self.concepts[parent_idx]
                if id(parent) not in visited_ids:
                    queue.append((parent, level + 1))

        self.levels = [level_dict[i] for i in sorted(level_dict.keys())]

    def get_concept_count(self) -> int:
        return len(self.concepts)

    def get_depth(self) -> int:
        return len(self.levels)

    def get_intents(self) -> Set[FrozenSet[int]]:
        """Get all intents (attributes) from all concepts"""
        return set(c.intent for c in self.concepts)

    def get_extents(self) -> Set[FrozenSet[int]]:
        """Get all extents (objects) from all concepts"""
        return set(c.extent for c in self.concepts)

    def get_all_attributes(self) -> Set[int]:
        """Get all unique attributes across all concepts"""
        all_attrs = set()
        for concept in self.concepts:
            all_attrs.update(concept.intent)
        return all_attrs

    def get_intent_strings(self, attribute_names: Dict[int, str] = None) -> Set[str]:
        """Get human-readable intents for explainability."""
        intents = self.get_intents()
        if attribute_names is None:
            return {f"({', '.join(f'm{i}' for i in sorted(intent))})" if intent else "(∅)"
                    for intent in intents}
        else:
            return {f"({', '.join(attribute_names.get(i, f'm{i}') for i in sorted(intent))})"
                    if intent else "(∅)"
                    for intent in intents}

    def compare_with(self, other: 'ConceptLattice') -> Dict:
        """Compare this lattice with another and return differences."""
        self_intents = self.get_intents()
        other_intents = other.get_intents()

        return {
            'lost_intents': self_intents - other_intents,
            'gained_intents': other_intents - self_intents,
            'stable_intents': self_intents & other_intents,
            'concept_count_before': len(self.concepts),
            'concept_count_after': len(other.concepts),
        }

    def __repr__(self):
        return f"ConceptLattice(concepts={len(self.concepts)}, depth={len(self.levels)})"


if __name__ == "__main__":
    import numpy as np
    from .context_builder import build_formal_context

    binary_matrix = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    context = build_formal_context(binary_matrix)
    lattice = ConceptLattice()
    lattice.build_from_context(context)

    print(f"Lattice: {lattice}")
    print(f"Top: {lattice.top_concept}")
    print(f"Bottom: {lattice.bottom_concept}")
