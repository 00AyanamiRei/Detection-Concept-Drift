"""
Concept Lattice Builder - FULL LATTICE

This is the CRITICAL module missing from the prototype!
"""
from dataclasses import dataclass
from typing import List, Set, Dict, FrozenSet, Optional
from .context_builder import FormalContext

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

    def build_from_context(self, formal_context: FormalContext):
        """Build complete lattice from formal context"""
        if not CONCEPTS_AVAILABLE:
            raise RuntimeError(
                "Cannot build lattice: 'concepts' library not installed. "
                "Install with: pip install concepts"
            )

        n_obj, n_attr = formal_context.incidence.shape

        objects = [f'g{i}' for i in range(n_obj)]
        attributes = [f'm{i}' for i in range(n_attr)]

        bools = []
        for i in range(n_obj):
            row = tuple(bool(formal_context.incidence[i, j]) for j in range(n_attr))
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
        """Build parent-child relations"""
        n = len(self.concepts)

        for i in range(n):
            self.hierarchy[i] = []

        for i, c1 in enumerate(self.concepts):
            for j, c2 in enumerate(self.concepts):
                if i == j:
                    continue

                if c1.extent < c2.extent:
                    is_direct = True
                    for k, c3 in enumerate(self.concepts):
                        if k == i or k == j:
                            continue
                        if c1.extent < c3.extent < c2.extent:
                            is_direct = False
                            break

                    if is_direct:
                        self.hierarchy[j].append(i)

    def _identify_extremes(self):
        """Find top and bottom concepts"""
        if not self.concepts:
            return

        self.top_concept = max(self.concepts, key=lambda c: len(c.extent))
        self.bottom_concept = min(self.concepts, key=lambda c: len(c.extent))

    def _compute_levels(self):
        """Organize concepts by levels"""
        if not self.bottom_concept:
            return

        visited = set()
        queue = [(self.bottom_concept, 0)]
        level_dict = {}

        while queue:
            concept, level = queue.pop(0)

            if concept in visited:
                continue
            visited.add(concept)

            if level not in level_dict:
                level_dict[level] = []
            level_dict[level].append(concept)

            concept_idx = self.concepts.index(concept)
            parents_indices = [
                parent_idx
                for parent_idx, children in self.hierarchy.items()
                if concept_idx in children
            ]

            for parent_idx in parents_indices:
                parent = self.concepts[parent_idx]
                if parent not in visited:
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
        """
        Get human-readable intents for explainability

        Args:
            attribute_names: Mapping from attribute index to name (e.g., {0: 'A', 1: 'B'})

        Returns:
            Set of string representations of intents
        """
        intents = self.get_intents()
        if attribute_names is None:
            # Default: use m0, m1, m2...
            return {f"({', '.join(f'm{i}' for i in sorted(intent))})" if intent else "(∅)"
                    for intent in intents}
        else:
            return {f"({', '.join(attribute_names.get(i, f'm{i}') for i in sorted(intent))})"
                    if intent else "(∅)"
                    for intent in intents}

    def compare_with(self, other: 'ConceptLattice') -> Dict:
        """
        Compare this lattice with another and return differences

        Returns:
            {
                'lost_intents': Set[FrozenSet[int]],
                'gained_intents': Set[FrozenSet[int]],
                'stable_intents': Set[FrozenSet[int]],
                'concept_count_before': int,
                'concept_count_after': int
            }
        """
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
