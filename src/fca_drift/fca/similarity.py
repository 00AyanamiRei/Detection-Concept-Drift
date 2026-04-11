"""
Lattice Similarity Calculator

Compares FULL lattice structures, not just intents!
"""
import numpy as np
from typing import Dict, Set
from .lattice_builder import ConceptLattice


class LatticeSimilarityCalculator:
    """
    Computes similarity between two concept lattices

    Sim(L1, L2) = α·Sim_concepts + β·Sim_hierarchy + γ·Sim_structure
    """

    def __init__(self,
                 alpha: float = 0.4,
                 beta: float = 0.3,
                 gamma: float = 0.3,
                 min_support_ratio: float = 0.1):
        """
        Args:
            alpha: Weight for concept similarity
            beta: Weight for hierarchy similarity
            gamma: Weight for structural similarity
            min_support_ratio: Concepts below this support are ignored to reduce
                small-window noise in intent-set comparisons.
        """
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1"

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_support_ratio = max(0.0, min(1.0, float(min_support_ratio)))

    def compute_similarity(self, L1: ConceptLattice, L2: ConceptLattice) -> float:
        """
        Compute overall lattice similarity
        """
        sim_concepts = self._concept_similarity(L1, L2)
        sim_hierarchy = self._hierarchy_similarity(L1, L2)
        sim_structure = self._structural_similarity(L1, L2)

        return (self.alpha * sim_concepts +
                self.beta * sim_hierarchy +
                self.gamma * sim_structure)

    def _concept_similarity(self, L1: ConceptLattice, L2: ConceptLattice) -> float:
        """
        Similarity on concept signatures using support-weighted intent overlap.

        IMPORTANT:
        Direct extent comparison is unstable for sliding windows because object IDs
        are positional (0..W-1) and shift every step. We therefore compare:
        - intents directly (stable semantic signature)
        - extent-size/intent-size signatures (shape of concepts)
        """
        support1 = self._intent_support_map(L1)
        support2 = self._intent_support_map(L2)
        jaccard_int = self._weighted_jaccard(support1, support2)

        sig1 = set((len(c.extent), len(c.intent)) for c in self._filtered_concepts(L1))
        sig2 = set((len(c.extent), len(c.intent)) for c in self._filtered_concepts(L2))
        jaccard_sig = self._jaccard(sig1, sig2)

        return 0.75 * jaccard_int + 0.25 * jaccard_sig

    def _hierarchy_similarity(self, L1: ConceptLattice, L2: ConceptLattice) -> float:
        """Similarity of parent-child relations"""
        edges1 = self._get_edges(L1)
        edges2 = self._get_edges(L2)

        return self._jaccard(edges1, edges2)

    def _structural_similarity(self, L1: ConceptLattice, L2: ConceptLattice) -> float:
        """Similarity of overall structure"""
        count1 = len(self._filtered_concepts(L1))
        count2 = len(self._filtered_concepts(L2))
        sim_count = 1 - abs(count1 - count2) / max(count1, count2, 1)

        depth1 = L1.get_depth()
        depth2 = L2.get_depth()
        sim_depth = 1 - abs(depth1 - depth2) / max(depth1, depth2, 1)

        return (sim_count + sim_depth) / 2

    def _get_edges(self, lattice: ConceptLattice) -> Set:
        """
        Extract parent-child edges using intent-only signatures.

        Using extents directly is unstable for sliding windows because object IDs
        are not persistent across consecutive windows.
        """
        edges = set()
        concept_support = self._concept_support_by_index(lattice)
        for parent_id, children in lattice.hierarchy.items():
            if concept_support.get(parent_id, 0.0) < self.min_support_ratio:
                continue
            parent = lattice.concepts[parent_id]
            for child_id in children:
                if concept_support.get(child_id, 0.0) < self.min_support_ratio:
                    continue
                child = lattice.concepts[child_id]
                edge = (
                    frozenset(parent.intent),
                    frozenset(child.intent)
                )
                edges.add(edge)
        return edges

    def _concept_support_by_index(self, lattice: ConceptLattice) -> Dict[int, float]:
        """Return support ratio for each concept index."""
        if not lattice.concepts:
            return {}
        n_objects = max((len(c.extent) for c in lattice.concepts), default=1)
        n_objects = max(n_objects, 1)
        return {
            idx: len(concept.extent) / n_objects
            for idx, concept in enumerate(lattice.concepts)
        }

    def _filtered_concepts(self, lattice: ConceptLattice):
        """Yield concepts above minimum support."""
        support = self._concept_support_by_index(lattice)
        return [
            concept
            for idx, concept in enumerate(lattice.concepts)
            if support.get(idx, 0.0) >= self.min_support_ratio
        ]

    def _intent_support_map(self, lattice: ConceptLattice) -> Dict[frozenset, float]:
        """Aggregate support weights by intent signature."""
        support = self._concept_support_by_index(lattice)
        intent_weights: Dict[frozenset, float] = {}
        for idx, concept in enumerate(lattice.concepts):
            w = support.get(idx, 0.0)
            if w < self.min_support_ratio:
                continue
            key = frozenset(concept.intent)
            # Keep strongest support per intent to avoid over-counting duplicates.
            intent_weights[key] = max(intent_weights.get(key, 0.0), w)
        return intent_weights

    @staticmethod
    def _weighted_jaccard(weights1: Dict[frozenset, float],
                          weights2: Dict[frozenset, float]) -> float:
        """Weighted Jaccard for intent-support maps."""
        if not weights1 and not weights2:
            return 1.0
        all_keys = set(weights1) | set(weights2)
        if not all_keys:
            return 1.0
        numerator = sum(min(weights1.get(k, 0.0), weights2.get(k, 0.0)) for k in all_keys)
        denominator = sum(max(weights1.get(k, 0.0), weights2.get(k, 0.0)) for k in all_keys)
        return numerator / denominator if denominator > 0 else 0.0

    @staticmethod
    def _jaccard(set1: Set, set2: Set) -> float:
        """Jaccard similarity"""
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    import numpy as np
    from .context_builder import build_formal_context
    from .lattice_builder import ConceptLattice

    # Two similar contexts
    matrix1 = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    matrix2 = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]])

    context1 = build_formal_context(matrix1)
    context2 = build_formal_context(matrix2)

    lattice1 = ConceptLattice()
    lattice1.build_from_context(context1)

    lattice2 = ConceptLattice()
    lattice2.build_from_context(context2)

    calc = LatticeSimilarityCalculator()
    sim = calc.compute_similarity(lattice1, lattice2)

    print(f"Similarity: {sim:.4f}")
