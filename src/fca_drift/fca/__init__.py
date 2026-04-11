"""
Formal Concept Analysis modules
"""
from .context_builder import FormalContext, build_formal_context
from .lattice_builder import ConceptLattice, FormalConcept
from .similarity import LatticeSimilarityCalculator

__all__ = [
    'FormalContext',
    'build_formal_context',
    'ConceptLattice',
    'FormalConcept',
    'LatticeSimilarityCalculator',
]
