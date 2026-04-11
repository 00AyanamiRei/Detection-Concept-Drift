"""
Formal Context Builder
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Set


@dataclass
class FormalContext:
    """
    Represents a formal context (G, M, I)

    G - objects (rows)
    M - attributes (columns)
    I - incidence relation (binary matrix)
    """
    objects: List[int]
    attributes: List[int]
    incidence: np.ndarray

    def get_object_intent(self, obj_idx: int) -> Set[int]:
        """Get intent of object: {m ∈ M | (obj, m) ∈ I}"""
        return set(
            attr_idx
            for attr_idx in self.attributes
            if self.incidence[obj_idx, attr_idx]
        )

    def get_attribute_extent(self, attr_idx: int) -> Set[int]:
        """Get extent of attribute: {g ∈ G | (g, attr) ∈ I}"""
        return set(
            obj_idx
            for obj_idx in self.objects
            if self.incidence[obj_idx, attr_idx]
        )

    def __repr__(self):
        return f"FormalContext({len(self.objects)}x{len(self.attributes)})"


def build_formal_context(binary_matrix: np.ndarray) -> FormalContext:
    """
    Build formal context from binary matrix

    Args:
        binary_matrix: Binary matrix (n_objects × n_attributes)

    Returns:
        FormalContext instance
    """
    if binary_matrix.ndim != 2:
        raise ValueError("Binary matrix must be 2D")

    n_objects, n_attributes = binary_matrix.shape

    return FormalContext(
        objects=list(range(n_objects)),
        attributes=list(range(n_attributes)),
        incidence=binary_matrix
    )


if __name__ == "__main__":
    matrix = np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 1],
    ])

    context = build_formal_context(matrix)
    print(context)
    print(f"Object 0 intent: {context.get_object_intent(0)}")
    print(f"Attribute 1 extent: {context.get_attribute_extent(1)}")
