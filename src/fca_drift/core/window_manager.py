"""
Sliding Window Manager
"""
from collections import deque
from typing import List, Any


class SlidingWindow:
    """
    Fixed-size sliding window with FIFO behavior
    """

    def __init__(self, size: int):
        """
        Args:
            size: Window size (number of instances)
        """
        if size <= 0:
            raise ValueError("Window size must be positive")

        self.size = size
        self.window = deque(maxlen=size)

    def append(self, instance: Any):
        """
        Add instance to window
        Automatically removes oldest if full
        """
        self.window.append(instance)

    def is_full(self) -> bool:
        """Check if window has reached size"""
        return len(self.window) == self.size

    def get_data(self) -> List[Any]:
        """Get all instances in window"""
        return list(self.window)

    def clear(self):
        """Clear window"""
        self.window.clear()

    def __len__(self):
        """Current window size"""
        return len(self.window)

    def __repr__(self):
        return f"SlidingWindow(size={self.size}, current={len(self.window)})"


if __name__ == "__main__":
    window = SlidingWindow(size=3)

    for i in range(5):
        window.append({'id': i})
        print(f"Step {i}: {window}, len={len(window)}")
