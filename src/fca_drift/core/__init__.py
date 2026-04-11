"""
Core stream processing modules
"""
from .stream_reader import StreamReader, StreamWrapper, get_available_datasets
from .window_manager import SlidingWindow
from .preprocessing import DataPreprocessor

__all__ = [
    'StreamReader',
    'StreamWrapper',
    'SlidingWindow',
    'DataPreprocessor',
    'get_available_datasets',
]
