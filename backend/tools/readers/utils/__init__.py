"""
Utilities for text readers and document processing.

This package contains common utilities that can be shared across
different document parsers, including text optimization, paragraph
processing, and language detection.
"""

from .para_optimizer import ParagraphOptimizer

__all__ = [
    'ParagraphOptimizer',
]