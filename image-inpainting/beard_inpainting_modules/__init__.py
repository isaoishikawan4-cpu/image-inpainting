"""Beard inpainting modules for Gradio application v4.

This package provides modular components for beard detection and removal:
- ImageHandler: Image format conversion utilities
- RegionSelector: Rectangle extraction from Gradio ImageEditor
- BeardDetector: Unified detector with Grounded SAM and rule-based backends
- BeardRegionManager: Region management and selection logic
- LamaInpainter: LaMa inpainting wrapper
- SkinColorCorrector: Post-inpainting color correction for blue beard
- BeardRemovalPipeline: Main orchestrator for the complete workflow
"""

from .image_handler import ImageHandler
from .region_selector import RegionSelector, SelectionShape
from .beard_detector import (
    BeardDetector,
    GroundedSAMBackend,
    RuleBasedBackend,
    DetectedRegion,
    DetectionBackend,
)
from .highlighter import BeardRegionManager, SelectionMode, SelectionResult
from .inpainter import LamaInpainter, OpenCVInpainter, InpaintingMethod
from .color_corrector import SkinColorCorrector, CorrectionMode, MaskType
from .pipeline import BeardRemovalPipeline

__all__ = [
    # Image utilities
    'ImageHandler',

    # Region selection
    'RegionSelector',
    'SelectionShape',

    # Detection
    'BeardDetector',
    'GroundedSAMBackend',
    'RuleBasedBackend',
    'DetectedRegion',
    'DetectionBackend',

    # Highlighting/Selection
    'BeardRegionManager',
    'SelectionMode',
    'SelectionResult',

    # Inpainting
    'LamaInpainter',
    'OpenCVInpainter',
    'InpaintingMethod',

    # Color Correction
    'SkinColorCorrector',
    'CorrectionMode',
    'MaskType',

    # Pipeline
    'BeardRemovalPipeline',
]

__version__ = '0.2.0'
