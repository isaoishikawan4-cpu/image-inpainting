"""Beard inpainting modules for Gradio application v6.

This package provides modular components for beard detection and removal:
- ImageHandler: Image format conversion utilities
- RegionSelector: Rectangle extraction from Gradio ImageEditor
- BeardDetector: Unified detector with Grounded SAM, rule-based, and YOLO v8 backends
- BeardRegionManager: Region management and selection logic
- LamaInpainter: LaMa inpainting wrapper
- MATInpainter: MAT (Mask-Aware Transformer) inpainting wrapper (v6 new)
- SkinColorCorrector: Post-inpainting color correction for blue beard
- BeardRemovalPipeline: Main orchestrator for the complete workflow
"""

from .image_handler import ImageHandler
from .region_selector import RegionSelector
from .beard_detector import (
    BeardDetector,
    GroundedSAMBackend,
    RuleBasedBackend,
    DetectedRegion,
    DetectionBackend,
)
from .highlighter import BeardRegionManager, SelectionMode, SelectionResult
from .inpainter import LamaInpainter, InpaintingMethod
from .mat_inpainter import MATInpainter
from .color_corrector import SkinColorCorrector, CorrectionMode
from .pipeline import BeardRemovalPipeline

# Single hair segmentation
from .single_hair_segmenter import (
    SingleHairSegmenter,
    SingleHairSegmentationPipeline,
    SeparationMethod,
    SegmentationConfig,
    visualize_single_hairs,
)

# Black/White hair detection
from .black_white_hair_detector import (
    BlackWhiteHairDetector,
    HairClassParams,
)

# Morphology utilities
from .morphology_utils import (
    extract_skeleton,
    find_branch_endpoints,
    split_skeleton_at_branches,
    restore_segment_thickness,
    simple_connected_component_separation,
    preprocess_beard_mask,
    filter_by_shape,
    calculate_centroid,
)

__all__ = [
    # Image utilities
    'ImageHandler',

    # Region selection
    'RegionSelector',

    # Detection
    'BeardDetector',
    'GroundedSAMBackend',
    'RuleBasedBackend',
    'DetectedRegion',
    'DetectionBackend',

    # Single Hair Segmentation
    'SingleHairSegmenter',
    'SingleHairSegmentationPipeline',
    'SeparationMethod',
    'SegmentationConfig',
    'visualize_single_hairs',

    # Black/White Hair Detection
    'BlackWhiteHairDetector',
    'HairClassParams',

    # Morphology Utilities
    'extract_skeleton',
    'find_branch_endpoints',
    'split_skeleton_at_branches',
    'restore_segment_thickness',
    'simple_connected_component_separation',
    'preprocess_beard_mask',
    'filter_by_shape',
    'calculate_centroid',

    # Highlighting/Selection
    'BeardRegionManager',
    'SelectionMode',
    'SelectionResult',

    # Inpainting
    'LamaInpainter',
    'MATInpainter',
    'InpaintingMethod',

    # Color Correction
    'SkinColorCorrector',
    'CorrectionMode',

    # Pipeline
    'BeardRemovalPipeline',
]

__version__ = '0.4.0'
