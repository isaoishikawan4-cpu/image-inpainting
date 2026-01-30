"""Individual beard hair segmentation using Grounded SAM + morphological processing.

This module provides the core functionality for segmenting a beard region
into individual hair strands using either skeleton-based or erosion-based methods.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from .beard_detector import GroundedSAMBackend, DetectedRegion
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


class SeparationMethod(Enum):
    """Available methods for separating individual hairs."""
    SKELETON = "skeleton"
    EROSION = "erosion"


@dataclass
class SegmentationConfig:
    """Configuration for single hair segmentation."""
    # Separation method
    separation_method: SeparationMethod = SeparationMethod.SKELETON

    # Filtering parameters
    min_hair_area: int = 10
    max_hair_area: int = 5000
    min_aspect_ratio: float = 1.5

    # Skeleton method parameters
    skeleton_min_length: int = 5
    dilation_radius: int = 3

    # Erosion method parameters
    erosion_iterations: int = 2

    # Preprocessing
    preprocess_kernel_size: int = 3
    remove_noise: bool = True
    fill_holes: bool = True


class SingleHairSegmenter:
    """Segments beard into individual hairs."""

    def __init__(self, config: Optional[SegmentationConfig] = None):
        """
        Initialize segmenter.

        Args:
            config: Segmentation configuration. Uses defaults if None.
        """
        self.config = config or SegmentationConfig()

    def segment_beard_mask(
        self,
        beard_mask: np.ndarray,
        original_image: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Segment a beard mask into individual hair masks.

        Args:
            beard_mask: Binary mask of entire beard region (from Grounded SAM)
            original_image: Optional original image for additional processing

        Returns:
            List of binary masks (uint8, 0/255), one per individual hair
        """
        # Preprocess mask
        preprocessed = preprocess_beard_mask(
            beard_mask,
            kernel_size=self.config.preprocess_kernel_size,
            remove_noise=self.config.remove_noise,
            fill_holes=self.config.fill_holes
        )

        # Choose separation method
        if self.config.separation_method == SeparationMethod.SKELETON:
            raw_masks = self._segment_by_skeleton(preprocessed)
        else:
            raw_masks = self._segment_by_erosion(preprocessed)

        # Filter by shape criteria
        filtered_masks = self._filter_masks(raw_masks)

        return filtered_masks

    def _segment_by_skeleton(self, beard_mask: np.ndarray) -> List[np.ndarray]:
        """
        Skeleton-based segmentation.

        Process:
        1. Extract skeleton (thin to 1-pixel width)
        2. Find branch points
        3. Split at branch points
        4. Restore thickness for each segment

        Args:
            beard_mask: Preprocessed binary mask

        Returns:
            List of individual hair masks
        """
        # Extract skeleton
        skeleton = extract_skeleton(beard_mask)

        if not np.any(skeleton):
            # No skeleton found, return original as single component
            return [beard_mask] if cv2.countNonZero(beard_mask) > 0 else []

        # Find branch points
        branch_points, endpoints = find_branch_endpoints(skeleton)

        # Split at branches
        segments = split_skeleton_at_branches(
            skeleton,
            branch_points,
            min_length=self.config.skeleton_min_length
        )

        if not segments:
            # No valid segments, try simpler connected component analysis
            return self._segment_by_erosion(beard_mask)

        # Restore thickness for each segment
        individual_hairs = []
        for segment in segments:
            restored = restore_segment_thickness(
                segment,
                beard_mask,
                dilation_radius=self.config.dilation_radius
            )
            if cv2.countNonZero(restored) > 0:
                individual_hairs.append(restored)

        return individual_hairs

    def _segment_by_erosion(self, beard_mask: np.ndarray) -> List[np.ndarray]:
        """
        Erosion-based segmentation (simpler approach).

        Process:
        1. Erode to separate touching hairs
        2. Connected component labeling
        3. Dilate and intersect with original

        Args:
            beard_mask: Preprocessed binary mask

        Returns:
            List of individual hair masks
        """
        return simple_connected_component_separation(
            beard_mask,
            erosion_iterations=self.config.erosion_iterations
        )

    def _filter_masks(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter hair masks by area and aspect ratio.

        Args:
            masks: List of candidate hair masks

        Returns:
            Filtered list of valid hair masks
        """
        filtered = []
        for mask in masks:
            if filter_by_shape(
                mask,
                min_area=self.config.min_hair_area,
                max_area=self.config.max_hair_area,
                min_aspect_ratio=self.config.min_aspect_ratio
            ):
                filtered.append(mask)

        return filtered


class SingleHairSegmentationPipeline:
    """Complete pipeline for single hair segmentation using Grounded SAM."""

    def __init__(
        self,
        sam_checkpoint: str = "sam_vit_h_4b8939.pth",
        grounding_dino_checkpoint: str = "groundingdino_swint_ogc.pth"
    ):
        """
        Initialize pipeline.

        Args:
            sam_checkpoint: SAM model checkpoint filename
            grounding_dino_checkpoint: Grounding DINO checkpoint filename
        """
        self._grounded_sam = GroundedSAMBackend(
            sam_checkpoint=sam_checkpoint,
            grounding_dino_checkpoint=grounding_dino_checkpoint
        )
        self._segmenter = SingleHairSegmenter()
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize Grounded SAM models.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        self._initialized = self._grounded_sam.initialize()
        return self._initialized

    def is_available(self) -> bool:
        """Check if pipeline is ready to use."""
        return self._initialized and self._grounded_sam.is_available()

    def process(
        self,
        image_rgb: np.ndarray,
        region_box: Tuple[int, int, int, int],
        text_prompt: str = "beard. facial hair. stubble.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        segmentation_config: Optional[SegmentationConfig] = None
    ) -> List[DetectedRegion]:
        """
        Complete processing pipeline.

        Args:
            image_rgb: Input RGB image
            region_box: (x1, y1, x2, y2) region of interest
            text_prompt: Text prompt for Grounded SAM detection
            box_threshold: Grounding DINO box threshold
            text_threshold: Grounding DINO text threshold
            segmentation_config: Configuration for hair separation

        Returns:
            List of DetectedRegion, one per individual hair
        """
        if segmentation_config:
            self._segmenter.config = segmentation_config

        # Initialize if needed
        if not self.is_available():
            if not self.initialize():
                raise RuntimeError("Failed to initialize Grounded SAM. Check checkpoint files.")

        # Step 1: Get overall beard mask from Grounded SAM
        print(f"Running Grounded SAM with prompt: '{text_prompt}'")
        beard_regions = self._grounded_sam.detect(
            image_rgb,
            region_box,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_area=50,  # Lower threshold for overall mask
            max_area=100000
        )

        if not beard_regions:
            print("No beard regions detected by Grounded SAM")
            return []

        print(f"Grounded SAM detected {len(beard_regions)} beard regions")

        # Combine all detected beard regions
        h, w = image_rgb.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for region in beard_regions:
            combined_mask = cv2.bitwise_or(combined_mask, region.mask)

        total_area = cv2.countNonZero(combined_mask)
        print(f"Combined beard mask area: {total_area} pixels")

        # Step 2: Segment into individual hairs
        method_name = self._segmenter.config.separation_method.value
        print(f"Separating individual hairs using '{method_name}' method...")
        individual_masks = self._segmenter.segment_beard_mask(combined_mask)
        print(f"Found {len(individual_masks)} individual hair segments")

        # Step 3: Convert to DetectedRegion format
        detected_regions = []
        for i, mask in enumerate(individual_masks):
            area = cv2.countNonZero(mask)
            cx, cy = calculate_centroid(mask)

            detected_regions.append(DetectedRegion(
                mask=mask,
                area=area,
                centroid=(cx, cy),
                confidence=1.0,  # Individual segments don't have confidence from SAM
                source='grounded_sam_single_hair',
                phrase=f"hair_{i+1}"
            ))

        return detected_regions

    def process_with_details(
        self,
        image_rgb: np.ndarray,
        region_box: Tuple[int, int, int, int],
        text_prompt: str = "beard. facial hair. stubble.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        segmentation_config: Optional[SegmentationConfig] = None
    ) -> Tuple[List[DetectedRegion], np.ndarray, dict]:
        """
        Process with additional details for debugging/visualization.

        Args:
            Same as process()

        Returns:
            Tuple of:
            - List of DetectedRegion
            - Combined beard mask before segmentation
            - Dictionary with processing stats
        """
        if segmentation_config:
            self._segmenter.config = segmentation_config

        if not self.is_available():
            if not self.initialize():
                raise RuntimeError("Failed to initialize Grounded SAM")

        # Step 1: Grounded SAM detection
        beard_regions = self._grounded_sam.detect(
            image_rgb,
            region_box,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_area=50,
            max_area=100000
        )

        stats = {
            'grounded_sam_regions': len(beard_regions),
            'text_prompt': text_prompt,
            'box_threshold': box_threshold,
            'text_threshold': text_threshold,
            'separation_method': self._segmenter.config.separation_method.value,
        }

        if not beard_regions:
            empty_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            stats['individual_hairs'] = 0
            stats['total_beard_area'] = 0
            return [], empty_mask, stats

        # Combine masks
        h, w = image_rgb.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for region in beard_regions:
            combined_mask = cv2.bitwise_or(combined_mask, region.mask)

        stats['total_beard_area'] = cv2.countNonZero(combined_mask)

        # Step 2: Segment individual hairs
        individual_masks = self._segmenter.segment_beard_mask(combined_mask)
        stats['individual_hairs'] = len(individual_masks)

        # Convert to DetectedRegion
        detected_regions = []
        for i, mask in enumerate(individual_masks):
            area = cv2.countNonZero(mask)
            cx, cy = calculate_centroid(mask)

            detected_regions.append(DetectedRegion(
                mask=mask,
                area=area,
                centroid=(cx, cy),
                confidence=1.0,
                source='grounded_sam_single_hair',
                phrase=f"hair_{i+1}"
            ))

        return detected_regions, combined_mask, stats


def visualize_single_hairs(
    image_rgb: np.ndarray,
    detected_regions: List[DetectedRegion],
    alpha: float = 0.5,
    show_markers: bool = True
) -> np.ndarray:
    """
    Visualize individual hairs with different colors.

    Args:
        image_rgb: Original RGB image
        detected_regions: List of DetectedRegion objects
        alpha: Transparency for overlay (0-1, lower = more transparent)
        show_markers: Whether to show center markers

    Returns:
        RGB image with colored hair overlay
    """
    result = image_rgb.copy()

    # Generate distinct colors using HSV
    num_regions = len(detected_regions)
    if num_regions == 0:
        return result

    colors = []
    for i in range(num_regions):
        hue = int(180 * i / num_regions)  # Distribute hues evenly
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(tuple(int(c) for c in rgb))

    # Apply colored overlay for each region
    overlay = result.copy()
    for region, color in zip(detected_regions, colors):
        mask_bool = region.mask > 0
        overlay[mask_bool] = color

    # Blend with original
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    # Draw centroids (markers)
    if show_markers:
        for region in detected_regions:
            cx, cy = region.centroid
            cv2.circle(result, (cx, cy), 2, (255, 255, 255), -1)

    return result
