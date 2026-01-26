"""Beard region management and highlight display.

This module manages detected beard regions, implements selection logic
(random, area-based, confidence-based), and generates colored visualizations.
"""

import numpy as np
import cv2
from enum import Enum
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random

from .beard_detector import DetectedRegion


class SelectionMode(Enum):
    """Region selection modes."""
    RANDOM = "random"
    AREA_LARGE = "area_large"
    AREA_SMALL = "area_small"
    CONFIDENCE = "confidence"


@dataclass
class SelectionResult:
    """Result of region selection."""
    active_indices: List[int]
    total_count: int
    selected_count: int
    seed: int


class BeardRegionManager:
    """Manages detected beard regions and selection logic."""

    # Color palette for visualization (RGB)
    COLORS_RGB = [
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Dark Red
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Blue
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]

    HIGHLIGHT_COLOR = (255, 0, 0)  # Red for selected regions

    def __init__(self):
        self._regions: List[DetectedRegion] = []
        self._active_indices: List[int] = []
        self._random_seed: int = 42

    @property
    def regions(self) -> List[DetectedRegion]:
        """Get all detected regions."""
        return self._regions

    @property
    def active_indices(self) -> List[int]:
        """Get indices of selected (active) regions."""
        return self._active_indices

    @property
    def random_seed(self) -> int:
        """Get current random seed."""
        return self._random_seed

    def clear(self) -> None:
        """Clear all regions and selections."""
        self._regions = []
        self._active_indices = []

    def add_regions(self, regions: List[DetectedRegion]) -> None:
        """Add new detected regions."""
        self._regions.extend(regions)

    def update_selection(
        self,
        removal_percentage: int,
        mode: SelectionMode,
        new_seed: bool = False
    ) -> SelectionResult:
        """
        Update which regions are selected for removal.

        Args:
            removal_percentage: Percentage of regions to select (0-100)
            mode: Selection strategy
            new_seed: Generate new random seed

        Returns:
            SelectionResult with indices and statistics
        """
        if not self._regions:
            self._active_indices = []
            return SelectionResult(
                active_indices=[],
                total_count=0,
                selected_count=0,
                seed=self._random_seed
            )

        if new_seed:
            self._random_seed = random.randint(0, 10000)

        total = len(self._regions)
        target_count = int(total * removal_percentage / 100)

        if mode == SelectionMode.RANDOM:
            self._active_indices = self._select_by_random(target_count)
        elif mode == SelectionMode.AREA_LARGE:
            self._active_indices = self._select_by_area(target_count, largest_first=True)
        elif mode == SelectionMode.AREA_SMALL:
            self._active_indices = self._select_by_area(target_count, largest_first=False)
        elif mode == SelectionMode.CONFIDENCE:
            self._active_indices = self._select_by_confidence(target_count)
        else:
            self._active_indices = list(range(target_count))

        return SelectionResult(
            active_indices=self._active_indices.copy(),
            total_count=total,
            selected_count=len(self._active_indices),
            seed=self._random_seed
        )

    def _select_by_random(self, count: int) -> List[int]:
        """Select regions randomly."""
        random.seed(self._random_seed)
        indices = list(range(len(self._regions)))
        random.shuffle(indices)
        return sorted(indices[:count])

    def _select_by_area(self, count: int, largest_first: bool) -> List[int]:
        """Select regions by area."""
        sorted_idx = sorted(
            range(len(self._regions)),
            key=lambda i: self._regions[i].area,
            reverse=largest_first
        )
        return sorted(sorted_idx[:count])

    def _select_by_confidence(self, count: int) -> List[int]:
        """Select regions by detection confidence."""
        sorted_idx = sorted(
            range(len(self._regions)),
            key=lambda i: self._regions[i].confidence,
            reverse=True
        )
        return sorted(sorted_idx[:count])

    def get_combined_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Get combined binary mask of all selected regions.

        Args:
            image_shape: (height, width) of the target image

        Returns:
            Binary mask (H, W) where 255 = selected region
        """
        h, w = image_shape
        combined = np.zeros((h, w), dtype=np.uint8)
        for idx in self._active_indices:
            if 0 <= idx < len(self._regions):
                combined = cv2.bitwise_or(combined, self._regions[idx].mask)
        return combined

    def get_dilated_mask(
        self,
        image_shape: Tuple[int, int],
        dilation: int = 2
    ) -> np.ndarray:
        """
        Get dilated mask for better inpainting coverage.

        Args:
            image_shape: (height, width) of the target image
            dilation: Dilation kernel radius

        Returns:
            Dilated binary mask
        """
        mask = self.get_combined_mask(image_shape)
        if dilation > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation * 2 + 1, dilation * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def create_colored_display(
        self,
        image_rgb: np.ndarray,
        highlight_active: bool = True
    ) -> np.ndarray:
        """
        Create visualization with each region in different color.

        Args:
            image_rgb: Base RGB image
            highlight_active: Show selected regions in red

        Returns:
            Annotated RGB image
        """
        display = image_rgb.copy()

        for i, region in enumerate(self._regions):
            mask = region.mask
            color = self.COLORS_RGB[i % len(self.COLORS_RGB)]

            if highlight_active and i in self._active_indices:
                # Selected for removal: highlight in red (semi-transparent)
                overlay = display.copy()
                display[mask > 0] = self.HIGHLIGHT_COLOR
                cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            else:
                # Keep: show in palette color (lighter)
                overlay = display.copy()
                display[mask > 0] = color
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        return display

    def get_all_masks_combined(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Get combined mask of all regions (not just selected).

        Args:
            image_shape: (height, width) of the target image

        Returns:
            Binary mask of all detected regions
        """
        h, w = image_shape
        combined = np.zeros((h, w), dtype=np.uint8)
        for region in self._regions:
            combined = cv2.bitwise_or(combined, region.mask)
        return combined

    def get_region_count(self) -> int:
        """Get total number of detected regions."""
        return len(self._regions)

    def get_selected_count(self) -> int:
        """Get number of selected regions."""
        return len(self._active_indices)
