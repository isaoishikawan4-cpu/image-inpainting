"""Hair thickness classification utilities.

Provides functions to classify detected hairs into thickness categories
based on the minimum bounding rectangle width of each hair mask.

Categories:
- 剛毛 (Coarse): Very thick hair
- 硬毛 (Thick): Thick hair
- 中間毛 (Medium): Medium thickness hair
- 軟毛 (Fine): Fine/thin hair
"""

import numpy as np
import cv2
from typing import List
from dataclasses import dataclass

from .beard_detector import DetectedRegion


# Hair thickness categories with colors (RGB for Gradio display)
THICKNESS_CATEGORIES = {
    "剛毛": {"color": (255, 0, 0), "label_en": "Coarse"},       # Red
    "硬毛": {"color": (255, 165, 0), "label_en": "Thick"},      # Orange
    "中間毛": {"color": (0, 200, 0), "label_en": "Medium"},     # Green
    "軟毛": {"color": (148, 0, 211), "label_en": "Fine"},       # Purple (Dark Violet)
}


@dataclass
class ClassifiedHair:
    """A detected hair with thickness classification."""
    detection: DetectedRegion
    width: float
    category: str


def classify_hair_thickness(
    width: float,
    threshold_coarse: float,
    threshold_thick: float,
    threshold_medium: float
) -> str:
    """Classify hair based on width thresholds."""
    if width >= threshold_coarse:
        return "剛毛"
    elif width >= threshold_thick:
        return "硬毛"
    elif width >= threshold_medium:
        return "中間毛"
    else:
        return "軟毛"


def calculate_hair_width(mask: np.ndarray) -> float:
    """Calculate hair width from mask using minAreaRect."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    w_rect, h_rect = rect[1]

    if w_rect == 0 or h_rect == 0:
        return 0.0

    return min(w_rect, h_rect)


def visualize_classified_hairs(
    image: np.ndarray,
    classified_hairs: List[ClassifiedHair],
    alpha: float = 0.4,
    show_markers: bool = True
) -> np.ndarray:
    """Visualize hairs with category-based colors."""
    result = image.copy()
    overlay = np.zeros_like(image)

    for hair in classified_hairs:
        color = THICKNESS_CATEGORIES[hair.category]["color"]
        mask_bool = hair.detection.mask > 0
        overlay[mask_bool] = color

        if show_markers:
            cx, cy = hair.detection.centroid
            cv2.circle(result, (cx, cy), 2, color, -1)

    mask_any = np.any(overlay > 0, axis=2)
    result[mask_any] = cv2.addWeighted(
        result[mask_any], 1 - alpha,
        overlay[mask_any], alpha,
        0
    )

    return result
