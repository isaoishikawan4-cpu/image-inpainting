"""Morphological processing utilities for single hair segmentation.

This module provides functions to separate connected beard regions
into individual hair strands using skeleton-based and erosion-based approaches.
"""

import numpy as np
import cv2
from scipy import ndimage
from typing import List, Tuple, Optional

# Try to import skimage for skeletonization
SKIMAGE_AVAILABLE = False
try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    pass


def extract_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract skeleton (1-pixel wide) from binary mask.

    Uses Zhang-Suen thinning algorithm via scikit-image if available,
    otherwise falls back to OpenCV morphological thinning.

    Args:
        binary_mask: Binary mask (0/255 or 0/1)

    Returns:
        Boolean array where True indicates skeleton pixels
    """
    # Normalize to boolean
    mask_bool = (binary_mask > 0)

    if SKIMAGE_AVAILABLE:
        # scikit-image skeletonize (Zhang-Suen algorithm)
        skeleton = skeletonize(mask_bool)
    else:
        # Fallback: OpenCV morphological thinning
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        skeleton_uint8 = cv2.ximgproc.thinning(mask_uint8) if hasattr(cv2, 'ximgproc') else _manual_thinning(mask_uint8)
        skeleton = skeleton_uint8 > 0

    return skeleton


def _manual_thinning(binary_image: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """
    Manual thinning implementation as fallback.

    Args:
        binary_image: Binary image (0/255)
        max_iterations: Maximum iterations

    Returns:
        Thinned binary image
    """
    img = binary_image.copy()
    prev = np.zeros_like(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for _ in range(max_iterations):
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        subset = cv2.subtract(eroded, opened)
        img = cv2.subtract(img, subset)

        if np.array_equal(img, prev):
            break
        prev = img.copy()

    return img


def find_branch_endpoints(skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find branch points and endpoints in skeleton.

    Branch points: pixels with 3+ neighbors on the skeleton
    Endpoints: pixels with exactly 1 neighbor on the skeleton

    Args:
        skeleton: Boolean skeleton array

    Returns:
        Tuple of (branch_points, endpoints) as boolean arrays
    """
    # 8-connectivity kernel for counting neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    skeleton_uint8 = skeleton.astype(np.uint8)

    # Count neighbors for each pixel
    neighbor_count = cv2.filter2D(skeleton_uint8, -1, kernel)

    # Only count on skeleton pixels
    neighbor_count = neighbor_count * skeleton_uint8

    # Branch points: 3 or more neighbors
    branch_points = (neighbor_count >= 3) & skeleton

    # Endpoints: exactly 1 neighbor
    endpoints = (neighbor_count == 1) & skeleton

    return branch_points, endpoints


def split_skeleton_at_branches(
    skeleton: np.ndarray,
    branch_points: np.ndarray,
    min_length: int = 5
) -> List[np.ndarray]:
    """
    Split skeleton at branch points into separate segments.

    Args:
        skeleton: Boolean skeleton array
        branch_points: Boolean array of branch point locations
        min_length: Minimum segment length (in pixels)

    Returns:
        List of boolean masks, one per segment
    """
    # Remove branch points to disconnect segments
    skeleton_split = skeleton.copy()
    skeleton_split[branch_points] = False

    # Also remove pixels adjacent to branch points for cleaner separation
    kernel = np.ones((3, 3), dtype=np.uint8)
    branch_dilated = cv2.dilate(branch_points.astype(np.uint8), kernel)
    skeleton_split = skeleton_split & (branch_dilated == 0)

    # Connected component labeling
    labeled, num_features = ndimage.label(skeleton_split)

    segments = []
    for i in range(1, num_features + 1):
        segment_mask = (labeled == i)
        pixel_count = np.sum(segment_mask)
        if pixel_count >= min_length:
            segments.append(segment_mask)

    return segments


def restore_segment_thickness(
    skeleton_segment: np.ndarray,
    original_mask: np.ndarray,
    dilation_radius: int = 3
) -> np.ndarray:
    """
    Restore original thickness to a skeleton segment.

    Dilates the skeleton segment and intersects with the original mask
    to recover the actual hair thickness.

    Args:
        skeleton_segment: Boolean mask of skeleton segment
        original_mask: Original binary mask with actual thickness
        dilation_radius: Radius for dilation operation

    Returns:
        Binary mask (uint8, 0/255) with restored thickness
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_radius * 2 + 1, dilation_radius * 2 + 1)
    )

    # Convert skeleton segment to uint8
    segment_uint8 = skeleton_segment.astype(np.uint8) * 255

    # Dilate to approximate original thickness
    dilated = cv2.dilate(segment_uint8, kernel)

    # Ensure original_mask is uint8
    if original_mask.dtype == bool:
        original_uint8 = original_mask.astype(np.uint8) * 255
    else:
        original_uint8 = original_mask

    # Intersect with original mask to get actual hair pixels
    restored = cv2.bitwise_and(dilated, original_uint8)

    return restored


def simple_connected_component_separation(
    beard_mask: np.ndarray,
    erosion_iterations: int = 2
) -> List[np.ndarray]:
    """
    Separate connected components using erosion-based approach.

    This is a simpler alternative when skeletonization produces too many
    fragments. Works well when hairs are not too densely packed.

    Process:
    1. Erode mask to separate touching hairs
    2. Find connected components
    3. Dilate each component and intersect with original mask

    Args:
        beard_mask: Binary mask of beard region (uint8, 0/255)
        erosion_iterations: Number of erosion iterations

    Returns:
        List of binary masks (uint8, 0/255), one per component
    """
    # Ensure mask is uint8
    if beard_mask.dtype == bool:
        mask = beard_mask.astype(np.uint8) * 255
    else:
        mask = beard_mask.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Erode to separate touching hairs
    eroded = cv2.erode(mask, kernel, iterations=erosion_iterations)

    # Find connected components
    num_labels, labels = cv2.connectedComponents(eroded)

    individual_hairs = []
    for label_id in range(1, num_labels):
        # Extract component mask
        component_mask = (labels == label_id).astype(np.uint8) * 255

        # Dilate back to approximate original size
        dilated = cv2.dilate(component_mask, kernel, iterations=erosion_iterations)

        # Intersect with original mask
        restored = cv2.bitwise_and(dilated, mask)

        if cv2.countNonZero(restored) > 0:
            individual_hairs.append(restored)

    return individual_hairs


def preprocess_beard_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    remove_noise: bool = True,
    fill_holes: bool = True
) -> np.ndarray:
    """
    Preprocess beard mask before segmentation.

    Args:
        mask: Input binary mask
        kernel_size: Morphological kernel size
        remove_noise: Apply opening to remove small noise
        fill_holes: Apply closing to fill small holes

    Returns:
        Preprocessed binary mask (uint8, 0/255)
    """
    # Ensure uint8
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.max() == 1:
        mask = mask.astype(np.uint8) * 255
    else:
        mask = mask.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    if remove_noise:
        # Opening removes small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if fill_holes:
        # Closing fills small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def filter_by_shape(
    mask: np.ndarray,
    min_area: int = 10,
    max_area: int = 5000,
    min_aspect_ratio: float = 1.5
) -> bool:
    """
    Check if a mask passes shape-based filtering criteria.

    Args:
        mask: Binary mask to evaluate
        min_area: Minimum pixel area
        max_area: Maximum pixel area
        min_aspect_ratio: Minimum aspect ratio (length/width)

    Returns:
        True if mask passes all criteria
    """
    area = cv2.countNonZero(mask)

    # Area filter
    if area < min_area or area > max_area:
        return False

    # Find contours for aspect ratio
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return False

    # Use minimum area rectangle for aspect ratio
    rect = cv2.minAreaRect(contours[0])
    w, h = rect[1]

    if w == 0 or h == 0:
        return False

    aspect_ratio = max(w, h) / min(w, h)

    return aspect_ratio >= min_aspect_ratio


def calculate_centroid(mask: np.ndarray) -> Tuple[int, int]:
    """
    Calculate centroid of a binary mask.

    Args:
        mask: Binary mask

    Returns:
        (cx, cy) centroid coordinates
    """
    M = cv2.moments(mask.astype(np.uint8))

    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        # Fallback to mean of non-zero coordinates
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            cy = int(np.mean(coords[0]))
            cx = int(np.mean(coords[1]))
        else:
            cy, cx = 0, 0

    return cx, cy
