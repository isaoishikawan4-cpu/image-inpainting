"""
Single Hair Detection using Edge Detection + SAM (Segment Anything Model)

This app uses SAM's Automatic Mask Generation (AMG) for precise hair detection:
- SAM generates many small masks from dense point sampling
- Filters masks by area, aspect ratio, and brightness for hair-like shapes
- Edge detection mode also available for comparison

Supported GPU:
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon M1/M2/M3)
- CPU (fallback)

Inspired by: https://github.com/ymgw55/segment-anything-edge-detection

Usage:
    python app_single_hair_edge.py

Requirements:
    pip install gradio numpy opencv-python pillow scipy
    pip install torch torchvision
    pip install git+https://github.com/facebookresearch/segment-anything.git
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple, List, Dict
import os
import sys
from scipy import ndimage
from dataclasses import dataclass

from beard_inpainting_modules import (
    RegionSelector,
    DetectedRegion,
    visualize_single_hairs,
)

# Check SAM availability
SAM_AVAILABLE = False
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    pass


@dataclass
class EdgeSegment:
    """Represents a detected edge segment (potential hair)."""
    contour: np.ndarray
    length: float
    orientation: float  # degrees
    centroid: Tuple[int, int]
    curvature: float  # average curvature
    darkness: float  # average darkness along edge


def get_device() -> str:
    """
    Detect the best available device for PyTorch.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if not SAM_AVAILABLE:
        return "cpu"

    # CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return "cuda"

    # MPS (Apple Silicon - M1/M2/M3)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def patch_torch_for_mps():
    """
    Patch torch.as_tensor to work with MPS by converting float64 to float32.
    MPS doesn't support float64.
    """
    if not SAM_AVAILABLE:
        return

    _original_as_tensor = torch.as_tensor

    def _patched_as_tensor(data, dtype=None, device=None):
        # Check if targeting MPS device
        if device is not None:
            if isinstance(device, str) and device == "mps":
                is_mps = True
            elif hasattr(device, 'type') and device.type == "mps":
                is_mps = True
            else:
                is_mps = False

            if is_mps:
                # Convert numpy float64 to float32 for MPS
                if isinstance(data, np.ndarray) and data.dtype == np.float64:
                    data = data.astype(np.float32)
                # Force float32 if dtype would be float64
                if dtype == torch.float64:
                    dtype = torch.float32

        return _original_as_tensor(data, dtype=dtype, device=device)

    torch.as_tensor = _patched_as_tensor
    print("torch.as_tensor patched for MPS compatibility (float64 -> float32)")


# Apply MPS patch at module load
if SAM_AVAILABLE:
    patch_torch_for_mps()


class EdgeBasedHairDetector:
    """
    Hair detector using edge detection and SAM AMG.

    Approach:
    1. Apply edge detection (Canny or Sobel) to find boundaries
    2. Extract and filter edge segments for hair-like properties
    3. Use SAM AMG for precise mask generation

    Supported devices:
    - CUDA (NVIDIA GPU)
    - MPS (Apple Silicon M1/M2/M3)
    - CPU (fallback)
    """

    def __init__(self):
        self._sam_predictor: Optional[SamPredictor] = None
        self._sam_amg: Optional[SamAutomaticMaskGenerator] = None
        self._sam_model = None
        self._sam_initialized = False
        self._current_points_per_side = 64
        self.device = get_device()
        print(f"EdgeBasedHairDetector initialized with device: {self.device}")

    def _update_amg_params(self, points_per_side: int = 64):
        """Update SAM AMG with new parameters."""
        if self._sam_model is None or not self._sam_initialized:
            return

        if points_per_side != self._current_points_per_side:
            print(f"Updating SAM AMG: points_per_side={points_per_side}")
            self._sam_amg = SamAutomaticMaskGenerator(
                self._sam_model,
                points_per_side=points_per_side,
                pred_iou_thresh=0.5,
                stability_score_thresh=0.6,
                min_mask_region_area=3,
            )
            self._current_points_per_side = points_per_side

    def _init_sam(self) -> Tuple[bool, str]:
        """Initialize SAM model."""
        if self._sam_initialized:
            return True, "SAM already initialized"

        if not SAM_AVAILABLE:
            return False, "SAM not available"

        checkpoint_paths = [
            os.path.join(os.path.dirname(__file__), "checkpoints", "sam_vit_h_4b8939.pth"),
            "checkpoints/sam_vit_h_4b8939.pth",
        ]

        sam_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                sam_path = path
                break

        if sam_path is None:
            return False, "SAM checkpoint not found"

        try:
            print(f"Loading SAM from: {sam_path}")
            sam = sam_model_registry["vit_h"](checkpoint=sam_path)
            sam.to(device=self.device)
            self._sam_predictor = SamPredictor(sam)

            # Store SAM model for dynamic AMG creation
            self._sam_model = sam
            # Create default AMG (will be recreated with UI parameters)
            self._sam_amg = SamAutomaticMaskGenerator(
                sam,
                points_per_side=64,  # Very dense sampling for thin hairs
                pred_iou_thresh=0.5,  # Lower threshold to catch more candidates
                stability_score_thresh=0.6,  # Lower for more masks
                min_mask_region_area=3,  # Very small regions for thin hairs
            )

            self._sam_initialized = True
            print(f"SAM initialized on {self.device}")
            return True, f"SAM initialized on {self.device}"
        except Exception as e:
            return False, f"SAM initialization error: {e}"

    def detect_edges(
        self,
        image_rgb: np.ndarray,
        method: str = "canny",
        low_threshold: int = 50,
        high_threshold: int = 150,
        use_clahe: bool = True
    ) -> np.ndarray:
        """
        Detect edges in image using various methods.

        Args:
            image_rgb: RGB image
            method: 'canny', 'sobel', 'laplacian', or 'combined'
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold
            use_clahe: Apply CLAHE enhancement before edge detection

        Returns:
            Binary edge map
        """
        # Convert to grayscale (use L channel from LAB for better contrast)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        gray = lab[:, :, 0]

        # Optional CLAHE enhancement
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        if method == "canny":
            edges = cv2.Canny(blurred, low_threshold, high_threshold)

        elif method == "sobel":
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges / edges.max() * 255).astype(np.uint8)
            _, edges = cv2.threshold(edges, low_threshold, 255, cv2.THRESH_BINARY)

        elif method == "laplacian":
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            edges = np.abs(laplacian)
            edges = (edges / edges.max() * 255).astype(np.uint8)
            _, edges = cv2.threshold(edges, low_threshold, 255, cv2.THRESH_BINARY)

        elif method == "combined":
            # Combine multiple edge detectors
            canny = cv2.Canny(blurred, low_threshold, high_threshold)

            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = (sobel / sobel.max() * 255).astype(np.uint8)
            _, sobel = cv2.threshold(sobel, low_threshold, 255, cv2.THRESH_BINARY)

            # Union of edges
            edges = cv2.bitwise_or(canny, sobel)
        else:
            raise ValueError(f"Unknown edge method: {method}")

        return edges

    def extract_edge_segments(
        self,
        edge_map: np.ndarray,
        image_gray: np.ndarray,
        min_length: int = 10,
        max_gap: int = 3
    ) -> List[EdgeSegment]:
        """
        Extract individual edge segments from edge map.

        Args:
            edge_map: Binary edge map
            image_gray: Grayscale image for darkness calculation
            min_length: Minimum segment length in pixels
            max_gap: Maximum gap to bridge in pixels

        Returns:
            List of EdgeSegment objects
        """
        # Optional: Bridge small gaps in edges
        if max_gap > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max_gap, max_gap))
            edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)

        # Find contours (edge segments)
        contours, _ = cv2.findContours(
            edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        segments = []
        for contour in contours:
            # Calculate arc length (perimeter)
            length = cv2.arcLength(contour, closed=False)
            if length < min_length:
                continue

            # Calculate centroid
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Calculate orientation using fitted line
            if len(contour) >= 5:
                [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
                orientation = np.degrees(np.arctan2(vy, vx))
            else:
                orientation = 0

            # Calculate curvature (simplified: ratio of arc length to chord length)
            if len(contour) >= 2:
                start = contour[0][0]
                end = contour[-1][0]
                chord_length = np.linalg.norm(end - start)
                curvature = length / max(chord_length, 1) - 1  # 0 = straight line
            else:
                curvature = 0

            # Calculate average darkness along edge
            mask = np.zeros(image_gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, 2)
            darkness = 255 - cv2.mean(image_gray, mask=mask)[0]

            segments.append(EdgeSegment(
                contour=contour,
                length=length,
                orientation=orientation,
                centroid=(cx, cy),
                curvature=curvature,
                darkness=darkness
            ))

        return segments

    def filter_hair_segments(
        self,
        segments: List[EdgeSegment],
        min_length: int = 10,
        max_curvature: float = 2.0,
        min_darkness: float = 30,
        orientation_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[List[EdgeSegment], Dict[str, int]]:
        """
        Filter edge segments for hair-like properties.

        Hair characteristics:
        - Relatively straight (low curvature)
        - Dark against skin
        - Typically oriented in similar directions

        Args:
            segments: List of EdgeSegment
            min_length: Minimum length
            max_curvature: Maximum curvature (0 = straight)
            min_darkness: Minimum darkness value
            orientation_range: Optional (min, max) orientation in degrees

        Returns:
            Tuple of (filtered list, diagnostic stats)
        """
        stats = {
            'total': len(segments),
            'filtered_length': 0,
            'filtered_curvature': 0,
            'filtered_darkness': 0,
            'filtered_orientation': 0,
            'passed': 0
        }

        filtered = []
        for seg in segments:
            if seg.length < min_length:
                stats['filtered_length'] += 1
                continue
            if seg.curvature > max_curvature:
                stats['filtered_curvature'] += 1
                continue
            if seg.darkness < min_darkness:
                stats['filtered_darkness'] += 1
                continue
            if orientation_range is not None:
                min_orient, max_orient = orientation_range
                if not (min_orient <= seg.orientation <= max_orient):
                    # Check wrapped angle
                    if not (min_orient <= seg.orientation + 180 <= max_orient):
                        stats['filtered_orientation'] += 1
                        continue
            filtered.append(seg)
            stats['passed'] += 1

        print(f"Edge filter stats: {stats}")
        return filtered, stats

    def segments_to_masks(
        self,
        segments: List[EdgeSegment],
        image_shape: Tuple[int, int],
        dilation_radius: int = 2
    ) -> List[DetectedRegion]:
        """
        Convert edge segments to detection masks.

        Args:
            segments: List of EdgeSegment
            image_shape: (height, width)
            dilation_radius: Radius to dilate edge into mask

        Returns:
            List of DetectedRegion
        """
        results = []
        h, w = image_shape

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_radius * 2 + 1, dilation_radius * 2 + 1)
        )

        for i, seg in enumerate(segments):
            # Draw contour and dilate
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [seg.contour], -1, 255, 1)
            mask = cv2.dilate(mask, kernel)

            area = cv2.countNonZero(mask)

            results.append(DetectedRegion(
                mask=mask,
                area=area,
                centroid=seg.centroid,
                confidence=seg.darkness / 255.0,  # Use darkness as confidence
                source='edge_detection',
                phrase=f"hair_{i+1}"
            ))

        return results

    def _calculate_tiles(
        self,
        region_width: int,
        region_height: int,
        tile_size: int = 400,
        overlap: int = 50
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate tile positions for a region.

        Args:
            region_width: Width of the region
            region_height: Height of the region
            tile_size: Target tile size (will not split if region is smaller)
            overlap: Overlap between adjacent tiles

        Returns:
            List of (x1, y1, x2, y2) tile coordinates relative to region
        """
        tiles = []

        # If region is smaller than tile_size, return single tile
        if region_width <= tile_size and region_height <= tile_size:
            return [(0, 0, region_width, region_height)]

        # Calculate step size (tile_size - overlap)
        step = max(tile_size - overlap, 1)

        # Generate tile positions
        y = 0
        while y < region_height:
            x = 0
            while x < region_width:
                # Calculate tile bounds
                x1 = x
                y1 = y
                x2 = min(x + tile_size, region_width)
                y2 = min(y + tile_size, region_height)

                # Ensure minimum tile size (at least overlap size)
                if x2 - x1 >= overlap and y2 - y1 >= overlap:
                    tiles.append((x1, y1, x2, y2))

                x += step
                # Break if we've reached the end
                if x2 >= region_width:
                    break

            y += step
            if y2 >= region_height:
                break

        return tiles

    def _process_single_tile(
        self,
        tile_rgb: np.ndarray,
        tile_offset: Tuple[int, int],
        region_offset: Tuple[int, int],
        full_image_shape: Tuple[int, int],
        min_area: int,
        max_area: int,
        min_aspect: float,
        brightness_threshold: float
    ) -> List[DetectedRegion]:
        """
        Process a single tile with SAM AMG.

        Args:
            tile_rgb: Tile image (RGB)
            tile_offset: (x, y) offset of tile within region
            region_offset: (x, y) offset of region within full image
            full_image_shape: (height, width) of full image
            min_area, max_area, min_aspect, brightness_threshold: Filter params

        Returns:
            List of DetectedRegion with coordinates in full image space
        """
        h, w = full_image_shape
        tile_x, tile_y = tile_offset
        region_x, region_y = region_offset

        # Run SAM AMG on tile
        masks = self._sam_amg.generate(tile_rgb)

        # Convert tile to grayscale for brightness check
        tile_gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(tile_gray)

        results = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            area = mask_data['area']

            # Filter by area
            if area < min_area or area > max_area:
                continue

            # Check aspect ratio
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            rect = cv2.minAreaRect(contours[0])
            w_rect, h_rect = rect[1]
            if w_rect == 0 or h_rect == 0:
                continue
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

            if aspect < min_aspect:
                continue

            # Brightness check
            mask_brightness = cv2.mean(tile_gray, mask=mask)[0]
            if mask_brightness > mean_brightness * brightness_threshold:
                continue

            # Create full-size mask with correct offset
            full_mask = np.zeros((h, w), dtype=np.uint8)
            # Offset: region_offset + tile_offset
            abs_x = region_x + tile_x
            abs_y = region_y + tile_y
            tile_h, tile_w = tile_rgb.shape[:2]

            # Place tile mask in full image
            full_mask[abs_y:abs_y+tile_h, abs_x:abs_x+tile_w] = mask

            # Calculate centroid in full image coordinates
            M = cv2.moments(mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00']) + abs_x
                cy = int(M['m01'] / M['m00']) + abs_y
            else:
                cx = abs_x + tile_w // 2
                cy = abs_y + tile_h // 2

            results.append(DetectedRegion(
                mask=full_mask,
                area=area,
                centroid=(cx, cy),
                confidence=mask_data.get('stability_score', 0.5),
                source='sam_amg_tiled',
                phrase=f"hair_{i+1}"
            ))

        return results

    def _remove_tile_duplicates(
        self,
        detections: List[DetectedRegion],
        overlap_threshold: float = 0.5
    ) -> List[DetectedRegion]:
        """
        Remove duplicate detections from overlapping tiles.

        Uses IoU (Intersection over Union) with the smaller mask as reference.

        Args:
            detections: List of DetectedRegion from all tiles
            overlap_threshold: Threshold for considering as duplicate

        Returns:
            Deduplicated list of DetectedRegion
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (higher first) to keep better detections
        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)

        unique = []
        for det in sorted_dets:
            is_duplicate = False
            for existing in unique:
                # Calculate intersection
                intersection = cv2.bitwise_and(det.mask, existing.mask)
                intersection_area = cv2.countNonZero(intersection)

                # Use smaller area as reference (more strict for small hairs)
                min_area = min(det.area, existing.area)
                if min_area > 0 and intersection_area / min_area > overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(det)

        return unique

    def detect_with_sam_amg(
        self,
        image_rgb: np.ndarray,
        region_box: Tuple[int, int, int, int],
        min_area: int = 10,
        max_area: int = 1000,
        min_aspect: float = 1.5,
        brightness_threshold: float = 1.10,
        points_per_side: int = 64,
        use_tiling: bool = False,
        tile_size: int = 400,
        tile_overlap: int = 50
    ) -> Tuple[List[DetectedRegion], List[np.ndarray], Dict[str, int]]:
        """
        Use SAM Automatic Mask Generation for hair detection.

        This approach generates many small masks and filters for hair-like shapes.

        Args:
            image_rgb: Full RGB image
            region_box: (x1, y1, x2, y2) ROI
            min_area: Minimum mask area
            max_area: Maximum mask area
            min_aspect: Minimum aspect ratio for hair-like shape
            brightness_threshold: Max brightness ratio vs mean (1.0 = same as mean)
            points_per_side: SAM sampling density (higher = more masks, slower)
            use_tiling: Enable tile-based processing for large regions
            tile_size: Size of each tile (default 400px, good for 1600x500 etc.)
            tile_overlap: Overlap between tiles (default 50px)

        Returns:
            Tuple of (List of DetectedRegion, List of all masks (unfiltered), diagnostic stats)
        """
        empty_stats = {'total': 0, 'filtered_area_small': 0, 'filtered_area_large': 0, 'filtered_aspect': 0, 'filtered_brightness': 0, 'passed': 0, 'tiles': 0}

        if not self._sam_initialized:
            success, msg = self._init_sam()
            if not success:
                print(f"SAM AMG not available: {msg}")
                return [], [], empty_stats

        # Update AMG parameters if needed
        self._update_amg_params(points_per_side)

        x1, y1, x2, y2 = region_box
        h, w = image_rgb.shape[:2]

        # Crop ROI
        roi = image_rgb[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Tiled processing for large regions
        if use_tiling:
            tiles = self._calculate_tiles(roi_w, roi_h, tile_size, tile_overlap)
            print(f"Tile-based processing: {len(tiles)} tiles (tile_size={tile_size}, overlap={tile_overlap})")
            print(f"Region size: {roi_w}x{roi_h}")

            all_results = []
            all_masks_unfiltered = []  # Store all masks before filtering
            stats = {
                'total': 0,
                'filtered_area_small': 0,
                'filtered_area_large': 0,
                'filtered_aspect': 0,
                'filtered_brightness': 0,
                'passed': 0,
                'tiles': len(tiles)
            }

            for idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                tile_rgb = roi[ty1:ty2, tx1:tx2]
                print(f"  Processing tile {idx+1}/{len(tiles)}: ({tx1},{ty1})-({tx2},{ty2}) size={tx2-tx1}x{ty2-ty1}")

                # Run SAM AMG on tile to get all masks
                tile_masks = self._sam_amg.generate(tile_rgb)
                stats['total'] += len(tile_masks)

                # Store all masks (unfiltered) with correct offset
                tile_h, tile_w = tile_rgb.shape[:2]
                abs_x = x1 + tx1
                abs_y = y1 + ty1
                for mask_data in tile_masks:
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    mask = mask_data['segmentation'].astype(np.uint8) * 255
                    full_mask[abs_y:abs_y+tile_h, abs_x:abs_x+tile_w] = mask
                    all_masks_unfiltered.append(full_mask)

                tile_results = self._process_single_tile(
                    tile_rgb,
                    tile_offset=(tx1, ty1),
                    region_offset=(x1, y1),
                    full_image_shape=(h, w),
                    min_area=min_area,
                    max_area=max_area,
                    min_aspect=min_aspect,
                    brightness_threshold=brightness_threshold
                )
                all_results.extend(tile_results)
                stats['passed'] += len(tile_results)

            # Remove duplicates from overlapping tiles
            print(f"Before deduplication: {len(all_results)} detections")
            all_results = self._remove_tile_duplicates(all_results, overlap_threshold=0.5)
            print(f"After deduplication: {len(all_results)} detections")

            stats['passed'] = len(all_results)
            return all_results, all_masks_unfiltered, stats

        # Non-tiled processing (original behavior)
        print("Running SAM Automatic Mask Generation...")
        masks = self._sam_amg.generate(roi)
        print(f"SAM AMG generated {len(masks)} masks")

        # Store all masks (unfiltered) for visualization
        all_masks_unfiltered = []
        for mask_data in masks:
            full_mask = np.zeros((h, w), dtype=np.uint8)
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            full_mask[y1:y2, x1:x2] = mask
            all_masks_unfiltered.append(full_mask)

        # Convert ROI to grayscale for darkness check
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(roi_gray)

        # Diagnostic counters
        stats = {
            'total': len(masks),
            'filtered_area_small': 0,
            'filtered_area_large': 0,
            'filtered_aspect': 0,
            'filtered_brightness': 0,
            'passed': 0,
            'tiles': 1  # Non-tiled = 1 tile (the whole region)
        }

        results = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            area = mask_data['area']

            # Filter by area
            if area < min_area:
                stats['filtered_area_small'] += 1
                continue
            if area > max_area:
                stats['filtered_area_large'] += 1
                continue

            # Check aspect ratio (hair should be elongated)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            rect = cv2.minAreaRect(contours[0])
            w_rect, h_rect = rect[1]
            if w_rect == 0 or h_rect == 0:
                continue
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

            # Hair should be elongated
            if aspect < min_aspect:
                stats['filtered_aspect'] += 1
                continue

            # Darkness check: hair should be darker than surrounding skin
            mask_brightness = cv2.mean(roi_gray, mask=mask)[0]
            # Filter out masks that are too bright (adjustable via brightness_threshold)
            if mask_brightness > mean_brightness * brightness_threshold:
                stats['filtered_brightness'] += 1
                continue

            stats['passed'] += 1

            # Create full-size mask
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask

            # Calculate centroid in full image coordinates
            M = cv2.moments(mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00']) + x1
                cy = int(M['m01'] / M['m00']) + y1
            else:
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

            results.append(DetectedRegion(
                mask=full_mask,
                area=area,
                centroid=(cx, cy),
                confidence=mask_data.get('stability_score', 0.5),
                source='sam_amg',
                phrase=f"hair_{i+1}"
            ))

        print(f"SAM AMG stats: {stats}")
        print(f"SAM AMG filtered to {len(results)} hair-like masks")
        return results, all_masks_unfiltered, stats


class EdgeDetectionApp:
    """Gradio application for edge-based hair detection."""

    def __init__(self):
        self._detector = EdgeBasedHairDetector()
        self._current_image: Optional[np.ndarray] = None
        self._detections: List[DetectedRegion] = []
        self._edge_map: Optional[np.ndarray] = None

    def detect_hairs(
        self,
        editor_data: dict,
        detection_mode: str,
        edge_method: str,
        low_threshold: int,
        high_threshold: int,
        min_edge_length: int,
        max_curvature: float,
        min_darkness: int,
        dilation_radius: int,
        use_clahe: bool,
        sam_points_per_side: int,
        sam_min_area: int,
        sam_max_area: int,
        sam_min_aspect: float,
        sam_brightness_threshold: float,
        use_tiling: bool = False,
        tile_size: int = 400,
        tile_overlap: int = 50,
        use_coordinates: bool = False,
        coord_x1: int = 0,
        coord_y1: int = 0,
        coord_x2: int = 100,
        coord_y2: int = 100,
        overlay_alpha: float = 0.3,
        show_markers: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int, str]:
        """Main detection function."""

        # Validate input
        if editor_data is None:
            return None, None, None, None, 0, "Please upload an image first"

        # Extract image
        if 'background' in editor_data:
            image = editor_data['background']
        elif 'composite' in editor_data:
            image = editor_data['composite']
        else:
            return None, None, None, None, 0, "Invalid image data"

        if image is None:
            return None, None, None, None, 0, "No image found"

        # Convert to numpy RGB
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        self._current_image = image
        h, w = image.shape[:2]

        # Extract rectangle region (from coordinates or drawn rectangle)
        if use_coordinates:
            # Use coordinate input
            x1 = max(0, int(coord_x1))
            y1 = max(0, int(coord_y1))
            x2 = min(w, int(coord_x2))
            y2 = min(h, int(coord_y2))

            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                return image, None, None, None, 0, f"Invalid coordinates: ({x1},{y1})-({x2},{y2}). X2>X1, Y2>Y1 required."
            if x2 - x1 < 10 or y2 - y1 < 10:
                return image, None, None, None, 0, f"Region too small: {x2-x1}x{y2-y1}. Minimum 10x10 required."

            rect = (x1, y1, x2, y2)
            print(f"Using coordinate input: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px")
        else:
            # Use drawn rectangle
            rect = RegionSelector.extract_rectangle(editor_data)
            if rect is None:
                return image, None, None, None, 0, "Please draw a rectangle to select the detection region"
            x1, y1, x2, y2 = rect

        # Crop ROI
        roi = image[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        detections = []
        all_masks_unfiltered = []  # Store all SAM masks before filtering
        edge_vis = None

        if detection_mode == "edge_detection":
            # Pure edge detection approach

            # Step 1: Detect edges
            edge_map = self._detector.detect_edges(
                roi,
                method=edge_method,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                use_clahe=use_clahe
            )
            self._edge_map = edge_map

            # Create edge visualization
            edge_vis_roi = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
            edge_vis = np.zeros((h, w, 3), dtype=np.uint8)
            edge_vis[y1:y2, x1:x2] = edge_vis_roi

            # Step 2: Extract edge segments
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            segments = self._detector.extract_edge_segments(
                edge_map, gray_roi,
                min_length=min_edge_length,
                max_gap=2
            )
            print(f"Extracted {len(segments)} edge segments")

            # Step 3: Filter for hair-like segments
            hair_segments, filter_stats = self._detector.filter_hair_segments(
                segments,
                min_length=min_edge_length,
                max_curvature=max_curvature,
                min_darkness=min_darkness
            )
            print(f"Filtered to {len(hair_segments)} hair-like segments")

            # Step 4: Convert to masks (offset to full image coords)
            for seg in hair_segments:
                seg.contour = seg.contour + np.array([x1, y1])
                seg.centroid = (seg.centroid[0] + x1, seg.centroid[1] + y1)

            detections = self._detector.segments_to_masks(
                hair_segments, (h, w),
                dilation_radius=dilation_radius
            )

            status = f"Edge detection ({edge_method}): {len(detections)} hairs\n"
            status += f"Edge segments: {len(segments)} -> Hair-like: {len(hair_segments)}\n"
            status += f"Filtered: length={filter_stats['filtered_length']}, curvature={filter_stats['filtered_curvature']}, darkness={filter_stats['filtered_darkness']}"

        elif detection_mode == "sam_amg":
            # SAM Automatic Mask Generation approach
            detections, all_masks_unfiltered, sam_stats = self._detector.detect_with_sam_amg(
                image, rect,
                min_area=sam_min_area,
                max_area=sam_max_area,
                min_aspect=sam_min_aspect,
                brightness_threshold=sam_brightness_threshold,
                points_per_side=sam_points_per_side,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap
            )

            # No edge map for this mode
            edge_vis = np.zeros((h, w, 3), dtype=np.uint8)

            # Build status message
            tiles_info = f", tiles={sam_stats.get('tiles', 1)}" if use_tiling else ""
            status = f"SAM AMG: {len(detections)} hair-like masks detected\n"
            status += f"Region: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px\n"
            status += f"Total masks: {sam_stats['total']} (points_per_side={sam_points_per_side}{tiles_info})\n"
            status += f"Filtered: area_small={sam_stats['filtered_area_small']}, area_large={sam_stats['filtered_area_large']}, aspect={sam_stats['filtered_aspect']}, brightness={sam_stats.get('filtered_brightness', 0)}"

        elif detection_mode == "edge_then_sam":
            # Hybrid: Use edges to find hair locations, then SAM to refine
            # Step 1: Edge detection
            edge_map = self._detector.detect_edges(
                roi,
                method=edge_method,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                use_clahe=use_clahe
            )
            self._edge_map = edge_map

            edge_vis_roi = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
            edge_vis = np.zeros((h, w, 3), dtype=np.uint8)
            edge_vis[y1:y2, x1:x2] = edge_vis_roi

            # Step 2: Extract and filter segments
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            segments = self._detector.extract_edge_segments(
                edge_map, gray_roi,
                min_length=min_edge_length,
                max_gap=2
            )
            hair_segments, filter_stats = self._detector.filter_hair_segments(
                segments,
                min_length=min_edge_length,
                max_curvature=max_curvature,
                min_darkness=min_darkness
            )

            # Step 3: Use segment centroids as SAM point prompts
            if SAM_AVAILABLE and len(hair_segments) > 0:
                success, msg = self._detector._init_sam()
                if success:
                    self._detector._sam_predictor.set_image(image)

                    for i, seg in enumerate(hair_segments):
                        cx = seg.centroid[0] + x1
                        cy = seg.centroid[1] + y1

                        try:
                            masks, scores, _ = self._detector._sam_predictor.predict(
                                point_coords=np.array([[cx, cy]]),
                                point_labels=np.array([1]),
                                multimask_output=True
                            )

                            # Select smallest mask (most likely to be single hair)
                            areas = [m.sum() for m in masks]
                            best_idx = np.argmin(areas)
                            mask = masks[best_idx].astype(np.uint8) * 255

                            # Validate mask size
                            mask_area = cv2.countNonZero(mask)
                            if sam_min_area <= mask_area <= sam_max_area:
                                detections.append(DetectedRegion(
                                    mask=mask,
                                    area=mask_area,
                                    centroid=(cx, cy),
                                    confidence=float(scores[best_idx]),
                                    source='edge_then_sam',
                                    phrase=f"hair_{i+1}"
                                ))
                        except Exception as e:
                            print(f"SAM error for point ({cx}, {cy}): {e}")

                    status = f"Edge->SAM hybrid: {len(detections)} hairs\n"
                    status += f"Edge segments: {len(hair_segments)}, SAM refined: {len(detections)}"
                else:
                    # Fallback to edge-only
                    for seg in hair_segments:
                        seg.contour = seg.contour + np.array([x1, y1])
                        seg.centroid = (seg.centroid[0] + x1, seg.centroid[1] + y1)
                    detections = self._detector.segments_to_masks(
                        hair_segments, (h, w), dilation_radius=dilation_radius
                    )
                    status = f"Edge detection (SAM unavailable): {len(detections)} hairs"
            else:
                # No SAM, use edge-only
                for seg in hair_segments:
                    seg.contour = seg.contour + np.array([x1, y1])
                    seg.centroid = (seg.centroid[0] + x1, seg.centroid[1] + y1)
                detections = self._detector.segments_to_masks(
                    hair_segments, (h, w), dilation_radius=dilation_radius
                )
                status = f"Edge detection: {len(detections)} hairs"

        else:
            return None, None, None, None, 0, f"Unknown mode: {detection_mode}"

        # Remove duplicates
        detections = self._remove_duplicates(detections)
        self._detections = detections

        # Create visualization
        if len(detections) > 0:
            result_image = visualize_single_hairs(
                image, detections,
                alpha=overlay_alpha,
                show_markers=show_markers
            )
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Combined mask with colors (filtered)
            mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
            num_det = len(detections)
            for i, det in enumerate(detections):
                # Generate color for each detection
                hue = int(180 * i / max(num_det, 1))
                hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
                color = tuple(int(c) for c in rgb)
                # Apply color to mask
                mask_bool = det.mask > 0
                mask_vis[mask_bool] = color
        else:
            result_image = image.copy()
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            mask_vis = np.zeros_like(image)

        # Create ALL Mask visualization (unfiltered but exclude huge masks)
        # Filter out masks that are too large (same max_area threshold)
        # and sort by area (large first) so small masks are drawn on top
        all_mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        if len(all_masks_unfiltered) > 0:
            # Calculate area and filter out huge masks
            masks_with_area = []
            for mask in all_masks_unfiltered:
                area = cv2.countNonZero(mask)
                if area <= sam_max_area:  # Exclude masks larger than max_area
                    masks_with_area.append((mask, area))

            # Sort by area descending (large masks drawn first, small on top)
            masks_with_area.sort(key=lambda x: x[1], reverse=True)

            num_all = len(masks_with_area)
            for i, (mask, area) in enumerate(masks_with_area):
                # Generate color for each mask
                hue = int(180 * i / max(num_all, 1))
                hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
                color = tuple(int(c) for c in rgb)
                # Apply color to mask
                mask_bool = mask > 0
                all_mask_vis[mask_bool] = color

        return result_image, edge_vis, mask_vis, all_mask_vis, len(detections), status

    def _remove_duplicates(
        self,
        detections: List[DetectedRegion],
        overlap_threshold: float = 0.5  # Increased from 0.3 to keep more detections
    ) -> List[DetectedRegion]:
        """Remove duplicate detections."""
        if len(detections) <= 1:
            return detections

        unique = []
        for det in detections:
            is_duplicate = False
            for existing in unique:
                intersection = cv2.bitwise_and(det.mask, existing.mask)
                intersection_area = cv2.countNonZero(intersection)
                min_area = min(det.area, existing.area)
                if min_area > 0 and intersection_area / min_area > overlap_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(det)
        return unique

    def preview_region(
        self,
        editor_data: dict,
        coord_x1: int,
        coord_y1: int,
        coord_x2: int,
        coord_y2: int
    ) -> Optional[np.ndarray]:
        """Preview the selected region with coordinates."""
        if editor_data is None:
            return None

        # Extract image
        if 'background' in editor_data:
            image = editor_data['background']
        elif 'composite' in editor_data:
            image = editor_data['composite']
        else:
            return None

        if image is None:
            return None

        # Convert to numpy RGB
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        h, w = image.shape[:2]

        # Clamp coordinates
        x1 = max(0, int(coord_x1))
        y1 = max(0, int(coord_y1))
        x2 = min(w, int(coord_x2))
        y2 = min(h, int(coord_y2))

        # Create preview with rectangle
        preview = image.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text showing size
        text = f"({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px"
        cv2.putText(preview, text, (x1, max(y1-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return preview

    def get_details(self) -> str:
        """Get detection details."""
        if not self._detections:
            return "No detections yet"

        lines = [f"Total: {len(self._detections)} hairs\n"]

        # Group by source
        sources = {}
        for d in self._detections:
            sources[d.source] = sources.get(d.source, 0) + 1

        lines.append("By source:")
        for source, count in sources.items():
            lines.append(f"  {source}: {count}")

        # Area stats
        areas = [d.area for d in self._detections]
        if areas:
            lines.append(f"\nArea statistics:")
            lines.append(f"  Min: {min(areas)} px")
            lines.append(f"  Max: {max(areas)} px")
            lines.append(f"  Avg: {sum(areas)/len(areas):.1f} px")

        return "\n".join(lines)


def create_app():
    """Create Gradio application."""
    app = EdgeDetectionApp()

    with gr.Blocks(
        title="Edge + SAM Hair Detection",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # Edge + SAM Hair Detection (app_single_hair_edge.py)

        SAM (Segment Anything Model) の Automatic Mask Generation を使用した髭検出:

        **推奨モード: sam_amg**
        - SAMが密なポイントサンプリングから多数のマスクを生成
        - 面積、アスペクト比、暗さでフィルタリングして髭を抽出
        - 精度が高く、細い髭も検出可能

        **参考**: https://github.com/ymgw55/segment-anything-edge-detection
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_editor = gr.ImageEditor(
                    label="Input Image (draw rectangle)",
                    type="numpy",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=5),
                    height=500,
                )

                with gr.Accordion("Region Selection (座標指定)", open=False):
                    use_coordinates = gr.Checkbox(
                        value=False,
                        label="Use Coordinate Input (座標入力を使用)",
                        info="ONにすると矩形描画ではなく座標入力で範囲指定"
                    )
                    with gr.Row():
                        coord_x1 = gr.Number(label="X1 (左)", value=0, precision=0)
                        coord_y1 = gr.Number(label="Y1 (上)", value=0, precision=0)
                    with gr.Row():
                        coord_x2 = gr.Number(label="X2 (右)", value=100, precision=0)
                        coord_y2 = gr.Number(label="Y2 (下)", value=100, precision=0)
                    preview_btn = gr.Button("Preview Region (プレビュー)", size="sm")
                    coord_preview = gr.Image(label="Coordinate Preview", type="numpy", height=200)
                    gr.Markdown("*座標は画像の左上が原点 (0,0)*")

                with gr.Accordion("Detection Mode", open=True):
                    detection_mode = gr.Radio(
                        choices=["edge_detection", "sam_amg", "edge_then_sam"],
                        value="sam_amg",
                        label="Detection Mode",
                        info="sam_amg推奨: SAM自動マスク生成 | edge_detection: エッジ検出 | edge_then_sam: ハイブリッド"
                    )

                with gr.Accordion("Edge Detection Settings", open=False):
                    edge_method = gr.Radio(
                        choices=["canny", "sobel", "laplacian", "combined"],
                        value="canny",
                        label="Edge Method"
                    )
                    with gr.Row():
                        low_threshold = gr.Slider(
                            minimum=10, maximum=200, value=50, step=5,
                            label="Low Threshold"
                        )
                        high_threshold = gr.Slider(
                            minimum=50, maximum=300, value=150, step=5,
                            label="High Threshold"
                        )
                    use_clahe = gr.Checkbox(value=True, label="Use CLAHE Enhancement")

                with gr.Accordion("Hair Filtering (for edge_detection)", open=False):
                    min_edge_length = gr.Slider(
                        minimum=3, maximum=50, value=8, step=1,
                        label="Min Edge Length (px)",
                        info="Lower = detect shorter edges. Try 5-15"
                    )
                    max_curvature = gr.Slider(
                        minimum=0.5, maximum=10.0, value=5.0, step=0.5,
                        label="Max Curvature",
                        info="0 = straight line only. Higher = allow more curved edges. Try 3-8"
                    )
                    min_darkness = gr.Slider(
                        minimum=0, maximum=100, value=10, step=5,
                        label="Min Darkness",
                        info="Higher = only detect darker edges. Try 5-30"
                    )
                    dilation_radius = gr.Slider(
                        minimum=1, maximum=5, value=2, step=1,
                        label="Dilation Radius",
                        info="Thickness of detected hair mask"
                    )

                with gr.Accordion("SAM Settings (for sam_amg/edge_then_sam)", open=True):
                    sam_points_per_side = gr.Slider(
                        minimum=32, maximum=128, value=64, step=8,
                        label="SAM Points Per Side",
                        info="Sampling density. Higher = more masks (64-96 for thin hairs, slower)"
                    )
                    sam_min_area = gr.Slider(
                        minimum=1, maximum=100, value=5, step=1,
                        label="SAM Min Area",
                        info="Minimum mask area. Lower = detect smaller hairs"
                    )
                    sam_max_area = gr.Slider(
                        minimum=100, maximum=5000, value=2000, step=100,
                        label="SAM Max Area",
                        info="Maximum mask area. Higher = include larger hairs"
                    )
                    sam_min_aspect = gr.Slider(
                        minimum=1.0, maximum=5.0, value=1.2, step=0.1,
                        label="SAM Min Aspect Ratio",
                        info="Hair shape filter. Lower = allow rounder shapes (1.2 recommended)"
                    )
                    sam_brightness_threshold = gr.Slider(
                        minimum=0.90, maximum=1.20, value=1.14, step=0.02,
                        label="SAM Brightness Threshold",
                        info="Higher = allow brighter masks (1.10-1.15 recommended)"
                    )

                with gr.Accordion("Tile Processing (for large images)", open=False):
                    use_tiling = gr.Checkbox(
                        value=False,
                        label="Enable Tile Processing",
                        info="Split large regions into tiles. Recommended for regions > 500x500px"
                    )
                    tile_size = gr.Slider(
                        minimum=200, maximum=800, value=400, step=50,
                        label="Tile Size (px)",
                        info="Size of each tile. 300-500 for most cases. Smaller = faster per tile"
                    )
                    tile_overlap = gr.Slider(
                        minimum=20, maximum=150, value=50, step=10,
                        label="Tile Overlap (px)",
                        info="Overlap between tiles to avoid missing hairs at boundaries"
                    )

                with gr.Accordion("Visualization Settings (可視化設定)", open=False):
                    overlay_alpha = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                        label="Overlay Alpha (オーバーレイ透明度)",
                        info="Low = more transparent, High = more opaque. 0.2-0.4 recommended"
                    )
                    show_markers = gr.Checkbox(
                        value=True,
                        label="Show Center Markers (中心マーカー表示)",
                        info="Show diamond markers at hair centers"
                    )

                detect_btn = gr.Button("Detect Hairs", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_image = gr.Image(label="Detection Result", type="numpy", height=300)
                edge_image = gr.Image(label="Edge Map", type="numpy", height=200)
                with gr.Row():
                    mask_image = gr.Image(label="Combined Mask (Filtered)", type="numpy", height=200)
                    all_mask_image = gr.Image(label="ALL Mask (Unfiltered)", type="numpy", height=200)

                with gr.Row():
                    hair_count = gr.Number(label="Hair Count", value=0, precision=0)
                status_text = gr.Textbox(label="Status", lines=4)

                details_btn = gr.Button("Show Details")
                details_text = gr.Textbox(label="Details", lines=8)

        detect_btn.click(
            fn=app.detect_hairs,
            inputs=[
                image_editor,
                detection_mode,
                edge_method,
                low_threshold,
                high_threshold,
                min_edge_length,
                max_curvature,
                min_darkness,
                dilation_radius,
                use_clahe,
                sam_points_per_side,
                sam_min_area,
                sam_max_area,
                sam_min_aspect,
                sam_brightness_threshold,
                use_tiling,
                tile_size,
                tile_overlap,
                use_coordinates,
                coord_x1,
                coord_y1,
                coord_x2,
                coord_y2,
                overlay_alpha,
                show_markers,
            ],
            outputs=[result_image, edge_image, mask_image, all_mask_image, hair_count, status_text]
        )

        preview_btn.click(
            fn=app.preview_region,
            inputs=[image_editor, coord_x1, coord_y1, coord_x2, coord_y2],
            outputs=[coord_preview]
        )

        details_btn.click(
            fn=app.get_details,
            outputs=[details_text]
        )

        gr.Markdown("""
        ---
        ### 手法比較

        | 項目 | rule_based (app_single_hair.py) | SAM AMG (このアプリ) |
        |------|--------------------------------|---------------------|
        | 検出原理 | 適応的閾値処理 | SAM自動マスク生成 |
        | 精度 | 良好 | より高い |
        | 速度 | 高速 | 遅い（GPU推奨） |
        | 細い髭 | やや苦手 | 検出可能 |

        ### パラメータ調整履歴

        | パラメータ | 旧値 | 新値（推奨） | 備考 |
        |-----------|------|-------------|------|
        | SAM Min Aspect Ratio | 1.5 | 1.2 | 細い髭も検出 |
        | SAM Brightness Threshold | 1.10 | 1.14 | より明るい髭も検出 |
        | SAM Points Per Side | 48 | 64 | より多くのマスク生成 |

        ### Tips

        - **sam_amg** モードを推奨（精度が高い）
        - 細い髭が検出されない場合: `SAM Points Per Side` を 80-96 に上げる
        - 処理が遅い場合: `SAM Points Per Side` を 48-56 に下げる
        - GPU（CUDA）使用で大幅に高速化

        ### タイル分割処理（大きい画像向け）

        大きい領域を選択した場合、処理時間が長くなります。**Tile Processing** を有効にすると：

        | 領域サイズ | タイル分割 | 推奨設定 |
        |-----------|-----------|----------|
        | 200x200以下 | 不要（OFF） | そのまま処理 |
        | 500x500程度 | 任意 | tile_size=400 |
        | 1000x1000以上 | 推奨（ON） | tile_size=400, overlap=50 |
        | 1600x500など横長 | 推奨（ON） | tile_size=400-500 |

        - **Tile Size**: 300-500px が推奨。小さいほど1タイルあたりは速いが、タイル数が増える
        - **Tile Overlap**: 50px で十分。境界の髭を拾うため
        - 小さい画像（200x200等）では OFF にしてください
        """)

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7863,  # Different port from other apps
        share=False
    )
