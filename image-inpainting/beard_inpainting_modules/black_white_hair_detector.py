"""Black and White Hair Detector using SAM Automatic Mask Generation.

This module provides detection for:
- Class 1: Black hair (dark hairs against lighter skin)
- Class 2: White hair (light/gray hairs against skin)

Each class has its own filter parameters for optimal detection.
Uses SAM (Segment Anything Model) Automatic Mask Generation for precise segmentation.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
import os
from dataclasses import dataclass

from .beard_detector import DetectedRegion

# Check SAM availability
SAM_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    pass


def get_device() -> str:
    """Detect the best available device for PyTorch."""
    if not TORCH_AVAILABLE:
        return "cpu"
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def patch_torch_for_mps():
    """Patch torch.as_tensor to work with MPS (float64 -> float32)."""
    if not TORCH_AVAILABLE:
        return

    import torch

    _original_as_tensor = torch.as_tensor

    def _patched_as_tensor(data, dtype=None, device=None):
        if device is not None:
            if isinstance(device, str) and device == "mps":
                is_mps = True
            elif hasattr(device, 'type') and device.type == "mps":
                is_mps = True
            else:
                is_mps = False

            if is_mps:
                if isinstance(data, np.ndarray) and data.dtype == np.float64:
                    data = data.astype(np.float32)
                if dtype == torch.float64:
                    dtype = torch.float32

        return _original_as_tensor(data, dtype=dtype, device=device)

    torch.as_tensor = _patched_as_tensor
    print("torch.as_tensor patched for MPS compatibility")


# Apply MPS patch at import time
if SAM_AVAILABLE:
    patch_torch_for_mps()


@dataclass
class HairClassParams:
    """Parameters for a specific hair color class."""
    min_area: int
    max_area: int
    min_aspect: float
    brightness_threshold: float
    brightness_mode: str  # 'darker' for black, 'brighter' for white
    dilation_kernel_size: int = 0   # 0=OFF, odd values recommended (3, 5, 7...)
    dilation_iterations: int = 1    # number of dilation iterations


class BlackWhiteHairDetector:
    """
    Hair detector with separate black/white hair detection.

    Uses SAM Automatic Mask Generation to detect individual hairs,
    then filters by brightness to classify as black or white hair.
    """

    def __init__(self):
        self._sam_model = None
        self._sam_amg: Optional['SamAutomaticMaskGenerator'] = None
        self._sam_initialized = False
        self._current_points_per_side = 64
        self.device = get_device()
        print(f"BlackWhiteHairDetector initialized with device: {self.device}")

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
            os.path.join(os.path.dirname(__file__), "..", "checkpoints", "sam_vit_h_4b8939.pth"),
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
            self._sam_model = sam
            self._sam_amg = SamAutomaticMaskGenerator(
                sam,
                points_per_side=64,
                pred_iou_thresh=0.5,
                stability_score_thresh=0.6,
                min_mask_region_area=3,
            )
            self._sam_initialized = True
            print(f"SAM initialized on {self.device}")
            return True, f"SAM initialized on {self.device}"
        except Exception as e:
            return False, f"SAM initialization error: {e}"

    def detect_with_class(
        self,
        image_rgb: np.ndarray,
        region_box: Tuple[int, int, int, int],
        hair_class: str,  # 'black' or 'white'
        params: HairClassParams,
        points_per_side: int = 64,
        use_tiling: bool = False,
        tile_size: int = 400,
        tile_overlap: int = 50,
        overlap_threshold: float = 0.5
    ) -> Tuple[List[DetectedRegion], List[np.ndarray], Dict]:
        """
        Detect hairs of a specific color class.

        Args:
            image_rgb: Full RGB image
            region_box: (x1, y1, x2, y2) ROI
            hair_class: 'black' or 'white'
            params: HairClassParams with filter settings
            points_per_side: SAM sampling density
            use_tiling: Enable tile-based processing
            tile_size: Size of each tile
            tile_overlap: Overlap between tiles

        Returns:
            Tuple of (filtered detections, all masks, stats dict)
        """
        empty_stats = {
            'total': 0, 'filtered_area_small': 0, 'filtered_area_large': 0,
            'filtered_aspect': 0, 'filtered_brightness': 0, 'passed': 0,
            'hair_class': hair_class
        }

        if not self._sam_initialized:
            success, msg = self._init_sam()
            if not success:
                print(f"SAM not available: {msg}")
                return [], [], empty_stats

        self._update_amg_params(points_per_side)

        x1, y1, x2, y2 = region_box
        h, w = image_rgb.shape[:2]

        # Crop ROI
        roi = image_rgb[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Tiled processing
        if use_tiling:
            tiles = self._calculate_tiles(roi_w, roi_h, tile_size, tile_overlap)
            print(f"Tile-based processing: {len(tiles)} tiles")

            all_results = []
            all_masks_unfiltered = []
            stats = {
                'total': 0, 'filtered_area_small': 0, 'filtered_area_large': 0,
                'filtered_no_contour': 0, 'filtered_zero_dim': 0,
                'filtered_aspect': 0, 'filtered_brightness': 0, 'passed': 0,
                'tiles': len(tiles), 'hair_class': hair_class
            }

            for idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                tile_rgb = roi[ty1:ty2, tx1:tx2]
                print(f"  Processing tile {idx+1}/{len(tiles)}")

                tile_masks = self._sam_amg.generate(tile_rgb)
                stats['total'] += len(tile_masks)

                # Store all masks
                tile_h, tile_w = tile_rgb.shape[:2]
                abs_x = x1 + tx1
                abs_y = y1 + ty1
                for mask_data in tile_masks:
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    mask = mask_data['segmentation'].astype(np.uint8) * 255
                    full_mask[abs_y:abs_y+tile_h, abs_x:abs_x+tile_w] = mask
                    all_masks_unfiltered.append(full_mask)

                tile_results, tile_stats = self._filter_tile_masks(
                    tile_masks, tile_rgb, (h, w),
                    (tx1, ty1), (x1, y1), params, hair_class
                )
                all_results.extend(tile_results)

                # Aggregate stats
                for key in ['filtered_area_small', 'filtered_area_large', 'filtered_no_contour', 'filtered_zero_dim', 'filtered_aspect', 'filtered_brightness']:
                    stats[key] += tile_stats.get(key, 0)

            # Remove duplicates
            all_results = self._remove_duplicates(all_results, overlap_threshold)
            stats['passed'] = len(all_results)
            return all_results, all_masks_unfiltered, stats

        # Non-tiled processing
        print("Running SAM Automatic Mask Generation...")
        masks = self._sam_amg.generate(roi)
        print(f"SAM AMG generated {len(masks)} masks")

        # Store all masks
        all_masks_unfiltered = []
        for mask_data in masks:
            full_mask = np.zeros((h, w), dtype=np.uint8)
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            full_mask[y1:y2, x1:x2] = mask
            all_masks_unfiltered.append(full_mask)

        # Filter masks
        results, stats = self._filter_masks(
            masks, roi, (h, w), region_box, params, hair_class
        )

        # Remove duplicates
        results = self._remove_duplicates(results, overlap_threshold)
        stats['passed'] = len(results)

        return results, all_masks_unfiltered, stats

    def _filter_masks(
        self,
        masks: List[Dict],
        roi: np.ndarray,
        full_shape: Tuple[int, int],
        region_box: Tuple[int, int, int, int],
        params: HairClassParams,
        hair_class: str
    ) -> Tuple[List[DetectedRegion], Dict]:
        """Filter SAM masks based on hair class parameters."""
        x1, y1, x2, y2 = region_box
        h, w = full_shape

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(roi_gray)

        stats = {
            'total': len(masks),
            'filtered_area_small': 0,
            'filtered_area_large': 0,
            'filtered_no_contour': 0,
            'filtered_zero_dim': 0,
            'filtered_aspect': 0,
            'filtered_brightness': 0,
            'passed': 0,
            'hair_class': hair_class,
            'mean_brightness': float(mean_brightness)
        }

        results = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            area = mask_data['area']

            # Area filter
            if area < params.min_area:
                stats['filtered_area_small'] += 1
                continue
            if area > params.max_area:
                stats['filtered_area_large'] += 1
                continue

            # Aspect ratio filter
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                stats['filtered_no_contour'] += 1
                continue

            rect = cv2.minAreaRect(contours[0])
            w_rect, h_rect = rect[1]
            # Handle zero dimensions (very thin masks) by setting minimum to 1px
            if w_rect == 0:
                w_rect = 1.0
            if h_rect == 0:
                h_rect = 1.0
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

            if aspect < params.min_aspect:
                stats['filtered_aspect'] += 1
                continue

            # Brightness filter (class-dependent)
            mask_brightness = cv2.mean(roi_gray, mask=mask)[0]

            if hair_class == 'black':
                # Black hair: mask should be DARKER than skin
                # mask_brightness < mean_brightness * threshold
                # threshold > 1.0 allows slightly brighter, < 1.0 requires darker
                if mask_brightness > mean_brightness * params.brightness_threshold:
                    stats['filtered_brightness'] += 1
                    continue
            else:  # white
                # White hair: mask should be BRIGHTER than skin
                # mask_brightness > mean_brightness * threshold
                # threshold < 1.0 allows slightly darker, > 1.0 requires brighter
                if mask_brightness < mean_brightness * params.brightness_threshold:
                    stats['filtered_brightness'] += 1
                    continue

            # Create full-size mask
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask

            # Apply dilation if enabled
            if params.dilation_kernel_size > 0:
                ksize = params.dilation_kernel_size
                if ksize % 2 == 0:
                    ksize += 1  # Ensure odd kernel size
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (ksize, ksize)
                )
                full_mask = cv2.dilate(
                    full_mask, kernel, iterations=params.dilation_iterations
                )

                # Re-check area after dilation
                new_area = cv2.countNonZero(full_mask)
                if new_area > params.max_area:
                    stats['filtered_area_large'] += 1
                    continue
                area = new_area

            # Calculate centroid (from full_mask after potential dilation)
            M = cv2.moments(full_mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

            stats['passed'] += 1

            results.append(DetectedRegion(
                mask=full_mask,
                area=area,
                centroid=(cx, cy),
                confidence=mask_data.get('stability_score', 0.5),
                source=f'sam_amg_{hair_class}',
                phrase=f"{hair_class}_hair_{i+1}"
            ))

        print(f"Filter stats ({hair_class}): {stats}")
        return results, stats

    def _filter_tile_masks(
        self,
        masks: List[Dict],
        tile_rgb: np.ndarray,
        full_shape: Tuple[int, int],
        tile_offset: Tuple[int, int],
        region_offset: Tuple[int, int],
        params: HairClassParams,
        hair_class: str
    ) -> Tuple[List[DetectedRegion], Dict]:
        """Filter masks from a single tile."""
        h, w = full_shape
        tx, ty = tile_offset
        rx, ry = region_offset
        tile_h, tile_w = tile_rgb.shape[:2]

        tile_gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(tile_gray)

        stats = {
            'filtered_area_small': 0, 'filtered_area_large': 0,
            'filtered_no_contour': 0, 'filtered_zero_dim': 0,
            'filtered_aspect': 0, 'filtered_brightness': 0
        }

        results = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            area = mask_data['area']

            if area < params.min_area:
                stats['filtered_area_small'] += 1
                continue
            if area > params.max_area:
                stats['filtered_area_large'] += 1
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                stats['filtered_no_contour'] += 1
                continue

            rect = cv2.minAreaRect(contours[0])
            w_rect, h_rect = rect[1]
            # Handle zero dimensions (very thin masks) by setting minimum to 1px
            if w_rect == 0:
                w_rect = 1.0
            if h_rect == 0:
                h_rect = 1.0
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

            if aspect < params.min_aspect:
                stats['filtered_aspect'] += 1
                continue

            mask_brightness = cv2.mean(tile_gray, mask=mask)[0]

            if hair_class == 'black':
                if mask_brightness > mean_brightness * params.brightness_threshold:
                    stats['filtered_brightness'] += 1
                    continue
            else:
                if mask_brightness < mean_brightness * params.brightness_threshold:
                    stats['filtered_brightness'] += 1
                    continue

            # Create full-size mask
            full_mask = np.zeros((h, w), dtype=np.uint8)
            abs_x = rx + tx
            abs_y = ry + ty
            full_mask[abs_y:abs_y+tile_h, abs_x:abs_x+tile_w] = mask

            # Apply dilation if enabled
            if params.dilation_kernel_size > 0:
                ksize = params.dilation_kernel_size
                if ksize % 2 == 0:
                    ksize += 1  # Ensure odd kernel size
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (ksize, ksize)
                )
                full_mask = cv2.dilate(
                    full_mask, kernel, iterations=params.dilation_iterations
                )

                # Re-check area after dilation
                new_area = cv2.countNonZero(full_mask)
                if new_area > params.max_area:
                    stats['filtered_area_large'] += 1
                    continue
                area = new_area

            # Calculate centroid (from full_mask after potential dilation)
            M = cv2.moments(full_mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx = abs_x + tile_w // 2
                cy = abs_y + tile_h // 2

            results.append(DetectedRegion(
                mask=full_mask,
                area=area,
                centroid=(cx, cy),
                confidence=mask_data.get('stability_score', 0.5),
                source=f'sam_amg_{hair_class}_tiled',
                phrase=f"{hair_class}_hair_{i+1}"
            ))

        return results, stats

    def _calculate_tiles(
        self, region_width: int, region_height: int,
        tile_size: int = 400, overlap: int = 50
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate tile positions."""
        tiles = []

        if region_width <= tile_size and region_height <= tile_size:
            return [(0, 0, region_width, region_height)]

        step = max(tile_size - overlap, 1)

        y = 0
        while y < region_height:
            x = 0
            while x < region_width:
                x1, y1 = x, y
                x2 = min(x + tile_size, region_width)
                y2 = min(y + tile_size, region_height)

                if x2 - x1 >= overlap and y2 - y1 >= overlap:
                    tiles.append((x1, y1, x2, y2))

                x += step
                if x2 >= region_width:
                    break

            y += step
            if y2 >= region_height:
                break

        return tiles

    def _remove_duplicates(
        self, detections: List[DetectedRegion], overlap_threshold: float = 0.5
    ) -> List[DetectedRegion]:
        """Remove duplicate detections."""
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        unique = []

        for det in sorted_dets:
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

    def detect_with_class_and_mask(
        self,
        image_rgb: np.ndarray,
        region_mask: np.ndarray,
        hair_class: str,
        params: HairClassParams,
        points_per_side: int = 64,
        use_tiling: bool = False,
        tile_size: int = 400,
        tile_overlap: int = 50,
        overlap_threshold: float = 0.5
    ) -> Tuple[List[DetectedRegion], List[np.ndarray], Dict]:
        """
        Detect hairs within a freeform mask region.

        Args:
            image_rgb: Full RGB image
            region_mask: Binary mask (255 = detection region)
            hair_class: 'black' or 'white'
            params: HairClassParams with filter settings
            points_per_side: SAM sampling density
            use_tiling: Enable tile-based processing
            tile_size: Size of each tile
            tile_overlap: Overlap between tiles

        Returns:
            Tuple of (filtered detections, all masks, stats dict)
        """
        # Ensure mask is binary
        mask_binary = (region_mask > 128).astype(np.uint8) * 255

        # Get bounding box from mask
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            empty_stats = {
                'total': 0, 'filtered_area_small': 0, 'filtered_area_large': 0,
                'filtered_aspect': 0, 'filtered_brightness': 0, 'passed': 0,
                'filtered_outside_mask': 0, 'hair_class': hair_class,
                'selection_mode': 'freeform'
            }
            return [], [], empty_stats

        # Compute bounding box that encompasses all contours
        x_min, y_min = image_rgb.shape[1], image_rgb.shape[0]
        x_max, y_max = 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        region_box = (x_min, y_min, x_max, y_max)

        # Run detection with bounding box to get all_masks (unfiltered)
        # Use overlap_threshold=1.0 here to skip duplicate removal (we'll do it later)
        _, all_masks_raw, base_stats = self.detect_with_class(
            image_rgb, region_box, hair_class, params,
            points_per_side=points_per_side,
            use_tiling=use_tiling,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            overlap_threshold=1.0  # Skip duplicate removal here
        )

        # Re-filter using freeform mask for accurate mean_brightness calculation
        # Calculate mean brightness ONLY within the freeform mask region
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        mask_pixels = image_gray[mask_binary > 0]
        if len(mask_pixels) > 0:
            mean_brightness = np.mean(mask_pixels)
        else:
            mean_brightness = np.mean(image_gray[y_min:y_max, x_min:x_max])

        h, w = image_rgb.shape[:2]
        filtered_detections = []
        filtered_outside_mask = 0
        filtered_area_small = 0
        filtered_area_large = 0
        filtered_aspect = 0
        filtered_brightness = 0

        for full_mask in all_masks_raw:
            # Clip mask to freeform region
            masked = cv2.bitwise_and(full_mask, mask_binary)
            area = cv2.countNonZero(masked)

            if area == 0:
                filtered_outside_mask += 1
                continue

            # Area filter
            if area < params.min_area:
                filtered_area_small += 1
                continue
            if area > params.max_area:
                filtered_area_large += 1
                continue

            # Aspect ratio filter
            contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            rect = cv2.minAreaRect(contours[0])
            w_rect, h_rect = rect[1]
            # Handle zero dimensions (very thin masks) by setting minimum to 1px
            if w_rect == 0:
                w_rect = 1.0
            if h_rect == 0:
                h_rect = 1.0
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

            if aspect < params.min_aspect:
                filtered_aspect += 1
                continue

            # Brightness filter using freeform-aware mean_brightness
            mask_brightness = cv2.mean(image_gray, mask=masked)[0]

            if hair_class == 'black':
                if mask_brightness > mean_brightness * params.brightness_threshold:
                    filtered_brightness += 1
                    continue
            else:  # white
                if mask_brightness < mean_brightness * params.brightness_threshold:
                    filtered_brightness += 1
                    continue

            # Apply dilation if enabled
            final_mask = masked
            if params.dilation_kernel_size > 0:
                ksize = params.dilation_kernel_size
                if ksize % 2 == 0:
                    ksize += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                final_mask = cv2.dilate(masked, kernel, iterations=params.dilation_iterations)
                # Re-clip to freeform mask after dilation
                final_mask = cv2.bitwise_and(final_mask, mask_binary)
                area = cv2.countNonZero(final_mask)
                if area > params.max_area:
                    filtered_area_large += 1
                    continue

            # Calculate centroid
            M = cv2.moments(final_mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2

            filtered_detections.append(DetectedRegion(
                mask=final_mask,
                area=area,
                centroid=(cx, cy),
                confidence=0.5,
                source=f'sam_amg_{hair_class}_freeform',
                phrase=f"{hair_class}_hair"
            ))

        # Remove duplicates
        filtered_detections = self._remove_duplicates(filtered_detections, overlap_threshold)

        stats = {
            'total': len(all_masks_raw),
            'filtered_area_small': filtered_area_small,
            'filtered_area_large': filtered_area_large,
            'filtered_aspect': filtered_aspect,
            'filtered_brightness': filtered_brightness,
            'filtered_outside_mask': filtered_outside_mask,
            'passed': len(filtered_detections),
            'hair_class': hair_class,
            'selection_mode': 'freeform',
            'mean_brightness': float(mean_brightness)
        }

        return filtered_detections, all_masks_raw, stats
