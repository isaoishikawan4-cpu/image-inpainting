"""LaMa inpainting integration for beard removal.

This module provides a wrapper around the existing core/inpainting.py
with Gradio-specific convenience methods.

Also includes OpenCV inpainting option (TELEA/Navier-Stokes algorithms).
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import cv2


class InpaintingMethod(Enum):
    """インペインティング手法の選択"""
    LAMA = "lama"                      # Simple LaMa (deep learning)
    OPENCV_TELEA = "opencv_telea"      # OpenCV Telea method
    OPENCV_NS = "opencv_ns"            # OpenCV Navier-Stokes method

# Check if core inpainting is available
LAMA_AVAILABLE = False
try:
    from core.inpainting import InpaintingEngine, BeardThinningProcessor, SIMPLE_LAMA_AVAILABLE
    from core.image_utils import (
        resize_image_if_needed,
        convert_to_binary_mask,
        numpy_to_pil
    )
    # Only mark as available if simple-lama-inpainting is actually installed
    LAMA_AVAILABLE = SIMPLE_LAMA_AVAILABLE
except ImportError:
    pass


class LamaInpainter:
    """LaMa Inpainting wrapper with lazy initialization."""

    def __init__(self):
        self._engine = None
        self._processor = None
        self._is_available: Optional[bool] = None

    @property
    def is_available(self) -> bool:
        """Check if LaMa is available (lazy check)."""
        if self._is_available is None:
            self._is_available = LAMA_AVAILABLE
        return self._is_available

    def _ensure_initialized(self) -> bool:
        """Initialize LaMa engine if not already done."""
        if not self.is_available:
            return False

        if self._processor is not None:
            return True

        try:
            print("Initializing LaMa engine...")
            self._processor = BeardThinningProcessor()
            print("LaMa engine: Initialized")
            return True
        except Exception as e:
            print(f"LaMa engine initialization error: {e}")
            self._is_available = False
            return False

    def inpaint_single(
        self,
        image: Image.Image,
        mask: np.ndarray
    ) -> Image.Image:
        """
        Perform single inpainting operation.

        Args:
            image: PIL Image (RGB)
            mask: Binary mask (H, W)

        Returns:
            Inpainted PIL Image
        """
        if not self._ensure_initialized():
            raise RuntimeError("LaMa is not available")

        # Ensure mask is binary
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        binary_mask = convert_to_binary_mask(mask)

        # Use the processor's engine for single inpainting
        mask_pil = numpy_to_pil(binary_mask)
        return self._processor.engine.inpaint(image, mask_pil)

    def process_thinning_levels(
        self,
        image: Image.Image,
        mask: np.ndarray,
        levels: List[int],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[List[Tuple[Image.Image, str]], str]:
        """
        Process multiple thinning levels.

        Args:
            image: Original PIL Image
            mask: Binary mask of beard region
            levels: List of thinning percentages (e.g., [30, 50, 70, 100])
            progress_callback: Optional progress reporter (current, total, description)

        Returns:
            Tuple of (gallery_items, status_message)
            where gallery_items is [(image, caption), ...]
        """
        if not self._ensure_initialized():
            return [], "LaMa is not available. Please install simple-lama-inpainting."

        if image is None:
            return [], "No image provided"

        if mask is None or np.max(mask) == 0:
            return [], "No mask provided"

        if not levels:
            return [], "No thinning levels selected"

        try:
            # Resize image if needed
            image = resize_image_if_needed(image)

            # Ensure mask is binary
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            binary_mask = convert_to_binary_mask(mask)

            # Process thinning
            def internal_progress(current, total, level):
                if progress_callback:
                    progress_callback(current, total, f"Processing: {level}")

            results, messages = self._processor.process_thinning(
                image,
                binary_mask,
                levels,
                progress_callback=internal_progress
            )

            if not results:
                return [], "Processing failed: " + "\n".join(messages)

            # Convert to gallery format
            gallery = [(image, "Original (0%)")]
            for level in sorted(results.keys()):
                caption = f"{level}% thinning" if level < 100 else "Complete removal (100%)"
                gallery.append((results[level], caption))

            status = f"Done! Generated {len(results)} thinning levels"
            return gallery, status

        except Exception as e:
            return [], f"Error: {str(e)}"

    def prepare_for_gradio(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """
        Prepare image and mask for inpainting from Gradio inputs.

        Args:
            image_rgb: RGB numpy array from Tab 1
            mask: Mask from region manager

        Returns:
            (PIL Image, processed binary mask)
        """
        if image_rgb is None:
            return None, None

        # Convert numpy to PIL
        if len(image_rgb.shape) == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        elif image_rgb.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

        image_pil = Image.fromarray(image_rgb)

        return image_pil, mask

    def get_status(self) -> str:
        """Get status message about LaMa availability."""
        if not LAMA_AVAILABLE:
            return "LaMa not installed. Run: pip install simple-lama-inpainting"
        if self._processor is not None:
            return "LaMa: Ready"
        return "LaMa: Available (not initialized)"


class OpenCVInpainter:
    """OpenCV-based inpainting using TELEA or Navier-Stokes algorithms."""

    def __init__(self, method: InpaintingMethod = InpaintingMethod.OPENCV_TELEA):
        """
        Initialize OpenCV inpainter.

        Args:
            method: InpaintingMethod.OPENCV_TELEA or OPENCV_NS
        """
        self._method = method
        self._inpaint_radius = 3  # Default inpainting radius

    @property
    def is_available(self) -> bool:
        """OpenCV is always available."""
        return True

    def set_radius(self, radius: int) -> None:
        """Set inpainting radius."""
        self._inpaint_radius = max(1, min(radius, 20))

    def set_method(self, method: InpaintingMethod) -> None:
        """Set inpainting method."""
        self._method = method

    def _get_cv_method(self) -> int:
        """Get OpenCV inpainting flag."""
        if self._method == InpaintingMethod.OPENCV_NS:
            return cv2.INPAINT_NS
        return cv2.INPAINT_TELEA

    def inpaint_single(
        self,
        image: Image.Image,
        mask: np.ndarray,
        radius: Optional[int] = None
    ) -> Image.Image:
        """
        Perform single inpainting operation.

        Args:
            image: PIL Image (RGB)
            mask: Binary mask (H, W) - white areas will be inpainted
            radius: Optional inpainting radius (overrides default)

        Returns:
            Inpainted PIL Image
        """
        # Convert PIL to OpenCV BGR
        image_rgb = np.array(image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Ensure mask is proper format
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # Ensure mask is uint8 binary
        mask_binary = (mask > 127).astype(np.uint8) * 255

        # Apply inpainting
        inpaint_radius = radius if radius is not None else self._inpaint_radius
        result_bgr = cv2.inpaint(
            image_bgr,
            mask_binary,
            inpaint_radius,
            self._get_cv_method()
        )

        # Convert back to RGB PIL
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

    def process_thinning_levels(
        self,
        image: Image.Image,
        mask: np.ndarray,
        levels: List[int],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[List[Tuple[Image.Image, str]], str]:
        """
        Process multiple thinning levels using mask erosion.

        For OpenCV inpainting, thinning is achieved by eroding the mask
        to progressively reduce the inpainted area.

        Args:
            image: Original PIL Image
            mask: Binary mask of beard region
            levels: List of thinning percentages (e.g., [30, 50, 70, 100])
            progress_callback: Optional progress reporter (current, total, description)

        Returns:
            Tuple of (gallery_items, status_message)
            where gallery_items is [(image, caption), ...]
        """
        if image is None:
            return [], "画像が指定されていません"

        if mask is None or np.max(mask) == 0:
            return [], "マスクが指定されていません"

        if not levels:
            return [], "薄め具合が選択されていません"

        try:
            # Ensure mask is grayscale
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            # Binary mask
            mask_binary = (mask > 127).astype(np.uint8) * 255

            # Gallery starts with original
            gallery = [(image, "Original (0%)")]
            method_name = "Telea" if self._method == InpaintingMethod.OPENCV_TELEA else "Navier-Stokes"

            sorted_levels = sorted(levels)
            total = len(sorted_levels)

            for i, level in enumerate(sorted_levels):
                if progress_callback:
                    progress_callback(i, total, f"OpenCV ({method_name}): {level}%")

                if level == 100:
                    # Full mask - no erosion
                    current_mask = mask_binary
                else:
                    # Erode mask to achieve partial removal
                    # Higher erosion = less area = lower level
                    # For 70%, we want to remove 70% of the mask area
                    erosion_factor = 1.0 - (level / 100.0)
                    kernel_size = max(3, int(erosion_factor * 30))

                    if kernel_size > 3:
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (kernel_size, kernel_size)
                        )
                        current_mask = cv2.erode(mask_binary, kernel, iterations=1)
                    else:
                        current_mask = mask_binary.copy()

                # Check if mask still has content
                if np.max(current_mask) == 0:
                    continue

                # Inpaint
                result = self.inpaint_single(image, current_mask)

                caption = f"{level}% thinning (OpenCV {method_name})" if level < 100 else f"Complete removal (OpenCV {method_name})"
                gallery.append((result, caption))

            if progress_callback:
                progress_callback(total, total, "完了")

            return gallery, f"完了! OpenCV {method_name}で{len(gallery)-1}段階の処理を生成"

        except Exception as e:
            return [], f"エラー: {str(e)}"

    def get_status(self) -> str:
        """Get status message about OpenCV inpainting."""
        method_name = "Telea" if self._method == InpaintingMethod.OPENCV_TELEA else "Navier-Stokes"
        return f"OpenCV Inpainting ({method_name}): Ready"
