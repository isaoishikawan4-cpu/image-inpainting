"""LaMa inpainting integration for beard removal.

This module provides a wrapper around the existing core/inpainting.py
with Gradio-specific convenience methods.
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Callable
import cv2

# Check if core inpainting is available
LAMA_AVAILABLE = False
try:
    from core.inpainting import InpaintingEngine, BeardThinningProcessor
    from core.image_utils import (
        resize_image_if_needed,
        convert_to_binary_mask,
        numpy_to_pil
    )
    LAMA_AVAILABLE = True
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
