"""Image format conversion and I/O utilities.

This module provides utilities for converting between different image formats
(numpy arrays, PIL Images) used throughout the beard inpainting pipeline.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Optional, Union


class ImageHandler:
    """Handles image format conversions between numpy/PIL/Gradio formats."""

    @staticmethod
    def ensure_rgb(image: np.ndarray) -> np.ndarray:
        """
        Convert any image format to RGB numpy array.

        Args:
            image: Input image (grayscale, RGB, or RGBA)

        Returns:
            RGB numpy array (H, W, 3)
        """
        if image is None:
            raise ValueError("Input image is None")

        if len(image.shape) == 2:
            # Grayscale -> RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA -> RGB
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            # Already RGB
            return image.copy()
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    @staticmethod
    def numpy_to_pil(image: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image.

        Args:
            image: RGB numpy array (H, W, 3) or grayscale (H, W)

        Returns:
            PIL Image in RGB or L mode
        """
        if image is None:
            raise ValueError("Input image is None")

        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        if len(image.shape) == 2:
            return Image.fromarray(image, mode='L')
        elif len(image.shape) == 3 and image.shape[2] == 3:
            return Image.fromarray(image, mode='RGB')
        elif len(image.shape) == 3 and image.shape[2] == 4:
            return Image.fromarray(image, mode='RGBA')
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    @staticmethod
    def pil_to_numpy(image: Image.Image) -> np.ndarray:
        """
        Convert PIL Image to numpy array.

        Args:
            image: PIL Image

        Returns:
            numpy array (RGB if color, grayscale if L mode)
        """
        if image is None:
            raise ValueError("Input image is None")

        return np.array(image)

    @staticmethod
    def ensure_grayscale_mask(mask: np.ndarray) -> np.ndarray:
        """
        Convert mask to single-channel grayscale.

        Args:
            mask: Input mask (possibly RGB or RGBA)

        Returns:
            Grayscale mask (H, W)
        """
        if mask is None:
            raise ValueError("Input mask is None")

        if len(mask.shape) == 2:
            return mask
        elif len(mask.shape) == 3:
            if mask.shape[2] == 3:
                return cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            elif mask.shape[2] == 4:
                return cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)

        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    @staticmethod
    def convert_to_binary_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
        """
        Convert mask to binary format (0 or 255).

        Args:
            mask: Input mask array
            threshold: Binarization threshold

        Returns:
            Binary mask (values are 0 or 255)
        """
        if mask is None:
            raise ValueError("Input mask is None")

        # Ensure grayscale
        if len(mask.shape) == 3:
            mask = ImageHandler.ensure_grayscale_mask(mask)

        # Binarize
        binary_mask = np.where(mask > threshold, 255, 0).astype(np.uint8)
        return binary_mask

    @staticmethod
    def extract_image_from_editor(editor_data: dict) -> Optional[np.ndarray]:
        """
        Extract RGB numpy image from Gradio ImageEditor data.

        Handles:
        - Extracting from 'background' key (preferred) or 'composite' key
        - PIL.Image to numpy conversion
        - Grayscale to RGB conversion
        - RGBA to RGB conversion

        Args:
            editor_data: Dictionary from gr.ImageEditor

        Returns:
            RGB numpy array (H, W, 3) or None if no valid image found
        """
        if editor_data is None:
            return None

        if 'background' in editor_data:
            image = editor_data['background']
        elif 'composite' in editor_data:
            image = editor_data['composite']
        else:
            return None

        if image is None:
            return None

        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image
