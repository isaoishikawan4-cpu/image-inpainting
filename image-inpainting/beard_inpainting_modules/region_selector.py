"""Region selection from Gradio ImageEditor.

This module extracts user-drawn regions from Gradio ImageEditor component,
supporting both rectangular and freeform selection modes.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from enum import Enum


class SelectionShape(Enum):
    """選択形状モード"""
    RECTANGLE = "rectangle"   # 矩形（バウンディングボックス）
    FREEFORM = "freeform"     # 自由形状（線で囲んだ領域）


class RegionSelector:
    """Extracts user-drawn regions from Gradio ImageEditor."""

    @staticmethod
    def extract_rectangle(editor_data: dict) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract bounding rectangle from ImageEditor brush strokes.

        Args:
            editor_data: Dictionary from gr.ImageEditor containing:
                - 'layers': List of layer arrays
                - 'composite': Composite image with alpha

        Returns:
            (x1, y1, x2, y2) bounding box or None if no region drawn
        """
        if editor_data is None:
            return None

        try:
            mask = RegionSelector._extract_mask_from_layers(editor_data)

            if mask is None or np.max(mask) == 0:
                return None

            return RegionSelector._get_bounding_box_from_mask(mask)

        except Exception as e:
            print(f"Error extracting rectangle: {e}")
            return None

    @staticmethod
    def _extract_mask_from_layers(editor_data: dict) -> Optional[np.ndarray]:
        """
        Extract binary mask from editor layers.

        Args:
            editor_data: Dictionary from gr.ImageEditor

        Returns:
            Binary mask array or None
        """
        if not isinstance(editor_data, dict):
            return None

        # Try to extract from layers
        if 'layers' in editor_data and len(editor_data['layers']) > 0:
            layer = editor_data['layers'][0]
            if isinstance(layer, np.ndarray) and len(layer.shape) == 3:
                # Convert to grayscale and threshold
                gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                mask = (gray > 128).astype(np.uint8) * 255
                return mask

        # Try to extract from composite (alpha channel)
        if 'composite' in editor_data:
            composite = editor_data['composite']
            if isinstance(composite, np.ndarray) and len(composite.shape) == 3:
                if composite.shape[2] == 4:
                    # Use alpha channel as mask
                    mask = composite[:, :, 3]
                    return mask

        return None

    @staticmethod
    def _get_bounding_box_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate bounding box from binary mask using contours.

        Args:
            mask: Binary mask array

        Returns:
            (x1, y1, x2, y2) bounding box or None
        """
        # Ensure binary mask
        mask_binary = (mask > 128).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            mask_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Calculate combined bounding box
        x_min, y_min = mask.shape[1], mask.shape[0]
        x_max, y_max = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        if x_max <= x_min or y_max <= y_min:
            return None

        return (x_min, y_min, x_max, y_max)

    @staticmethod
    def extract_freeform_mask(editor_data: dict) -> Optional[np.ndarray]:
        """
        Extract freeform mask from ImageEditor brush strokes.

        線で囲んだ領域を閉じた形状として認識し、その内部を塗りつぶしたマスクを返す。

        Args:
            editor_data: Dictionary from gr.ImageEditor containing:
                - 'layers': List of layer arrays
                - 'composite': Composite image with alpha

        Returns:
            Binary mask (255 = inside region) or None if no region drawn
        """
        if editor_data is None:
            return None

        try:
            mask = RegionSelector._extract_mask_from_layers(editor_data)

            if mask is None or np.max(mask) == 0:
                return None

            # 線で囲んだ領域を閉じた形状として処理
            filled_mask = RegionSelector._fill_enclosed_region(mask)

            return filled_mask

        except Exception as e:
            print(f"Error extracting freeform mask: {e}")
            return None

    @staticmethod
    def _fill_enclosed_region(mask: np.ndarray) -> np.ndarray:
        """
        線で囲まれた領域を塗りつぶす。

        Args:
            mask: 線が描かれたバイナリマスク

        Returns:
            内部が塗りつぶされたマスク
        """
        # バイナリ化
        mask_binary = (mask > 128).astype(np.uint8) * 255

        # 線を少し太くして隙間を埋める
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)

        # 輪郭を検出
        contours, _ = cv2.findContours(
            mask_dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return mask_binary

        # 塗りつぶし用の新しいマスクを作成
        filled_mask = np.zeros_like(mask_binary)

        # すべての輪郭を塗りつぶす
        cv2.drawContours(filled_mask, contours, -1, 255, -1)  # -1 = 塗りつぶし

        return filled_mask

    @staticmethod
    def validate_region(
        region: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
        min_size: int = 10
    ) -> bool:
        """
        Validate that a region is valid and within image bounds.

        Args:
            region: (x1, y1, x2, y2) bounding box
            image_shape: (height, width) of the image
            min_size: Minimum dimension size

        Returns:
            True if region is valid
        """
        if region is None:
            return False

        x1, y1, x2, y2 = region
        h, w = image_shape

        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False

        # Check minimum size
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return False

        return True

    @staticmethod
    def validate_mask(mask: np.ndarray, min_pixels: int = 100) -> bool:
        """
        Validate that a freeform mask has sufficient coverage.

        Args:
            mask: Binary mask array
            min_pixels: Minimum number of white pixels

        Returns:
            True if mask is valid
        """
        if mask is None:
            return False

        return np.sum(mask > 0) >= min_pixels
