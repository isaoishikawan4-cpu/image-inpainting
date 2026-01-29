"""Main beard removal pipeline orchestrating all modules.

This module provides the BeardRemovalPipeline class that coordinates
the complete workflow: detection -> selection -> inpainting -> color correction.
"""

import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
import cv2

from .image_handler import ImageHandler
from .region_selector import RegionSelector, SelectionShape
from .beard_detector import BeardDetector, DetectionBackend
from .highlighter import BeardRegionManager, SelectionMode
from .inpainter import LamaInpainter, OpenCVInpainter, InpaintingMethod
from .color_corrector import SkinColorCorrector, CorrectionMode, MaskType


class BeardRemovalPipeline:
    """
    Orchestrates the complete beard detection and removal workflow.

    This class maintains state across Gradio callbacks and coordinates
    the flow between detection, selection, inpainting, and color correction.
    """

    def __init__(self):
        self._detector = BeardDetector()
        self._region_manager = BeardRegionManager()
        self._inpainter = LamaInpainter()
        self._opencv_inpainter = OpenCVInpainter()
        self._color_corrector = SkinColorCorrector()
        self._current_image: Optional[np.ndarray] = None
        self._last_inpaint_result: Optional[np.ndarray] = None

    @property
    def current_image(self) -> Optional[np.ndarray]:
        """Get the current working image."""
        return self._current_image

    @property
    def region_manager(self) -> BeardRegionManager:
        """Get the region manager for direct access."""
        return self._region_manager

    def process_detection(
        self,
        image: np.ndarray,
        editor_data: dict,
        use_grounded_sam: bool,
        # Grounded SAM parameters
        text_prompt: str = "beard. facial hair. stubble.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        # Rule-based parameters
        threshold_value: int = 80,
        min_area: int = 10,
        max_area: int = 5000,
        # Selection shape
        selection_shape: str = "矩形"
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Process beard detection (Tab 1 main function).

        Args:
            image: Input image from Gradio
            editor_data: Rectangle/freeform selection from ImageEditor
            use_grounded_sam: Use Grounded SAM (True) or rule-based (False)
            text_prompt: Detection text prompt (for Grounded SAM)
            box_threshold: Box detection threshold (for Grounded SAM)
            text_threshold: Text matching threshold (for Grounded SAM)
            threshold_value: Binarization threshold (for rule-based)
            min_area: Minimum region area
            max_area: Maximum region area
            selection_shape: "矩形" or "自由形状"

        Returns:
            (display_image, mask, status_message)
        """
        if image is None:
            return None, None, "Please upload an image"

        # Store current image
        self._current_image = image.copy()
        self._region_manager.clear()

        # Ensure RGB
        image_rgb = ImageHandler.ensure_rgb(image)

        # 選択形状に応じて処理を分岐
        use_freeform = selection_shape == "自由形状"

        if use_freeform:
            # 自由形状モード: マスクを抽出
            freeform_mask = RegionSelector.extract_freeform_mask(editor_data)
            if freeform_mask is None or not RegionSelector.validate_mask(freeform_mask, min_pixels=100):
                return image_rgb, None, "線で領域を囲んでください（閉じた形状になるように描画）"

            # バウンディングボックスも取得（表示用）
            rect = RegionSelector.extract_rectangle(editor_data)
            x1, y1, x2, y2 = rect if rect else (0, 0, image_rgb.shape[1], image_rgb.shape[0])

            try:
                if use_grounded_sam:
                    # Grounded SAM with freeform mask
                    backend = self._detector.get_backend(DetectionBackend.GROUNDED_SAM)
                    if not backend.is_available():
                        if not backend.initialize():
                            return image_rgb, None, "Grounded SAM not available. Check checkpoints."

                    print(f"Grounded SAM detection (freeform)...")
                    regions = self._detector.detect_with_mask(
                        image_rgb, freeform_mask,
                        backend=DetectionBackend.GROUNDED_SAM,
                        text_prompt=text_prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        min_area=min_area,
                        max_area=max_area
                    )
                    mode_name = "Grounded SAM（自由形状）"
                else:
                    # Rule-based with freeform mask
                    print(f"Rule-based detection (freeform)... threshold={threshold_value}")
                    regions = self._detector.detect_with_mask(
                        image_rgb, freeform_mask,
                        backend=DetectionBackend.RULE_BASED,
                        threshold_value=threshold_value,
                        min_area=min_area,
                        max_area=max_area
                    )
                    mode_name = "ルールベース（自由形状）"

                self._region_manager.add_regions(regions)

                # Create colored display
                display = self._region_manager.create_colored_display(image_rgb, highlight_active=False)

                # 自由形状の輪郭を描画（青色）
                contours, _ = cv2.findContours(
                    (freeform_mask > 128).astype(np.uint8) * 255,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(display, contours, -1, (0, 150, 255), 2)

                # Create combined mask
                mask = self._region_manager.get_all_masks_combined(image_rgb.shape[:2])

                status = f"検出完了 [{mode_name}]: {len(regions)} 本の髭を検出"
                return display, mask, status

            except Exception as e:
                return image_rgb, None, f"Detection error: {str(e)}"
        else:
            # 矩形モード（従来の処理）
            rect = RegionSelector.extract_rectangle(editor_data)
            if rect is None:
                return image_rgb, None, "矩形で範囲を囲んでください（白色ブラシで塗りつぶし）"

            x1, y1, x2, y2 = rect

            try:
                if use_grounded_sam:
                    # Grounded SAM detection
                    backend = self._detector.get_backend(DetectionBackend.GROUNDED_SAM)
                    if not backend.is_available():
                        if not backend.initialize():
                            return image_rgb, None, "Grounded SAM not available. Check checkpoints."

                    print(f"Grounded SAM detection... region=({x1}, {y1}, {x2}, {y2})")
                    regions = backend.detect(
                        image_rgb, rect,
                        text_prompt=text_prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        min_area=min_area,
                        max_area=max_area
                    )
                    mode_name = "Grounded SAM"
                else:
                    # Rule-based detection
                    print(f"Rule-based detection... region=({x1}, {y1}, {x2}, {y2}), threshold={threshold_value}")
                    regions = self._detector.detect(
                        image_rgb, rect,
                        backend=DetectionBackend.RULE_BASED,
                        threshold_value=threshold_value,
                        min_area=min_area,
                        max_area=max_area
                    )
                    mode_name = "ルールベース"

                self._region_manager.add_regions(regions)

                # Create colored display
                display = self._region_manager.create_colored_display(image_rgb, highlight_active=False)

                # Create combined mask
                mask = self._region_manager.get_all_masks_combined(image_rgb.shape[:2])

                # Draw detection region rectangle
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

                status = f"検出完了 [{mode_name}]: {len(regions)} 本の髭を検出 | 領域: ({x1},{y1})-({x2},{y2})"
                return display, mask, status

            except Exception as e:
                return image_rgb, None, f"Detection error: {str(e)}"

    def update_selection(
        self,
        removal_percentage: int,
        selection_mode: str,
        new_seed: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Update region selection (Tab 1 selection update).

        Args:
            removal_percentage: Percentage to remove (0-100)
            selection_mode: "ランダム", "面積大", "面積小", "信頼度順"
            new_seed: Generate new random seed

        Returns:
            (display_image, selection_mask, status_message)
        """
        if self._current_image is None or not self._region_manager.regions:
            return None, None, "Please run detection first"

        # Map selection mode
        mode = self._map_selection_mode(selection_mode)

        # Update selection
        result = self._region_manager.update_selection(removal_percentage, mode, new_seed)

        # Ensure RGB
        image_rgb = ImageHandler.ensure_rgb(self._current_image)

        # Create colored display (with active regions highlighted)
        display = self._region_manager.create_colored_display(image_rgb, highlight_active=True)

        # Get dilated mask for selected regions
        combined_mask = self._region_manager.get_dilated_mask(image_rgb.shape[:2])

        status = f"Selection: {result.selected_count}/{result.total_count} ({removal_percentage}%) | Mode: {selection_mode} | Seed: {result.seed}"

        return display, combined_mask, status

    def transfer_for_inpainting(
        self,
        mask: np.ndarray
    ) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """
        Transfer current image and mask for inpainting (Tab 1 -> Tab 2).

        Args:
            mask: Selection mask from Tab 1

        Returns:
            (PIL Image, mask) for Tab 2 inputs
        """
        if self._current_image is None:
            return None, None

        return self._inpainter.prepare_for_gradio(self._current_image, mask)

    def process_inpainting(
        self,
        image: Image.Image,
        mask: np.ndarray,
        thinning_levels: List[int],
        progress=None,
        method: str = "lama",
        opencv_radius: int = 3
    ) -> Tuple[List[Tuple[Image.Image, str]], str]:
        """
        Process inpainting (Tab 2 main function).

        Args:
            image: Input PIL image
            mask: Binary mask
            thinning_levels: List of levels [30, 50, 70, 100]
            progress: Gradio progress tracker
            method: "lama", "opencv_telea", or "opencv_ns"
            opencv_radius: Inpainting radius for OpenCV methods (1-20)

        Returns:
            (gallery_items, status_message)
        """
        # Create progress callback
        def progress_callback(current, total, desc):
            if progress is not None:
                try:
                    progress(current / total, desc=desc)
                except:
                    pass

        # Select inpainting method
        if method == "opencv_telea":
            self._opencv_inpainter.set_method(InpaintingMethod.OPENCV_TELEA)
            self._opencv_inpainter.set_radius(opencv_radius)
            return self._opencv_inpainter.process_thinning_levels(
                image, mask, thinning_levels, progress_callback
            )
        elif method == "opencv_ns":
            self._opencv_inpainter.set_method(InpaintingMethod.OPENCV_NS)
            self._opencv_inpainter.set_radius(opencv_radius)
            return self._opencv_inpainter.process_thinning_levels(
                image, mask, thinning_levels, progress_callback
            )
        else:
            # Default: LaMa
            return self._inpainter.process_thinning_levels(
                image, mask, thinning_levels, progress_callback
            )

    def reset(self) -> None:
        """Reset all state for new image."""
        self._current_image = None
        self._region_manager.clear()

    def _map_selection_mode(self, mode_str: str) -> SelectionMode:
        """Map Japanese mode string to SelectionMode enum."""
        mode_map = {
            "ランダム": SelectionMode.RANDOM,
            "面積大": SelectionMode.AREA_LARGE,
            "面積小": SelectionMode.AREA_SMALL,
            "信頼度順": SelectionMode.CONFIDENCE,
            # English fallbacks
            "random": SelectionMode.RANDOM,
            "area_large": SelectionMode.AREA_LARGE,
            "area_small": SelectionMode.AREA_SMALL,
            "confidence": SelectionMode.CONFIDENCE,
        }
        return mode_map.get(mode_str, SelectionMode.RANDOM)

    def is_grounded_sam_available(self) -> bool:
        """Check if Grounded SAM is available."""
        return self._detector.is_grounded_sam_available()

    def is_lama_available(self) -> bool:
        """Check if LaMa is available."""
        return self._inpainter.is_available

    # =========================================================================
    # Tab 3: Color Correction
    # =========================================================================

    def process_color_correction(
        self,
        image: np.ndarray,
        target_editor_data: dict,
        source_editor_data: Optional[dict],
        correction_mode: str,
        strength: float = 0.8,
        edge_blur: int = 15,
        a_adjustment_factor: float = 0.3,
        b_adjustment_factor: float = 0.6,
        l_adjustment_factor: float = 0.5,
        use_scattered_mode: bool = False,
        use_direct_fill: bool = False
    ) -> Tuple[np.ndarray, str]:
        """
        Process color correction (Tab 3 main function).

        Args:
            image: Input image (numpy array, RGB or BGR)
            target_editor_data: Target region (blue beard area) from ImageEditor
            source_editor_data: Source region (cheek skin) from ImageEditor (optional)
            correction_mode: "青み除去（推奨）", "色味転送", "自動補正"
            strength: Correction strength (0.0-1.0)
            edge_blur: Edge blur size for blending
            a_adjustment_factor: a*（赤-緑軸）の調整係数 (0.0-1.0) デフォルト0.3
            b_adjustment_factor: b*（青-黄軸）の調整係数 (0.0-1.0) デフォルト0.6
            l_adjustment_factor: L（明度）の調整係数 (0.0-1.0) デフォルト0.5
            use_scattered_mode: Tab 1の散らばった髭領域を使用する場合True

        Returns:
            (result_image, status_message)
        """
        if image is None:
            return None, "画像をアップロードしてください"

        # Extract target mask
        target_mask = self._extract_mask_from_editor(target_editor_data)
        if target_mask is None or not np.any(target_mask > 0):
            return image, "対象領域（青髭部分）を塗ってください"

        # Map correction mode
        mode = self._map_correction_mode(correction_mode)

        # Determine mask type
        mask_type = MaskType.SCATTERED if use_scattered_mode else MaskType.MANUAL

        # For color transfer mode, extract source mask
        source_mask = None
        if mode == CorrectionMode.COLOR_TRANSFER:
            source_mask = self._extract_mask_from_editor(source_editor_data)
            if source_mask is None or not np.any(source_mask > 0):
                return image, "色味転送モードではスポイト領域（頬など）を塗ってください"

        try:
            # Convert to BGR if needed (assuming input is RGB from Gradio)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Apply color correction
            result_bgr = self._color_corrector.correct_color(
                image_bgr,
                target_mask,
                source_mask,
                strength=strength,
                edge_blur=edge_blur,
                mode=mode,
                a_adjustment_factor=a_adjustment_factor,
                b_adjustment_factor=b_adjustment_factor,
                l_adjustment_factor=l_adjustment_factor,
                mask_type=mask_type,
                use_direct_fill=use_direct_fill
            )

            # Convert back to RGB
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

            mode_name = {
                CorrectionMode.BLUE_REMOVAL: "青み除去",
                CorrectionMode.COLOR_TRANSFER: "色味転送",
                CorrectionMode.AUTO_DETECT: "自動補正"
            }.get(mode, "不明")

            mask_mode_text = "（散らばった領域モード）" if use_scattered_mode else ""
            status = f"色調補正完了 [{mode_name}]{mask_mode_text} | 強度: {int(strength * 100)}% | LAB: a*={a_adjustment_factor:.0%}, b*={b_adjustment_factor:.0%}, L={l_adjustment_factor:.0%}"
            return result_rgb, status

        except Exception as e:
            return image, f"色調補正エラー: {str(e)}"

    def transfer_inpaint_result_for_correction(
        self,
        gallery_data
    ) -> Optional[np.ndarray]:
        """
        Transfer inpainting result to Tab 3 for color correction.

        Args:
            gallery_data: Gallery data from Tab 2

        Returns:
            numpy array of the last inpainting result
        """
        if gallery_data is None or len(gallery_data) == 0:
            return self._last_inpaint_result

        # Get the last image from gallery (usually 100% removal)
        try:
            last_item = gallery_data[-1]
            if isinstance(last_item, tuple):
                img = last_item[0]
            else:
                img = last_item

            if isinstance(img, Image.Image):
                return np.array(img)
            elif isinstance(img, np.ndarray):
                return img
        except Exception as e:
            print(f"Gallery data extraction error: {e}")

        return self._last_inpaint_result

    def store_inpaint_result(self, image: np.ndarray) -> None:
        """Store the latest inpainting result for later use."""
        self._last_inpaint_result = image

    def _extract_mask_from_editor(self, editor_data: dict) -> Optional[np.ndarray]:
        """Extract binary mask from ImageEditor data."""
        if editor_data is None:
            return None

        try:
            if isinstance(editor_data, dict):
                if 'layers' in editor_data and len(editor_data['layers']) > 0:
                    layer = editor_data['layers'][0]
                    if isinstance(layer, np.ndarray) and len(layer.shape) == 3:
                        # Convert to grayscale and threshold
                        gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                        mask = (gray > 30).astype(np.uint8) * 255
                        return mask
                elif 'composite' in editor_data:
                    composite = editor_data['composite']
                    if isinstance(composite, np.ndarray):
                        if len(composite.shape) == 3 and composite.shape[2] == 4:
                            # Use alpha channel
                            mask = composite[:, :, 3]
                            return (mask > 30).astype(np.uint8) * 255
                        elif len(composite.shape) == 3:
                            gray = cv2.cvtColor(composite[:, :, :3], cv2.COLOR_RGB2GRAY)
                            return (gray > 30).astype(np.uint8) * 255
        except Exception as e:
            print(f"Mask extraction error: {e}")

        return None

    def _map_correction_mode(self, mode_str: str) -> CorrectionMode:
        """Map Japanese mode string to CorrectionMode enum."""
        mode_map = {
            "青み除去（推奨）": CorrectionMode.BLUE_REMOVAL,
            "青み除去": CorrectionMode.BLUE_REMOVAL,
            "色味転送": CorrectionMode.COLOR_TRANSFER,
            "自動補正": CorrectionMode.AUTO_DETECT,
            # English fallbacks
            "blue_removal": CorrectionMode.BLUE_REMOVAL,
            "color_transfer": CorrectionMode.COLOR_TRANSFER,
            "auto_detect": CorrectionMode.AUTO_DETECT,
        }
        return mode_map.get(mode_str, CorrectionMode.BLUE_REMOVAL)
