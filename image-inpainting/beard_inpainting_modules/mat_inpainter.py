"""MAT (Mask-Aware Transformer) Inpainter for beard removal.

This module provides a high-level wrapper around MATEngine,
matching the interface of LamaInpainter for seamless integration.

Enhanced with:
- Texture preservation (frequency separation)
- Blue tint correction for beard shadow removal
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Callable
import cv2

# Check if MAT is available
MAT_AVAILABLE = False
try:
    from core.mat_engine import MATEngine, check_mat_availability, MAT_MODELS
    MAT_AVAILABLE = True
except ImportError as e:
    print(f"[MATInpainter] Import error: {e}")


def frequency_separation_blend(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    texture_strength: float = 0.8,
    blur_size: int = 15
) -> np.ndarray:
    """周波数分離によるテクスチャ保持ブレンド.

    MAT出力の色・構造（低周波）+ 元画像のテクスチャ（高周波）を合成。

    Args:
        original: 元画像 (H, W, 3) RGB uint8
        inpainted: MAT出力画像 (H, W, 3) RGB uint8
        mask: バイナリマスク (H, W) 255=処理領域
        texture_strength: テクスチャ復元強度 (0.0-1.0)
        blur_size: 低周波抽出のブラーサイズ（奇数）

    Returns:
        テクスチャ保持されたブレンド画像 (H, W, 3) RGB uint8
    """
    # グレースケールに変換
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 低周波成分を抽出（ぼかし）
    if blur_size % 2 == 0:
        blur_size += 1
    original_lowfreq = cv2.GaussianBlur(original_gray, (blur_size, blur_size), 0)
    inpainted_lowfreq = cv2.GaussianBlur(inpainted_gray, (blur_size, blur_size), 0)

    # 高周波成分（テクスチャ）= 元画像 - 低周波
    original_texture = original_gray - original_lowfreq

    # 合成: MAT出力の低周波 + 元画像のテクスチャ
    combined_gray = inpainted_lowfreq + original_texture * texture_strength

    # グレースケールの変化量を計算
    gray_diff = combined_gray - inpainted_gray

    # マスク領域のみにテクスチャを適用
    mask_float = (mask > 127).astype(np.float32)
    mask_float = cv2.GaussianBlur(mask_float, (5, 5), 0)  # エッジをソフトに

    # 各チャンネルにテクスチャの変化を適用
    result = inpainted.astype(np.float32).copy()
    for c in range(3):
        result[:, :, c] += gray_diff * mask_float

    return np.clip(result, 0, 255).astype(np.uint8)


def remove_blue_tint_lab(
    image: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.7,
    a_factor: float = 0.3,
    b_factor: float = 0.6,
    l_factor: float = 0.3
) -> np.ndarray:
    """LAB色空間で青髭の色調を補正.

    Args:
        image: 入力画像 (H, W, 3) RGB uint8
        mask: バイナリマスク (H, W) 255=補正領域
        strength: 全体的な補正強度 (0.0-1.0)
        a_factor: a*（赤-緑軸）の調整係数
        b_factor: b*（青-黄軸）の調整係数
        l_factor: L（明度）の調整係数

    Returns:
        色調補正された画像 (H, W, 3) RGB uint8
    """
    if not np.any(mask > 0):
        return image

    # RGB → BGR → LAB
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 周辺の肌色を自動サンプリング
    # マスクを膨張させて周辺領域を取得
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    excluded = cv2.dilate(mask, kernel_small, iterations=1)
    surrounding = cv2.subtract(dilated, excluded)

    # 周辺領域の肌色サンプル
    if np.any(surrounding > 0):
        skin_pixels = lab[surrounding > 0]
        # 明るめの肌色を目標に（75パーセンタイル）
        target_l = np.percentile(skin_pixels[:, 0], 75)
        target_a = np.median(skin_pixels[:, 1])
        target_b = np.median(skin_pixels[:, 2])
    else:
        # デフォルト（日本人の平均的な肌色）
        target_l = 185.0
        target_a = 140.0
        target_b = 145.0

    # 対象領域の座標
    target_y, target_x = np.where(mask > 127)

    if len(target_y) == 0:
        return image

    # 元の値を取得
    original_l = lab[target_y, target_x, 0]
    original_a = lab[target_y, target_x, 1]
    original_b = lab[target_y, target_x, 2]

    # 色調整
    new_l = original_l + (target_l - original_l) * strength * l_factor
    new_a = original_a + (target_a - original_a) * strength * a_factor
    new_b = original_b + (target_b - original_b) * strength * b_factor

    # 値を更新
    lab[target_y, target_x, 0] = new_l
    lab[target_y, target_x, 1] = new_a
    lab[target_y, target_x, 2] = new_b

    # LAB → BGR → RGB
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    bgr_result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb_result = cv2.cvtColor(bgr_result, cv2.COLOR_BGR2RGB)

    # マスク境界をソフトにブレンド
    mask_float = mask.astype(np.float32) / 255.0
    mask_float = cv2.GaussianBlur(mask_float, (15, 15), 0)
    mask_3ch = np.stack([mask_float] * 3, axis=-1)

    result = image.astype(np.float32) * (1.0 - mask_3ch) + rgb_result.astype(np.float32) * mask_3ch
    return np.clip(result, 0, 255).astype(np.uint8)


def masked_alpha_blend(
    original: Image.Image,
    inpainted: Image.Image,
    mask: np.ndarray,
    alpha: float,
    feather_radius: int = 5
) -> Image.Image:
    """Blend original and inpainted images using mask and alpha.

    Args:
        original: Original PIL Image
        inpainted: Fully inpainted PIL Image
        mask: Binary mask (H, W) with 255=inpaint region
        alpha: Blend factor (0.0=original, 1.0=fully inpainted)
        feather_radius: Gaussian blur radius for smooth edges

    Returns:
        Blended PIL Image
    """
    # Convert to numpy
    orig_np = np.array(original).astype(np.float32)
    inpaint_np = np.array(inpainted).astype(np.float32)

    # Normalize mask to 0-1
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask_normalized = mask.astype(np.float32) / 255.0

    # Feather edges with Gaussian blur
    if feather_radius > 0:
        blur_size = feather_radius * 2 + 1
        mask_normalized = cv2.GaussianBlur(
            mask_normalized,
            (blur_size, blur_size),
            0
        )

    # Apply alpha
    blend_mask = mask_normalized * alpha

    # Expand to 3 channels
    blend_mask_3ch = np.stack([blend_mask] * 3, axis=-1)

    # Blend: output = original * (1 - blend_mask) + inpainted * blend_mask
    result = orig_np * (1.0 - blend_mask_3ch) + inpaint_np * blend_mask_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


class MATInpainter:
    """MAT Inpainter wrapper with lazy initialization.

    Provides the same interface as LamaInpainter for easy integration.
    """

    def __init__(self, model_type: str = "ffhq"):
        """Initialize MAT Inpainter.

        Args:
            model_type: "ffhq" or "celeba"
        """
        self._engine: Optional[MATEngine] = None
        self._model_type = model_type
        self._is_available: Optional[bool] = None

    @property
    def is_available(self) -> bool:
        """Check if MAT is available (lazy check)."""
        if self._is_available is None:
            if not MAT_AVAILABLE:
                self._is_available = False
            else:
                availability = check_mat_availability()
                self._is_available = availability.get(self._model_type, False)
        return self._is_available

    def _ensure_initialized(self) -> bool:
        """Initialize MAT engine if not already done."""
        if not MAT_AVAILABLE:
            return False

        if self._engine is not None and self._engine._initialized:
            return True

        try:
            print(f"[MATInpainter] Initializing MAT engine ({self._model_type})...")
            self._engine = MATEngine(model_type=self._model_type)

            if not self._engine.is_available:
                print(f"[MATInpainter] Model file not found for {self._model_type}")
                self._is_available = False
                return False

            # Force initialization
            self._engine._ensure_initialized()
            print(f"[MATInpainter] MAT engine ready")
            return True

        except Exception as e:
            print(f"[MATInpainter] Initialization error: {e}")
            import traceback
            traceback.print_exc()
            self._engine = None  # Reset engine on failure to allow retry
            self._is_available = False
            return False

    def inpaint_single(
        self,
        image: Image.Image,
        mask: np.ndarray,
        preserve_texture: bool = False,
        remove_blue_tint: bool = False,
        texture_strength: float = 0.8,
        color_correction_strength: float = 0.7
    ) -> Image.Image:
        """Perform single inpainting operation.

        Args:
            image: PIL Image (RGB)
            mask: Binary mask (H, W) with 255=inpaint region
            preserve_texture: テクスチャ保持モード（周波数分離）
            remove_blue_tint: 青髭補正モード
            texture_strength: テクスチャ復元強度 (0.0-1.0)
            color_correction_strength: 色補正強度 (0.0-1.0)

        Returns:
            Inpainted PIL Image
        """
        if not self._ensure_initialized():
            raise RuntimeError(f"MAT ({self._model_type}) is not available")

        # Convert PIL to numpy
        image_np = np.array(image)

        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # Ensure binary
        binary_mask = ((mask > 127).astype(np.uint8)) * 255

        # Run MAT inpainting
        result_np = self._engine.inpaint(image_np, binary_mask)

        # テクスチャ保持モード
        if preserve_texture:
            print(f"[MATInpainter] テクスチャ保持モード (強度={texture_strength:.0%})")
            result_np = frequency_separation_blend(
                image_np, result_np, binary_mask,
                texture_strength=texture_strength,
                blur_size=15
            )

        # 青髭補正モード
        if remove_blue_tint:
            print(f"[MATInpainter] 青髭補正モード (強度={color_correction_strength:.0%})")
            result_np = remove_blue_tint_lab(
                result_np, binary_mask,
                strength=color_correction_strength
            )

        return Image.fromarray(result_np)

    def inpaint_enhanced(
        self,
        image: Image.Image,
        mask: np.ndarray,
        texture_strength: float = 0.8,
        color_correction_strength: float = 0.7
    ) -> Image.Image:
        """強化版inpainting: テクスチャ保持 + 青髭補正.

        髭除去に最適化された処理パイプライン:
        1. MAT inpainting（顔構造を生成）
        2. 周波数分離でテクスチャ復元
        3. LAB色空間で青髭補正

        Args:
            image: PIL Image (RGB)
            mask: Binary mask (H, W) with 255=inpaint region
            texture_strength: テクスチャ復元強度 (0.0-1.0)
            color_correction_strength: 色補正強度 (0.0-1.0)

        Returns:
            Enhanced inpainted PIL Image
        """
        return self.inpaint_single(
            image, mask,
            preserve_texture=True,
            remove_blue_tint=True,
            texture_strength=texture_strength,
            color_correction_strength=color_correction_strength
        )

    def process_thinning_levels(
        self,
        image: Image.Image,
        mask: np.ndarray,
        levels: List[int],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        enhanced_mode: bool = False,
        texture_strength: float = 0.8,
        color_correction_strength: float = 0.7
    ) -> Tuple[List[Tuple[Image.Image, str]], str]:
        """Process multiple thinning levels.

        Args:
            image: Original PIL Image
            mask: Binary mask of beard region (255=inpaint)
            levels: List of thinning percentages (e.g., [30, 50, 70, 100])
            progress_callback: Optional progress reporter (current, total, description)
            enhanced_mode: 強化モード（テクスチャ保持+青髭補正）
            texture_strength: テクスチャ復元強度 (0.0-1.0)
            color_correction_strength: 色補正強度 (0.0-1.0)

        Returns:
            Tuple of (gallery_items, status_message)
            where gallery_items is [(image, caption), ...]
        """
        if not self._ensure_initialized():
            return [], f"MAT ({self._model_type}) is not available. Check model installation."

        if image is None:
            return [], "No image provided"

        if mask is None or np.max(mask) == 0:
            return [], "No mask provided"

        if not levels:
            return [], "No thinning levels selected"

        try:
            # Ensure mask is 2D binary
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            binary_mask = ((mask > 127).astype(np.uint8)) * 255

            image_np = np.array(image)

            # Step 1: Full inpainting (100% removal)
            if progress_callback:
                desc = "MAT Enhanced Inpainting..." if enhanced_mode else "MAT Inpainting..."
                progress_callback(1, len(levels) + 1, desc)

            fully_inpainted_np = self._engine.inpaint(image_np, binary_mask)

            # 強化モード: テクスチャ保持 + 青髭補正
            if enhanced_mode:
                if progress_callback:
                    progress_callback(1, len(levels) + 1, "テクスチャ復元...")

                # テクスチャ保持
                fully_inpainted_np = frequency_separation_blend(
                    image_np, fully_inpainted_np, binary_mask,
                    texture_strength=texture_strength,
                    blur_size=15
                )

                if progress_callback:
                    progress_callback(1, len(levels) + 1, "青髭補正...")

                # 青髭補正
                fully_inpainted_np = remove_blue_tint_lab(
                    fully_inpainted_np, binary_mask,
                    strength=color_correction_strength
                )

            fully_inpainted = Image.fromarray(fully_inpainted_np)

            # Step 2: Alpha blend for each level
            gallery = [(image, "Original (0%)")]

            sorted_levels = sorted(levels)
            mode_suffix = " Enhanced" if enhanced_mode else ""
            for idx, level in enumerate(sorted_levels):
                if progress_callback:
                    progress_callback(idx + 2, len(levels) + 1, f"Blending {level}%...")

                alpha = level / 100.0

                if level == 100:
                    # No blending needed for 100%
                    blended = fully_inpainted
                    caption = f"Complete removal (MAT{mode_suffix})"
                else:
                    blended = masked_alpha_blend(
                        image,
                        fully_inpainted,
                        binary_mask,
                        alpha,
                        feather_radius=5
                    )
                    caption = f"{level}% (MAT{mode_suffix})"

                gallery.append((blended, caption))

            status = f"MAT{mode_suffix} ({self._model_type}) completed! Generated {len(sorted_levels)} thinning levels"
            return gallery, status

        except Exception as e:
            import traceback
            traceback.print_exc()
            return [], f"MAT Error: {str(e)}"

    def prepare_for_gradio(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """Prepare image and mask for inpainting from Gradio inputs.

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
        """Get status message about MAT availability."""
        if not MAT_AVAILABLE:
            return f"MAT not available (import error)"

        availability = check_mat_availability()
        if not availability.get(self._model_type, False):
            return f"MAT ({self._model_type}) model not found. Download from https://github.com/fenglinglwb/MAT"

        if self._engine is not None:
            return f"MAT ({self._model_type}): Ready"

        return f"MAT ({self._model_type}): Available (not initialized)"
