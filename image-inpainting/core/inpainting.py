"""LaMa Inpainting wrapper for beard thinning."""

from PIL import Image
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from simple_lama_inpainting import SimpleLama

from core.image_utils import (
    numpy_to_pil,
    validate_image_and_mask,
    masked_alpha_blend
)


def get_device():
    """Get the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class InpaintingEngine:
    """SimpleLamaのラッパークラス。"""

    def __init__(self):
        """Inpaintingエンジンを初期化する。"""
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """SimpleLamaモデルを初期化する。"""
        try:
            device = get_device()
            print(f"LaMa Inpainting デバイス: {device}")
            self.model = SimpleLama(device=device)
        except Exception as e:
            raise RuntimeError(f"SimpleLamaモデルの初期化に失敗しました: {str(e)}")

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        単一の画像にInpaintingを実行する。

        Args:
            image: 入力PIL Image (RGB)
            mask: マスクPIL Image (Lモード、白=修復領域)

        Returns:
            Inpainting済みPIL Image
        """
        is_valid, error_msg = validate_image_and_mask(image, mask)
        if not is_valid:
            raise ValueError(error_msg)

        if mask.mode != 'L':
            mask = mask.convert('L')

        if image.mode != 'RGB':
            image = image.convert('RGB')

        try:
            result = self.model(image, mask)
            return result
        except Exception as e:
            raise RuntimeError(f"Inpainting処理に失敗しました: {str(e)}")

    def is_initialized(self) -> bool:
        """モデルが正しく初期化されているかチェックする。"""
        return self.model is not None


class BeardThinningProcessor:
    """髭薄め処理のプロセッサ（アルファブレンディング使用）"""

    def __init__(self):
        """プロセッサを初期化する。"""
        self.engine = InpaintingEngine()
        self._cached_inpainted: Optional[Image.Image] = None

    def process_thinning(
        self,
        original_image: Image.Image,
        beard_mask: np.ndarray,
        thinning_levels: List[int],
        progress_callback: Optional[callable] = None
    ) -> Tuple[Dict[int, Image.Image], List[str]]:
        """
        複数のアルファレベルで髭薄め結果を生成する。

        Args:
            original_image: 元のPIL画像（髭あり）
            beard_mask: 髭領域のバイナリマスク
            thinning_levels: 薄め具合のリスト（例: [30, 50, 70, 100]）
            progress_callback: 進捗コールバック(current, total, level)

        Returns:
            Tuple of:
            - Dict mapping 薄め具合 to ブレンド画像
            - ステータスメッセージのリスト

        処理フロー:
        1. LaMa Inpaintingで髭を完全除去した画像を1回生成
        2. 各薄め具合レベルで元画像とブレンド
        """
        results = {}
        messages = []

        total_steps = len(thinning_levels) + 1  # +1 for inpainting step

        # Step 1: 完全に髭を除去した画像を生成
        if progress_callback:
            progress_callback(1, total_steps, "inpainting")

        mask_pil = numpy_to_pil(beard_mask)

        try:
            fully_inpainted = self.engine.inpaint(original_image, mask_pil)
            self._cached_inpainted = fully_inpainted
            messages.append("Inpainting完了: 髭除去画像を生成しました")
        except Exception as e:
            messages.append(f"Inpaintingエラー: {str(e)}")
            return {}, messages

        # Step 2: 各レベルでブレンド画像を生成
        for idx, level in enumerate(sorted(thinning_levels)):
            step = idx + 2
            if progress_callback:
                progress_callback(step, total_steps, level)

            alpha = level / 100.0

            # マスク領域のみブレンド（自然な仕上がりに）
            blended = masked_alpha_blend(
                original_image,
                fully_inpainted,
                beard_mask,
                alpha
            )

            results[level] = blended
            messages.append(f"Level {level}%: ブレンド完了")

        return results, messages

    def get_cached_inpainted(self) -> Optional[Image.Image]:
        """キャッシュされた完全除去画像を返す。"""
        return self._cached_inpainted
