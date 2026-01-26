"""Utility functions for image and mask processing."""

import numpy as np
from PIL import Image
from typing import Tuple, Optional
import cv2
import config


def resize_image_if_needed(image: Image.Image, max_size: int = None) -> Image.Image:
    """
    画像のアスペクト比を維持しながらリサイズする。

    Args:
        image: PIL Image
        max_size: 最大サイズ（幅または高さ）

    Returns:
        リサイズされたPIL Image
    """
    if max_size is None:
        max_size = config.MAX_IMAGE_SIZE

    width, height = image.size

    if width <= max_size and height <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def convert_to_binary_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    マスクをバイナリ形式（0または255）に変換する。

    Args:
        mask: 入力マスク配列
        threshold: 二値化の閾値

    Returns:
        バイナリマスク（値は0または255）
    """
    if len(mask.shape) == 3:
        mask = np.mean(mask, axis=2)

    binary_mask = np.where(mask > threshold, 255, 0).astype(np.uint8)
    return binary_mask


def extract_mask_from_editor(editor_output: dict) -> Optional[np.ndarray]:
    """
    Gradio ImageEditorの出力からマスクを抽出する。

    Args:
        editor_output: ImageEditorからの辞書出力

    Returns:
        マスクのnumpy配列、またはマスクがない場合はNone
    """
    if editor_output is None:
        return None

    if isinstance(editor_output, dict):
        if 'layers' in editor_output and len(editor_output['layers']) > 0:
            mask_layer = editor_output['layers'][0]
            if isinstance(mask_layer, np.ndarray):
                return convert_to_binary_mask(mask_layer)

        if 'composite' in editor_output:
            composite = editor_output['composite']
            if isinstance(composite, np.ndarray):
                return convert_to_binary_mask(composite)

    if isinstance(editor_output, np.ndarray):
        return convert_to_binary_mask(editor_output)

    return None


def has_mask_content(mask: np.ndarray, threshold: float = 0.001) -> bool:
    """
    マスクに意味のある内容があるかチェックする。

    Args:
        mask: バイナリマスク配列
        threshold: 非ゼロピクセルの最小割合（0-1）

    Returns:
        マスクに内容がある場合True
    """
    if mask is None:
        return False

    non_zero_pixels = np.count_nonzero(mask)
    total_pixels = mask.size

    return (non_zero_pixels / total_pixels) > threshold


def merge_masks(masks: list) -> Optional[np.ndarray]:
    """
    複数のバイナリマスクをOR演算で結合する。

    Args:
        masks: バイナリマスク配列のリスト

    Returns:
        結合されたマスク配列
    """
    if not masks:
        return None

    merged = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        merged = np.maximum(merged, mask)

    return merged


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    numpy配列をPIL Imageに変換する。

    Args:
        array: numpy配列 (H, W, C) または (H, W)

    Returns:
        PIL Image
    """
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)

    if len(array.shape) == 2:
        return Image.fromarray(array, mode='L')
    else:
        return Image.fromarray(array, mode='RGB')


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    PIL Imageをnumpy配列に変換する。

    Args:
        image: PIL Image

    Returns:
        numpy配列
    """
    return np.array(image)


def validate_image_and_mask(
    image: Image.Image,
    mask: Image.Image
) -> Tuple[bool, Optional[str]]:
    """
    画像とマスクの寸法が互換性があるか確認する。

    Args:
        image: 入力PIL Image
        mask: マスクPIL Image

    Returns:
        (有効かどうか, エラーメッセージ)のタプル
    """
    if image is None:
        return False, "画像が指定されていません"

    if mask is None:
        return False, "マスクが指定されていません"

    if image.size != mask.size:
        return False, f"画像とマスクのサイズが一致しません。画像: {image.size}, マスク: {mask.size}"

    return True, None


def masked_alpha_blend(
    original: Image.Image,
    inpainted: Image.Image,
    mask: np.ndarray,
    alpha: float,
    feather_radius: int = None
) -> Image.Image:
    """
    マスク領域内のみをブレンドし、境界をぼかして自然な仕上がりにする。

    Args:
        original: 元のPIL画像
        inpainted: Inpainting済みPIL画像
        mask: バイナリマスク (255 = ブレンド領域)
        alpha: マスク領域のブレンド係数 (0.0 = 元画像, 1.0 = 完全除去)
        feather_radius: マスク境界のぼかしピクセル数

    Returns:
        ブレンドされたPIL画像（マスク外は元画像を維持）
    """
    if feather_radius is None:
        feather_radius = config.FEATHER_RADIUS

    if original.size != inpainted.size:
        inpainted = inpainted.resize(original.size, Image.Resampling.LANCZOS)

    original_array = np.array(original, dtype=np.float32)
    inpainted_array = np.array(inpainted, dtype=np.float32)

    # マスクを0-1に正規化
    mask_normalized = mask.astype(np.float32) / 255.0

    # マスクの境界をぼかす（自然な境界遷移のため）
    if feather_radius > 0:
        mask_normalized = cv2.GaussianBlur(
            mask_normalized,
            (feather_radius * 2 + 1, feather_radius * 2 + 1),
            0
        )

    # アルファ値を適用したマスク
    blend_mask = mask_normalized * alpha

    # 3チャンネルに拡張
    if len(blend_mask.shape) == 2:
        blend_mask = np.stack([blend_mask] * 3, axis=-1)

    # ブレンド: マスク領域のみブレンド、それ以外は元画像
    blended = original_array * (1.0 - blend_mask) + inpainted_array * blend_mask
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended, mode='RGB')
