#!/usr/bin/env python3
"""
Gradio Beard Detection & Removal Application v6

v6 新機能:
- MAT (Mask-Aware Transformer) Inpainting: CVPR 2022 Best Paper Finalist
- 顔画像に特化した高品質なinpainting
- FFHQとCelebA-HQの2つのプリトレーニングモデルに対応

v5からの機能:
- 髭オーバーレイ機能: 色補正後に残す髭を元画像からオーバーレイ
- 自然な髭の薄め効果を実現

ワークフロー:
1. Tab 1: 髭検出 → 削除対象を選択
2. Tab 2: 削除対象をInpainting (MAT/LaMa/OpenCV)
3. Tab 3: 色補正 → 残した髭をオーバーレイ → 最終出力

Usage:
    python app_gradio_v6.py

Required checkpoints:
    - sam_vit_h_4b8939.pth (for Grounded SAM detection)
    - groundingdino_swint_ogc.pth (for Grounded SAM detection)
    - checkpoints/mat/FFHQ-512.pkl (for MAT FFHQ)
    - checkpoints/mat/CelebA-HQ.pkl (for MAT CelebA-HQ)
"""

import gradio as gr
from beard_inpainting_modules import BeardRemovalPipeline
import numpy as np

# Global pipeline instance
pipeline = BeardRemovalPipeline()

# =============================================================================
# v5: Global storage for original image and masks
# =============================================================================
_v5_storage = {
    'original_image': None,      # Tab 1の元画像
    'detect_mask': None,         # 全検出マスク
    'selection_mask': None,      # 削除対象マスク
}


def store_original_image(image):
    """Store original image for later overlay."""
    global _v5_storage
    if image is not None:
        _v5_storage['original_image'] = image.copy()
        print(f"[v5] 元画像を保存: {image.shape}")


def store_masks(detect_mask, selection_mask):
    """Store detection and selection masks."""
    global _v5_storage
    if detect_mask is not None:
        _v5_storage['detect_mask'] = detect_mask.copy()
    if selection_mask is not None:
        _v5_storage['selection_mask'] = selection_mask.copy()


def get_remaining_beard_mask():
    """Calculate remaining beard mask (detect - selection)."""
    global _v5_storage
    import cv2

    detect = _v5_storage.get('detect_mask')
    selection = _v5_storage.get('selection_mask')

    if detect is None:
        return None

    # Convert to grayscale if needed
    if len(detect.shape) == 3:
        detect_gray = cv2.cvtColor(detect, cv2.COLOR_RGB2GRAY)
    else:
        detect_gray = detect

    if selection is None:
        # No selection = keep all detected beard
        return (detect_gray > 30).astype(np.uint8) * 255

    if len(selection.shape) == 3:
        selection_gray = cv2.cvtColor(selection, cv2.COLOR_RGB2GRAY)
    else:
        selection_gray = selection

    # Remaining = detected - selected for removal
    detect_binary = (detect_gray > 30).astype(np.uint8) * 255
    selection_binary = (selection_gray > 30).astype(np.uint8) * 255
    remaining = cv2.subtract(detect_binary, selection_binary)

    return remaining


# =============================================================================
# Callback wrappers
# =============================================================================

def preview_coordinate_region(image, x1: int, y1: int, x2: int, y2: int):
    """座標指定領域のプレビューを表示"""
    import cv2

    if image is None:
        return None

    h, w = image.shape[:2]

    # 座標のバリデーション
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))

    preview = image.copy()

    # 半透明の塗りつぶし領域
    overlay = preview.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.3, preview, 0.7, 0, preview)

    # 枠線
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 座標情報を表示
    coord_text = f"({x1}, {y1}) - ({x2}, {y2})"
    size_text = f"Size: {x2-x1} x {y2-y1}"
    cv2.putText(preview, coord_text, (x1, max(y1 - 10, 20)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(preview, size_text, (x1, min(y2 + 20, h - 5)),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return preview


def process_beard_detection(
    image, editor_data, detection_mode,
    text_prompt, box_threshold, text_threshold,
    threshold_value, min_area, max_area,
    selection_shape, enable_denoise,
    use_coordinates=False, coord_x1=0, coord_y1=0, coord_x2=100, coord_y2=100
):
    """Wrapper for pipeline.process_detection."""
    import cv2

    # v5: Store original image
    store_original_image(image)

    # ========== 前処理: ノイズ除去（オプション） ==========
    processed_image = image
    if enable_denoise and image is not None:
        processed_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        print("[前処理] バイラテラルフィルターでノイズ除去を適用")

    # ========== 座標入力モードの場合、editor_dataを上書き ==========
    if use_coordinates and image is not None:
        h, w = image.shape[:2]
        x1 = max(0, min(int(coord_x1), w - 1))
        y1 = max(0, min(int(coord_y1), h - 1))
        x2 = max(x1 + 1, min(int(coord_x2), w))
        y2 = max(y1 + 1, min(int(coord_y2), h))

        # 座標からマスクを生成してeditor_dataとして渡す
        coord_mask = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(coord_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)

        editor_data = {
            'background': image,
            'layers': [coord_mask],
            'composite': image
        }
        print(f"[座標入力] 領域: ({x1}, {y1}) - ({x2}, {y2})")

    use_grounded_sam = "Grounded SAM" in detection_mode
    result = pipeline.process_detection(
        image=processed_image,
        editor_data=editor_data,
        use_grounded_sam=use_grounded_sam,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        threshold_value=threshold_value,
        min_area=min_area,
        max_area=max_area,
        selection_shape=selection_shape
    )

    # v5: Store detect_mask
    if len(result) >= 2 and result[1] is not None:
        _v5_storage['detect_mask'] = result[1].copy()
        print(f"[v5] 検出マスクを保存")

    return result


def update_selection_preview(removal_percentage, selection_mode, new_seed):
    """Wrapper for pipeline.update_selection."""
    result = pipeline.update_selection(
        removal_percentage=removal_percentage,
        selection_mode=selection_mode,
        new_seed=new_seed
    )

    # v5: Store selection_mask
    if len(result) >= 2 and result[1] is not None:
        _v5_storage['selection_mask'] = result[1].copy()
        print(f"[v5] 選択マスクを保存")

    return result


def transfer_to_inpainting(mask):
    """Wrapper for pipeline.transfer_for_inpainting."""
    return pipeline.transfer_for_inpainting(mask)


def extract_manual_mask(editor_data, base_image):
    """
    ImageEditorから手動描画マスクを抽出してinpaint_maskに適用

    Args:
        editor_data: ImageEditorのデータ
        base_image: 元画像（サイズ参照用）

    Returns:
        (mask, status): 抽出されたマスク画像とステータスメッセージ
    """
    import cv2

    if editor_data is None:
        return None, "マスクエディタに画像がありません"

    try:
        mask = None

        if isinstance(editor_data, dict):
            # layersから描画内容を取得
            if 'layers' in editor_data and len(editor_data['layers']) > 0:
                layer = editor_data['layers'][0]
                if isinstance(layer, np.ndarray):
                    if len(layer.shape) == 3:
                        # RGB/RGBAからグレースケールに変換
                        if layer.shape[2] == 4:
                            # アルファチャンネルを考慮
                            gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                        else:
                            gray = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
                        mask = (gray > 30).astype(np.uint8) * 255
                    else:
                        mask = (layer > 30).astype(np.uint8) * 255

        if mask is None or np.max(mask) == 0:
            return None, "描画された領域がありません。白色ブラシで塗ってください。"

        pixel_count = np.sum(mask > 0)
        return mask, f"手動マスク適用完了: {pixel_count} pixels"

    except Exception as e:
        print(f"手動マスク抽出エラー: {e}")
        return None, f"マスク抽出エラー: {str(e)}"


def transfer_image_to_manual_editor(image):
    """元画像を手動マスクエディタに転送"""
    if image is None:
        return None

    # PILの場合はnumpyに変換
    if hasattr(image, 'convert'):
        import numpy as np
        image = np.array(image)

    return image


def process_lama_inpainting(image, mask, thinning_levels, inpaint_method, mat_enhanced, texture_strength, color_correction_strength, opencv_radius, progress=gr.Progress()):
    """Wrapper for pipeline.process_inpainting."""
    from PIL import Image

    method_map = {
        "MAT (FFHQ - 顔専用)": "mat_ffhq",
        "MAT (CelebA-HQ - 顔専用)": "mat_celeba",
        "Simple LaMa": "lama",
        "OpenCV Telea": "opencv_telea",
        "OpenCV Navier-Stokes": "opencv_ns"
    }
    method = method_map.get(inpaint_method, "mat_ffhq")

    gallery_items, status = pipeline.process_inpainting(
        image=image,
        mask=mask,
        thinning_levels=thinning_levels,
        progress=progress,
        method=method,
        opencv_radius=opencv_radius,
        mat_enhanced=mat_enhanced,
        texture_strength=texture_strength,
        color_correction_strength=color_correction_strength
    )

    if gallery_items and len(gallery_items) > 0:
        last_item = gallery_items[-1]
        if isinstance(last_item, tuple):
            img = last_item[0]
        else:
            img = last_item
        if isinstance(img, Image.Image):
            pipeline.store_inpaint_result(np.array(img))
        elif isinstance(img, np.ndarray):
            pipeline.store_inpaint_result(img)

    last_image = pipeline.transfer_inpaint_result_for_correction(gallery_items)

    if last_image is not None:
        editor_data = {
            "background": last_image,
            "layers": [],
            "composite": last_image
        }
    else:
        editor_data = None

    return gallery_items, status, last_image, editor_data, editor_data, editor_data


# =============================================================================
# Tab 3: Color Correction Callbacks
# =============================================================================

def transfer_from_inpainting(gallery_data):
    """Transfer inpainting result to Tab 3."""
    img = pipeline.transfer_inpaint_result_for_correction(gallery_data)
    if img is not None:
        editor_data = {
            "background": img,
            "layers": [],
            "composite": img
        }
    else:
        editor_data = None
    return img, editor_data, editor_data, editor_data


def transfer_tab1_mask(selection_mask):
    """Transfer Tab 1's selection mask to Tab 3 for preview."""
    if selection_mask is None:
        return None, "Tab 1で髭を検出・選択してください"
    return selection_mask, "Tab 1のマスクを取得しました"


def transfer_tab1_full_mask(detect_mask):
    """Transfer Tab 1's full detection mask for exclusion."""
    if detect_mask is None:
        return None, "Tab 1で髭を検出してください"
    return detect_mask, "Tab 1の全検出マスクを除外領域に追加しました"


def get_remaining_beard_preview():
    """Get preview of remaining beard mask."""
    remaining = get_remaining_beard_mask()
    if remaining is None:
        return None, "Tab 1で髭を検出・選択してください"

    pixel_count = np.sum(remaining > 0)
    return remaining, f"残す髭: {pixel_count} pixels"


# =============================================================================
# v5: Beard Overlay Function
# =============================================================================

def overlay_beard_on_image(color_corrected_image, overlay_strength=1.0, edge_blur=3):
    """Overlay remaining beard pixels onto color-corrected image."""
    import cv2

    global _v5_storage

    original = _v5_storage.get('original_image')
    remaining_mask = get_remaining_beard_mask()

    if original is None:
        return color_corrected_image, "元画像がありません。Tab 1で画像をアップロードしてください"

    if remaining_mask is None or not np.any(remaining_mask > 0):
        return color_corrected_image, "残す髭がありません"

    if color_corrected_image is None:
        return None, "色補正画像がありません"

    # Ensure same size
    if original.shape[:2] != color_corrected_image.shape[:2]:
        return color_corrected_image, "画像サイズが一致しません"

    # Create blurred mask for smooth blending
    if edge_blur > 0:
        blur_size = edge_blur * 2 + 1
        mask_float = cv2.GaussianBlur(
            remaining_mask.astype(np.float32),
            (blur_size, blur_size), 0
        ) / 255.0
    else:
        mask_float = remaining_mask.astype(np.float32) / 255.0

    # Apply overlay strength
    mask_float = mask_float * overlay_strength

    # Expand to 3 channels
    mask_3ch = np.stack([mask_float] * 3, axis=-1)

    # Blend: color_corrected * (1 - mask) + original * mask
    result = (
        color_corrected_image.astype(np.float32) * (1.0 - mask_3ch) +
        original.astype(np.float32) * mask_3ch
    ).astype(np.uint8)

    pixel_count = np.sum(remaining_mask > 0)
    return result, f"髭オーバーレイ完了 | 残した髭: {pixel_count} pixels | 強度: {overlay_strength:.0%}"


def fill_mask_gaps(mask, closing_size=15, edge_blur=21):
    """
    マスクの隙間を埋めてエッジをスムーズにする

    Args:
        mask: 入力マスク (numpy array)
        closing_size: クロージング処理のカーネルサイズ（大きいほど隙間が埋まる）
        edge_blur: エッジブラーのサイズ（大きいほど滑らかに遷移）

    Returns:
        処理後のマスク（0.0-1.0のfloat、グラデーション付き）
    """
    import cv2

    if mask is None:
        return None

    # グレースケールに変換
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask.copy()

    # バイナリ化
    _, binary = cv2.threshold(mask_gray, 30, 255, cv2.THRESH_BINARY)

    # モルフォロジー クロージング（膨張→収縮）で隙間を埋める
    if closing_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (closing_size * 2 + 1, closing_size * 2 + 1)
        )
        filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    else:
        filled = binary

    # エッジをガウシアンブラーでスムーズに
    if edge_blur > 0:
        blur_size = edge_blur * 2 + 1
        smoothed = cv2.GaussianBlur(filled.astype(np.float32), (blur_size, blur_size), 0)
    else:
        smoothed = filled.astype(np.float32)

    # 0.0-1.0に正規化
    result = smoothed / 255.0

    return result


def process_color_correction_with_overlay(
    image, target_editor, source_editor,
    correction_mode, strength,
    mask_source, tab1_mask,
    a_factor, b_factor, l_factor,
    enable_scattered_mode,
    exclusion_editor, enable_exclusion, tab1_full_mask,
    enable_overlay, overlay_strength, overlay_edge_blur,
    use_direct_fill=False,
    enable_gap_fill=False, gap_fill_size=15, gap_edge_blur=21
):
    """Color correction with optional beard overlay."""
    import cv2

    # マスク隙間埋め処理
    processed_tab1_mask = tab1_mask
    if enable_gap_fill and tab1_mask is not None:
        filled_mask = fill_mask_gaps(tab1_mask, gap_fill_size, gap_edge_blur)
        # float maskをuint8に戻す（255スケール、グラデーション保持）
        processed_tab1_mask = (filled_mask * 255).astype(np.uint8)
        print(f"[隙間埋め] closing_size={gap_fill_size}, edge_blur={gap_edge_blur}")

    # First, do color correction
    corrected_image, status = process_color_correction_with_source(
        image, target_editor, source_editor,
        correction_mode, strength,
        mask_source, processed_tab1_mask,
        a_factor, b_factor, l_factor,
        enable_scattered_mode,
        exclusion_editor, enable_exclusion, tab1_full_mask,
        use_direct_fill
    )

    if corrected_image is None:
        return corrected_image, status

    # Then, overlay beard if enabled
    if enable_overlay:
        result, overlay_status = overlay_beard_on_image(
            corrected_image,
            overlay_strength=overlay_strength,
            edge_blur=int(overlay_edge_blur)
        )
        return result, f"{status} → {overlay_status}"
    else:
        return corrected_image, status


def process_color_correction_with_source(
    image, target_editor, source_editor,
    correction_mode, strength,
    mask_source, tab1_mask,
    a_factor, b_factor, l_factor,
    enable_scattered_mode,
    exclusion_editor, enable_exclusion, tab1_full_mask,
    use_direct_fill=False
):
    """Wrapper for color correction with mask source selection and exclusion mask."""
    import cv2

    use_scattered_mode = False

    if mask_source == "Tab 1の選択マスクを使用":
        if tab1_mask is None:
            return image, "Tab 1のマスクが設定されていません"

        target_editor_data = {
            "layers": [tab1_mask],
            "background": image
        }
        use_scattered_mode = enable_scattered_mode
    else:
        target_editor_data = target_editor

    exclusion_mask = None
    if enable_exclusion:
        if exclusion_editor is not None:
            try:
                if isinstance(exclusion_editor, dict) and 'layers' in exclusion_editor:
                    if len(exclusion_editor['layers']) > 0:
                        layer = exclusion_editor['layers'][0]
                        if isinstance(layer, np.ndarray) and len(layer.shape) == 3:
                            gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                            exclusion_mask = (gray > 30).astype(np.uint8) * 255
            except Exception as e:
                print(f"除外マスク抽出エラー: {e}")

        if tab1_full_mask is not None:
            full_mask = tab1_full_mask.copy()
            if len(full_mask.shape) == 3:
                full_mask = cv2.cvtColor(full_mask, cv2.COLOR_RGB2GRAY)
            full_mask = (full_mask > 30).astype(np.uint8) * 255

            if exclusion_mask is not None:
                exclusion_mask = cv2.bitwise_or(exclusion_mask, full_mask)
            else:
                exclusion_mask = full_mask

    if exclusion_mask is not None and np.any(exclusion_mask > 0):
        return process_color_correction_with_exclusion(
            image=image,
            target_editor_data=target_editor_data,
            source_editor_data=source_editor,
            exclusion_mask=exclusion_mask,
            correction_mode=correction_mode,
            strength=strength,
            a_factor=a_factor,
            b_factor=b_factor,
            l_factor=l_factor,
            use_scattered_mode=use_scattered_mode,
            use_direct_fill=use_direct_fill
        )
    else:
        return pipeline.process_color_correction(
            image=image,
            target_editor_data=target_editor_data,
            source_editor_data=source_editor,
            correction_mode=correction_mode,
            strength=strength,
            a_adjustment_factor=a_factor,
            b_adjustment_factor=b_factor,
            l_adjustment_factor=l_factor,
            use_scattered_mode=use_scattered_mode,
            use_direct_fill=use_direct_fill
        )


def process_color_correction_with_exclusion(
    image, target_editor_data, source_editor_data,
    exclusion_mask, correction_mode, strength,
    a_factor, b_factor, l_factor, use_scattered_mode,
    use_direct_fill=False
):
    """除外マスクを考慮した色補正処理"""
    import cv2

    if image is None:
        return None, "画像がありません"

    target_mask = None
    if target_editor_data is not None:
        if isinstance(target_editor_data, dict) and 'layers' in target_editor_data:
            if len(target_editor_data['layers']) > 0:
                layer = target_editor_data['layers'][0]
                if isinstance(layer, np.ndarray):
                    if len(layer.shape) == 3:
                        gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                        target_mask = (gray > 30).astype(np.uint8) * 255
                    else:
                        target_mask = (layer > 30).astype(np.uint8) * 255

    if target_mask is None or not np.any(target_mask > 0):
        return image, "対象領域が指定されていません"

    if use_scattered_mode:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        target_mask = cv2.dilate(target_mask, kernel, iterations=1)

    correction_mask = target_mask.copy()
    if exclusion_mask is not None and np.any(exclusion_mask > 0):
        correction_mask = cv2.bitwise_or(correction_mask, exclusion_mask)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    skin_mask = detect_surrounding_skin_with_exclusion(
        image_bgr, correction_mask, exclusion_mask,
        dilation_size=50, erosion_size=15
    )

    if not np.any(skin_mask > 0):
        skin_mask = detect_surrounding_skin_with_exclusion(
            image_bgr, correction_mask, exclusion_mask,
            dilation_size=100, erosion_size=20
        )

    if not np.any(skin_mask > 0):
        return image, "参照できる肌色がありません"

    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    skin_pixels_lab = lab_image[skin_mask > 0]
    target_l = np.percentile(skin_pixels_lab[:, 0], 75)
    target_a = np.median(skin_pixels_lab[:, 1])
    target_b = np.median(skin_pixels_lab[:, 2])

    corr_y, corr_x = np.where(correction_mask > 0)
    original_l = lab_image[corr_y, corr_x, 0]
    original_a = lab_image[corr_y, corr_x, 1]
    original_b = lab_image[corr_y, corr_x, 2]

    new_l = original_l + (target_l - original_l) * strength * l_factor
    new_a = original_a + (target_a - original_a) * strength * a_factor
    new_b = original_b + (target_b - original_b) * strength * b_factor

    lab_image[corr_y, corr_x, 0] = new_l
    lab_image[corr_y, corr_x, 1] = new_a
    lab_image[corr_y, corr_x, 2] = new_b

    lab_image = np.clip(lab_image, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    smoothed = cv2.bilateralFilter(result_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    smooth_strength = min(strength * 0.7, 0.8)

    mask_float = correction_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_float] * 3, axis=-1)
    result_bgr = (
        result_bgr.astype(np.float32) * (1.0 - smooth_strength * mask_3ch) +
        smoothed.astype(np.float32) * (smooth_strength * mask_3ch)
    ).astype(np.uint8)

    edge_blur = 15
    blur_mask = cv2.GaussianBlur(correction_mask.astype(np.float32), (edge_blur * 2 + 1, edge_blur * 2 + 1), 0) / 255.0
    blur_mask_3ch = np.stack([blur_mask] * 3, axis=-1)

    result_bgr = (
        image_bgr.astype(np.float32) * (1.0 - blur_mask_3ch) +
        result_bgr.astype(np.float32) * blur_mask_3ch
    ).astype(np.uint8)

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    corr_pixels = np.sum(correction_mask > 0)
    fill_mode = "美肌塗りつぶし" if use_direct_fill else "自動検出"
    status = f"色補正完了 [{fill_mode}] | 補正: {corr_pixels} px | 参照: {len(skin_pixels_lab)} px"
    return result_rgb, status


def detect_surrounding_skin_with_exclusion(
    image, target_mask, exclusion_mask,
    dilation_size=50, erosion_size=15
):
    """除外マスクを考慮した周辺肌色検出"""
    import cv2

    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size * 2 + 1, dilation_size * 2 + 1)
    )
    dilated = cv2.dilate(target_mask, kernel_dilate, iterations=1)

    kernel_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_size * 2 + 1, erosion_size * 2 + 1)
    )
    excluded_target = cv2.dilate(target_mask, kernel_erode, iterations=1)

    surrounding = cv2.subtract(dilated, excluded_target)

    if exclusion_mask is not None and np.any(exclusion_mask > 0):
        kernel_exclusion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        exclusion_dilated = cv2.dilate(exclusion_mask, kernel_exclusion, iterations=1)
        surrounding = cv2.subtract(surrounding, exclusion_dilated)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 200, 255], dtype=np.uint8)
    skin_mask_hsv = cv2.inRange(hsv, lower_skin, upper_skin)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower_skin_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)

    skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)

    result_mask = cv2.bitwise_and(surrounding, skin_mask)

    return result_mask


# =============================================================================
# Gradio UI
# =============================================================================

def create_app():
    """Create Gradio application with modular backend."""

    with gr.Blocks(
        title="髭検出・修復アプリ v6"
    ) as app:
        gr.Markdown("""
        # 髭検出・修復アプリケーション v6

        **ワークフロー:** Tab 1（髭検出）→ Tab 2（Inpainting）→ Tab 3（色調補正 + 髭オーバーレイ）

        **v6 新機能:**
        - **MAT (Mask-Aware Transformer)**: CVPR 2022 Best Paper Finalist
        - 顔画像に特化した高品質inpainting（512x512処理）
        - FFHQとCelebA-HQの2モデル対応

        **v5からの機能:**
        - Tab 1: 髭検出（ルールベース / Grounded SAM）
        - Tab 2: MAT / LaMa / OpenCV Inpainting
        - Tab 3: LAB色空間ベースの青髭補正 + 髭オーバーレイ
        """)

        with gr.Tabs():
            # =================================================================
            # Tab 1: 髭検出
            # =================================================================
            with gr.TabItem("1. 髭検出"):
                gr.Markdown("""
                ### 使い方
                1. 画像をアップロード
                2. 髭の範囲を選択（矩形で塗りつぶし or 自由形状で線を囲む）
                3. 検出モードを選択して「髭を検出」をクリック
                4. Remove % で削除対象を選択（**残りはTab 3でオーバーレイ可能**）
                5. 「マスクを Tab 2 に転送」
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="画像をアップロード",
                            type="numpy",
                            sources=["upload"]
                        )

                        selection_shape_radio = gr.Radio(
                            choices=["矩形", "自由形状"],
                            value="矩形",
                            label="選択形状",
                            info="矩形: 塗りつぶし / 自由形状: 線で囲む"
                        )

                        rect_editor = gr.ImageEditor(
                            label="髭の範囲を選択",
                            type="numpy",
                            brush=gr.Brush(
                                default_size=30,
                                colors=["white"],
                                default_color="white"
                            ),
                            eraser=gr.Eraser(default_size=30)
                        )

                        # 座標入力セクション
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

                        detection_mode_radio = gr.Radio(
                            choices=["ルールベース（1本ずつ検出）", "Grounded SAM"],
                            value="ルールベース（1本ずつ検出）",
                            label="検出モード",
                            info="ルールベースは髭を1本ずつ高精度に検出します（推奨）"
                        )

                        enable_denoise_checkbox = gr.Checkbox(
                            label="前処理: ノイズ除去を有効化",
                            value=False,
                            info="毛穴などの細かいノイズを髭と誤検出する場合にON"
                        )

                        with gr.Accordion("ルールベース パラメータ", open=True):
                            threshold_slider = gr.Slider(
                                minimum=20, maximum=200, value=80, step=5,
                                label="二値化閾値",
                                info="小さいほど薄い髭も検出"
                            )
                            min_area_slider = gr.Slider(
                                minimum=1, maximum=500, value=10, step=1,
                                label="最小面積"
                            )
                            max_area_slider = gr.Slider(
                                minimum=100, maximum=10000, value=5000, step=100,
                                label="最大面積"
                            )

                        with gr.Accordion("Grounded SAM パラメータ", open=False):
                            text_prompt_input = gr.Textbox(
                                label="検出プロンプト",
                                value="beard. facial hair. stubble."
                            )
                            box_threshold_slider = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                                label="Box Threshold"
                            )
                            text_threshold_slider = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.20, step=0.05,
                                label="Text Threshold"
                            )

                        detect_btn = gr.Button("髭を検出", variant="primary")

                    with gr.Column(scale=1):
                        detect_result = gr.Image(
                            label="検出結果（各髭を色分け表示）- クリックで拡大",
                            type="numpy",
                            interactive=False
                        )
                        detect_mask = gr.Image(
                            label="検出マスク（全髭）",
                            type="numpy",
                            interactive=False
                        )
                        detect_status = gr.Textbox(
                            label="ステータス",
                            interactive=False
                        )

                gr.Markdown("---")
                gr.Markdown("### 削除対象の選択")

                with gr.Row():
                    with gr.Column(scale=1):
                        removal_slider = gr.Slider(
                            minimum=0, maximum=100, value=50, step=1,
                            label="Remove %（削除割合）",
                            info="残りの髭はTab 3でオーバーレイ可能"
                        )
                        selection_mode_radio = gr.Radio(
                            choices=["ランダム", "面積大", "面積小", "信頼度順"],
                            value="ランダム",
                            label="選択モード"
                        )
                        with gr.Row():
                            new_seed_btn = gr.Button("新しいシード")
                            update_selection_btn = gr.Button("選択を更新", variant="secondary")

                    with gr.Column(scale=1):
                        selection_result = gr.Image(
                            label="選択結果（赤=削除対象, 他の色=保持→オーバーレイ）- クリックで拡大",
                            type="numpy",
                            interactive=False
                        )
                        selection_mask = gr.Image(
                            label="削除対象マスク",
                            type="numpy",
                            interactive=False
                        )
                        selection_status = gr.Textbox(
                            label="選択ステータス",
                            interactive=False
                        )

                transfer_btn = gr.Button(
                    "マスクを Tab 2 に転送 →",
                    variant="primary",
                    size="lg"
                )

                # イベント
                image_input.change(
                    fn=lambda img: img,
                    inputs=[image_input],
                    outputs=[rect_editor]
                )

                # イベント: 座標プレビュー
                preview_btn.click(
                    fn=preview_coordinate_region,
                    inputs=[image_input, coord_x1, coord_y1, coord_x2, coord_y2],
                    outputs=[coord_preview]
                )

                # イベント: 座標入力時の自動プレビュー
                for coord_input in [coord_x1, coord_y1, coord_x2, coord_y2]:
                    coord_input.change(
                        fn=preview_coordinate_region,
                        inputs=[image_input, coord_x1, coord_y1, coord_x2, coord_y2],
                        outputs=[coord_preview]
                    )

                detect_btn.click(
                    fn=process_beard_detection,
                    inputs=[
                        image_input, rect_editor, detection_mode_radio,
                        text_prompt_input, box_threshold_slider, text_threshold_slider,
                        threshold_slider, min_area_slider, max_area_slider,
                        selection_shape_radio, enable_denoise_checkbox,
                        use_coordinates, coord_x1, coord_y1, coord_x2, coord_y2
                    ],
                    outputs=[detect_result, detect_mask, detect_status]
                )

                update_selection_btn.click(
                    fn=update_selection_preview,
                    inputs=[removal_slider, selection_mode_radio, gr.State(False)],
                    outputs=[selection_result, selection_mask, selection_status]
                )

                new_seed_btn.click(
                    fn=update_selection_preview,
                    inputs=[removal_slider, selection_mode_radio, gr.State(True)],
                    outputs=[selection_result, selection_mask, selection_status]
                )

            # =================================================================
            # Tab 2: Inpainting
            # =================================================================
            with gr.TabItem("2. 髭除去"):
                gr.Markdown("""
                ### 使い方
                1. Tab 1 からマスクを転送、または**手動でマスクを描画**
                2. インペインティング手法を選択
                3. 薄め具合を選択
                4. 「髭薄めを実行」をクリック
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        inpaint_image = gr.Image(
                            label="元画像",
                            type="pil",
                            sources=["upload"]
                        )
                        inpaint_mask = gr.Image(
                            label="マスク画像（白=除去領域）",
                            type="numpy",
                            sources=["upload"]
                        )

                        # 手動マスク編集機能
                        with gr.Accordion("手動マスク編集（オプション）", open=False):
                            gr.Markdown("""
                            **ブラシで任意の領域をInpainting対象に指定**
                            - Tab 1のマスクを使わず、直接描画で領域指定が可能
                            - 白色で塗った部分がInpainting対象になります
                            """)
                            manual_mask_editor = gr.ImageEditor(
                                label="手動マスク描画（白=Inpainting対象）",
                                type="numpy",
                                brush=gr.Brush(
                                    default_size=20,
                                    colors=["white"],
                                    default_color="white"
                                ),
                                eraser=gr.Eraser(default_size=20)
                            )
                            apply_manual_mask_btn = gr.Button(
                                "描画したマスクを適用",
                                variant="secondary"
                            )

                        thinning_checkboxes = gr.CheckboxGroup(
                            choices=[30, 50, 70, 100],
                            value=[30, 50, 70, 100],
                            label="薄め具合（%）"
                        )

                        inpaint_method_radio = gr.Radio(
                            choices=[
                                "MAT (FFHQ - 顔専用)",
                                "MAT (CelebA-HQ - 顔専用)",
                                "Simple LaMa",
                                "OpenCV Telea",
                                "OpenCV Navier-Stokes"
                            ],
                            value="MAT (FFHQ - 顔専用)",
                            label="インペインティング手法",
                            info="MAT: 顔画像に最適化（512x512で処理）"
                        )

                        # MAT Enhanced Mode Options
                        with gr.Group(visible=True) as mat_options_group:
                            mat_enhanced_checkbox = gr.Checkbox(
                                value=True,
                                label="MAT強化モード（推奨）",
                                info="テクスチャ保持 + 青髭補正"
                            )
                            with gr.Row():
                                texture_strength_slider = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.8, step=0.1,
                                    label="テクスチャ強度",
                                    info="元画像の肌の質感をどれだけ復元するか"
                                )
                                color_correction_slider = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.7, step=0.1,
                                    label="青髭補正強度",
                                    info="青みをどれだけ除去するか"
                                )

                        opencv_radius_slider = gr.Slider(
                            minimum=1, maximum=20, value=3, step=1,
                            label="OpenCV 補間半径",
                            visible=False
                        )

                        inpaint_btn = gr.Button(
                            "髭薄めを実行",
                            variant="primary",
                            size="lg"
                        )

                        def toggle_method_options(method):
                            is_mat = method in ["MAT (FFHQ - 顔専用)", "MAT (CelebA-HQ - 顔専用)"]
                            is_opencv = method in ["OpenCV Telea", "OpenCV Navier-Stokes"]
                            return gr.update(visible=is_mat), gr.update(visible=is_opencv)

                        inpaint_method_radio.change(
                            fn=toggle_method_options,
                            inputs=[inpaint_method_radio],
                            outputs=[mat_options_group, opencv_radius_slider]
                        )

                    with gr.Column(scale=2):
                        inpaint_gallery = gr.Gallery(
                            label="結果",
                            columns=2,
                            rows=2,
                            height="auto",
                            object_fit="contain",
                            preview=True
                        )
                        inpaint_status = gr.Textbox(
                            label="ステータス",
                            interactive=False
                        )

                transfer_btn.click(
                    fn=transfer_to_inpainting,
                    inputs=[selection_mask],
                    outputs=[inpaint_image, inpaint_mask]
                )

                # 元画像が変更されたら手動マスクエディタにも転送
                inpaint_image.change(
                    fn=transfer_image_to_manual_editor,
                    inputs=[inpaint_image],
                    outputs=[manual_mask_editor]
                )

                # 手動マスク適用ボタン
                apply_manual_mask_btn.click(
                    fn=extract_manual_mask,
                    inputs=[manual_mask_editor, inpaint_image],
                    outputs=[inpaint_mask, inpaint_status]
                )

            # =================================================================
            # Tab 3: 色調補正 + 髭オーバーレイ
            # =================================================================
            with gr.TabItem("3. 色調補正 + オーバーレイ"):
                gr.Markdown("""
                ### 青髭補正 + 髭オーバーレイ（v5新機能）

                **ワークフロー:**
                1. Tab 2の結果を取得（Inpainting済み画像）
                2. 青髭エリアを色補正
                3. **髭オーバーレイ**: 削除しなかった髭を元画像から重ねる

                **最終出力 = 青み除去した肌 + 残した髭**
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        color_image = gr.Image(
                            label="画像（Tab 2の結果または直接アップロード）",
                            type="numpy",
                            sources=["upload"]
                        )

                        transfer_from_inpaint_btn = gr.Button(
                            "← Tab 2の結果を取得",
                            variant="secondary"
                        )

                        correction_mode_radio = gr.Radio(
                            choices=["青み除去（推奨）", "色味転送", "自動補正"],
                            value="青み除去（推奨）",
                            label="補正モード"
                        )

                        gr.Markdown("#### 対象領域の指定方法")
                        mask_source_radio = gr.Radio(
                            choices=["手動で塗る", "Tab 1の選択マスクを使用"],
                            value="手動で塗る",
                            label="マスク指定方法"
                        )

                        transfer_mask_btn = gr.Button(
                            "← Tab 1の選択マスクを取得",
                            variant="secondary"
                        )

                        scattered_mode_checkbox = gr.Checkbox(
                            label="散らばった領域モードを有効化",
                            value=True
                        )

                        # マスク隙間埋め設定
                        with gr.Accordion("マスク隙間埋め設定", open=False):
                            gr.Markdown("""
                            **髭検出マスクの隙間を埋めて滑らかに補正**
                            - 点々に見えず、自然なグラデーションで補正されます
                            - Tab 1のマスクを使用時のみ有効
                            """)
                            enable_gap_fill_checkbox = gr.Checkbox(
                                label="隙間埋めを有効化",
                                value=False,
                                info="検出マスクの隙間を埋めて補正範囲を拡大"
                            )
                            gap_fill_size_slider = gr.Slider(
                                minimum=5, maximum=50, value=15, step=5,
                                label="隙間埋めサイズ",
                                info="大きいほど広い隙間が埋まる（15-25推奨）"
                            )
                            gap_edge_blur_slider = gr.Slider(
                                minimum=5, maximum=51, value=21, step=2,
                                label="エッジぼかし",
                                info="大きいほど境界が滑らかに（21-31推奨）"
                            )

                        tab1_mask_preview = gr.Image(
                            label="Tab 1のマスク（プレビュー）",
                            type="numpy",
                            interactive=False
                        )

                        gr.Markdown("#### 手動塗り")
                        target_color_editor = gr.ImageEditor(
                            label="対象領域（青髭部分を塗る）",
                            type="numpy",
                            brush=gr.Brush(
                                default_size=50,
                                colors=["#FF0000"],
                                default_color="#FF0000"
                            ),
                            eraser=gr.Eraser(default_size=50)
                        )

                        source_color_editor = gr.ImageEditor(
                            label="スポイト領域（頬など）※色味転送モードのみ",
                            type="numpy",
                            brush=gr.Brush(
                                default_size=50,
                                colors=["#00FF00"],
                                default_color="#00FF00"
                            ),
                            eraser=gr.Eraser(default_size=50)
                        )

                        # 美肌塗りつぶしモード
                        direct_fill_checkbox = gr.Checkbox(
                            label="美肌塗りつぶしモード（色味転送時）",
                            value=False,
                            info="ONで参照色で直接塗りつぶし＋自然ブレンド"
                        )

                        # 除外マスク設定
                        with gr.Accordion("除外マスク設定", open=False):
                            enable_exclusion_checkbox = gr.Checkbox(
                                label="除外マスクを有効化",
                                value=False
                            )

                            exclusion_editor = gr.ImageEditor(
                                label="除外領域",
                                type="numpy",
                                brush=gr.Brush(
                                    default_size=50,
                                    colors=["#0000FF"],
                                    default_color="#0000FF"
                                ),
                                eraser=gr.Eraser(default_size=50)
                            )

                            transfer_full_mask_btn = gr.Button(
                                "← Tab 1の全検出マスクを除外に追加",
                                variant="secondary"
                            )

                            tab1_full_mask_preview = gr.Image(
                                label="Tab 1の全検出マスク",
                                type="numpy",
                                interactive=False
                            )

                        # ========== v5: 髭オーバーレイ設定 ==========
                        with gr.Accordion("髭オーバーレイ設定（v5新機能）", open=True):
                            gr.Markdown("""
                            **残した髭を元画像から重ねる**
                            - Tab 1で「削除対象」として選ばなかった髭を、色補正後の画像に重ねます
                            - 青みが消えた肌 + 本来の髭 = 自然な薄め効果
                            """)

                            enable_overlay_checkbox = gr.Checkbox(
                                label="髭オーバーレイを有効化",
                                value=True,
                                info="色補正後に残した髭を重ねる"
                            )

                            overlay_strength_slider = gr.Slider(
                                minimum=0, maximum=1, value=1.0, step=0.05,
                                label="オーバーレイ強度",
                                info="1.0 = 完全に元の髭、0.5 = 半透明"
                            )

                            overlay_edge_blur_slider = gr.Slider(
                                minimum=0, maximum=10, value=3, step=1,
                                label="エッジぼかし",
                                info="髭の境界をぼかして自然に馴染ませる"
                            )

                            get_remaining_btn = gr.Button(
                                "残す髭マスクを確認",
                                variant="secondary"
                            )

                            remaining_beard_preview = gr.Image(
                                label="残す髭マスク（検出 - 削除対象）",
                                type="numpy",
                                interactive=False
                            )

                            remaining_status = gr.Textbox(
                                label="残す髭の状態",
                                interactive=False
                            )

                        # 補正強度
                        color_strength_slider = gr.Slider(
                            minimum=0, maximum=1, value=0.8, step=0.05,
                            label="補正強度"
                        )

                        with gr.Accordion("LABパラメータ調整", open=False):
                            a_factor_slider = gr.Slider(
                                minimum=0, maximum=1, value=0.3, step=0.05,
                                label="a* 調整係数"
                            )
                            b_factor_slider = gr.Slider(
                                minimum=0, maximum=1, value=0.6, step=0.05,
                                label="b* 調整係数"
                            )
                            l_factor_slider = gr.Slider(
                                minimum=0, maximum=1, value=0.5, step=0.05,
                                label="L 調整係数"
                            )

                        apply_color_btn = gr.Button(
                            "色調補正 + オーバーレイを適用",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        color_result = gr.Image(
                            label="最終結果（色補正 + 髭オーバーレイ）",
                            type="numpy",
                            interactive=False
                        )
                        color_status = gr.Textbox(
                            label="ステータス",
                            interactive=False
                        )

                # イベント
                transfer_from_inpaint_btn.click(
                    fn=transfer_from_inpainting,
                    inputs=[inpaint_gallery],
                    outputs=[color_image, target_color_editor, source_color_editor, exclusion_editor]
                )

                transfer_mask_btn.click(
                    fn=transfer_tab1_mask,
                    inputs=[selection_mask],
                    outputs=[tab1_mask_preview, color_status]
                )

                transfer_full_mask_btn.click(
                    fn=transfer_tab1_full_mask,
                    inputs=[detect_mask],
                    outputs=[tab1_full_mask_preview, color_status]
                )

                get_remaining_btn.click(
                    fn=get_remaining_beard_preview,
                    inputs=[],
                    outputs=[remaining_beard_preview, remaining_status]
                )

                apply_color_btn.click(
                    fn=process_color_correction_with_overlay,
                    inputs=[
                        color_image, target_color_editor, source_color_editor,
                        correction_mode_radio, color_strength_slider,
                        mask_source_radio, tab1_mask_preview,
                        a_factor_slider, b_factor_slider, l_factor_slider,
                        scattered_mode_checkbox,
                        exclusion_editor, enable_exclusion_checkbox, tab1_full_mask_preview,
                        enable_overlay_checkbox, overlay_strength_slider, overlay_edge_blur_slider,
                        direct_fill_checkbox,
                        enable_gap_fill_checkbox, gap_fill_size_slider, gap_edge_blur_slider
                    ],
                    outputs=[color_result, color_status]
                )

        # Cross-Tab Event
        inpaint_btn.click(
            fn=process_lama_inpainting,
            inputs=[inpaint_image, inpaint_mask, thinning_checkboxes, inpaint_method_radio, mat_enhanced_checkbox, texture_strength_slider, color_correction_slider, opencv_radius_slider],
            outputs=[inpaint_gallery, inpaint_status, color_image, target_color_editor, source_color_editor, exclusion_editor]
        )

        # 技術情報
        with gr.Accordion("技術情報", open=False):
            gr.Markdown("""
            ## v5 新機能: 髭オーバーレイ

            ### 処理フロー
            ```
            1. Tab 1: 髭検出 → 全検出マスク
            2. Tab 1: 削除対象選択 → 削除マスク
            3. Tab 2: 削除対象をInpainting
            4. Tab 3: Inpainted画像を色補正（青み除去）
            5. Tab 3: 残す髭マスク = 全検出 - 削除対象
            6. Tab 3: 色補正画像 + 元画像の残す髭 = 最終出力
            ```

            ### 数式
            ```
            remaining_beard = detect_mask - selection_mask
            final_output = color_corrected * (1 - mask) + original * mask
            ```

            ### 使用技術
            - **LAB色空間**: 青み除去
            - **アルファブレンディング**: 髭オーバーレイ
            - **ガウシアンブラー**: エッジぼかし
            """)

    return app


def main():
    """Main entry point."""
    app = create_app()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7867  # v6 port
    )


if __name__ == "__main__":
    main()
