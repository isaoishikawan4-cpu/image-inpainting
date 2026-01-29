#!/usr/bin/env python3
"""
Gradio Beard Detection & Removal Application v4

Modular refactoring of v3:
- All processing logic extracted to beard_inpainting_modules/
- Clean separation of UI and business logic
- Same functionality as v3

Usage:
    python app_gradio_v4.py

Required checkpoints:
    - sam_vit_h_4b8939.pth
    - groundingdino_swint_ogc.pth
"""

import gradio as gr
from beard_inpainting_modules import BeardRemovalPipeline

# Global pipeline instance
pipeline = BeardRemovalPipeline()


# =============================================================================
# Callback wrappers
# =============================================================================

import numpy as np


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

    # ========== 前処理: ノイズ除去（オプション） ==========
    processed_image = image
    if enable_denoise and image is not None:
        # バイラテラルフィルターで毛穴などの細かいノイズを除去
        # d=9: フィルタ直径、sigmaColor=75: 色空間の標準偏差、sigmaSpace=75: 座標空間の標準偏差
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
    return pipeline.process_detection(
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


def update_selection_preview(removal_percentage, selection_mode, new_seed):
    """Wrapper for pipeline.update_selection."""
    return pipeline.update_selection(
        removal_percentage=removal_percentage,
        selection_mode=selection_mode,
        new_seed=new_seed
    )


def transfer_to_inpainting(mask):
    """Wrapper for pipeline.transfer_for_inpainting."""
    return pipeline.transfer_for_inpainting(mask)


def process_lama_inpainting(image, mask, thinning_levels, inpaint_method, opencv_radius, progress=gr.Progress()):
    """Wrapper for pipeline.process_inpainting."""
    import numpy as np
    from PIL import Image

    # Map Japanese UI names to method strings
    method_map = {
        "Simple LaMa": "lama",
        "OpenCV Telea": "opencv_telea",
        "OpenCV Navier-Stokes": "opencv_ns"
    }
    method = method_map.get(inpaint_method, "lama")

    gallery_items, status = pipeline.process_inpainting(
        image=image,
        mask=mask,
        thinning_levels=thinning_levels,
        progress=progress,
        method=method,
        opencv_radius=opencv_radius
    )

    # Store the last result for Tab 3 auto-transfer
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

    # Return gallery, status, and images for Tab 3 (color_image + editors)
    last_image = pipeline.transfer_inpaint_result_for_correction(gallery_items)

    # ImageEditor requires {"background": image, "layers": [], "composite": image} format
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
    """Transfer inpainting result to Tab 3 (color_image + editors)."""
    img = pipeline.transfer_inpaint_result_for_correction(gallery_data)
    # ImageEditor requires {"background": image, "layers": [], "composite": image} format
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


def process_color_correction_with_source(
    image, target_editor, source_editor,
    correction_mode, strength,
    mask_source, tab1_mask,
    a_factor, b_factor, l_factor,
    enable_scattered_mode,
    exclusion_editor, enable_exclusion, tab1_full_mask
):
    """Wrapper for color correction with mask source selection and exclusion mask."""
    import cv2
    import numpy as np

    # Determine which mask to use
    use_scattered_mode = False

    if mask_source == "Tab 1の選択マスクを使用":
        # Use Tab 1's selection mask
        if tab1_mask is None:
            return image, "Tab 1のマスクが設定されていません。「Tab 1の選択マスクを取得」ボタンを押してください"

        # Create a dummy editor data with the Tab 1 mask as the layer
        target_editor_data = {
            "layers": [tab1_mask],
            "background": image
        }
        # 散らばった領域モードはチェックボックスで制御
        use_scattered_mode = enable_scattered_mode
    else:
        # Use manual painting (existing behavior)
        target_editor_data = target_editor

    # ========== 除外マスクの処理 ==========
    exclusion_mask = None
    if enable_exclusion:
        # 除外マスクを取得（エディタから）
        if exclusion_editor is not None:
            try:
                if isinstance(exclusion_editor, dict) and 'layers' in exclusion_editor:
                    if len(exclusion_editor['layers']) > 0:
                        layer = exclusion_editor['layers'][0]
                        if isinstance(layer, np.ndarray) and len(layer.shape) == 3:
                            # 青チャンネルを使用（青で塗る想定）
                            gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                            exclusion_mask = (gray > 30).astype(np.uint8) * 255
            except Exception as e:
                print(f"除外マスク抽出エラー: {e}")

        # Tab 1の全検出マスクも除外に追加（オプション）
        if tab1_full_mask is not None:
            full_mask = tab1_full_mask.copy()
            if len(full_mask.shape) == 3:
                full_mask = cv2.cvtColor(full_mask, cv2.COLOR_RGB2GRAY)
            full_mask = (full_mask > 30).astype(np.uint8) * 255

            if exclusion_mask is not None:
                # 両方を合成
                exclusion_mask = cv2.bitwise_or(exclusion_mask, full_mask)
            else:
                exclusion_mask = full_mask

    # 除外マスク付きの色補正を実行
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
            use_scattered_mode=use_scattered_mode
        )
    else:
        # 従来の処理
        return pipeline.process_color_correction(
            image=image,
            target_editor_data=target_editor_data,
            source_editor_data=source_editor,
            correction_mode=correction_mode,
            strength=strength,
            a_adjustment_factor=a_factor,
            b_adjustment_factor=b_factor,
            l_adjustment_factor=l_factor,
            use_scattered_mode=use_scattered_mode
        )


def process_color_correction_with_exclusion(
    image, target_editor_data, source_editor_data,
    exclusion_mask, correction_mode, strength,
    a_factor, b_factor, l_factor, use_scattered_mode
):
    """除外マスクを考慮した色補正処理

    改善版:
    - 除外マスク = 参照サンプリングから除外 AND 色補正の対象にも含める
    - 対象マスク（髭検出）+ 除外マスク（青髭領域）= 合成して色補正
    """
    import cv2
    import numpy as np

    if image is None:
        return None, "画像がありません"

    # 対象マスクを取得
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

    # 散らばった領域の場合、マスクを膨張
    if use_scattered_mode:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        target_mask = cv2.dilate(target_mask, kernel, iterations=1)

    # ========== 色補正対象マスク = 対象マスク + 除外マスク ==========
    # 除外マスクも色補正の対象に含める（点々の隙間も補正）
    correction_mask = target_mask.copy()
    if exclusion_mask is not None and np.any(exclusion_mask > 0):
        correction_mask = cv2.bitwise_or(correction_mask, exclusion_mask)
        print(f"[除外マスク統合] 対象: {np.sum(target_mask > 0)} + 除外: {np.sum(exclusion_mask > 0)} = 合計: {np.sum(correction_mask > 0)} pixels")

    # BGR変換
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ========== 除外マスクを考慮した周辺肌色検出 ==========
    # 参照サンプリングからは除外マスク領域を除外
    skin_mask = detect_surrounding_skin_with_exclusion(
        image_bgr, correction_mask, exclusion_mask,
        dilation_size=50, erosion_size=15
    )

    if not np.any(skin_mask > 0):
        # 広い範囲で再試行
        skin_mask = detect_surrounding_skin_with_exclusion(
            image_bgr, correction_mask, exclusion_mask,
            dilation_size=100, erosion_size=20
        )

    if not np.any(skin_mask > 0):
        return image, "除外領域を考慮すると、参照できる肌色がありません。除外領域を狭めてください。"

    # LAB色空間で補正
    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # 参照領域から目標色を取得
    skin_pixels_lab = lab_image[skin_mask > 0]
    target_l = np.percentile(skin_pixels_lab[:, 0], 75)  # 明るめ
    target_a = np.median(skin_pixels_lab[:, 1])
    target_b = np.median(skin_pixels_lab[:, 2])

    print(f"[除外マスク適用] 参照肌色: L={target_l:.1f}, a={target_a:.1f}, b={target_b:.1f}")
    print(f"参照ピクセル数: {len(skin_pixels_lab)} (除外後)")

    # ========== 合成マスク全体を補正（点々 + 隙間） ==========
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

    # クリップしてBGRに戻す
    lab_image = np.clip(lab_image, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # バイラテラルフィルターで美肌（合成マスク全体に適用）
    smoothed = cv2.bilateralFilter(result_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    smooth_strength = min(strength * 0.7, 0.8)

    # マスク領域のみスムージングを適用
    mask_float = correction_mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_float] * 3, axis=-1)
    result_bgr = (
        result_bgr.astype(np.float32) * (1.0 - smooth_strength * mask_3ch) +
        smoothed.astype(np.float32) * (smooth_strength * mask_3ch)
    ).astype(np.uint8)

    # エッジぼかしでブレンド（合成マスクで）
    edge_blur = 15
    blur_mask = cv2.GaussianBlur(correction_mask.astype(np.float32), (edge_blur * 2 + 1, edge_blur * 2 + 1), 0) / 255.0
    blur_mask_3ch = np.stack([blur_mask] * 3, axis=-1)

    result_bgr = (
        image_bgr.astype(np.float32) * (1.0 - blur_mask_3ch) +
        result_bgr.astype(np.float32) * blur_mask_3ch
    ).astype(np.uint8)

    # RGB変換して返す
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    corr_pixels = np.sum(correction_mask > 0)
    status = f"除外マスク統合補正完了 | 補正対象: {corr_pixels} pixels | 参照: {len(skin_pixels_lab)} pixels"
    return result_rgb, status


def detect_surrounding_skin_with_exclusion(
    image, target_mask, exclusion_mask,
    dilation_size=50, erosion_size=15
):
    """除外マスクを考慮した周辺肌色検出"""
    import cv2
    import numpy as np

    h, w = image.shape[:2]

    # 対象領域を膨張させて周辺領域を作成
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size * 2 + 1, dilation_size * 2 + 1)
    )
    dilated = cv2.dilate(target_mask, kernel_dilate, iterations=1)

    # 対象領域を少し膨張させて除外
    kernel_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_size * 2 + 1, erosion_size * 2 + 1)
    )
    excluded_target = cv2.dilate(target_mask, kernel_erode, iterations=1)

    # 周辺領域 = 膨張 - 対象
    surrounding = cv2.subtract(dilated, excluded_target)

    # ========== 除外マスクを適用 ==========
    # 除外マスクも膨張させて確実に除外
    if exclusion_mask is not None and np.any(exclusion_mask > 0):
        kernel_exclusion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        exclusion_dilated = cv2.dilate(exclusion_mask, kernel_exclusion, iterations=1)
        surrounding = cv2.subtract(surrounding, exclusion_dilated)
        print(f"除外マスク適用: {np.sum(exclusion_mask > 0)} pixels を除外")

    # HSV色空間で肌色をフィルタリング
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 200, 255], dtype=np.uint8)
    skin_mask_hsv = cv2.inRange(hsv, lower_skin, upper_skin)

    # YCrCb空間でも肌色検出
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower_skin_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)

    # 両方を組み合わせ
    skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)

    # 周辺領域と肌色マスクのAND
    result_mask = cv2.bitwise_and(surrounding, skin_mask)

    return result_mask


# =============================================================================
# Gradio UI
# =============================================================================

def create_app():
    """Create Gradio application with modular backend."""

    with gr.Blocks(
        title="髭検出・修復アプリ v4",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown("""
        # 髭検出・修復アプリケーション v4

        **ワークフロー:** Tab 1（髭検出）→ Tab 2（Inpainting）→ Tab 3（色調補正）

        **機能:**
        - **Tab 1: 髭検出** - ルールベース / Grounded SAM で髭を1本ずつ検出
        - **Tab 2: LaMa Inpainting** - 高品質な髭除去・段階的な薄め
        - **Tab 3: 色調補正** - 青髭を素肌に近づける（LAB色空間ベース）

        **v4 の改善点:**
        - モジュール化されたコードベース（beard_inpainting_modules/）
        - 青髭補正機能を追加
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
                4. Remove % で削除対象を選択
                5. 「マスクを Tab 2 に転送」
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # 画像入力
                        image_input = gr.Image(
                            label="画像をアップロード",
                            type="numpy",
                            sources=["upload"]
                        )

                        # 選択形状モード
                        selection_shape_radio = gr.Radio(
                            choices=["矩形", "自由形状"],
                            value="矩形",
                            label="選択形状",
                            info="矩形: 塗りつぶし / 自由形状: 線で囲む"
                        )

                        # 描画エディタ
                        rect_editor = gr.ImageEditor(
                            label="髭の範囲を選択（矩形で塗りつぶし or 自由形状で線を囲む）",
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

                        # 検出モード選択
                        detection_mode_radio = gr.Radio(
                            choices=["ルールベース（1本ずつ検出）", "Grounded SAM"],
                            value="ルールベース（1本ずつ検出）",
                            label="検出モード",
                            info="ルールベースは髭を1本ずつ高精度に検出します（推奨）"
                        )

                        # 前処理オプション
                        enable_denoise_checkbox = gr.Checkbox(
                            label="前処理: ノイズ除去を有効化",
                            value=False,
                            info="毛穴などの細かいノイズを髭と誤検出する場合にON（バイラテラルフィルター）"
                        )

                        with gr.Accordion("ルールベース パラメータ", open=True):
                            threshold_slider = gr.Slider(
                                minimum=20, maximum=200, value=80, step=5,
                                label="二値化閾値",
                                info="小さいほど薄い髭も検出（暗い=髭）"
                            )
                            min_area_slider = gr.Slider(
                                minimum=1, maximum=500, value=10, step=1,
                                label="最小面積",
                                info="検出領域の最小ピクセル数"
                            )
                            max_area_slider = gr.Slider(
                                minimum=100, maximum=10000, value=5000, step=100,
                                label="最大面積",
                                info="検出領域の最大ピクセル数"
                            )

                        with gr.Accordion("Grounded SAM パラメータ", open=False):
                            text_prompt_input = gr.Textbox(
                                label="検出プロンプト",
                                value="beard. facial hair. stubble.",
                                info="髭を表すテキスト（ピリオド区切り）"
                            )
                            box_threshold_slider = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                                label="Box Threshold",
                                info="ボックス検出の信頼度閾値"
                            )
                            text_threshold_slider = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.20, step=0.05,
                                label="Text Threshold",
                                info="テキストマッチングの閾値"
                            )

                        detect_btn = gr.Button("髭を検出", variant="primary")

                    with gr.Column(scale=1):
                        detect_result = gr.Image(
                            label="検出結果（各髭を色分け表示）",
                            type="numpy",
                            interactive=False
                        )
                        detect_mask = gr.Image(
                            label="検出マスク",
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
                            label="Remove %（削除割合）"
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
                            label="選択結果（赤=削除対象, 他の色=保持）- クリックで拡大",
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

                # イベント: 画像アップロード時に rect_editor へ自動転送
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

                # イベント: 髭検出
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

                # イベント: 選択更新
                update_selection_btn.click(
                    fn=update_selection_preview,
                    inputs=[removal_slider, selection_mode_radio, gr.State(False)],
                    outputs=[selection_result, selection_mask, selection_status]
                )

                # イベント: 新しいシード
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
                1. Tab 1 からマスクを転送（または直接アップロード）
                2. インペインティング手法を選択
                3. 薄め具合を選択
                4. 「髭薄めを実行」をクリック

                **手法の違い:**
                - **Simple LaMa**: ディープラーニングベース（高品質だがやや遅い）
                - **OpenCV Telea**: Fast Marching Method（高速・シャープ）
                - **OpenCV Navier-Stokes**: 流体力学ベース（滑らかな補間）
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

                        thinning_checkboxes = gr.CheckboxGroup(
                            choices=[30, 50, 70, 100],
                            value=[30, 50, 70, 100],
                            label="薄め具合（%）"
                        )

                        inpaint_method_radio = gr.Radio(
                            choices=["Simple LaMa", "OpenCV Telea", "OpenCV Navier-Stokes"],
                            value="Simple LaMa",
                            label="インペインティング手法",
                            info="LaMa: 高品質 / OpenCV: 高速"
                        )

                        opencv_radius_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=3,
                            step=1,
                            label="OpenCV 補間半径",
                            info="OpenCV使用時のみ有効（大きいほど広範囲を補間）",
                            visible=False
                        )

                        inpaint_btn = gr.Button(
                            "髭薄めを実行",
                            variant="primary",
                            size="lg"
                        )

                        # OpenCV選択時に半径スライダーを表示
                        def toggle_opencv_radius(method):
                            return gr.update(visible=method in ["OpenCV Telea", "OpenCV Navier-Stokes"])

                        inpaint_method_radio.change(
                            fn=toggle_opencv_radius,
                            inputs=[inpaint_method_radio],
                            outputs=[opencv_radius_slider]
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

                # イベント: 転送
                transfer_btn.click(
                    fn=transfer_to_inpainting,
                    inputs=[selection_mask],
                    outputs=[inpaint_image, inpaint_mask]
                )

                # Note: inpaint_btn.click is registered after Tab 3 (see below)

            # =================================================================
            # Tab 3: 色調補正（青髭補正）
            # =================================================================
            with gr.TabItem("3. 色調補正"):
                gr.Markdown("""
                ### 青髭補正ツール
                髭を剃った後の青髭を素肌に近づけます。

                **モード:**
                - **青み除去（推奨）**: 青っぽさを自動で除去。スポイト不要
                - **色味転送**: 頬などの肌色を指定して転送
                - **自動補正**: 周辺の肌色を自動サンプリング
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # 画像入力
                        color_image = gr.Image(
                            label="画像（Tab 2の結果または直接アップロード）",
                            type="numpy",
                            sources=["upload"]
                        )

                        # Tab 2から転送ボタン
                        transfer_from_inpaint_btn = gr.Button(
                            "← Tab 2の結果を取得",
                            variant="secondary"
                        )

                        # モード選択
                        correction_mode_radio = gr.Radio(
                            choices=["青み除去（推奨）", "色味転送", "自動補正"],
                            value="青み除去（推奨）",
                            label="補正モード",
                            info="青み除去がおすすめです"
                        )

                        # ========== マスク指定方法 ==========
                        gr.Markdown("#### 対象領域の指定方法")
                        mask_source_radio = gr.Radio(
                            choices=["手動で塗る", "Tab 1の選択マスクを使用"],
                            value="手動で塗る",
                            label="マスク指定方法",
                            info="Tab 1で選択した領域を再利用できます"
                        )

                        # Tab 1のマスクを転送するボタン
                        transfer_mask_btn = gr.Button(
                            "← Tab 1の選択マスクを取得",
                            variant="secondary"
                        )

                        # 散らばった領域モードのチェックボックス
                        scattered_mode_checkbox = gr.Checkbox(
                            label="散らばった領域モードを有効化",
                            value=True,
                            info="Tab 1マスク使用時、補正効果を強化（エッジぼかし減・LAB係数増）"
                        )

                        # Tab 1のマスク表示
                        tab1_mask_preview = gr.Image(
                            label="Tab 1のマスク（プレビュー）",
                            type="numpy",
                            interactive=False
                        )

                        # 対象領域エディタ（手動塗り用）
                        gr.Markdown("#### 手動塗り（ブラシサイズはエディタ内で調整可能）")
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

                        # スポイト領域エディタ（色味転送用）
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

                        # ========== 除外マスク設定（オプション機能） ==========
                        with gr.Accordion("除外マスク設定（青髭領域を参照から除外）", open=False):
                            gr.Markdown("""
                            **周辺の青みを除外**: 補正時に参照として使用する周辺肌色から、
                            青みを含む領域を除外できます。
                            - 青髭が周囲に広がっている場合に有効
                            - Tab 1で検出した全マスクを除外に追加することも可能
                            """)

                            enable_exclusion_checkbox = gr.Checkbox(
                                label="除外マスクを有効化",
                                value=False,
                                info="チェックを入れると、指定した領域を周辺参照から除外"
                            )

                            exclusion_editor = gr.ImageEditor(
                                label="除外領域（青髭が広がっている部分を塗る）",
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
                                label="Tab 1の全検出マスク（プレビュー）",
                                type="numpy",
                                interactive=False
                            )

                        # 強度スライダー
                        color_strength_slider = gr.Slider(
                            minimum=0, maximum=1, value=0.8, step=0.05,
                            label="補正強度",
                            info="0.8程度がおすすめ。1.0だと不自然になる場合あり"
                        )

                        # LABパラメータ調整（詳細設定）
                        with gr.Accordion("LABパラメータ調整（上級者向け）", open=False):
                            gr.Markdown("""
                            **最適な肌色補正**の設定です。通常はデフォルトのままで問題ありません。
                            - **a* (赤-緑軸)**: 小さいほどオレンジ化を防ぐ
                            - **b* (青-黄軸)**: 青み除去の強さ
                            - **L (明度)**: 明るさの調整
                            """)
                            a_factor_slider = gr.Slider(
                                minimum=0, maximum=1, value=0.3, step=0.05,
                                label="a* 調整係数（赤-緑軸）",
                                info="デフォルト 0.3（30%）。小さいほど赤み追加を抑制"
                            )
                            b_factor_slider = gr.Slider(
                                minimum=0, maximum=1, value=0.6, step=0.05,
                                label="b* 調整係数（青-黄軸）",
                                info="デフォルト 0.6（60%）。青み除去の主要パラメータ"
                            )
                            l_factor_slider = gr.Slider(
                                minimum=0, maximum=1, value=0.5, step=0.05,
                                label="L 調整係数（明度）",
                                info="デフォルト 0.5（50%）。大きいほど明るく補正"
                            )

                        # 適用ボタン
                        apply_color_btn = gr.Button(
                            "色調補正を適用",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        # 結果表示
                        color_result = gr.Image(
                            label="補正結果",
                            type="numpy",
                            interactive=False
                        )
                        color_status = gr.Textbox(
                            label="ステータス",
                            interactive=False
                        )

                # イベント: Tab 2から転送 (color_image + editors に転送)
                transfer_from_inpaint_btn.click(
                    fn=transfer_from_inpainting,
                    inputs=[inpaint_gallery],
                    outputs=[color_image, target_color_editor, source_color_editor, exclusion_editor]
                )

                # イベント: Tab 1のマスクを転送
                transfer_mask_btn.click(
                    fn=transfer_tab1_mask,
                    inputs=[selection_mask],
                    outputs=[tab1_mask_preview, color_status]
                )

                # イベント: Tab 1の全検出マスクを除外に転送
                transfer_full_mask_btn.click(
                    fn=transfer_tab1_full_mask,
                    inputs=[detect_mask],
                    outputs=[tab1_full_mask_preview, color_status]
                )

                # イベント: 色調補正適用
                apply_color_btn.click(
                    fn=process_color_correction_with_source,
                    inputs=[
                        color_image, target_color_editor, source_color_editor,
                        correction_mode_radio, color_strength_slider,
                        mask_source_radio, tab1_mask_preview,
                        a_factor_slider, b_factor_slider, l_factor_slider,
                        scattered_mode_checkbox,
                        exclusion_editor, enable_exclusion_checkbox, tab1_full_mask_preview
                    ],
                    outputs=[color_result, color_status]
                )

        # =================================================================
        # Cross-Tab Event: Inpainting -> Tab 3 auto-transfer
        # (Defined here because color_image is in Tab 3)
        # =================================================================
        inpaint_btn.click(
            fn=process_lama_inpainting,
            inputs=[inpaint_image, inpaint_mask, thinning_checkboxes, inpaint_method_radio, opencv_radius_slider],
            outputs=[inpaint_gallery, inpaint_status, color_image, target_color_editor, source_color_editor, exclusion_editor]
        )

        # 説明パネル
        with gr.Accordion("技術情報", open=False):
            gr.Markdown("""
            ## 使用技術

            ### Tab 1: 髭検出
            - **Grounding DINO**: テキストプロンプトからオブジェクト検出
            - **SAM**: セグメンテーションマスク生成
            - **ルールベース**: 閾値処理＋輪郭検出
            - **選択モード**: ランダム / 面積大 / 面積小 / 信頼度順

            ### Tab 2: LaMa Inpainting
            - **SimpleLama**: フーリエ畳み込みベースの高品質画像修復
            - **アルファブレンディング**: 段階的な髭薄め効果

            ### Tab 3: 色調補正
            - **LAB色空間**: 明度を保持しながら色味のみを調整
            - **青み除去**: b*チャンネル（青-黄軸）を補正
            - **色味転送**: スポイト領域の色味をLUTで転送
            - **自動補正**: 周辺の肌色を自動サンプリング

            ### v4 アーキテクチャ
            ```
            beard_inpainting_modules/
            ├── __init__.py         # パッケージエクスポート
            ├── image_handler.py    # 画像I/O
            ├── region_selector.py  # 矩形選択
            ├── beard_detector.py   # 検出ロジック
            ├── highlighter.py      # 選択・表示
            ├── inpainter.py        # LaMa wrapper
            ├── color_corrector.py  # 色調補正
            └── pipeline.py         # オーケストレーター
            ```

            ### 必要なパッケージ
            ```bash
            pip install gradio opencv-python numpy pillow torch
            pip install segment-anything groundingdino-py
            pip install simple-lama-inpainting
            ```

            ### チェックポイントのダウンロード
            - SAM: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
            - Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
            """)

    return app


def main():
    """Main entry point."""
    app = create_app()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7864  # Different port from v3 (7863)
    )


if __name__ == "__main__":
    main()
