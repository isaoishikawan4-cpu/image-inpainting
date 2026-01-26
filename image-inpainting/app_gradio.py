#!/usr/bin/env python3
"""
Gradio版 髭検出・修復アプリケーション

2つの機能を分離して軽量化:
- Tab 1: 髭検出（マスク生成）
- Tab 2: LaMa Inpainting（画像修復）

使用方法:
    python app_gradio.py
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, List
import random

# LaMa Inpainting モジュール
try:
    from core.inpainting import InpaintingEngine, BeardThinningProcessor
    from core.image_utils import (
        resize_image_if_needed,
        merge_masks,
        convert_to_binary_mask,
        numpy_to_pil,
        pil_to_numpy
    )
    LAMA_AVAILABLE = True
    print("LaMa Inpainting: 利用可能")
except ImportError as e:
    LAMA_AVAILABLE = False
    print(f"警告: LaMa Inpainting が利用できません: {e}")

import config


# =============================================================================
# Tab 1: 髭検出（マスク生成）
# =============================================================================

class BeardDetector:
    """髭検出器（ルールベース）"""

    def __init__(self):
        self.min_area = 10
        self.max_area = 50000
        self.threshold_value = 80

    def detect_in_region(
        self,
        image: np.ndarray,
        region_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, int]:
        """
        指定領域内で髭を検出

        Args:
            image: BGR画像
            region_mask: 検出対象領域のマスク（None=全体）

        Returns:
            (検出マスク, 検出数)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        _, binary = cv2.threshold(
            blurred, self.threshold_value, 255, cv2.THRESH_BINARY_INV
        )
        mask = cv2.bitwise_and(adaptive, binary)

        if region_mask is not None:
            mask = cv2.bitwise_and(mask, region_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                cv2.drawContours(result_mask, [contour], -1, 255, -1)
                count += 1

        return result_mask, count


# グローバルインスタンス
beard_detector = BeardDetector()
detected_regions: List[dict] = []
random_seed = 42


def process_detection(
    image: np.ndarray,
    editor_data: dict,
    min_area: int,
    max_area: int,
    threshold: int
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    髭検出を実行

    Args:
        image: 入力画像
        editor_data: ImageEditorからのデータ
        min_area: 最小面積
        max_area: 最大面積
        threshold: 二値化閾値

    Returns:
        (表示用画像, マスク画像, ステータス)
    """
    global detected_regions

    if image is None:
        return None, None, "画像をアップロードしてください"

    # パラメータ更新
    beard_detector.min_area = min_area
    beard_detector.max_area = max_area
    beard_detector.threshold_value = threshold

    # BGR変換
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # エディタからマスク領域を取得
    region_mask = None
    if editor_data is not None:
        try:
            # ImageEditorの構造を解析
            if isinstance(editor_data, dict):
                if 'layers' in editor_data and len(editor_data['layers']) > 0:
                    layer = editor_data['layers'][0]
                    if isinstance(layer, np.ndarray):
                        if len(layer.shape) == 3 and layer.shape[2] >= 3:
                            gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                            region_mask = (gray > 128).astype(np.uint8) * 255
                elif 'composite' in editor_data:
                    composite = editor_data['composite']
                    if isinstance(composite, np.ndarray) and len(composite.shape) == 3:
                        if composite.shape[2] == 4:
                            # アルファチャンネルをマスクとして使用
                            region_mask = composite[:, :, 3]
        except Exception as e:
            print(f"マスク解析エラー: {e}")

    # 髭検出
    mask, count = beard_detector.detect_in_region(image_bgr, region_mask)

    # 検出結果を保存
    detected_regions = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        region_mask_single = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(region_mask_single, [contour], -1, 255, -1)
        M = cv2.moments(region_mask_single)
        cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else 0
        detected_regions.append({
            'mask': region_mask_single,
            'area': area,
            'centroid': (cx, cy)
        })

    # 表示用画像を作成
    display = image_bgr.copy()
    overlay = display.copy()
    display[mask > 0] = (0, 255, 0)  # 緑でハイライト
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

    status = f"検出完了: {count} 個の髭領域を検出しました"
    return display_rgb, mask, status


def update_removal_mask(
    image: np.ndarray,
    removal_percentage: int,
    selection_mode: str,
    new_seed: bool
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    削除対象マスクを更新

    Args:
        image: 入力画像
        removal_percentage: 削除割合 (%)
        selection_mode: 選択モード
        new_seed: 新しいシードを使用するか

    Returns:
        (表示用画像, マスク画像, ステータス)
    """
    global random_seed

    if image is None or not detected_regions:
        return None, None, "先に髭検出を実行してください"

    if new_seed:
        random_seed = random.randint(0, 10000)

    # BGR変換
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    total_count = len(detected_regions)
    target_count = int(total_count * removal_percentage / 100)

    # 選択モードに応じてインデックスを決定
    if selection_mode == "ランダム":
        random.seed(random_seed)
        all_indices = list(range(total_count))
        random.shuffle(all_indices)
        active_indices = sorted(all_indices[:target_count])
    elif selection_mode == "面積大":
        sorted_indices = sorted(
            range(total_count),
            key=lambda i: detected_regions[i]['area'],
            reverse=True
        )
        active_indices = sorted(sorted_indices[:target_count])
    elif selection_mode == "面積小":
        sorted_indices = sorted(
            range(total_count),
            key=lambda i: detected_regions[i]['area']
        )
        active_indices = sorted(sorted_indices[:target_count])
    else:
        active_indices = list(range(target_count))

    # マスクを生成
    h, w = image_bgr.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in active_indices:
        combined_mask = cv2.bitwise_or(combined_mask, detected_regions[idx]['mask'])

    # 膨張処理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # 表示用画像を作成
    display = image_bgr.copy()
    for i, region in enumerate(detected_regions):
        mask = region['mask']
        if i in active_indices:
            # 削除対象: 赤
            overlay = display.copy()
            display[mask > 0] = (0, 0, 255)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        else:
            # 非削除: 緑
            overlay = display.copy()
            display[mask > 0] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

    status = f"選択: {len(active_indices)}/{total_count} 個 ({removal_percentage}%) | シード: {random_seed}"
    return display_rgb, combined_mask, status


# =============================================================================
# Tab 2: LaMa Inpainting（画像修復）
# =============================================================================

# プロセッサのグローバルインスタンス（遅延初期化）
_processor: Optional[BeardThinningProcessor] = None


def get_processor() -> Optional[BeardThinningProcessor]:
    """プロセッサを取得（遅延初期化）"""
    global _processor
    if _processor is None and LAMA_AVAILABLE:
        try:
            print("LaMa プロセッサを初期化中...")
            _processor = BeardThinningProcessor()
            print("LaMa プロセッサ: 初期化成功")
        except Exception as e:
            print(f"LaMa プロセッサ初期化エラー: {e}")
            return None
    return _processor


def process_inpainting(
    image: Image.Image,
    mask: np.ndarray,
    thinning_levels: List[int],
    progress=gr.Progress()
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    """
    LaMa Inpainting を実行

    Args:
        image: 入力画像 (PIL)
        mask: マスク画像
        thinning_levels: 薄め具合レベル
        progress: Gradio プログレス

    Returns:
        (結果画像リスト, ステータス)
    """
    if image is None:
        return [], "画像をアップロードしてください"

    if mask is None or np.max(mask) == 0:
        return [], "マスクを指定してください（Tab 1で生成するか、直接描画）"

    if not thinning_levels:
        return [], "少なくとも1つの薄め具合を選択してください"

    processor = get_processor()
    if processor is None:
        return [], "LaMa が利用できません。simple-lama-inpainting をインストールしてください。"

    try:
        # 画像をリサイズ
        image = resize_image_if_needed(image)

        # マスクを二値化
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        binary_mask = convert_to_binary_mask(mask)

        # 処理実行
        def progress_callback(current, total, level):
            progress((current / total), desc=f"処理中: {level}")

        results, messages = processor.process_thinning(
            image,
            binary_mask,
            thinning_levels,
            progress_callback=progress_callback
        )

        if not results:
            return [], "処理に失敗しました: " + "\n".join(messages)

        # 結果をギャラリー形式に変換
        gallery_images = [(image, "オリジナル (0% 薄め)")]
        for level in sorted(results.keys()):
            caption = f"{level}% 薄め" if level < 100 else "完全除去 (100%)"
            gallery_images.append((results[level], caption))

        status = f"完了！ {len(results)}段階の髭薄め画像を生成しました"
        return gallery_images, status

    except Exception as e:
        return [], f"エラーが発生しました: {str(e)}"


def process_inpainting_from_editor(
    editor_data: dict,
    thinning_levels: List[int],
    progress=gr.Progress()
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    """
    ImageEditorから直接 Inpainting を実行

    Args:
        editor_data: ImageEditorからのデータ
        thinning_levels: 薄め具合レベル
        progress: Gradio プログレス

    Returns:
        (結果画像リスト, ステータス)
    """
    if editor_data is None:
        return [], "画像をアップロードしてマスクを描画してください"

    try:
        # ImageEditorから画像とマスクを抽出
        background = None
        mask = None

        if isinstance(editor_data, dict):
            if 'background' in editor_data:
                background = editor_data['background']
            if 'layers' in editor_data and len(editor_data['layers']) > 0:
                layer = editor_data['layers'][0]
                if isinstance(layer, np.ndarray):
                    if len(layer.shape) == 3 and layer.shape[2] >= 3:
                        gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                        mask = (gray > 128).astype(np.uint8) * 255
            if 'composite' in editor_data and mask is None:
                composite = editor_data['composite']
                if isinstance(composite, np.ndarray) and composite.shape[2] == 4:
                    mask = composite[:, :, 3]

        if background is None:
            return [], "画像が見つかりません"

        if isinstance(background, np.ndarray):
            image = Image.fromarray(background)
        else:
            image = background

        return process_inpainting(image, mask, thinning_levels, progress)

    except Exception as e:
        return [], f"エラー: {str(e)}"


def transfer_mask_to_inpainting(
    original_image: np.ndarray,
    mask: np.ndarray
) -> Tuple[Image.Image, np.ndarray]:
    """
    Tab 1 からマスクを Tab 2 に転送

    Args:
        original_image: 元画像
        mask: マスク画像

    Returns:
        (PIL画像, マスク)
    """
    if original_image is None:
        return None, None

    if len(original_image.shape) == 2:
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 4:
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
    elif original_image.shape[2] == 3:
        image_rgb = original_image
    else:
        image_rgb = original_image

    image_pil = Image.fromarray(image_rgb)

    return image_pil, mask


# =============================================================================
# Gradio UI
# =============================================================================

def create_app():
    """Gradio アプリを作成"""

    with gr.Blocks(
        title="髭検出・修復アプリ",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown("""
        # 髭検出・修復アプリケーション

        2つの機能を分離して軽量化しました:
        - **Tab 1**: 髭検出（マスク生成）- ルールベースの画像処理
        - **Tab 2**: LaMa Inpainting（画像修復）- AIによる高品質修復
        """)

        # 共有状態
        shared_image = gr.State(None)
        shared_mask = gr.State(None)

        with gr.Tabs():
            # =================================================================
            # Tab 1: 髭検出
            # =================================================================
            with gr.TabItem("1. 髭検出（マスク生成）"):
                gr.Markdown("""
                ### 使い方
                1. 画像をアップロード
                2. （オプション）検出領域を白で描画して限定
                3. 「髭検出」ボタンをクリック
                4. Remove % スライダーで削除対象を選択
                5. 「マスクを Tab 2 に転送」で修復へ
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        detect_image_input = gr.Image(
                            label="画像をアップロード",
                            type="numpy",
                            sources=["upload"]
                        )

                        detect_editor = gr.ImageEditor(
                            label="検出領域を描画（オプション：白で塗りつぶし）",
                            type="numpy",
                            brush=gr.Brush(
                                default_size=20,
                                colors=["white"],
                                default_color="white"
                            ),
                            eraser=gr.Eraser(default_size=20)
                        )

                        with gr.Accordion("検出パラメータ", open=False):
                            min_area_slider = gr.Slider(
                                minimum=1, maximum=500, value=10,
                                label="最小面積", step=1
                            )
                            max_area_slider = gr.Slider(
                                minimum=1000, maximum=100000, value=50000,
                                label="最大面積", step=1000
                            )
                            threshold_slider = gr.Slider(
                                minimum=50, maximum=150, value=80,
                                label="二値化閾値", step=1
                            )

                        detect_btn = gr.Button("髭検出", variant="primary")

                    with gr.Column(scale=1):
                        detect_result_image = gr.Image(
                            label="検出結果（緑=検出された髭）",
                            type="numpy",
                            interactive=False
                        )

                        detect_mask_output = gr.Image(
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
                            minimum=0, maximum=100, value=50,
                            label="Remove %（削除割合）", step=1
                        )
                        selection_mode = gr.Radio(
                            choices=["ランダム", "面積大", "面積小"],
                            value="ランダム",
                            label="選択モード"
                        )
                        new_seed_btn = gr.Button("新しいランダムシード")
                        update_selection_btn = gr.Button("選択を更新", variant="secondary")

                    with gr.Column(scale=1):
                        selection_result_image = gr.Image(
                            label="選択結果（赤=削除対象, 緑=保持）",
                            type="numpy",
                            interactive=False
                        )
                        selection_mask_output = gr.Image(
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

                # イベント接続
                detect_btn.click(
                    fn=process_detection,
                    inputs=[
                        detect_image_input,
                        detect_editor,
                        min_area_slider,
                        max_area_slider,
                        threshold_slider
                    ],
                    outputs=[detect_result_image, detect_mask_output, detect_status]
                )

                update_selection_btn.click(
                    fn=update_removal_mask,
                    inputs=[
                        detect_image_input,
                        removal_slider,
                        selection_mode,
                        gr.State(False)
                    ],
                    outputs=[selection_result_image, selection_mask_output, selection_status]
                )

                new_seed_btn.click(
                    fn=update_removal_mask,
                    inputs=[
                        detect_image_input,
                        removal_slider,
                        selection_mode,
                        gr.State(True)
                    ],
                    outputs=[selection_result_image, selection_mask_output, selection_status]
                )

            # =================================================================
            # Tab 2: LaMa Inpainting
            # =================================================================
            with gr.TabItem("2. LaMa Inpainting（画像修復）"):
                gr.Markdown("""
                ### 使い方
                1. Tab 1 からマスクを転送、または直接画像をアップロードしてマスクを描画
                2. 薄め具合を選択
                3. 「髭薄めを実行」ボタンをクリック
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        inpaint_image_input = gr.Image(
                            label="元画像",
                            type="pil",
                            sources=["upload"]
                        )

                        inpaint_mask_input = gr.Image(
                            label="マスク画像（白=修復領域）",
                            type="numpy",
                            sources=["upload"]
                        )

                        gr.Markdown("**または**")

                        inpaint_editor = gr.ImageEditor(
                            label="直接マスクを描画",
                            type="numpy",
                            brush=gr.Brush(
                                default_size=config.BRUSH_RADIUS_DEFAULT,
                                colors=["white"],
                                default_color="white"
                            ),
                            eraser=gr.Eraser(default_size=config.BRUSH_RADIUS_DEFAULT)
                        )

                        thinning_checkboxes = gr.CheckboxGroup(
                            choices=[30, 50, 70, 100],
                            value=[30, 50, 70, 100],
                            label="薄め具合（%）"
                        )

                        inpaint_btn = gr.Button(
                            "髭薄めを実行",
                            variant="primary",
                            size="lg"
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

                # Tab 1 から転送
                transfer_btn.click(
                    fn=transfer_mask_to_inpainting,
                    inputs=[detect_image_input, selection_mask_output],
                    outputs=[inpaint_image_input, inpaint_mask_input]
                )

                # Inpainting 実行（画像 + マスク）
                inpaint_btn.click(
                    fn=process_inpainting,
                    inputs=[
                        inpaint_image_input,
                        inpaint_mask_input,
                        thinning_checkboxes
                    ],
                    outputs=[inpaint_gallery, inpaint_status]
                )

        # 説明パネル
        with gr.Accordion("使い方・技術説明", open=False):
            gr.Markdown("""
            ## 処理の仕組み

            ### Tab 1: 髭検出
            - **ルールベース処理**: OpenCV を使用した画像処理
            - CLAHE（コントラスト強調）→ 適応閾値化 → 形態学処理 → 輪郭検出
            - GPUを使用しないため高速

            ### Tab 2: LaMa Inpainting
            - **AI処理**: SimpleLama モデルによる高品質な画像修復
            - フーリエ畳み込みを使用した最新のInpaintingアルゴリズム
            - GPU使用時は高速、CPU使用時はやや時間がかかる

            ### パフォーマンス
            | 処理 | CPU | GPU |
            |-----|-----|-----|
            | 髭検出 | ~100ms | - |
            | LaMa Inpainting | ~5秒 | ~500ms |

            ### 必要なパッケージ
            ```bash
            pip install gradio opencv-python numpy pillow
            pip install simple-lama-inpainting  # LaMa 用
            ```
            """)

    return app


def main():
    """メインエントリーポイント"""
    app = create_app()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7861
    )


if __name__ == "__main__":
    main()
