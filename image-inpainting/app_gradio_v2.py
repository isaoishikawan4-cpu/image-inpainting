#!/usr/bin/env python3
"""
Gradio版 髭検出・修復アプリケーション v2

改良点:
- Tab 1: SAM による髭検出（矩形範囲指定でボックス入力）
- Tab 2: LaMa Inpainting（100% 完全除去のみ）
- ランダム選択機能を維持

使用方法:
    python app_gradio_v2.py
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, List, Dict
import random
import os

# PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    print(f"PyTorch: 利用可能 (CUDA: {torch.cuda.is_available()})")
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch がインストールされていません")

# SAM (Segment Anything Model)
SAM_AVAILABLE = False
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print("SAM: 利用可能")
except ImportError:
    print("警告: segment-anything がインストールされていません")
    print("  pip install segment-anything")

# LaMa Inpainting
LAMA_AVAILABLE = False
try:
    from core.inpainting import InpaintingEngine
    from core.image_utils import resize_image_if_needed, convert_to_binary_mask
    LAMA_AVAILABLE = True
    print("LaMa Inpainting: 利用可能")
except ImportError as e:
    print(f"警告: LaMa Inpainting が利用できません: {e}")


# =============================================================================
# SAM Detector
# =============================================================================

class SAMDetector:
    """Meta SAM を使った髭検出器（ボックス入力対応）"""

    def __init__(
        self,
        checkpoint: str = "sam_vit_h_4b8939.pth",
        model_type: str = "vit_h"
    ):
        self.predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._checkpoint = checkpoint
        self._model_type = model_type
        self._initialized = False

    def initialize(self) -> bool:
        """SAM モデルを初期化（遅延初期化）"""
        if self._initialized:
            return True

        if not SAM_AVAILABLE:
            print("SAM ライブラリが利用できません")
            return False

        # チェックポイントのパスを探索
        checkpoint_paths = [
            self._checkpoint,
            os.path.join(os.path.dirname(__file__), self._checkpoint),
            os.path.join(os.path.dirname(__file__), "..", self._checkpoint),
            os.path.expanduser(f"~/{self._checkpoint}"),
        ]

        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break

        if checkpoint_path is None:
            print(f"SAM チェックポイントが見つかりません: {self._checkpoint}")
            print("ダウンロード: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            return False

        try:
            print(f"SAM モデルを読み込み中: {checkpoint_path}")
            sam = sam_model_registry[self._model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            self._initialized = True
            print(f"SAM: 初期化完了 (device={self.device})")
            return True
        except Exception as e:
            print(f"SAM 初期化エラー: {e}")
            return False

    def is_available(self) -> bool:
        """SAM が利用可能かどうか"""
        return self._initialized and self.predictor is not None

    def segment_with_box(
        self,
        image_rgb: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        ボックス指定でセグメンテーション

        Args:
            image_rgb: RGB 画像 (H, W, 3)
            box: [x_min, y_min, x_max, y_max]

        Returns:
            マスク画像 (H, W) uint8, 0 or 255
        """
        if not self.is_available():
            raise RuntimeError("SAM が初期化されていません")

        self.predictor.set_image(image_rgb)

        masks, scores, _ = self.predictor.predict(
            box=np.array(box),
            multimask_output=True
        )

        # 最もスコアの高いマスクを選択
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8) * 255

        return mask


# =============================================================================
# グローバル状態
# =============================================================================

sam_detector: Optional[SAMDetector] = None
detected_regions: List[Dict] = []
random_seed: int = 42
current_image: Optional[np.ndarray] = None


def get_sam_detector() -> Optional[SAMDetector]:
    """SAM Detector を取得（遅延初期化）"""
    global sam_detector
    if sam_detector is None:
        sam_detector = SAMDetector()
    if not sam_detector.is_available():
        sam_detector.initialize()
    return sam_detector if sam_detector.is_available() else None


# =============================================================================
# ユーティリティ関数
# =============================================================================

def extract_box_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    マスクからバウンディングボックスを抽出

    Args:
        mask: バイナリマスク (0/255 または 0/1)

    Returns:
        (x_min, y_min, x_max, y_max) または None
    """
    if mask is None:
        return None

    # 二値化
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    if np.max(mask) == 0:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 全輪郭の最大バウンディングボックス
    x_min, y_min = mask.shape[1], mask.shape[0]
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    return (x_min, y_min, x_max, y_max)


def extract_mask_from_editor(editor_data: dict) -> Optional[np.ndarray]:
    """ImageEditor からマスクを抽出"""
    if editor_data is None:
        return None

    try:
        if isinstance(editor_data, dict):
            # layers から取得
            if 'layers' in editor_data and len(editor_data['layers']) > 0:
                layer = editor_data['layers'][0]
                if isinstance(layer, np.ndarray):
                    if len(layer.shape) == 3 and layer.shape[2] >= 3:
                        gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                        return (gray > 128).astype(np.uint8) * 255

            # composite から取得
            if 'composite' in editor_data:
                composite = editor_data['composite']
                if isinstance(composite, np.ndarray) and len(composite.shape) == 3:
                    if composite.shape[2] == 4:
                        return composite[:, :, 3]
    except Exception as e:
        print(f"マスク解析エラー: {e}")

    return None


def combine_masks(regions: List[Dict], indices: List[int]) -> np.ndarray:
    """指定インデックスのマスクを統合"""
    if not regions:
        return None

    h, w = regions[0]['mask'].shape[:2]
    combined = np.zeros((h, w), dtype=np.uint8)

    for idx in indices:
        if 0 <= idx < len(regions):
            combined = cv2.bitwise_or(combined, regions[idx]['mask'])

    return combined


# =============================================================================
# Tab 1: SAM による髭検出
# =============================================================================

def process_sam_detection(
    image: np.ndarray,
    editor_data: dict
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    SAM で髭を検出

    Args:
        image: 入力画像
        editor_data: ImageEditor からのデータ（矩形描画）

    Returns:
        (表示用画像, マスク画像, ステータス)
    """
    global detected_regions, current_image

    if image is None:
        return None, None, "画像をアップロードしてください"

    # 画像を保存
    current_image = image.copy()

    # RGB 変換
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image.copy()

    # エディタからマスク（矩形描画）を取得
    drawn_mask = extract_mask_from_editor(editor_data)

    if drawn_mask is None or np.max(drawn_mask) == 0:
        return image_rgb, None, "矩形を描画してください（白色ブラシで範囲を指定）"

    # マスクからボックスを抽出
    box = extract_box_from_mask(drawn_mask)

    if box is None:
        return image_rgb, None, "有効な矩形が検出されませんでした"

    x1, y1, x2, y2 = box

    # SAM で検出
    detector = get_sam_detector()
    if detector is None:
        return image_rgb, None, "SAM が利用できません。チェックポイントを確認してください。"

    try:
        print(f"SAM 検出中... Box=({x1}, {y1}, {x2}, {y2})")
        mask = detector.segment_with_box(image_rgb, box)

        # 検出された領域を分析
        detected_regions = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # 小さすぎる領域は除外
                continue

            region_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)

            M = cv2.moments(region_mask)
            cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else 0
            cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else 0

            detected_regions.append({
                'mask': region_mask,
                'area': area,
                'centroid': (cx, cy)
            })

        # 表示用画像を作成（検出領域を緑でハイライト）
        display = image_rgb.copy()
        overlay = display.copy()
        display[mask > 0] = (0, 255, 0)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        # ボックスを描画
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)

        status = f"SAM 検出完了: {len(detected_regions)} 個の領域を検出 | Box=({x1},{y1},{x2},{y2})"
        return display, mask, status

    except Exception as e:
        return image_rgb, None, f"SAM エラー: {str(e)}"


def update_removal_selection(
    removal_percentage: int,
    selection_mode: str,
    new_seed: bool
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    削除対象の選択を更新

    Args:
        removal_percentage: 削除割合 (%)
        selection_mode: 選択モード
        new_seed: 新しいシードを使用するか

    Returns:
        (表示用画像, マスク画像, ステータス)
    """
    global random_seed, detected_regions, current_image

    if current_image is None or not detected_regions:
        return None, None, "先に SAM で検出を実行してください"

    if new_seed:
        random_seed = random.randint(0, 10000)

    # RGB 変換
    if len(current_image.shape) == 2:
        image_rgb = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)
    elif current_image.shape[2] == 4:
        image_rgb = cv2.cvtColor(current_image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = current_image.copy()

    total = len(detected_regions)
    target_count = int(total * removal_percentage / 100)

    # 選択モードに応じてインデックスを決定
    if selection_mode == "ランダム":
        random.seed(random_seed)
        indices = list(range(total))
        random.shuffle(indices)
        active_indices = sorted(indices[:target_count])
    elif selection_mode == "面積大":
        sorted_idx = sorted(
            range(total),
            key=lambda i: detected_regions[i]['area'],
            reverse=True
        )
        active_indices = sorted(sorted_idx[:target_count])
    elif selection_mode == "面積小":
        sorted_idx = sorted(
            range(total),
            key=lambda i: detected_regions[i]['area']
        )
        active_indices = sorted(sorted_idx[:target_count])
    else:
        active_indices = list(range(target_count))

    # マスクを統合
    combined_mask = combine_masks(detected_regions, active_indices)

    if combined_mask is None:
        return image_rgb, None, "マスクの生成に失敗しました"

    # 膨張処理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    # 表示用画像を作成
    display = image_rgb.copy()
    for i, region in enumerate(detected_regions):
        mask = region['mask']
        if i in active_indices:
            # 削除対象: 赤
            overlay = display.copy()
            display[mask > 0] = (255, 0, 0)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        else:
            # 保持: 緑
            overlay = display.copy()
            display[mask > 0] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    status = f"選択: {len(active_indices)}/{total} 個 ({removal_percentage}%) | シード: {random_seed}"
    return display, combined_mask, status


# =============================================================================
# Tab 2: LaMa Inpainting
# =============================================================================

_inpainting_engine: Optional[InpaintingEngine] = None


def get_inpainting_engine() -> Optional[InpaintingEngine]:
    """Inpainting エンジンを取得（遅延初期化）"""
    global _inpainting_engine
    if _inpainting_engine is None and LAMA_AVAILABLE:
        try:
            print("LaMa エンジンを初期化中...")
            _inpainting_engine = InpaintingEngine()
            print("LaMa エンジン: 初期化完了")
        except Exception as e:
            print(f"LaMa エンジン初期化エラー: {e}")
            return None
    return _inpainting_engine


def process_inpainting(
    image: Image.Image,
    mask: np.ndarray,
    progress=gr.Progress()
) -> Tuple[Image.Image, str]:
    """
    LaMa Inpainting を実行（100% 完全除去）

    Args:
        image: 入力画像 (PIL)
        mask: マスク画像

    Returns:
        (結果画像, ステータス)
    """
    if image is None:
        return None, "画像をアップロードしてください"

    if mask is None or np.max(mask) == 0:
        return None, "マスクを指定してください（Tab 1 で生成するか、直接アップロード）"

    engine = get_inpainting_engine()
    if engine is None:
        return None, "LaMa が利用できません。simple-lama-inpainting をインストールしてください。"

    try:
        progress(0.1, desc="画像を準備中...")

        # 画像をリサイズ
        image = resize_image_if_needed(image)

        # マスクを二値化
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        binary_mask = convert_to_binary_mask(mask)

        # PIL Image に変換
        mask_pil = Image.fromarray(binary_mask, mode='L')

        progress(0.3, desc="LaMa Inpainting 実行中...")

        # Inpainting 実行
        result = engine.inpaint(image, mask_pil)

        progress(1.0, desc="完了")

        return result, "完了！ 髭を 100% 除去しました"

    except Exception as e:
        return None, f"エラーが発生しました: {str(e)}"


def transfer_to_inpainting(
    mask: np.ndarray
) -> Tuple[Image.Image, np.ndarray]:
    """Tab 1 から Tab 2 にデータを転送"""
    global current_image

    if current_image is None:
        return None, None

    # PIL Image に変換
    if len(current_image.shape) == 2:
        image_rgb = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)
    elif current_image.shape[2] == 4:
        image_rgb = cv2.cvtColor(current_image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = current_image

    image_pil = Image.fromarray(image_rgb)

    return image_pil, mask


# =============================================================================
# Gradio UI
# =============================================================================

def create_app():
    """Gradio アプリを作成"""

    with gr.Blocks(title="髭検出・修復アプリ v2") as app:
        gr.Markdown("""
        # 髭検出・修復アプリケーション v2

        **改良点:**
        - SAM による高精度な髭検出（矩形範囲指定）
        - LaMa による 100% 完全除去
        """)

        with gr.Tabs():
            # =================================================================
            # Tab 1: SAM による髭検出
            # =================================================================
            with gr.TabItem("1. SAM 髭検出"):
                gr.Markdown("""
                ### 使い方
                1. 画像をアップロード
                2. 髭の範囲を **矩形で囲む**（白色ブラシで描画）
                3. 「SAM で検出」をクリック
                4. Remove % で削除対象を選択
                5. 「マスクを Tab 2 に転送」
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # 画像アップロード
                        image_input = gr.Image(
                            label="画像をアップロード",
                            type="numpy",
                            sources=["upload"]
                        )

                        # 矩形描画エディタ
                        box_editor = gr.ImageEditor(
                            label="髭の範囲を矩形で囲む（白色ブラシ）",
                            type="numpy",
                            brush=gr.Brush(
                                default_size=30,
                                colors=["white"],
                                default_color="white"
                            ),
                            eraser=gr.Eraser(default_size=30)
                        )

                        detect_btn = gr.Button("SAM で検出", variant="primary")

                    with gr.Column(scale=1):
                        # 検出結果
                        detect_result = gr.Image(
                            label="検出結果（緑=検出領域, 青枠=ボックス）",
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
                            minimum=0, maximum=100, value=50,
                            label="Remove %（削除割合）", step=1
                        )
                        selection_mode = gr.Radio(
                            choices=["ランダム", "面積大", "面積小"],
                            value="ランダム",
                            label="選択モード"
                        )
                        with gr.Row():
                            new_seed_btn = gr.Button("新しいシード")
                            update_btn = gr.Button("選択を更新", variant="secondary")

                    with gr.Column(scale=1):
                        selection_result = gr.Image(
                            label="選択結果（赤=削除対象, 緑=保持）",
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

                # イベント: SAM 検出
                detect_btn.click(
                    fn=process_sam_detection,
                    inputs=[image_input, box_editor],
                    outputs=[detect_result, detect_mask, detect_status]
                )

                # イベント: 選択更新
                update_btn.click(
                    fn=update_removal_selection,
                    inputs=[removal_slider, selection_mode, gr.State(False)],
                    outputs=[selection_result, selection_mask, selection_status]
                )

                # イベント: 新しいシード
                new_seed_btn.click(
                    fn=update_removal_selection,
                    inputs=[removal_slider, selection_mode, gr.State(True)],
                    outputs=[selection_result, selection_mask, selection_status]
                )

            # =================================================================
            # Tab 2: LaMa Inpainting
            # =================================================================
            with gr.TabItem("2. LaMa 髭除去"):
                gr.Markdown("""
                ### 使い方
                1. Tab 1 からマスクを転送（または直接アップロード）
                2. 「髭を除去」をクリック
                3. 結果を確認・保存
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
                        inpaint_btn = gr.Button(
                            "髭を除去",
                            variant="primary",
                            size="lg"
                        )

                    with gr.Column(scale=1):
                        inpaint_result = gr.Image(
                            label="結果（100% 除去）",
                            type="pil",
                            interactive=False
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

                # イベント: Inpainting
                inpaint_btn.click(
                    fn=process_inpainting,
                    inputs=[inpaint_image, inpaint_mask],
                    outputs=[inpaint_result, inpaint_status]
                )

        # 説明
        with gr.Accordion("技術情報", open=False):
            gr.Markdown("""
            ## 使用技術

            ### Tab 1: SAM (Segment Anything Model)
            - **モデル**: Meta SAM (vit_h)
            - **入力**: 画像 + バウンディングボックス
            - **出力**: セグメンテーションマスク
            - **チェックポイント**: sam_vit_h_4b8939.pth

            ### Tab 2: LaMa Inpainting
            - **モデル**: SimpleLama
            - **アルゴリズム**: フーリエ畳み込みベースの画像修復
            - **出力**: 100% 完全除去画像

            ### 必要なパッケージ
            ```bash
            pip install gradio opencv-python numpy pillow torch
            pip install segment-anything
            pip install simple-lama-inpainting
            ```

            ### SAM チェックポイントのダウンロード
            ```bash
            wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
            ```
            """)

    return app


def main():
    """メインエントリーポイント"""
    app = create_app()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7862
    )


if __name__ == "__main__":
    main()
