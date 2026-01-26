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

def process_beard_detection(
    image, editor_data, detection_mode,
    text_prompt, box_threshold, text_threshold,
    threshold_value, min_area, max_area
):
    """Wrapper for pipeline.process_detection."""
    use_grounded_sam = "Grounded SAM" in detection_mode
    return pipeline.process_detection(
        image=image,
        editor_data=editor_data,
        use_grounded_sam=use_grounded_sam,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        threshold_value=threshold_value,
        min_area=min_area,
        max_area=max_area
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


def process_lama_inpainting(image, mask, thinning_levels, progress=gr.Progress()):
    """Wrapper for pipeline.process_inpainting."""
    import numpy as np
    from PIL import Image

    gallery_items, status = pipeline.process_inpainting(
        image=image,
        mask=mask,
        thinning_levels=thinning_levels,
        progress=progress
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

    return gallery_items, status, last_image, editor_data, editor_data


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
    return img, editor_data, editor_data


def transfer_tab1_mask(selection_mask):
    """Transfer Tab 1's selection mask to Tab 3 for preview."""
    if selection_mask is None:
        return None, "Tab 1で髭を検出・選択してください"
    return selection_mask, "Tab 1のマスクを取得しました"


def process_color_correction_with_source(
    image, target_editor, source_editor,
    correction_mode, strength,
    mask_source, tab1_mask,
    a_factor, b_factor, l_factor
):
    """Wrapper for color correction with mask source selection."""
    # Determine which mask to use
    if mask_source == "Tab 1の選択マスクを使用":
        # Use Tab 1's selection mask
        if tab1_mask is None:
            return image, "Tab 1のマスクが設定されていません。「Tab 1の選択マスクを取得」ボタンを押してください"

        # Create a dummy editor data with the Tab 1 mask as the layer
        target_editor_data = {
            "layers": [tab1_mask],
            "background": image
        }
    else:
        # Use manual painting (existing behavior)
        target_editor_data = target_editor

    return pipeline.process_color_correction(
        image=image,
        target_editor_data=target_editor_data,
        source_editor_data=source_editor,
        correction_mode=correction_mode,
        strength=strength,
        a_adjustment_factor=a_factor,
        b_adjustment_factor=b_factor,
        l_adjustment_factor=l_factor
    )


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
                2. 髭の範囲を **矩形で囲む**（白色ブラシで塗りつぶし）
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

                        # 矩形描画エディタ
                        rect_editor = gr.ImageEditor(
                            label="髭の範囲を矩形で囲む（白色ブラシで塗りつぶし）",
                            type="numpy",
                            brush=gr.Brush(
                                default_size=30,
                                colors=["white"],
                                default_color="white"
                            ),
                            eraser=gr.Eraser(default_size=30)
                        )

                        # 検出モード選択
                        detection_mode_radio = gr.Radio(
                            choices=["ルールベース（1本ずつ検出）", "Grounded SAM"],
                            value="ルールベース（1本ずつ検出）",
                            label="検出モード",
                            info="ルールベースは髭を1本ずつ高精度に検出します（推奨）"
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
                            label="選択結果（赤=削除対象, 他の色=保持）",
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

                # イベント: 髭検出
                detect_btn.click(
                    fn=process_beard_detection,
                    inputs=[
                        image_input, rect_editor, detection_mode_radio,
                        text_prompt_input, box_threshold_slider, text_threshold_slider,
                        threshold_slider, min_area_slider, max_area_slider
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
            # Tab 2: LaMa Inpainting
            # =================================================================
            with gr.TabItem("2. LaMa 髭除去"):
                gr.Markdown("""
                ### 使い方
                1. Tab 1 からマスクを転送（または直接アップロード）
                2. 薄め具合を選択
                3. 「髭薄めを実行」をクリック
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
                    outputs=[color_image, target_color_editor, source_color_editor]
                )

                # イベント: Tab 1のマスクを転送
                transfer_mask_btn.click(
                    fn=transfer_tab1_mask,
                    inputs=[selection_mask],
                    outputs=[tab1_mask_preview, color_status]
                )

                # イベント: 色調補正適用
                apply_color_btn.click(
                    fn=process_color_correction_with_source,
                    inputs=[
                        color_image, target_color_editor, source_color_editor,
                        correction_mode_radio, color_strength_slider,
                        mask_source_radio, tab1_mask_preview,
                        a_factor_slider, b_factor_slider, l_factor_slider
                    ],
                    outputs=[color_result, color_status]
                )

        # =================================================================
        # Cross-Tab Event: Inpainting -> Tab 3 auto-transfer
        # (Defined here because color_image is in Tab 3)
        # =================================================================
        inpaint_btn.click(
            fn=process_lama_inpainting,
            inputs=[inpaint_image, inpaint_mask, thinning_checkboxes],
            outputs=[inpaint_gallery, inpaint_status, color_image, target_color_editor, source_color_editor]
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
