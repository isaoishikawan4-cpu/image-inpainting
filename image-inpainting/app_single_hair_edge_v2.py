"""
Single Hair Detection v2 - Black & White Beard Detection

This app extends v1 with separate detection modes for:
- Class 1: Black beard (dark hairs against lighter skin)
- Class 2: White beard (light/gray hairs against skin)

Each class has its own filter parameters for optimal detection.

Based on SAM (Segment Anything Model) Automatic Mask Generation.

Usage:
    python app_single_hair_edge_v2.py

Requirements:
    pip install gradio numpy opencv-python pillow scipy
    pip install torch torchvision
    pip install git+https://github.com/facebookresearch/segment-anything.git
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple, List

from beard_inpainting_modules import (
    RegionSelector,
    DetectedRegion,
    visualize_single_hairs,
    BlackWhiteHairDetector,
    HairClassParams,
)


class EdgeDetectionAppV2:
    """Gradio application v2 for black & white hair detection."""

    def __init__(self):
        self._detector = BlackWhiteHairDetector()
        self._current_image: Optional[np.ndarray] = None
        self._detections: List[DetectedRegion] = []
        self._freeform_mask: Optional[np.ndarray] = None  # Store for visualization

    def detect_hairs(
        self,
        editor_data: dict,
        hair_class: str,
        # SAM common params
        sam_points_per_side: int,
        use_tiling: bool,
        tile_size: int,
        tile_overlap: int,
        # Black hair params
        black_min_area: int,
        black_max_area: int,
        black_min_aspect: float,
        black_brightness_threshold: float,
        black_dilation_kernel: int,
        black_dilation_iterations: int,
        # White hair params
        white_min_area: int,
        white_max_area: int,
        white_min_aspect: float,
        white_brightness_threshold: float,
        white_dilation_kernel: int,
        white_dilation_iterations: int,
        # Duplicate removal
        overlap_threshold: float,
        # Region selection
        selection_mode: str,  # 'freeform', 'rectangle', 'coordinates'
        coord_x1: int,
        coord_y1: int,
        coord_x2: int,
        coord_y2: int,
        # Visualization
        overlay_alpha: float,
        show_markers: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int, str]:
        """Main detection function."""

        if editor_data is None:
            return None, None, None, 0, "Please upload an image first"

        if 'background' in editor_data:
            image = editor_data['background']
        elif 'composite' in editor_data:
            image = editor_data['composite']
        else:
            return None, None, None, 0, "Invalid image data"

        if image is None:
            return None, None, None, 0, "No image found"

        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        self._current_image = image
        self._freeform_mask = None
        h, w = image.shape[:2]

        # Extract region based on selection mode
        rect = None
        freeform_mask = None

        if selection_mode == "freeform":
            freeform_mask = RegionSelector.extract_freeform_mask(editor_data)
            if freeform_mask is None:
                return image, None, None, 0, "線で領域を囲んでください (Draw a closed shape to select the region)"

            self._freeform_mask = freeform_mask

            # Get bounding box for display
            contours, _ = cv2.findContours(
                freeform_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for contour in contours:
                    x, y, cw, ch = cv2.boundingRect(contour)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + cw)
                    y_max = max(y_max, y + ch)
                x1, y1, x2, y2 = x_min, y_min, x_max, y_max
            else:
                return image, None, None, 0, "領域を検出できませんでした"

        elif selection_mode == "rectangle":
            rect = RegionSelector.extract_rectangle(editor_data)
            if rect is None:
                return image, None, None, 0, "矩形を描いてください (Draw a rectangle to select the region)"
            x1, y1, x2, y2 = rect

        else:  # coordinates
            x1 = max(0, int(coord_x1))
            y1 = max(0, int(coord_y1))
            x2 = min(w, int(coord_x2))
            y2 = min(h, int(coord_y2))

            if x2 <= x1 or y2 <= y1:
                return image, None, None, 0, f"Invalid coordinates: ({x1},{y1})-({x2},{y2})"
            if x2 - x1 < 10 or y2 - y1 < 10:
                return image, None, None, 0, f"Region too small: {x2-x1}x{y2-y1}"

            rect = (x1, y1, x2, y2)

        # Select parameters based on hair class
        if hair_class == "black":
            params = HairClassParams(
                min_area=black_min_area,
                max_area=black_max_area,
                min_aspect=black_min_aspect,
                brightness_threshold=black_brightness_threshold,
                brightness_mode='darker',
                dilation_kernel_size=black_dilation_kernel,
                dilation_iterations=black_dilation_iterations,
            )
        else:  # white
            params = HairClassParams(
                min_area=white_min_area,
                max_area=white_max_area,
                min_aspect=white_min_aspect,
                brightness_threshold=white_brightness_threshold,
                brightness_mode='brighter',
                dilation_kernel_size=white_dilation_kernel,
                dilation_iterations=white_dilation_iterations,
            )

        # Run detection based on selection mode
        if selection_mode == "freeform" and freeform_mask is not None:
            detections, all_masks, stats = self._detector.detect_with_class_and_mask(
                image, freeform_mask, hair_class, params,
                points_per_side=sam_points_per_side,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                overlap_threshold=overlap_threshold
            )
        else:
            detections, all_masks, stats = self._detector.detect_with_class(
                image, rect, hair_class, params,
                points_per_side=sam_points_per_side,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                overlap_threshold=overlap_threshold
            )

        self._detections = detections

        # Create visualization
        if len(detections) > 0:
            result_image = visualize_single_hairs(
                image, detections,
                alpha=overlay_alpha,
                show_markers=show_markers
            )

            # Draw region outline based on selection mode
            if selection_mode == "freeform" and freeform_mask is not None:
                # Show freeform mask area with semi-transparent overlay
                mask_overlay = np.zeros_like(result_image)
                mask_overlay[freeform_mask > 0] = (0, 255, 255)  # Yellow for mask area
                result_image = cv2.addWeighted(result_image, 1.0, mask_overlay, 0.15, 0)

                contours, _ = cv2.findContours(
                    freeform_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result_image, contours, -1, (255, 255, 255), 2)
            else:
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Combined mask
            mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
            num_det = len(detections)
            for i, det in enumerate(detections):
                hue = int(180 * i / max(num_det, 1))
                hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
                color = tuple(int(c) for c in rgb)
                mask_bool = det.mask > 0
                mask_vis[mask_bool] = color
        else:
            result_image = image.copy()
            # Draw region outline based on selection mode
            if selection_mode == "freeform" and freeform_mask is not None:
                # Show freeform mask area with semi-transparent overlay
                mask_overlay = np.zeros_like(result_image)
                mask_overlay[freeform_mask > 0] = (0, 255, 255)  # Yellow for mask area
                result_image = cv2.addWeighted(result_image, 1.0, mask_overlay, 0.15, 0)

                contours, _ = cv2.findContours(
                    freeform_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result_image, contours, -1, (255, 255, 255), 2)
            else:
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            mask_vis = np.zeros_like(image)

        # ALL Mask visualization
        all_mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        if len(all_masks) > 0:
            masks_with_area = []
            max_area = black_max_area if hair_class == 'black' else white_max_area
            for mask in all_masks:
                area = cv2.countNonZero(mask)
                if area <= max_area:
                    masks_with_area.append((mask, area))

            masks_with_area.sort(key=lambda x: x[1], reverse=True)

            num_all = len(masks_with_area)
            for i, (mask, area) in enumerate(masks_with_area):
                hue = int(180 * i / max(num_all, 1))
                hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
                color = tuple(int(c) for c in rgb)
                mask_bool = mask > 0
                all_mask_vis[mask_bool] = color

        # Build status
        class_label = "黒髭 (Black)" if hair_class == "black" else "白髭 (White)"
        mode_labels = {"freeform": "フリーハンド", "rectangle": "矩形", "coordinates": "座標入力"}
        mode_label = mode_labels.get(selection_mode, selection_mode)
        tiles_info = f", tiles={stats.get('tiles', 1)}" if use_tiling else ""

        status = f"【{class_label}】 検出数: {len(detections)}\n"
        status += f"選択モード: {mode_label}\n"
        if selection_mode == "freeform" and freeform_mask is not None:
            mask_pixels = np.sum(freeform_mask > 0)
            status += f"Mask area: {mask_pixels} pixels (黄色で表示)\n"
        status += f"Region: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px\n"
        status += f"Total masks: {stats['total']} (points_per_side={sam_points_per_side}{tiles_info})\n"
        status += f"Filtered: area_small={stats['filtered_area_small']}, "
        status += f"area_large={stats['filtered_area_large']}, "
        status += f"aspect={stats['filtered_aspect']}, "
        status += f"brightness={stats['filtered_brightness']}"

        if selection_mode == "freeform":
            status += f", outside_mask={stats.get('filtered_outside_mask', 0)}"

        status += "\n"

        if 'mean_brightness' in stats:
            status += f"Mean brightness: {stats['mean_brightness']:.1f}"

        return result_image, mask_vis, all_mask_vis, len(detections), status

    def preview_region(
        self, editor_data: dict,
        coord_x1: int, coord_y1: int, coord_x2: int, coord_y2: int
    ) -> Optional[np.ndarray]:
        """Preview the selected region."""
        if editor_data is None:
            return None

        if 'background' in editor_data:
            image = editor_data['background']
        elif 'composite' in editor_data:
            image = editor_data['composite']
        else:
            return None

        if image is None:
            return None

        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        h, w = image.shape[:2]
        x1 = max(0, int(coord_x1))
        y1 = max(0, int(coord_y1))
        x2 = min(w, int(coord_x2))
        y2 = min(h, int(coord_y2))

        preview = image.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px"
        cv2.putText(preview, text, (x1, max(y1-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return preview

    def get_details(self) -> str:
        """Get detection details."""
        if not self._detections:
            return "No detections yet"

        lines = [f"Total: {len(self._detections)} hairs\n"]

        sources = {}
        for d in self._detections:
            sources[d.source] = sources.get(d.source, 0) + 1

        lines.append("By source:")
        for source, count in sources.items():
            lines.append(f"  {source}: {count}")

        areas = [d.area for d in self._detections]
        if areas:
            lines.append(f"\nArea statistics:")
            lines.append(f"  Min: {min(areas)} px")
            lines.append(f"  Max: {max(areas)} px")
            lines.append(f"  Avg: {sum(areas)/len(areas):.1f} px")

        return "\n".join(lines)


def create_app():
    """Create Gradio application v2."""
    app = EdgeDetectionAppV2()

    with gr.Blocks(
        title="Hair Detection v2 - Black & White",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # Hair Detection v2 - 黒髭・白髭検出

        SAM (Segment Anything Model) を使用した髭検出 v2:
        - **クラス1 (黒髭)**: 肌より暗い髭を検出
        - **クラス2 (白髭)**: 肌より明るい髭を検出

        各クラスで独立したフィルターパラメータを設定可能
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_editor = gr.ImageEditor(
                    label="Input Image (線で領域を囲む / Draw region)",
                    type="numpy",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=5),
                    height=500,
                )

                with gr.Accordion("Region Selection (領域選択)", open=True):
                    selection_mode = gr.Radio(
                        choices=["freeform", "rectangle", "coordinates"],
                        value="freeform",
                        label="Selection Mode (選択モード)",
                        info="freeform: 自由線で囲む | rectangle: 矩形描画 | coordinates: 座標入力"
                    )
                    gr.Markdown("**ヒント**: フリーハンドモードでは、検出したい髭領域を線で囲んでください。")

                with gr.Accordion("Coordinate Input (座標入力)", open=False):
                    gr.Markdown("*selection_mode が 'coordinates' の場合に使用*")
                    with gr.Row():
                        coord_x1 = gr.Number(label="X1 (左)", value=0, precision=0)
                        coord_y1 = gr.Number(label="Y1 (上)", value=0, precision=0)
                    with gr.Row():
                        coord_x2 = gr.Number(label="X2 (右)", value=100, precision=0)
                        coord_y2 = gr.Number(label="Y2 (下)", value=100, precision=0)
                    preview_btn = gr.Button("Preview Region", size="sm")
                    coord_preview = gr.Image(label="Coordinate Preview", type="numpy", height=200)

                # Hair Class Selection
                with gr.Accordion("Hair Class Selection (髭クラス選択)", open=True):
                    hair_class = gr.Radio(
                        choices=["black", "white"],
                        value="black",
                        label="Hair Color Class",
                        info="black: 黒髭（肌より暗い） | white: 白髭（肌より明るい）"
                    )

                # SAM Common Settings
                with gr.Accordion("SAM Settings (共通)", open=True):
                    sam_points_per_side = gr.Slider(
                        minimum=32, maximum=128, value=64, step=8,
                        label="SAM Points Per Side",
                        info="Sampling density (64-96 recommended)"
                    )

                with gr.Accordion("Tile Processing", open=False):
                    use_tiling = gr.Checkbox(value=False, label="Enable Tile Processing")
                    tile_size = gr.Slider(
                        minimum=200, maximum=800, value=400, step=50,
                        label="Tile Size (px)"
                    )
                    tile_overlap = gr.Slider(
                        minimum=20, maximum=150, value=50, step=10,
                        label="Tile Overlap (px)"
                    )

                # Black Hair Parameters
                with gr.Accordion("Black Hair Parameters (黒髭用)", open=True):
                    gr.Markdown("*肌より暗い髭を検出*")
                    black_min_area = gr.Slider(
                        minimum=1, maximum=100, value=5, step=1,
                        label="Min Area"
                    )
                    black_max_area = gr.Slider(
                        minimum=100, maximum=5000, value=2000, step=100,
                        label="Max Area"
                    )
                    black_min_aspect = gr.Slider(
                        minimum=1.0, maximum=5.0, value=1.2, step=0.1,
                        label="Min Aspect Ratio"
                    )
                    black_brightness_threshold = gr.Slider(
                        minimum=0.80, maximum=1.30, value=1.14, step=0.02,
                        label="Brightness Threshold",
                        info="マスクの明るさ < 平均×閾値 で検出 (高い=より明るい髭も許容)"
                    )
                    gr.Markdown("**Dilation (膨張処理)**")
                    black_dilation_kernel = gr.Slider(
                        minimum=0, maximum=15, value=0, step=1,
                        label="Dilation Kernel Size",
                        info="0=OFF, 奇数値推奨 (3, 5, 7...) 検出領域を拡大"
                    )
                    black_dilation_iterations = gr.Slider(
                        minimum=1, maximum=5, value=1, step=1,
                        label="Dilation Iterations",
                        info="膨張処理の繰り返し回数"
                    )

                # White Hair Parameters
                with gr.Accordion("White Hair Parameters (白髭用)", open=True):
                    gr.Markdown("*肌より明るい髭を検出*")
                    white_min_area = gr.Slider(
                        minimum=1, maximum=100, value=5, step=1,
                        label="Min Area"
                    )
                    white_max_area = gr.Slider(
                        minimum=100, maximum=5000, value=2000, step=100,
                        label="Max Area"
                    )
                    white_min_aspect = gr.Slider(
                        minimum=1.0, maximum=5.0, value=1.2, step=0.1,
                        label="Min Aspect Ratio"
                    )
                    white_brightness_threshold = gr.Slider(
                        minimum=0.70, maximum=1.20, value=0.95, step=0.02,
                        label="Brightness Threshold",
                        info="マスクの明るさ > 平均×閾値 で検出 (低い=より暗い髭も許容)"
                    )
                    gr.Markdown("**Dilation (膨張処理)**")
                    white_dilation_kernel = gr.Slider(
                        minimum=0, maximum=15, value=0, step=1,
                        label="Dilation Kernel Size",
                        info="0=OFF, 奇数値推奨 (3, 5, 7...) 検出領域を拡大"
                    )
                    white_dilation_iterations = gr.Slider(
                        minimum=1, maximum=5, value=1, step=1,
                        label="Dilation Iterations",
                        info="膨張処理の繰り返し回数"
                    )

                # Duplicate Removal
                with gr.Accordion("Duplicate Removal (重複除去)", open=True):
                    overlap_threshold = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="Overlap Threshold",
                        info="重複判定の閾値。高い値=より多くのマスクを保持、低い値=より厳しく重複除去"
                    )

                # Visualization
                with gr.Accordion("Visualization Settings", open=False):
                    overlay_alpha = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                        label="Overlay Alpha"
                    )
                    show_markers = gr.Checkbox(value=True, label="Show Center Markers")

                detect_btn = gr.Button("Detect Hairs", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_image = gr.Image(label="Detection Result", type="numpy", height=350)
                with gr.Row():
                    mask_image = gr.Image(label="Filtered Mask", type="numpy", height=200)
                    all_mask_image = gr.Image(label="ALL Mask (Unfiltered)", type="numpy", height=200)

                with gr.Row():
                    hair_count = gr.Number(label="Hair Count", value=0, precision=0)
                status_text = gr.Textbox(label="Status", lines=5)

                details_btn = gr.Button("Show Details")
                details_text = gr.Textbox(label="Details", lines=8)

        detect_btn.click(
            fn=app.detect_hairs,
            inputs=[
                image_editor,
                hair_class,
                sam_points_per_side,
                use_tiling,
                tile_size,
                tile_overlap,
                black_min_area,
                black_max_area,
                black_min_aspect,
                black_brightness_threshold,
                black_dilation_kernel,
                black_dilation_iterations,
                white_min_area,
                white_max_area,
                white_min_aspect,
                white_brightness_threshold,
                white_dilation_kernel,
                white_dilation_iterations,
                overlap_threshold,
                selection_mode,
                coord_x1,
                coord_y1,
                coord_x2,
                coord_y2,
                overlay_alpha,
                show_markers,
            ],
            outputs=[result_image, mask_image, all_mask_image, hair_count, status_text]
        )

        preview_btn.click(
            fn=app.preview_region,
            inputs=[image_editor, coord_x1, coord_y1, coord_x2, coord_y2],
            outputs=[coord_preview]
        )

        details_btn.click(
            fn=app.get_details,
            outputs=[details_text]
        )

        gr.Markdown("""
        ---
        ### 選択モード説明

        | モード | 使い方 |
        |--------|--------|
        | **freeform** (デフォルト) | 線で自由に領域を囲む。囲んだ形状内のみ検出 |
        | **rectangle** | 矩形を描画して領域を選択 |
        | **coordinates** | 座標入力で矩形領域を指定 |

        ### パラメータ説明

        | クラス | Brightness Threshold | 意味 |
        |--------|---------------------|------|
        | 黒髭 | 1.14 (default) | マスク明るさ < 平均×1.14 で検出。高い値=より明るい髭も許容 |
        | 白髭 | 0.95 (default) | マスク明るさ > 平均×0.95 で検出。低い値=より暗い髭も許容 |

        ### 調整のヒント

        **黒髭が検出されない場合:**
        - `Brightness Threshold` を上げる (1.2-1.3)

        **白髭が検出されない場合:**
        - `Brightness Threshold` を下げる (0.85-0.90)

        **誤検出が多い場合:**
        - `Min Aspect Ratio` を上げる (1.5-2.0)
        - `Brightness Threshold` を調整 (黒髭:下げる、白髭:上げる)
        """)

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7864,  # Different port from v1
        share=False
    )
