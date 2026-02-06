"""
Single Hair Detection v3 - Hair Thickness Classification

This app extends v2 with hair thickness classification:
- 剛毛 (Coarse): Very thick hair
- 硬毛 (Thick): Thick hair
- 中間毛 (Medium): Medium thickness hair
- 軟毛 (Fine): Fine/thin hair

Classification is based on the minimum bounding rectangle width of each detected hair.
Uses SAM (Segment Anything Model) for detection.

Usage:
    python app_single_hair_edge_v3.py

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
from dataclasses import dataclass

from beard_inpainting_modules import (
    RegionSelector,
    DetectedRegion,
    BlackWhiteHairDetector,
    HairClassParams,
)


# Hair thickness categories with colors (RGB for Gradio display)
THICKNESS_CATEGORIES = {
    "剛毛": {"color": (255, 0, 0), "label_en": "Coarse"},       # Red
    "硬毛": {"color": (255, 165, 0), "label_en": "Thick"},      # Orange
    "中間毛": {"color": (0, 200, 0), "label_en": "Medium"},     # Green
    "軟毛": {"color": (148, 0, 211), "label_en": "Fine"},       # Purple (Dark Violet)
}


@dataclass
class ClassifiedHair:
    """A detected hair with thickness classification."""
    detection: DetectedRegion
    width: float
    category: str


def classify_hair_thickness(
    width: float,
    threshold_coarse: float,
    threshold_thick: float,
    threshold_medium: float
) -> str:
    """Classify hair based on width thresholds."""
    if width >= threshold_coarse:
        return "剛毛"
    elif width >= threshold_thick:
        return "硬毛"
    elif width >= threshold_medium:
        return "中間毛"
    else:
        return "軟毛"


def calculate_hair_width(mask: np.ndarray) -> float:
    """Calculate hair width from mask using minAreaRect."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    w_rect, h_rect = rect[1]

    if w_rect == 0 or h_rect == 0:
        return 0.0

    return min(w_rect, h_rect)


def visualize_classified_hairs(
    image: np.ndarray,
    classified_hairs: List[ClassifiedHair],
    alpha: float = 0.4,
    show_markers: bool = True
) -> np.ndarray:
    """Visualize hairs with category-based colors."""
    result = image.copy()
    overlay = np.zeros_like(image)

    for hair in classified_hairs:
        color = THICKNESS_CATEGORIES[hair.category]["color"]
        mask_bool = hair.detection.mask > 0
        overlay[mask_bool] = color

        if show_markers:
            cx, cy = hair.detection.centroid
            cv2.circle(result, (cx, cy), 2, color, -1)

    mask_any = np.any(overlay > 0, axis=2)
    result[mask_any] = cv2.addWeighted(
        result[mask_any], 1 - alpha,
        overlay[mask_any], alpha,
        0
    )

    return result


class EdgeDetectionAppV3:
    """Gradio application v3 for hair thickness classification."""

    def __init__(self):
        self._detector = BlackWhiteHairDetector()
        self._current_image: Optional[np.ndarray] = None
        self._classified_hairs: List[ClassifiedHair] = []
        self._freeform_mask: Optional[np.ndarray] = None

    def detect_and_classify(
        self,
        editor_data: dict,
        hair_class: str,
        # SAM params
        sam_points_per_side: int,
        use_tiling: bool,
        tile_size: int,
        tile_overlap: int,
        # Filter params (inherited from v2)
        min_area: int,
        max_area: int,
        min_aspect: float,
        brightness_threshold: float,
        dilation_kernel: int,
        dilation_iterations: int,
        # Thickness thresholds (v3 new)
        threshold_coarse: float,
        threshold_thick: float,
        threshold_medium: float,
        # Duplicate removal
        overlap_threshold: float,
        # Region selection
        selection_mode: str,
        coord_x1: int,
        coord_y1: int,
        coord_x2: int,
        coord_y2: int,
        # Visualization
        overlay_alpha: float,
        show_markers: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str, str]:
        """Main detection and classification function.

        Returns:
            result_image: Detection result with category colors
            category_mask: Category-colored mask
            all_mask: All SAM masks (before filtering)
            filtered_mask: Filtered masks (after filtering, before category coloring)
            count_display: Category counts text
            status: Status text
        """

        if editor_data is None:
            return None, None, None, None, "0", "Please upload an image first"

        if 'background' in editor_data:
            image = editor_data['background']
        elif 'composite' in editor_data:
            image = editor_data['composite']
        else:
            return None, None, None, None, "0", "Invalid image data"

        if image is None:
            return None, None, None, None, "0", "No image found"

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
                return image, None, None, None, "0", "線で領域を囲んでください"
            self._freeform_mask = freeform_mask
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
                return image, None, None, None, "0", "領域を検出できませんでした"

        elif selection_mode == "rectangle":
            rect = RegionSelector.extract_rectangle(editor_data)
            if rect is None:
                return image, None, None, None, "0", "矩形を描いてください"
            x1, y1, x2, y2 = rect

        else:  # coordinates
            x1 = max(0, int(coord_x1))
            y1 = max(0, int(coord_y1))
            x2 = min(w, int(coord_x2))
            y2 = min(h, int(coord_y2))
            if x2 <= x1 or y2 <= y1:
                return image, None, None, None, "0", f"Invalid coordinates: ({x1},{y1})-({x2},{y2})"
            rect = (x1, y1, x2, y2)

        # Set up detection parameters (inherited from v2)
        params = HairClassParams(
            min_area=min_area,
            max_area=max_area,
            min_aspect=min_aspect,
            brightness_threshold=brightness_threshold,
            brightness_mode='darker' if hair_class == 'black' else 'brighter',
            dilation_kernel_size=dilation_kernel,
            dilation_iterations=dilation_iterations,
        )

        # Run SAM detection
        if selection_mode == "freeform" and freeform_mask is not None:
            detections, all_masks, stats = self._detector.detect_with_class_and_mask(
                image, freeform_mask, hair_class, params,
                points_per_side=sam_points_per_side,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                overlap_threshold=overlap_threshold,
            )
        else:
            detections, all_masks, stats = self._detector.detect_with_class(
                image, rect, hair_class, params,
                points_per_side=sam_points_per_side,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                overlap_threshold=overlap_threshold,
            )

        # Classify detected hairs by thickness
        self._classified_hairs = []
        category_counts = {"剛毛": 0, "硬毛": 0, "中間毛": 0, "軟毛": 0}
        width_stats = []

        for det in detections:
            width = calculate_hair_width(det.mask)
            category = classify_hair_thickness(
                width, threshold_coarse, threshold_thick, threshold_medium
            )
            self._classified_hairs.append(ClassifiedHair(
                detection=det,
                width=width,
                category=category
            ))
            category_counts[category] += 1
            width_stats.append(width)

        # Create All Masks visualization (before filtering)
        all_mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        if len(all_masks) > 0:
            masks_with_area = []
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

        # Create Filtered Masks visualization (after filtering, rainbow colors)
        filtered_mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        if len(detections) > 0:
            num_det = len(detections)
            for i, det in enumerate(detections):
                hue = int(180 * i / max(num_det, 1))
                hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
                color = tuple(int(c) for c in rgb)
                mask_bool = det.mask > 0
                filtered_mask_vis[mask_bool] = color

        # Create visualization
        if len(self._classified_hairs) > 0:
            result_image = visualize_classified_hairs(
                image, self._classified_hairs,
                alpha=overlay_alpha,
                show_markers=show_markers
            )

            # Draw region outline
            if selection_mode == "freeform" and freeform_mask is not None:
                contours, _ = cv2.findContours(
                    freeform_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result_image, contours, -1, (255, 255, 255), 2)
            else:
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Create category mask visualization
            category_mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
            for hair in self._classified_hairs:
                color = THICKNESS_CATEGORIES[hair.category]["color"]
                mask_bool = hair.detection.mask > 0
                category_mask_vis[mask_bool] = color
        else:
            result_image = image.copy()
            if selection_mode == "freeform" and freeform_mask is not None:
                contours, _ = cv2.findContours(
                    freeform_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result_image, contours, -1, (255, 255, 255), 2)
            else:
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            category_mask_vis = np.zeros_like(image)

        # Build status text
        total = len(self._classified_hairs)
        class_label = "黒髭" if hair_class == "black" else "白髭"
        mode_labels = {"freeform": "フリーハンド", "rectangle": "矩形", "coordinates": "座標入力"}
        mode_label = mode_labels.get(selection_mode, selection_mode)

        status_lines = [
            f"【{class_label}】 総検出数: {total}",
            f"選択モード: {mode_label}",
            f"Region: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px",
            "",
            "━━━ カテゴリ別検出数 ━━━",
            f"  剛毛 (Coarse):  {category_counts['剛毛']}本 (>={threshold_coarse:.1f}px)",
            f"  硬毛 (Thick):   {category_counts['硬毛']}本 (>={threshold_thick:.1f}px)",
            f"  中間毛 (Medium): {category_counts['中間毛']}本 (>={threshold_medium:.1f}px)",
            f"  軟毛 (Fine):    {category_counts['軟毛']}本 (<{threshold_medium:.1f}px)",
            "",
            "━━━ フィルタ統計 ━━━",
            f"SAM masks: {stats['total']} (points_per_side={sam_points_per_side})",
            f"Filtered: area_small={stats['filtered_area_small']}, "
            f"area_large={stats['filtered_area_large']}, "
            f"aspect={stats['filtered_aspect']}, "
            f"brightness={stats['filtered_brightness']}",
        ]

        if 'mean_brightness' in stats:
            status_lines.append(f"Mean brightness: {stats['mean_brightness']:.1f}")

        if width_stats:
            status_lines.extend([
                "",
                "━━━ 幅の統計 ━━━",
                f"Min: {min(width_stats):.1f}px, Max: {max(width_stats):.1f}px, "
                f"Avg: {sum(width_stats)/len(width_stats):.1f}px"
            ])

        status = "\n".join(status_lines)

        # Count display
        count_display = (
            f"総数:{total} | "
            f"剛毛:{category_counts['剛毛']} | "
            f"硬毛:{category_counts['硬毛']} | "
            f"中間毛:{category_counts['中間毛']} | "
            f"軟毛:{category_counts['軟毛']}"
        )

        return result_image, category_mask_vis, all_mask_vis, filtered_mask_vis, count_display, status

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
        """Get detailed classification results."""
        if not self._classified_hairs:
            return "No detections yet"

        lines = [f"Total: {len(self._classified_hairs)} hairs\n"]

        # Group by category
        by_category = {"剛毛": [], "硬毛": [], "中間毛": [], "軟毛": []}
        for hair in self._classified_hairs:
            by_category[hair.category].append(hair.width)

        for cat in ["剛毛", "硬毛", "中間毛", "軟毛"]:
            widths = by_category[cat]
            if widths:
                lines.append(f"\n{cat} ({len(widths)}本):")
                lines.append(f"  Width range: {min(widths):.1f} - {max(widths):.1f}px")
                lines.append(f"  Average: {sum(widths)/len(widths):.1f}px")

        return "\n".join(lines)


def create_app():
    """Create Gradio application v3."""
    app = EdgeDetectionAppV3()

    with gr.Blocks(
        title="Hair Detection v3 - Thickness Classification",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # Hair Detection v3 - 髭太さ分類

        SAMで検出した髭を太さ（幅）に基づいて4種類に分類:
        - **剛毛 (Coarse)**: 非常に太い髭 (赤)
        - **硬毛 (Thick)**: 太い髭 (オレンジ)
        - **中間毛 (Medium)**: 中程度の髭 (緑)
        - **軟毛 (Fine)**: 細い髭 (紫)

        各カテゴリの閾値はスライダーで調整可能
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_editor = gr.ImageEditor(
                    label="Input Image",
                    type="numpy",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=5),
                    height=500,
                )

                with gr.Accordion("Region Selection (領域選択)", open=True):
                    selection_mode = gr.Radio(
                        choices=["freeform", "rectangle", "coordinates"],
                        value="coordinates",
                        label="Selection Mode",
                        info="freeform: 自由線で囲む | rectangle: 矩形描画 | coordinates: 座標入力"
                    )

                with gr.Accordion("Coordinate Input (座標入力)", open=True):
                    with gr.Row():
                        coord_x1 = gr.Number(label="X1 (左)", value=400, precision=0)
                        coord_y1 = gr.Number(label="Y1 (上)", value=440, precision=0)
                    with gr.Row():
                        coord_x2 = gr.Number(label="X2 (右)", value=690, precision=0)
                        coord_y2 = gr.Number(label="Y2 (下)", value=555, precision=0)
                    preview_btn = gr.Button("Preview Region", size="sm")
                    coord_preview = gr.Image(label="Coordinate Preview", type="numpy", height=200)

                with gr.Accordion("Hair Color (髭色クラス)", open=True):
                    hair_class = gr.Radio(
                        choices=["black", "white"],
                        value="black",
                        label="Hair Color Class",
                        info="black: 黒髭（肌より暗い） | white: 白髭（肌より明るい）"
                    )

                with gr.Accordion("SAM Settings (共通)", open=False):
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

                with gr.Accordion("Filter Parameters (フィルタ)", open=True):
                    min_area = gr.Slider(
                        minimum=1, maximum=50, value=5, step=1,
                        label="Min Area"
                    )
                    max_area = gr.Slider(
                        minimum=50, maximum=5000, value=100, step=10,
                        label="Max Area"
                    )
                    min_aspect = gr.Slider(
                        minimum=1.0, maximum=5.0, value=1.0, step=0.1,
                        label="Min Aspect Ratio"
                    )
                    brightness_threshold = gr.Slider(
                        minimum=0.90, maximum=1.20, value=1.00, step=0.01,
                        label="Brightness Threshold",
                        info="黒髭: マスク明るさ < 平均×閾値 | 白髭: マスク明るさ > 平均×閾値"
                    )
                    gr.Markdown("**Dilation (膨張処理)**")
                    dilation_kernel = gr.Slider(
                        minimum=0, maximum=15, value=0, step=1,
                        label="Dilation Kernel Size",
                        info="0=OFF, 奇数値推奨 (3, 5, 7...)"
                    )
                    dilation_iterations = gr.Slider(
                        minimum=1, maximum=5, value=1, step=1,
                        label="Dilation Iterations"
                    )

                with gr.Accordion("Thickness Thresholds (太さ閾値)", open=True):
                    gr.Markdown("""
                    髭の太さ（外接矩形の短辺）で分類する閾値:
                    - 剛毛 >= Coarse閾値
                    - 硬毛 >= Thick閾値 (かつ < Coarse閾値)
                    - 中間毛 >= Medium閾値 (かつ < Thick閾値)
                    - 軟毛 < Medium閾値
                    """)
                    threshold_coarse = gr.Slider(
                        minimum=1, maximum=20, value=8, step=0.5,
                        label="剛毛 (Coarse) 閾値 [px]"
                    )
                    threshold_thick = gr.Slider(
                        minimum=1, maximum=15, value=6, step=0.5,
                        label="硬毛 (Thick) 閾値 [px]"
                    )
                    threshold_medium = gr.Slider(
                        minimum=1, maximum=10, value=4, step=0.5,
                        label="中間毛 (Medium) 閾値 [px]"
                    )

                with gr.Accordion("Duplicate Removal (重複除去)", open=True):
                    overlap_threshold = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="Overlap Threshold",
                        info="重複判定の閾値。高い値=より多くのマスクを保持、低い値=より厳しく重複除去"
                    )

                with gr.Accordion("Visualization", open=False):
                    overlay_alpha = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                        label="Overlay Alpha"
                    )
                    show_markers = gr.Checkbox(value=True, label="Show Center Markers")

                detect_btn = gr.Button("Detect & Classify", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_image = gr.Image(
                    label="Detection Result (Color = Category)",
                    type="numpy",
                    height=350
                )
                with gr.Row():
                    category_mask_image = gr.Image(
                        label="Category Mask",
                        type="numpy",
                        height=200
                    )
                    filtered_mask_image = gr.Image(
                        label="Filtered Mask",
                        type="numpy",
                        height=200
                    )
                all_mask_image = gr.Image(
                    label="All Masks (Unfiltered)",
                    type="numpy",
                    height=200
                )

                count_display = gr.Textbox(label="Category Counts", lines=1)
                status_text = gr.Textbox(label="Status", lines=14)

                details_btn = gr.Button("Show Details")
                details_text = gr.Textbox(label="Details", lines=10)

        # Legend
        gr.Markdown("""
        ---
        ### 色の凡例
        | 色 | カテゴリ | 説明 |
        |----|----------|------|
        | 赤 | 剛毛 (Coarse) | 非常に太い髭 |
        | オレンジ | 硬毛 (Thick) | 太い髭 |
        | 緑 | 中間毛 (Medium) | 中程度の髭 |
        | 紫 | 軟毛 (Fine) | 細い髭 |

        ### 調整のヒント
        - まず検出を実行し、「幅の統計」のMin/Max/Avgを参考に閾値を調整
        - 閾値は **剛毛 > 硬毛 > 中間毛** の順に大きくなるよう設定
        """)

        detect_btn.click(
            fn=app.detect_and_classify,
            inputs=[
                image_editor,
                hair_class,
                sam_points_per_side,
                use_tiling,
                tile_size,
                tile_overlap,
                min_area,
                max_area,
                min_aspect,
                brightness_threshold,
                dilation_kernel,
                dilation_iterations,
                threshold_coarse,
                threshold_thick,
                threshold_medium,
                overlap_threshold,
                selection_mode,
                coord_x1,
                coord_y1,
                coord_x2,
                coord_y2,
                overlay_alpha,
                show_markers,
            ],
            outputs=[result_image, category_mask_image, all_mask_image, filtered_mask_image, count_display, status_text]
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

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7865,
        share=False
    )
