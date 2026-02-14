"""

1. 毛質別のカウント:
   SAMで検出した髭を太さ（幅）に基づいて4種類に分類:
   - 剛毛: 赤
   - 硬毛: オレンジ
   - 中間毛: 緑
   - 軟毛: 紫

2. 色別のカウント:
   髭を色（明るさ）で分類してカウント:
   - 黒髭: 肌より暗い髭を検出
   - 白髭: 肌より明るい髭を検出

   明るさの閾値 パラメータ:
   - 黒髭 (デフォルト: 1.14, 範囲: 0.80-1.30):
     マスクの明るさ < ROI平均 × 閾値 で検出。
     値を上げると、より明るい髭も許容（検出数増加）。
   - 白髭 (デフォルト: 0.95, 範囲: 0.70-1.20):
     マスクの明るさ > ROI平均 × 閾値 で検出。
     値を下げると、より暗い髭も許容（検出数増加）。

   最小/最大面積: マスクのピクセル面積でフィルタ。
   最小アスペクト比: 細長さでフィルタ（1.0=すべて許可, 2.0=細長いもののみ）。
   膨張処理: 検出マスクを膨張させて隣接ピクセルを含める（0=OFF）。

Usage:
    python app_sam_edge-detection.py

Requirements:
    pip install gradio numpy opencv-python pillow scipy
    pip install torch torchvision
    pip install git+https://github.com/facebookresearch/segment-anything.git
"""

import gradio as gr
import numpy as np
import cv2
from typing import Optional, Tuple, List

from beard_inpainting_modules import (
    RegionSelector,
    DetectedRegion,
    BlackWhiteHairDetector,
    HairClassParams,
    ImageHandler,
    visualize_single_hairs,
    # Thickness classification
    THICKNESS_CATEGORIES,
    ClassifiedHair,
    calculate_hair_width,
    classify_hair_thickness,
    visualize_classified_hairs,
)

class EdgeDetectionAppV3:
    """Gradio application v3 for hair thickness classification and black/white count."""

    def __init__(self):
        self._detector = BlackWhiteHairDetector()
        self._current_image: Optional[np.ndarray] = None
        self._classified_hairs: List[ClassifiedHair] = []
        self._detections: List[DetectedRegion] = []
        self._freeform_mask: Optional[np.ndarray] = None
        self._last_mode: str = "毛質別のカウント"

    def detect_and_classify(
        self,
        editor_data: dict,
        detection_mode: str,
        hair_class: str,
        # SAM params
        sam_points_per_side: int,
        pred_iou_thresh: float,
        stability_score_thresh: float,
        use_tiling: bool,
        tile_size: int,
        tile_overlap: int,
        # Thickness mode: filter params
        min_area: int,
        max_area: int,
        min_aspect: float,
        brightness_threshold: float,
        dilation_kernel: int,
        dilation_iterations: int,
        # Thickness mode: thickness thresholds
        threshold_coarse: float,
        threshold_thick: float,
        threshold_medium: float,
        # BW Count mode: params
        bw_min_area: int,
        bw_max_area: int,
        bw_min_aspect: float,
        bw_brightness_threshold: float,
        bw_dilation_kernel: int,
        bw_dilation_iterations: int,
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
            category_mask: Category-colored mask (thickness mode) or None (bw_count mode)
            all_mask: All SAM masks (before filtering)
            filtered_mask: Filtered masks (after filtering)
            count_display: Category counts text
            status: Status text
        """

        # Extract image from editor
        image = ImageHandler.extract_image_from_editor(editor_data)
        if image is None:
            return None, None, None, None, "0", "画像をアップロードしてください"

        self._current_image = image
        self._freeform_mask = None
        self._last_mode = detection_mode
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

        # Build HairClassParams based on detection mode
        if detection_mode == "色別のカウント":
            params = HairClassParams(
                min_area=bw_min_area,
                max_area=bw_max_area,
                min_aspect=bw_min_aspect,
                brightness_threshold=bw_brightness_threshold,
                brightness_mode='darker' if hair_class == 'black' else 'brighter',
                dilation_kernel_size=bw_dilation_kernel,
                dilation_iterations=bw_dilation_iterations,
            )
            current_max_area = bw_max_area
        else:  # 毛質別のカウント
            params = HairClassParams(
                min_area=min_area,
                max_area=max_area,
                min_aspect=min_aspect,
                brightness_threshold=brightness_threshold,
                brightness_mode='darker' if hair_class == 'black' else 'brighter',
                dilation_kernel_size=dilation_kernel,
                dilation_iterations=dilation_iterations,
            )
            current_max_area = max_area

        # Run SAM detection
        if selection_mode == "freeform" and freeform_mask is not None:
            detections, all_masks, stats = self._detector.detect_with_class_and_mask(
                image, freeform_mask, hair_class, params,
                points_per_side=sam_points_per_side,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                overlap_threshold=overlap_threshold,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh
            )
        else:
            detections, all_masks, stats = self._detector.detect_with_class(
                image, rect, hair_class, params,
                points_per_side=sam_points_per_side,
                use_tiling=use_tiling,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                overlap_threshold=overlap_threshold,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh
            )

        # Create All Masks visualization (before filtering)
        all_mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        if len(all_masks) > 0:
            masks_with_area = []
            for mask in all_masks:
                area = cv2.countNonZero(mask)
                if area <= current_max_area:
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

        # Branch based on detection mode
        if detection_mode == "色別のカウント":
            return self._process_bw_count_mode(
                image, detections, all_mask_vis, filtered_mask_vis, stats,
                hair_class, selection_mode, freeform_mask,
                x1, y1, x2, y2, sam_points_per_side, use_tiling,
                overlay_alpha, show_markers,
            )
        else:
            return self._process_thickness_mode(
                image, detections, all_mask_vis, filtered_mask_vis, stats,
                hair_class, selection_mode, freeform_mask,
                x1, y1, x2, y2, sam_points_per_side,
                threshold_coarse, threshold_thick, threshold_medium,
                overlay_alpha, show_markers,
            )

    def _process_thickness_mode(
        self,
        image: np.ndarray,
        detections: List[DetectedRegion],
        all_mask_vis: np.ndarray,
        filtered_mask_vis: np.ndarray,
        stats: dict,
        hair_class: str,
        selection_mode: str,
        freeform_mask: Optional[np.ndarray],
        x1: int, y1: int, x2: int, y2: int,
        sam_points_per_side: int,
        threshold_coarse: float,
        threshold_thick: float,
        threshold_medium: float,
        overlay_alpha: float,
        show_markers: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray, str, str]:
        """Process detections in thickness classification mode."""
        h, w = image.shape[:2]

        # Classify detected hairs by thickness
        self._classified_hairs = []
        self._detections = []
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
            f"【毛質別のカウント - {class_label}】 総検出数: {total}",
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

        count_display = (
            f"総数:{total} | "
            f"剛毛:{category_counts['剛毛']} | "
            f"硬毛:{category_counts['硬毛']} | "
            f"中間毛:{category_counts['中間毛']} | "
            f"軟毛:{category_counts['軟毛']}"
        )

        return result_image, category_mask_vis, all_mask_vis, filtered_mask_vis, count_display, status

    def _process_bw_count_mode(
        self,
        image: np.ndarray,
        detections: List[DetectedRegion],
        all_mask_vis: np.ndarray,
        filtered_mask_vis: np.ndarray,
        stats: dict,
        hair_class: str,
        selection_mode: str,
        freeform_mask: Optional[np.ndarray],
        x1: int, y1: int, x2: int, y2: int,
        sam_points_per_side: int,
        use_tiling: bool,
        overlay_alpha: float,
        show_markers: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray, str, str]:
        """Process detections in black/white count mode."""
        self._detections = detections
        self._classified_hairs = []

        # Create visualization using rainbow colors
        if len(detections) > 0:
            result_image = visualize_single_hairs(
                image, detections,
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
        else:
            result_image = image.copy()
            if selection_mode == "freeform" and freeform_mask is not None:
                contours, _ = cv2.findContours(
                    freeform_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result_image, contours, -1, (255, 255, 255), 2)
            else:
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Build status text
        class_label = "黒髭 (Black)" if hair_class == "black" else "白髭 (White)"
        mode_labels = {"freeform": "フリーハンド", "rectangle": "矩形", "coordinates": "座標入力"}
        mode_label = mode_labels.get(selection_mode, selection_mode)
        tiles_info = f", tiles={stats.get('tiles', 1)}" if use_tiling else ""

        status_lines = [
            f"【色別のカウント - {class_label}】 検出数: {len(detections)}",
            f"選択モード: {mode_label}",
        ]

        if selection_mode == "freeform" and freeform_mask is not None:
            mask_pixels = int(np.sum(freeform_mask > 0))
            status_lines.append(f"Mask area: {mask_pixels} pixels")

        status_lines.extend([
            f"Region: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px",
            f"Total masks: {stats['total']} (points_per_side={sam_points_per_side}{tiles_info})",
            f"Filtered: area_small={stats['filtered_area_small']}, "
            f"area_large={stats['filtered_area_large']}, "
            f"aspect={stats['filtered_aspect']}, "
            f"brightness={stats['filtered_brightness']}",
        ])

        if selection_mode == "freeform":
            status_lines.append(f"  outside_mask={stats.get('filtered_outside_mask', 0)}")

        if 'mean_brightness' in stats:
            status_lines.append(f"Mean brightness: {stats['mean_brightness']:.1f}")

        status = "\n".join(status_lines)
        count_display = f"検出数: {len(detections)}"

        # category_mask is None for bw_count mode
        return result_image, None, all_mask_vis, filtered_mask_vis, count_display, status

    def preview_region(
        self, editor_data: dict,
        coord_x1: int, coord_y1: int, coord_x2: int, coord_y2: int
    ) -> Optional[np.ndarray]:
        """Preview the selected region."""
        image = ImageHandler.extract_image_from_editor(editor_data)
        if image is None:
            return None

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
        if self._last_mode == "毛質別のカウント":
            return self._get_thickness_details()
        else:
            return self._get_bw_count_details()

    def _get_thickness_details(self) -> str:
        """Get details for thickness classification mode."""
        if not self._classified_hairs:
            return "No detections yet"

        lines = [f"Total: {len(self._classified_hairs)} hairs\n"]

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

    def _get_bw_count_details(self) -> str:
        """Get details for black/white count mode."""
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
    """Create Gradio application"""
    app = EdgeDetectionAppV3()

    with gr.Blocks(
        title="Hair Detection v3 - Thickness & BW Count",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 髭検出・分類

        - **毛質別のカウント**: 太さ（幅）に基づいて4種類に分類
        - **色別のカウント**: 黒髭・白髭を個別にカウント
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_editor = gr.ImageEditor(
                    label="Input Image",
                    type="numpy",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=5),
                    height=500,
                )

                with gr.Accordion("検出モード", open=True):
                    detection_mode = gr.Radio(
                        choices=["毛質別のカウント", "色別のカウント"],
                        value="毛質別のカウント",
                        label="検出モード",
                    )

                with gr.Accordion("領域選択", open=True):
                    selection_mode = gr.Radio(
                        choices=[("自由な範囲指定", "freeform"), ("矩形描画", "rectangle"), ("座標", "coordinates")],
                        value="coordinates",
                        label="Selection Mode",
                    )

                with gr.Accordion("座標入力", open=True):
                    with gr.Row():
                        coord_x1 = gr.Number(label="X1 (左)", value=400, precision=0)
                        coord_y1 = gr.Number(label="Y1 (上)", value=440, precision=0)
                    with gr.Row():
                        coord_x2 = gr.Number(label="X2 (右)", value=690, precision=0)
                        coord_y2 = gr.Number(label="Y2 (下)", value=555, precision=0)
                    preview_btn = gr.Button("Preview Region", size="sm")
                    coord_preview = gr.Image(label="Coordinate Preview", type="numpy", height=200)

                with gr.Accordion("髭色クラス", open=True):
                    hair_class = gr.Radio(
                        choices=[("黒ヒゲ", "black"), ("白ヒゲ", "white")],
                        value="black",
                        label="Hair Color Class",
                    )

                with gr.Accordion("SAM Settings (共通)", open=False):
                    sam_points_per_side = gr.Slider(
                        minimum=32, maximum=128, value=64, step=8,
                        label="SAM Points Per Side",
                        info="Sampling density (64-96 recommended)"
                    )
                    pred_iou_thresh = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                        label="Predicted IoU Threshold",
                        info="論文デフォルト: 0.88 (高い=厳しい, マスク数減少)"
                    )
                    stability_score_thresh = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.6, step=0.01,
                        label="Stability Score Threshold",
                        info="論文デフォルト: 0.95 (高い=厳しい, QualityFilter処理時間短縮)"
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

                # ===== Thickness mode parameters =====
                with gr.Group(visible=True) as thickness_params_group:
                    with gr.Accordion("Filter Parameters (フィルタ) - 毛質別モード", open=True):
                        min_area = gr.Slider(
                            minimum=1, maximum=50, value=5, step=1,
                            label="最小面積"
                        )
                        max_area = gr.Slider(
                            minimum=50, maximum=5000, value=100, step=10,
                            label="最大面積"
                        )
                        min_aspect = gr.Slider(
                            minimum=1.0, maximum=5.0, value=1.0, step=0.1,
                            label="最小アスペクト比"
                        )
                        brightness_threshold = gr.Slider(
                            minimum=0.90, maximum=1.20, value=1.00, step=0.01,
                            label="明るさの閾値",
                            info="黒髭: マスク明るさ < 平均×閾値 | 白髭: マスク明るさ > 平均×閾値"
                        )
                        gr.Markdown("**膨張処理**")
                        dilation_kernel = gr.Slider(
                            minimum=0, maximum=15, value=0, step=1,
                            label="Dilation Kernel Size",
                            info="0=OFF, 奇数値推奨 (3, 5, 7...)"
                        )
                        dilation_iterations = gr.Slider(
                            minimum=1, maximum=5, value=1, step=1,
                            label="Dilation Iterations"
                        )

                    with gr.Accordion("太さ閾値", open=True):
                        gr.Markdown("""
                        髭の太さ（外接矩形の短辺）で分類する閾値:
                        - 剛毛 >= Coarse閾値
                        - 硬毛 >= Thick閾値 (かつ < Coarse閾値)
                        - 中間毛 >= Medium閾値 (かつ < Thick閾値)
                        - 軟毛 < Medium閾値
                        """)
                        threshold_coarse = gr.Slider(
                            minimum=1, maximum=20, value=8, step=0.5,
                            label="剛毛閾値 [px]"
                        )
                        threshold_thick = gr.Slider(
                            minimum=1, maximum=15, value=6, step=0.5,
                            label="硬毛閾値 [px]"
                        )
                        threshold_medium = gr.Slider(
                            minimum=1, maximum=10, value=4, step=0.5,
                            label="中間毛閾値 [px]"
                        )

                # ===== BW Count mode parameters =====
                with gr.Group(visible=False) as bw_count_params_group:
                    with gr.Accordion("ヒゲの検出用パラメータ", open=True):
                        bw_min_area = gr.Slider(
                            minimum=1, maximum=100, value=5, step=1,
                            label="最小面積"
                        )
                        bw_max_area = gr.Slider(
                            minimum=100, maximum=5000, value=2000, step=100,
                            label="最大面積"
                        )
                        bw_min_aspect = gr.Slider(
                            minimum=1.0, maximum=5.0, value=1.2, step=0.1,
                            label="最小アスペクト比"
                        )
                        bw_brightness_threshold = gr.Slider(
                            minimum=0.70, maximum=1.30, value=1.14, step=0.02,
                            label="明るさの閾値",
                            info="黒髭: マスク明るさ < 平均×閾値 | 白髭: マスク明るさ > 平均×閾値"
                        )
                        gr.Markdown("**膨張処理**")
                        bw_dilation_kernel = gr.Slider(
                            minimum=0, maximum=15, value=3, step=1,
                            label="Dilation Kernel Size",
                            info="0=OFF, 奇数値推奨 (3, 5, 7...) 検出領域を拡大"
                        )
                        bw_dilation_iterations = gr.Slider(
                            minimum=1, maximum=5, value=1, step=1,
                            label="膨張処理の繰り返し回数",
                        )

                    gr.Markdown("""
                    ### 黒白分類パラメータの説明

                    **Brightness Threshold (明るさ閾値)** は検出の核となるパラメータです:

                    | クラス | 判定ロジック |
                    |--------|------------|
                    | 黒髭 | マスク明るさ < ROI平均×閾値 → 検出 |
                    | 白髭 | マスク明るさ > ROI平均×閾値 → 検出 |

                    **調整のヒント:**
                    - 黒髭が検出されない → 閾値を上げる (1.2-1.3)
                    - 白髭が検出されない → 閾値を下げる (0.85-0.90)
                    - 誤検出が多い → 最小アスペクト比を上げる (1.5-2.0)

                    **最小/最大面積**: マスクのピクセル面積。小さすぎる/大きすぎる検出を除外
                    **最小アスペクト比**: 細長さのフィルタ。1.0=すべて許可、2.0=細長いもののみ
                    **膨張処理**: 検出マスクを膨張させて隣接ピクセルを含める。0=OFF
                    """)

                with gr.Accordion("検出したマスクの重複除去", open=True):
                    overlap_threshold = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                        label="Overlap Threshold",
                        info="重複判定の閾値。高い値=より多くのマスクを保持、低い値=より厳しく重複除去"
                    )

                with gr.Accordion("検出結果の可視化", open=False):
                    overlay_alpha = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.15, step=0.05,
                        label="Overlay Alpha"
                    )
                    show_markers = gr.Checkbox(value=False, label="Show Center Markers")

                detect_btn = gr.Button("検出開始", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_image = gr.Image(
                    label="検出結果 (色 = 毛質)",
                    type="numpy",
                    height=350
                )
                with gr.Row():
                    category_mask_image = gr.Image(
                        label="毛質別のマスク",
                        type="numpy",
                        height=200
                    )
                    filtered_mask_image = gr.Image(
                        label="フィルター後のマスク",
                        type="numpy",
                        height=200
                    )
                all_mask_image = gr.Image(
                    label="フィルター前の全マスク",
                    type="numpy",
                    height=200
                )

                count_display = gr.Textbox(label="カウント結果", lines=1)
                status_text = gr.Textbox(label="ステータス", lines=14)

                details_btn = gr.Button("詳細を表示")
                details_text = gr.Textbox(label="詳細", lines=10)

        # Mode visibility toggle
        def toggle_mode(mode):
            is_thickness = (mode == "毛質別のカウント")
            return (
                gr.update(visible=is_thickness),
                gr.update(visible=not is_thickness),
            )

        detection_mode.change(
            fn=toggle_mode,
            inputs=[detection_mode],
            outputs=[thickness_params_group, bw_count_params_group]
        )

        # Legend
        gr.Markdown("""
        ---
        ### 色の凡例

        **毛質別のカウントモード:**

        | 色 | カテゴリ | 説明 |
        |----|----------|------|
        | 赤 | 剛毛 (Coarse) | 非常に太い髭 |
        | オレンジ | 硬毛 (Thick) | 太い髭 |
        | 緑 | 中間毛 (Medium) | 中程度の髭 |
        | 紫 | 軟毛 (Fine) | 細い髭 |

        **色別のカウントモード:**
        - 各髭はレインボーカラーで個別に色分け表示

        ### 黒白分類のパラメータ解説

        **Brightness Threshold の仕組み:**
        - 各マスクの平均明るさを、ROI（検出領域）全体の平均明るさと比較
        - **黒髭**: `マスク明るさ < ROI平均 × 閾値` で検出。閾値↑ = 感度上昇
        - **白髭**: `マスク明るさ > ROI平均 × 閾値` で検出。閾値↓ = 感度上昇
        """)

        detect_btn.click(
            fn=app.detect_and_classify,
            inputs=[
                image_editor,
                detection_mode,
                hair_class,
                sam_points_per_side,
                pred_iou_thresh,
                stability_score_thresh,
                use_tiling,
                tile_size,
                tile_overlap,
                # Thickness mode params
                min_area,
                max_area,
                min_aspect,
                brightness_threshold,
                dilation_kernel,
                dilation_iterations,
                threshold_coarse,
                threshold_thick,
                threshold_medium,
                # BW Count mode params
                bw_min_area,
                bw_max_area,
                bw_min_aspect,
                bw_brightness_threshold,
                bw_dilation_kernel,
                bw_dilation_iterations,
                # Shared
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