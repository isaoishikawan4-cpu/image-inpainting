#!/usr/bin/env python3
"""
髭カウント＆Inpaintingアプリケーション
- 髭を検出してカウント
- スライドバーで除去する髭の割合(%)を指定
- 指定%に応じたマスクを生成
- OpenCV inpaintingでプレビュー
- マスクと結果画像を保存
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import random


class BeardCounterFloodFill:
    def __init__(self, image_path: str):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")

        self.image_path = image_path
        self.display_image = self.original_image.copy()
        self.inpaint_preview = None
        self.drawing = False
        self.mode = 'lasso'  # 'lasso', 'rectangle', 'click_fill'
        self.start_point = None
        self.lasso_points: List[Tuple[int, int]] = []

        # 検出された髭領域のリスト（マスク、面積、重心座標）
        self.beard_regions: List[dict] = []
        self.active_mask_indices: List[int] = []  # 現在有効なマスクのインデックス

        # パラメータ
        self.removal_percentage = 0  # 除去する髭の割合(0-100)
        self.tolerance = 30
        self.min_area = 10
        self.max_area = 5000
        self.threshold_value = 80
        self.inpaint_radius = 3  # inpaintingの半径
        self.selection_mode = 'random'  # 'random', 'area_large', 'area_small', 'sequential'

        self.show_inpaint_preview = False
        self.random_seed = 42  # 再現性のため

        # カラーパレット
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]

    def on_percentage_change(self, value):
        """スライドバーの値が変わったときのコールバック"""
        self.removal_percentage = value
        self.update_active_masks()
        self.update_display()

    def on_inpaint_radius_change(self, value):
        """Inpaint半径スライドバーのコールバック"""
        self.inpaint_radius = max(1, value)
        if self.show_inpaint_preview:
            self.update_inpaint_preview()

    def on_threshold_change(self, value):
        """Threshold値スライドバーのコールバック"""
        self.threshold_value = max(1, value)
        print(f"Threshold changed: {self.threshold_value}")
        # 既存の検出結果はそのまま保持、次回検出から新しい値を適用

    def update_active_masks(self):
        """%に基づいて有効なマスクを更新"""
        if not self.beard_regions:
            self.active_mask_indices = []
            return

        total_count = len(self.beard_regions)
        target_count = int(total_count * self.removal_percentage / 100)

        if self.selection_mode == 'random':
            # ランダム選択（シード固定で再現性確保）
            random.seed(self.random_seed)
            all_indices = list(range(total_count))
            random.shuffle(all_indices)
            self.active_mask_indices = sorted(all_indices[:target_count])

        elif self.selection_mode == 'area_large':
            # 面積が大きい順
            sorted_indices = sorted(
                range(total_count),
                key=lambda i: self.beard_regions[i]['area'],
                reverse=True
            )
            self.active_mask_indices = sorted(sorted_indices[:target_count])

        elif self.selection_mode == 'area_small':
            # 面積が小さい順
            sorted_indices = sorted(
                range(total_count),
                key=lambda i: self.beard_regions[i]['area']
            )
            self.active_mask_indices = sorted(sorted_indices[:target_count])

        elif self.selection_mode == 'sequential':
            # 検出順
            self.active_mask_indices = list(range(target_count))

        if self.show_inpaint_preview:
            self.update_inpaint_preview()

    def get_combined_mask(self) -> np.ndarray:
        """有効な髭領域を結合したマスクを取得"""
        h, w = self.original_image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for idx in self.active_mask_indices:
            if 0 <= idx < len(self.beard_regions):
                mask = self.beard_regions[idx]['mask']
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        return combined_mask

    def get_dilated_mask(self, dilation: int = 2) -> np.ndarray:
        """マスクを膨張させる（inpainting用に少し大きめに）"""
        mask = self.get_combined_mask()
        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation*2+1, dilation*2+1))
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def update_inpaint_preview(self):
        """Inpaintingのプレビューを更新"""
        mask = self.get_dilated_mask(dilation=2)

        if cv2.countNonZero(mask) > 0:
            # OpenCV inpainting（Telea法またはNavier-Stokes法）
            self.inpaint_preview = cv2.inpaint(
                self.original_image, mask, self.inpaint_radius, cv2.INPAINT_TELEA
            )
        else:
            self.inpaint_preview = self.original_image.copy()

    def mouse_callback(self, event, x, y, flags, param):
        """マウスイベントのコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.lasso_points = [(x, y)]

            if self.mode == 'click_fill':
                self.flood_fill_at_point(x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.mode in ['lasso', 'rectangle']:
                if self.mode == 'lasso':
                    self.lasso_points.append((x, y))
                    temp_image = self.get_base_display_image()
                    if len(self.lasso_points) > 1:
                        pts = np.array(self.lasso_points, dtype=np.int32)
                        cv2.polylines(temp_image, [pts], False, (0, 255, 0), 2)
                    self.display_image = temp_image
                elif self.mode == 'rectangle':
                    temp_image = self.get_base_display_image()
                    cv2.rectangle(temp_image, self.start_point, (x, y), (0, 255, 0), 2)
                    self.display_image = temp_image

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

            if self.mode == 'lasso' and len(self.lasso_points) > 2:
                self.detect_beards_in_lasso()
            elif self.mode == 'rectangle':
                self.detect_beards_in_rectangle((x, y))

    def get_base_display_image(self) -> np.ndarray:
        """表示用の基本画像を取得"""
        if self.show_inpaint_preview and self.inpaint_preview is not None:
            image = self.inpaint_preview.copy()
        else:
            image = self.original_image.copy()

            # 有効な髭領域を色付けして表示
            for i, region in enumerate(self.beard_regions):
                color = self.colors[i % len(self.colors)]
                mask = region['mask']

                if i in self.active_mask_indices:
                    # 有効（除去対象）：赤系で強調
                    overlay = image.copy()
                    image[mask > 0] = (0, 0, 255)  # 赤
                    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                else:
                    # 無効（残す）：元の色で薄く
                    overlay = image.copy()
                    image[mask > 0] = color
                    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        return image

    def flood_fill_at_point(self, x: int, y: int):
        """指定点からFlood Fill"""
        h, w = self.original_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        if not (0 <= x < w and 0 <= y < h):
            return

        cv2.floodFill(
            gray, mask, (x, y),
            newVal=255,
            loDiff=self.tolerance,
            upDiff=self.tolerance,
            flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        )

        result_mask = mask[1:-1, 1:-1]
        area = cv2.countNonZero(result_mask)

        if area < self.min_area or area > self.max_area:
            print(f"領域面積 ({area}) が範囲外です")
            return

        # 重複チェック
        for region in self.beard_regions:
            overlap = cv2.bitwise_and(result_mask, region['mask'])
            if cv2.countNonZero(overlap) > area * 0.5:
                print("既存の領域と重複")
                return

        # 重心を計算
        M = cv2.moments(result_mask)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = x, y

        self.beard_regions.append({
            'mask': result_mask,
            'area': area,
            'centroid': (cx, cy)
        })

        self.update_active_masks()
        self.update_display()
        print(f"髭 #{len(self.beard_regions)} を検出 (面積: {area})")

    def detect_beards_in_rectangle(self, end_point: Tuple[int, int]):
        """矩形領域内の髭を検出"""
        x1 = min(self.start_point[0], end_point[0])
        y1 = min(self.start_point[1], end_point[1])
        x2 = max(self.start_point[0], end_point[0])
        y2 = max(self.start_point[1], end_point[1])

        if x2 - x1 < 10 or y2 - y1 < 10:
            return

        roi = self.original_image[y1:y2, x1:x2]
        self._detect_beards_in_roi(roi, x1, y1)

    def detect_beards_in_lasso(self):
        """投げ縄領域内の髭を検出"""
        pts = np.array(self.lasso_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)

        if w < 10 or h < 10:
            return

        lasso_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(lasso_mask, [pts], 255)

        roi = self.original_image[y:y+h, x:x+w]
        roi_mask = lasso_mask[y:y+h, x:x+w]

        self._detect_beards_in_roi(roi, x, y, roi_mask)

    def _detect_beards_in_roi(self, roi: np.ndarray, offset_x: int, offset_y: int,
                               region_mask: Optional[np.ndarray] = None):
        """ROI内の髭を検出"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(adaptive, binary)

        if region_mask is not None:
            mask = cv2.bitwise_and(mask, region_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        added_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                full_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
                offset_contour = contour + np.array([offset_x, offset_y])
                cv2.drawContours(full_mask, [offset_contour], -1, 255, -1)

                # 重複チェック
                is_duplicate = False
                for region in self.beard_regions:
                    overlap = cv2.bitwise_and(full_mask, region['mask'])
                    if cv2.countNonZero(overlap) > area * 0.3:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    M = cv2.moments(full_mask)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                    else:
                        cx, cy = offset_x, offset_y

                    self.beard_regions.append({
                        'mask': full_mask,
                        'area': area,
                        'centroid': (cx, cy)
                    })
                    added_count += 1

        self.update_active_masks()
        self.update_display()
        print(f"領域内で {added_count} 本の髭を検出（合計: {len(self.beard_regions)}）")

    def update_display(self):
        """表示を更新"""
        self.display_image = self.get_base_display_image()

        # 情報表示
        total = len(self.beard_regions)
        active = len(self.active_mask_indices)

        info_lines = [
            f"Total: {total} | Remove: {active} ({self.removal_percentage}%)",
            f"Mode: {self.selection_mode}"
        ]

        y_offset = 30
        for line in info_lines:
            cv2.rectangle(self.display_image, (5, y_offset - 20), (350, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(self.display_image, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_offset += 25

    def save_mask(self, filename: str = "beard_mask.png"):
        """マスクを保存"""
        mask = self.get_dilated_mask(dilation=2)
        cv2.imwrite(filename, mask)
        print(f"マスクを保存しました: {filename}")
        return mask

    def save_inpainted(self, filename: str = "beard_inpainted.png"):
        """Inpainting結果を保存"""
        if self.inpaint_preview is None:
            self.update_inpaint_preview()
        cv2.imwrite(filename, self.inpaint_preview)
        print(f"Inpainting結果を保存しました: {filename}")
        return self.inpaint_preview

    def run(self):
        """メインループ"""
        window_name = 'Beard Counter + Inpainting'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # スライドバーを追加
        cv2.createTrackbar('Remove %', window_name, 0, 100, self.on_percentage_change)
        cv2.createTrackbar('Inpaint R', window_name, self.inpaint_radius, 20, self.on_inpaint_radius_change)
        cv2.createTrackbar('Threshold', window_name, self.threshold_value, 255, self.on_threshold_change)

        print("=" * 65)
        print("髭カウント＆Inpaintingアプリケーション")
        print("=" * 65)
        print("操作方法:")
        print("  [モード切替]")
        print("    '1': クリック塗りつぶしモード")
        print("    '2': 投げ縄モード")
        print("    '3': 矩形モード")
        print("  [選択モード]")
        print("    'r': ランダム選択")
        print("    'l': 面積大きい順")
        print("    's': 面積小さい順")
        print("    'd': 検出順（sequential）")
        print("  [表示]")
        print("    'p': Inpaintingプレビュー切替")
        print("    'n': 新しいランダムシード")
        print("  [保存]")
        print("    'm': マスクを保存")
        print("    'i': Inpainting結果を保存")
        print("  [その他]")
        print("    'u': Undo")
        print("    'c': 全クリア")
        print("    'q'/ESC: 終了")
        print("=" * 65)
        print("スライドバーで除去割合(%)とThreshold(検出感度)を調整できます")
        print("=" * 65)

        while True:
            display = self.display_image.copy()

            # ステータスバー
            mode_names = {'click_fill': 'クリック', 'lasso': '投げ縄', 'rectangle': '矩形'}
            preview_status = "ON" if self.show_inpaint_preview else "OFF"
            status = f"Mode: {mode_names.get(self.mode, self.mode)} | " \
                    f"Select: {self.selection_mode} | Preview: {preview_status}"
            cv2.putText(display, status,
                       (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):
                self.mode = 'click_fill'
                print("モード: クリック塗りつぶし")
            elif key == ord('2'):
                self.mode = 'lasso'
                print("モード: 投げ縄")
            elif key == ord('3'):
                self.mode = 'rectangle'
                print("モード: 矩形")
            elif key == ord('r'):
                self.selection_mode = 'random'
                self.update_active_masks()
                self.update_display()
                print("選択モード: ランダム")
            elif key == ord('l'):
                self.selection_mode = 'area_large'
                self.update_active_masks()
                self.update_display()
                print("選択モード: 面積大きい順")
            elif key == ord('s'):
                self.selection_mode = 'area_small'
                self.update_active_masks()
                self.update_display()
                print("選択モード: 面積小さい順")
            elif key == ord('d'):
                self.selection_mode = 'sequential'
                self.update_active_masks()
                self.update_display()
                print("選択モード: 検出順")
            elif key == ord('p'):
                self.show_inpaint_preview = not self.show_inpaint_preview
                if self.show_inpaint_preview:
                    self.update_inpaint_preview()
                self.update_display()
                print(f"Inpaintingプレビュー: {'ON' if self.show_inpaint_preview else 'OFF'}")
            elif key == ord('n'):
                self.random_seed = random.randint(0, 10000)
                self.update_active_masks()
                self.update_display()
                print(f"新しいランダムシード: {self.random_seed}")
            elif key == ord('m'):
                self.save_mask()
            elif key == ord('i'):
                self.save_inpainted()
            elif key == ord('u'):
                if self.beard_regions:
                    self.beard_regions.pop()
                    self.update_active_masks()
                    self.update_display()
                    print(f"取り消しました（残り: {len(self.beard_regions)}）")
            elif key == ord('c'):
                self.beard_regions = []
                self.active_mask_indices = []
                self.inpaint_preview = None
                self.display_image = self.original_image.copy()
                print("全クリアしました")

        cv2.destroyAllWindows()

        total = len(self.beard_regions)
        active = len(self.active_mask_indices)
        print(f"\n最終結果: {total} 本中 {active} 本 ({self.removal_percentage}%) を除去対象")


def main():
    import sys

    if len(sys.argv) < 2:
        image_path = "beard_sample.jpg"
        print(f"使用方法: python beard_counter_floodfill.py <画像パス>")
        print(f"画像パスが指定されていないため、'{image_path}' を使用します")
    else:
        image_path = sys.argv[1]

    try:
        app = BeardCounterFloodFill(image_path)
        app.run()
    except ValueError as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
