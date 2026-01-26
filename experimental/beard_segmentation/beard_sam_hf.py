#!/usr/bin/env python3
"""
髭検出アプリケーション - Hugging Face SAM版（簡易セットアップ）

Grounded SAMより簡単にセットアップ可能なバージョン。
Hugging Face Transformersを使用。

セットアップ:
  pip install transformers torch torchvision
  pip install opencv-python

SAMのみ使用（自動マスク生成 or ポイント/ボックス指定）
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import random
import sys

# Transformers SAMのインポート
SAM_AVAILABLE = False
try:
    import torch
    from transformers import SamModel, SamProcessor
    SAM_AVAILABLE = True
    print("Hugging Face SAM: 利用可能")
except ImportError:
    print("警告: transformers がインストールされていません")
    print("  pip install transformers torch torchvision")


def get_best_device():
    """利用可能な最速デバイスを取得（CUDA > MPS > CPU）"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def inputs_to_device(inputs, device):
    """入力をデバイスに転送（MPS用にfloat32変換）"""
    result = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            # MPSはfloat64をサポートしないのでfloat32に変換
            if v.dtype == torch.float64:
                v = v.float()
            result[k] = v.to(device)
        else:
            result[k] = v
    return result


class HuggingFaceSAMDetector:
    """Hugging Face版SAMを使った検出器"""

    def __init__(self, model_name: str = "facebook/sam-vit-base"):
        """
        Args:
            model_name: SAMモデル名
                - "facebook/sam-vit-base" (軽量・高速)
                - "facebook/sam-vit-large"
                - "facebook/sam-vit-huge" (高精度)
        """
        self.device = get_best_device()
        print(f"デバイス: {self.device}")
        print(f"モデルを読み込み中: {model_name}")

        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        print("SAM読み込み完了")

    def segment_with_points(self, image: np.ndarray,
                           input_points: List[Tuple[int, int]],
                           input_labels: List[int] = None) -> np.ndarray:
        """
        ポイント指定でセグメンテーション

        Args:
            image: BGR画像
            input_points: クリックポイントのリスト [(x, y), ...]
            input_labels: 各ポイントのラベル (1=前景, 0=背景)

        Returns:
            マスク画像
        """
        if input_labels is None:
            input_labels = [1] * len(input_points)  # 全て前景

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.processor(
            image_rgb,
            input_points=[input_points],
            input_labels=[input_labels],
            return_tensors="pt"
        )
        inputs = inputs_to_device(inputs, self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        # 最もスコアの高いマスクを選択
        scores = outputs.iou_scores.cpu().numpy()[0][0]
        best_idx = np.argmax(scores)
        mask = masks[0][0][best_idx].numpy().astype(np.uint8) * 255

        return mask

    def segment_with_box(self, image: np.ndarray,
                        box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        バウンディングボックス指定でセグメンテーション

        Args:
            image: BGR画像
            box: [x1, y1, x2, y2]

        Returns:
            マスク画像
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = self.processor(
            image_rgb,
            input_boxes=[[list(box)]],
            return_tensors="pt"
        )
        inputs = inputs_to_device(inputs, self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )

        scores = outputs.iou_scores.cpu().numpy()[0][0]
        best_idx = np.argmax(scores)
        mask = masks[0][0][best_idx].numpy().astype(np.uint8) * 255

        return mask

    def auto_segment(self, image: np.ndarray,
                    points_per_side: int = 16) -> List[Dict]:
        """
        自動マスク生成（グリッドポイントを使用）

        Args:
            image: BGR画像
            points_per_side: グリッドのポイント数

        Returns:
            検出されたマスクのリスト
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # グリッドポイントを生成
        results = []
        processed_areas = np.zeros((h, w), dtype=np.uint8)

        step_x = w // points_per_side
        step_y = h // points_per_side

        for i in range(points_per_side):
            for j in range(points_per_side):
                px = step_x // 2 + i * step_x
                py = step_y // 2 + j * step_y

                # 既に処理済みの領域はスキップ
                if processed_areas[py, px] > 0:
                    continue

                try:
                    # このポイントでセグメンテーション
                    inputs = self.processor(
                        image_rgb,
                        input_points=[[(px, py)]],
                        input_labels=[[1]],
                        return_tensors="pt"
                    )
                    inputs = inputs_to_device(inputs, self.device)

                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    masks = self.processor.image_processor.post_process_masks(
                        outputs.pred_masks.cpu(),
                        inputs["original_sizes"].cpu(),
                        inputs["reshaped_input_sizes"].cpu()
                    )

                    scores = outputs.iou_scores.cpu().numpy()[0][0]
                    best_idx = np.argmax(scores)
                    mask = masks[0][0][best_idx].numpy().astype(np.uint8) * 255

                    area = cv2.countNonZero(mask)

                    # 小さすぎる or 大きすぎるマスクはスキップ
                    if area < 50 or area > (h * w * 0.5):
                        continue

                    # 既存マスクとの重複チェック
                    overlap = cv2.bitwise_and(processed_areas, mask)
                    if cv2.countNonZero(overlap) > area * 0.5:
                        continue

                    results.append({
                        'mask': mask,
                        'area': area,
                    })

                    # 処理済みとしてマーク
                    processed_areas = cv2.bitwise_or(processed_areas, mask)

                except Exception as e:
                    continue

        print(f"グリッド探索で {len(results)} 個の領域を検出")
        return results


def split_mask_into_individual_hairs(image: np.ndarray, region_mask: np.ndarray,
                                      threshold: int = 80,
                                      min_area: int = 5,
                                      max_area: int = 3000) -> List[np.ndarray]:
    """
    大きなマスク領域内から個々のヒゲを分離する

    Args:
        image: 元画像（BGR）
        region_mask: SAM等で得た大きなマスク
        threshold: ヒゲ検出の閾値（暗さ）
        min_area: 最小ヒゲ面積
        max_area: 最大ヒゲ面積

    Returns:
        個々のヒゲマスクのリスト
    """
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # マスク領域内のみ処理
    masked_gray = gray.copy()
    masked_gray[region_mask == 0] = 255  # マスク外は白に

    # コントラスト強調（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked_gray)

    # ガウシアンブラー
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 適応的閾値処理（ヒゲは暗いので反転）
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 通常閾値処理
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    # 両方のマスクを組み合わせ
    hair_mask = cv2.bitwise_and(adaptive, binary)

    # 元の領域マスク内のみに制限
    hair_mask = cv2.bitwise_and(hair_mask, region_mask)

    # モルフォロジー演算でノイズ除去＆ヒゲを分離
    # まず細いカーネルでオープニング（細かいノイズ除去）
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel_small)

    # エロージョンでヒゲ同士を分離
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    hair_mask = cv2.erode(hair_mask, kernel_erode, iterations=1)

    # 連結成分分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hair_mask)

    individual_masks = []
    for i in range(1, num_labels):  # 0は背景
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            # この連結成分だけのマスクを作成
            mask = (labels == i).astype(np.uint8) * 255

            # 少し膨張させて元のサイズに近づける
            mask = cv2.dilate(mask, kernel_erode, iterations=1)

            individual_masks.append(mask)

    return individual_masks


class BeardCounterSAMHF:
    """Hugging Face SAM版 髭カウンター"""

    def __init__(self, image_path: str):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")

        self.display_image = self.original_image.copy()
        self.inpaint_preview = None
        self.drawing = False
        self.mode = 'point'  # 'point', 'box', 'lasso', 'auto', 'split'
        self.start_point = None
        self.lasso_points: List[Tuple[int, int]] = []
        self.click_points: List[Tuple[int, int]] = []  # SAM用のポイント
        self.click_labels: List[int] = []  # 1=前景, 0=背景

        self.beard_regions: List[dict] = []
        self.active_mask_indices: List[int] = []

        # 未分割の大きな領域（SAMの結果）
        self.pending_large_mask: Optional[np.ndarray] = None
        self.auto_split_mode = True  # ボックス選択時に自動でヒゲ分離

        self.removal_percentage = 0
        self.min_area = 5  # 個々のヒゲ用に小さく
        self.max_area = 3000
        self.hair_threshold = 80  # ヒゲ検出閾値
        self.inpaint_radius = 3
        self.selection_mode = 'random'
        self.show_inpaint_preview = False
        self.random_seed = 42

        # SAM検出器（遅延初期化）
        self.sam_detector = None

        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

    def init_sam(self, model_size: str = "base"):
        """SAMを初期化"""
        if self.sam_detector is not None:
            return True

        if not SAM_AVAILABLE:
            print("SAMは利用できません")
            return False

        model_map = {
            "base": "facebook/sam-vit-base",
            "large": "facebook/sam-vit-large",
            "huge": "facebook/sam-vit-huge"
        }
        model_name = model_map.get(model_size, model_map["base"])

        try:
            self.sam_detector = HuggingFaceSAMDetector(model_name)
            return True
        except Exception as e:
            print(f"SAM初期化エラー: {e}")
            return False

    def segment_current_points(self):
        """現在のクリックポイントでセグメンテーション"""
        if not self.click_points:
            print("ポイントがありません。画像をクリックしてください。")
            return

        if not self.init_sam():
            return

        print(f"セグメンテーション中... ({len(self.click_points)} ポイント)")
        mask = self.sam_detector.segment_with_points(
            self.original_image,
            self.click_points,
            self.click_labels
        )

        area = cv2.countNonZero(mask)
        if area < self.min_area:
            print(f"面積が小さすぎます ({area})")
            return

        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else 0

        self.beard_regions.append({
            'mask': mask,
            'area': area,
            'centroid': (cx, cy),
            'source': 'sam_point'
        })

        # ポイントをクリア
        self.click_points = []
        self.click_labels = []

        self.update_active_masks()
        self.update_display()
        print(f"領域を追加 (面積: {area})")

    def segment_with_box(self, box: Tuple[int, int, int, int], auto_split: bool = True):
        """ボックスでセグメンテーション"""
        if not self.init_sam():
            return

        print("ボックスからセグメンテーション中...")
        mask = self.sam_detector.segment_with_box(self.original_image, box)

        area = cv2.countNonZero(mask)
        if area < 100:
            print(f"面積が小さすぎます ({area})")
            return

        if auto_split:
            # 大きな領域の場合は自動的にヒゲに分離
            print(f"領域 (面積: {area}) を個々のヒゲに分離中...")
            self.split_mask_into_hairs(mask)
        else:
            # 分離せずそのまま追加
            self.pending_large_mask = mask
            M = cv2.moments(mask)
            cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else 0
            cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else 0

            self.beard_regions.append({
                'mask': mask,
                'area': area,
                'centroid': (cx, cy),
                'source': 'sam_box'
            })
            self.update_active_masks()
            self.update_display()
            print(f"領域を追加 (面積: {area}) - 'h'キーでヒゲに分離できます")

    def split_mask_into_hairs(self, mask: np.ndarray):
        """大きなマスクを個々のヒゲに分離して追加"""
        individual_masks = split_mask_into_individual_hairs(
            self.original_image, mask,
            threshold=self.hair_threshold,
            min_area=self.min_area,
            max_area=self.max_area
        )

        added = 0
        for hair_mask in individual_masks:
            area = cv2.countNonZero(hair_mask)
            M = cv2.moments(hair_mask)
            cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else 0
            cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else 0

            self.beard_regions.append({
                'mask': hair_mask,
                'area': area,
                'centroid': (cx, cy),
                'source': 'hair_split'
            })
            added += 1

        self.update_active_masks()
        self.update_display()
        print(f"{added} 本のヒゲを検出しました")

    def split_last_region(self):
        """最後に追加した大きな領域をヒゲに分離"""
        if not self.beard_regions:
            print("分離する領域がありません")
            return

        # 最後の領域を取得して削除
        last_region = self.beard_regions.pop()
        mask = last_region['mask']

        print(f"領域 (面積: {last_region['area']}) を分離中...")
        self.split_mask_into_hairs(mask)

    def auto_segment_all(self):
        """自動で全領域をセグメント"""
        if not self.init_sam():
            return

        print("自動セグメンテーション中（時間がかかる場合があります）...")
        results = self.sam_detector.auto_segment(self.original_image)

        added = 0
        for result in results:
            area = result['area']
            if self.min_area < area < self.max_area:
                mask = result['mask']
                M = cv2.moments(mask)
                cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else 0
                cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else 0

                self.beard_regions.append({
                    'mask': mask,
                    'area': area,
                    'centroid': (cx, cy),
                    'source': 'sam_auto'
                })
                added += 1

        self.update_active_masks()
        self.update_display()
        print(f"{added} 個の領域を追加")

    def on_percentage_change(self, value):
        self.removal_percentage = value
        self.update_active_masks()
        self.update_display()

    def update_active_masks(self):
        if not self.beard_regions:
            self.active_mask_indices = []
            return

        total = len(self.beard_regions)
        target = int(total * self.removal_percentage / 100)

        if self.selection_mode == 'random':
            random.seed(self.random_seed)
            indices = list(range(total))
            random.shuffle(indices)
            self.active_mask_indices = sorted(indices[:target])
        elif self.selection_mode == 'area_large':
            sorted_idx = sorted(range(total),
                              key=lambda i: self.beard_regions[i]['area'],
                              reverse=True)
            self.active_mask_indices = sorted(sorted_idx[:target])
        else:
            self.active_mask_indices = list(range(target))

        if self.show_inpaint_preview:
            self.update_inpaint_preview()

    def get_combined_mask(self) -> np.ndarray:
        h, w = self.original_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for idx in self.active_mask_indices:
            if 0 <= idx < len(self.beard_regions):
                mask = cv2.bitwise_or(mask, self.beard_regions[idx]['mask'])
        return mask

    def update_inpaint_preview(self):
        mask = self.get_combined_mask()
        if cv2.countNonZero(mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel)
            self.inpaint_preview = cv2.inpaint(
                self.original_image, mask, self.inpaint_radius, cv2.INPAINT_TELEA
            )
        else:
            self.inpaint_preview = self.original_image.copy()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

            if self.mode == 'point':
                # 前景ポイントを追加
                self.click_points.append((x, y))
                self.click_labels.append(1)
                self.update_display()
                print(f"前景ポイント追加: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == 'point':
                # 背景ポイントを追加
                self.click_points.append((x, y))
                self.click_labels.append(0)
                self.update_display()
                print(f"背景ポイント追加: ({x}, {y})")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.mode == 'box':
                temp = self.get_base_display_image()
                cv2.rectangle(temp, self.start_point, (x, y), (0, 255, 0), 2)
                self.display_image = temp

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == 'box' and self.start_point:
                x1 = min(self.start_point[0], x)
                y1 = min(self.start_point[1], y)
                x2 = max(self.start_point[0], x)
                y2 = max(self.start_point[1], y)
                if x2 - x1 > 10 and y2 - y1 > 10:
                    self.segment_with_box((x1, y1, x2, y2), auto_split=self.auto_split_mode)

    def get_base_display_image(self) -> np.ndarray:
        if self.show_inpaint_preview and self.inpaint_preview is not None:
            image = self.inpaint_preview.copy()
        else:
            image = self.original_image.copy()

            for i, region in enumerate(self.beard_regions):
                color = self.colors[i % len(self.colors)]
                mask = region['mask']

                if i in self.active_mask_indices:
                    overlay = image.copy()
                    image[mask > 0] = (0, 0, 255)
                    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                else:
                    overlay = image.copy()
                    image[mask > 0] = color
                    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # クリックポイントを描画
        for pt, label in zip(self.click_points, self.click_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(image, pt, 5, color, -1)
            cv2.circle(image, pt, 7, (255, 255, 255), 1)

        return image

    def update_display(self):
        self.display_image = self.get_base_display_image()

        total = len(self.beard_regions)
        active = len(self.active_mask_indices)
        info = f"Total: {total} | Remove: {active} ({self.removal_percentage}%)"
        cv2.rectangle(self.display_image, (5, 10), (400, 35), (0, 0, 0), -1)
        cv2.putText(self.display_image, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    def save_mask(self):
        mask = self.get_combined_mask()
        cv2.imwrite("beard_mask.png", mask)
        print("マスクを保存: beard_mask.png")

    def on_threshold_change(self, value):
        self.hair_threshold = value

    def run(self):
        window = 'Beard Counter (SAM)'
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self.mouse_callback)
        cv2.createTrackbar('Remove %', window, 0, 100, self.on_percentage_change)
        cv2.createTrackbar('Hair Thresh', window, self.hair_threshold, 150, self.on_threshold_change)

        print("=" * 70)
        print("髭検出アプリ - Hugging Face SAM版 (個別ヒゲ分離機能付き)")
        print("=" * 70)
        print("操作:")
        print("  [モード]")
        print("    '1': ポイントモード（左クリック=前景, 右クリック=背景）")
        print("    '2': ボックスモード（ドラッグで矩形選択→自動でヒゲ分離）")
        print("    '3': ボックスモード（分離なし）")
        print("  [検出]")
        print("    Enter: ポイントからセグメント実行")
        print("    'h': 最後の領域を個々のヒゲに分離")
        print("  [選択]")
        print("    'r': ランダム選択 / 'l': 面積大きい順")
        print("  [表示]")
        print("    'p': Inpaintingプレビュー切替")
        print("  [保存]")
        print("    'm': マスク保存")
        print("  [その他]")
        print("    'u': Undo / 'c': 全クリア / 'q': 終了")
        print("=" * 70)
        print("※ボックスモードでドラッグすると、SAMで領域検出→自動でヒゲに分離します")
        print("=" * 70)

        self.update_display()

        while True:
            cv2.imshow(window, self.display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):
                self.mode = 'point'
                print("モード: ポイント")
            elif key == ord('2'):
                self.mode = 'box'
                self.auto_split_mode = True
                print("モード: ボックス（自動ヒゲ分離ON）")
            elif key == ord('3'):
                self.mode = 'box'
                self.auto_split_mode = False
                print("モード: ボックス（分離なし）")
            elif key == 13:  # Enter
                self.segment_current_points()
            elif key == ord('h'):
                # 最後の領域をヒゲに分離
                self.split_last_region()
            elif key == ord('p'):
                self.show_inpaint_preview = not self.show_inpaint_preview
                if self.show_inpaint_preview:
                    self.update_inpaint_preview()
                self.update_display()
            elif key == ord('r'):
                self.selection_mode = 'random'
                self.update_active_masks()
                self.update_display()
            elif key == ord('l'):
                self.selection_mode = 'area_large'
                self.update_active_masks()
                self.update_display()
            elif key == ord('m'):
                self.save_mask()
            elif key == ord('c'):
                self.beard_regions = []
                self.active_mask_indices = []
                self.click_points = []
                self.click_labels = []
                self.pending_large_mask = None
                self.display_image = self.original_image.copy()
                print("全クリアしました")
            elif key == ord('u'):
                if self.beard_regions:
                    self.beard_regions.pop()
                    self.update_active_masks()
                    self.update_display()
                    print(f"Undo: 残り {len(self.beard_regions)} 領域")
                elif self.click_points:
                    self.click_points.pop()
                    self.click_labels.pop()
                    self.update_display()

        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("使用方法: python beard_sam_hf.py <画像パス>")
        sys.exit(1)

    app = BeardCounterSAMHF(sys.argv[1])
    app.run()


if __name__ == "__main__":
    main()
