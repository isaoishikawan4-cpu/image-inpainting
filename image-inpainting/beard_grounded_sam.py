#!/usr/bin/env python3
"""
髭検出アプリケーション - Grounded SAM版 + LaMa Inpainting統合
- Grounding DINO + SAM でテキストプロンプトから髭を自動検出
- 従来のルールベース検出も併用可能
- スライドバーで除去する髭の割合(%)を指定
- LaMa Inpainting機能付き（高品質）
- OpenCV Inpainting機能付き（高速フォールバック）

必要なセットアップ:
1. pip install torch torchvision  # CUDA対応版推奨
2. pip install segment-anything
3. pip install groundingdino-py
4. pip install simple-lama-inpainting
5. モデルのダウンロード（下記参照）
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import random
import os
import sys
from PIL import Image

# Grounded SAM関連のインポート（オプション）
GROUNDED_SAM_AVAILABLE = False
try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    GROUNDED_SAM_AVAILABLE = True
    print("PyTorch & SAM: 利用可能")
except ImportError:
    print("警告: segment-anything がインストールされていません")
    print("  pip install segment-anything")

try:
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    GROUNDING_DINO_AVAILABLE = True
    print("Grounding DINO: 利用可能")
except ImportError:
    GROUNDING_DINO_AVAILABLE = False
    print("警告: groundingdino がインストールされていません")
    print("  pip install groundingdino-py")

# LaMa Inpainting モジュール（同一ディレクトリの core から）
LAMA_MODULE_AVAILABLE = False
try:
    from core.inpainting import InpaintingEngine
    LAMA_MODULE_AVAILABLE = True
    print("LaMa Inpainting: 利用可能")
except ImportError as e:
    print(f"警告: LaMa Inpainting が利用できません: {e}")
    print("  pip install simple-lama-inpainting")


class LamaInpaintingBridge:
    """LaMa Inpainting へのブリッジ（遅延初期化・フォールバック対応）"""

    def __init__(self):
        self._engine = None
        self._available = None
        self._initialization_error = None

    def is_available(self) -> bool:
        """LaMa が利用可能かチェック（初回呼び出し時に初期化）"""
        if self._available is None:
            self._try_initialize()
        return self._available

    def _try_initialize(self):
        """遅延初期化を試行"""
        if not LAMA_MODULE_AVAILABLE:
            self._available = False
            self._initialization_error = "LaMa モジュールがインポートできません"
            print(f"LaMa Inpainting: {self._initialization_error}")
            return

        try:
            print("LaMa Inpainting モデルを初期化中...")
            self._engine = InpaintingEngine()
            self._available = True
            print("LaMa Inpainting: 初期化成功")
        except Exception as e:
            self._available = False
            self._initialization_error = f"初期化エラー: {e}"
            print(f"LaMa Inpainting: {self._initialization_error}")

    def inpaint(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        LaMa Inpainting を実行

        Args:
            image_bgr: OpenCV BGR 画像
            mask: バイナリマスク (0/255)

        Returns:
            Inpainted BGR 画像

        Raises:
            RuntimeError: LaMa が利用できない場合
        """
        if not self.is_available():
            raise RuntimeError(f"LaMa Inpainting は利用できません: {self._initialization_error}")

        # BGR -> RGB -> PIL
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # マスクを PIL L モードに変換
        mask_pil = Image.fromarray(mask, mode='L')

        # LaMa Inpainting 実行
        result_pil = self._engine.inpaint(image_pil, mask_pil)

        # PIL -> RGB -> BGR
        result_rgb = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        return result_bgr


class GroundedSAMDetector:
    """Grounded SAM を使った髭検出器"""

    def __init__(self,
                 sam_checkpoint: str = "sam_vit_h_4b8939.pth",
                 sam_model_type: str = "vit_h",
                 grounding_dino_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 grounding_dino_checkpoint: str = "groundingdino_swint_ogc.pth",
                 device: str = "cuda"):

        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"デバイス: {self.device}")

        self.sam_predictor = None
        self.grounding_dino_model = None

        # SAMの読み込み
        if os.path.exists(sam_checkpoint):
            print(f"SAMモデルを読み込み中: {sam_checkpoint}")
            sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            print("SAM: 読み込み完了")
        else:
            print(f"警告: SAMチェックポイントが見つかりません: {sam_checkpoint}")
            print("  ダウンロード: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")

        # Grounding DINOの読み込み
        if GROUNDING_DINO_AVAILABLE and os.path.exists(grounding_dino_checkpoint):
            print(f"Grounding DINOモデルを読み込み中: {grounding_dino_checkpoint}")
            self.grounding_dino_model = load_model(grounding_dino_config, grounding_dino_checkpoint)
            print("Grounding DINO: 読み込み完了")
        else:
            print(f"警告: Grounding DINOチェックポイントが見つかりません")

    def detect_with_prompt(self, image: np.ndarray, text_prompt: str,
                          box_threshold: float = 0.3,
                          text_threshold: float = 0.25) -> List[Dict]:
        """
        テキストプロンプトで対象を検出しセグメンテーション

        Args:
            image: BGR画像 (OpenCV形式)
            text_prompt: 検出対象のテキスト（例: "beard. facial hair. stubble."）
            box_threshold: バウンディングボックスの信頼度閾値
            text_threshold: テキストマッチングの閾値

        Returns:
            検出結果のリスト [{'mask': np.ndarray, 'box': [x1,y1,x2,y2], 'confidence': float}, ...]
        """
        if self.grounding_dino_model is None or self.sam_predictor is None:
            print("エラー: モデルが読み込まれていません")
            return []

        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Grounding DINOで検出
        # groundingdinoの入力形式に変換
        import groundingdino.datasets.transforms as T
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image_pil = Image.fromarray(image_rgb)
        image_transformed, _ = transform(image_pil, None)

        boxes, logits, phrases = predict(
            model=self.grounding_dino_model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        if len(boxes) == 0:
            print("検出された領域がありません")
            return []

        # ボックスを画像サイズにスケール
        h, w = image.shape[:2]
        boxes_scaled = boxes * torch.tensor([w, h, w, h])
        boxes_xyxy = boxes_scaled.cpu().numpy()

        # SAMでセグメンテーション
        self.sam_predictor.set_image(image_rgb)

        results = []
        for i, (box, conf) in enumerate(zip(boxes_xyxy, logits)):
            # SAMにボックスを入力
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )

            # 最もスコアの高いマスクを選択
            best_idx = np.argmax(scores)
            mask = masks[best_idx].astype(np.uint8) * 255

            results.append({
                'mask': mask,
                'box': box.tolist(),
                'confidence': float(conf),
                'phrase': phrases[i] if i < len(phrases) else ""
            })

        print(f"{len(results)} 個の領域を検出しました")
        return results

    def is_available(self) -> bool:
        """モデルが利用可能かどうか"""
        return self.sam_predictor is not None and self.grounding_dino_model is not None


class BeardCounterGroundedSAM:
    """Grounded SAM + 従来手法のハイブリッド髭検出"""

    def __init__(self, image_path: str):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")

        self.image_path = image_path
        self.display_image = self.original_image.copy()
        self.inpaint_preview = None
        self.drawing = False
        self.mode = 'grounded_sam'  # 'grounded_sam', 'lasso', 'rectangle', 'click_fill'
        self.start_point = None
        self.lasso_points: List[Tuple[int, int]] = []

        # 検出された髭領域
        self.beard_regions: List[dict] = []
        self.active_mask_indices: List[int] = []

        # パラメータ
        self.removal_percentage = 0
        self.tolerance = 30
        self.min_area = 10
        self.max_area = 50000
        self.threshold_value = 80
        self.inpaint_radius = 3
        self.selection_mode = 'random'

        # Grounded SAM パラメータ
        self.text_prompt = "beard. facial hair. stubble."
        self.box_threshold = 0.25
        self.text_threshold = 0.20

        self.show_inpaint_preview = False
        self.random_seed = 42

        # Grounded SAM 検出器（遅延初期化）
        self.gsam_detector = None

        # LaMa Inpainting ブリッジ（遅延初期化）
        self.lama_bridge = None
        self.use_lama = True  # デフォルトで LaMa を使用

        # カラーパレット
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
        ]

    def init_grounded_sam(self):
        """Grounded SAMを初期化（重いので必要時のみ）"""
        if self.gsam_detector is not None:
            return self.gsam_detector.is_available()

        if not GROUNDED_SAM_AVAILABLE or not GROUNDING_DINO_AVAILABLE:
            print("Grounded SAMは利用できません。従来手法を使用してください。")
            return False

        print("Grounded SAMを初期化中...")
        try:
            self.gsam_detector = GroundedSAMDetector()
            return self.gsam_detector.is_available()
        except Exception as e:
            print(f"Grounded SAM初期化エラー: {e}")
            return False

    def init_lama_inpainting(self) -> bool:
        """LaMa Inpainting を初期化（必要時のみ）"""
        if self.lama_bridge is not None:
            return self.lama_bridge.is_available()

        self.lama_bridge = LamaInpaintingBridge()
        return self.lama_bridge.is_available()

    def detect_with_grounded_sam(self):
        """Grounded SAMで髭を検出"""
        if not self.init_grounded_sam():
            print("Grounded SAMが利用できません")
            return

        print(f"プロンプト: {self.text_prompt}")
        print(f"閾値 - Box: {self.box_threshold}, Text: {self.text_threshold}")

        results = self.gsam_detector.detect_with_prompt(
            self.original_image,
            self.text_prompt,
            self.box_threshold,
            self.text_threshold
        )

        for result in results:
            mask = result['mask']
            area = cv2.countNonZero(mask)

            if area < self.min_area or area > self.max_area:
                continue

            # 重心を計算
            M = cv2.moments(mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0

            self.beard_regions.append({
                'mask': mask,
                'area': area,
                'centroid': (cx, cy),
                'confidence': result.get('confidence', 0),
                'source': 'grounded_sam'
            })

        self.update_active_masks()
        self.update_display()
        print(f"Grounded SAMで {len(results)} 個の領域を検出")

    def on_percentage_change(self, value):
        self.removal_percentage = value
        self.update_active_masks()
        self.update_display()

    def on_inpaint_radius_change(self, value):
        self.inpaint_radius = max(1, value)
        if self.show_inpaint_preview:
            self.update_inpaint_preview()

    def on_box_threshold_change(self, value):
        self.box_threshold = value / 100.0

    def on_text_threshold_change(self, value):
        self.text_threshold = value / 100.0

    def update_active_masks(self):
        if not self.beard_regions:
            self.active_mask_indices = []
            return

        total_count = len(self.beard_regions)
        target_count = int(total_count * self.removal_percentage / 100)

        if self.selection_mode == 'random':
            random.seed(self.random_seed)
            all_indices = list(range(total_count))
            random.shuffle(all_indices)
            self.active_mask_indices = sorted(all_indices[:target_count])
        elif self.selection_mode == 'area_large':
            sorted_indices = sorted(range(total_count),
                                   key=lambda i: self.beard_regions[i]['area'],
                                   reverse=True)
            self.active_mask_indices = sorted(sorted_indices[:target_count])
        elif self.selection_mode == 'area_small':
            sorted_indices = sorted(range(total_count),
                                   key=lambda i: self.beard_regions[i]['area'])
            self.active_mask_indices = sorted(sorted_indices[:target_count])
        elif self.selection_mode == 'confidence':
            # 信頼度順（Grounded SAM用）
            sorted_indices = sorted(range(total_count),
                                   key=lambda i: self.beard_regions[i].get('confidence', 0),
                                   reverse=True)
            self.active_mask_indices = sorted(sorted_indices[:target_count])
        else:
            self.active_mask_indices = list(range(target_count))

        if self.show_inpaint_preview:
            self.update_inpaint_preview()

    def get_combined_mask(self) -> np.ndarray:
        h, w = self.original_image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for idx in self.active_mask_indices:
            if 0 <= idx < len(self.beard_regions):
                mask = self.beard_regions[idx]['mask']
                combined_mask = cv2.bitwise_or(combined_mask, mask)
        return combined_mask

    def get_dilated_mask(self, dilation: int = 2) -> np.ndarray:
        mask = self.get_combined_mask()
        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation*2+1, dilation*2+1))
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def update_inpaint_preview(self):
        """Inpainting のプレビューを更新（LaMa または OpenCV）"""
        mask = self.get_dilated_mask(dilation=2)

        if cv2.countNonZero(mask) == 0:
            self.inpaint_preview = self.original_image.copy()
            return

        # LaMa Inpainting を試行
        if self.use_lama:
            try:
                if self.init_lama_inpainting():
                    print("LaMa Inpainting を実行中...")
                    self.inpaint_preview = self.lama_bridge.inpaint(
                        self.original_image, mask
                    )
                    print("LaMa Inpainting 完了")
                    return
                else:
                    print("LaMa が利用できないため、OpenCV にフォールバック")
            except Exception as e:
                print(f"LaMa Inpainting エラー: {e}")
                print("OpenCV にフォールバック")

        # OpenCV Inpainting (フォールバック)
        self.inpaint_preview = cv2.inpaint(
            self.original_image, mask, self.inpaint_radius, cv2.INPAINT_TELEA
        )

    def mouse_callback(self, event, x, y, flags, param):
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
        if self.show_inpaint_preview and self.inpaint_preview is not None:
            return self.inpaint_preview.copy()

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

        return image

    def flood_fill_at_point(self, x: int, y: int):
        """従来のFlood Fill検出"""
        h, w = self.original_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        if not (0 <= x < w and 0 <= y < h):
            return

        cv2.floodFill(gray, mask, (x, y), 255,
                     self.tolerance, self.tolerance,
                     cv2.FLOODFILL_MASK_ONLY | (255 << 8))

        result_mask = mask[1:-1, 1:-1]
        area = cv2.countNonZero(result_mask)

        if area < self.min_area or area > self.max_area:
            return

        M = cv2.moments(result_mask)
        cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else x
        cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else y

        self.beard_regions.append({
            'mask': result_mask,
            'area': area,
            'centroid': (cx, cy),
            'source': 'flood_fill'
        })

        self.update_active_masks()
        self.update_display()

    def detect_beards_in_rectangle(self, end_point: Tuple[int, int]):
        x1 = min(self.start_point[0], end_point[0])
        y1 = min(self.start_point[1], end_point[1])
        x2 = max(self.start_point[0], end_point[0])
        y2 = max(self.start_point[1], end_point[1])

        if x2 - x1 < 10 or y2 - y1 < 10:
            return

        roi = self.original_image[y1:y2, x1:x2]
        self._detect_beards_in_roi(roi, x1, y1)

    def detect_beards_in_lasso(self):
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
        """従来のルールベース検出"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        adaptive = cv2.adaptiveThreshold(blurred, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
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

                M = cv2.moments(full_mask)
                cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else offset_x
                cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else offset_y

                self.beard_regions.append({
                    'mask': full_mask,
                    'area': area,
                    'centroid': (cx, cy),
                    'source': 'rule_based'
                })
                added_count += 1

        self.update_active_masks()
        self.update_display()
        print(f"領域内で {added_count} 本の髭を検出")

    def update_display(self):
        self.display_image = self.get_base_display_image()

        total = len(self.beard_regions)
        active = len(self.active_mask_indices)
        inpaint_method = "LaMa" if self.use_lama else "OpenCV"

        info = f"Total: {total} | Remove: {active} ({self.removal_percentage}%) | Mode: {self.selection_mode} | Inpaint: {inpaint_method}"
        cv2.rectangle(self.display_image, (5, 10), (650, 35), (0, 0, 0), -1)
        cv2.putText(self.display_image, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    def save_mask(self, filename: str = "beard_mask.png"):
        mask = self.get_dilated_mask(dilation=2)
        cv2.imwrite(filename, mask)
        print(f"マスクを保存: {filename}")

    def save_inpainted(self, filename: str = None):
        """Inpainting 結果を保存"""
        if self.inpaint_preview is None:
            self.update_inpaint_preview()

        # ファイル名の決定
        if filename is None:
            suffix = "_lama" if self.use_lama else "_opencv"
            filename = f"beard_inpainted{suffix}.png"

        cv2.imwrite(filename, self.inpaint_preview)
        method = "LaMa" if self.use_lama else "OpenCV"
        print(f"Inpainting結果を保存 ({method}): {filename}")

    def run(self):
        window_name = 'Beard Counter (Grounded SAM + LaMa)'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        cv2.createTrackbar('Remove %', window_name, 0, 100, self.on_percentage_change)
        cv2.createTrackbar('Inpaint R', window_name, self.inpaint_radius, 20, self.on_inpaint_radius_change)
        cv2.createTrackbar('Box Thresh', window_name, int(self.box_threshold*100), 100, self.on_box_threshold_change)
        cv2.createTrackbar('Text Thresh', window_name, int(self.text_threshold*100), 100, self.on_text_threshold_change)

        print("=" * 70)
        print("髭検出アプリケーション - Grounded SAM版 + LaMa Inpainting")
        print("=" * 70)
        print("操作方法:")
        print("  [検出モード]")
        print("    'g': Grounded SAMで自動検出（要GPU・モデル）")
        print("    '1': クリック塗りつぶしモード")
        print("    '2': 投げ縄モード")
        print("    '3': 矩形モード")
        print("  [選択モード]")
        print("    'r': ランダム / 'l': 面積大 / 's': 面積小 / 'f': 信頼度順")
        print("  [表示]")
        print("    'p': LaMa Inpaintingプレビュー（高品質・GPU推奨）")
        print("    'o': OpenCV Inpaintingプレビュー（高速・CPU）")
        print("    'n': 新しいランダムシード")
        print("  [保存]")
        print("    'm': マスク保存 / 'i': Inpainting結果保存")
        print("  [その他]")
        print("    't': プロンプト変更 / 'u': Undo / 'c': 全クリア / 'q': 終了")
        print("=" * 70)
        print(f"プロンプト: {self.text_prompt}")
        print(f"Inpainting方式: {'LaMa (高品質)' if self.use_lama else 'OpenCV (高速)'}")

        while True:
            display = self.display_image.copy()

            mode_names = {'grounded_sam': 'Grounded SAM', 'click_fill': 'クリック',
                         'lasso': '投げ縄', 'rectangle': '矩形'}
            inpaint_mode = "LaMa" if self.use_lama else "OpenCV"
            status = f"Mode: {mode_names.get(self.mode, self.mode)} | Inpaint: {inpaint_mode} | Prompt: {self.text_prompt[:25]}"
            cv2.putText(display, status, (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('g'):
                print("Grounded SAMで検出中...")
                self.detect_with_grounded_sam()
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
            elif key == ord('l'):
                self.selection_mode = 'area_large'
                self.update_active_masks()
                self.update_display()
            elif key == ord('s'):
                self.selection_mode = 'area_small'
                self.update_active_masks()
                self.update_display()
            elif key == ord('f'):
                self.selection_mode = 'confidence'
                self.update_active_masks()
                self.update_display()
            elif key == ord('p'):
                # LaMa Inpainting を有効化してプレビュー
                self.use_lama = True
                self.show_inpaint_preview = not self.show_inpaint_preview
                if self.show_inpaint_preview:
                    print("LaMa Inpainting プレビューを有効化")
                    self.update_inpaint_preview()
                self.update_display()
            elif key == ord('o'):
                # OpenCV Inpainting (従来方式)
                self.use_lama = False
                self.show_inpaint_preview = not self.show_inpaint_preview
                if self.show_inpaint_preview:
                    print("OpenCV Inpainting プレビューを有効化")
                    self.update_inpaint_preview()
                self.update_display()
            elif key == ord('n'):
                self.random_seed = random.randint(0, 10000)
                self.update_active_masks()
                self.update_display()
            elif key == ord('t'):
                print(f"現在のプロンプト: {self.text_prompt}")
                new_prompt = input("新しいプロンプト（空白でキャンセル）: ").strip()
                if new_prompt:
                    self.text_prompt = new_prompt
                    print(f"プロンプトを変更: {self.text_prompt}")
            elif key == ord('m'):
                self.save_mask()
            elif key == ord('i'):
                self.save_inpainted()
            elif key == ord('u'):
                if self.beard_regions:
                    self.beard_regions.pop()
                    self.update_active_masks()
                    self.update_display()
            elif key == ord('c'):
                self.beard_regions = []
                self.active_mask_indices = []
                self.inpaint_preview = None
                self.display_image = self.original_image.copy()

        cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        image_path = "beard_sample.jpg"
        print(f"使用方法: python beard_grounded_sam.py <画像パス>")
    else:
        image_path = sys.argv[1]

    try:
        app = BeardCounterGroundedSAM(image_path)
        app.run()
    except ValueError as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
