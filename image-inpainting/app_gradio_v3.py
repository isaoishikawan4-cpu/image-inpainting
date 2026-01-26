#!/usr/bin/env python3
"""
Gradio版 髭検出・修復アプリケーション v3

改良点:
- Grounded SAM による高精度髭検出（テキストプロンプト対応）
- 矩形選択ツールによる検出領域指定（白ブラシで塗りつぶし）
- ランダム/面積ベースの髭選択（削除割合を指定）
- LaMa Inpainting による高品質修復

使用方法:
    python app_gradio_v3.py

必要なチェックポイント:
    - sam_vit_h_4b8939.pth
    - groundingdino_swint_ogc.pth
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, List, Dict
import random
import os
import sys

# =============================================================================
# 依存ライブラリのインポートと利用可能フラグ
# =============================================================================

# PyTorch
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    print(f"PyTorch: 利用可能 (CUDA: {torch.cuda.is_available()})")
except ImportError:
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

# Grounding DINO
GROUNDING_DINO_AVAILABLE = False
try:
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    GROUNDING_DINO_AVAILABLE = True
    print("Grounding DINO: 利用可能")
except ImportError:
    print("警告: groundingdino がインストールされていません")
    print("  pip install groundingdino-py")

# LaMa Inpainting
LAMA_AVAILABLE = False
try:
    from core.inpainting import InpaintingEngine, BeardThinningProcessor
    from core.image_utils import (
        resize_image_if_needed,
        convert_to_binary_mask,
        numpy_to_pil
    )
    LAMA_AVAILABLE = True
    print("LaMa Inpainting: 利用可能")
except ImportError as e:
    print(f"警告: LaMa Inpainting が利用できません: {e}")

# Config
try:
    import config
except ImportError:
    # フォールバック設定
    class config:
        MAX_IMAGE_SIZE = 2048
        BRUSH_RADIUS_DEFAULT = 20
        DEFAULT_THINNING_LEVELS = [30, 50, 70, 100]


# =============================================================================
# GroundedSAMDetector クラス
# =============================================================================

class GroundedSAMDetector:
    """Grounded SAM を使った髭検出器"""

    def __init__(
        self,
        sam_checkpoint: str = "sam_vit_h_4b8939.pth",
        sam_model_type: str = "vit_h",
        grounding_dino_config: str = None,  # None の場合は pip パッケージから自動取得
        grounding_dino_checkpoint: str = "groundingdino_swint_ogc.pth"
    ):
        self._sam_checkpoint = sam_checkpoint
        self._sam_model_type = sam_model_type
        self._dino_config = grounding_dino_config
        self._dino_checkpoint = grounding_dino_checkpoint
        self._initialized = False
        self.sam_predictor = None
        self.grounding_dino_model = None
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    def _find_checkpoint(self, filename: str) -> Optional[str]:
        """チェックポイントファイルを探索"""
        search_paths = [
            filename,
            os.path.join(os.path.dirname(__file__), filename),
            os.path.join(os.path.dirname(__file__), "..", filename),
            os.path.expanduser(f"~/{filename}"),
        ]
        for path in search_paths:
            if os.path.exists(path):
                return path
        return None

    def initialize(self) -> bool:
        """モデルを初期化（遅延初期化）"""
        if self._initialized:
            return True

        if not SAM_AVAILABLE or not GROUNDING_DINO_AVAILABLE:
            print("Grounded SAM ライブラリが利用できません")
            return False

        # SAM の読み込み
        sam_path = self._find_checkpoint(self._sam_checkpoint)
        if sam_path is None:
            print(f"SAM チェックポイントが見つかりません: {self._sam_checkpoint}")
            return False

        try:
            print(f"SAM モデルを読み込み中: {sam_path}")
            sam = sam_model_registry[self._sam_model_type](checkpoint=sam_path)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            print(f"SAM: 読み込み完了 (device={self.device})")
        except Exception as e:
            print(f"SAM 初期化エラー: {e}")
            return False

        # Grounding DINO の読み込み
        dino_path = self._find_checkpoint(self._dino_checkpoint)
        if dino_path is None:
            print(f"Grounding DINO チェックポイントが見つかりません: {self._dino_checkpoint}")
            return False

        # Config ファイルを探索（pip パッケージ内から自動取得）
        dino_config_path = None

        # 1. 明示的に指定された場合
        if self._dino_config and os.path.exists(self._dino_config):
            dino_config_path = self._dino_config
        else:
            # 2. pip パッケージ内から取得（推奨）
            try:
                import groundingdino
                package_config = os.path.join(
                    os.path.dirname(groundingdino.__file__),
                    "config",
                    "GroundingDINO_SwinT_OGC.py"
                )
                if os.path.exists(package_config):
                    dino_config_path = package_config
                    print(f"Grounding DINO config: pip パッケージから取得")
            except ImportError:
                pass

            # 3. フォールバック: 従来のパス探索
            if dino_config_path is None:
                fallback_paths = [
                    os.path.join(os.path.dirname(__file__), "..", "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"),
                    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                ]
                for path in fallback_paths:
                    if os.path.exists(path):
                        dino_config_path = path
                        break

        if dino_config_path is None:
            print(f"Grounding DINO config が見つかりません（pip install groundingdino-py を実行してください）")
            return False

        try:
            print(f"Grounding DINO モデルを読み込み中: {dino_path}")
            self.grounding_dino_model = load_model(dino_config_path, dino_path)
            print("Grounding DINO: 読み込み完了")
        except Exception as e:
            print(f"Grounding DINO 初期化エラー: {e}")
            return False

        self._initialized = True
        return True

    def is_available(self) -> bool:
        """モデルが利用可能かどうか"""
        return self._initialized and self.sam_predictor is not None and self.grounding_dino_model is not None

    def detect_with_prompt(
        self,
        image_rgb: np.ndarray,
        text_prompt: str = "beard. facial hair. stubble.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20
    ) -> List[Dict]:
        """
        テキストプロンプトで対象を検出しセグメンテーション（beard_grounded_sam.pyと同じロジック）

        Args:
            image_rgb: RGB画像
            text_prompt: 検出用テキスト
            box_threshold: ボックス検出閾値
            text_threshold: テキストマッチング閾値

        Returns:
            検出結果のリスト
        """
        if not self.is_available():
            raise RuntimeError("Grounded SAM が初期化されていません")

        # Grounding DINO で検出
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
            print("Grounding DINO: 検出された領域がありません")
            return []

        # ボックスを画像サイズにスケール
        h, w = image_rgb.shape[:2]
        boxes_scaled = boxes * torch.tensor([w, h, w, h])
        boxes_xyxy = boxes_scaled.cpu().numpy()

        # SAM でセグメンテーション
        self.sam_predictor.set_image(image_rgb)

        results = []
        for i, (box, conf) in enumerate(zip(boxes_xyxy, logits)):
            # SAM でマスク生成
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )

            best_idx = np.argmax(scores)
            mask = masks[best_idx].astype(np.uint8) * 255

            results.append({
                'mask': mask,
                'box': box.tolist(),
                'confidence': float(conf),
                'phrase': phrases[i] if i < len(phrases) else "",
                'source': 'grounded_sam'
            })

        print(f"Grounded SAM: {len(results)} 個の領域を検出")
        return results

    def detect_with_prompt_in_region(
        self,
        image_rgb: np.ndarray,
        region_box: Tuple[int, int, int, int],
        text_prompt: str = "beard. facial hair. stubble.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        min_area: int = 10,
        max_area: int = 50000
    ) -> List[Dict]:
        """
        矩形領域内でテキストプロンプトによる髭検出
        クロップした領域に対してGrounded SAMを適用

        Args:
            image_rgb: RGB画像
            region_box: 検出領域 (x1, y1, x2, y2)
            text_prompt: 検出用テキスト
            box_threshold: ボックス検出閾値
            text_threshold: テキストマッチング閾値
            min_area: 最小面積
            max_area: 最大面積

        Returns:
            検出結果のリスト
        """
        if not self.is_available():
            raise RuntimeError("Grounded SAM が初期化されていません")

        x1, y1, x2, y2 = region_box
        h_orig, w_orig = image_rgb.shape[:2]

        # 領域をクロップ
        roi = image_rgb[y1:y2, x1:x2].copy()
        print(f"ROI size: {roi.shape[1]}x{roi.shape[0]}")

        # クロップした領域に対してGrounded SAMを適用
        print(f"Grounded SAM検出パラメータ: prompt='{text_prompt}', box_thresh={box_threshold}, text_thresh={text_threshold}")
        roi_results = self.detect_with_prompt(
            roi, text_prompt, box_threshold, text_threshold
        )

        print(f"Grounded SAM raw results: {len(roi_results)} 個")
        for i, r in enumerate(roi_results):
            area = cv2.countNonZero(r['mask'])
            print(f"  [{i}] phrase='{r['phrase']}', confidence={r['confidence']:.3f}, area={area}")

        if not roi_results:
            print("ROI内で検出された領域がありません")
            return []

        # 結果を元画像座標系に変換
        results = []
        for result in roi_results:
            roi_mask = result['mask']
            area = cv2.countNonZero(roi_mask)

            # 面積フィルタ
            if area < min_area or area > max_area:
                print(f"  領域を除外（面積フィルタ）: area={area} (範囲: {min_area}-{max_area})")
                continue

            # フルサイズのマスクを作成
            full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = roi_mask

            # 重心を計算（元画像座標系）
            M = cv2.moments(full_mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

            results.append({
                'mask': full_mask,
                'area': area,
                'centroid': (cx, cy),
                'confidence': result['confidence'],
                'phrase': result['phrase'],
                'source': 'grounded_sam'
            })

        print(f"Grounded SAM (ROI): {len(results)} 個の領域を検出")
        return results


# =============================================================================
# BeardRegionManager クラス
# =============================================================================

class BeardRegionManager:
    """髭領域の管理と選択ロジック"""

    def __init__(self):
        self.regions: List[Dict] = []
        self.active_indices: List[int] = []
        self.random_seed: int = 42

        # カラーパレット（beard_grounded_sam.pyと同様）
        self.colors = [
            (0, 255, 0),    # 緑
            (0, 0, 255),    # 青
            (255, 255, 0),  # 黄
            (255, 0, 255),  # マゼンタ
            (0, 255, 255),  # シアン
            (128, 0, 0),    # 暗赤
            (0, 128, 0),    # 暗緑
            (0, 0, 128),    # 暗青
            (255, 128, 0),  # オレンジ
            (128, 0, 255),  # 紫
        ]

    def clear(self):
        """全領域をクリア"""
        self.regions = []
        self.active_indices = []

    def add_regions(self, new_regions: List[Dict]):
        """検出された領域を追加"""
        self.regions.extend(new_regions)

    def update_selection(
        self,
        removal_percentage: int,
        selection_mode: str,
        new_seed: bool = False
    ) -> List[int]:
        """
        削除対象の選択を更新

        Args:
            removal_percentage: 削除割合 (0-100%)
            selection_mode: 'random', 'area_large', 'area_small', 'confidence'
            new_seed: 新しいシードを使用

        Returns:
            選択されたインデックスのリスト
        """
        if not self.regions:
            self.active_indices = []
            return []

        if new_seed:
            self.random_seed = random.randint(0, 10000)

        total = len(self.regions)
        target_count = int(total * removal_percentage / 100)

        if selection_mode == 'random':
            random.seed(self.random_seed)
            indices = list(range(total))
            random.shuffle(indices)
            self.active_indices = sorted(indices[:target_count])
        elif selection_mode == 'area_large':
            sorted_idx = sorted(
                range(total),
                key=lambda i: self.regions[i]['area'],
                reverse=True
            )
            self.active_indices = sorted(sorted_idx[:target_count])
        elif selection_mode == 'area_small':
            sorted_idx = sorted(
                range(total),
                key=lambda i: self.regions[i]['area']
            )
            self.active_indices = sorted(sorted_idx[:target_count])
        elif selection_mode == 'confidence':
            sorted_idx = sorted(
                range(total),
                key=lambda i: self.regions[i].get('confidence', 0),
                reverse=True
            )
            self.active_indices = sorted(sorted_idx[:target_count])
        else:
            self.active_indices = list(range(target_count))

        return self.active_indices

    def get_combined_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """選択された領域の統合マスクを取得"""
        h, w = image_shape
        combined = np.zeros((h, w), dtype=np.uint8)
        for idx in self.active_indices:
            if 0 <= idx < len(self.regions):
                combined = cv2.bitwise_or(combined, self.regions[idx]['mask'])
        return combined

    def get_dilated_mask(self, image_shape: Tuple[int, int], dilation: int = 2) -> np.ndarray:
        """膨張処理済みマスクを取得"""
        mask = self.get_combined_mask(image_shape)
        if dilation > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation * 2 + 1, dilation * 2 + 1)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def create_colored_display(
        self,
        image_rgb: np.ndarray,
        highlight_active: bool = True
    ) -> np.ndarray:
        """
        各髭を異なる色でハイライトした表示用画像を作成
        (beard_grounded_sam.py の get_base_display_image と同等)

        Args:
            image_rgb: RGB画像
            highlight_active: 削除対象を赤で強調表示するか

        Returns:
            ハイライトされた画像
        """
        display = image_rgb.copy()

        for i, region in enumerate(self.regions):
            mask = region['mask']
            color = self.colors[i % len(self.colors)]

            if highlight_active and i in self.active_indices:
                # 削除対象: 赤色で強調（半透明）
                overlay = display.copy()
                display[mask > 0] = (255, 0, 0)  # 赤
                cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            else:
                # 保持: カラーパレットの色で表示（薄め）
                overlay = display.copy()
                display[mask > 0] = color
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        return display


# =============================================================================
# グローバル状態
# =============================================================================

detector: Optional[GroundedSAMDetector] = None
region_manager: BeardRegionManager = BeardRegionManager()
current_image: Optional[np.ndarray] = None
_inpainting_engine: Optional[InpaintingEngine] = None
_thinning_processor: Optional[BeardThinningProcessor] = None


def get_detector() -> Optional[GroundedSAMDetector]:
    """検出器を取得（遅延初期化）"""
    global detector
    if detector is None:
        detector = GroundedSAMDetector()
    if not detector._initialized:
        detector.initialize()
    return detector if detector.is_available() else None


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
    return _inpainting_engine


def get_thinning_processor() -> Optional[BeardThinningProcessor]:
    """薄め処理プロセッサを取得（遅延初期化）"""
    global _thinning_processor
    if _thinning_processor is None and LAMA_AVAILABLE:
        try:
            print("LaMa プロセッサを初期化中...")
            _thinning_processor = BeardThinningProcessor()
            print("LaMa プロセッサ: 初期化完了")
        except Exception as e:
            print(f"LaMa プロセッサ初期化エラー: {e}")
    return _thinning_processor


# =============================================================================
# ユーティリティ関数
# =============================================================================

def extract_rectangle_from_editor(editor_data: dict) -> Optional[Tuple[int, int, int, int]]:
    """
    ImageEditor のデータから矩形座標を抽出

    Returns:
        (x1, y1, x2, y2) または None
    """
    if editor_data is None:
        return None

    try:
        mask = None
        if isinstance(editor_data, dict):
            if 'layers' in editor_data and len(editor_data['layers']) > 0:
                layer = editor_data['layers'][0]
                if isinstance(layer, np.ndarray) and len(layer.shape) == 3:
                    gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                    mask = (gray > 128).astype(np.uint8) * 255
            elif 'composite' in editor_data:
                composite = editor_data['composite']
                if isinstance(composite, np.ndarray) and len(composite.shape) == 3:
                    if composite.shape[2] == 4:
                        mask = composite[:, :, 3]

        if mask is None or np.max(mask) == 0:
            return None

        # 輪郭からバウンディングボックスを取得
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        x_min, y_min = mask.shape[1], mask.shape[0]
        x_max, y_max = 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        return (x_min, y_min, x_max, y_max)

    except Exception as e:
        print(f"矩形抽出エラー: {e}")
        return None


# =============================================================================
# ルールベース髭検出（beard_grounded_sam.pyと同等）
# =============================================================================

def detect_beards_rule_based(
    image_rgb: np.ndarray,
    region_box: Tuple[int, int, int, int],
    threshold_value: int = 80,
    min_area: int = 10,
    max_area: int = 5000
) -> List[Dict]:
    """
    ルールベースで髭を1本ずつ検出（beard_grounded_sam.pyの_detect_beards_in_roiと同等）

    Args:
        image_rgb: RGB画像
        region_box: 検出領域 (x1, y1, x2, y2)
        threshold_value: 二値化閾値
        min_area: 最小面積
        max_area: 最大面積

    Returns:
        検出結果のリスト
    """
    x1, y1, x2, y2 = region_box
    h_orig, w_orig = image_rgb.shape[:2]

    # ROIを切り出し（BGRに変換）
    roi_rgb = image_rgb[y1:y2, x1:x2]
    roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

    # グレースケール変換
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE（コントラスト強調）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ガウシアンブラー
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 適応的閾値処理
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # 固定閾値処理
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # 両方のマスクをAND
    mask = cv2.bitwise_and(adaptive, binary)

    # モルフォロジー処理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 輪郭検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # フルサイズのマスクを作成
            full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            offset_contour = contour + np.array([x1, y1])
            cv2.drawContours(full_mask, [offset_contour], -1, 255, -1)

            # 重心を計算
            M = cv2.moments(full_mask)
            cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else x1
            cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else y1

            results.append({
                'mask': full_mask,
                'area': area,
                'centroid': (cx, cy),
                'confidence': 1.0,  # ルールベースは信頼度1.0
                'source': 'rule_based'
            })

    print(f"ルールベース検出: {len(results)} 本の髭を検出")
    return results


# =============================================================================
# Tab 1: 髭検出
# =============================================================================

def process_beard_detection(
    image: np.ndarray,
    editor_data: dict,
    detection_mode: str,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    threshold_value: int,
    min_area: int,
    max_area: int
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    髭を検出（Grounded SAM または ルールベース）

    Args:
        image: 入力画像
        editor_data: ImageEditor からのデータ（矩形描画）
        detection_mode: 検出モード（"ルールベース（1本ずつ検出）" or "Grounded SAM"）
        text_prompt: 検出用テキストプロンプト（Grounded SAM用）
        box_threshold: ボックス検出閾値（Grounded SAM用）
        text_threshold: テキストマッチング閾値（Grounded SAM用）
        threshold_value: 二値化閾値（ルールベース用）
        min_area: 最小面積
        max_area: 最大面積

    Returns:
        (表示用画像, マスク画像, ステータス)
    """
    global current_image, region_manager

    if image is None:
        return None, None, "画像をアップロードしてください"

    current_image = image.copy()
    region_manager.clear()

    # RGB 変換
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = image.copy()

    # 矩形座標を取得
    rect = extract_rectangle_from_editor(editor_data)
    if rect is None:
        return image_rgb, None, "矩形を描画してください（白色ブラシで領域を塗りつぶし）"

    x1, y1, x2, y2 = rect

    try:
        use_rule_based = "ルールベース" in detection_mode

        if use_rule_based:
            # ルールベース検出（1本ずつ検出）
            print(f"ルールベース検出中... 領域=({x1}, {y1}, {x2}, {y2}), 閾値={threshold_value}")
            regions = detect_beards_rule_based(
                image_rgb, rect, threshold_value, min_area, max_area
            )
        else:
            # Grounded SAM 検出
            det = get_detector()
            if det is None:
                return image_rgb, None, "Grounded SAM が利用できません。チェックポイントを確認してください。"

            print(f"Grounded SAM 検出中... 領域=({x1}, {y1}, {x2}, {y2})")
            regions = det.detect_with_prompt_in_region(
                image_rgb, rect, text_prompt, box_threshold, text_threshold, min_area, max_area
            )

        region_manager.add_regions(regions)

        # 各髭を異なる色でハイライト（beard_grounded_sam.pyと同様）
        display = region_manager.create_colored_display(image_rgb, highlight_active=False)

        # 統合マスクを作成
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        for region in regions:
            mask = cv2.bitwise_or(mask, region['mask'])

        # 検出領域の矩形を描画
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)

        mode_name = "ルールベース" if use_rule_based else "Grounded SAM"
        status = f"検出完了 [{mode_name}]: {len(regions)} 本の髭を検出 | 領域: ({x1},{y1})-({x2},{y2})"
        return display, mask, status

    except Exception as e:
        return image_rgb, None, f"検出エラー: {str(e)}"


def update_selection_preview(
    removal_percentage: int,
    selection_mode: str,
    new_seed: bool
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    削除対象の選択を更新
    """
    global current_image, region_manager

    if current_image is None or not region_manager.regions:
        return None, None, "先に Grounded SAM で検出を実行してください"

    # 選択モードをマッピング
    mode_map = {
        "ランダム": "random",
        "面積大": "area_large",
        "面積小": "area_small",
        "信頼度順": "confidence"
    }
    mode = mode_map.get(selection_mode, "random")

    active_indices = region_manager.update_selection(removal_percentage, mode, new_seed)

    # RGB 変換
    if len(current_image.shape) == 2:
        image_rgb = cv2.cvtColor(current_image, cv2.COLOR_GRAY2RGB)
    elif current_image.shape[2] == 4:
        image_rgb = cv2.cvtColor(current_image, cv2.COLOR_RGBA2RGB)
    else:
        image_rgb = current_image.copy()

    # 各髭を色分け表示（削除対象は赤、保持はカラーパレット）
    display = region_manager.create_colored_display(image_rgb, highlight_active=True)

    # 統合マスクを取得（削除対象のみ）
    combined_mask = region_manager.get_dilated_mask(image_rgb.shape[:2])

    total = len(region_manager.regions)
    status = f"選択: {len(active_indices)}/{total} 個 ({removal_percentage}%) | モード: {selection_mode} | シード: {region_manager.random_seed}"

    return display, combined_mask, status


# =============================================================================
# Tab 2: LaMa Inpainting
# =============================================================================

def process_lama_inpainting(
    image: Image.Image,
    mask: np.ndarray,
    thinning_levels: List[int],
    progress=gr.Progress()
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    """
    LaMa Inpainting を実行（複数薄め段階）
    """
    if image is None:
        return [], "画像をアップロードしてください"

    if mask is None or np.max(mask) == 0:
        return [], "マスクを指定してください（Tab 1 で生成するか、直接アップロード）"

    if not thinning_levels:
        return [], "少なくとも1つの薄め具合を選択してください"

    processor = get_thinning_processor()
    if processor is None:
        return [], "LaMa が利用できません。simple-lama-inpainting をインストールしてください。"

    try:
        progress(0.1, desc="画像を準備中...")

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
        gallery = [(image, "オリジナル (0%)")]
        for level in sorted(results.keys()):
            caption = f"{level}% 薄め" if level < 100 else "完全除去 (100%)"
            gallery.append((results[level], caption))

        progress(1.0, desc="完了")
        status = f"完了！ {len(results)} 段階の髭薄め画像を生成しました"
        return gallery, status

    except Exception as e:
        return [], f"エラーが発生しました: {str(e)}"


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

    with gr.Blocks(
        title="髭検出・修復アプリ v3",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown("""
        # 髭検出・修復アプリケーション v3

        **新機能:**
        - **ルールベース検出**: 髭を1本ずつ高精度に検出（推奨）
        - **Grounded SAM**: テキストプロンプトによる自動髭検出
        - **矩形選択**: 白ブラシで検出領域を指定
        - **カラーハイライト**: 検出した各髭を異なる色でハイライト表示
        - **削除対象選択**: ランダム/面積大/面積小/信頼度順で削除対象を選択（赤色で強調）
        - **LaMa Inpainting**: 高品質な髭除去・段階的な薄め
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

                # イベント: Inpainting
                inpaint_btn.click(
                    fn=process_lama_inpainting,
                    inputs=[inpaint_image, inpaint_mask, thinning_checkboxes],
                    outputs=[inpaint_gallery, inpaint_status]
                )

        # 説明パネル
        with gr.Accordion("技術情報", open=False):
            gr.Markdown("""
            ## 使用技術

            ### Tab 1: Grounded SAM
            - **Grounding DINO**: テキストプロンプトからオブジェクト検出
            - **SAM**: セグメンテーションマスク生成
            - **選択モード**: ランダム / 面積大 / 面積小 / 信頼度順

            ### Tab 2: LaMa Inpainting
            - **SimpleLama**: フーリエ畳み込みベースの高品質画像修復
            - **アルファブレンディング**: 段階的な髭薄め効果

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
    """メインエントリーポイント"""
    app = create_app()
    app.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7863
    )


if __name__ == "__main__":
    main()
