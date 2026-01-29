"""肌色調整（色調補正）モジュール - 青髭補正特化

髭を剃った後の青髭を素肌に近づけるための色補正ツール。
LAB色空間を使用して、明度を保持しながら色味のみを調整。

元ファイル: experimental/color_fill_tool/color_fill_tool.py
"""

import numpy as np
import cv2
from enum import Enum
from typing import Tuple, Optional


class CorrectionMode(Enum):
    """色補正モード"""
    BLUE_REMOVAL = "blue_removal"       # 青み除去に特化（推奨）
    COLOR_TRANSFER = "color_transfer"   # スポイト領域の色味を転送
    AUTO_DETECT = "auto_detect"         # 自動検出+補正


class MaskType(Enum):
    """マスクの種類（補正強度の調整に使用）"""
    MANUAL = "manual"           # 手動塗り（連続した領域）
    SCATTERED = "scattered"     # Tab 1からの散らばった髭領域


class SkinColorCorrector:
    """LAB色空間を使った肌色補正（青髭補正特化）"""

    def __init__(self):
        self._lut_smoothing_kernel = 21

    def correct_color(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        source_mask: Optional[np.ndarray] = None,
        strength: float = 1.0,
        edge_blur: int = 15,
        mode: CorrectionMode = CorrectionMode.BLUE_REMOVAL,
        a_adjustment_factor: float = 0.3,
        b_adjustment_factor: float = 0.6,
        l_adjustment_factor: float = 0.5,
        mask_type: MaskType = MaskType.MANUAL,
        use_direct_fill: bool = False
    ) -> np.ndarray:
        """
        対象領域の色を補正する（統合インターフェース）

        Args:
            image: 入力画像 (BGR or RGB)
            target_mask: 色調整する領域のマスク（青髭領域）
            source_mask: 色を取得する領域のマスク（頬など）※COLOR_TRANSFERで必須
            strength: 補正強度 (0.0-1.0)
            edge_blur: エッジぼかしサイズ
            mode: 補正モード
            a_adjustment_factor: a*（赤-緑軸）の調整係数 (0.0-1.0)
            b_adjustment_factor: b*（青-黄軸）の調整係数 (0.0-1.0)
            l_adjustment_factor: L（明度）の調整係数 (0.0-1.0)
            mask_type: マスクの種類（MANUAL=手動塗り, SCATTERED=Tab1の散らばった領域）
            use_direct_fill: 美肌塗りつぶしモード（色味転送時）

        Returns:
            色調補正された画像
        """
        if mode == CorrectionMode.BLUE_REMOVAL:
            return self.remove_blue_tint(
                image, target_mask, strength, edge_blur,
                a_adjustment_factor, b_adjustment_factor, l_adjustment_factor,
                mask_type
            )
        elif mode == CorrectionMode.COLOR_TRANSFER:
            if source_mask is None:
                raise ValueError("COLOR_TRANSFER モードでは source_mask が必要です")
            return self.transfer_color_from_source(
                image, target_mask, source_mask, strength, edge_blur,
                use_direct_fill=use_direct_fill
            )
        elif mode == CorrectionMode.AUTO_DETECT:
            return self.auto_correct(
                image, target_mask, strength, edge_blur,
                a_adjustment_factor, b_adjustment_factor, l_adjustment_factor,
                mask_type
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def remove_blue_tint(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        strength: float = 1.0,
        edge_blur: int = 15,
        a_adjustment_factor: float = 0.3,
        b_adjustment_factor: float = 0.6,
        l_adjustment_factor: float = 0.5,
        mask_type: MaskType = MaskType.MANUAL
    ) -> np.ndarray:
        """
        青髭を素肌に近づける（美肌補正版）

        周辺の肌色を自動サンプリングし、青髭領域を自然な美肌に補正する。
        LAB色空間で色味を肌色に近づけ、テクスチャスムージングで滑らかに。

        Args:
            image: 入力画像 (BGR)
            target_mask: 青髭領域のマスク
            strength: 補正強度 (0.0-1.0)
            edge_blur: エッジぼかしサイズ
            a_adjustment_factor: a*（赤-緑軸）の調整係数 (0.0-1.0) デフォルト0.3
            b_adjustment_factor: b*（青-黄軸）の調整係数 (0.0-1.0) デフォルト0.6
            l_adjustment_factor: L（明度）の調整係数 (0.0-1.0) デフォルト0.5
            mask_type: マスクの種類（散らばった領域の場合は補正を強化）

        Returns:
            美肌補正された画像 (BGR)
        """
        if image is None or target_mask is None:
            return image

        if not np.any(target_mask > 0):
            return image

        # ========== 散らばった領域向けの最適化 ==========
        # Tab 1のマスク（散らばった小さな髭領域）の場合、
        # エッジぼかしとスムージングを減らして補正効果を維持
        use_texture_preservation = False
        if mask_type == MaskType.SCATTERED:
            use_texture_preservation = True

            # 散らばった領域用のパラメータ調整
            # 1. マスクを少し膨張させて領域を繋げる
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            target_mask = cv2.dilate(target_mask, kernel, iterations=1)

            # 2. エッジぼかしを減らす（小さい領域では効果が打ち消されるため）
            edge_blur = max(3, edge_blur // 3)

            # 3. LAB調整係数を強化（散らばった領域では効果が見えにくいため）
            a_adjustment_factor = min(1.0, a_adjustment_factor * 1.5)
            b_adjustment_factor = min(1.0, b_adjustment_factor * 1.5)
            l_adjustment_factor = min(1.0, l_adjustment_factor * 1.3)

            print(f"[散らばった領域モード] edge_blur={edge_blur}, LAB強化係数適用, テクスチャ保持=ON")

        result = image.copy()

        # LAB色空間に変換
        lab_image = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 周辺の肌色を自動サンプリング（広めの範囲から取得）
        skin_mask = self._detect_surrounding_skin(image, target_mask, dilation_size=50, erosion_size=15)

        # 肌色サンプルが取得できた場合
        if np.any(skin_mask > 0):
            skin_pixels_lab = lab_image[skin_mask > 0]

            # 美肌補正: 明るめの肌色を目標にする（75パーセンタイル使用）
            # 平均値だと暗い部分も含まれるため、明るめを採用
            target_l = np.percentile(skin_pixels_lab[:, 0], 75)
            target_a = np.median(skin_pixels_lab[:, 1])  # 中央値で安定
            target_b = np.median(skin_pixels_lab[:, 2])  # 中央値で安定

            print(f"肌色サンプル（美肌）: L={target_l:.1f}, a={target_a:.1f}, b={target_b:.1f}")
        else:
            # サンプルが取れない場合は日本人の平均的な明るい肌色
            target_l = 185.0
            target_a = 140.0
            target_b = 145.0
            print("肌色サンプル取得できず、デフォルト値を使用")

        # 対象領域の座標を取得（ベクトル化処理）
        target_y, target_x = np.where(target_mask > 0)

        if len(target_y) == 0:
            return image

        # 対象領域のピクセル値を取得
        original_l = lab_image[target_y, target_x, 0]
        original_a = lab_image[target_y, target_x, 1]
        original_b = lab_image[target_y, target_x, 2]

        # ========== 色調整（控えめに設定） ==========
        # a*: 赤-緑軸の調整（デフォルト30%、赤みの追加を最小限に）
        # b*: 青-黄軸の調整（デフォルト60%、青み除去がメインだが控えめに）
        # L: 明度の調整（デフォルト50%、上げすぎると不自然）
        #
        # ※パラメータはUI（Tab 3）から調整可能
        print(f"LAB調整係数: a*={a_adjustment_factor:.0%}, b*={b_adjustment_factor:.0%}, L={l_adjustment_factor:.0%}")

        new_a = original_a + (target_a - original_a) * strength * a_adjustment_factor
        new_b = original_b + (target_b - original_b) * strength * b_adjustment_factor

        # 明度調整
        l_adjustment = (target_l - original_l) * strength * l_adjustment_factor
        new_l = original_l + l_adjustment

        # 値を更新
        lab_image[target_y, target_x, 0] = new_l
        lab_image[target_y, target_x, 1] = new_a
        lab_image[target_y, target_x, 2] = new_b

        # 値をクリップ
        lab_image = np.clip(lab_image, 0, 255).astype(np.uint8)

        # BGRに戻す
        result_bgr = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

        # ========== テクスチャ保持モード ==========
        if use_texture_preservation:
            # 元画像から高周波成分（テクスチャ）を抽出
            # 周波数分離: 元画像 = 低周波（色）+ 高周波（テクスチャ）
            original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            result_gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # 低周波成分を取得（ぼかし）
            blur_size = 15
            original_lowfreq = cv2.GaussianBlur(original_gray, (blur_size, blur_size), 0)
            result_lowfreq = cv2.GaussianBlur(result_gray, (blur_size, blur_size), 0)

            # 高周波成分（テクスチャ）= 元画像 - 低周波
            texture = original_gray - original_lowfreq

            # 補正結果の低周波 + 元のテクスチャ
            # これにより色は補正されるがテクスチャは保持される
            new_gray = result_lowfreq + texture

            # グレースケールの変化量を計算
            gray_diff = new_gray - result_gray

            # 各チャンネルにテクスチャの変化を適用
            result_bgr = result_bgr.astype(np.float32)
            for c in range(3):
                result_bgr[:, :, c] += gray_diff * 0.7  # 70%のテクスチャを復元
            result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)

            # 微細なノイズを追加して自然さを演出
            noise = np.random.normal(0, 2, result_bgr.shape).astype(np.float32)
            result_bgr = np.clip(result_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            print("[テクスチャ保持] 高周波成分を復元、微細ノイズ追加")

            # スムージングは行わない（テクスチャを維持するため）
        else:
            # ========== 従来の美肌スムージング ==========
            smooth_strength = min(strength * 0.8, 0.9)

            # バイラテラルフィルタでエッジを保持しながらテクスチャを滑らかに
            smoothed = cv2.bilateralFilter(result_bgr, d=9, sigmaColor=75, sigmaSpace=75)

            # 元画像とスムージング画像をstrengthでブレンド
            result_bgr = cv2.addWeighted(
                result_bgr, 1.0 - smooth_strength,
                smoothed, smooth_strength,
                0
            )

        # エッジをぼかしてブレンド（境界を自然に）
        result = self._blend_with_edge_blur(image, result_bgr, target_mask, edge_blur)

        return result

    def transfer_color_from_source(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        source_mask: np.ndarray,
        strength: float = 1.0,
        edge_blur: int = 15,
        use_direct_fill: bool = False
    ) -> np.ndarray:
        """
        スポイト領域の色味を転送

        Args:
            image: 入力画像 (BGR)
            target_mask: 色調整する領域のマスク
            source_mask: 色を取得する領域のマスク（頬など）
            strength: 補正強度 (0.0-1.0)
            edge_blur: エッジぼかしサイズ
            use_direct_fill: True=美肌塗りつぶしモード, False=LUTベース（従来）

        Returns:
            色調補正された画像 (BGR)
        """
        if image is None:
            return image

        if not np.any(target_mask > 0):
            return image

        if not np.any(source_mask > 0):
            print("警告: スポイト領域が選択されていません")
            return image

        # モード分岐
        if use_direct_fill:
            return self._transfer_color_direct_fill(
                image, target_mask, source_mask, strength, edge_blur
            )
        else:
            return self._transfer_color_lut_based(
                image, target_mask, source_mask, strength, edge_blur
            )

    def _transfer_color_lut_based(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        source_mask: np.ndarray,
        strength: float,
        edge_blur: int
    ) -> np.ndarray:
        """LUTベースの色味転送（従来方式）"""
        result = image.copy()

        # LAB色空間に変換
        lab_image = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)

        # LUTを構築
        l_to_a, l_to_b = self._build_luminance_to_ab_lut(lab_image, source_mask)

        # 対象領域の各ピクセルを処理
        target_coords = np.where(target_mask > 0)

        for y, x in zip(target_coords[0], target_coords[1]):
            l_val = int(lab_image[y, x, 0])
            original_a = lab_image[y, x, 1]
            original_b = lab_image[y, x, 2]

            # 目標の色味を取得
            target_a = l_to_a[l_val]
            target_b = l_to_b[l_val]

            # 強度に応じてブレンド
            lab_image[y, x, 1] = original_a + (target_a - original_a) * strength
            lab_image[y, x, 2] = original_b + (target_b - original_b) * strength

        # 値をクリップ
        lab_image = np.clip(lab_image, 0, 255).astype(np.uint8)

        # BGRに戻す
        result_bgr = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

        # エッジをぼかしてブレンド
        result = self._blend_with_edge_blur(image, result_bgr, target_mask, edge_blur)

        return result

    def _transfer_color_direct_fill(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        source_mask: np.ndarray,
        strength: float,
        edge_blur: int
    ) -> np.ndarray:
        """美肌塗りつぶしモード：参照色で直接塗りつぶし＋自然なブレンド"""
        result = image.copy()

        # LAB色空間に変換
        lab_image = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)

        # ========== 参照領域の美肌色を取得 ==========
        source_pixels_lab = lab_image[source_mask > 0]

        # 明るめの肌色を目標にする（60パーセンタイル - 少し控えめに）
        target_l = np.percentile(source_pixels_lab[:, 0], 60)
        target_a = np.median(source_pixels_lab[:, 1])
        target_b = np.median(source_pixels_lab[:, 2])

        print(f"[美肌塗りつぶし] 参照色: L={target_l:.1f}, a={target_a:.1f}, b={target_b:.1f}, 強度={strength:.0%}")

        # ========== 対象領域を美肌色で塗りつぶし ==========
        target_y, target_x = np.where(target_mask > 0)

        if len(target_y) == 0:
            return image

        # 元の値を保存
        original_l = lab_image[target_y, target_x, 0]
        original_a = lab_image[target_y, target_x, 1]
        original_b = lab_image[target_y, target_x, 2]

        # 元の色のバリエーション（標準偏差）を計算
        original_l_std = np.std(original_l)
        original_a_std = np.std(original_a)
        original_b_std = np.std(original_b)

        # 美肌色に近づける（強度に応じて）
        new_l = original_l + (target_l - original_l) * strength
        new_a = original_a + (target_a - original_a) * strength
        new_b = original_b + (target_b - original_b) * strength

        # ========== 色のバリエーションを復元（ベタ塗り防止） ==========
        # 元の色の偏差を一部残す
        variation_keep = 0.4  # 元のバリエーションを40%残す
        l_deviation = original_l - np.mean(original_l)
        a_deviation = original_a - np.mean(original_a)
        b_deviation = original_b - np.mean(original_b)

        new_l = new_l + l_deviation * variation_keep
        new_a = new_a + a_deviation * variation_keep
        new_b = new_b + b_deviation * variation_keep

        lab_image[target_y, target_x, 0] = new_l
        lab_image[target_y, target_x, 1] = new_a
        lab_image[target_y, target_x, 2] = new_b

        # 値をクリップ
        lab_image = np.clip(lab_image, 0, 255).astype(np.uint8)

        # BGRに戻す
        result_bgr = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

        # ========== テクスチャ保持（肌のきめを強く復元） ==========
        original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        result_gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 高周波成分（テクスチャ）を抽出 - より細かいテクスチャを保持
        blur_size = 9  # 小さくしてより細かいテクスチャを保持
        original_lowfreq = cv2.GaussianBlur(original_gray, (blur_size, blur_size), 0)
        result_lowfreq = cv2.GaussianBlur(result_gray, (blur_size, blur_size), 0)

        texture = original_gray - original_lowfreq

        # テクスチャを強く復元（0.7→1.0）
        texture_strength = 1.0
        new_gray = result_lowfreq + texture * texture_strength
        gray_diff = new_gray - result_gray

        result_bgr = result_bgr.astype(np.float32)
        texture_mask_float = (target_mask > 0).astype(np.float32)

        for c in range(3):
            result_bgr[:, :, c] += gray_diff * texture_strength * texture_mask_float

        result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)

        # ========== スムージングを大幅に軽減 ==========
        # スムージングはほぼ無しに（テクスチャを壊さない）
        smooth_strength = min(strength * 0.2, 0.3)  # 大幅に軽減
        if smooth_strength > 0.1:
            smoothed = cv2.bilateralFilter(result_bgr, d=5, sigmaColor=30, sigmaSpace=30)
            mask_float = target_mask.astype(np.float32) / 255.0
            mask_3ch = np.stack([mask_float] * 3, axis=-1)
            result_bgr = (
                result_bgr.astype(np.float32) * (1.0 - smooth_strength * mask_3ch) +
                smoothed.astype(np.float32) * (smooth_strength * mask_3ch)
            ).astype(np.uint8)

        # ========== 周囲との自然なブレンド（多段階） ==========
        # 1. まずマスク境界を大きくぼかしてグラデーション作成
        feather_size = 31  # フェザリングサイズ
        feathered_mask = cv2.GaussianBlur(
            target_mask.astype(np.float32),
            (feather_size, feather_size), 0
        ) / 255.0

        # 2. フェザリングマスクでブレンド（グラデーション効果）
        feathered_3ch = np.stack([feathered_mask] * 3, axis=-1)
        result_bgr = (
            image.astype(np.float32) * (1.0 - feathered_3ch) +
            result_bgr.astype(np.float32) * feathered_3ch
        ).astype(np.uint8)

        # 3. さらにエッジブラーで仕上げ
        effective_edge_blur = max(edge_blur, 15)
        result = self._blend_with_edge_blur(image, result_bgr, target_mask, effective_edge_blur)

        return result

    def auto_correct(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        strength: float = 1.0,
        edge_blur: int = 15,
        a_adjustment_factor: float = 0.3,
        b_adjustment_factor: float = 0.6,
        l_adjustment_factor: float = 0.5,
        mask_type: MaskType = MaskType.MANUAL
    ) -> np.ndarray:
        """
        自動検出+補正

        対象領域周辺の肌色を自動サンプリングし、
        青髭領域を自然な肌色に補正する。

        Args:
            image: 入力画像 (BGR)
            target_mask: 青髭領域のマスク
            strength: 補正強度 (0.0-1.0)
            edge_blur: エッジぼかしサイズ
            a_adjustment_factor: a*（赤-緑軸）の調整係数 (0.0-1.0)
            b_adjustment_factor: b*（青-黄軸）の調整係数 (0.0-1.0)
            l_adjustment_factor: L（明度）の調整係数 (0.0-1.0)
            mask_type: マスクの種類

        Returns:
            自動補正された画像 (BGR)
        """
        if image is None or target_mask is None:
            return image

        if not np.any(target_mask > 0):
            return image

        # 周辺の肌色を自動サンプリング
        source_mask = self._detect_surrounding_skin(image, target_mask)

        if not np.any(source_mask > 0):
            # 周辺肌色が検出できない場合は青み除去にフォールバック
            print("周辺肌色を検出できませんでした。青み除去モードで処理します。")
            return self.remove_blue_tint(
                image, target_mask, strength, edge_blur,
                a_adjustment_factor, b_adjustment_factor, l_adjustment_factor,
                mask_type
            )

        # 色味転送を実行
        return self.transfer_color_from_source(image, target_mask, source_mask, strength, edge_blur)

    def _build_luminance_to_ab_lut(
        self,
        lab_image: np.ndarray,
        source_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """L値から(a,b)値へのLUTを構築"""
        # スポイト領域のピクセルを取得
        color_pixels_lab = lab_image[source_mask > 0]

        # L値は0-255の範囲
        l_to_a = np.zeros(256, dtype=np.float32)
        l_to_b = np.zeros(256, dtype=np.float32)
        l_counts = np.zeros(256, dtype=np.float32)

        # スポイト領域のピクセルからL→(a,b)のマッピングを作成
        for pixel in color_pixels_lab:
            l_idx = int(pixel[0])
            l_to_a[l_idx] += pixel[1]
            l_to_b[l_idx] += pixel[2]
            l_counts[l_idx] += 1

        # 平均を計算（データがある部分のみ）
        valid = l_counts > 0
        l_to_a[valid] /= l_counts[valid]
        l_to_b[valid] /= l_counts[valid]

        # データがない明度レベルを補間で埋める
        l_to_a = self._interpolate_lut(l_to_a, l_counts)
        l_to_b = self._interpolate_lut(l_to_b, l_counts)

        # LUTをスムージング（ガウシアンフィルタ）
        kernel_size = self._lut_smoothing_kernel
        l_to_a = cv2.GaussianBlur(l_to_a.reshape(1, -1), (kernel_size, 1), 0).flatten()
        l_to_b = cv2.GaussianBlur(l_to_b.reshape(1, -1), (kernel_size, 1), 0).flatten()

        return l_to_a, l_to_b

    def _interpolate_lut(self, lut: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """データがない明度レベルを線形補間で埋める"""
        result = lut.copy()
        valid_indices = np.where(counts > 0)[0]

        if len(valid_indices) == 0:
            return result

        # 最初と最後の有効値で端を埋める
        if valid_indices[0] > 0:
            result[:valid_indices[0]] = result[valid_indices[0]]
        if valid_indices[-1] < 255:
            result[valid_indices[-1]+1:] = result[valid_indices[-1]]

        # 中間の欠損値を線形補間
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            if end_idx - start_idx > 1:
                start_val = result[start_idx]
                end_val = result[end_idx]
                for j in range(start_idx + 1, end_idx):
                    t = (j - start_idx) / (end_idx - start_idx)
                    result[j] = start_val + (end_val - start_val) * t

        return result

    def _detect_surrounding_skin(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        dilation_size: int = 30,
        erosion_size: int = 5
    ) -> np.ndarray:
        """
        対象領域周辺の肌色を自動検出

        Args:
            image: 入力画像 (BGR)
            target_mask: 対象領域のマスク
            dilation_size: 検出領域の拡張サイズ
            erosion_size: 対象領域からの距離

        Returns:
            周辺肌色領域のマスク
        """
        h, w = image.shape[:2]

        # 対象領域を膨張させて周辺領域を作成
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_size * 2 + 1, dilation_size * 2 + 1)
        )
        dilated = cv2.dilate(target_mask, kernel_dilate, iterations=1)

        # 対象領域を少し膨張させて除外領域を作成
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erosion_size * 2 + 1, erosion_size * 2 + 1)
        )
        excluded = cv2.dilate(target_mask, kernel_erode, iterations=1)

        # 周辺領域 = 膨張 - 除外
        surrounding = cv2.subtract(dilated, excluded)

        # HSV色空間で肌色をフィルタリング
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 肌色の範囲（調整可能）
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 200, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 周辺領域と肌色マスクのAND
        result_mask = cv2.bitwise_and(surrounding, skin_mask)

        return result_mask

    def _blend_with_edge_blur(
        self,
        original: np.ndarray,
        corrected: np.ndarray,
        mask: np.ndarray,
        edge_blur: int = 15
    ) -> np.ndarray:
        """エッジをぼかしてブレンド"""
        if edge_blur <= 0:
            # ぼかしなしの場合、マスク領域のみ置換
            result = original.copy()
            result[mask > 0] = corrected[mask > 0]
            return result

        # マスクの境界をぼかす
        blur_mask = cv2.GaussianBlur(
            mask.astype(np.float32),
            (edge_blur * 2 + 1, edge_blur * 2 + 1),
            0
        )
        blur_mask = blur_mask / 255.0

        # 3チャンネルに拡張
        blur_mask_3ch = np.stack([blur_mask] * 3, axis=-1)

        # ブレンド
        result = (
            original.astype(np.float32) * (1.0 - blur_mask_3ch) +
            corrected.astype(np.float32) * blur_mask_3ch
        ).astype(np.uint8)

        return result

    def get_mask_from_editor(self, editor_data: dict, color: str = "red") -> Optional[np.ndarray]:
        """
        Gradio ImageEditorからマスクを抽出

        Args:
            editor_data: ImageEditorの出力
            color: 抽出する色 ("red" or "green")

        Returns:
            バイナリマスク
        """
        if editor_data is None:
            return None

        try:
            if isinstance(editor_data, dict):
                if 'layers' in editor_data and len(editor_data['layers']) > 0:
                    layer = editor_data['layers'][0]
                    if isinstance(layer, np.ndarray) and len(layer.shape) == 3:
                        # 指定色のチャンネルを抽出
                        if color == "red":
                            # 赤チャンネルが高い領域
                            mask = (layer[:, :, 0] > 128).astype(np.uint8) * 255
                        elif color == "green":
                            # 緑チャンネルが高い領域
                            mask = (layer[:, :, 1] > 128).astype(np.uint8) * 255
                        else:
                            # グレースケール
                            gray = cv2.cvtColor(layer[:, :, :3], cv2.COLOR_RGB2GRAY)
                            mask = (gray > 128).astype(np.uint8) * 255
                        return mask
        except Exception as e:
            print(f"マスク抽出エラー: {e}")

        return None
