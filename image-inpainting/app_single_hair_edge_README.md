# SAM による髭検出アプリケーション 黒髭・白髭対応

SAM (Segment Anything Model) を使用した髭検出アプリケーション

## 概要

SAMのAutomatic Mask Generation (AMG) を活用して、画像内の髭を高精度に検出するGradioベースのWebアプリケーションです。

**v2の新機能:**
- **クラス1 (黒髭)**: 肌より暗い髭を検出
- **クラス2 (白髭)**: 肌より明るい髭を検出
- 各クラスで独立したフィルターパラメータを設定可能

## 必要環境

### Python パッケージ

```bash
pip install gradio numpy opencv-python pillow scipy
pip install torch torchvision
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### SAM モデルチェックポイント

SAM ViT-H モデルが必要です。以下のいずれかのパスに配置してください：

- `./checkpoints/sam_vit_h_4b8939.pth`
- `image-inpainting/checkpoints/sam_vit_h_4b8939.pth`

ダウンロード: https://github.com/facebookresearch/segment-anything#model-checkpoints

## 対応GPU

| デバイス | 説明 |
|---------|------|
| CUDA | NVIDIA GPU（推奨、最速） |
| MPS | Apple Silicon (M1/M2/M3/M4) |
| CPU | フォールバック（低速） |

## 起動方法

```bash
cd image-inpainting
python app_sam_edge-detection.py
```

ブラウザで `http://127.0.0.1:7864` にアクセス

## 使い方

### 基本的な流れ

1. **画像をアップロード** - Input Image エリアに画像をドロップ
2. **検出領域を指定** - 矩形を描画するか、座標を入力
3. **Hair Color Class を選択** - `black`（黒髭）または `white`（白髭）
4. **Detect Hairs をクリック** - 検出実行

### 領域指定方法

#### 方法1: 矩形描画（デフォルト）
画像上で矩形をドラッグして描画

#### 方法2: 座標入力
1. "Region Selection (座標指定)" を開く
2. "Use Coordinate Input" にチェック
3. X1, Y1（左上）と X2, Y2（右下）を入力
4. "Preview Region" で確認

## 髭クラス選択

| クラス | 説明 | 用途 |
|--------|------|------|
| **black** | 黒髭（肌より暗い） | 濃い髭、黒い髭の検出 |
| **white** | 白髭（肌より明るい） | 白髪の髭、グレーの髭の検出 |

## パラメータ説明

### SAM Settings（共通）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| SAM Points Per Side | 64 | サンプリング密度。高いほど多くのマスク生成（64-96推奨） |

### Black Hair Parameters（黒髭用）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Min Area | 5 | 最小マスク面積 |
| Max Area | 2000 | 最大マスク面積 |
| Min Aspect Ratio | 1.2 | アスペクト比フィルタ |
| Brightness Threshold | 1.14 | マスク明るさ < 平均×閾値 で検出。高い値=より明るい髭も許容 |

### White Hair Parameters（白髭用）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Min Area | 5 | 最小マスク面積 |
| Max Area | 2000 | 最大マスク面積 |
| Min Aspect Ratio | 1.2 | アスペクト比フィルタ |
| Brightness Threshold | 0.95 | マスク明るさ > 平均×閾値 で検出。低い値=より暗い髭も許容 |

### Brightness Threshold の仕組み

| クラス | 閾値 | 判定ロジック |
|--------|-----|--------------|
| 黒髭 | 1.14 (default) | マスク明るさ < 平均×1.14 で検出 |
| 白髭 | 0.95 (default) | マスク明るさ > 平均×0.95 で検出 |

### Tile Processing（大きい画像用）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Enable Tile Processing | OFF | タイル分割処理を有効化 |
| Tile Size | 400 | 各タイルのサイズ（px） |
| Tile Overlap | 50 | タイル間のオーバーラップ（px） |

### Visualization Settings

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| Overlay Alpha | 0.3 | オーバーレイ透明度（0.2-0.4推奨） |
| Show Center Markers | ON | 髭の中心にマーカー表示 |

## 出力

| 出力 | 説明 |
|------|------|
| Detection Result | 検出結果のオーバーレイ画像 |
| Filtered Mask | フィルタ後のマスク（カラー表示） |
| ALL Mask (Unfiltered) | フィルタ前の全マスク |
| Hair Count | 検出された髭の数 |
| Status | 処理ステータスと統計 |

## チューニングのヒント

### 黒髭が検出されない場合
- `Brightness Threshold` を上げる (1.2-1.3)
- `Min Area` を下げる (3-5)

### 白髭が検出されない場合
- `Brightness Threshold` を下げる (0.85-0.90)
- `Min Area` を下げる (3-5)

### 誤検出が多い場合
- `Min Aspect Ratio` を上げる (1.5-2.0)
- 黒髭: `Brightness Threshold` を下げる
- 白髭: `Brightness Threshold` を上げる

### 細い髭が検出されない場合
- `SAM Points Per Side` を 80-96 に上げる
- `Min Aspect Ratio` を 1.1-1.2 に下げる

### 処理が遅い場合
- `SAM Points Per Side` を 48-56 に下げる
- GPU（CUDA/MPS）を使用する
- 検出領域を小さくする

## 関連ファイル

- `beard_inpainting_modules/` - 共通モジュール
  - `RegionSelector` - 領域選択
  - `DetectedRegion` - 検出結果データクラス
  - `visualize_single_hairs` - 可視化関数

## 参考

- [Segment Anything](https://github.com/facebookresearch/segment-anything)

## ポート

デフォルト: `7864`（v1の7863とは別ポート）
