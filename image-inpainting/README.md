# 髭検出・修復アプリケーション v3/v4

髭を検出し、LaMa Inpainting で自然に除去・薄めするGradioアプリケーション。

## 機能

- **ルールベース検出**: 髭を1本ずつ高精度に検出（推奨）
- **Grounded SAM**: テキストプロンプトによる自動髭検出（オプション）
- **矩形選択**: 白ブラシで検出領域を指定
- **カラーハイライト**: 検出した各髭を異なる色でハイライト表示
- **削除対象選択**: ランダム/面積大/面積小/信頼度順で削除対象を選択（赤色で強調）
- **LaMa Inpainting**: 高品質な髭除去・段階的な薄め（30%, 50%, 70%, 100%）
- **色調補正（v4）**: LAB色空間ベースの青髭補正機能

## 必要な環境

- Python 3.10 以上（3.11 または 3.12 推奨）
- CUDA 対応 GPU（推奨、CPUでも動作可能）

## インストール

### 1. 必須パッケージのインストール

```bash
pip install gradio numpy opencv-python pillow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install simple-lama-inpainting
```

### 2. オプション: Grounded SAM（高度な検出機能）

Grounded SAM を使用する場合のみ必要です。ルールベース検出のみ使用する場合は不要です。

```bash
pip install segment-anything groundingdino-py
```

**チェックポイントのダウンロード:**

| ファイル | ダウンロードURL | 配置場所 |
|---------|----------------|---------|
| `sam_vit_h_4b8939.pth` | [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | `image-inpainting/` |
| `groundingdino_swint_ogc.pth` | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO/releases) | `image-inpainting/` または親フォルダ |

## ファイル構成

### 必須ファイル

```
image-inpainting/
├── app_gradio_v3.py          # v3 アプリケーション
├── app_gradio_v4.py          # v4 アプリケーション（モジュラー版）
├── config.py                 # 設定ファイル（MAX_IMAGE_SIZE, FEATHER_RADIUS等）
├── core/                     # コアモジュール
│   ├── __init__.py
│   ├── inpainting.py         # LaMa Inpainting ラッパー（SimpleLama使用）
│   └── image_utils.py        # 画像処理ユーティリティ
├── beard_inpainting_modules/ # v4 モジュール（色調補正機能含む）
│   ├── __init__.py
│   ├── image_handler.py      # 画像I/O
│   ├── region_selector.py    # 矩形選択
│   ├── beard_detector.py     # 検出ロジック
│   ├── highlighter.py        # 選択・表示
│   ├── inpainter.py          # LaMa wrapper
│   ├── color_corrector.py    # 色調補正（青髭補正）
│   └── pipeline.py           # オーケストレーター
├── requirements.txt          # 依存パッケージ一覧
└── README.md                 # このファイル
```

### オプション（Grounded SAM 使用時のみ）

```
image-inpainting/
├── sam_vit_h_4b8939.pth      # SAM チェックポイント（約2.4GB）
└── groundingdino_swint_ogc.pth  # Grounding DINO チェックポイント（約694MB）
    ※ 親フォルダに配置しても自動検出されます
```

### 不要なファイル（削除可能）

以下は旧バージョンまたは開発用ファイルです：

```
image-inpainting/
├── app.py                    # 旧バージョン
├── app_gradio.py             # v1
├── app_gradio_v2.py          # v2
├── beard_grounded_sam.py     # OpenCV版（参考実装）
├── ui/                       # 未使用UIコンポーネント
└── *.png                     # テスト出力画像
```

### 参考: 親フォルダの構成

```
image_inpaintin+SAM_移管用/
├── image-inpainting/         # ← このアプリ
├── GroundingDINO/            # ※ pip install groundingdino-py で不要に
├── groundingdino_swint_ogc.pth  # チェックポイント（ここでも検出可能）
├── sam_vit_b_01ec64.pth      # SAM ViT-B（軽量版、オプション）
└── experimental/             # 実験用コード
```

## 使い方

### 起動

```bash
cd image-inpainting

# v3（従来版）
python app_gradio_v3.py  # Port 7863

# v4（モジュラー版 + 色調補正機能）
python app_gradio_v4.py  # Port 7864
```

- v3: `http://127.0.0.1:7863`
- v4: `http://127.0.0.1:7864`

### 起動時のログメッセージ

起動時に以下のようなログが表示されます：

```
PyTorch: 利用可能 (CUDA: True/False)
SAM: 利用可能
Grounding DINO: 利用可能
LaMa Inpainting: 利用可能
```

**各項目の意味:**

| メッセージ | 説明 |
|-----------|------|
| `PyTorch: 利用可能 (CUDA: True)` | GPU で高速処理が可能 |
| `PyTorch: 利用可能 (CUDA: False)` | CPU モードで動作（やや遅いが問題なし） |
| `SAM: 利用可能` | Grounded SAM 検出モードが使用可能 |
| `Grounding DINO: 利用可能` | Grounded SAM 検出モードが使用可能 |
| `LaMa Inpainting: 利用可能` | 髭除去機能が使用可能（必須） |

### 無視してよい警告メッセージ

以下の警告は動作に影響しないため、無視して構いません：

```
FutureWarning: Importing from timm.models.layers is deprecated...
```
→ `timm` パッケージの内部警告。将来のバージョンで解消予定。

```
UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
```
→ Grounding DINO のカスタム CUDA 演算子が未コンパイル。CPU で代替処理されるため問題なし。

```
DeprecationWarning: The 'theme' parameter...
```
→ Gradio の API 変更に関する警告。動作に影響なし。

### ワークフロー

#### Tab 1: 髭検出

1. **画像をアップロード**
2. **髭の範囲を矩形で囲む**（白色ブラシで塗りつぶし）
3. **検出モードを選択**
   - ルールベース（1本ずつ検出）: 推奨、高精度
   - Grounded SAM: テキストプロンプトベース
4. **「髭を検出」をクリック**
   - 各髭が異なる色でハイライト表示される
5. **Remove % で削除対象を選択**
   - スライダーで削除割合を指定
   - 選択モード: ランダム / 面積大 / 面積小 / 信頼度順
   - 削除対象は赤色で強調表示
6. **「マスクを Tab 2 に転送」をクリック**

#### Tab 2: LaMa 髭除去

1. **薄め具合を選択**（30%, 50%, 70%, 100%）
2. **「髭薄めを実行」をクリック**
3. **結果をギャラリーで確認・ダウンロード**

#### Tab 3: 色調補正（v4のみ）

1. **Tab 2の結果を取得**（自動転送 or 「← Tab 2の結果を取得」ボタン）
2. **補正モードを選択**（青み除去がおすすめ）
3. **対象領域を指定**
   - 「手動で塗る」: エディタで青髭部分を赤ブラシで塗る
   - 「Tab 1の選択マスクを使用」: 検出済みのマスクを再利用
4. **LABパラメータを調整**（必要に応じて）
5. **「色調補正を適用」をクリック**

## パラメータ説明

### ルールベース検出

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| 二値化閾値 | 80 | 小さいほど薄い髭も検出（暗い=髭） |
| 最小面積 | 10 | 検出領域の最小ピクセル数 |
| 最大面積 | 5000 | 検出領域の最大ピクセル数 |

### Grounded SAM（オプション）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| 検出プロンプト | `beard. facial hair. stubble.` | 髭を表すテキスト |
| Box Threshold | 0.25 | ボックス検出の信頼度閾値 |
| Text Threshold | 0.20 | テキストマッチングの閾値 |

### 色調補正（v4 Tab 3）- 最適な肌色補正

LAB色空間を使用して青髭を自然な肌色に補正します。

#### パラメータ調整の経緯

色調補正のLABパラメータは、実際のテスト画像を使用して最適化されました。

**変更前（初期値）:**
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| a* 調整係数 | 100% (1.0) | 赤-緑軸の調整 |
| b* 調整係数 | 100% (1.0) | 青-黄軸の調整 |
| L 調整係数 | 70% (0.7) | 明度の調整 |

初期値では色調整が強すぎて、補正後の肌がオレンジ/暖色に偏りすぎる問題がありました。

**変更後（最適な肌色補正）:**
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| a* 調整係数 | **30% (0.3)** | 赤みの追加を最小限に抑制（オレンジ化防止） |
| b* 調整係数 | **60% (0.6)** | 青み除去を控えめに（自然な色味を維持） |
| L 調整係数 | **50% (0.5)** | 明度の上げすぎを抑制 |

この設定により、元の肌色・テクスチャを保持しながら青髭を自然に補正できます。

#### 補正モード

| モード | 用途 | スポイト領域 |
|--------|------|-------------|
| **青み除去（推奨）** | 青髭補正 | 不要 |
| 色味転送 | 汎用色補正 | 必要 |
| 自動補正 | 手軽な補正 | 不要 |

## トラブルシューティング

### LaMa が動作しない

```bash
pip install simple-lama-inpainting --no-deps
pip install numpy pillow opencv-python torch torchvision
```

### Grounded SAM の config が見つからない

`groundingdino-py` を pip でインストールすると、config ファイルは自動的にパッケージ内から取得されます。

```bash
pip install groundingdino-py
```

### CUDA メモリ不足

- 画像サイズを小さくする
- `config.py` の `MAX_IMAGE_SIZE` を調整

## 技術スタック

- **Gradio**: Web UI フレームワーク
- **OpenCV**: 画像処理（ルールベース検出）
- **SimpleLama**: LaMa ベースの Inpainting
- **Segment Anything (SAM)**: セグメンテーション（オプション）
- **Grounding DINO**: テキストベース物体検出（オプション）

## ライセンス

MIT License
