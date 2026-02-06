# 髭検出・除去シミュレーション　アプリ v6

髭を検出し、MAT/LaMaによる修復、色補正を行うアプリケーション。

## 機能

### v6 新機能
- **MAT (Mask-Aware Transformer) Inpainting**: CVPR 2022 Best Paper Finalist
- **顔画像に特化**: 512x512で処理する高品質なinpainting
- **2つのプリトレーニングモデル**: FFHQ / CelebA-HQ 対応
- **MAT強化モード**: テクスチャ保持 + 青髭補正の組み合わせ

### 基本機能
- **ルールベース検出**: 髭を1本ずつ高精度に検出（推奨）
- **Grounded SAM**: テキストプロンプトによる自動髭検出（オプション）
- **矩形選択 / 座標入力**: 検出領域を指定
- **カラーハイライト**: 検出した各髭を異なる色でハイライト表示
- **削除対象選択**: ランダム/面積大/面積小/信頼度順で削除対象を選択（赤色で強調）
- **MAT / LaMa / OpenCV Inpainting**: 高品質な髭除去・段階的な薄め（30%, 50%, 70%, 100%）
- **色調補正**: LAB色空間ベースの青髭補正機能
- **髭オーバーレイ**: 残した髭を元画像から重ねて自然な薄め効果を実現

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

### 2. オプション: MAT Inpainting（v6推奨）

MAT (Mask-Aware Transformer) を使用する場合に必要です。

**チェックポイントのダウンロード:**

| ファイル | 配置場所 |
|---------|---------|
| `MAT_FFHQ_512_fp16.safetensors` | `checkpoints/mat/` |
| `MAT_CelebA-HQ_512_fp16.safetensors` | `checkpoints/mat/` |

※ チェックポイントは [Hugging Face](https://huggingface.co/spacepxl/MAT-inpainting-fp16) からダウンロード
※ オリジナルの `.pkl` 形式も使用可能（[MAT公式リポジトリ](https://github.com/fenglinglwb/MAT)）

### 3. オプション: Grounded SAM（高度な検出機能）

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
├── app_gradio_v6.py          # v6 アプリケーション（MAT対応）
├── config.py                 # 設定ファイル（MAX_IMAGE_SIZE, FEATHER_RADIUS等）
├── core/                     # コアモジュール
│   ├── __init__.py
│   ├── inpainting.py         # LaMa Inpainting ラッパー（SimpleLama使用）
│   ├── mat_engine.py         # MAT Inpainting エンジン
│   ├── mat/                  # MAT モデル関連ファイル
│   └── image_utils.py        # 画像処理ユーティリティ
├── beard_inpainting_modules/ # 髭除去モジュール
│   ├── __init__.py
│   ├── pipeline.py           # オーケストレーター
│   ├── inpainter.py          # Inpainting wrapper
│   ├── mat_inpainter.py      # MAT Inpainting wrapper
│   └── region_selector.py    # 領域選択
├── checkpoints/              # チェックポイント
│   └── mat/                  # MAT モデル
│       ├── MAT_FFHQ_512_fp16.safetensors      # FFHQ モデル
│       └── MAT_CelebA-HQ_512_fp16.safetensors # CelebA-HQ モデル
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

## 使い方

### クイックスタート（コマンドライン）

```bash
# 1. プロジェクトディレクトリに移動
cd image-inpainting

# 2. Python 3.12 仮想環境を作成
python3.12 -m venv venv

# 3. 仮想環境を有効化
source venv/bin/activate

# 4. 依存パッケージをインストール
pip install gradio numpy opencv-python pillow torch torchvision
pip install simple-lama-inpainting

# 5. アプリを起動
cd image-inpainting
python app_gradio_v6.py
```

ブラウザで http://127.0.0.1:7867 を開いてアプリを使用できます。

### 2回目以降の起動

```bash
cd image-inpainting
source venv/bin/activate
cd image-inpainting
python app_gradio_v6.py
```

### デバイス（GPU/CPU）

アプリは以下の優先順位でデバイスを自動選択します：
1. **MPS** (Apple Silicon Mac)
2. **CUDA** (NVIDIA GPU)
3. **CPU** (フォールバック)

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
2. **髭の範囲を選択**
   - 矩形で塗りつぶし、または座標入力で指定
3. **検出モードを選択**
   - ルールベース（1本ずつ検出）: 推奨、高精度
   - Grounded SAM: テキストプロンプトベース
4. **「髭を検出」をクリック**
   - 各髭が異なる色でハイライト表示される
5. **Remove % で削除対象を選択**
   - スライダーで削除割合を指定
   - 選択モード: ランダム / 面積大 / 面積小 / 信頼度順
   - 削除対象は赤色で強調表示
   - **残りの髭はTab 3でオーバーレイ可能**
6. **「マスクを Tab 2 に転送」をクリック**

#### Tab 2: 髭除去（MAT / LaMa）

1. **インペインティング手法を選択**
   - **MAT (FFHQ - 顔専用)**: 顔画像に最適化（v6推奨）
   - **MAT (CelebA-HQ - 顔専用)**: 別のプリトレーニングモデル
   - Simple LaMa: 汎用的なInpainting
   - OpenCV Telea / Navier-Stokes: 軽量な従来手法
2. **MAT強化モード設定**（MAT使用時）
   - テクスチャ強度: 元画像の肌の質感をどれだけ復元するか
   - 青髭補正強度: 青みをどれだけ除去するか
3. **薄め具合を選択**（30%, 50%, 70%, 100%）
4. **「髭薄めを実行」をクリック**
5. **結果をギャラリーで確認・ダウンロード**

**手動マスク編集:**
Tab 1のマスクを使わず、直接ブラシで描画して領域指定も可能です。

#### Tab 3: 色調補正 + 髭オーバーレイ

1. **Tab 2の結果を取得**（自動転送 or 「← Tab 2の結果を取得」ボタン）
2. **補正モードを選択**（青み除去がおすすめ）
3. **対象領域を指定**
   - 「手動で塗る」: エディタで青髭部分を赤ブラシで塗る
   - 「Tab 1の選択マスクを使用」: 検出済みのマスクを再利用
4. **LABパラメータを調整**（必要に応じて）
5. **髭オーバーレイを設定**（v5新機能）
   - 「髭オーバーレイを有効化」にチェック
   - オーバーレイ強度を調整（1.0 = 完全に元の髭）
   - エッジぼかしで境界を滑らかに
6. **「色調補正 + オーバーレイを適用」をクリック**

##### 髭オーバーレイ機能

Tab 1で「削除対象」として選ばなかった髭を、色補正後の画像に重ねます。

**仕組み:**
```
残す髭マスク = 全検出マスク - 削除対象マスク
最終出力 = 色補正画像 × (1 - マスク) + 元画像 × マスク
```

**効果:**
- 青みが消えた肌 + 本来の髭 = 自然な髭の薄め効果
- 完全除去ではなく、自然に薄くなった見た目を実現

##### マスク隙間埋め機能

検出マスクが点々に見える場合、隙間を埋めて滑らかに補正できます。

**使い方:**
1. 「マスク隙間埋め設定」アコーディオンを開く
2. 「隙間埋めを有効化」にチェック
3. 隙間埋めサイズとエッジぼかしを調整

##### 除外マスク機能

周辺の青みを含む領域を参照から除外し、より自然な色補正を行う機能です。

**使い方:**
1. 「除外マスク設定」アコーディオンを開く
2. 「除外マスクを有効化」にチェック
3. 除外領域を指定（以下のいずれか）:
   - **手動で塗る**: 青ブラシで青髭が広がっている部分を塗る
   - **Tab 1マスクを使用**: 「← Tab 1の全検出マスクを除外に追加」ボタン
4. 「色調補正 + オーバーレイを適用」をクリック

**動作原理:**
```
色補正対象 = 対象マスク（髭検出・点々）∪ 除外マスク（青髭領域）
参照サンプリング = 周辺肌色 − 除外マスク（青みを含む領域を除外）
```

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

### MAT Inpainting（Tab 2）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| インペインティング手法 | MAT (FFHQ) | FFHQ: 一般的な顔画像、CelebA-HQ: 有名人の顔画像 |
| MAT強化モード | ON | テクスチャ保持 + 青髭補正を有効化 |
| テクスチャ強度 | 0.8 | 元画像の肌の質感をどれだけ復元するか |
| 青髭補正強度 | 0.7 | 青みをどれだけ除去するか |

### 色調補正（Tab 3）

LAB色空間を使用して青髭を自然な肌色に補正します。

#### LABパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| a* 調整係数 | 30% (0.3) | 赤みの追加を最小限に抑制（オレンジ化防止） |
| b* 調整係数 | 60% (0.6) | 青み除去を控えめに（自然な色味を維持） |
| L 調整係数 | 50% (0.5) | 明度の上げすぎを抑制 |

この設定により、元の肌色・テクスチャを保持しながら青髭を自然に補正できます。

#### 補正モード

| モード | 用途 | スポイト領域 |
|--------|------|-------------|
| **青み除去（推奨）** | 青髭補正 | 不要 |
| 色味転送 | 汎用色補正 | 必要 |
| 自動補正 | 手軽な補正 | 不要 |

### 髭オーバーレイ（Tab 3）

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| オーバーレイ強度 | 1.0 | 1.0 = 完全に元の髭、0.5 = 半透明 |
| エッジぼかし | 3 | 髭の境界をぼかして自然に馴染ませる |

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

- **Gradio 5.x**: Web UI フレームワーク
- **MAT (Mask-Aware Transformer)**: CVPR 2022 Best Paper Finalist の高品質Inpainting
- **OpenCV**: 画像処理（ルールベース検出）
- **SimpleLama**: LaMa ベースの Inpainting
- **Segment Anything (SAM)**: セグメンテーション（オプション）
- **Grounding DINO**: テキストベース物体検出（オプション）

## ライセンス

MIT License
