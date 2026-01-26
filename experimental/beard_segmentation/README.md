# Experimental - 髭検出・カウント・Inpaintingツール

髭（ヒゲ）を検出・カウントし、指定した割合だけ除去（Inpainting）するための実験的ツール群です。

## ファイル一覧

| ファイル | 説明 | 必要環境 |
|---------|------|----------|
| `beard_counter_floodfill.py` | ルールベース髭検出（メイン） | OpenCV |
| `beard_sam_hf.py` | SAM（Segment Anything）版 | PyTorch, Transformers |
| `beard_grounded_sam.py` | Grounded SAM版（テキストプロンプト対応） | PyTorch, GroundingDINO, SAM |
| `beard_counter_inpaint.py` | 旧版（beard_counter_floodfill.pyに統合済み） | OpenCV |
| `setup_grounded_sam.sh` | Grounded SAMセットアップスクリプト | - |

---

## 1. beard_counter_floodfill.py（推奨）

**ルールベースの髭検出ツール。軽量で高速。**

### セットアップ
```bash
pip install opencv-python numpy
```

### 使い方
```bash
python beard_counter_floodfill.py <画像パス>
```

### 操作方法
| キー | 機能 |
|------|------|
| `1` | クリック塗りつぶしモード |
| `2` | 投げ縄（フリーハンド）モード |
| `3` | 矩形モード |
| `r` | 選択モード：ランダム |
| `l` | 選択モード：面積大きい順 |
| `s` | 選択モード：面積小さい順 |
| `d` | 選択モード：検出順 |
| `p` | Inpaintingプレビュー切替 |
| `n` | 新しいランダムシード |
| `m` | マスク保存 |
| `i` | Inpainting結果保存 |
| `u` | Undo |
| `c` | 全クリア |
| `q` | 終了 |

### スライドバー
- **Remove %**: 除去する髭の割合（0-100%）
- **Inpaint R**: Inpainting半径

---

## 2. beard_sam_hf.py（SAM版）

**Segment Anything Model（SAM）を使った高精度検出。個々のヒゲ分離機能付き。**

### セットアップ
```bash
pip install torch torchvision transformers opencv-python
```

### 使い方
```bash
python beard_sam_hf.py <画像パス>
```

### 操作方法
| キー | 機能 |
|------|------|
| `1` | ポイントモード（左クリック=前景、右クリック=背景） |
| `2` | ボックスモード（自動ヒゲ分離ON） |
| `3` | ボックスモード（分離なし） |
| `Enter` | ポイントからセグメント実行 |
| `h` | 最後の領域を個々のヒゲに分離 |
| `r` | 選択モード：ランダム |
| `l` | 選択モード：面積大きい順 |
| `p` | Inpaintingプレビュー切替 |
| `m` | マスク保存 |
| `u` | Undo |
| `c` | 全クリア |
| `q` | 終了 |

### スライドバー
- **Remove %**: 除去する髭の割合
- **Hair Thresh**: ヒゲ検出閾値（暗さ）

### 特徴
- SAMで大きな領域を検出後、自動的に個々のヒゲに分離
- MPS（Mac GPU）/CUDA/CPU自動検出

---

## 3. beard_grounded_sam.py（Grounded SAM版）

**テキストプロンプトで髭を自動検出（"beard"等で指定）。**

### セットアップ
```bash
chmod +x setup_grounded_sam.sh
./setup_grounded_sam.sh
```

または手動で：
```bash
pip install torch torchvision segment-anything groundingdino-py
# モデルダウンロード
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 使い方
```bash
python beard_grounded_sam.py <画像パス>
```

### 操作方法
| キー | 機能 |
|------|------|
| `g` | Grounded SAMで自動検出 |
| `1` | クリック塗りつぶしモード |
| `2` | 投げ縄モード |
| `3` | 矩形モード |
| `t` | プロンプト変更 |
| `f` | 選択モード：信頼度順 |
| `p` | Inpaintingプレビュー |
| `m` | マスク保存 |
| `i` | Inpainting結果保存 |
| `q` | 終了 |

### デフォルトプロンプト
```
beard. facial hair. stubble.
```

---

## 共通機能

### 出力ファイル
- `beard_mask.png`: Inpainting用マスク（白=除去領域）
- `beard_inpainted.png`: OpenCV Inpainting結果

### ワークフロー例

1. **髭を検出**（投げ縄/ボックス/SAMで領域選択）
2. **Remove %スライダー**で除去する割合を調整
3. **`p`キー**でInpaintingプレビュー確認
4. **`m`キー**でマスク保存 → 外部Inpaintingモデル（Stable Diffusion等）で使用

### 比較表

| 機能 | floodfill | sam_hf | grounded_sam |
|------|-----------|--------|--------------|
| セットアップ | 簡単 | 中程度 | 複雑 |
| GPU | 不要 | 推奨 | 必須 |
| 自動検出 | × | △ | ○ |
| テキストプロンプト | × | × | ○ |
| 個別ヒゲ分離 | ○ | ○ | △ |
| 処理速度 | 高速 | 中程度 | 遅い |

---

## 技術詳細

### ルールベース検出（floodfill）
1. CLAHE（コントラスト強調）
2. 適応的閾値処理 + 通常閾値処理
3. モルフォロジー演算（Opening/Closing）
4. 連結成分分析

### SAMベース検出
1. SAMで大きな領域をセグメント
2. 領域内でルールベース処理
3. 連結成分分析で個々のヒゲに分離

### Inpainting
- OpenCV Telea法（高速）
- 外部モデル用マスク出力対応
