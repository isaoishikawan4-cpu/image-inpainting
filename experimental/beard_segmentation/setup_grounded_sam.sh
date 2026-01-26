#!/bin/bash
# Grounded SAM セットアップスクリプト
#
# 使用方法:
#   chmod +x setup_grounded_sam.sh
#   ./setup_grounded_sam.sh

set -e

echo "=========================================="
echo "Grounded SAM セットアップ"
echo "=========================================="

# 作業ディレクトリ
WORK_DIR=$(pwd)
MODEL_DIR="${WORK_DIR}/models"
mkdir -p "${MODEL_DIR}"

# 1. 必要なPythonパッケージのインストール
echo ""
echo "[1/4] Pythonパッケージをインストール中..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pycocotools matplotlib
pip install segment-anything
pip install supervision==0.21.0

# 2. Grounding DINOのインストール
echo ""
echo "[2/4] Grounding DINOをインストール中..."
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
fi
cd GroundingDINO
pip install -e .
cd "${WORK_DIR}"

# 3. SAMモデルのダウンロード
echo ""
echo "[3/4] SAMモデルをダウンロード中..."
SAM_CHECKPOINT="${MODEL_DIR}/sam_vit_h_4b8939.pth"
if [ ! -f "${SAM_CHECKPOINT}" ]; then
    echo "SAM ViT-H モデルをダウンロード (2.4GB)..."
    wget -P "${MODEL_DIR}" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
    echo "SAMモデルは既に存在します"
fi

# 4. Grounding DINOモデルのダウンロード
echo ""
echo "[4/4] Grounding DINOモデルをダウンロード中..."
DINO_CHECKPOINT="${MODEL_DIR}/groundingdino_swint_ogc.pth"
if [ ! -f "${DINO_CHECKPOINT}" ]; then
    echo "Grounding DINO SwinT モデルをダウンロード..."
    wget -P "${MODEL_DIR}" https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "Grounding DINOモデルは既に存在します"
fi

echo ""
echo "=========================================="
echo "セットアップ完了！"
echo "=========================================="
echo ""
echo "モデルの場所:"
echo "  SAM: ${SAM_CHECKPOINT}"
echo "  Grounding DINO: ${DINO_CHECKPOINT}"
echo ""
echo "使用方法:"
echo "  python beard_grounded_sam.py <画像パス>"
echo ""
echo "注意: GPUが必要です（CUDA対応）"
echo "=========================================="
