"""Configuration constants for the beard thinning application."""

# 髭薄め段階（%）
DEFAULT_THINNING_LEVELS = [30, 50, 70, 100]

# 画像処理設定
MAX_IMAGE_SIZE = 2048
BRUSH_RADIUS_DEFAULT = 20

# 対応フォーマット
SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'webp']

# マスク境界のぼかし設定
FEATHER_RADIUS = 5

# MAT (Mask-Aware Transformer) 設定
MAT_MODEL_DIR = "checkpoints/mat"
MAT_IMAGE_SIZE = 512
# 対応モデルファイル（優先順）
MAT_MODELS = {
    "ffhq": [
        "MAT_FFHQ_512_fp16.safetensors",  # Hugging Face mirror
        "FFHQ-512.pkl",                    # Original format
    ],
    "celeba": [
        "MAT_CelebA-HQ_512_fp16.safetensors",  # Hugging Face mirror
        "CelebA-HQ.pkl",                        # Original format
    ]
}
