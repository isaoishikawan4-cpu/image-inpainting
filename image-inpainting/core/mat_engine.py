"""MAT (Mask-Aware Transformer) Inpainting Engine.

This module provides the low-level interface for loading and running
MAT models for face inpainting.

CVPR 2022 Best Paper Finalist
https://github.com/fenglinglwb/MAT

Supports both original .pkl format and Hugging Face safetensors format.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import cv2
import torch

# Configuration
MAT_IMAGE_SIZE = 512
MAT_MODEL_DIR = Path(__file__).parent.parent / "checkpoints" / "mat"

# Supported model files (in order of preference)
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


class MATEngine:
    """MAT (Mask-Aware Transformer) inpainting engine.

    This engine handles:
    - Model loading from pickle files
    - Device management (CUDA/MPS/CPU with fallback)
    - Image resizing to 512x512 and restoration
    - Mask format conversion (v5 format to MAT format)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "ffhq",
        device: Optional[str] = None
    ):
        """Initialize MAT engine.

        Args:
            model_path: Path to .pkl checkpoint. If None, uses default path.
            model_type: "ffhq" or "celeba" (used if model_path is None)
            device: "cuda", "mps", or "cpu". If None, auto-detects.
        """
        self.model_type = model_type
        self._model = None
        self._device = None
        self._model_path = model_path or self._get_default_model_path(model_type)
        self._requested_device = device
        self._initialized = False

    def _get_default_model_path(self, model_type: str) -> Optional[str]:
        """Get default model path for given type.

        Searches for available model files in order of preference.
        """
        if model_type not in MAT_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MAT_MODELS.keys())}")

        # Search for available model file
        for filename in MAT_MODELS[model_type]:
            path = MAT_MODEL_DIR / filename
            if path.exists():
                return str(path)

        # Return first option as default (for error messages)
        return str(MAT_MODEL_DIR / MAT_MODELS[model_type][0])

    def _get_device(self) -> torch.device:
        """Get best available device with MPS fallback support."""
        if self._requested_device:
            return torch.device(self._requested_device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            # MPS may have issues with some operations
            # We'll try it and fall back to CPU if needed
            return torch.device("mps")

        return torch.device("cpu")

    @property
    def is_available(self) -> bool:
        """Check if MAT model is available."""
        return os.path.exists(self._model_path)

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the model."""
        if self._initialized:
            return True

        if not self.is_available:
            print(f"[MATEngine] Model not found at {self._model_path}")
            return False

        try:
            self._device = self._get_device()
            self._load_model()
            self._initialized = True
            print(f"[MATEngine] Initialized on {self._device}")
            return True
        except Exception as e:
            print(f"[MATEngine] Initialization failed: {e}")
            return False

    def _load_model(self):
        """Load MAT model from pickle or safetensors file."""
        from .mat.networks.mat import Generator

        print(f"[MATEngine] Loading model from {self._model_path}")

        # Create fresh Generator
        self._model = Generator(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=MAT_IMAGE_SIZE,
            img_channels=3
        )

        # Load weights based on file format
        if self._model_path.endswith('.safetensors'):
            self._load_safetensors()
        else:
            self._load_pkl()

        # Move to device and set to eval mode
        self._model = self._model.to(self._device).eval().requires_grad_(False)

        print(f"[MATEngine] Model loaded successfully")

    def _load_pkl(self):
        """Load model from original pickle format."""
        from .mat import dnnlib
        from .mat import legacy

        with dnnlib.util.open_url(self._model_path) as f:
            G_saved = legacy.load_network_pkl(f)['G_ema']

        # Copy parameters
        self._copy_params_and_buffers(G_saved, self._model, require_all=True)

    def _load_safetensors(self):
        """Load model from Hugging Face safetensors format."""
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors package is required for loading .safetensors files. "
                "Install with: pip install safetensors"
            )

        print(f"[MATEngine] Loading safetensors file...")
        state_dict = load_file(self._model_path)

        # The safetensors from Hugging Face has flattened keys
        # We need to load them directly into the model
        model_state = self._model.state_dict()

        # Try direct loading first
        missing_keys = []
        for key in model_state.keys():
            if key in state_dict:
                # Handle fp16 to fp32 conversion if needed
                src_tensor = state_dict[key]
                if src_tensor.dtype == torch.float16:
                    src_tensor = src_tensor.float()
                model_state[key].copy_(src_tensor)
            else:
                missing_keys.append(key)

        if missing_keys:
            print(f"[MATEngine] Warning: {len(missing_keys)} keys not found in safetensors")
            # Try to match with partial key names
            loaded = 0
            for key in missing_keys:
                # Try without module prefix
                short_key = key.split('.')[-1] if '.' in key else key
                for st_key, st_val in state_dict.items():
                    if st_key.endswith(short_key) or short_key in st_key:
                        if model_state[key].shape == st_val.shape:
                            if st_val.dtype == torch.float16:
                                st_val = st_val.float()
                            model_state[key].copy_(st_val)
                            loaded += 1
                            break
            print(f"[MATEngine] Loaded {loaded} additional keys by partial matching")

    def _copy_params_and_buffers(self, src_module, dst_module, require_all=False):
        """Copy parameters and buffers from source to destination module."""
        src_tensors = {name: tensor for name, tensor in self._named_params_and_buffers(src_module)}
        for name, tensor in self._named_params_and_buffers(dst_module):
            if name in src_tensors:
                tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)
            elif require_all:
                raise KeyError(f"Missing parameter: {name}")

    def _named_params_and_buffers(self, module):
        """Get all named parameters and buffers."""
        return list(module.named_parameters()) + list(module.named_buffers())

    def _convert_mask_v5_to_mat(self, mask: np.ndarray) -> np.ndarray:
        """Convert v5 mask format to MAT format.

        v5 format: 255 = inpaint region, 0 = preserve
        MAT format: 0 = inpaint (masked), 1 = preserve (remained)

        Args:
            mask: Binary mask (H, W) with values 0 or 255

        Returns:
            MAT format mask (H, W) with values 0.0-1.0
        """
        # Normalize to 0-1, then invert
        mat_mask = 1.0 - (mask.astype(np.float32) / 255.0)
        return mat_mask

    def _prepare_for_mat(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Resize and pad image/mask to 512x512.

        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W) in v5 format (255=inpaint)

        Returns:
            (processed_image, processed_mask, restore_info)
        """
        h, w = image.shape[:2]

        # Calculate scale to fit within 512x512
        scale = min(MAT_IMAGE_SIZE / h, MAT_IMAGE_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Resize mask (use NEAREST to preserve binary values)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Center padding
        pad_h = (MAT_IMAGE_SIZE - new_h) // 2
        pad_w = (MAT_IMAGE_SIZE - new_w) // 2

        # Create padded image (fill with zeros)
        padded_image = np.zeros((MAT_IMAGE_SIZE, MAT_IMAGE_SIZE, 3), dtype=np.uint8)
        padded_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_image

        # Create padded mask (fill with 0 = preserve in MAT format, so no inpainting on padding)
        # First convert to MAT format, then pad
        mat_mask = self._convert_mask_v5_to_mat(resized_mask)
        padded_mask = np.ones((MAT_IMAGE_SIZE, MAT_IMAGE_SIZE), dtype=np.float32)  # 1 = preserve
        padded_mask[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = mat_mask

        restore_info = {
            'original_size': (h, w),
            'resized_size': (new_h, new_w),
            'padding': (pad_h, pad_w),
            'scale': scale
        }

        return padded_image, padded_mask, restore_info

    def _restore_from_mat(
        self,
        result: np.ndarray,
        restore_info: Dict
    ) -> np.ndarray:
        """Restore result image to original size.

        Args:
            result: MAT output (512, 512, 3)
            restore_info: Dict with original size and padding info

        Returns:
            Restored image at original resolution
        """
        h, w = restore_info['original_size']
        new_h, new_w = restore_info['resized_size']
        pad_h, pad_w = restore_info['padding']

        # Remove padding
        cropped = result[pad_h:pad_h+new_h, pad_w:pad_w+new_w]

        # Resize back to original
        restored = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

        return restored

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        truncation_psi: float = 1.0,
        noise_mode: str = 'const'
    ) -> np.ndarray:
        """Perform inpainting with automatic resize handling.

        Args:
            image: RGB image (H, W, 3) as uint8
            mask: Binary mask (H, W) in v5 format (255 = inpaint region)
            truncation_psi: Truncation parameter for StyleGAN
            noise_mode: 'const', 'random', or 'none'

        Returns:
            Inpainted image (H, W, 3) at original size
        """
        if not self._ensure_initialized():
            raise RuntimeError("MATEngine failed to initialize. Check model path.")

        # Prepare inputs
        prepared_image, prepared_mask, restore_info = self._prepare_for_mat(image, mask)

        # Convert to tensors
        # Image: normalize to [-1, 1]
        image_tensor = torch.from_numpy(prepared_image).float().permute(2, 0, 1) / 127.5 - 1
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 512, 512)

        # Mask: (1, 1, 512, 512)
        mask_tensor = torch.from_numpy(prepared_mask).float().unsqueeze(0).unsqueeze(0)

        # Random latent
        z = torch.randn(1, 512)

        # No labels
        label = torch.zeros(1, 0)

        # Move to device
        try:
            image_tensor = image_tensor.to(self._device)
            mask_tensor = mask_tensor.to(self._device)
            z = z.to(self._device)
            label = label.to(self._device)

            # Run inference
            with torch.inference_mode():
                output = self._model(
                    image_tensor,
                    mask_tensor,
                    z,
                    label,
                    truncation_psi=truncation_psi,
                    noise_mode=noise_mode
                )

        except RuntimeError as e:
            # MPS fallback to CPU
            if "mps" in str(e).lower() or self._device.type == "mps":
                print(f"[MATEngine] MPS error, falling back to CPU: {e}")
                self._device = torch.device("cpu")
                self._model = self._model.to(self._device)

                image_tensor = image_tensor.to(self._device)
                mask_tensor = mask_tensor.to(self._device)
                z = z.to(self._device)
                label = label.to(self._device)

                with torch.inference_mode():
                    output = self._model(
                        image_tensor,
                        mask_tensor,
                        z,
                        label,
                        truncation_psi=truncation_psi,
                        noise_mode=noise_mode
                    )
            else:
                raise

        # Convert output to numpy
        # Output is in [-1, 1], convert to [0, 255]
        output = output.permute(0, 2, 3, 1)  # (1, 512, 512, 3)
        output = (output * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()

        # Restore to original size
        result = self._restore_from_mat(output, restore_info)

        return result


def get_mat_model_path(model_type: str) -> str:
    """Get path to MAT model checkpoint.

    Args:
        model_type: "ffhq" or "celeba"

    Returns:
        Path to model file

    Raises:
        FileNotFoundError: If model not found
    """
    if model_type not in MAT_MODELS:
        raise ValueError(f"Unknown model type: {model_type}")

    # Check for available model files
    for filename in MAT_MODELS[model_type]:
        # Check in main directory
        path = MAT_MODEL_DIR / filename
        if path.exists():
            return str(path)

        # Check in parent checkpoints
        path = Path(__file__).parent.parent / "checkpoints" / "mat" / filename
        if path.exists():
            return str(path)

        # Check in user cache
        path = Path.home() / ".cache" / "mat" / filename
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"MAT model '{model_type}' not found. "
        f"Download from https://huggingface.co/spacepxl/MAT-inpainting-fp16 "
        f"and place in {MAT_MODEL_DIR}/"
    )


def check_mat_availability() -> Dict[str, bool]:
    """Check which MAT models are available.

    Returns:
        Dict mapping model type to availability
    """
    result = {}
    for model_type, filenames in MAT_MODELS.items():
        available = False
        for filename in filenames:
            path = MAT_MODEL_DIR / filename
            if path.exists():
                available = True
                break
        result[model_type] = available
    return result
