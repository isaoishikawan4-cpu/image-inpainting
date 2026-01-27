"""Beard detection using Grounded SAM and rule-based methods.

This module provides multiple detection backends for identifying beard regions
in images, including AI-based (Grounded SAM) and traditional CV approaches.
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import os


class DetectionBackend(Enum):
    """Available detection backends."""
    GROUNDED_SAM = "grounded_sam"
    RULE_BASED = "rule_based"


@dataclass
class DetectedRegion:
    """Represents a single detected beard region."""
    mask: np.ndarray          # Binary mask (H, W)
    area: int                 # Pixel count
    centroid: Tuple[int, int] # (cx, cy)
    confidence: float         # Detection confidence (0-1)
    source: str               # 'grounded_sam' or 'rule_based'
    phrase: str = ""          # Detected phrase (for Grounded SAM)


# Check library availability
TORCH_AVAILABLE = False
SAM_AVAILABLE = False
GROUNDING_DINO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    pass

try:
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    pass


class DetectorBackendBase(ABC):
    """Abstract base class for detection backends."""

    @abstractmethod
    def detect(
        self,
        image_rgb: np.ndarray,
        region_box: Tuple[int, int, int, int],
        **kwargs
    ) -> List[DetectedRegion]:
        """
        Detect beard regions within specified box.

        Args:
            image_rgb: RGB image array
            region_box: (x1, y1, x2, y2) detection region
            **kwargs: Backend-specific parameters

        Returns:
            List of DetectedRegion objects
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class GroundedSAMBackend(DetectorBackendBase):
    """Grounded SAM detection backend."""

    def __init__(
        self,
        sam_checkpoint: str = "sam_vit_h_4b8939.pth",
        sam_model_type: str = "vit_h",
        grounding_dino_config: Optional[str] = None,
        grounding_dino_checkpoint: str = "groundingdino_swint_ogc.pth"
    ):
        self._sam_checkpoint = sam_checkpoint
        self._sam_model_type = sam_model_type
        self._dino_config = grounding_dino_config
        self._dino_checkpoint = grounding_dino_checkpoint
        self._initialized = False
        self.sam_predictor = None
        self.grounding_dino_model = None
        # Device selection: MPS (Apple Silicon) > CUDA > CPU
        if TORCH_AVAILABLE:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

    def _find_checkpoint(self, filename: str) -> Optional[str]:
        """Search for checkpoint file in common locations."""
        search_paths = [
            filename,
            os.path.join(os.path.dirname(__file__), "..", filename),
            os.path.join(os.path.dirname(__file__), "..", "..", filename),
            os.path.expanduser(f"~/{filename}"),
        ]
        for path in search_paths:
            if os.path.exists(path):
                return path
        return None

    def initialize(self) -> bool:
        """Lazy initialization of models."""
        if self._initialized:
            return True

        if not SAM_AVAILABLE or not GROUNDING_DINO_AVAILABLE:
            print("Grounded SAM libraries not available")
            return False

        # Load SAM
        sam_path = self._find_checkpoint(self._sam_checkpoint)
        if sam_path is None:
            print(f"SAM checkpoint not found: {self._sam_checkpoint}")
            return False

        try:
            print(f"Loading SAM model: {sam_path}")
            sam = sam_model_registry[self._sam_model_type](checkpoint=sam_path)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            print(f"SAM: Loaded (device={self.device})")
        except Exception as e:
            print(f"SAM initialization error: {e}")
            return False

        # Load Grounding DINO
        dino_path = self._find_checkpoint(self._dino_checkpoint)
        if dino_path is None:
            print(f"Grounding DINO checkpoint not found: {self._dino_checkpoint}")
            return False

        # Find config file
        dino_config_path = None

        if self._dino_config and os.path.exists(self._dino_config):
            dino_config_path = self._dino_config
        else:
            try:
                import groundingdino
                package_config = os.path.join(
                    os.path.dirname(groundingdino.__file__),
                    "config",
                    "GroundingDINO_SwinT_OGC.py"
                )
                if os.path.exists(package_config):
                    dino_config_path = package_config
                    print("Grounding DINO config: loaded from pip package")
            except ImportError:
                pass

            if dino_config_path is None:
                fallback_paths = [
                    os.path.join(os.path.dirname(__file__), "..", "..", "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py"),
                    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                ]
                for path in fallback_paths:
                    if os.path.exists(path):
                        dino_config_path = path
                        break

        if dino_config_path is None:
            print("Grounding DINO config not found")
            return False

        try:
            print(f"Loading Grounding DINO model: {dino_path}")
            self.grounding_dino_model = load_model(dino_config_path, dino_path)
            print("Grounding DINO: Loaded")
        except Exception as e:
            print(f"Grounding DINO initialization error: {e}")
            return False

        self._initialized = True
        return True

    def is_available(self) -> bool:
        """Check if SAM and Grounding DINO are loaded."""
        return (
            self._initialized
            and self.sam_predictor is not None
            and self.grounding_dino_model is not None
        )

    def _detect_with_prompt(
        self,
        image_rgb: np.ndarray,
        text_prompt: str = "beard. facial hair. stubble.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20
    ) -> List[Dict]:
        """Detect objects using text prompt."""
        if not self.is_available():
            raise RuntimeError("Grounded SAM not initialized")

        # Grounding DINO detection
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
            print("Grounding DINO: No regions detected")
            return []

        # Scale boxes to image size
        h, w = image_rgb.shape[:2]
        boxes_scaled = boxes * torch.tensor([w, h, w, h])
        boxes_xyxy = boxes_scaled.cpu().numpy()

        # SAM segmentation
        self.sam_predictor.set_image(image_rgb)

        results = []
        for i, (box, conf) in enumerate(zip(boxes_xyxy, logits)):
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

        print(f"Grounded SAM: {len(results)} regions detected")
        return results

    def detect(
        self,
        image_rgb: np.ndarray,
        region_box: Tuple[int, int, int, int],
        text_prompt: str = "beard. facial hair. stubble.",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        min_area: int = 10,
        max_area: int = 50000
    ) -> List[DetectedRegion]:
        """Detect using Grounded SAM within region."""
        if not self.is_available():
            if not self.initialize():
                raise RuntimeError("Grounded SAM not available")

        x1, y1, x2, y2 = region_box
        h_orig, w_orig = image_rgb.shape[:2]

        # Crop region
        roi = image_rgb[y1:y2, x1:x2].copy()
        print(f"ROI size: {roi.shape[1]}x{roi.shape[0]}")

        # Detect in cropped region
        print(f"Grounded SAM params: prompt='{text_prompt}', box_thresh={box_threshold}, text_thresh={text_threshold}")
        roi_results = self._detect_with_prompt(
            roi, text_prompt, box_threshold, text_threshold
        )

        print(f"Grounded SAM raw results: {len(roi_results)}")
        for i, r in enumerate(roi_results):
            area = cv2.countNonZero(r['mask'])
            print(f"  [{i}] phrase='{r['phrase']}', confidence={r['confidence']:.3f}, area={area}")

        if not roi_results:
            print("No regions detected in ROI")
            return []

        # Convert to full image coordinates
        results = []
        for result in roi_results:
            roi_mask = result['mask']
            area = cv2.countNonZero(roi_mask)

            # Area filter
            if area < min_area or area > max_area:
                print(f"  Region filtered (area): area={area} (range: {min_area}-{max_area})")
                continue

            # Create full-size mask
            full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = roi_mask

            # Calculate centroid
            M = cv2.moments(full_mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

            results.append(DetectedRegion(
                mask=full_mask,
                area=area,
                centroid=(cx, cy),
                confidence=result['confidence'],
                source='grounded_sam',
                phrase=result['phrase']
            ))

        print(f"Grounded SAM (ROI): {len(results)} regions detected")
        return results


class RuleBasedBackend(DetectorBackendBase):
    """Rule-based detection using image processing."""

    def is_available(self) -> bool:
        """Always available (uses OpenCV only)."""
        return True

    def detect(
        self,
        image_rgb: np.ndarray,
        region_box: Tuple[int, int, int, int],
        threshold_value: int = 80,
        min_area: int = 10,
        max_area: int = 5000
    ) -> List[DetectedRegion]:
        """Detect using adaptive thresholding and contours."""
        x1, y1, x2, y2 = region_box
        h_orig, w_orig = image_rgb.shape[:2]

        # Crop ROI and convert to BGR
        roi_rgb = image_rgb[y1:y2, x1:x2]
        roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # CLAHE (contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Fixed threshold
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)

        # Combine both masks
        mask = cv2.bitwise_and(adaptive, binary)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Create full-size mask
                full_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
                offset_contour = contour + np.array([x1, y1])
                cv2.drawContours(full_mask, [offset_contour], -1, 255, -1)

                # Calculate centroid
                M = cv2.moments(full_mask)
                cx = int(M['m10'] / M['m00']) if M['m00'] > 0 else x1
                cy = int(M['m01'] / M['m00']) if M['m00'] > 0 else y1

                results.append(DetectedRegion(
                    mask=full_mask,
                    area=int(area),
                    centroid=(cx, cy),
                    confidence=1.0,  # Rule-based has confidence 1.0
                    source='rule_based',
                    phrase=""
                ))

        print(f"Rule-based detection: {len(results)} beards detected")
        return results


class BeardDetector:
    """Unified beard detector with multiple backends."""

    def __init__(self):
        self._grounded_sam: Optional[GroundedSAMBackend] = None
        self._rule_based: RuleBasedBackend = RuleBasedBackend()

    def get_backend(self, backend_type: DetectionBackend) -> DetectorBackendBase:
        """Get or initialize the specified backend."""
        if backend_type == DetectionBackend.GROUNDED_SAM:
            if self._grounded_sam is None:
                self._grounded_sam = GroundedSAMBackend()
            return self._grounded_sam
        else:
            return self._rule_based

    def detect(
        self,
        image_rgb: np.ndarray,
        region_box: Tuple[int, int, int, int],
        backend: DetectionBackend = DetectionBackend.RULE_BASED,
        **kwargs
    ) -> List[DetectedRegion]:
        """
        Detect beard regions using specified backend.

        Args:
            image_rgb: RGB image array
            region_box: (x1, y1, x2, y2) detection region
            backend: Which detection backend to use
            **kwargs: Backend-specific parameters

        Returns:
            List of DetectedRegion objects
        """
        detector = self.get_backend(backend)
        return detector.detect(image_rgb, region_box, **kwargs)

    def is_grounded_sam_available(self) -> bool:
        """Check if Grounded SAM backend is available."""
        if self._grounded_sam is None:
            self._grounded_sam = GroundedSAMBackend()
        return self._grounded_sam.is_available() or self._grounded_sam.initialize()
