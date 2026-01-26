#!/usr/bin/env python3
"""
DINOv2 Inference Viewer with Trained Classifier (Pixel-Level Edition)

Interactive viewer using trained DINOv2+MLP model for pixel-level similarity visualization.

Features:
    - PIXEL-LEVEL similarity maps using bilinear upsampling + Gaussian smoothing
    - Based on AnomalyDINO methodology for high-resolution anomaly detection
    - GPU-accelerated computation with intelligent caching
    - 14x14 patch features → Full pixel resolution (e.g., 896x896 pixels)

Usage:
    python dinov2_inference_viewer.py <image_path> <model_path> [--model MODEL] [--device DEVICE]

Arguments:
    --model: DINOv2 model name (default: dinov2_vitg14)
             Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    --device: Device to use (cuda/mps/cpu, default: auto)

Controls:
    - Mode button: Toggle between Point/Rectangle mode
    - Binary button: Toggle binary visualization on/off
    - Threshold slider: Skin region threshold (0-100%, similarity)
    - Min Area slider: Minimum blob area in pixels
    - Max Area slider: Maximum blob area in pixels
    - Circularity slider: Minimum circularity (0-100%)
    - Point mode: Left click to select a single pixel
    - Rectangle mode: Click two points to define a rectangle
    - Right click: Reset selected features
    - 'q' or ESC: Quit

Note:
    - Beards detected using SimpleBlobDetector (dark blobs)
    - Stage 1: Create skin mask from similarity (RED regions)
    - Stage 2: Detect dark blobs within skin regions
    - Threshold: Defines skin region boundary
    - Binary mode: Shows skin regions (red) vs non-skin (black)
    - Green circles with yellow crosses mark detected beards
    - Adjust Min/Max Area and Circularity for better detection

Technical Details:
    - Upsampling: Bilinear interpolation from patch grid to pixel grid
    - Smoothing: Gaussian filter (σ=4.0, kernel_size=17)
    - Reference: AnomalyDINO (Damm et al., WACV 2025)
"""

import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms


class MLPClassifier(nn.Module):
    """MLP classifier on top of DINOv2 features"""

    def __init__(self, input_dim=384, hidden_dim=256, num_classes=3, projection_dim=128):
        super().__init__()

        # Classification head: 5 hidden layers + output
        self.mlp = self._make_mlp_layers(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_hidden_layers=5,
            dropout_rate=0.3
        )

        # Projection head for contrastive learning: 2 hidden layers + output
        self.projection = self._make_mlp_layers(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            num_hidden_layers=2,
            dropout_rate=0.2
        )

    def _make_mlp_layers(self, input_dim, hidden_dim, output_dim, num_hidden_layers, dropout_rate):
        """Create MLP layers with specified architecture"""
        layers = []

        # First layer: input_dim -> hidden_dim
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_hidden_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        # Output layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, x, return_embedding=False):
        logits = self.mlp(x)

        if return_embedding:
            embeddings = self.projection(x)
            embeddings = F.normalize(embeddings, dim=1)
            return logits, embeddings

        return logits


class DINOv2InferenceViewer:
    CLASSES = ['black_beard', 'white_beard', 'other']
    CLASS_COLORS = {
        'black_beard': (0, 0, 255),      # Red
        'white_beard': (255, 0, 0),      # Blue
        'other': (0, 255, 0)             # Green
    }

    def __init__(self, image_path, model_path, model_name='dinov2_vitg14', device=None):
        self.image_path = image_path
        self.model_path = model_path

        # Select device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load DINOv2 model
        print(f"Loading DINOv2 model: {model_name}...")
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dinov2_model.to(self.device)
        self.dinov2_model.eval()

        # Freeze DINOv2
        for param in self.dinov2_model.parameters():
            param.requires_grad = False

        # Load trained classifier
        print(f"Loading trained classifier from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        input_dim = checkpoint['input_dim']
        projection_dim = checkpoint.get('projection_dim', 128)

        self.classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=256,
            num_classes=3,
            projection_dim=projection_dim
        )
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.to(self.device)
        self.classifier.eval()

        print(f"Model loaded (accuracy: {checkpoint.get('accuracy', 'N/A')})")

        # Extract features at patch resolution
        print("Extracting features...")
        self.features, self.feature_h, self.feature_w = self._extract_features()
        print(f"Feature map: {self.feature_h}×{self.feature_w} patches")
        print(f"Display image: {self.display_h}×{self.display_w} pixels")

        # Precompute embeddings for all patches
        print("Computing embeddings...")
        self.embeddings = self._compute_embeddings()
        print(f"Embeddings shape: {self.embeddings.shape}")

        # Compute class probabilities for all patches
        print("Computing class probabilities...")
        self.class_probs = self._compute_class_probabilities()
        print(f"Class probabilities shape: {self.class_probs.shape}")

        self.selected_embeddings = []
        self.click_positions = []
        self.selected_rectangles = []
        self.similarity_map = None  # Patch-level similarity map for peak detection
        self.similarity_map_pixel = None  # Pixel-level similarity map for visualization

        # Store image dimensions
        self.img_height = self.original_image.shape[0]
        self.img_width = self.original_image.shape[1]

        # Mode: 'point' or 'rectangle'
        self.mode = 'point'

        # Binary mode
        self.binary_mode = False
        self.threshold = 70  # 0-100 scale, maps to 0.0-1.0 similarity threshold (for skin region mask)

        # SimpleBlobDetector parameters for beard detection
        self.blob_min_area = 5  # Minimum blob area in pixels
        self.blob_max_area = 100  # Maximum blob area in pixels
        self.blob_circularity = 30  # Minimum circularity (0-100 scale)

        # Blob detection results
        self.detected_blobs = None  # Will store detected blob keypoints

        # Cache for Gaussian kernel (computed once, reused for all similarity computations)
        self.gaussian_kernel = None

        # Note: self.gray_image is already set in _extract_features()

        # Rectangle drawing state
        self.rect_start = None
        self.rect_end = None
        self.drawing = False
        self.mouse_pos = None

        # Button dimensions
        self.button_height = 50
        self.button_width = 150
        self.button_margin = 10

        # Setup window
        cv2.namedWindow('DINOv2 Inference Viewer')
        cv2.setMouseCallback('DINOv2 Inference Viewer', self._mouse_callback)
        cv2.createTrackbar('Threshold', 'DINOv2 Inference Viewer',
                          self.threshold, 100, self._on_threshold_change)
        cv2.createTrackbar('Min Area', 'DINOv2 Inference Viewer',
                          self.blob_min_area, 200, self._on_blob_min_area_change)
        cv2.createTrackbar('Max Area', 'DINOv2 Inference Viewer',
                          self.blob_max_area, 500, self._on_blob_max_area_change)
        cv2.createTrackbar('Circularity', 'DINOv2 Inference Viewer',
                          self.blob_circularity, 100, self._on_blob_circularity_change)

    def _extract_features(self):
        """Extract features at native patch resolution"""
        # Load original image
        pil_image = Image.open(self.image_path).convert('RGB')
        orig_w, orig_h = pil_image.size

        # Save original image for display and grayscale for blob detection
        self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        # Resize to multiple of 14 for DINOv2
        patch_size = 14
        dino_w = max(patch_size, (orig_w // patch_size) * patch_size)
        dino_h = max(patch_size, (orig_h // patch_size) * patch_size)

        # Resize image to exact multiple of patch_size
        pil_image = pil_image.resize((dino_w, dino_h), Image.BILINEAR)

        # Save display dimensions
        self.display_w = dino_w
        self.display_h = dino_h

        # Update display image to match DINOv2 input
        self.original_image = cv2.resize(self.original_image, (dino_w, dino_h))
        self.gray_image = cv2.resize(self.gray_image, (dino_w, dino_h))

        # DINOv2 preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.dinov2_model.forward_features(img_tensor)
            patch_features = features['x_norm_patchtokens']  # [1, num_patches, feature_dim]

        # Get patch grid dimensions
        patch_h = dino_h // patch_size
        patch_w = dino_w // patch_size

        # Reshape to spatial grid: [patch_h, patch_w, feature_dim]
        feature_map = patch_features.reshape(1, patch_h, patch_w, -1)
        feature_map = feature_map.squeeze(0)

        # Normalize features
        feature_map = F.normalize(feature_map, dim=2)

        return feature_map.cpu(), patch_h, patch_w

    def _compute_embeddings(self):
        """Compute MLP embeddings for all patches"""
        # Reshape features to [num_patches, feature_dim]
        features_flat = self.features.reshape(-1, self.features.shape[2])

        # Move to device and compute embeddings in batches
        batch_size = 256
        embeddings_list = []

        with torch.no_grad():
            for i in range(0, features_flat.shape[0], batch_size):
                batch = features_flat[i:i+batch_size].to(self.device)
                _, batch_embeddings = self.classifier(batch, return_embedding=True)
                embeddings_list.append(batch_embeddings.cpu())

        # Concatenate and reshape back to spatial grid
        embeddings_flat = torch.cat(embeddings_list, dim=0)
        embeddings = embeddings_flat.reshape(self.feature_h, self.feature_w, -1)

        return embeddings

    def _compute_class_probabilities(self):
        """Compute class probabilities for all patches"""
        # Reshape features to [num_patches, feature_dim]
        features_flat = self.features.reshape(-1, self.features.shape[2])

        # Compute logits in batches
        batch_size = 256
        probs_list = []

        with torch.no_grad():
            for i in range(0, features_flat.shape[0], batch_size):
                batch = features_flat[i:i+batch_size].to(self.device)
                logits = self.classifier(batch)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs.cpu())

        # Concatenate and reshape back to spatial grid
        probs_flat = torch.cat(probs_list, dim=0)
        probs = probs_flat.reshape(self.feature_h, self.feature_w, len(self.CLASSES))

        return probs.numpy()

    def _pixel_to_patch(self, x, y):
        """Convert pixel coordinates to patch coordinates"""
        patch_x = min(x // 14, self.feature_w - 1)
        patch_y = min(y // 14, self.feature_h - 1)
        return patch_x, patch_y

    def _on_threshold_change(self, value):
        """Callback for similarity threshold trackbar (for skin region mask)"""
        self.threshold = value
        if self.similarity_map_pixel is not None:
            self._detect_beards_blob()  # Re-detect beards with new threshold
            self._update_display()

    def _on_blob_min_area_change(self, value):
        """Callback for blob minimum area trackbar"""
        self.blob_min_area = max(1, value)  # Ensure at least 1
        if self.similarity_map_pixel is not None:
            self._detect_beards_blob()  # Re-detect beards with new parameters
            self._update_display()

    def _on_blob_max_area_change(self, value):
        """Callback for blob maximum area trackbar"""
        self.blob_max_area = max(self.blob_min_area + 1, value)  # Must be > min_area
        if self.similarity_map_pixel is not None:
            self._detect_beards_blob()  # Re-detect beards with new parameters
            self._update_display()

    def _on_blob_circularity_change(self, value):
        """Callback for blob circularity trackbar"""
        self.blob_circularity = value
        if self.similarity_map_pixel is not None:
            self._detect_beards_blob()  # Re-detect beards with new parameters
            self._update_display()

    def _is_button_click(self, x, y, button_index):
        """Check if click is on a specific button (0=mode, 1=binary)"""
        button_x = self.button_margin + (self.button_width + self.button_margin) * button_index
        button_y = self.button_margin
        return (button_x <= x <= button_x + self.button_width and
                button_y <= y <= button_y + self.button_height)

    def _mouse_callback(self, event, x, y, _flags, _param):
        """Handle mouse click events"""
        # Adjust y coordinate for button area
        button_area_height = self.button_height + self.button_margin * 2
        display_y = y - button_area_height

        # Right click: reset
        if event == cv2.EVENT_RBUTTONDOWN:
            self.reset()
            return

        # Check for button clicks
        if event == cv2.EVENT_LBUTTONDOWN and y < button_area_height:
            # Mode button (0)
            if self._is_button_click(x, y, 0):
                self.mode = 'rectangle' if self.mode == 'point' else 'point'
                self.rect_start = None
                self.rect_end = None
                self.drawing = False
                print(f"Switched to {self.mode.upper()} mode")
                self._update_display()
                return

            # Binary button (1)
            if self._is_button_click(x, y, 1):
                self.binary_mode = not self.binary_mode
                print(f"Binary mode: {'ON' if self.binary_mode else 'OFF'}")
                if self.similarity_map_pixel is not None:
                    self._detect_beards_blob()  # Re-detect beards with new mode
                self._update_display()
                return

        # Mouse move: track position for rectangle preview
        if event == cv2.EVENT_MOUSEMOVE:
            if display_y >= 0 and x < self.img_width:
                self.mouse_pos = (x, display_y)
                if self.drawing and self.mode == 'rectangle':
                    self._update_display()
            return

        # Left click on image area
        if event == cv2.EVENT_LBUTTONDOWN and display_y >= 0:
            # Check bounds
            if x < 0 or x >= self.img_width or display_y < 0 or display_y >= self.img_height:
                return

            if self.mode == 'point':
                # Get patch coordinates
                patch_x, patch_y = self._pixel_to_patch(x, display_y)

                # Extract embedding at this patch
                embedding = self.embeddings[patch_y, patch_x]

                self.selected_embeddings.append(embedding)
                self.click_positions.append((x, display_y))

                # Get class prediction for this pixel
                probs = self.class_probs[patch_y, patch_x]
                pred_class_idx = np.argmax(probs)
                pred_class = self.CLASSES[pred_class_idx]
                confidence = probs[pred_class_idx] * 100

                print(f"Selected point: ({x}, {display_y}) -> patch ({patch_x}, {patch_y})")
                print(f"  Predicted class: {pred_class} ({confidence:.1f}%)")
                print(f"  Probabilities: black={probs[0]*100:.1f}%, white={probs[1]*100:.1f}%, other={probs[2]*100:.1f}%")

                self._compute_similarity()
                self._update_display()

            elif self.mode == 'rectangle':
                if not self.drawing:
                    # First click: start rectangle
                    self.rect_start = (x, display_y)
                    self.drawing = True
                    print(f"Rectangle start: ({x}, {display_y})")
                else:
                    # Second click: finish rectangle
                    self.rect_end = (x, display_y)
                    self.drawing = False

                    # Get rectangle bounds
                    x1, y1 = min(self.rect_start[0], self.rect_end[0]), min(self.rect_start[1], self.rect_end[1])
                    x2, y2 = max(self.rect_start[0], self.rect_end[0]), max(self.rect_start[1], self.rect_end[1])

                    # Convert to patch coordinates
                    patch_x1, patch_y1 = self._pixel_to_patch(x1, y1)
                    patch_x2, patch_y2 = self._pixel_to_patch(x2, y2)

                    # Extract embeddings in rectangle and average
                    rect_embeddings = self.embeddings[patch_y1:patch_y2+1, patch_x1:patch_x2+1]  # [h, w, emb_dim]
                    avg_embedding = rect_embeddings.mean(dim=(0, 1))  # [emb_dim]
                    avg_embedding = F.normalize(avg_embedding.unsqueeze(0), dim=1).squeeze(0)

                    self.selected_embeddings.append(avg_embedding)
                    self.selected_rectangles.append(((x1, y1), (x2, y2)))

                    print(f"Selected rectangle: ({x1}, {y1}) to ({x2}, {y2})")
                    print(f"  Patch region: ({patch_x1}, {patch_y1}) to ({patch_x2}, {patch_y2})")

                    self.rect_start = None
                    self.rect_end = None
                    self._compute_similarity()
                    self._update_display()

    # def _compute_similarity(self):
    #     """Compute similarity map based on selected embeddings

    #     Computes pixel-level similarity using bilinear upsampling and Gaussian smoothing,
    #     following the AnomalyDINO methodology for high-resolution anomaly maps.
    #     """
    #     if len(self.selected_embeddings) == 0:
    #         self.similarity_map = None
    #         self.similarity_map_pixel = None
    #         self.detected_blobs = None
    #         return

    #     print("Computing similarity map...")

    #     # Average all selected embeddings
    #     avg_embedding = torch.stack(self.selected_embeddings).mean(dim=0)  # [emb_dim]

    #     # Compute cosine similarity with all patches on GPU for efficiency
    #     # embeddings: [patch_h, patch_w, emb_dim]
    #     # avg_embedding: [emb_dim]
    #     embeddings_gpu = self.embeddings.to(self.device)
    #     avg_embedding_gpu = avg_embedding.to(self.device)

    #     similarity_map = torch.matmul(embeddings_gpu, avg_embedding_gpu)  # [patch_h, patch_w]

    #     # Keep patch-level similarity for peak detection
    #     self.similarity_map = similarity_map.cpu().numpy()

    #     # Upsample to pixel-level resolution using bilinear interpolation
    #     # Add batch and channel dimensions: [1, 1, patch_h, patch_w]
    #     similarity_map_4d = similarity_map.unsqueeze(0).unsqueeze(0)

    #     # Bilinear upsampling to pixel resolution
    #     similarity_map_upsampled = F.interpolate(
    #         similarity_map_4d,
    #         size=(self.display_h, self.display_w),
    #         mode='bilinear',
    #         align_corners=False
    #     )  # [1, 1, display_h, display_w]

    #     # Apply Gaussian smoothing for spatial coherence (σ=4.0 as in AnomalyDINO)
    #     # Use cached Gaussian kernel for efficiency
    #     if self.gaussian_kernel is None:
    #         # Create Gaussian kernel (only computed once)
    #         kernel_size = 17  # Should be odd, large enough to capture σ=4.0
    #         sigma = 4.0

    #         # Create 1D Gaussian kernel
    #         x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
    #         gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
    #         gauss_1d = gauss_1d / gauss_1d.sum()

    #         # Create 2D Gaussian kernel via outer product
    #         gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
    #         gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_size, kernel_size]

    #         # Cache for future use
    #         self.gaussian_kernel = gauss_2d
    #         print(f"Gaussian kernel created and cached (size={kernel_size}, σ={sigma})")

    #     # Apply Gaussian filter with padding
    #     kernel_size = self.gaussian_kernel.shape[-1]
    #     padding = kernel_size // 2
    #     similarity_map_smoothed = F.conv2d(
    #         similarity_map_upsampled,
    #         self.gaussian_kernel,
    #         padding=padding
    #     )  # [1, 1, display_h, display_w]

    #     # Store pixel-level similarity map
    #     self.similarity_map_pixel = similarity_map_smoothed.squeeze().cpu().numpy()

    #     print(f"Pixel-level similarity map computed: {self.similarity_map_pixel.shape}")
    #     print(f"  Range: [{self.similarity_map_pixel.min():.3f}, {self.similarity_map_pixel.max():.3f}]")
    #     print(f"  Mean: {self.similarity_map_pixel.mean():.3f}")

    #     # Detect beards using SimpleBlobDetector
    #     self._detect_beards_blob()

    def _compute_similarity(self):
        """Compute similarity map based on angular distance (AMC-Loss theory)"""
        if len(self.selected_embeddings) == 0:
            self.similarity_map = None
            self.similarity_map_pixel = None
            self.detected_blobs = None
            return

        print("Computing angular similarity map...")

        # 1. 選択された特徴量の平均
        avg_embedding = torch.stack(self.selected_embeddings).mean(dim=0)
        
        # 2. コサイン類似度の計算 (GPU)
        embeddings_gpu = self.embeddings.to(self.device)
        avg_embedding_gpu = avg_embedding.to(self.device)
        cosine_sim = torch.matmul(embeddings_gpu, avg_embedding_gpu) # [patch_h, patch_w]

        # 3. 角度空間への変換 (AMC-Lossの理論的整合性)
        # acosで角度θ(0〜π)を算出
        eps = 1e-6
        theta = torch.acos(torch.clamp(cosine_sim, -1.0 + eps, 1.0 - eps))

        # 4. 指数関数によるスコアリング
        # 学習したマージン(例: 0.5)を基準に、角度が離れると急激にスコアを下げる
        # 係数 4.0 は、可視化時の「赤色の広がり」を調整するための感度パラメータです
        margin_val = getattr(self, 'margin', 0.5) # 初期化時に保存していない場合は 0.5
        similarity_map = torch.exp(-4.0 * (theta / margin_val))

        # --- 以下、アップサンプリングと平滑化（既存のロジック） ---
        self.similarity_map = similarity_map.cpu().numpy()

        similarity_map_4d = similarity_map.unsqueeze(0).unsqueeze(0)
        similarity_map_upsampled = F.interpolate(
            similarity_map_4d,
            size=(self.display_h, self.display_w),
            mode='bilinear',
            align_corners=False
        )

        if self.gaussian_kernel is None:
            kernel_size = 17
            sigma = 4.0
            x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
            gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
            self.gaussian_kernel = gauss_2d.unsqueeze(0).unsqueeze(0)

        similarity_map_smoothed = F.conv2d(
            similarity_map_upsampled,
            self.gaussian_kernel,
            padding=self.gaussian_kernel.shape[-1] // 2
        )

        self.similarity_map_pixel = similarity_map_smoothed.squeeze().cpu().numpy()

        # 髭（Blob）の検出を実行
        self._detect_beards_blob()

    def _detect_beards_blob(self):
        """Detect beards using SimpleBlobDetector within skin regions

        Two-stage process:
        1. Create skin region mask from similarity map (similarity > threshold)
        2. Detect dark blobs (beards) in grayscale image within skin regions
        """
        if self.similarity_map_pixel is None or self.gray_image is None:
            self.detected_blobs = None
            return

        # Stage 1: Create skin region mask (high similarity regions)
        absolute_threshold = self.threshold / 100.0
        skin_mask = (self.similarity_map_pixel > absolute_threshold).astype(np.uint8)

        # Stage 2: Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()

        # Don't filter by color initially (we'll use all blobs)
        # Note: SimpleBlobDetector's blobColor parameter can be unreliable
        params.filterByColor = False

        # Filter by area
        params.filterByArea = True
        params.minArea = float(self.blob_min_area)
        params.maxArea = float(self.blob_max_area)

        # Filter by circularity (0.0 to 1.0)
        params.filterByCircularity = True
        params.minCircularity = self.blob_circularity / 100.0

        # Disable other filters
        params.filterByConvexity = False
        params.filterByInertia = False

        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs in grayscale image
        keypoints = detector.detect(self.gray_image)

        # Filter blobs: keep only those within skin region mask
        filtered_blobs = []
        blobs_outside = 0
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            # Check if blob center is within skin mask
            if 0 <= y < skin_mask.shape[0] and 0 <= x < skin_mask.shape[1]:
                if skin_mask[y, x] > 0:
                    filtered_blobs.append(kp)
                else:
                    blobs_outside += 1
            else:
                blobs_outside += 1

        self.detected_blobs = filtered_blobs

        print(f"\n=== BLOB DETECTION (SimpleBlobDetector) ===")
        print(f"Similarity range: [{self.similarity_map_pixel.min():.3f}, {self.similarity_map_pixel.max():.3f}]")
        print(f"Skin mask threshold: {self.threshold}% -> {absolute_threshold:.3f}")
        print(f"Skin region pixels: {np.sum(skin_mask)} / {skin_mask.size} ({100*np.sum(skin_mask)/skin_mask.size:.1f}%)")
        print(f"Blob parameters:")
        print(f"  Min area: {self.blob_min_area} pixels")
        print(f"  Max area: {self.blob_max_area} pixels")
        print(f"  Min circularity: {self.blob_circularity}%")
        print(f"Total blobs detected: {len(keypoints)}")
        print(f"Blobs within skin region: {len(filtered_blobs)}")
        print(f"Blobs outside skin region: {blobs_outside}")

        # Show first few blobs for debugging
        if len(filtered_blobs) > 0:
            print(f"First {min(5, len(filtered_blobs))} blobs in skin region:")
            for i, kp in enumerate(filtered_blobs[:5], 1):
                x, y = int(kp.pt[0]), int(kp.pt[1])
                intensity = self.gray_image[y, x]
                print(f"  {i}. pos=({kp.pt[0]:.1f}, {kp.pt[1]:.1f}), size={kp.size:.1f}, intensity={intensity}")
        print(f"==========================================\n")

    def reset(self):
        """Reset all selections"""
        self.selected_embeddings = []
        self.click_positions = []
        self.selected_rectangles = []
        self.similarity_map = None
        self.similarity_map_pixel = None
        self.detected_blobs = None
        self.rect_start = None
        self.rect_end = None
        self.drawing = False
        self._update_display()
        print("Reset")

    def _draw_buttons(self, canvas):
        """Draw control buttons"""
        # Mode button
        button_x = self.button_margin
        button_y = self.button_margin

        mode_color = (100, 150, 200) if self.mode == 'rectangle' else (100, 200, 100)
        mode_text = f"Mode: {self.mode.upper()}"

        cv2.rectangle(canvas, (button_x, button_y),
                     (button_x + self.button_width, button_y + self.button_height),
                     mode_color, -1)
        cv2.rectangle(canvas, (button_x, button_y),
                     (button_x + self.button_width, button_y + self.button_height),
                     (255, 255, 255), 2)

        text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = button_x + (self.button_width - text_size[0]) // 2
        text_y = button_y + (self.button_height + text_size[1]) // 2
        cv2.putText(canvas, mode_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Binary button
        button_x = self.button_margin + (self.button_width + self.button_margin)
        binary_color = (150, 100, 200) if self.binary_mode else (80, 80, 80)
        binary_text = f"Binary: {'ON' if self.binary_mode else 'OFF'}"

        cv2.rectangle(canvas, (button_x, button_y),
                     (button_x + self.button_width, button_y + self.button_height),
                     binary_color, -1)
        cv2.rectangle(canvas, (button_x, button_y),
                     (button_x + self.button_width, button_y + self.button_height),
                     (255, 255, 255), 2)

        text_size = cv2.getTextSize(binary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = button_x + (self.button_width - text_size[0]) // 2
        text_y = button_y + (self.button_height + text_size[1]) // 2
        cv2.putText(canvas, binary_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # def _create_heatmap(self, similarity, use_pixel_level=True):
    #     """Create heatmap visualization at pixel or patch resolution

    #     JET colormap: Red = High similarity, Blue/Purple = Low similarity
    #     Threshold is used for binary mode to show regions above similarity threshold

    #     Args:
    #         similarity: Similarity map (either patch-level or pixel-level)
    #         use_pixel_level: If True and pixel-level map exists, use it for visualization
    #     """
    #     # Use pixel-level similarity map if available and requested
    #     if use_pixel_level and self.similarity_map_pixel is not None:
    #         similarity_to_viz = self.similarity_map_pixel
    #     else:
    #         similarity_to_viz = similarity

    #     if self.binary_mode:
    #         # Binary visualization: show only values above absolute threshold
    #         # Convert threshold from 0-100 scale to 0.0-1.0 similarity scale
    #         absolute_threshold = self.threshold / 100.0

    #         # Create binary mask based on absolute threshold
    #         binary_mask = (similarity_to_viz > absolute_threshold).astype(np.uint8) * 255
    #         heatmap = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
    #     else:
    #         # Continuous heatmap: Blue (low) -> Green -> Yellow -> Red (high)
    #         # Normalize to 0-255 (relative to min-max range)
    #         normalized = ((similarity_to_viz - similarity_to_viz.min()) /
    #                      (similarity_to_viz.max() - similarity_to_viz.min() + 1e-8) * 255).astype(np.uint8)
    #         heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    #     return heatmap
    
    def _create_heatmap(self, similarity, use_pixel_level=True):
        similarity_to_viz = self.similarity_map_pixel if (use_pixel_level and self.similarity_map_pixel is not None) else similarity

        if self.binary_mode:
            # 0.0〜1.0 の絶対値でスライス
            absolute_threshold = self.threshold / 100.0
            mask = (similarity_to_viz > absolute_threshold).astype(np.uint8) * 255
            # 赤いマスクを作成
            heatmap = np.zeros((similarity_to_viz.shape[0], similarity_to_viz.shape[1], 3), dtype=np.uint8)
            heatmap[mask > 0] = [0, 0, 255] # BGRで赤
        else:
            # スコア(0〜1)をそのまま255倍してJET適用
            # 最小・最大で正規化しないことで、画面全体の赤かぶりを防ぎます
            viz_norm = np.clip(similarity_to_viz * 255, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(viz_norm, cv2.COLORMAP_JET)

        return heatmap

    def _update_display(self):
        """Update the display"""
        display = self.original_image.copy()

        if self.similarity_map is not None:
            # Create heatmap at pixel resolution (no resizing needed!)
            heatmap = self._create_heatmap(self.similarity_map, use_pixel_level=True)

            # If using pixel-level map, heatmap is already at display size
            if self.similarity_map_pixel is not None:
                # Pixel-level heatmap is already at correct resolution
                heatmap_display = heatmap
            else:
                # Fall back to resizing patch-level heatmap
                heatmap_display = cv2.resize(heatmap, (self.display_w, self.display_h),
                                            interpolation=cv2.INTER_LINEAR)

            # Blend with original image
            display = cv2.addWeighted(display, 0.5, heatmap_display, 0.5, 0)

        # Draw detected blobs (beards)
        if self.detected_blobs is not None and len(self.detected_blobs) > 0:
            for blob in self.detected_blobs:
                # blob is cv2.KeyPoint object
                pixel_x = int(blob.pt[0])
                pixel_y = int(blob.pt[1])
                blob_size = blob.size

                # Draw circle for detected blob
                cv2.circle(display, (pixel_x, pixel_y), 8, (0, 255, 0), 2)  # Green circle
                cv2.drawMarker(display, (pixel_x, pixel_y), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)  # Yellow cross

                # Optional: Draw blob size circle
                # cv2.circle(display, (pixel_x, pixel_y), int(blob_size / 2), (255, 0, 255), 1)

        # Draw count (always show, even if 0 blobs)
        if self.similarity_map_pixel is not None:
            blob_count = len(self.detected_blobs) if self.detected_blobs is not None else 0
            count_text = f"Beards: {blob_count}"
            # Draw with outline for better visibility
            cv2.putText(display, count_text, (10, self.display_h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(display, count_text, (10, self.display_h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        # Draw selected points
        for pos in self.click_positions:
            cv2.drawMarker(display, pos, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Draw selected rectangles
        for (p1, p2) in self.selected_rectangles:
            cv2.rectangle(display, p1, p2, (0, 255, 0), 2)

        # Draw current rectangle being drawn
        if self.drawing and self.rect_start and self.mouse_pos:
            cv2.rectangle(display, self.rect_start, self.mouse_pos, (0, 255, 255), 2)

        # Create canvas with button area
        button_area_height = self.button_height + self.button_margin * 2
        canvas = np.zeros((button_area_height + display.shape[0], display.shape[1], 3),
                         dtype=np.uint8)
        canvas[:] = (50, 50, 50)  # Dark gray background for button area

        # Draw buttons
        self._draw_buttons(canvas)

        # Place image below buttons
        canvas[button_area_height:, :] = display

        cv2.imshow('DINOv2 Inference Viewer', canvas)

    def run(self):
        """Main loop"""
        print("\n" + "="*80)
        print("DINOv2 Inference Viewer - PIXEL-LEVEL EDITION")
        print("="*80)
        print("\nFeatures:")
        print("  ✓ Pixel-level similarity maps (no more 14x14 patch limitation!)")
        print("  ✓ Bilinear upsampling + Gaussian smoothing (σ=4.0)")
        print("  ✓ GPU-accelerated with intelligent caching")
        print("  ✓ Based on AnomalyDINO methodology (WACV 2025)")
        print("\nControls:")
        print("  Mode button: Toggle Point/Rectangle mode")
        print("  Binary button: Toggle binary visualization")
        print("  Threshold slider: Skin region threshold (similarity)")
        print("  Min Area slider: Minimum blob area in pixels")
        print("  Max Area slider: Maximum blob area in pixels")
        print("  Circularity slider: Minimum circularity (0-100%)")
        print("  Left click: Select point/rectangle")
        print("  Right click: Reset")
        print("  'q' or ESC: Quit")
        print("\nDetection Method:")
        print("  - SimpleBlobDetector for beard counting")
        print("  - Stage 1: Create skin mask (RED regions = high similarity)")
        print("  - Stage 2: Detect dark blobs (beards) within skin regions")
        print("  - Threshold: Defines skin region boundary")
        print("  - Binary mode: Shows skin regions (red) vs non-skin (black)")
        print("  - Green circles + yellow crosses mark detected beards")
        print("  - Adjust Min/Max Area and Circularity for better detection")
        print("  - Enjoy smooth, high-resolution pixel-level similarity visualization!\n")

        self._update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='DINOv2 Inference Viewer')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('model_path', help='Path to trained model (.pth)')
    parser.add_argument('--model', default='dinov2_vitg14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                       help='DINOv2 model name (default: dinov2_vitg14)')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto-detect)')

    args = parser.parse_args()

    try:
        viewer = DINOv2InferenceViewer(
            image_path=args.image_path,
            model_path=args.model_path,
            model_name=args.model,
            device=args.device
        )
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
