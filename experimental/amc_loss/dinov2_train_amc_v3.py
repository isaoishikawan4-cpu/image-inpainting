#!/usr/bin/env python3
"""
DINOv2 Contrastive Learning Training Script

Train a classifier on top of frozen DINOv2 features using contrastive learning.

Usage:
    python dinov2_train_contrastive.py <annotation_files...> --output model.pth

Example:
    python dinov2_train_contrastive.py image1_annotations.json image2_annotations.json --output classifier.pth
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


class DINOv2FeatureExtractor:
    """Extract features using frozen DINOv2"""

    def __init__(self, model_name='dinov2_vitg14', device=None):
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
        print(f"Loading DINOv2 model: {model_name}...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_path: str):
        """Extract pixel-wise features from an image"""
        # Load and preprocess image
        pil_image = Image.open(image_path).convert('RGB')

        # Resize to multiple of 14
        orig_w, orig_h = pil_image.size
        new_w = max(14, (orig_w // 14) * 14)
        new_h = max(14, (orig_h // 14) * 14)
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)

        # Transform and extract features
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
            patch_features = features['x_norm_patchtokens']

        # Reshape to spatial grid
        patch_h = img_tensor.shape[2] // 14
        patch_w = img_tensor.shape[3] // 14
        feature_map = patch_features.reshape(1, patch_h, patch_w, -1)
        feature_map = feature_map.permute(0, 3, 1, 2)  # [1, feature_dim, h, w]

        # Interpolate to original image size
        feature_map = F.interpolate(
            feature_map,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )

        # Normalize features
        feature_map = feature_map.squeeze(0)  # [feature_dim, h, w]
        feature_map = F.normalize(feature_map, dim=0)

        return feature_map.cpu()


class AnnotationDataset(Dataset):
    """Dataset of annotated regions"""

    CLASSES = ['black_beard', 'white_beard', 'other']

    def __init__(self, annotation_files: List[str], feature_extractor: DINOv2FeatureExtractor):
        self.samples = []
        self.feature_extractor = feature_extractor

        # Load all annotations
        for ann_file in annotation_files:
            self._load_annotation_file(ann_file)

        print(f"Loaded {len(self.samples)} samples from {len(annotation_files)} files")

        # Count samples per class
        class_counts = {c: 0 for c in self.CLASSES}
        for sample in self.samples:
            class_counts[sample['class']] += 1
        print("Class distribution:", class_counts)

    def _load_annotation_file(self, annotation_file: str):
        """Load annotations from a JSON file"""
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        image_path = data['image_path']
        annotations = data['annotations']

        # Extract features once for this image
        print(f"Extracting features from {image_path}...")
        features = self.feature_extractor.extract_features(image_path)

        # Process each annotation
        for ann in annotations:
            class_name = ann['class']
            class_idx = self.CLASSES.index(class_name)

            if ann['type'] == 'point':
                x, y = ann['coords']['x'], ann['coords']['y']
                # Clamp to feature map bounds
                y = min(y, features.shape[1] - 1)
                x = min(x, features.shape[2] - 1)
                feature = features[:, y, x]  # [feature_dim]

                self.samples.append({
                    'feature': feature,
                    'class': class_name,
                    'class_idx': class_idx,
                    'image_path': image_path
                })

            elif ann['type'] == 'rectangle':
                x1, y1 = ann['coords']['x1'], ann['coords']['y1']
                x2, y2 = ann['coords']['x2'], ann['coords']['y2']

                # Clamp to feature map bounds
                y1 = min(y1, features.shape[1] - 1)
                y2 = min(y2, features.shape[1] - 1)
                x1 = min(x1, features.shape[2] - 1)
                x2 = min(x2, features.shape[2] - 1)

                # Average features in rectangle
                rect_features = features[:, y1:y2+1, x1:x2+1]
                feature = rect_features.mean(dim=(1, 2))  # [feature_dim]

                self.samples.append({
                    'feature': feature,
                    'class': class_name,
                    'class_idx': class_idx,
                    'image_path': image_path
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'feature': sample['feature'],
            'label': sample['class_idx'],
            'class_name': sample['class']
        }


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
        """
        Args:
            x: Input features [batch_size, input_dim]
            return_embedding: If True, return projection embeddings for contrastive loss

        Returns:
            logits or (logits, embeddings)
        """
        logits = self.mlp(x)

        if return_embedding:
            embeddings = self.projection(x)
            embeddings = F.normalize(embeddings, dim=1)
            return logits, embeddings

        return logits


class AngularMarginContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, scale=64.0):
        super().__init__()
        self.margin = margin  # ラジアン単位 (0.5 rad ≈ 28.6度)
        self.scale = scale
        self.eps = 1e-6

    def forward(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # 1. Cosine similarity & Angular distance
        cosine_sim = torch.matmul(embeddings, embeddings.T)
        cosine_sim = torch.clamp(cosine_sim, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(cosine_sim) # [0, pi]

        # 2. Mask creation
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()
        eye_mask = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - eye_mask # 自分自身を除外
        mask_neg = 1.0 - torch.eq(labels, labels.T).float()

        # 3. Compute Losses (AMC-Loss Logic)
        # 正例: 角度を0に近づける
        loss_pos = (theta ** 2) * mask_pos
        
        # 負例: マージンより近い(角度が小さい)場合にペナルティ
        # margin 0.5 rad 以内に敵クラスがいたらロスを出す
        loss_neg = (torch.clamp(self.margin - theta, min=0.0) ** 2) * mask_neg

        # 4. Average
        num_pos = mask_pos.sum() + self.eps
        num_neg = mask_neg.sum() + self.eps

        # 全体にscaleをかけてLossの大きさを調整
        loss = self.scale * ((loss_pos.sum() / num_pos) + (loss_neg.sum() / num_neg))

        return loss


def train_epoch(model, dataloader, optimizer, ce_criterion, contrastive_criterion, device, epoch, contrast_weight=2.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_contrast_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        features = batch['feature'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits, embeddings = model(features, return_embedding=True)

        # Cross-entropy loss
        ce_loss = ce_criterion(logits, labels)

        # Contrastive loss
        contrast_loss = contrastive_criterion(embeddings, labels)

        # Combined loss
        loss = ce_loss + contrast_weight * contrast_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_contrast_loss += contrast_loss.item()

        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'cont': f'{contrast_loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_contrast_loss = total_contrast_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, avg_ce_loss, avg_contrast_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train DINOv2 + MLP with contrastive learning')
    parser.add_argument('annotation_files', nargs='+', help='Annotation JSON files')
    parser.add_argument('--output', '-o', required=True, help='Output model file (.pth)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], help='Device to use (default: auto-detect)')
    parser.add_argument('--model', default='dinov2_vitg14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                       help='DINOv2 model name (default: dinov2_vitg14)')
    parser.add_argument('--margin', type=float, default=0.5, help='Angular margin for AMC loss (default: 0.5)')
    parser.add_argument('--scale', type=float, default=64.0, help='Feature scaling for AMC loss (default: 64.0)')
    parser.add_argument('--contrast-weight', type=float, default=2.0, help='Weight for contrastive loss (default: 2.0)')
    parser.add_argument('--projection-dim', type=int, default=128, help='Projection embedding dimension (default: 128)')

    args = parser.parse_args()

    # Validate annotation files
    for ann_file in args.annotation_files:
        if not Path(ann_file).exists():
            print(f"Error: Annotation file not found: {ann_file}")
            sys.exit(1)

    # Extract features
    feature_extractor = DINOv2FeatureExtractor(model_name=args.model, device=args.device)
    device = feature_extractor.device

    # Create dataset
    dataset = AnnotationDataset(args.annotation_files, feature_extractor)

    if len(dataset) == 0:
        print("Error: No samples found in annotation files")
        sys.exit(1)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Get feature dimension from first sample
    input_dim = dataset[0]['feature'].shape[0]
    print(f"Feature dimension: {input_dim}")

    # Create model
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=256,
        num_classes=3,
        projection_dim=args.projection_dim
    )
    model.to(device)

    # Loss and optimizer
    ce_criterion = nn.CrossEntropyLoss()
    contrastive_criterion = AngularMarginContrastiveLoss(margin=args.margin, scale=args.scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_ce_loss, avg_contrast_loss, accuracy = train_epoch(
            model, dataloader, optimizer, ce_criterion, contrastive_criterion,
            device, epoch, contrast_weight=args.contrast_weight
        )

        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs}: "
              f"Loss={avg_loss:.4f} (CE={avg_ce_loss:.4f}, Contrast={avg_contrast_loss:.4f}), "
              f"Accuracy={accuracy:.2f}%")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'input_dim': input_dim,
                'projection_dim': args.projection_dim,
                'classes': dataset.CLASSES
            }, args.output)
            print(f"Saved best model with accuracy {accuracy:.2f}%")

    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
