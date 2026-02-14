"""Timed SAM Automatic Mask Generator.

This module provides a subclass of SamAutomaticMaskGenerator that measures
encoding and decoding times separately.
"""

import time
import numpy as np
from typing import Any, Dict, List

try:
    import torch
    from segment_anything import SamAutomaticMaskGenerator
    from segment_anything.utils.amg import (
        MaskData,
        batch_iterator,
        batched_mask_to_box,
        calculate_stability_score,
        is_box_near_crop_edge,
        mask_to_rle_pytorch,
        uncrop_masks,
    )
    from torchvision.ops import batched_nms
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    # Create dummy base class for type checking
    class SamAutomaticMaskGenerator:
        pass


class TimedSamAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    """SAM Automatic Mask Generator with separate encoding/decoding timing.

    This class extends SamAutomaticMaskGenerator to measure:
    - Image encoding time (image -> features via ViT)
    - Mask decoding time (features + prompts -> masks)

    Timing information is accumulated across all crops and batches within
    a single generate() call, and can be retrieved via get_timings().
    """

    def __init__(self, *args, **kwargs):
        """Initialize the timed mask generator."""
        if not SAM_AVAILABLE:
            raise ImportError("segment_anything library is not available")

        super().__init__(*args, **kwargs)
        self.reset_timings()

    def reset_timings(self):
        """Reset timing accumulators to zero."""
        self._encode_time = 0.0
        self._decode_time = 0.0
        self._quality_filter_time = 0.0
        self._binarize_time = 0.0
        self._edge_filter_time = 0.0
        self._nms_time = 0.0
        self._rle_time = 0.0
        self._total_time = 0.0

    def get_timings(self) -> Dict[str, float]:
        """Get accumulated timing information.

        Returns:
            Dictionary with encode, decode, nms, rle, postprocess, and total times
        """
        measured = (self._encode_time + self._decode_time
                    + self._quality_filter_time + self._binarize_time
                    + self._edge_filter_time + self._nms_time + self._rle_time)
        postprocess_time = self._total_time - measured
        return {
            'encode_time': self._encode_time,
            'decode_time': self._decode_time,
            'quality_filter_time': self._quality_filter_time,
            'binarize_time': self._binarize_time,
            'edge_filter_time': self._edge_filter_time,
            'nms_time': self._nms_time,
            'rle_time': self._rle_time,
            'postprocess_time': max(0.0, postprocess_time),
            'total_time': self._total_time,
        }

    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate masks with timing measurement.

        This method resets timings before calling the parent generate(),
        so timing information reflects only the current generate() call.

        Args:
            image: RGB image array (H, W, 3)

        Returns:
            List of mask dictionaries (same as SamAutomaticMaskGenerator)
        """
        self.reset_timings()
        t_start = time.perf_counter()
        result = super().generate(image)
        self._total_time = time.perf_counter() - t_start
        return result

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: tuple,
    ) -> "MaskData":
        """Process a single crop with encoding time measurement.

        This method overrides the parent to measure the time spent in
        predictor.set_image(), which performs image encoding via ViT.

        Args:
            image: Full input image
            crop_box: Crop coordinates [x0, y0, x1, y1]
            crop_layer_idx: Index of the point grid layer
            orig_size: Original image size (h, w)

        Returns:
            MaskData containing masks for this crop
        """
        # Extract crop
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]

        # MEASURE ENCODING TIME
        t_encode_start = time.perf_counter()
        self.predictor.set_image(cropped_im)
        self._encode_time += time.perf_counter() - t_encode_start

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches (decoding happens here)
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)

        self.predictor.reset_image()

        # MEASURE NMS TIME
        t_nms_start = time.perf_counter()
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            data["class_ids"],
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)
        self._nms_time += time.perf_counter() - t_nms_start

        # Apply cropping offset
        offset = torch.as_tensor([crop_box[0], crop_box[1], crop_box[0], crop_box[1]],
                                  device=data["boxes"].device, dtype=data["boxes"].dtype)
        data["boxes"] = data["boxes"] + offset
        data["crop_boxes"] = torch.zeros((len(data["boxes"]), 4), dtype=torch.int, device=data["boxes"].device)
        data["crop_boxes"][:] = torch.as_tensor(crop_box, device=data["boxes"].device, dtype=torch.int)

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: tuple,
        crop_box: List[int],
        orig_size: tuple,
    ) -> "MaskData":
        """Process a batch of points with decoding time measurement.

        This method overrides the parent to measure the time spent in
        predictor.predict_torch(), which performs mask decoding.

        Args:
            points: Point prompts array (N, 2)
            im_size: Cropped image size (h, w)
            crop_box: Crop coordinates [x0, y0, x1, y1]
            orig_size: Original image size (h, w)

        Returns:
            MaskData containing decoded masks for this batch
        """
        # Transform points for model input
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)

        # MEASURE DECODING TIME
        t_decode_start = time.perf_counter()
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )
        self._decode_time += time.perf_counter() - t_decode_start

        # Serialize predictions and filter by IoU
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=in_points[:, None, :].expand(-1, masks.shape[1], -1).flatten(0, 1),
        )
        del masks

        # MEASURE QUALITY FILTERING TIME
        t_qf_start = time.perf_counter()
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)
        self._quality_filter_time += time.perf_counter() - t_qf_start

        # MEASURE BINARIZATION TIME
        t_bin_start = time.perf_counter()
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])
        self._binarize_time += time.perf_counter() - t_bin_start

        # MEASURE EDGE FILTERING TIME
        t_edge_start = time.perf_counter()
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_size[1], orig_size[0]])
        if not torch.all(keep_mask):
            data.filter(keep_mask)
        self._edge_filter_time += time.perf_counter() - t_edge_start

        # MEASURE RLE ENCODING TIME
        t_rle_start = time.perf_counter()
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_size[0], orig_size[1])
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]
        self._rle_time += time.perf_counter() - t_rle_start

        # Store class IDs for NMS (all same class in this batch)
        data["class_ids"] = torch.zeros(len(data["boxes"]), dtype=torch.int, device=data["boxes"].device)

        return data


# For backwards compatibility when SAM is not available
if not SAM_AVAILABLE:
    TimedSamAutomaticMaskGenerator = None
