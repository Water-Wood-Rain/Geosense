"""VGGT depth geometry encoder implementation."""

import torch

from .base import BaseGeometryEncoder, GeometryEncoderConfig


class VGGTDepthEncoder(BaseGeometryEncoder):
    """VGGT geometry encoder that re-encodes VGGT-predicted depth maps."""

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)

        from ..vggt.models.vggt import VGGT

        self.depth_predictor = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=True,
            enable_track=False,
        )
        self.depth_encoder = VGGT(
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )

        if self.freeze_encoder:
            for param in self.depth_predictor.parameters():
                param.requires_grad = False
            for param in self.depth_encoder.parameters():
                param.requires_grad = False

        self.reference_frame = config.reference_frame
        self.patch_size = 14
        self.depth_predictor.eval()
        self.depth_encoder.eval()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into depth-conditioned VGGT geometry features."""
        device = next(self.depth_encoder.parameters()).device
        images = self._apply_reference_frame_transform(images).to(device)
        use_cuda = device.type == "cuda"
        dtype = torch.bfloat16 if use_cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=use_cuda):
            depth_predictions = self.depth_predictor(images[None])
            depth = depth_predictions["depth"].permute(0, 1, 4, 2, 3).contiguous()
            depth = self._normalize_depth(depth)
            depth = depth.repeat(1, 1, 3, 1, 1)

            aggregated_tokens_list, patch_start_idx = self.depth_encoder.aggregator(depth)
            features = aggregated_tokens_list[-2][0, :, patch_start_idx:]

        features = self._apply_inverse_reference_frame_transform(features)
        return features

    def get_feature_dim(self) -> int:
        """Get VGGT feature dimension."""
        return 2048

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for compatibility."""
        return self.encode(images)

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Normalize per-frame depth to [0, 1] before re-encoding."""
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        bsz, num_frames, _, height, width = depth.shape
        depth = depth.view(bsz, num_frames, -1)
        depth_min = depth.amin(dim=-1, keepdim=True)
        depth_max = depth.amax(dim=-1, keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min).clamp_min(1e-6)
        return depth.view(bsz, num_frames, 1, height, width)

    def _apply_reference_frame_transform(self, images: torch.Tensor) -> torch.Tensor:
        """Apply reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(images, dims=(0,))
        return images

    def _apply_inverse_reference_frame_transform(self, features: torch.Tensor) -> torch.Tensor:
        """Apply inverse reference frame transformation if needed."""
        if self.reference_frame != "first":
            return torch.flip(features, dims=(0,))
        return features

    def load_model(self, model_path: str) -> None:
        """Load pretrained VGGT weights for both depth prediction and re-encoding."""
        from ..vggt.models.vggt import VGGT

        self.depth_predictor = VGGT.from_pretrained(
            model_path,
            enable_camera=False,
            enable_point=False,
            enable_depth=True,
            enable_track=False,
        )
        self.depth_encoder = VGGT.from_pretrained(
            model_path,
            enable_camera=False,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )

        if self.freeze_encoder:
            for param in self.depth_predictor.parameters():
                param.requires_grad = False
            for param in self.depth_encoder.parameters():
                param.requires_grad = False

        self.depth_predictor.eval()
        self.depth_encoder.eval()
