# ---------------------------------------------
# EC-3DSSC (ECSSC). Built on VoxFormer (https://github.com/NVlabs/VoxFormer),
# NVIDIA Source Code License-NC. See LICENSE.
# ---------------------------------------------
"""
Event encoders of ECSSC (event temporal decomposition).

  EgoEventEncoderGN    (E_ego): SUM over temporal bins -> spatial event density (ego-motion / depth cue).
                                Auxiliary head: dense depth prediction.
  MotionEventEncoderGN (E_obj): temporal DIFF between bins + magnitude -> object-motion cue.
                                Auxiliary head: moving-object segmentation.

Input event_voxel: (B, T=5, H, W) voxel grid (see utils/event_io.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EgoEventEncoderGN(nn.Module):
    """
    Spatial distribution of the event voxel.
    SUM over temporal bins -> ego-motion / depth cue.
    Auxiliary: depth prediction head
    """
    def __init__(self, out_channels=128, num_groups=8):
        super().__init__()
        self.out_channels = out_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(num_groups, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(num_groups, out_channels), nn.ReLU(inplace=True),
        )
        self.depth_head = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 64, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, 64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Softplus()
        )

    def forward(self, event_voxel):
        # SUM over time -> spatial event density (ego-motion proxy)
        spatial = event_voxel.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        return self.encoder(spatial)                      # (B, C, H/4, W/4)

    def predict_depth(self, feat):
        return self.depth_head(feat)


# ============================================================
# E_obj: temporal DIFF encoder
# ============================================================
class MotionEventEncoderGN(nn.Module):
    """
    Event temporal decomposition: DIFF-based object-motion encoder.

    DIFF:
      diffs = event_voxel[:, 1:] - event_voxel[:, :-1]
            = (B, T-1, H, W)  <- change between consecutive bins

      mag   = diffs.abs().sum(dim=1)
            = (B, 1, H, W)    <- overall motion magnitude

      input = cat([diffs, mag])
            = (B, T, H, W)    = (B, 5, H, W) for T=5

    Intuition:
      - background (ego-motion events on static structure):
        events shift consistently over bins 0->4 -> diffs follow the ego-motion pattern,
        roughly uniform over the image

      - dynamic objects (own motion):
        ego-motion + object velocity -> diffs deviate from the background pattern,
        strong DIFF responses only around the object boundaries

    Auxiliary: moving object segmentation head
    """
    def __init__(self, num_bins=5, out_channels=128, num_groups=8):
        super().__init__()
        self.out_channels = out_channels
        self.num_bins = num_bins
        in_channels = num_bins  # (T-1) diffs + 1 mag = T

        self.encoder = nn.Sequential(
            # input: (B, T, H/2, W/2) after spatial downsample
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(num_groups, out_channels), nn.ReLU(inplace=True),
        )  # (B, T, H/2, W/2) → (B, C, H/8, W/8) → upsample → (B, C, H/4, W/4)

        # upsample to the E_ego output resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1),
            nn.GroupNorm(num_groups, out_channels), nn.ReLU(inplace=True),
        )  # (B, C, H/8, W/8) → (B, C, H/4, W/4)

        # Auxiliary moving object head
        self.dynamic_head = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, event_voxel):
        """
        event_voxel: (B, num_bins, H, W)
        returns: feat (B, C, H/4, W/4)
        """
        B, T, H, W = event_voxel.shape

        # ── temporal DIFF ─────────────────────────────
        diffs = event_voxel[:, 1:] - event_voxel[:, :-1]  # (B, T-1, H, W)
        mag   = diffs.abs().sum(dim=1, keepdim=True)        # (B, 1,   H, W)
        motion = torch.cat([diffs, mag], dim=1)             # (B, T,   H, W)

        # ── spatial downsample (memory) ───────────────
        # (B, T, H, W) → (B, T, H/2, W/2)
        motion_down = F.avg_pool2d(
            motion.view(B * T, 1, H, W), kernel_size=2, stride=2
        ).view(B, T, H // 2, W // 2)

        # ── Encoding ───────────────────────────────────────
        feat = self.encoder(motion_down)    # (B, C, H/8, W/8)
        feat = self.upsample(feat)          # (B, C, H/4, W/4)
        return feat, mag, mag                                    # mag is also returned (unused by ECSSC)

    def predict_dynamic(self, feat):
        return self.dynamic_head(feat)      # (B, 1, H/2, W/2)


