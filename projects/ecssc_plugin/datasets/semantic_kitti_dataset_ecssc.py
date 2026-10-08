# ---------------------------------------------
# EC-3DSSC (ECSSC). Built on VoxFormer (https://github.com/NVlabs/VoxFormer),
# NVIDIA Source Code License-NC. See LICENSE.
# ---------------------------------------------
"""
SemanticKittiDatasetECSSC: SemanticKITTI stage-2 dataset + event / auxiliary-supervision paths.

Adds to img_metas (see projects/ecssc_plugin/ecssc/utils/event_io.py for the loaders):
  event_path       : events/{seq}/image_0/{frame}.npy (raw events; the voxel grid is read from
                     the matching events_voxel/ path) or events/sequences/{seq}/{frame}.npy
  depth_path       : depth/sequences/{seq}/{frame}.npy   (stereo pseudo-depth, auxiliary target of E_ego)
  moving_mask_path : moving_masks/{seq}/{frame}.npy      (auxiliary target of E_obj)
Inputs (RGB + temporal frames) and SSC targets are the same as SemanticKittiDatasetStage2.
"""

import os, glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from torchvision import transforms

from .semantic_kitti_dataset_stage2 import SemanticKittiDatasetStage2


@DATASETS.register_module()
class SemanticKittiDatasetECSSC(SemanticKittiDatasetStage2):
    """
    ECSSC dataset.
    Adds event_path, depth_path and moving_mask_path to img_metas.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # event / auxiliary supervision roots
        data_root = self.data_root
        self.event_root       = os.path.join(data_root, "events", "sequences")
        self.depth_root       = os.path.join(data_root, "depth", "sequences")
        self.moving_mask_root = os.path.join(data_root, "moving_masks")

        # event voxel parameters
        self.num_event_bins = 5
        self.ev_H, self.ev_W = 352, 1216

        print(f"[ECSSC Dataset] event_root:       {self.event_root}")
        print(f"[ECSSC Dataset] depth_root:        {self.depth_root}")
        print(f"[ECSSC Dataset] moving_mask_root:  {self.moving_mask_root}")

    def get_meta_info(self, scan, sequence, frame_id, proposal_path):
        """Stage-2 meta info + event_path, depth_path, moving_mask_path."""
        meta_dict = super().get_meta_info(scan, sequence, frame_id, proposal_path)

        # event path
        # seq 00-03: events/sequences/{seq}/{frame}.npy
        # seq 04-10: events/{seq}/image_0/{frame}.npy
        ev_candidates = [
            os.path.join(self.event_root, sequence, f"{frame_id}.npy"),
            os.path.join(self.data_root, "events", sequence,
                         "image_0", f"{frame_id}.npy"),
        ]
        event_path = next((p for p in ev_candidates if os.path.exists(p)), None)

        # depth target path
        depth_path = os.path.join(
            self.depth_root, sequence, f"{frame_id}.npy"
        )
        if not os.path.exists(depth_path):
            depth_path = None

        # moving-object mask path
        moving_mask_path = os.path.join(
            self.moving_mask_root, sequence, f"{frame_id}.npy"
        )
        if not os.path.exists(moving_mask_path):
            moving_mask_path = None

        meta_dict.update(dict(
            event_path       = event_path,
            depth_path       = depth_path,
            moving_mask_path = moving_mask_path,
            sequence_id      = sequence,
            frame_id         = frame_id,
        ))
        return meta_dict