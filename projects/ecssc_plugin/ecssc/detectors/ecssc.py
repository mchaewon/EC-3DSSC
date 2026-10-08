# ---------------------------------------------
# EC-3DSSC (ECSSC). Built on VoxFormer (https://github.com/NVlabs/VoxFormer),
# NVIDIA Source Code License-NC. See LICENSE.
# ---------------------------------------------
"""
ECSSC: event-camera assisted camera-based semantic scene completion.

Built on VoxFormer-T (stage 2). The RGB branch is unchanged; two event encoders are added:
  E_ego (EgoEventEncoderGN)    : SUM of event bins  -> F_ego, auxiliary depth head
  E_obj (MotionEventEncoderGN) : temporal DIFF      -> F_obj, auxiliary moving-object head
F_ego and F_obj are projected separately and passed to ECSSCHead, which runs three
independent 3D deformable cross-attentions (RGB / ego / obj) and fuses them with a
query-wise 3-way gate.

Options
  ablation      : None (full model) | 'ego_only' | 'obj_only'.
                  The disabled branch's encoder output is replaced by zeros before its projection
                  (same scheme as the earlier ablations) and its auxiliary loss is set to 0.
  lambda_ego / lambda_obj : auxiliary loss weights (0 disables the depth / moving-mask supervision).
  blur_severity / blur_angle : test-time motion blur of the RGB input (0 = off), see utils/blur.py.
  event_zero    : feed all-zero event voxels (test-time ablation of the event content).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.ecssc_plugin.ecssc.modules.event_encoders import EgoEventEncoderGN, MotionEventEncoderGN
from projects.ecssc_plugin.ecssc.utils.blur import motion_blur
from projects.ecssc_plugin.ecssc.utils.event_io import (
    load_event_voxel_batch, load_depth_batch, load_moving_mask_batch)

ABLATIONS = (None, 'ego_only', 'obj_only')


@DETECTORS.register_module()
class ECSSC(MVXTwoStageDetector):
    """pts_bbox_head must be ECSSCHead."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None, pts_voxel_encoder=None,
                 pts_middle_encoder=None, pts_fusion_layer=None,
                 img_backbone=None, pts_backbone=None,
                 img_neck=None, pts_neck=None, pts_bbox_head=None,
                 img_roi_head=None, img_rpn_head=None,
                 train_cfg=None, test_cfg=None, pretrained=None,
                 event_channels=128, num_event_bins=5, num_groups=8,
                 lambda_ego=0.1, lambda_obj=0.1,
                 img_H=370, img_W=1220, ev_H=352, ev_W=1216,
                 ablation=None,
                 blur_severity=0, blur_angle=0.0,
                 event_zero=False,
                 ):
        super().__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
            pts_fusion_layer, img_backbone, pts_backbone, img_neck,
            pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained
        )
        if ablation not in ABLATIONS:
            raise ValueError(f'ablation must be one of {ABLATIONS}, got {ablation!r}')

        self.ego_encoder = EgoEventEncoderGN(event_channels, num_groups)
        self.obj_encoder = MotionEventEncoderGN(num_event_bins, event_channels, num_groups)

        # separate projections for F_ego and F_obj (kept apart, fused in the head)
        self.ego_proj = nn.Sequential(
            nn.Conv2d(event_channels, event_channels, 1, bias=False),
            nn.GroupNorm(num_groups, event_channels),
            nn.ReLU(inplace=True),
        )
        self.obj_proj = nn.Sequential(
            nn.Conv2d(event_channels, event_channels, 1, bias=False),
            nn.GroupNorm(num_groups, event_channels),
            nn.ReLU(inplace=True),
        )

        self.lambda_ego = lambda_ego
        self.lambda_obj = lambda_obj
        self.T = num_event_bins
        self.img_H, self.img_W = img_H, img_W
        self.ev_H, self.ev_W = ev_H, ev_W
        self.ablation = ablation
        self.blur_severity = blur_severity
        self.blur_angle = blur_angle
        self.event_zero = event_zero
        if ablation:
            print(f'[ECSSC] ablation: {ablation}')

    def _get_event_feats(self, img_metas, fpn_shape, device):
        """Returns (ev, F_ego, F_obj, ego_feats, obj_feats); ego/obj feats at FPN resolution."""
        ev = load_event_voxel_batch(
            img_metas, self.T, self.ev_H, self.ev_W,
            self.img_H, self.img_W, device, event_zero=self.event_zero
        )
        F_ego = self.ego_encoder(ev)

        obj_out = self.obj_encoder(ev)
        F_obj = obj_out[0] if isinstance(obj_out, (tuple, list)) else obj_out

        # ablation: disable one branch by zeroing its encoder output
        if self.ablation == 'obj_only':
            F_ego = torch.zeros_like(F_ego)
        elif self.ablation == 'ego_only':
            F_obj = torch.zeros_like(F_obj)

        B, N, C, Hf, Wf = fpn_shape
        ego_feats = self.ego_proj(
            F.interpolate(F_ego, (Hf, Wf), mode='bilinear', align_corners=False)
        )  # (B, C, Hf, Wf)
        obj_feats = self.obj_proj(
            F.interpolate(F_obj, (Hf, Wf), mode='bilinear', align_corners=False)
        )  # (B, C, Hf, Wf)

        return ev, F_ego, F_obj, ego_feats, obj_feats

    def compute_aux_losses(self, F_ego, F_obj, img_metas, device):
        losses = {}
        depth_pred = self.ego_encoder.predict_depth(F_ego)
        if self.ablation == 'obj_only' or self.lambda_ego == 0:
            losses['loss_ego_depth'] = depth_pred.sum() * 0.
        else:
            depth_gt = load_depth_batch(img_metas, self.img_H, self.img_W, device=device)
            if depth_pred.shape != depth_gt.shape:
                depth_pred = F.interpolate(depth_pred, depth_gt.shape[2:],
                                           mode='bilinear', align_corners=False)
            vd = (depth_gt > 0).float()
            losses['loss_ego_depth'] = (
                F.l1_loss(depth_pred*vd, depth_gt*vd, reduction='sum')
                / (vd.sum()+1e-6)
            ) * self.lambda_ego if vd.sum() > 0 else depth_pred.sum()*0.

        dyn_pred = self.obj_encoder.predict_dynamic(F_obj)
        if self.ablation == 'ego_only' or self.lambda_obj == 0:
            losses['loss_obj_dyn'] = dyn_pred.sum() * 0.
        else:
            moving_gt = load_moving_mask_batch(img_metas, self.img_H, self.img_W, device=device)
            if dyn_pred.shape != moving_gt.shape:
                dyn_pred = F.interpolate(dyn_pred, moving_gt.shape[2:],
                                         mode='bilinear', align_corners=False)
            # pixels without a LiDAR point (-1) are masked out: input logit and target are both 0,
            # so they contribute a constant to the loss value but no gradient.
            vm = (moving_gt >= 0).float()
            losses['loss_obj_dyn'] = (
                F.binary_cross_entropy_with_logits(
                    dyn_pred*vm, (moving_gt >= 0.5).float()*vm, reduction='sum'
                ) / (vm.sum()+1e-6)
            ) * self.lambda_obj if vm.sum() > 0 else dyn_pred.sum()*0.
        return losses

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                B, N, C, H, W = img.size(); img = img.reshape(B*N, C, H, W)
            img = motion_blur(img, self.blur_severity, self.blur_angle)
            feats = self.img_backbone(img)
            if isinstance(feats, dict): feats = list(feats.values())
        else:
            return None
        if self.with_img_neck: feats = self.img_neck(feats)
        out = []
        for f in feats:
            BN, C, H, W = f.size()
            out.append(f.view(int(B/len_queue), len_queue, int(BN/B), C, H, W)
                       if len_queue else f.view(B, int(BN/B), C, H, W))
        return out

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        return self.extract_img_feat(img, img_metas, len_queue)

    def forward(self, return_loss=True, **kw):
        return self.forward_train(**kw) if return_loss else self.forward_test(**kw)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self, img_metas=None, img=None, target=None):
        lq    = img.size(1)
        metas = [each[lq-1] for each in img_metas]
        img   = img[:, -1, ...]
        dev   = img.device

        img_feats = self.extract_feat(img=img)
        _, F_ego, F_obj, ego_feats, obj_feats = self._get_event_feats(
            metas, img_feats[0].shape, dev
        )

        losses = {}
        outs = self.pts_bbox_head(
            img_feats, metas, target,
            ego_feats=ego_feats,
            obj_feats=obj_feats,
        )
        losses.update(self.pts_bbox_head.training_step(outs, target, metas))
        losses.update(self.compute_aux_losses(F_ego, F_obj, metas, dev))
        return losses

    def forward_test(self, img_metas=None, img=None, target=None, **kw):
        lq    = img.size(1)
        metas = [each[lq-1] for each in img_metas]
        img   = img[:, -1, ...]
        dev   = img.device

        img_feats = self.extract_feat(img=img)
        _, F_ego, F_obj, ego_feats, obj_feats = self._get_event_feats(
            metas, img_feats[0].shape, dev
        )
        outs = self.pts_bbox_head(
            img_feats, metas, target,
            ego_feats=ego_feats,
            obj_feats=obj_feats,
        )
        res = self.pts_bbox_head.validation_step(outs, target, metas)
        return [res] if isinstance(res, dict) else res
