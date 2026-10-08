# ---------------------------------------------
# EC-3DSSC (ECSSC). Built on VoxFormer (https://github.com/NVlabs/VoxFormer),
# NVIDIA Source Code License-NC. See LICENSE.
# ---------------------------------------------
"""
Loading of event voxels and auxiliary supervision for ECSSC.

  load_event_voxel_batch : events_voxel/{seq}/image_0/{frame}.npy (5, 352, 1216), resized to the RGB crop (370, 1220).
                           Falls back to voxelizing raw events (events/{seq}/image_0/{frame}.npy, (N, 4) [x, y, t, p]).
                           event_zero=True returns all-zero voxels (ablation of the event content at test time).
  load_depth_batch       : depth/sequences/{seq}/{frame}.npy (stereo pseudo-depth target), cropped and clipped to 80 m.
  load_moving_mask_batch : moving_masks/{seq}/{frame}.npy (-1 = no LiDAR point, 0 = static, 1 = moving),
                           see tools/data/precompute_moving_masks.py.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F


def load_event_voxel_batch(img_metas, num_bins=5, ev_H=352, ev_W=1216,
                            img_H=370, img_W=1220, device='cpu', event_zero=False):
    if event_zero:
        return torch.zeros(len(img_metas), num_bins, img_H, img_W, device=device)
    batch = []
    for meta in img_metas:
        path  = meta.get('event_path', None)
        voxel = None
        if path:
            voxel_path = path.replace('/events/', '/events_voxel/')
            if os.path.exists(voxel_path):
                data = np.load(voxel_path)
                if data.ndim == 3:
                    voxel = data.astype(np.float32)
            if voxel is None and os.path.exists(path):
                events = np.load(path)
                if events.ndim == 3:
                    voxel = events.astype(np.float32)
                elif events.ndim == 2 and events.shape[1] == 4:
                    voxel = _raw_events_to_voxel(events, num_bins, ev_H, ev_W)
        if voxel is None:
            voxel = np.zeros((num_bins, ev_H, ev_W), dtype=np.float32)
        batch.append(voxel)
    ev = torch.from_numpy(np.stack(batch)).float().to(device)
    if ev.shape[2:] != (img_H, img_W):
        ev = F.interpolate(ev, (img_H, img_W), mode='bilinear', align_corners=False)
    return ev


def _raw_events_to_voxel(events, num_bins, ev_H, ev_W):
    voxel = np.zeros((num_bins, ev_H, ev_W), dtype=np.float32)
    x = events[:,0].astype(np.int32); y = events[:,1].astype(np.int32)
    t = events[:,2].astype(np.float64); p = events[:,3].astype(np.float32)
    ok = (x>=0)&(x<ev_W)&(y>=0)&(y<ev_H)
    x,y,t,p = x[ok],y[ok],t[ok],p[ok]
    if len(t) > 0:
        tn  = (t-t.min())/(t.max()-t.min()+1e-8)*(num_bins-1)
        pol = np.where(p>0,1.,-1.).astype(np.float32)
        tf  = tn.astype(np.int32); tc = tf+1
        wc  = (tn-tf).astype(np.float32); wf = 1.-wc
        mf  = tf<num_bins; mc = tc<num_bins
        if mf.sum(): np.add.at(voxel,(tf[mf],y[mf],x[mf]),pol[mf]*wf[mf])
        if mc.sum(): np.add.at(voxel,(tc[mc],y[mc],x[mc]),pol[mc]*wc[mc])
    return voxel


def load_depth_batch(img_metas, img_H=370, img_W=1220, max_depth=80., device='cpu'):
    batch = []
    for meta in img_metas:
        path = meta.get('depth_path', None)
        if path and os.path.exists(path):
            d = np.load(path).astype(np.float32)
            dh, dw = d.shape
            if dh >= img_H and dw >= img_W:
                d = d[:img_H, :img_W]
            d = np.clip(d, 0, max_depth)
        else:
            d = np.zeros((img_H, img_W), dtype=np.float32)
        batch.append(d)
    return torch.from_numpy(np.stack(batch)[:,np.newaxis]).float().to(device)


def load_moving_mask_batch(img_metas, img_H=370, img_W=1220, device='cpu'):
    batch = []
    for meta in img_metas:
        path = meta.get('moving_mask_path', None)
        if path and os.path.exists(path):
            m = np.load(path).astype(np.float32)
            if m.shape != (img_H, img_W):
                from PIL import Image as PILImage
                m_u = ((m+1)*64).astype(np.uint8)
                m   = np.array(
                    PILImage.fromarray(m_u).resize((img_W, img_H), PILImage.NEAREST)
                ).astype(np.float32)/64 - 1
        else:
            m = np.full((img_H, img_W), -1, dtype=np.float32)
        batch.append(m)
    return torch.from_numpy(np.stack(batch)[:,np.newaxis]).float().to(device)


# ============================================================
# Main Detector: VoxFormerExp3Diff
