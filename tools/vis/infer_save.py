"""
Inference that saves what is needed for qualitative figures (clean or blurred input).

Builds the dataset / model the same way as tools/test.py, runs single_gpu_test and dataset.evaluate,
and additionally saves, per frame:
  - predictions : learning labels 0-19 as uint8 npz  {labels_root}/sequences/{seq}/predictions/{frame}.npz
                  (read back and checked against the prediction right after saving)
  - --rgb-out   : the current RGB frame exactly as fed to the image backbone (after blur), de-normalized
                  to a uint8 PNG (370 x 1220). Captured with a forward pre-hook on img_backbone.
  - --hash-out  : sha256 of the backbone input (5 frames / current frame) and of the prediction per frame,
                  e.g. to check that two models received identical inputs.
Blur and event options are passed to the model config (same as tools/test.py).

Usage (from the repository root):
  python tools/vis/infer_save.py projects/configs/ecssc/ecssc.py <ckpt> \
      --labels-root outputs/blur_s1/ecssc/labels --blur-severity 1 \
      [--rgb-out outputs/blur_s1/rgb_blurred] [--hash-out outputs/blur_s1/ecssc_hashes.json] [--event-zero] [--indices 0 3 407]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pred_io  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--labels-root', required=True)
    p.add_argument('--blur-severity', type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument('--blur-angle', type=float, default=0.0)
    p.add_argument('--event-zero', action='store_true')
    p.add_argument('--rgb-out', default=None)
    p.add_argument('--hash-out', default=None)
    p.add_argument('--indices', type=int, nargs='+', default=None, help='dataset indices (default: all)')
    p.add_argument('--metrics-out', default=None)
    return p.parse_args()


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def to_uint8_rgb(x_cur):
    """(3,H,W) float32 in normalized space -> (H,W,3) uint8."""
    return (np.clip(x_cur * STD + MEAN, 0.0, 1.0) * 255.0).round().astype(np.uint8).transpose(1, 2, 0)


def main():
    a = parse_args()
    os.chdir(ROOT)
    pdir = pred_io.pred_dir(a.labels_root, '08')
    for d in (pdir, a.rgb_out):
        if d and os.path.exists(d) and any(os.scandir(d)):
            sys.exit(f'output dir not empty: {d}')
    if a.rgb_out:
        os.makedirs(a.rgb_out, exist_ok=True)

    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.apis import set_random_seed
    from mmdet3d.apis import single_gpu_test
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from PIL import Image
    import importlib

    cfg = Config.fromfile(a.config)
    importlib.import_module('projects.ecssc_plugin')
    from projects.ecssc_plugin.datasets.builder import build_dataloader
    cfg.model.blur_severity = a.blur_severity
    cfg.model.blur_angle = a.blur_angle
    if a.event_zero:
        cfg.model.event_zero = True
    print(f'[infer] blur_severity={a.blur_severity} blur_angle={a.blur_angle} event_zero={a.event_zero}', flush=True)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    set_random_seed(0, deterministic=False)

    dataset = build_dataset(cfg.data.test)
    data = dataset if a.indices is None else torch.utils.data.Subset(dataset, a.indices)
    loader = build_dataloader(data, samples_per_gpu=samples_per_gpu, workers_per_gpu=cfg.data.workers_per_gpu,
                              dist=False, shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    det, head = model.module, model.module.pts_bbox_head

    captured, records = [], []
    det.img_backbone.register_forward_pre_hook(lambda mod, inp: captured.append(inp[0].detach()))
    orig_vs = head.validation_step

    def validation_step(out_dict, target, img_metas):
        res = orig_vs(out_dict, target, img_metas)
        x = captured.pop().float().cpu().numpy()        # (5, 3, 370, 1220): current + temporal frames
        captured.clear()
        meta = img_metas[0]
        fid, seq = meta['frame_id'], meta['sequence_id']
        y = np.asarray(res['y_pred'])
        path = pred_io.save_pred_npz(a.labels_root, seq, fid, y)
        with np.load(path) as z:
            if not np.array_equal(z['pred'].astype(np.int64), y.reshape(pred_io.GRID).astype(np.int64)):
                raise RuntimeError(f'npz roundtrip mismatch: {path}')
        if a.rgb_out:
            Image.fromarray(to_uint8_rgb(x[0])).save(os.path.join(a.rgb_out, fid + '.png'))
        records.append(dict(frame=fid, input_sha256=sha(x), input_cur_sha256=sha(x[0]),
                            pred_sha256=sha(y.reshape(pred_io.GRID).astype(np.uint8))))
        return res

    head.validation_step = validation_step
    t0 = time.time()
    outputs = single_gpu_test(model, loader)
    print(f'\n[infer] {len(outputs)} frames, {time.time() - t0:.1f}s', flush=True)

    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(metric=['ssc'])
    detail = dataset.evaluate(outputs, **eval_kwargs)
    print(detail, flush=True)
    print(f"[infer] mIoU={100 * float(detail['ssc_SemanticKITTI/mIoU']):.4f} "
          f"IoU={100 * float(detail['ssc_SemanticKITTI/IoU']):.4f}", flush=True)
    if a.hash_out:
        with open(a.hash_out, 'w') as fh:
            json.dump(dict(config=a.config, checkpoint=a.checkpoint, blur_severity=a.blur_severity,
                           blur_angle=a.blur_angle, event_zero=a.event_zero, frames=records), fh, indent=1)
    if a.metrics_out:
        with open(a.metrics_out, 'w') as fh:
            json.dump({k: float(v) for k, v in detail.items()}, fh, indent=1)


if __name__ == '__main__':
    main()
