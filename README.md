# EC-3DSSC: Event-Camera assisted Camera-based 3D Semantic Scene Completion

This repository contains the code of **ECSSC**, a camera-based semantic scene completion model that adds an event-camera branch to [VoxFormer-T](https://github.com/NVlabs/VoxFormer). It also contains the evaluation tools used in the paper: test-time motion-blur robustness, event ablations, and qualitative visualization.

## Method overview

ECSSC keeps the VoxFormer-T pipeline: stage-1 depth-based query proposals, then stage-2 RGB deformable cross-attention and 3D self-attention. It adds an event branch on top.

- **E_ego (`EgoEventEncoderGN`).** Encodes the sum of the event voxel bins, which is a spatial event-density / ego-motion cue. During training, an auxiliary depth head is supervised with the stereo pseudo-depth target.
- **E_obj (`MotionEventEncoderGN`).** Encodes the temporal differences between bins plus their magnitude, which is an object-motion cue. During training, an auxiliary moving-object head is supervised with LiDAR-projected moving masks.
- **ECSSCHead.** Runs three separate 3D deformable cross-attentions on the RGB, F_ego and F_obj features for each query proposal. A query-wise 3-way gate fuses them:
  `seed = g_rgb * seed_rgb + g_ego * seed_ego + g_obj * seed_obj`.
- The auxiliary heads are used only for training. Inference outputs the SSC volume.

## Repository structure

```
EC-3DSSC/
├── projects/
│   ├── configs/
│   │   ├── ecssc/              # ECSSC (full model) and ablation configs
│   │   └── voxformer/          # VoxFormer-T baseline (stage 2) and query-proposal network (stage 1, qpn.py)
│   └── ecssc_plugin/           # mmdet3d plugin
│       ├── datasets/           # SemanticKITTI stage-1 / stage-2 datasets, SemanticKittiDatasetECSSC
│       └── ecssc/
│           ├── detectors/      # ecssc.py (ECSSC), voxformer.py (VoxFormer-T baseline), lmscnet.py (stage 1)
│           ├── dense_heads/    # ecssc_head.py (ECSSCHead), voxformer_head.py
│           ├── modules/        # VoxFormer transformer modules, event_encoders.py
│           └── utils/          # blur.py (test-time motion blur), event_io.py, SSC loss / metric
├── tools/
│   ├── train.py, test.py, dist_train.sh, dist_test.sh
│   ├── data/                   # moving-mask generation (precompute_moving_masks.py, _v2.py)
│   └── vis/                    # qualitative rendering, blur figures, event visualization
├── preprocess/                 # VoxFormer stage-1 preprocessing + event voxelization
├── deform_attn_3d/             # CUDA op for 3D deformable self-attention (build before use)
└── docs/                       # original VoxFormer install / data / getting-started notes
```

## Installation

The code was developed with Python 3.8, CUDA 11.1, PyTorch 1.9.1, mmcv-full 1.4.0, mmdet 2.14.0, mmsegmentation 0.14.1 and mmdet3d 0.17.1. The provided `Dockerfile` installs this environment.

```bash
# PyTorch / mmcv / mmdet (see Dockerfile for the exact commands)
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1 timm einops

# mmdetection3d v0.17.1
git clone https://github.com/open-mmlab/mmdetection3d.git && cd mmdetection3d && git checkout v0.17.1 && pip install -v -e . && cd ..

# 3D deformable attention op (imported from ./deform_attn_3d; override with DEFORM_ATTN_3D_PATH)
cd deform_attn_3d && python setup.py build_ext --inplace && cd ..
```

Run all commands from the repository root with the repository on `PYTHONPATH`.

## Data preparation

Place SemanticKITTI under `dataset/semantickitti/`. A symlink is fine. The expected layout is:

```
dataset/semantickitti/
├── sequences/{00..21}/            # calib.txt, poses.txt, image_2/, image_3/, velodyne/, voxels/
├── labels/{seq}/*_1_1.npy         # SSC targets (VoxFormer preprocessing)
├── sequences_msnet3d_sweep10/     # stage-1 query proposals (VoxFormer)
├── depth/sequences/{seq}/*.npy    # MobileStereoNet stereo depth (VoxFormer preprocessing); aux. depth target
├── lidarseg/{seq}/labels/*.label  # SemanticKITTI point labels (for moving masks)
├── events/{seq}/image_0/*.npy     # raw events (N, 4) [x, y, t, p], SemanticKITTI-E
├── events_voxel/{seq}/image_0/*.npy   # 5-bin event voxel grids (352 x 1216)
└── moving_masks/{seq}/*.npy       # moving-object masks (-1 no LiDAR point / 0 static / 1 moving)
```

1. **SemanticKITTI, SSC labels, query proposals and stereo depth.** Follow the VoxFormer instructions in `docs/prepare_dataset.md` and `preprocess/README.md`. The MobileStereoNet checkpoint is not included; download it as described there.
2. **Event voxels.** Convert the raw events to 5-bin voxel grids:
   ```bash
   python preprocess/preprocess_event_to_voxel.py --event_root dataset/semantickitti/events \
       --output_root dataset/semantickitti/events_voxel --workers 8
   ```
3. **Moving-object masks** (auxiliary target of E_obj). Project the LiDAR points with their SemanticKITTI labels into the image and mark moving classes (252–259, except 256):
   ```bash
   python tools/data/precompute_moving_masks.py --dataset_root dataset/semantickitti --workers 8      # sparse masks (used for training)
   python tools/data/precompute_moving_masks_v2.py --dataset_root dataset/semantickitti --workers 8   # dense variant -> moving_masks_v2/
   ```
   The training configs read `moving_masks/` (v1). v2 dilates the sparse masks and intersects them with the event density.
4. **ImageNet ResNet-50.** Download the weights to `ckpts/resnet50-19c8e357.pth`.

## Training

```bash
# ECSSC (full model), 1 GPU
python tools/train.py projects/configs/ecssc/ecssc.py
# multi-GPU
bash tools/dist_train.sh projects/configs/ecssc/ecssc.py 4

# ablations (same ECSSC architecture)
python tools/train.py projects/configs/ecssc/ecssc_ablation_ego_only.py   # E_obj disabled
python tools/train.py projects/configs/ecssc/ecssc_ablation_obj_only.py   # E_ego disabled
python tools/train.py projects/configs/ecssc/ecssc_ablation_no_depth.py   # no auxiliary depth loss (lambda_ego = 0)
python tools/train.py projects/configs/ecssc/ecssc_ablation_no_mask.py    # no auxiliary moving-object loss (lambda_obj = 0)
```

In the `ego_only` / `obj_only` ablations, the disabled branch's encoder output is replaced by zeros before its projection, and that branch's auxiliary loss is set to 0.

## Evaluation

```bash
# ECSSC
python tools/test.py projects/configs/ecssc/ecssc.py <ecssc_checkpoint.pth> --eval ssc
# VoxFormer-T baseline
python tools/test.py projects/configs/voxformer/voxformer-T_deform3D.py <voxformer-T_checkpoint.pth> --eval ssc
# save predictions as SemanticKITTI .label files
python tools/test.py projects/configs/ecssc/ecssc.py <ckpt> --eval ssc --save-pred outputs/ecssc/labels
```

Pretrained checkpoints are not included in this repository.

## Robustness to test-time motion blur

`tools/test.py` can blur the RGB input at test time. No retraining is involved.

```bash
python tools/test.py projects/configs/ecssc/ecssc.py <ckpt> --eval ssc --blur-severity 2
python tools/test.py projects/configs/voxformer/voxformer-T_deform3D.py <ckpt> --eval ssc --blur-severity 2
```

| `--blur-severity` | kernel | description |
|---|---|---|
| 0 | – | clean input (default) |
| 1 | 7 px | horizontal linear motion blur, uniform kernel 1/k |
| 2 | 15 px | 〃 |
| 3 | 25 px | 〃 |

- **Where it is applied.** In the detector, right before the image backbone (`projects/ecssc_plugin/ecssc/utils/blur.py`). It is applied to the 1220×370 crop of the current frame and of all temporal frames (−12, −9, −6, −3).
- **What it equals.** Blurring is done after ImageNet normalization with replicate padding and a kernel that sums to 1, so it is equivalent to blurring the RGB image. It is deterministic.
- **What it does not touch.** The event input and the pre-computed depth-based query proposals are not blurred.
- **Other options.**
  - `--blur-angle` sets the blur direction in degrees. The default 0 is horizontal.
  - `--event-zero` (ECSSC only) feeds all-zero event voxels. It removes the event content at test time without retraining.
- **Config overrides.** The same options can be set in a config or with `--cfg-options model.blur_severity=2 model.blur_angle=0 model.event_zero=True`.

## Visualization

Event visualization:

```bash
# one event frame: per-pixel ON - OFF count (raw events or sum of the voxel bins), native 1216x352, no axes
python tools/vis/visualize_event_voxel.py --seq 08 --frame 000355 --mode frame --source raw --out outputs/event_vis/08_000355_frame.png
# full figure: RGB, event frame, the 5 voxel bins, and an overlay aligned to the model input
python tools/vis/visualize_event_voxel.py --seq 08 --frame 000355 --out outputs/event_vis/08_000355.png
# SUM / DIFF decomposition (inputs of E_ego / E_obj)
python tools/vis/visualize_event_sum_diff.py --event_path dataset/semantickitti/events/08/image_0/000355.npy --out_dir outputs/event_vis
```

Qualitative SSC results, clean or blurred:

```bash
# 1) inference that saves uint8 npz predictions and the blurred RGB actually fed to the model
python tools/vis/infer_save.py projects/configs/ecssc/ecssc.py <ckpt> --blur-severity 1 \
    --labels-root outputs/blur_s1/ecssc/labels --rgb-out outputs/blur_s1/rgb_blurred
# 2) render voxels: oblique view (default) or a raised camera behind the RGB camera
python tools/vis/render_ssc.py --seq 08 --pred ecssc=outputs/blur_s1/ecssc/labels voxformer_t=outputs/blur_s1/voxformer_t/labels \
    --out outputs/blur_s1 --comparison
python tools/vis/render_ssc.py --seq 08 --pred ecssc=... voxformer_t=... --out outputs/blur_s1 --comparison --with-rgb \
    --rgb-dir outputs/blur_s1/rgb_blurred --rgb-title 'Blurred RGB (k=7)' \
    --view camera_raised --raise-h 7 --raise-back 8 --pitch 22 --hfov 90 --img-w 1220 --img-h 740
# 3) severity sweep: one image per frame, rows clean / k=7 / 15 / 25
python tools/vis/severity_sweep.py --frames 003385 --clean-root outputs/qualitative --blur-root outputs/qualitative_blur \
    --out outputs/qualitative_blur/severity_sweep
```

Metrics from saved predictions (`.label` or `.npz`) are computed with `tools/vis/eval_saved_labels.py`. Per-frame metrics, for picking frames only, are computed with `tools/vis/per_frame_metrics.py`.

## Acknowledgement and license

This code is built on [VoxFormer](https://github.com/NVlabs/VoxFormer). VoxFormer in turn builds on [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [LMSCNet](https://github.com/astra-vision/LMSCNet), [MobileStereoNet](https://github.com/cogsys-tuebingen/mobilestereonet) and [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).

The VoxFormer-derived code is released under the **NVIDIA Source Code License-NC** (see `LICENSE`), which permits non-commercial use only. The original copyright headers are kept in the derived files.
