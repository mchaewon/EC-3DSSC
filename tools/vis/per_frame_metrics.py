"""
프레임별 SSC 지표 CSV (대표 프레임 선택용 참고값, 공식 지표 아님).

  - 공식 지표(dataset.evaluate)는 815 프레임의 TP/FP/FN 을 먼저 합산한 뒤 IoU 를 계산한다.
    여기서는 같은 SSCMetrics 를 프레임마다 새로 만들어 한 프레임씩 계산한다. 평균은 공식 mIoU/IoU 가 아니다.
  - 행 순서 / dataset_index : test.py 가 처리하는 dataset 순서 (cfg.data.test 의 scans 순서, frame id 순서 아님).
  - {m}_IoU            : completion IoU (empty vs occupied), GT 255 제외
  - {m}_mIoU           : 공식 식을 한 프레임에 적용 (class 1~19 평균, GT 와 예측 모두에 없는 class 는 0)
  - {m}_mIoU_present   : GT 또는 예측에 있는 class 만 평균 (TP+FP+FN > 0, GT 255 제외 영역)
  - diff_*             : ours - baseline
  - {m}_IoU_{class}    : class IoU. GT 와 예측 모두에 없으면 빈 칸
  - 값은 0~1 비율, 소수 6자리.
예측은 .label / .npz 모두 읽는다 (tools/vis/pred_io.py).

Usage (from the repository root):
  python tools/vis/per_frame_metrics.py --baseline <labels_root> --ours <labels_root> --out <csv> [--compare-csv <csv>]
"""
import argparse
import csv
import os
import sys

import numpy as np
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pred_io  # noqa: E402

CLASS_NAMES = ["empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
               "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence",
               "vegetation", "trunk", "terrain", "pole", "traffic-sign"]
MODELS = ('baseline', 'ours')


def dataset_frames(config, seq):
    """cfg.data.test 의 scans 순서 그대로 frame id 목록 (이미지/GT 는 읽지 않음)."""
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.ecssc_plugin.datasets.builder  # noqa: F401  (plugin dataset 등록)
    cfg = Config.fromfile(config)
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    ds = build_dataset(cfg.data.test)
    out = []
    for s in ds.scans:
        if s['sequence'] != seq:
            raise ValueError(f'unexpected sequence {s["sequence"]}')
        out.append(os.path.splitext(os.path.basename(s['proposal_path']))[0])
    return out


def frame_metrics(y_pred, y_true):
    from projects.ecssc_plugin.ecssc.utils.ssc_metric import SSCMetrics
    m = SSCMetrics(len(CLASS_NAMES))
    m.add_batch(y_pred.reshape(1, 256, 256, 32), y_true.reshape(1, 256, 256, 32).copy())
    s = m.get_stats()
    present = (m.tps + m.fps + m.fns)[1:] > 0
    iou_c = s['iou_ssc'][1:]
    return dict(IoU=float(s['iou']), mIoU=float(s['iou_ssc_mean']),
                mIoU_present=float(iou_c[present].mean()) if present.any() else float('nan'),
                cls=[float(v) if p else None for v, p in zip(iou_c, present)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True, help='baseline labels root')
    p.add_argument('--ours', required=True, help='ours labels root')
    p.add_argument('--out', required=True)
    p.add_argument('--config', default='projects/configs/voxformer/voxformer-T_deform3D.py',
                   help='config whose data.test defines the dataset order')
    p.add_argument('--gt', default='dataset/semantickitti/labels')
    p.add_argument('--seq', default='08')
    p.add_argument('--yaml', default='preprocess/label/semantic-kitti.yaml')
    p.add_argument('--compare-csv', default=None, help='existing CSV to compare with (cell by cell)')
    a = p.parse_args()
    os.chdir(ROOT)
    if os.path.exists(a.out):
        print(f'[pfm] ABORT: output exists: {a.out}'); sys.exit(3)

    y = yaml.safe_load(open(a.yaml))
    lut = np.full(65536, -1, np.int64)
    for k, v in y['learning_map'].items():
        lut[k] = v
    frames = dataset_frames(a.config, a.seq)
    roots = dict(baseline=a.baseline, ours=a.ours)
    miss = {m: [f for f in frames if pred_io.pred_file(r, a.seq, f) is None] for m, r in roots.items()}
    if any(miss.values()):
        print(f'[pfm] ABORT: missing predictions: { {m: v[:3] for m, v in miss.items() if v} }'); sys.exit(1)

    cls = CLASS_NAMES[1:]
    header = (['frame', 'dataset_index'] + [f'{m}_{k}' for m in MODELS for k in ('IoU', 'mIoU')]
              + ['diff_IoU', 'diff_mIoU'] + [f'{m}_mIoU_present' for m in MODELS] + ['diff_mIoU_present']
              + [f'{m}_IoU_{c}' for m in MODELS for c in cls])
    fmt = lambda v: '' if v is None or v != v else f'{v:.6f}'
    rows = []
    for i, f in enumerate(frames):
        gt = np.load(os.path.join(a.gt, a.seq, f + '_1_1.npy'))
        r = {m: frame_metrics(pred_io.load_pred(roots[m], a.seq, f, lut), gt) for m in MODELS}
        row = [f, str(i)] + [fmt(r[m][k]) for m in MODELS for k in ('IoU', 'mIoU')]
        row += [fmt(r['ours'][k] - r['baseline'][k]) for k in ('IoU', 'mIoU')]
        row += [fmt(r[m]['mIoU_present']) for m in MODELS] + [fmt(r['ours']['mIoU_present'] - r['baseline']['mIoU_present'])]
        row += [fmt(v) for m in MODELS for v in r[m]['cls']]
        rows.append(row)
        if (i + 1) % 100 == 0 or i + 1 == len(frames):
            print(f'[pfm] {i + 1}/{len(frames)}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    print(f'[pfm] wrote {a.out}: {len(rows)} rows x {len(header)} columns')

    if a.compare_csv:
        ref = list(csv.reader(open(a.compare_csv)))
        new = [header] + rows
        same_shape = len(ref) == len(new) and all(len(x) == len(z) for x, z in zip(ref, new))
        ndiff = sum(c1 != c2 for r1, r2 in zip(ref, new) for c1, c2 in zip(r1, r2)) if same_shape else -1
        print(f'[pfm] compare with {a.compare_csv}: shape_match={same_shape} differing_cells={ndiff} '
              f'IDENTICAL={same_shape and ndiff == 0}')
        sys.exit(0 if same_shape and ndiff == 0 else 1)


if __name__ == '__main__':
    main()
