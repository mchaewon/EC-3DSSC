"""
저장된 예측 .label 로 SSC 지표를 다시 계산하고, test.py 가 출력한 지표와 비교한다.

  pred : {pred_root}/sequences/{seq}/predictions/{frame}.label   (uint16, 원본 SemanticKITTI id, head.save_pred 형식)
         -> semantic-kitti.yaml learning_map -> reshape(256,256,32)  (learning label 0~19)
         또는 같은 위치의 {frame}.npz (key 'pred', uint8 learning label 0~19)   (tools/vis/pred_io.py)
  gt   : {gt_root}/{seq}/{frame}_1_1.npy                            (dataset.get_gt_info 와 같은 파일, 255 = ignore)
  지표 : projects/.../utils/ssc_metric.SSCMetrics (dataset.evaluate 와 같은 클래스 / 같은 key 이름)

GT 프레임 전체에 대응하는 예측이 하나라도 없으면 실패로 끝낸다 (프레임을 건너뛰지 않음).

Usage (from the repository root):
  python tools/vis/eval_saved_labels.py --pred outputs/<model>/labels --seq 08 \
      [--compare-log <test.py stdout log>] [--frames 000000 000015 ...]
"""
import argparse
import ast
import glob
import json
import os
import re
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
PREFIX = 'ssc_SemanticKITTI'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True)
    p.add_argument('--gt', default='dataset/semantickitti/labels')
    p.add_argument('--seq', default='08')
    p.add_argument('--yaml', default='preprocess/label/semantic-kitti.yaml')
    p.add_argument('--frames', nargs='+', default=None, help='subset of frame ids (default: all GT frames)')
    p.add_argument('--compare-log', default=None, help='test.py log containing the printed metric dict')
    p.add_argument('--out-json', default=None)
    a = p.parse_args()
    os.chdir(ROOT)
    from projects.ecssc_plugin.ecssc.utils.ssc_metric import SSCMetrics

    y = yaml.safe_load(open(a.yaml))
    lut = np.full(65536, -1, np.int64)
    for k, v in y['learning_map'].items():
        lut[k] = v
    allowed = set(y['learning_map_inv'].values())

    gt_frames = sorted(os.path.basename(f)[:6] for f in glob.glob(os.path.join(a.gt, a.seq, '*_1_1.npy')))
    frames = a.frames if a.frames else gt_frames
    pred_dir = os.path.join(a.pred, 'sequences', a.seq, 'predictions')
    missing = [f for f in frames if pred_io.pred_file(a.pred, a.seq, f) is None]
    if missing:
        print(f'[eval] FAIL: {len(missing)} prediction files missing, e.g. {missing[:5]}')
        sys.exit(1)

    m = SSCMetrics(len(CLASS_NAMES))
    for f in frames:
        pf = pred_io.pred_file(a.pred, a.seq, f)
        if pf.endswith('.label'):
            raw = np.fromfile(pf, dtype=np.uint16)
            bad = set(np.unique(raw).tolist()) - allowed
            if raw.size != 256 * 256 * 32 or bad:
                print(f'[eval] FAIL: {f}: size={raw.size} unexpected ids={sorted(bad)}')
                sys.exit(1)
        y_pred = pred_io.load_pred(a.pred, a.seq, f, lut).reshape(1, 256, 256, 32)
        y_true = np.load(os.path.join(a.gt, a.seq, f + '_1_1.npy')).reshape(1, 256, 256, 32)
        m.add_batch(y_pred, y_true.copy())
    s = m.get_stats()
    detail = {f'{PREFIX}/SemIoU_{c}': float(s['iou_ssc'][i]) for i, c in enumerate(CLASS_NAMES)}
    detail.update({f'{PREFIX}/mIoU': float(s['iou_ssc_mean']), f'{PREFIX}/IoU': float(s['iou']),
                   f'{PREFIX}/Precision': float(s['precision']), f'{PREFIX}/Recall': float(s['recall'])})
    print(f'[eval] frames={len(frames)}  mIoU={100 * detail[PREFIX + "/mIoU"]:.4f}  IoU={100 * detail[PREFIX + "/IoU"]:.4f}')
    out = dict(pred=a.pred, seq=a.seq, n_frames=len(frames), metrics=detail)

    if a.compare_log:
        txt = open(a.compare_log, errors='replace').read()
        found = re.findall(r"\{'ssc_SemanticKITTI/SemIoU_empty'.*?\}", txt)
        if not found:
            print('[eval] FAIL: no metric dict in log'); sys.exit(1)
        ref = ast.literal_eval(found[-1])
        diffs = {k: abs(detail[k] - ref[k]) for k in ref}
        mx = max(diffs.values())
        out['compare'] = dict(log=a.compare_log, max_abs_diff=mx, identical=mx == 0.0, per_key=diffs)
        print(f'[eval] vs test.py log: max |diff| = {mx:.3e}  identical={mx == 0.0}')
    if a.out_json:
        with open(a.out_json, 'w') as fh:
            json.dump(out, fh, indent=2)
    print(json.dumps({k.split('/')[-1]: round(100 * v, 4) for k, v in detail.items()}))


if __name__ == '__main__':
    main()
