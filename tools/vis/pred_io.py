"""
예측 파일 읽기/쓰기 (render_ssc / eval_saved_labels / per_frame_metrics 공용).

  .label : {root}/sequences/{seq}/predictions/{frame}.label  uint16 원본 SemanticKITTI id (head.save_pred 형식)
                                                             -> learning_map -> 0~19
  .npz   : {root}/sequences/{seq}/predictions/{frame}.npz    key 'pred', uint8 (256,256,32) learning label 0~19
                                                             (np.savez_compressed, learning_map 적용하지 않음)
같은 프레임에 두 형식이 함께 있으면 어느 쪽을 읽을지 모호하므로 실패한다.
"""
import os

import numpy as np

GRID = (256, 256, 32)
N_CLASSES = 20


def pred_dir(root, seq):
    return os.path.join(root, 'sequences', seq, 'predictions')


def pred_file(root, seq, frame):
    """존재하는 예측 파일 경로 (.label 또는 .npz). 없으면 None, 둘 다 있으면 ValueError."""
    base = os.path.join(pred_dir(root, seq), frame)
    found = [base + ext for ext in ('.label', '.npz') if os.path.exists(base + ext)]
    if len(found) > 1:
        raise ValueError(f'both .label and .npz exist for {base}')
    return found[0] if found else None


def load_pred(root, seq, frame, lut):
    """-> (256,256,32) int64 learning label 0~19. lut: learning_map lookup table (65536,), 미정의 = -1."""
    p = pred_file(root, seq, frame)
    if p is None:
        raise FileNotFoundError(f'no prediction for {frame} under {pred_dir(root, seq)}')
    if p.endswith('.label'):
        raw = np.fromfile(p, dtype=np.uint16)
        if raw.size != np.prod(GRID):
            raise ValueError(f'{p}: invalid size {raw.size}')
        lab = lut[raw].reshape(GRID)
    else:
        with np.load(p) as z:
            arr = z['pred']
        if arr.dtype != np.uint8 or arr.shape != GRID:
            raise ValueError(f'{p}: expected uint8 {GRID}, got {arr.dtype} {arr.shape}')
        lab = arr.astype(np.int64)
    if lab.min() < 0 or lab.max() >= N_CLASSES:
        raise ValueError(f'{p}: labels out of range [{lab.min()}, {lab.max()}]')
    return lab


def save_pred_npz(root, seq, frame, y_pred):
    """y_pred: learning label 0~19 (원소 수 256*256*32). 이미 있으면 덮어쓰지 않고 실패. -> 저장 경로."""
    arr = np.asarray(y_pred).reshape(GRID)
    if arr.min() < 0 or arr.max() >= N_CLASSES:
        raise ValueError(f'{frame}: labels out of range [{arr.min()}, {arr.max()}]')
    d = pred_dir(root, seq)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, frame + '.npz')
    if os.path.exists(out) or os.path.exists(os.path.join(d, frame + '.label')):
        raise FileExistsError(f'refusing to overwrite existing prediction: {out}')
    np.savez_compressed(out, pred=arr.astype(np.uint8))
    return out
