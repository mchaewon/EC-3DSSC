"""
blur severity sweep 이미지: 한 프레임에 대해 행 = severity (맨 위 clean), 열 = [RGB 입력 | VoxFormer-T | Ours] (camera_raised).

새로 렌더링하지 않고 이미 만든 이미지를 조합만 한다.
  clean (severity 0) : RGB = image_2 crop [:370, :1220],   예측 = {clean_root}/{model}/images_camera_raised/{frame}.png
  severity s         : RGB = {blur_root}/severity_{s}/rgb_blurred/{frame}.png (모델에 실제 입력된 blur 이미지)
                       예측 = {blur_root}/severity_{s}/{model}/images_camera_raised/{frame}.png
모든 패널은 RGB 높이(370 px)에 맞춰 LANCZOS 로 축소 (render_ssc.compose_row 와 같은 방식). 출력이 이미 있으면 중단.

Usage (from the repository root):
  python tools/vis/severity_sweep.py --frames 001200 003385 --out outputs/qualitative_blur/severity_sweep
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_ssc as rs  # noqa: E402

MODELS = ['voxformer_t', 'ecssc']
KSIZE = {1: 7, 2: 15, 3: 25}


def to_u8(im, H):
    im8 = (np.clip(im[..., :3], 0, 1) * 255).round().astype(np.uint8) if im.dtype != np.uint8 else im[..., :3]
    if im8.shape[0] != H:
        w = int(round(im8.shape[1] * H / im8.shape[0]))
        im8 = np.asarray(Image.fromarray(im8).resize((w, H), Image.LANCZOS))
    return im8


def compose_sweep(frame, seq, rows, colors, path, label_w=150, title_h=40, legend_h=110, gap=8):
    """rows: [(row_label, [HxWx3 arrays])], 열 순서 = [RGB, baseline, ours]."""
    plt = rs.plt
    H = rows[0][1][0].shape[0]
    rows = [(lab, [to_u8(im, H) for im in ims]) for lab, ims in rows]
    widths = [im.shape[1] for im in rows[0][1]]
    Wt = label_w + sum(widths) + gap * (len(widths) - 1)
    Ht = title_h + len(rows) * H + (len(rows) - 1) * gap + legend_h
    fig = plt.figure(figsize=(Wt / rs.DPI, Ht / rs.DPI), dpi=rs.DPI, facecolor=rs.BG)
    titles = ['RGB input'] + [rs.DISPLAY_NAMES[m] for m in MODELS]
    x = label_w
    for t, w in zip(titles, widths):
        fig.text((x + w / 2) / Wt, 1 - (title_h / 2) / Ht, t, ha='center', va='center', fontsize=11, color='#1f1f1e')
        x += w + gap
    fig.text(0.003, 1 - (title_h / 2) / Ht, f'Seq {seq} / frame {frame}', ha='left', va='center', fontsize=9, color='#6b6a66')
    for r, (lab, ims) in enumerate(rows):
        y_top = title_h + r * (H + gap)
        y0 = (Ht - y_top - H) / Ht
        fig.text((label_w / 2) / Wt, y0 + (H / 2) / Ht, lab, ha='center', va='center', fontsize=11, color='#1f1f1e')
        x = label_w
        for im in ims:
            ax = fig.add_axes([x / Wt, y0, im.shape[1] / Wt, H / Ht]); ax.imshow(im, interpolation='none'); ax.axis('off')
            x += im.shape[1] + gap
    lax = fig.add_axes([0.01, 0.0, 0.98, legend_h / Ht]); lax.axis('off')
    per_row = 10
    for c in range(1, 20):
        row, col = divmod(c - 1, per_row)
        x0 = col / per_row; y0 = 0.62 - row * 0.42
        lax.add_patch(plt.Rectangle((x0, y0), 0.008, 0.28, color=colors[c], transform=lax.transAxes))
        lax.text(x0 + 0.011, y0 + 0.14, rs.CLASS_NAMES[c], va='center', fontsize=8, color='#1f1f1e', transform=lax.transAxes)
    fig.savefig(path, dpi=rs.DPI, facecolor=rs.BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', nargs='+', required=True)
    ap.add_argument('--seq', default='08')
    ap.add_argument('--clean-root', default='outputs/qualitative')
    ap.add_argument('--blur-root', default='outputs/qualitative_blur')
    ap.add_argument('--severities', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--out', required=True)
    ap.add_argument('--yaml', default='preprocess/label/semantic-kitti.yaml')
    a = ap.parse_args()
    os.chdir(rs.ROOT)
    _, colors = rs.load_maps(a.yaml)
    outs = {f: os.path.join(a.out, f + '.png') for f in a.frames}
    exist = [p for p in outs.values() if os.path.exists(p)]
    if exist:
        print(f'[sweep] ABORT: {len(exist)} output files already exist, e.g. {exist[:3]}'); sys.exit(3)
    srcs = {}
    for f in a.frames:
        srcs[(f, 0)] = [None] + [os.path.join(a.clean_root, m, 'images_camera_raised', f + '.png') for m in MODELS]
        for s in a.severities:
            d = os.path.join(a.blur_root, f'severity_{s}')
            srcs[(f, s)] = [os.path.join(d, 'rgb_blurred', f + '.png')] + \
                           [os.path.join(d, m, 'images_camera_raised', f + '.png') for m in MODELS]
    miss = [p for v in srcs.values() for p in v if p is not None and not os.path.exists(p)]
    if miss:
        print(f'[sweep] ABORT: {len(miss)} source images missing, e.g. {miss[:3]}'); sys.exit(1)
    os.makedirs(a.out, exist_ok=True)
    for f in a.frames:
        rows = []
        for s in [0] + a.severities:
            rgb_p, *pred_p = srcs[(f, s)]
            rgb = rs.load_rgb(a.seq, f) if rgb_p is None else rs.load_rgb(a.seq, f, os.path.dirname(rgb_p))
            lab = 'clean' if s == 0 else f'blur sev {s}\n(k={KSIZE[s]})'
            rows.append((lab, [rgb] + [rs.plt.imread(p) for p in pred_p]))
        compose_sweep(f, a.seq, rows, colors, outs[f])
        print(f'[sweep] {outs[f]}', flush=True)


if __name__ == '__main__':
    main()
