"""
Event voxel 시각화 (임의 프레임). RGB + raw event frame + 5 voxel bins + model-aligned overlay.
blur 관련 패널만 제외.

Usage (from the repository root):
    # full figure (panels a-d below)
    python tools/vis/visualize_event_voxel.py --seq 08 --frame 000355 --out outputs/event_vis/event_08_000355.png
    # a single event frame only (per-pixel ON - OFF count, native 1216x352, no title / axes)
    python tools/vis/visualize_event_voxel.py --seq 08 --frame 000355 --mode frame --source raw \
        --out outputs/event_vis/event_08_000355_frame.png

패널
  (a) RGB (image_2, 370x1220 crop)
  (b) event frame: raw events 의 픽셀별 ON-OFF 개수 (native 352x1216)
  (c) voxel bin 0-4: 모델 입력 (events_voxel/{seq}/image_0/{frame}.npy, 5 temporal bins)
  (d) 모델 정렬 기준 overlay: voxel 합을 370x1220 bilinear resize (load_event_voxel_batch 와 동일) 해서 RGB 위에 겹침
출력 파일이 이미 있으면 중단한다.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
IMG_H, IMG_W = 370, 1220      # dataset crop
EV_H, EV_W = 352, 1216        # event sensor resolution

# diverging: blue(OFF, -) <-> neutral gray <-> red(ON, +)
SURFACE, INK, INK_MUTED = '#fcfcfb', '#1f1f1e', '#6b6a66'
POL_CMAP = LinearSegmentedColormap.from_list('pol', ['#2a78d6', '#f0efec', '#e34948'])


def sym_limit(a, q=99.0):
    nz = np.abs(a[a != 0])
    return float(np.percentile(nz, q)) if nz.size else 1.0


def overlay(ax, rgb01, ev, lim, title):
    gray = rgb01 @ np.array([0.299, 0.587, 0.114])
    ax.imshow(gray, cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
    rgba = POL_CMAP(TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)(np.clip(ev, -lim, lim)))
    rgba[..., 3] = np.clip(np.abs(ev) / lim, 0, 1) ** 0.6 * 0.9
    ax.imshow(rgba, interpolation='bilinear')
    ax.set_title(title, color=INK, fontsize=10)
    ax.axis('off')


def save_event_frame(path, source, out, clip_pct=99.0):
    """One event image: per-pixel ON - OFF count (raw) or sum of the voxel bins (voxel), blue = OFF, red = ON."""
    if source == 'raw':
        e = np.load(path)                                   # (N, 4) [x, y, t(ns), p(+-1)]
        x, y, p = e[:, 0].astype(int), e[:, 1].astype(int), e[:, 3]
        idx = y * EV_W + x
        img = (np.bincount(idx[p > 0], minlength=EV_H * EV_W)
               - np.bincount(idx[p < 0], minlength=EV_H * EV_W)).reshape(EV_H, EV_W).astype(np.float32)
    else:
        img = np.load(path).astype(np.float32).sum(0)       # (5, 352, 1216) -> (352, 1216)
    lim = sym_limit(img, clip_pct)
    plt.imsave(out, POL_CMAP(TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)(np.clip(img, -lim, lim))))
    print(f'saved {out} ({img.shape[1]}x{img.shape[0]}, clip +-{lim:.2f}, source={source})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='08')
    ap.add_argument('--frame', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', choices=['panels', 'frame'], default='panels',
                    help='panels: RGB + event frame + voxel bins + overlay; frame: one event image only')
    ap.add_argument('--source', choices=['raw', 'voxel'], default='raw',
                    help='(--mode frame) raw events (events/) or sum of the voxel bins (events_voxel/)')
    ap.add_argument('--clip-pct', type=float, default=99.0, help='symmetric color clipping percentile')
    a = ap.parse_args()
    SEQ, FRAME = a.seq, a.frame
    DATA = os.path.join(ROOT, 'dataset/semantickitti')
    RGB_PATH = f'{DATA}/sequences/{SEQ}/image_2/{FRAME}.png'
    RAW_PATH = f'{DATA}/events/{SEQ}/image_0/{FRAME}.npy'
    VOX_PATH = f'{DATA}/events_voxel/{SEQ}/image_0/{FRAME}.npy'
    TIMES = f'{DATA}/sequences/{SEQ}/times.txt'
    OUT = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    if os.path.exists(OUT):
        print(f'ABORT: output exists: {OUT}'); sys.exit(3)
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    if a.mode == 'frame':
        save_event_frame(RAW_PATH if a.source == 'raw' else VOX_PATH, a.source, OUT, a.clip_pct)
        return
    rgb = np.asarray(Image.open(RGB_PATH).convert('RGB'), np.float32)[:IMG_H, :IMG_W] / 255.
    raw = np.load(RAW_PATH)                        # (N, 4) [x, y, t(ns), p(+-1)]
    vox = np.load(VOX_PATH).astype(np.float32)     # (5, 352, 1216)
    times = np.loadtxt(TIMES)
    t_rgb = times[int(FRAME)]

    x, y, t, p = raw[:, 0], raw[:, 1], raw[:, 2] / 1e9, raw[:, 3]
    t0, t1 = t.min(), t.max()
    idx = y * EV_W + x
    n_on = np.bincount(idx[p > 0], minlength=EV_H * EV_W).reshape(EV_H, EV_W)
    n_off = np.bincount(idx[p < 0], minlength=EV_H * EV_W).reshape(EV_H, EV_W)
    net = (n_on - n_off).astype(np.float32)

    # 모델이 보는 정렬: (5,352,1216) -> bilinear -> (5,370,1220)
    vox_model = F.interpolate(torch.from_numpy(vox)[None], (IMG_H, IMG_W),
                              mode='bilinear', align_corners=False)[0].numpy()
    ev_sum_model = vox_model.sum(0)
    span_ms = 1e3 * (t1 - t0)

    # ── layout (inch 단위 수동 배치: 이미지 종횡비 유지, 여백 최소화) ──
    FW, M, G, TT = 12.0, 0.25, 0.15, 0.30          # fig width, margin, column gap, title space
    wf = FW - 2 * M
    wh = (wf - G) / 2
    h_rgb_f, h_ev_f = wf * IMG_H / IMG_W, wf * EV_H / EV_W
    h_ev_h, h_rgb_h = wh * EV_H / EV_W, wh * IMG_H / IMG_W
    heights = [0.95, TT + h_rgb_f, 0.2, TT + h_ev_f, 0.62] + [TT + h_ev_h] * 3 + [0.2, TT + h_rgb_h, 0.2]
    FH = sum(heights)
    fig = plt.figure(figsize=(FW, FH), facecolor=SURFACE)

    def ax_at(x, top, w, h):
        return fig.add_axes([x / FW, 1 - (top + h) / FH, w / FW, h / FH])

    fig.text(0.5, 1 - 0.32 / FH, f'SemanticKITTI seq {SEQ} / {FRAME}  -  RGB and event input',
             ha='center', va='center', color=INK, fontsize=13, fontweight='bold')
    fig.text(0.5, 1 - 0.68 / FH,
             f'event window {span_ms:.0f} ms  ({t0:.3f}-{t1:.3f} s),  RGB timestamp {t_rgb:.3f} s '
             f'(+{1e3 * (t_rgb - t0):.0f} ms),  {len(raw):,} events  (ON {int((p > 0).sum()):,} / OFF {int((p < 0).sum()):,})',
             ha='center', va='center', color=INK_MUTED, fontsize=9)
    y = heights[0]

    ax = ax_at(M, y + TT, wf, h_rgb_f)
    ax.imshow(rgb)
    ax.set_title('(a) RGB  image_2  (370x1220)', color=INK, fontsize=10)
    ax.axis('off')
    y += heights[1] + heights[2]

    lim_net = sym_limit(net)
    ax = ax_at(M, y + TT, wf, h_ev_f)
    im = ax.imshow(net, cmap=POL_CMAP, norm=TwoSlopeNorm(vmin=-lim_net, vcenter=0, vmax=lim_net),
                   interpolation='nearest')
    ax.set_title('(b) Event frame  raw events, per-pixel ON - OFF count  (image_0, native 352x1216)',
                 color=INK, fontsize=10)
    ax.axis('off')
    y += heights[3]
    cb = fig.colorbar(im, cax=ax_at(M + wf * 0.3, y + 0.08, wf * 0.4, 0.12), orientation='horizontal')
    cb.set_label('ON - OFF count per pixel  (blue = OFF,  red = ON,  clipped at 99th pct)', color=INK_MUTED, fontsize=8)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=7)
    y += heights[4]

    lim_vox = sym_limit(vox)
    t_rgb_ms = 1e3 * (t_rgb - t0)
    for k in range(vox.shape[0]):
        r, c = divmod(k, 2)
        ax = ax_at(M + c * (wh + G), y + r * heights[5] + TT, wh, h_ev_h)
        im_b = ax.imshow(vox[k], cmap=POL_CMAP, norm=TwoSlopeNorm(vmin=-lim_vox, vcenter=0, vmax=lim_vox),
                         interpolation='nearest')
        tag = '  = RGB time' if abs(k * span_ms / 4 - t_rgb_ms) < span_ms / 8 else ''
        ax.set_title(f'(c) voxel bin {k}  (centre +{k * span_ms / 4:.0f} ms){tag}', color=INK, fontsize=9.5)
        ax.axis('off')
    # 6번째 칸: voxel colorbar + 설명
    top6 = y + 2 * heights[5] + TT
    cb2 = fig.colorbar(im_b, cax=ax_at(M + wh + G + 0.3, top6 + 0.35, wh - 0.6, 0.12), orientation='horizontal')
    cb2.ax.tick_params(colors=INK_MUTED, labelsize=7)
    fig.text((M + wh + G + wh / 2) / FW, 1 - (top6 + 0.95) / FH,
             'voxel value (model input, events_voxel/*.npy)\n'
             'blue = OFF / negative,  red = ON / positive\n'
             'shared scale across bins, clipped at 99th pct\n'
             'bins use linear temporal interpolation (5 bins)',
             ha='center', va='top', color=INK_MUTED, fontsize=8.5, linespacing=1.5)
    y += 3 * heights[5] + heights[8]

    lim_m = sym_limit(ev_sum_model)
    overlay(ax_at(M, y + TT, wh, h_rgb_h), rgb, ev_sum_model, lim_m,
            '(d) RGB + event (voxel sum, model alignment)')
    ax = ax_at(M + wh + G, y + TT, wh, h_rgb_h)
    ax.imshow(ev_sum_model, cmap=POL_CMAP, norm=TwoSlopeNorm(vmin=-lim_m, vcenter=0, vmax=lim_m), interpolation='nearest')
    ax.set_title('(d) voxel sum, resized to 370x1220 (model input alignment)', color=INK, fontsize=10)
    ax.axis('off')

    # ── RGB(image_2) vs event(model 정렬) 전역 오프셋 측정 ──
    gray = rgb @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    evm = np.abs(vox_model).sum(0)
    R = 20
    a = grad[R:-R, R:-R]
    a = (a - a.mean()) / a.std()
    best, ncc0 = (-1, 0, 0), None
    for dy in range(-R // 2, R // 2 + 1):
        for dx in range(-R, R + 1):
            b = evm[R + dy:IMG_H - R + dy, R + dx:IMG_W - R + dx]
            v = float((a * (b - b.mean()) / b.std()).mean())
            if dx == 0 and dy == 0:
                ncc0 = v
            if v > best[0]:
                best = (v, dx, dy)
    print(f'alignment NCC(|grad RGB|, |event|): shift(0,0)={ncc0:.3f}, best={best[0]:.3f} at dx={best[1]} dy={best[2]} px')

    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print('saved', OUT)
    print(f'window {t0:.6f}-{t1:.6f} s, rgb {t_rgb:.6f} s, N={len(raw)}, '
          f'net clip {lim_net:.1f}, vox clip {lim_vox:.3f}, model-sum clip {lim_m:.3f}')


if __name__ == '__main__':
    main()
