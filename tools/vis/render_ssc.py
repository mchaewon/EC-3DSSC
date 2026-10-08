"""
SSC voxel 렌더러 (numpy + matplotlib, headless).  GT / 예측을 같은 카메라·해상도·colormap 으로 그린다.

입력
  GT   : {gt}/{seq}/{frame}_1_1.npy           learning label 0~19, 255 = invalid (평가 제외)
  pred : {pred}/sequences/{seq}/predictions/{frame}.label   uint16 원본 id -> learning_map -> 0~19
         또는 같은 위치의 {frame}.npz (key 'pred', uint8 learning label 0~19)   (tools/vis/pred_io.py)
  --gt-images DIR : GT 를 다시 그리지 않고 DIR/{frame}.png (기존 GT 렌더) 를 comparison 의 GT 패널로 사용
  --rgb-dir DIR   : (--with-rgb) RGB 패널을 image_2 대신 DIR/{frame}.png (370 x 1220) 로, 제목은 --rgb-title
렌더링 규칙
  - empty(0) 는 그리지 않는다. GT 의 255 도 그리지 않는다.
  - 예측: 기본은 예측 전체를 그린다 (no-mask).  --mask-gt-invalid 이면 GT == 255 인 voxel 을 예측에서도 숨긴다
    (평가 영역과 동일, 출력 폴더 images_mask/, comparison_mask/).
  - colormap: preprocess/label/semantic-kitti.yaml color_map[learning_map_inv[c]] (BGR -> RGB)
  - 카메라: 정사영(orthographic), 고정 azimuth/elevation, 고정 축 범위 -> 모든 이미지가 같은 시점·해상도.
    voxel 의 드러난 면(이웃이 비어 있는 면)만 PolyCollection 으로 그리고, 면 중심 깊이 순(먼 것부터)으로 칠한다.
    면 방향별 음영: top 1.0 / side 0.80 / back 0.65.  --draw-bbox 이면 장면 경계 상자(51.2 x 51.2 x 6.4 m)를 회색 선으로 그린다
    (기본: 그리지 않음). 축 범위는 박스 유무와 무관하게 고정 (여백 crop 없음).
  - 좌표: x = 전방(index 0 = 차량 위치), y index 0 = 오른쪽(-25.6 m), z = 위.  화면 오른쪽 = 차량 오른쪽.
출력 (이미 존재하는 파일이 하나라도 있으면 시작 전에 중단)
  {out}/gt/images/{frame}.png, {out}/{name}/images[_mask]/{frame}.png, {out}/comparison[_mask]/{frame}.png

Usage (from the repository root):
  python tools/vis/render_ssc.py --seq 08 \
      --pred voxformer_t=outputs/voxformer_t/labels \
             ecssc=outputs/ecssc/labels \
      --out outputs/qualitative --comparison [--mask-gt-invalid] [--draw-bbox] [--frames 000000 ...] [--workers 16]
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pred_io  # noqa: E402
GRID = (256, 256, 32)
AZIM, ELEV = 30.0, 40.0                 # camera behind-left of the vehicle, looking forward-down
FIG_W, FIG_H, DPI = 8.0, 5.6, 150       # 1200 x 840 px
SHADE = {'top': 1.0, 'side': 0.80, 'back': 0.65}
BG = '#ffffff'
DISPLAY_NAMES = {'gt': 'Ground Truth', 'voxformer_t': 'VoxFormer-T', 'ecssc': 'ECSSC (Ours)'}
CLASS_NAMES = ["empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
               "motorcyclist", "road", "parking", "sidewalk", "other-ground", "building", "fence",
               "vegetation", "trunk", "terrain", "pole", "traffic-sign"]


def load_maps(yaml_path):
    y = yaml.safe_load(open(yaml_path))
    lut = np.full(65536, -1, np.int64)
    for k, v in y['learning_map'].items():
        lut[k] = v
    colors = np.zeros((20, 3), np.float64)
    for c in range(20):
        b, g, r = y['color_map'][y['learning_map_inv'][c]]
        colors[c] = (r / 255.0, g / 255.0, b / 255.0)            # BGR -> RGB
    return lut, colors


def camera():
    az, el = np.radians(AZIM), np.radians(ELEV)
    to_cam = np.array([-np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # scene -> camera
    f = -to_cam
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f, r, u, to_cam


F, R, U, TO_CAM = camera()
# 보이는 면: 법선 n 에 대해 n . to_cam > 0 인 축 방향 면 (behind-left-above -> -x, +y, +z)
FACES = []
for axis in range(3):
    for sgn in (-1, 1):
        n = np.zeros(3); n[axis] = sgn
        if n @ TO_CAM > 1e-9:
            kind = 'top' if axis == 2 else ('back' if axis == 0 else 'side')
            FACES.append((axis, sgn, kind))


def face_quads(axis, sgn):
    """unit cube [0,1]^3 의 해당 면 4 꼭짓점 (voxel 원점 기준)."""
    o = [a for a in range(3) if a != axis]
    c = np.zeros((4, 3))
    c[:, axis] = 1.0 if sgn > 0 else 0.0
    for i, (p, q) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        c[i, o[0]], c[i, o[1]] = p, q
    return c


QUAD = {(a, s): face_quads(a, s) for a, s, _ in FACES}


def project(pts):
    return np.stack([pts @ R, pts @ U], -1)


# 고정 축 범위: 전체 grid 의 8 꼭짓점 투영
_corners = np.array([[x, y, z] for x in (0, GRID[0]) for y in (0, GRID[1]) for z in (0, GRID[2])], float)
_pc = project(_corners)
PAD = 6.0
XLIM = (_pc[:, 0].min() - PAD, _pc[:, 0].max() + PAD)
YLIM = (_pc[:, 1].min() - PAD, _pc[:, 1].max() + PAD)
BOX_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if (np.abs(_corners[i] - _corners[j]) > 0).sum() == 1]


def build_polys(labels, colors):
    """labels: (256,256,32) int, 0 = 그리지 않음. -> verts (N,4,2), facecolors (N,3), depth 순 정렬."""
    occ = labels > 0
    verts, cols, depth = [], [], []
    for axis, sgn, kind in FACES:
        nb = np.zeros_like(occ)                           # 해당 방향 이웃 점유 여부 (경계 밖 = 비어 있음)
        src = [slice(None)] * 3; dst = [slice(None)] * 3
        if sgn > 0:
            src[axis] = slice(1, None); dst[axis] = slice(0, -1)
        else:
            src[axis] = slice(0, -1); dst[axis] = slice(1, None)
        nb[tuple(dst)] = occ[tuple(src)]
        exp = occ & ~nb
        idx = np.argwhere(exp).astype(np.float64)        # (M,3) voxel 원점
        if len(idx) == 0:
            continue
        q = QUAD[(axis, sgn)]
        pts = idx[:, None, :] + q[None, :, :]             # (M,4,3)
        verts.append(project(pts.reshape(-1, 3)).reshape(-1, 4, 2))
        lab = labels[exp]
        cols.append(colors[lab] * SHADE[kind])
        depth.append(pts.mean(1) @ F)
    if not verts:
        return np.zeros((0, 4, 2)), np.zeros((0, 3))
    verts, cols, depth = np.concatenate(verts), np.concatenate(cols), np.concatenate(depth)
    order = np.argsort(-depth, kind='stable')             # 먼 면부터
    return verts[order], cols[order]


def render(labels, colors, path, draw_bbox=False):
    verts, cols = build_polys(labels, colors)
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG)
    if draw_bbox:
        box = project(_corners)
        ax.add_collection(LineCollection([box[[i, j]] for i, j in BOX_EDGES], colors='#b4b4b4', linewidths=0.6, zorder=0))
    if len(verts):
        ax.add_collection(PolyCollection(verts, facecolors=cols, edgecolors=cols * 0.85, linewidths=0.15,
                                         antialiased=True, zorder=1))
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect('equal'); ax.axis('off')
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return len(verts)


def compose(frame, seq, panels, colors, path):
    """panels: [(display_name, png_path)] -> 가로 배치 + 범례 (영문)."""
    imgs = [plt.imread(p) for _, p in panels]
    h, w = imgs[0].shape[:2]
    fig = plt.figure(figsize=(len(imgs) * w / DPI, (h + 150) / DPI), dpi=DPI, facecolor=BG)
    top = 40 / (h + 150); leg = 110 / (h + 150)
    for i, ((name, _), im) in enumerate(zip(panels, imgs)):
        ax = fig.add_axes([i / len(imgs), leg, 1 / len(imgs), 1 - top - leg])
        ax.imshow(im); ax.axis('off')
        fig.text((i + 0.5) / len(imgs), 1 - top / 2, name, ha='center', va='center', fontsize=11, color='#1f1f1e')
    fig.text(0.005, 1 - top / 2, f'Seq {seq} / frame {frame}', ha='left', va='center', fontsize=9, color='#6b6a66')
    lax = fig.add_axes([0.01, 0.0, 0.98, leg]); lax.axis('off')
    n = 19; per_row = 10
    for c in range(1, 20):
        k = c - 1; row, col = divmod(k, per_row)
        x0 = col / per_row; y0 = 0.62 - row * 0.42
        lax.add_patch(plt.Rectangle((x0, y0), 0.012, 0.28, color=colors[c], transform=lax.transAxes))
        lax.text(x0 + 0.016, y0 + 0.14, CLASS_NAMES[c], va='center', fontsize=8, color='#1f1f1e', transform=lax.transAxes)
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)


# ============================================================
# 원근(perspective) 시점: --view camera / camera_raised
# ============================================================
VOX_SIZE = 0.2
VOX_ORIGIN = np.array([0.0, -25.6, -2.0])     # voxel (0,0,0) 의 LiDAR 좌표 (voxformer_head.get_ref_3d 와 동일)
RGB_H, RGB_W = 370, 1220                       # dataset 의 image_2 crop (img[:370, :1220])
NEAR = 0.1                                     # m, 카메라 앞 near plane (뒤/근접 voxel 제외)
FACES6 = [(axis, sgn, 'top' if axis == 2 else ('back' if axis == 0 else 'side')) for axis in range(3) for sgn in (-1, 1)]
QUAD6 = {(a, s): face_quads(a, s) for a, s, _ in FACES6}


def read_calib(seq):
    """dataset.read_calib 와 같은 파싱: P2 (3x4), Tr (4x4, velo -> rectified cam0)."""
    d = {}
    for line in open(os.path.join('dataset/semantickitti/sequences', seq, 'calib.txt')):
        if ':' in line:
            k, v = line.split(':', 1)
            d[k.strip()] = np.array([float(x) for x in v.split()])
    Tr = np.eye(4); Tr[:3, :4] = d['Tr'].reshape(3, 4)
    return d['P2'].reshape(3, 4), Tr


class PerspCam:
    """world(LiDAR) -> camera -> pixel.
    camera       : 정확한 cam2.  pixel = P2 @ Tr @ X  (P2 = K [I | t], 카메라 중심 c2 = -t)
    camera_raised: cam2 와 같은 시선(rect 좌표 축) 에서 위로 h, 뒤로 d 이동 후 pitch 만큼 아래로 기울임.
                   intrinsics 는 hfov / 이미지 크기로 새로 정의 (주점 = 이미지 중심)."""

    def __init__(self, view, P2, Tr, h=0.0, back=0.0, pitch=0.0, hfov=90.0, W=RGB_W, H=RGB_H):
        K2 = P2[:, :3]
        c2 = -np.linalg.solve(K2, P2[:, 3])
        self.Tr = Tr
        if view == 'camera':
            self.C, self.Rwc, self.K, self.W, self.H = c2, np.eye(3), K2, RGB_W, RGB_H
        else:
            th = np.radians(pitch)
            ex, ey, ez = np.array([1.0, 0, 0]), np.array([0, np.cos(th), -np.sin(th)]), np.array([0, np.sin(th), np.cos(th)])
            self.Rwc = np.stack([ex, ey, ez], 1)                 # columns = camera axes in rect cam0 frame (y down)
            self.C = c2 + np.array([0.0, -h, -back])             # up = -y, back = -z
            fx = (W / 2.0) / np.tan(np.radians(hfov) / 2.0)
            self.K = np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1.0]])
            self.W, self.H = W, H

    def to_cam(self, X):                       # X: (N,3) LiDAR
        Xr = X @ self.Tr[:3, :3].T + self.Tr[:3, 3]
        return (Xr - self.C) @ self.Rwc

    def dir_to_cam(self, n):                   # 방향 벡터 (LiDAR) -> camera
        return (self.Tr[:3, :3] @ n) @ self.Rwc

    def project(self, Xc):
        uv = Xc @ self.K.T
        return uv[..., :2] / uv[..., 2:3]


def build_polys_persp(labels, colors, cam):
    occ = labels > 0
    verts, cols, depth = [], [], []
    for axis, sgn, kind in FACES6:
        nb = np.zeros_like(occ)
        src = [slice(None)] * 3; dst = [slice(None)] * 3
        if sgn > 0:
            src[axis] = slice(1, None); dst[axis] = slice(0, -1)
        else:
            src[axis] = slice(0, -1); dst[axis] = slice(1, None)
        nb[tuple(dst)] = occ[tuple(src)]
        exp = occ & ~nb
        idx = np.argwhere(exp).astype(np.float64)
        if len(idx) == 0:
            continue
        pts = VOX_ORIGIN + (idx[:, None, :] + QUAD6[(axis, sgn)][None]) * VOX_SIZE      # (M,4,3) LiDAR
        pc = cam.to_cam(pts.reshape(-1, 3)).reshape(-1, 4, 3)
        ctr = pc.mean(1)
        n = np.zeros(3); n[axis] = sgn
        keep = (pc[..., 2] > NEAR).all(1) & ((ctr @ cam.dir_to_cam(n)) < 0)            # 카메라 앞 + 카메라를 향한 면
        if not keep.any():
            continue
        pc, ctr = pc[keep], ctr[keep]
        uv = cam.project(pc)
        inview = ~((uv[..., 0] < 0).all(1) | (uv[..., 0] > cam.W).all(1) | (uv[..., 1] < 0).all(1) | (uv[..., 1] > cam.H).all(1))
        verts.append(uv[inview]); cols.append(colors[labels[exp][keep][inview]] * SHADE[kind])
        depth.append(np.linalg.norm(ctr[inview], axis=1))
    if not verts:
        return np.zeros((0, 4, 2)), np.zeros((0, 3))
    verts, cols, depth = np.concatenate(verts), np.concatenate(cols), np.concatenate(depth)
    order = np.argsort(-depth, kind='stable')                                          # 카메라에서 먼 면부터
    return verts[order], cols[order]


def render_persp(labels, colors, path, cam):
    verts, cols = build_polys_persp(labels, colors, cam)
    fig = plt.figure(figsize=(cam.W / DPI, cam.H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG)
    if len(verts):
        ax.add_collection(PolyCollection(verts, facecolors=cols, edgecolors=cols * 0.85, linewidths=0.15,
                                         antialiased=True, zorder=1))
    ax.set_xlim(0, cam.W); ax.set_ylim(cam.H, 0); ax.axis('off')
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return len(verts)


def load_rgb(seq, frame, rgb_dir=None):
    p = (os.path.join('dataset/semantickitti/sequences', seq, 'image_2', frame + '.png') if rgb_dir is None
         else os.path.join(rgb_dir, frame + '.png'))
    im = plt.imread(p)[:RGB_H, :RGB_W, :3]
    return im.astype(np.float64) if im.dtype != np.uint8 else im.astype(np.float64) / 255.0


def overlay_rgb(seq, frame, render_png, path, alpha=0.55):
    rgb = load_rgb(seq, frame)
    r = plt.imread(render_png)[..., :3].astype(np.float64)
    fg = (r < 0.999).any(-1, keepdims=True)                 # 배경(흰색)이 아닌 픽셀
    out = np.where(fg, (1 - alpha) * rgb + alpha * r, rgb)
    plt.imsave(path, np.clip(out, 0, 1))


def compose_row(frame, seq, panels, colors, path, title_h=40, legend_h=110):
    """panels: [(display_name, HxWx3 array)] -> 모든 패널을 첫 패널(RGB) 높이에 맞춰 가로 배치 + 범례."""
    from PIL import Image
    H = panels[0][1].shape[0]
    ims = []
    for name, im in panels:
        im8 = (np.clip(im[..., :3], 0, 1) * 255).round().astype(np.uint8)
        if im8.shape[0] != H:
            w = int(round(im8.shape[1] * H / im8.shape[0]))
            im8 = np.asarray(Image.fromarray(im8).resize((w, H), Image.LANCZOS))
        ims.append((name, im8))
    gap = 8
    Wt = sum(im.shape[1] for _, im in ims) + gap * (len(ims) - 1)
    Ht = H + title_h + legend_h
    fig = plt.figure(figsize=(Wt / DPI, Ht / DPI), dpi=DPI, facecolor=BG)
    x = 0
    for name, im in ims:
        w = im.shape[1]
        ax = fig.add_axes([x / Wt, legend_h / Ht, w / Wt, H / Ht]); ax.imshow(im, interpolation='none'); ax.axis('off')
        fig.text((x + w / 2) / Wt, 1 - (title_h / 2) / Ht, name, ha='center', va='center', fontsize=11, color='#1f1f1e')
        x += w + gap
    fig.text(0.003, 1 - (title_h / 2) / Ht, f'Seq {seq} / frame {frame}', ha='left', va='center', fontsize=9, color='#6b6a66')
    lax = fig.add_axes([0.01, 0.0, 0.98, legend_h / Ht]); lax.axis('off')
    per_row = 10
    for c in range(1, 20):
        row, col = divmod(c - 1, per_row)
        x0 = col / per_row; y0 = 0.62 - row * 0.42
        lax.add_patch(plt.Rectangle((x0, y0), 0.008, 0.28, color=colors[c], transform=lax.transAxes))
        lax.text(x0 + 0.011, y0 + 0.14, CLASS_NAMES[c], va='center', fontsize=8, color='#1f1f1e', transform=lax.transAxes)
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)


def job(task):
    frame, a = task
    lut, colors = load_maps(a['yaml'])
    t0 = time.time()
    gt = np.load(os.path.join(a['gt'], a['seq'], frame + '_1_1.npy')).astype(np.int64)
    invalid = gt == 255
    persp = a['view'] != 'oblique'
    if persp:
        P2, Tr = read_calib(a['seq'])
        cam = PerspCam(a['view'], P2, Tr, **a['cam'])
        draw = lambda lab, path: render_persp(lab, colors, path, cam)
    else:
        draw = lambda lab, path: render(lab, colors, path, a['draw_bbox'])
    gl = gt.copy(); gl[invalid] = 0
    p = a['paths'][('gt', frame)]
    if p is not None:
        draw(gl, p)
    if a['overlay']:
        overlay_rgb(a['seq'], frame, a['paths'][('gt_any', frame)], a['paths'][('ovl', frame)])
    panels = [(DISPLAY_NAMES['gt'], a['paths'][('gt_any', frame)])]
    for name, root in a['preds']:
        lab = pred_io.load_pred(root, a['seq'], frame, lut)
        if a['mask_gt_invalid']:
            lab = lab.copy(); lab[invalid] = 0
        draw(lab, a['paths'][(name, frame)])
        panels.append((DISPLAY_NAMES.get(name, name), a['paths'][(name, frame)]))
    if a['comparison']:
        if a['with_rgb']:
            row = [(a['rgb_title'], load_rgb(a['seq'], frame, a['rgb_dir']))] + [(n, plt.imread(pp)) for n, pp in panels]
            compose_row(frame, a['seq'], row, colors, a['paths'][('cmp', frame)])
        else:
            compose(frame, a['seq'], panels, colors, a['paths'][('cmp', frame)])
    return frame, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='08')
    ap.add_argument('--gt', default='dataset/semantickitti/labels')
    ap.add_argument('--pred', nargs='*', default=[], help='name=labels_root')
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', nargs='+', default=None)
    ap.add_argument('--mask-gt-invalid', action='store_true', help='hide predictions where GT == 255 (default: draw all)')
    ap.add_argument('--draw-bbox', action='store_true', help='draw the scene bounding box wireframe (default: off; oblique only)')
    ap.add_argument('--comparison', action='store_true')
    ap.add_argument('--with-rgb', action='store_true', help='comparison row: RGB | GT | preds, all scaled to the RGB height')
    ap.add_argument('--view', choices=['oblique', 'camera', 'camera_raised'], default='oblique')
    ap.add_argument('--overlay', action='store_true', help='(--view camera) GT render blended over RGB -> {out}/overlay/')
    ap.add_argument('--raise-h', type=float, default=8.0, help='camera_raised: up shift from cam2 [m]')
    ap.add_argument('--raise-back', type=float, default=8.0, help='camera_raised: backward shift from cam2 [m]')
    ap.add_argument('--pitch', type=float, default=25.0, help='camera_raised: downward tilt [deg]')
    ap.add_argument('--hfov', type=float, default=90.0, help='camera_raised: horizontal FOV [deg] (KITTI cam2 ~81.9)')
    ap.add_argument('--img-w', type=int, default=1220, help='camera_raised image width [px]')
    ap.add_argument('--img-h', type=int, default=740, help='camera_raised image height [px]')
    ap.add_argument('--gt-images', default=None, help='use existing GT renders DIR/{frame}.png instead of rendering GT')
    ap.add_argument('--rgb-dir', default=None, help='(--with-rgb) RGB panel from DIR/{frame}.png instead of image_2')
    ap.add_argument('--rgb-title', default='RGB (image_2)', help='(--with-rgb) RGB panel title')
    ap.add_argument('--yaml', default='preprocess/label/semantic-kitti.yaml')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.overlay and a.view != 'camera':
        ap.error('--overlay requires --view camera')
    if a.rgb_dir and not a.with_rgb:
        ap.error('--rgb-dir requires --with-rgb')
    if a.gt_images and a.overlay:
        ap.error('--gt-images cannot be combined with --overlay')
    preds = [tuple(s.split('=', 1)) for s in a.pred]
    frames = a.frames or sorted(os.path.basename(f)[:6] for f in glob.glob(os.path.join(a.gt, a.seq, '*_1_1.npy')))
    vsfx = '' if a.view == 'oblique' else '_' + a.view
    sfx = vsfx + ('_mask' if a.mask_gt_invalid else '')
    paths = {}
    for f in frames:
        g = os.path.join(a.out, 'gt', 'images' + vsfx, f + '.png')
        # GT 이미지는 마스크 옵션과 무관 -> mask 버전 실행 시 이미 있으면 재사용, 없으면 생성
        paths[('gt', f)] = None if os.path.exists(g) and a.mask_gt_invalid else g
        paths[('gt_any', f)] = g
        if a.gt_images:                                   # 기존 GT 렌더 재사용 (GT 는 그리지 않음)
            paths[('gt', f)] = None
            paths[('gt_any', f)] = os.path.join(a.gt_images, f + '.png')
        for name, _ in preds:
            paths[(name, f)] = os.path.join(a.out, name, 'images' + sfx, f + '.png')
        if a.comparison:
            paths[('cmp', f)] = os.path.join(a.out, 'comparison' + sfx, f + '.png')
        if a.overlay:
            paths[('ovl', f)] = os.path.join(a.out, 'overlay', f + '.png')
    targets = [p for k, p in paths.items() if p is not None and k[0] != 'gt_any']
    exist = [p for p in targets if os.path.exists(p)]
    if exist:
        print(f'[render] ABORT: {len(exist)} output files already exist, e.g. {exist[:3]}')
        sys.exit(3)
    for name, root in preds:
        miss = [f for f in frames if pred_io.pred_file(root, a.seq, f) is None]
        if miss:
            print(f'[render] ABORT: {name}: {len(miss)} label files missing, e.g. {miss[:3]}')
            sys.exit(1)
    for flag, d in (('--gt-images', a.gt_images), ('--rgb-dir', a.rgb_dir)):
        miss = [f for f in frames if d and not os.path.exists(os.path.join(d, f + '.png'))]
        if miss:
            print(f'[render] ABORT: {flag}: {len(miss)} images missing in {d}, e.g. {miss[:3]}')
            sys.exit(1)
    for p in targets:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    camp = dict(h=a.raise_h, back=a.raise_back, pitch=a.pitch, hfov=a.hfov, W=a.img_w, H=a.img_h)
    vdesc = (f'azim={AZIM} elev={ELEV} size={int(FIG_W * DPI)}x{int(FIG_H * DPI)}' if a.view == 'oblique' else
             'P2@Tr size=1220x370' if a.view == 'camera' else
             f'raise_h={a.raise_h} back={a.raise_back} pitch={a.pitch} hfov={a.hfov} size={a.img_w}x{a.img_h}')
    print(f'[render] view={a.view} ({vdesc}) frames={len(frames)} preds={[n for n, _ in preds]} '
          f'mask_gt_invalid={a.mask_gt_invalid} draw_bbox={a.draw_bbox} comparison={a.comparison} with_rgb={a.with_rgb} '
          f'overlay={a.overlay} gt_images={a.gt_images} rgb_dir={a.rgb_dir} rgb_title={a.rgb_title!r}', flush=True)
    cfg = dict(yaml=a.yaml, gt=a.gt, seq=a.seq, preds=preds, mask_gt_invalid=a.mask_gt_invalid, draw_bbox=a.draw_bbox,
               comparison=a.comparison, with_rgb=a.with_rgb, view=a.view, cam=camp, overlay=a.overlay, paths=paths,
               rgb_dir=a.rgb_dir, rgb_title=a.rgb_title)
    t0 = time.time(); times = []
    with Pool(a.workers) as pool:
        for k, (f, dt) in enumerate(pool.imap(job, [(f, cfg) for f in frames]), 1):
            times.append(dt)
            if k % 50 == 0 or k == len(frames):
                print(f'[render] {k}/{len(frames)}  wall {time.time() - t0:.0f}s', flush=True)
    print(f'[render] done: {len(frames)} frames, per-frame (all panels, 1 worker) mean {np.mean(times):.2f}s '
          f'max {np.max(times):.2f}s, wall {time.time() - t0:.1f}s with {a.workers} workers')


if __name__ == '__main__':
    main()
