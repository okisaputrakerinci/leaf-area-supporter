# gui_labeler.py
from __future__ import annotations
import os, sys, shutil, random, time
from typing import List, Tuple, Optional

import numpy as np
import cv2

from PySide6.QtCore import Qt, QPoint, QRect, QSize, QTimer, QCoreApplication
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QPen, QAction, QKeySequence, QPainterPath, QIcon
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QSlider, QDoubleSpinBox, QCheckBox, QLineEdit, QStatusBar, QComboBox, QSpinBox,
    QInputDialog, QMenuBar
)

def _set_win_app_user_model_id(app_id: str = "OSAI.LeafLabeler"):
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass

# ===================== App metadata =====================
try:
    from .meta import APP_NAME, APP_VERSION, APP_PUBLISHER, COPYRIGHT_TXT
except Exception:
    try:
        from leaf_labeler.meta import APP_NAME, APP_VERSION, APP_PUBLISHER, COPYRIGHT_TXT
    except Exception:
        APP_NAME      = "LeafLabeler"
        APP_VERSION   = "1.0.0"
        APP_PUBLISHER = "OS AI Corp"
        COPYRIGHT_TXT = "© 2025 Oki Saputra - OS AI Corp. All rights reserved."


# ===================== Licensing (robust import) =====================
# Mencoba relative import (saat dijalankan sebagai package), lalu absolute (saat path berbeda),
# jika gagal pakai fallback supaya dev masih bisa jalan.
try:
    from .licensing.license_client import check_license, save_license_key, hardware_fingerprint
except Exception:
    try:
        from leaf_labeler.licensing.license_client import check_license, save_license_key, hardware_fingerprint
    except Exception:
        def check_license() -> dict:
            # fallback dev mode (trial dummy)
            return {"valid": True, "type": "trial", "remaining": 14, "expired": False, "name": None, "features": None}
        def save_license_key(_text: str):
            return False, "Modul lisensi tidak ditemukan. Pastikan src/leaf_labeler/licensing/license_client.py ada."
        def hardware_fingerprint() -> str:
            # fallback: node-based (tidak terkunci kuat, hanya untuk dev)
            try:
                import uuid, hashlib
                n = uuid.getnode()
                return hashlib.sha256(str(n).encode("utf-8")).hexdigest()
            except Exception:
                return "NO-FP"

# ===================== Asset helper =====================
def app_asset(filename: str) -> Optional[str]:
    import sys
    exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    src_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = []

    # PyInstaller (one-file / one-folder)
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
        candidates += [
            os.path.join(base, "assets", filename),
            os.path.join(base, "leaf_labeler", "assets", filename),
        ]

    # Sampingan EXE/script (one-folder PyInstaller & dev)
    candidates += [
        os.path.join(exe_dir, "assets", filename),
        os.path.join(exe_dir, "leaf_labeler", "assets", filename),
    ]

    # Sumber modul (dev)
    candidates += [
        os.path.join(src_dir, "assets", filename),
        os.path.join(os.path.dirname(src_dir), "assets", filename),
    ]

    for p in candidates:
        if os.path.isfile(p):
            return p

    # Fallback via importlib.resources (kalau paket terinstall)
    try:
        from importlib.resources import files
        r = files("leaf_labeler.assets") / filename
        if r.is_file():
            return str(r)
    except Exception:
        pass
    return None


# ===================== Utils =====================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def imread(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def imwrite(path: str, img: np.ndarray) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ext = os.path.splitext(path)[1]
        ok, buf = cv2.imencode(ext if ext else ".png", img)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False

def copy_text(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(src):
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
    else:
        with open(dst, "w", encoding="utf-8") as fdst:
            fdst.write("")

def copy_image(src: str, dst: str) -> bool:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img = imread(src)
    if img is None:
        try:
            shutil.copyfile(src, dst)
            return True
        except Exception:
            return False
    return imwrite(dst, img)

def list_images_in(dirpath: str) -> List[str]:
    if not dirpath or not os.path.isdir(dirpath):
        return []
    files = [f for f in os.listdir(dirpath) if f.lower().endswith(IMG_EXTS)]
    files.sort()
    return files

# ===================== Viewer =====================
class ImageViewer(QWidget):
    def _bi_key(self, tol: float, brush: int, fast: bool) -> tuple[int,int,int]:
        tol_k = int(round(tol * 2.0))      # quantize 0.5 ΔE
        br_k  = int(round(brush / 6.0))    # quantize per ~6px
        return (1 if fast else 0, tol_k, br_k)

    def _get_bilateral(self, tol: float, brush: int, fast: bool) -> Optional[np.ndarray]:
        lab_ds = self._wand_cache["lab_ds"]
        if lab_ds is None:
            return None
        key = self._bi_key(tol, brush, fast)
        if fast:
            if key == self._bi_key_fast and self._bi_fast is not None:
                return self._bi_fast
            sc = max(6.0, tol*3.0 + 6.0)
            ss = max(3.0, brush*0.6)
            out = cv2.bilateralFilter(lab_ds, d=0, sigmaColor=sc, sigmaSpace=ss)
            self._bi_key_fast = key; self._bi_fast = out
            return out
        else:
            if key == self._bi_key_full and self._bi_full is not None:
                return self._bi_full
            sc = max(10.0, tol*4.0 + 8.0)
            ss = max(5.0, brush*0.9)
            out = cv2.bilateralFilter(lab_ds, d=0, sigmaColor=sc, sigmaSpace=ss)
            out = cv2.bilateralFilter(out, d=0, sigmaColor=sc*0.6, sigmaSpace=ss*0.8)
            self._bi_key_full = key; self._bi_full = out
            return out

    def _roi_flood_lab(self, lab_ds: np.ndarray, seed_xy_ds: tuple[int,int], patch_r: int, tol_de: float):
        Hs, Ws = lab_ds.shape[:2]
        cx, cy = int(np.clip(seed_xy_ds[0], 0, Ws-1)), int(np.clip(seed_xy_ds[1], 0, Hs-1))
        roi_r = int(max(16, patch_r * 2.5))
        x0, x1 = max(0, cx - roi_r), min(Ws, cx + roi_r + 1)
        y0, y1 = max(0, cy - roi_r), min(Hs, cy + roi_r + 1)
        if x1 - x0 <= 2 or y1 - y0 <= 2:
            return np.zeros((1,1), np.uint8), (0,0,0,0)

        roi = lab_ds[y0:y1, x0:x1].copy()
        tL, tA, tB = self._tol_to_channel_diff(tol_de)
        mask = np.zeros((roi.shape[0] + 2, roi.shape[1] + 2), np.uint8)
        flags = cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | 8 | (255 << 8)
        cv2.floodFill(roi, mask, (cx - x0, cy - y0), (0,0,0), (tL,tA,tB), (tL,tA,tB), flags)
        region_roi = (mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255

        if region_roi.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            region_roi = cv2.morphologyEx(region_roi, cv2.MORPH_OPEN, k, 1)
        return region_roi, (x0, y0, x1, y1)

    def __init__(self, parent=None):
        super().__init__(parent)
        ico_path = app_asset("logo_os_ai_corp.ico")
        if ico_path:
            self.setWindowIcon(QIcon(ico_path))

        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.NoContextMenu)

        # Image & Lab
        self.image: Optional[np.ndarray] = None
        self.lab: Optional[np.ndarray] = None
        self.display_qpix: Optional[QPixmap] = None

        # Seeds & selection
        self.mask_sel_temp: Optional[np.ndarray] = None
        self.seed_fg: Optional[np.ndarray] = None
        self.seed_bg: Optional[np.ndarray] = None
        self.wand_painted: Optional[np.ndarray] = None

        # Instances: list of (mask_u8, class_id)
        self.instances: List[Tuple[np.ndarray, int]] = []

        # Tools
        self.tool: str = "fg"      # 'fg'|'bg'|'wand'
        self.brush = 18
        self.wand_tol = 12.0
        self.wand_to_bg = False
        self.wand_global = False   # global (non-contiguous)

        # View transform
        self.view_scale = 1.0
        self.view_offset = QPoint(0, 0)
        self.auto_fit = True
        self.min_scale = 0.05
        self.max_scale = 10.0

        # Interaction
        self._panning = False
        self._pan_start = QPoint(0, 0)
        self._wand_sweep = False
        self._wand_btn: Optional[int] = None
        self._last_wand_xy: Optional[Tuple[int, int]] = None
        self._last_wand_t = 0.0

        self.cursor_img_xy: Optional[Tuple[int, int]] = None

        # Overlays
        self.show_instances = True
        self.show_wand_painted = True
        self.show_seeds = True
        self.show_selection = True
        self.show_ants = True

        # Overlay caches
        self._cache_instances_pix: Optional[QPixmap] = None
        self._cache_fg_pix: Optional[QPixmap] = None
        self._cache_bg_pix: Optional[QPixmap] = None
        self._cache_sel_pix: Optional[QPixmap] = None
        self._cache_wand_pix: Optional[QPixmap] = None
        self._dirty_instances = True
        self._dirty_fg = False
        self._dirty_bg = False
        self._dirty_sel = False
        self._dirty_wand = False

        # Marching ants
        self._ants_phase = 0.0
        self._ants_timer = QTimer(self)
        self._ants_timer.timeout.connect(self._on_tick_ants)
        self._ants_timer.start(120)
        self._ants_cnts: List[np.ndarray] = []
        self._ants_dirty = True

        # Wand caches (downscale)
        self._wand_cache = dict(scale=1.0, bgr_ds=None, lab_ds=None)

        # cache bilateral (fast/full)
        self._bi_key_fast = None
        self._bi_key_full = None
        self._bi_fast = None
        self._bi_full = None

        # Undo/Redo
        self._history: List[dict] = []
        self._future: List[dict] = []
        self._stroke_active = False
        self._history_limit = 60

    # ---------- setters ----------
    def set_tool(self, name: str):
        self.tool = name
        self.setCursor(Qt.CrossCursor if name == "wand" else Qt.ArrowCursor)
        self.update()

    def set_brush(self, size: int):
        self.brush = max(1, int(size)); self.update()

    def set_wand_tol(self, tol: float):
        self.wand_tol = float(max(0.0, tol)); self.update()

    def set_wand_to_bg(self, flag: bool):
        self.wand_to_bg = bool(flag)

    def set_wand_global(self, flag: bool):
        self.wand_global = bool(flag)

    # toggles
    def set_show_instances(self, v: bool): self.show_instances = bool(v); self.update()
    def set_show_wand(self, v: bool): self.show_wand_painted = bool(v); self.update()
    def set_show_seeds(self, v: bool): self.show_seeds = bool(v); self.update()
    def set_show_selection(self, v: bool): self.show_selection = bool(v); self.update()
    def set_show_ants(self, v: bool): self.show_ants = bool(v); self.update()

    # ---------- image I/O ----------
    def set_image(self, bgr: Optional[np.ndarray]):
        self.image = None if bgr is None else bgr.copy()
        self.lab = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        self.display_qpix = None
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
            self.display_qpix = QPixmap.fromImage(qimg)
        if bgr is None:
            self.mask_sel_temp = self.seed_fg = self.seed_bg = None
            self.wand_painted = None
            self.instances.clear()
        else:
            H, W = bgr.shape[:2]
            self.mask_sel_temp = np.zeros((H, W), np.uint8)
            self.seed_fg = np.zeros((H, W), np.uint8)
            self.seed_bg = np.zeros((H, W), np.uint8)
            self.wand_painted = np.zeros((H, W), np.uint8)
            self.instances.clear()

        # reset caches
        self._cache_instances_pix = self._cache_fg_pix = self._cache_bg_pix = None
        self._cache_sel_pix = self._cache_wand_pix = None
        self._dirty_instances = self._dirty_fg = self._dirty_bg = self._dirty_sel = self._dirty_wand = True
        self._ants_cnts = []; self._ants_dirty = True

        # wand cache
        self._prepare_wand_cache()

        # history
        self._history.clear(); self._future.clear(); self._stroke_active = False

        self.auto_fit = True
        self.fit_to_window()
        self.update()

    def _prepare_wand_cache(self):
        self._wand_cache = dict(scale=1.0, bgr_ds=None, lab_ds=None)
        self._bi_key_fast = self._bi_key_full = None
        self._bi_fast = self._bi_full = None
        if self.image is None: return
        H, W = self.image.shape[:2]
        target = 1100
        s = 1.0 if max(H, W) <= target else (target / float(max(H, W)))
        if s < 0.25: s = 0.25
        bgr_ds = self.image if abs(s-1.0) < 1e-3 else cv2.resize(self.image, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
        lab_ds = cv2.cvtColor(bgr_ds, cv2.COLOR_BGR2Lab)
        self._wand_cache.update(scale=s, bgr_ds=bgr_ds, lab_ds=lab_ds)

    def mark_instances_dirty(self):
        self._dirty_instances = True
        self._cache_instances_pix = None
        self.update()

    def clear_seeds_and_temp(self):
        if self.image is None:
            return
        self._push_history()
        self.seed_fg[:] = 0; self.seed_bg[:] = 0
        self.mask_sel_temp[:] = 0
        if self.wand_painted is not None: self.wand_painted[:] = 0
        self._dirty_fg = self._dirty_bg = self._dirty_sel = self._dirty_wand = True
        self._cache_fg_pix = self._cache_bg_pix = self._cache_sel_pix = self._cache_wand_pix = None
        self._ants_cnts = []; self._ants_dirty = True
        self.update()

    # ---------- Undo/Redo ----------
    def _snapshot(self) -> dict:
        return dict(
            seed_fg=self.seed_fg.copy(),
            seed_bg=self.seed_bg.copy(),
            wand=self.wand_painted.copy(),
            sel=self.mask_sel_temp.copy()
        )

    def _restore(self, snap: dict):
        self.seed_fg[:] = snap["seed_fg"]
        self.seed_bg[:] = snap["seed_bg"]
        self.wand_painted[:] = snap["wand"]
        self.mask_sel_temp[:] = snap["sel"]
        self._dirty_fg = self._dirty_bg = self._dirty_sel = self._dirty_wand = True
        self._cache_fg_pix = self._cache_bg_pix = self._cache_sel_pix = self._cache_wand_pix = None
        self._ants_dirty = True
        self.update()

    def _push_history(self):
        if self.image is None: return
        self._history.append(self._snapshot())
        if len(self._history) > self._history_limit:
            self._history.pop(0)
        self._future.clear()

    def undo(self):
        if not self._history: return
        cur = self._snapshot()
        last = self._history.pop()
        self._future.append(cur)
        self._restore(last)

    def redo(self):
        if not self._future: return
        cur = self._snapshot()
        nxt = self._future.pop()
        self._history.append(cur)
        self._restore(nxt)

    # ---------- view transforms ----------
    def fit_to_window(self):
        if self.display_qpix is None: return
        iw = self.display_qpix.width()
        ih = self.display_qpix.height()
        ww = max(1, self.width()); wh = max(1, self.height())
        s = min(ww/iw, wh/ih)
        self.view_scale = s
        dx = int((ww - iw*s)/2); dy = int((wh - ih*s)/2)
        self.view_offset = QPoint(dx, dy)

    def _image_to_widget_pt(self, xy: Tuple[int, int]) -> QPoint:
        x, y = xy
        return QPoint(int(self.view_offset.x() + x*self.view_scale),
                      int(self.view_offset.y() + y*self.view_scale))

    def _widget_to_image_xy(self, pos: QPoint) -> Optional[Tuple[int,int]]:
        if self.display_qpix is None: return None
        x = (pos.x() - self.view_offset.x()) / (self.view_scale + 1e-9)
        y = (pos.y() - self.view_offset.y()) / (self.view_scale + 1e-9)
        w = self.display_qpix.width(); h = self.display_qpix.height()
        if x < 0 or y < 0 or x >= w or y >= h: return None
        return int(x), int(y)

    def wheelEvent(self, ev):
        if self.display_qpix is None: return
        delta = ev.angleDelta().y()
        if delta == 0: return
        old_scale = self.view_scale
        step = 1.15
        new_scale = old_scale * (step if delta > 0 else 1.0/step)
        new_scale = max(self.min_scale, min(self.max_scale, new_scale))
        if abs(new_scale - old_scale) < 1e-6: return
        mouse_pos = ev.position().toPoint() if hasattr(ev, 'position') else ev.pos()
        img_xy = self._widget_to_image_xy(mouse_pos)
        if img_xy is None:
            img_xy = (self.display_qpix.width()//2, self.display_qpix.height()//2)
        self.auto_fit = False
        wx, wy = mouse_pos.x(), mouse_pos.y()
        ix, iy = img_xy
        self.view_scale = new_scale
        self.view_offset = QPoint(int(wx - ix*new_scale), int(wy - iy*new_scale))
        self.update()

    def resizeEvent(self, ev):
        if self.auto_fit: self.fit_to_window()
        super().resizeEvent(ev)

    # ---------- overlay caches ----------
    def _mask_to_qpix_cached(self, mask_u8: np.ndarray, color: QColor, cache_name: str) -> QPixmap:
        if cache_name == "instances":
            if (self._cache_instances_pix is not None) and (not self._dirty_instances): return self._cache_instances_pix
        elif cache_name == "fg":
            if (self._cache_fg_pix is not None) and (not self._dirty_fg): return self._cache_fg_pix
        elif cache_name == "bg":
            if (self._cache_bg_pix is not None) and (not self._dirty_bg): return self._cache_bg_pix
        elif cache_name == "sel":
            if (self._cache_sel_pix is not None) and (not self._dirty_sel): return self._cache_sel_pix
        elif cache_name == "wand":
            if (self._cache_wand_pix is not None) and (not self._dirty_wand): return self._cache_wand_pix

        h, w = mask_u8.shape[:2]
        alpha = (mask_u8.astype(np.float32) / 255.0) * color.alphaF()
        rgba = np.zeros((h, w, 4), np.float32)
        rgba[..., 0] = color.redF()
        rgba[..., 1] = color.greenF()
        rgba[..., 2] = color.blueF()
        rgba[..., 3] = alpha
        rgba8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        qimg = QImage(rgba8.data, w, h, 4*w, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)

        if cache_name == "instances": self._cache_instances_pix = pix; self._dirty_instances = False
        elif cache_name == "fg": self._cache_fg_pix = pix; self._dirty_fg = False
        elif cache_name == "bg": self._cache_bg_pix = pix; self._dirty_bg = False
        elif cache_name == "sel": self._cache_sel_pix = pix; self._dirty_sel = False
        elif cache_name == "wand": self._cache_wand_pix = pix; self._dirty_wand = False
        return pix

    def _instances_overlay_qpix(self) -> QPixmap:
        if self.image is None: return QPixmap()
        if (self._cache_instances_pix is not None) and (not self._dirty_instances): return self._cache_instances_pix
        H, W = self.image.shape[:2]
        rgba = np.zeros((H, W, 4), np.float32)
        for m, cid in self.instances:
            if m is None or not np.any(m): continue
            r, g, b = _color_for_class(cid); a = 0.35
            on = (m > 0)
            rgba[on, 0] = r/255.0; rgba[on, 1] = g/255.0; rgba[on, 2] = b/255.0; rgba[on, 3] = a
        rgba8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        qimg = QImage(rgba8.data, W, H, 4*W, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg)
        self._cache_instances_pix = pix; self._dirty_instances = False
        return pix

    # ---------- paint ----------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(self.rect(), QColor(28, 28, 28))
        if self.display_qpix is None:
            p.end(); return

        w = int(self.display_qpix.width() * self.view_scale)
        h = int(self.display_qpix.height() * self.view_scale)
        target = QRect(self.view_offset, QSize(w, h))
        p.drawPixmap(target, self.display_qpix)

        if self.show_instances and self.instances:
            p.drawPixmap(target, self._instances_overlay_qpix())

        if self.show_wand_painted and self.wand_painted is not None and np.any(self.wand_painted):
            p.drawPixmap(target, self._mask_to_qpix_cached(self.wand_painted, QColor(255, 220, 30, 90), "wand"))
            if self.show_ants:
                self._draw_marching_ants_cached(p, target)

        if self.show_selection and self.mask_sel_temp is not None and np.any(self.mask_sel_temp):
            p.drawPixmap(target, self._mask_to_qpix_cached(self.mask_sel_temp, QColor(255, 60, 60, 110), "sel"))

        if self.show_seeds and self.seed_fg is not None and np.any(self.seed_fg):
            p.drawPixmap(target, self._mask_to_qpix_cached(self.seed_fg, QColor(50, 220, 90, 140), "fg"))
        if self.show_seeds and self.seed_bg is not None and np.any(self.seed_bg):
            p.drawPixmap(target, self._mask_to_qpix_cached(self.seed_bg, QColor(60, 170, 255, 140), "bg"))

        if self.cursor_img_xy is not None:
            wp = self._image_to_widget_pt(self.cursor_img_xy)
            p.setRenderHint(QPainter.Antialiasing, True)
            r = int(max(3, self.brush * self.view_scale))
            if self.tool in ("fg", "bg"):
                p.setPen(QPen(QColor(255, 255, 255, 220), 1)); p.drawEllipse(wp, r, r)
            elif self.tool == "wand":
                cross = max(12, r)
                p.setPen(QPen(QColor(255, 255, 0, 220), 1))
                p.drawEllipse(wp, r, r)
                p.drawLine(wp + QPoint(-cross, 0), wp + QPoint(cross, 0))
                p.drawLine(wp + QPoint(0, -cross), wp + QPoint(0, cross))
        p.end()

    def _rebuild_ants_cnts(self):
        if self.wand_painted is None or not np.any(self.wand_painted):
            self._ants_cnts = []; self._ants_dirty = False; return
        cnts, _ = cv2.findContours((self.wand_painted > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        H, W = self.wand_painted.shape[:2]
        min_area = max(16, (H*W)//8000)
        self._ants_cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
        self._ants_dirty = False

    def _draw_marching_ants_cached(self, painter: QPainter, _target: QRect):
        if self._ants_dirty: self._rebuild_ants_cnts()
        if not self._ants_cnts: return
        white_pen = QPen(QColor(255, 255, 255), 1)
        black_pen = QPen(QColor(0, 0, 0), 1)
        dash = [4, 4]
        white_pen.setDashPattern(dash); black_pen.setDashPattern(dash)
        white_pen.setDashOffset(self._ants_phase); black_pen.setDashOffset(self._ants_phase + 4)
        sx = self.view_scale; ox = self.view_offset.x(); oy = self.view_offset.y()
        path = QPainterPath()
        for c in self._ants_cnts:
            pts = c.reshape(-1, 2)
            if len(pts) < 2: continue
            x0 = ox + pts[0, 0]*sx; y0 = oy + pts[0, 1]*sx
            path.moveTo(x0, y0)
            for i in range(1, len(pts)):
                x = ox + pts[i, 0]*sx; y = oy + pts[i, 1]*sx
                path.lineTo(x, y)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(white_pen); painter.drawPath(path)
        painter.setPen(black_pen); painter.drawPath(path)

    def _on_tick_ants(self):
        self._ants_phase = (self._ants_phase + 1.0) % 8.0
        if self.show_ants and self.show_wand_painted and self.wand_painted is not None and np.any(self.wand_painted):
            self.update()

    # ---------- interaction ----------
    def mousePressEvent(self, ev):
        if self.image is None:
            return
        btn = ev.button()

        # Pan (tombol tengah)
        if btn == Qt.MiddleButton:
            self._panning = True
            self._pan_start = ev.position().toPoint() if hasattr(ev, 'position') else ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if not self._stroke_active:
            self._push_history()
            self._stroke_active = True

        # WAND: mulai drag → mode cepat
        if self.tool == "wand" and (btn == Qt.LeftButton or btn == Qt.RightButton):
            xy = self._widget_to_image_xy(ev.position().toPoint() if hasattr(ev, 'position') else ev.pos())
            if xy is not None:
                self._wand_sweep = True
                self._wand_btn = btn
                self._last_wand_xy = None
                self._apply_wand(xy, fast=True)
                self._last_wand_xy = xy
            return

        # Brush FG/BG
        if btn == Qt.LeftButton and self.tool in ("fg", "bg"):
            xy = self._widget_to_image_xy(ev.position().toPoint() if hasattr(ev, 'position') else ev.pos())
            if xy is not None:
                self._paint_seed(xy, self.tool)

    def mouseMoveEvent(self, ev):
        if self.image is None:
            return
        xy = self._widget_to_image_xy(ev.position().toPoint() if hasattr(ev, 'position') else ev.pos())
        self.cursor_img_xy = xy

        # Pan
        if self._panning and (ev.buttons() & Qt.MiddleButton):
            cur = ev.position().toPoint() if hasattr(ev, 'position') else ev.pos()
            d = cur - self._pan_start
            self._pan_start = cur
            self.view_offset += d
            self.auto_fit = False
            self.update()
            return

        # WAND drag → panggil fast
        if self._wand_sweep and self.tool == "wand" and xy is not None:
            if self._should_apply_wand(xy):
                self._apply_wand(xy, fast=True)
                self._last_wand_xy = xy
            return

        # Brush FG/BG drag
        if (ev.buttons() & Qt.LeftButton) and self.tool in ("fg", "bg") and xy is not None:
            self._paint_seed(xy, self.tool)

        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.set_tool(self.tool)

        # Selesai WAND → sekali final (tajam + GrabCut)
        if self._wand_sweep and self._wand_btn is not None and ev.button() == self._wand_btn:
            last_xy = self._last_wand_xy
            if last_xy is None:
                lp = ev.position().toPoint() if hasattr(ev, 'position') else ev.pos()
                last_xy = self._widget_to_image_xy(lp)
            if last_xy is not None:
                self._apply_wand(last_xy, fast=False)
            self._wand_sweep = False
            self._wand_btn = None
            self._last_wand_xy = None
            self.set_tool(self.tool)

        if self._stroke_active:
            self._stroke_active = False

    def _should_apply_wand(self, xy: Tuple[int, int]) -> bool:
        now = time.time()
        if self._last_wand_xy is None:
            self._last_wand_xy = xy; self._last_wand_t = now; return True
        lx, ly = self._last_wand_xy
        dx = xy[0]-lx; dy = xy[1]-ly
        thr = max(1, int(self.brush/3))
        moved = (dx*dx + dy*dy) >= (thr*thr)
        timed = (now - self._last_wand_t) > 0.05
        if moved or timed:
            self._last_wand_xy = xy; self._last_wand_t = now; return True
        return False

    # ---------- tools ----------
    def _paint_seed(self, xy: Tuple[int,int], tool: str):
        if self.image is None: return
        x, y = xy; r = int(max(1, self.brush))
        if tool == "fg":
            cv2.circle(self.seed_fg, (x, y), r, 255, -1); self._dirty_fg = True; self._cache_fg_pix = None
        else:
            cv2.circle(self.seed_bg, (x, y), r, 255, -1); self._dirty_bg = True; self._cache_bg_pix = None
        self.update()

    # ====== SMART WAND ======
    def _tol_to_channel_diff(self, tol_de: float) -> Tuple[int,int,int]:
        tL = int(max(1, round(tol_de * 2.2)))
        tA = int(max(1, round(tol_de * 1.6)))
        tB = int(max(1, round(tol_de * 1.6)))
        return min(255,tL), min(255,tA), min(255,tB)

    def _fullimage_flood_lab(self, lab_img: np.ndarray, seed_xy: Tuple[int,int], tol_de: float, global_mode: bool) -> np.ndarray:
        Hs, Ws = lab_img.shape[:2]
        tL, tA, tB = self._tol_to_channel_diff(tol_de)
        lo = (tL, tA, tB); up = (tL, tA, tB)
        cx, cy = seed_xy
        cx = int(np.clip(cx, 0, Ws-1)); cy = int(np.clip(cy, 0, Hs-1))
        work = lab_img.copy()
        mask = np.zeros((Hs+2, Ws+2), np.uint8)
        flags = (cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | 8 | (255<<8))
        cv2.floodFill(work, mask, (cx, cy), (0,0,0), lo, up, flags)
        region = mask[1:-1, 1:-1]
        if global_mode:
            sv = lab_img[cy, cx].astype(np.float32)
            d = np.sqrt(((lab_img.astype(np.float32) - sv)**2).sum(axis=2))
            region = np.where(d <= tol_de, 255, region)
        if region.any():
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            region = cv2.morphologyEx(region, cv2.MORPH_OPEN, k, 1)
        return region

    def _refine_with_grabcut(self, init_mask: np.ndarray) -> np.ndarray:
        if self.image is None: return init_mask
        H, W = init_mask.shape[:2]
        ys, xs = np.where(init_mask>0)
        if len(xs)==0: return init_mask
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        m = max(6, int(0.02*max(H,W)))
        x0 = max(0, x0-m); y0 = max(0, y0-m)
        x1 = min(W-1, x1+m); y1 = min(H-1, y1+m)
        roi = self.image[y0:y1+1, x0:x1+1]
        mk  = init_mask[y0:y1+1, x0:x1+1]
        gc = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
        er = max(1, int(max(roi.shape[:2]) * 0.004))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er*2+1, er*2+1))
        fg_certain = cv2.erode((mk>0).astype(np.uint8)*255, k, 1)
        bg_certain = cv2.erode((mk==0).astype(np.uint8)*255, k, 1)
        gc[fg_certain>0] = cv2.GC_FGD
        gc[bg_certain>0] = cv2.GC_BGD
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        try:
            cv2.grabCut(roi, gc, None, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_MASK)
            out = ((gc==cv2.GC_FGD) | (gc==cv2.GC_PR_FGD)).astype(np.uint8)*255
        except Exception:
            out = mk.copy()
        refined = np.zeros((H,W), np.uint8)
        refined[y0:y1+1, x0:x1+1] = out
        return refined

    def _apply_wand(self, xy: Tuple[int,int], fast: bool = False):
        if self.image is None or self.lab is None:
            return
        s = float(self._wand_cache["scale"])
        xs, ys = int(round(xy[0]*s)), int(round(xy[1]*s))
        brush_ds = max(1, int(round(self.brush * s)))

        if fast:
            lab_bi_fast = self._get_bilateral(self.wand_tol, brush_ds, fast=True)
            if lab_bi_fast is None:
                return
            region_ds, (x0d,y0d,x1d,y1d) = self._roi_flood_lab(lab_bi_fast, (xs, ys), brush_ds, float(self.wand_tol))
            if region_ds.size == 0:
                return
            x0 = int(round(x0d / s)); y0 = int(round(y0d / s))
            x1 = int(round(x1d / s)); y1 = int(round(y1d / s))
            H, W = self.lab.shape[:2]
            x0 = max(0, min(W-1, x0)); x1 = max(1, min(W, x1))
            y0 = max(0, min(H-1, y0)); y1 = max(1, min(H, y1))
            if x1 <= x0 or y1 <= y0:
                return
            region = cv2.resize(region_ds, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            if self.wand_to_bg:
                sub = self.seed_bg[y0:y1, x0:x1]
                cv2.bitwise_or(sub, region, sub); self._dirty_bg = True; self._cache_bg_pix = None
            else:
                sub = self.seed_fg[y0:y1, x0:x1]
                cv2.bitwise_or(sub, region, sub); self._dirty_fg = True; self._cache_fg_pix = None
            if self.wand_painted is not None:
                subw = self.wand_painted[y0:y1, x0:x1]
                cv2.bitwise_or(subw, region, subw)
                self._dirty_wand = True; self._cache_wand_pix = None; self._ants_dirty = True
            self.update()
            return

        lab_bi_full = self._get_bilateral(self.wand_tol, brush_ds, fast=False)
        if lab_bi_full is None:
            return
        region_ds = self._fullimage_flood_lab(lab_bi_full, (xs, ys), float(self.wand_tol), self.wand_global)
        if abs(s - 1.0) < 1e-3:
            region = region_ds
        else:
            H, W = self.lab.shape[:2]
            region = cv2.resize(region_ds, (W, H), interpolation=cv2.INTER_NEAREST)
        region = self._refine_with_grabcut(region)
        if self.wand_to_bg:
            self.seed_bg = cv2.bitwise_or(self.seed_bg, region); self._dirty_bg = True; self._cache_bg_pix = None
        else:
            self.seed_fg = cv2.bitwise_or(self.seed_fg, region); self._dirty_fg = True; self._cache_fg_pix = None
        if self.wand_painted is not None:
            self.wand_painted = cv2.bitwise_or(self.wand_painted, region)
            self._dirty_wand = True; self._cache_wand_pix = None; self._ants_dirty = True
        self.update()

    # ---------- Quick Select ----------
    def quick_select(self) -> bool:
        if self.image is None: return False
        self._push_history()
        H, W = self.image.shape[:2]
        gc = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
        if np.any(self.seed_bg): gc[self.seed_bg > 0] = cv2.GC_BGD
        if np.any(self.seed_fg): gc[self.seed_fg > 0] = cv2.GC_FGD
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(self.image, gc, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
            m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)*255
        except Exception:
            m = (self.seed_fg > 0).astype(np.uint8)*255
        ksz = max(3, int(min(H, W)/120) | 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 1)
        inv = 255 - m
        inv_pad = cv2.copyMakeBorder(inv, 1,1,1,1, cv2.BORDER_CONSTANT, value=255)
        ffmask = np.zeros((inv_pad.shape[0]+2, inv_pad.shape[1]+2), np.uint8)
        cv2.floodFill(inv_pad, ffmask, (0,0), 0)
        m = 255 - inv_pad[1:-1, 1:-1]
        self.mask_sel_temp = m
        self._dirty_sel = True; self._cache_sel_pix = None
        self.update(); return True

    # ---------- export helpers ----------
    def mask_to_polygons(self, mask: np.ndarray, min_area_frac: float = 0.002, max_points: int = 400) -> List[np.ndarray]:
        H, W = mask.shape[:2]
        min_area = max(32, int(min_area_frac * H * W))
        cnts, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys: List[np.ndarray] = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area: continue
            per = cv2.arcLength(c, True)
            eps = 0.002 * per
            approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if approx.shape[0] < 3: continue
            if approx.shape[0] > max_points:
                idx = np.linspace(0, approx.shape[0]-1, max_points).astype(int)
                approx = approx[idx]
            polys.append(approx.astype(np.int32))
        return polys

    def current_selection_mask(self) -> Optional[np.ndarray]:
        if self.mask_sel_temp is not None and np.any(self.mask_sel_temp):
            return ((self.mask_sel_temp > 0).astype(np.uint8) * 255)
        if self.wand_painted is not None and np.any(self.wand_painted):
            return ((self.wand_painted > 0).astype(np.uint8) * 255)
        if self.instances:
            H, W = self.image.shape[:2]
            acc = np.zeros((H, W), np.uint8)
            for m, _cid in self.instances:
                if m is not None and np.any(m):
                    cv2.bitwise_or(acc, (m > 0).astype(np.uint8) * 255, acc)
            if np.any(acc):
                return acc
        return None

# ===================== Class management =====================
DEFAULT_CLASSES = ["leaf"]

def _classes_file(labels_dir: str) -> str:
    return os.path.join(labels_dir, "classes.txt")

def load_classes(labels_dir: str) -> List[str]:
    p = _classes_file(labels_dir)
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                cls = [line.strip() for line in f if line.strip()]
            if cls: return cls
        except Exception: pass
    return DEFAULT_CLASSES.copy()

def save_classes(labels_dir: str, classes: list[str]):
    os.makedirs(labels_dir, exist_ok=True)
    data = "".join((str(c).strip()+"\n") for c in classes if str(c).strip())
    with open(_classes_file(labels_dir), "w", encoding="utf-8", newline="\n") as f:
        f.write(data)

def _color_for_class(cid: int) -> Tuple[int,int,int]:
    palette = [
        (239, 83, 80), (102, 187, 106), (66, 165, 245), (255, 202, 40),
        (171, 71, 188), (255, 112, 67), (38, 198, 218), (156, 204, 101),
    ]
    return palette[cid % len(palette)]

# ===================== Main Window =====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Judul awal; akan di-update oleh _check_license_at_start()
        self.setWindowTitle(f"{APP_NAME} – Annotation + Masking for YOLO Seg")
        ico_path = app_asset("logo_os_ai_corp.ico")
        if ico_path: self.setWindowIcon(QIcon(ico_path))
        self.resize(1520, 920)

        # State
        self.image_dir: str = ""
        self.labels_dir: str = ""
        self.classes: List[str] = DEFAULT_CLASSES.copy()
        self.active_class_id: int = 0
        self.min_area_frac: float = 0.0015

        # Export
        self.out_dir: str = ""
        self.train_pct: int = 80
        self.split_seed: int = 42
        self.make_yaml: bool = True

        # Widgets
        self.viewer = ImageViewer(self)
        self.list_images = QListWidget(self)
        self.list_images.itemSelectionChanged.connect(self._on_pick_item)

        # Left
        left = QVBoxLayout()
        left.addWidget(QLabel("Daftar Gambar"))
        left.addWidget(self.list_images, 1)
        btn_open_imgs = QPushButton("📁 Buka Folder Gambar")
        btn_open_imgs.clicked.connect(self._choose_image_folder)
        left.addWidget(btn_open_imgs)

        # Right
        right = QVBoxLayout()
        right.addWidget(self.viewer, 1)

        # Toolbar
        tools = QHBoxLayout()
        self.btn_fg = QPushButton("FG Brush [1]")
        self.btn_bg = QPushButton("BG Brush [2]")
        self.btn_wand = QPushButton("Magic Wand [3]")
        for b in (self.btn_fg, self.btn_bg, self.btn_wand): b.setCheckable(True)
        self.btn_fg.setChecked(True)
        self.btn_fg.clicked.connect(lambda: self._set_tool("fg"))
        self.btn_bg.clicked.connect(lambda: self._set_tool("bg"))
        self.btn_wand.clicked.connect(lambda: self._set_tool("wand"))
        self.brush_slider = QSlider(Qt.Horizontal); self.brush_slider.setRange(1,128); self.brush_slider.setValue(18)
        self.brush_slider.valueChanged.connect(lambda v: (self.viewer.set_brush(v), self._update_status()))
        self.wand_tol = QDoubleSpinBox(); self.wand_tol.setRange(0,60); self.wand_tol.setSingleStep(0.5); self.wand_tol.setValue(12.0)
        self.wand_tol.valueChanged.connect(lambda v: (self.viewer.set_wand_tol(v), self._update_status()))
        self.cb_wand_bg = QCheckBox("Wand→BG"); self.cb_wand_bg.stateChanged.connect(lambda st: self.viewer.set_wand_to_bg(st==Qt.Checked))
        self.cb_wand_global = QCheckBox("Global"); self.cb_wand_global.stateChanged.connect(lambda st: self.viewer.set_wand_global(st==Qt.Checked))
        tools.addWidget(self.btn_fg); tools.addWidget(self.btn_bg); tools.addWidget(self.btn_wand)
        tools.addWidget(QLabel("Brush")); tools.addWidget(self.brush_slider)
        tools.addWidget(QLabel("Wand tol")); tools.addWidget(self.wand_tol)
        tools.addWidget(self.cb_wand_bg); tools.addWidget(self.cb_wand_global)
        right.addLayout(tools)

        # Overlays
        overlays = QHBoxLayout()
        self.cb_show_instances  = QCheckBox("Show Instances [O]"); self.cb_show_instances.setChecked(True)
        self.cb_show_wand       = QCheckBox("Show Wand [W]");      self.cb_show_wand.setChecked(True)
        self.cb_show_seeds      = QCheckBox("Show Seeds [E]");     self.cb_show_seeds.setChecked(True)
        self.cb_show_selection  = QCheckBox("Show Selection [V]"); self.cb_show_selection.setChecked(True)
        self.cb_show_ants       = QCheckBox("Marching Ants [M]");  self.cb_show_ants.setChecked(True)
        self.cb_show_instances.toggled.connect(self.viewer.set_show_instances)
        self.cb_show_wand.toggled.connect(self.viewer.set_show_wand)
        self.cb_show_seeds.toggled.connect(self.viewer.set_show_seeds)
        self.cb_show_selection.toggled.connect(self.viewer.set_show_selection)
        self.cb_show_ants.toggled.connect(self.viewer.set_show_ants)
        for cb in (self.cb_show_instances, self.cb_show_wand, self.cb_show_seeds, self.cb_show_selection, self.cb_show_ants):
            overlays.addWidget(cb)
        overlays.addStretch(1)
        right.addLayout(overlays)

        # Classes
        clsrow = QHBoxLayout()
        self.cmb_class = QComboBox(); self.cmb_class.setEditable(True); self.cmb_class.addItems(self.classes)
        self.cmb_class.currentIndexChanged.connect(self._on_class_change)
        self.cmb_class.lineEdit().editingFinished.connect(self._on_class_edited)
        self.btn_add_inst = QPushButton("+ Add Instance (from selection)")
        self.btn_add_inst.clicked.connect(self._add_instance_from_selection)
        self.btn_del_inst = QPushButton("− Delete Instance")
        self.btn_del_inst.clicked.connect(self._delete_selected_instance)
        clsrow.addWidget(QLabel("Active Class:")); clsrow.addWidget(self.cmb_class, 2)
        clsrow.addWidget(self.btn_add_inst); clsrow.addWidget(self.btn_del_inst)
        right.addLayout(clsrow)

        # Instances
        instrow = QHBoxLayout()
        self.list_instances = QListWidget()
        instrow.addWidget(self.list_instances, 2)
        side = QVBoxLayout()
        btn_quick = QPushButton("⚡ Quick Select [R]"); btn_quick.clicked.connect(self._run_quick)
        btn_clear = QPushButton("🧹 Clear Seeds [C]"); btn_clear.clicked.connect(self._clear_seeds)
        btn_save = QPushButton("💾 Save [S]"); btn_save.clicked.connect(self._save_label)
        btn_savenext = QPushButton("💾 Save & Next [Shift+S]"); btn_savenext.clicked.connect(lambda: (self._save_label(), self._next_image()))
        side.addWidget(btn_quick); side.addWidget(btn_clear); side.addWidget(btn_save); side.addWidget(btn_savenext)
        side.addSpacing(8); side.addWidget(QLabel("Set Class → Selected Instance"))
        self.cmb_inst_class = QComboBox(); self.cmb_inst_class.setEditable(True); self.cmb_inst_class.addItems(self.classes)
        self.btn_apply_inst_class = QPushButton("Apply to Selected"); self.btn_apply_inst_class.clicked.connect(self._apply_class_to_selected_instance)
        side.addWidget(self.cmb_inst_class); side.addWidget(self.btn_apply_inst_class)
        side.addSpacing(8)
        btn_undo = QPushButton("↶ Undo (Ctrl+Z)"); btn_redo = QPushButton("↷ Redo (Ctrl+Y)")
        btn_undo.clicked.connect(self.viewer.undo); btn_redo.clicked.connect(self.viewer.redo)
        side.addWidget(btn_undo); side.addWidget(btn_redo)

        side.addSpacing(8)
        side.addWidget(QLabel("Export Binary Mask"))
        self.btn_mask_white = QPushButton("⬜ Sel=Putih (PNG)")
        self.btn_mask_white.clicked.connect(lambda: self._export_mask(True))
        self.btn_mask_black = QPushButton("⬛ Sel=Hitam (PNG)")
        self.btn_mask_black.clicked.connect(lambda: self._export_mask(False))
        side.addWidget(self.btn_mask_white)
        side.addWidget(self.btn_mask_black)

        side.addStretch(1)
        instrow.addLayout(side, 1)
        right.addLayout(instrow)

        # Paths
        pathrow = QHBoxLayout()
        self.ed_labels = QLineEdit("")
        btn_lbl = QPushButton("🏷️ Folder Labels"); btn_lbl.clicked.connect(self._choose_labels_folder)
        pathrow.addWidget(btn_lbl); pathrow.addWidget(self.ed_labels, 1)
        right.addLayout(pathrow)

        # Export
        exprow1 = QHBoxLayout()
        exprow1.addWidget(QLabel("📦 Output Dataset Dir:"))
        self.ed_outdir = QLineEdit("")
        btn_out = QPushButton("Pilih..."); btn_out.clicked.connect(self._choose_outdir)
        exprow1.addWidget(self.ed_outdir, 1); exprow1.addWidget(btn_out)
        right.addLayout(exprow1)

        exprow2 = QHBoxLayout()
        exprow2.addWidget(QLabel("Train %"))
        self.sb_train = QSpinBox(); self.sb_train.setRange(50,95); self.sb_train.setValue(80)
        exprow2.addWidget(self.sb_train); exprow2.addWidget(QLabel("Seed"))
        self.sb_seed = QSpinBox(); self.sb_seed.setRange(0,10000); self.sb_seed.setValue(42)
        exprow2.addWidget(self.sb_seed)
        self.cb_yaml = QCheckBox("Tulis data.yaml"); self.cb_yaml.setChecked(True)
        exprow2.addWidget(self.cb_yaml)
        self.btn_export = QPushButton("⇪ Export YOLO Dataset"); self.btn_export.clicked.connect(self._export_dataset)
        exprow2.addWidget(self.btn_export)
        right.addLayout(exprow2)

        # Info
        self.lbl_info = QLabel("Path: -")
        right.addWidget(self.lbl_info)

        # Root
        root = QHBoxLayout(); root.addLayout(left, 1); root.addLayout(right, 3)
        central = QWidget(); central.setLayout(root); self.setCentralWidget(central)

        # Status
        self.setStatusBar(QStatusBar()); self._update_status()

        # Menu Bantuan + About + Lisensi + Footer copyright
        self._setup_menubar()
        self._init_footer()
        self._check_license_at_start()

        # Shortcuts
        self._add_shortcut("1", lambda: self._set_tool("fg"))
        self._add_shortcut("2", lambda: self._set_tool("bg"))
        self._add_shortcut("3", lambda: self._set_tool("wand"))
        self._add_shortcut("R", self._run_quick)
        self._add_shortcut("S", self._save_label)
        self._add_shortcut("Shift+S", lambda: (self._save_label(), self._next_image()))
        self._add_shortcut("C", self._clear_seeds)
        self._add_shortcut("A", self._prev_image)
        self._add_shortcut("D", self._next_image)
        self._add_shortcut("Ctrl+0", self._fit)
        self._add_shortcut("[", lambda: self._adjust_brush(-2))
        self._add_shortcut("]", lambda: self._adjust_brush(+2))
        self._add_shortcut("O", self._toggle_instances)
        self._add_shortcut("W", self._toggle_wand)
        self._add_shortcut("E", self._toggle_seeds)
        self._add_shortcut("V", self._toggle_selection)
        self._add_shortcut("M", self._toggle_ants)
        self._add_shortcut("Ctrl+Z", self.viewer.undo)
        self._add_shortcut("Ctrl+Y", self.viewer.redo)
        self._add_shortcut("Ctrl+Shift+W", lambda: self._export_mask(True))
        self._add_shortcut("Ctrl+Shift+B", lambda: self._export_mask(False))
        self._add_shortcut("Ctrl+L", self._enter_license_key)   # lisensi cepat
        self._add_shortcut("Ctrl+Shift+F", self._copy_fingerprint)  # copy fingerprint

    # ------------ Menu, About, Lisensi ------------
    def _setup_menubar(self):
        if not self.menuBar():
            self.setMenuBar(QMenuBar())
        m_help = self.menuBar().addMenu("&Help")

        act_enter = QAction("Input License Key…", self)
        act_enter.setShortcut(QKeySequence("Ctrl+L"))
        act_enter.triggered.connect(self._enter_license_key)
        m_help.addAction(act_enter)

        # --- Tambahan: salin hardware fingerprint ---
        act_fp = QAction("Copy Hardware Fingerprint", self)
        act_fp.triggered.connect(self._copy_fingerprint)
        m_help.addAction(act_fp)

        m_help.addSeparator()

        act_about = QAction(f"About {APP_NAME}…", self)
        act_about.triggered.connect(self._show_about)
        m_help.addAction(act_about)

    def _init_footer(self):
        lbl = QLabel(COPYRIGHT_TXT)
        lbl.setStyleSheet("color:#bbb;")
        self.statusBar().addPermanentWidget(lbl)

    def _check_license_at_start(self):
        st = check_license()
        base = f"{APP_NAME} – Leaf Manual Labeler – YOLOseg Optimizer"
        if st.get("valid") and st.get("type") == "full":
            owner = st.get("name") or "Licensed"
            self.setWindowTitle(f"{base} — {owner}")
            self.statusBar().showMessage("License: Full")
            return
        if st.get("expired"):
            r = QMessageBox.question(self, "Lisensi", "Trial kedaluwarsa. Masukkan license key sekarang?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if r == QMessageBox.Yes and self._enter_license_key():
                return
            QMessageBox.warning(self, "Lisensi", "Aplikasi ditutup karena lisensi tidak valid/expired.")
            QApplication.instance().quit(); return
        days = int(st.get("remaining") or 0)
        self.setWindowTitle(f"{base} — Trial ({days} hari)")
        self.statusBar().showMessage(f"Trial mode: sisa {days} hari")

    def _enter_license_key(self) -> bool:
        key, ok = QInputDialog.getMultiLineText(self, "Masukkan License Key", "Tempel license key Anda:")
        if not ok or not key.strip(): return False
        ok2, msg = save_license_key(key.strip())
        if ok2:
            QMessageBox.information(self, "Lisensi", "Lisensi tersimpan. Terima kasih!")
            st = check_license()
            base = f"{APP_NAME} – Leaf Manual Labeler – YOLOseg Optimizer"
            if st.get("valid") and st.get("type") == "full":
                owner = st.get("name") or "Licensed"
                self.setWindowTitle(f"{base} — {owner}")
                self.statusBar().showMessage("License: Full")
            return True
        QMessageBox.warning(self, "Lisensi", f"Gagal menyimpan lisensi: {msg}")
        return False

    def _copy_fingerprint(self):
        fp = hardware_fingerprint()
        QApplication.clipboard().setText(fp)
        QMessageBox.information(self, "Hardware Fingerprint",
            f"Fingerprint berikut telah disalin ke clipboard:\n\n{fp}\n\n"
            "Kirim nilai ini ke penerbit untuk pembuatan license yang terkunci ke PC Anda.")

    def _show_about(self):
        st = check_license()
        if st.get("valid") and st.get("type") == "full":
            lic_line = "Lisensi: Full" + (f" (atas nama: {st.get('name')})" if st.get("name") else "")
        else:
            lic_line = "Lisensi: Trial"
            if st.get("expired"):
                lic_line += " (kedaluwarsa)"
            elif st.get("remaining") is not None:
                lic_line += f" — sisa {st.get('remaining')} hari"

        html = f"""
        <div style='min-width:360px'>
          <h2 style='margin:0'>{APP_NAME}</h2>
          <div>Versi {APP_VERSION}</div>
          <div>{APP_PUBLISHER}</div>
          <hr/>
          <div>{lic_line}</div>
          <div style='margin-top:6px'>{COPYRIGHT_TXT}</div>
          <div style='color:#888'>Smart Wand (edge-aware), Quick Select, Undo/Redo, Export YOLO.</div>
        </div>
        """
        box = QMessageBox(self)
        box.setWindowTitle(f"About {APP_NAME}")
        box.setIcon(QMessageBox.Information)
        box.setTextFormat(Qt.RichText)
        box.setText(html)
        logo = app_asset("logo_os_ai_corp.png") or app_asset("logo_os_ai_corp.ico")
        if logo and os.path.isfile(logo): box.setWindowIcon(QIcon(logo))
        box.exec()

    # ------------ helpers & actions ------------
    def _add_shortcut(self, key: str, fn):
        act = QAction(self); act.setShortcut(QKeySequence(key)); act.triggered.connect(fn); self.addAction(act)

    def _adjust_brush(self, delta: int):
        v = int(self.brush_slider.value()) + int(delta)
        v = max(self.brush_slider.minimum(), min(self.brush_slider.maximum(), v))
        self.brush_slider.setValue(v); self._update_status()

    def _fit(self):
        self.viewer.auto_fit = True; self.viewer.fit_to_window(); self.viewer.update(); self._update_status()

    def _update_status(self):
        self.statusBar().showMessage(
            f"Zoom: {self.viewer.view_scale*100:.1f}% | Tool: {self.viewer.tool} | "
            f"Class: {self.active_class_name()} | Brush: {self.brush_slider.value()} | "
            f"Tol: {self.wand_tol.value():.1f} | Global: {'ON' if self.cb_wand_global.isChecked() else 'OFF'}"
        )

    def _toggle_instances(self):
        self.cb_show_instances.setChecked(not self.cb_show_instances.isChecked())

    def _toggle_wand(self):
        self.cb_show_wand.setChecked(not self.cb_show_wand.isChecked())

    def _toggle_seeds(self):
        self.cb_show_seeds.setChecked(not self.cb_show_seeds.isChecked())

    def _toggle_selection(self):
        self.cb_show_selection.setChecked(not self.cb_show_selection.isChecked())

    def _toggle_ants(self):
        self.cb_show_ants.setChecked(not self.cb_show_ants.isChecked())

    def _choose_image_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pilih folder gambar", self.image_dir or ".")
        if not d: return
        self.image_dir = d; self._populate_images()

    def _choose_labels_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pilih folder labels", self.labels_dir or (self.image_dir or "."))
        if not d: return
        self.labels_dir = d; self.ed_labels.setText(d)
        self.classes = load_classes(self.labels_dir)
        self._refresh_classes_ui(); self.viewer.mark_instances_dirty()

    def _choose_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Pilih output dataset", self.out_dir or (self.image_dir or "."))
        if not d: return
        self.out_dir = d; self.ed_outdir.setText(d)

    def _populate_images(self):
        self.list_images.clear()
        if not self.image_dir or not os.path.isdir(self.image_dir): return
        files = list_images_in(self.image_dir)
        for f in files: self.list_images.addItem(QListWidgetItem(f))
        if files: self.list_images.setCurrentRow(0)

    def _on_pick_item(self):
        it = self.list_images.currentItem()
        if not it: return
        fn = it.text(); path = os.path.join(self.image_dir, fn)
        bgr = imread(path)
        if bgr is None:
            QMessageBox.warning(self, "Error", f"Gagal membaca gambar: {path}"); return
        self.viewer.set_image(bgr)
        if not self.labels_dir:
            self.labels_dir = os.path.join(os.path.dirname(self.image_dir), "labels")
            self.ed_labels.setText(self.labels_dir)
        self.classes = load_classes(self.labels_dir); self._refresh_classes_ui()
        self.lbl_info.setText(f"Path: {path}")
        self._load_existing_label(); self._update_status()

    def _refresh_classes_ui(self):
        shead = self.classes[:] if self.classes else ["leaf"]
        self.cmb_class.blockSignals(True); self.cmb_class.clear(); self.cmb_class.addItems(shead)
        self.cmb_class.setCurrentIndex(min(self.active_class_id, len(shead)-1)); self.cmb_class.blockSignals(False)
        self.cmb_inst_class.blockSignals(True); self.cmb_inst_class.clear(); self.cmb_inst_class.addItems(shead)
        self.cmb_inst_class.setCurrentIndex(min(self.active_class_id, len(shead)-1)); self.cmb_inst_class.blockSignals(False)

    def _on_class_change(self, idx: int):
        self.active_class_id = max(0, idx)
        save_classes(self.labels_dir or (self.image_dir and os.path.join(os.path.dirname(self.image_dir), "labels")) or ".", self.classes)
        self._update_status()

    def _on_class_edited(self):
        name = self.cmb_class.currentText().strip()
        if not name: return
        if name not in self.classes:
            self.classes.append(name)
            save_classes(self.labels_dir or (self.image_dir and os.path.join(os.path.dirname(self.image_dir), "labels")) or ".", self.classes)
            self._refresh_classes_ui(); self.cmb_class.setCurrentIndex(self.classes.index(name))
            self.viewer.mark_instances_dirty()

    def _set_tool(self, name: str):
        self.viewer.set_tool(name)
        self.btn_fg.setChecked(name=="fg"); self.btn_bg.setChecked(name=="bg"); self.btn_wand.setChecked(name=="wand")
        self._update_status()

    def _run_quick(self):
        ok = self.viewer.quick_select()
        if not ok:
            QMessageBox.information(self, "Info", "Tambahkan seed dengan brush atau gunakan Magic Wand lebih dulu.")
        else:
            self.statusBar().showMessage("Quick Select selesai — klik '+ Add Instance' untuk menambah instance")

    def _clear_seeds(self):
        self.viewer.clear_seeds_and_temp()

    def active_class_name(self) -> str:
        if not self.classes: return "class0"
        return self.classes[min(self.active_class_id, len(self.classes)-1)]

    def _add_instance_from_selection(self):
        if self.viewer.mask_sel_temp is None or not np.any(self.viewer.mask_sel_temp):
            QMessageBox.information(self, "Info", "Belum ada seleksi. Jalankan Quick Select dulu."); return
        self.viewer._push_history()
        m = (self.viewer.mask_sel_temp>0).astype(np.uint8)*255
        self.viewer.instances.append((m, self.active_class_id))
        self.viewer.mask_sel_temp[:] = 0
        self.viewer.seed_fg[:] = 0; self.viewer.seed_bg[:] = 0
        self.viewer._dirty_sel = True; self.viewer._cache_sel_pix = None
        self.viewer.mark_instances_dirty(); self._refresh_instances_list(); self.viewer.update()

    def _refresh_instances_list(self):
        self.list_instances.clear()
        for i,(m,cid) in enumerate(self.viewer.instances):
            px = int((m>0).sum()); name = self.classes[cid] if cid<len(self.classes) else f"class{cid}"
            self.list_instances.addItem(QListWidgetItem(f"[{i}] {name} — {px} px"))

    def _delete_selected_instance(self):
        r = self.list_instances.currentRow()
        if 0<=r<len(self.viewer.instances):
            self.viewer._push_history(); self.viewer.instances.pop(r)
            self.viewer.mark_instances_dirty(); self._refresh_instances_list(); self.viewer.update()

    def _apply_class_to_selected_instance(self):
        r = self.list_instances.currentRow()
        if r<0 or r>=len(self.viewer.instances):
            QMessageBox.information(self, "Info", "Pilih instance di daftar terlebih dulu."); return
        name = self.cmb_inst_class.currentText().strip()
        if not name: return
        if name not in self.classes:
            self.classes.append(name)
            save_classes(self.labels_dir or (self.image_dir and os.path.join(os.path.dirname(self.image_dir), "labels")) or ".", self.classes)
            self._refresh_classes_ui()
        new_cid = self.classes.index(name)
        m,_ = self.viewer.instances[r]; self.viewer.instances[r] = (m, new_cid)
        self.viewer.mark_instances_dirty(); self._refresh_instances_list(); self.viewer.update()

    def _next_image(self):
        r = self.list_images.currentRow()
        if r < self.list_images.count()-1: self.list_images.setCurrentRow(r+1)

    def _prev_image(self):
        r = self.list_images.currentRow()
        if r > 0: self.list_images.setCurrentRow(r-1)

    def _load_existing_label(self):
        it = self.list_images.currentItem()
        if not it or self.viewer.image is None: return
        txtp = os.path.join(self.labels_dir, os.path.splitext(it.text())[0]+".txt")
        H, W = self.viewer.image.shape[:2]
        self.viewer.instances.clear()
        if not os.path.isfile(txtp):
            self._refresh_instances_list(); self.viewer.mark_instances_dirty(); self.viewer.update(); return
        try:
            with open(txtp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    cid = int(parts[0]) if parts[0].isdigit() else 0
                    xy = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1,2)
                    xs = np.clip((xy[:,0]*W).round().astype(np.int32), 0, W-1)
                    ys = np.clip((xy[:,1]*H).round().astype(np.int32), 0, H-1)
                    pts = np.stack([xs,ys], axis=1)
                    mask = np.zeros((H,W), np.uint8); cv2.fillPoly(mask, [pts], 255)
                    self.viewer.instances.append((mask, cid))
        except Exception as e:
            print("label load error:", e)
        self._refresh_instances_list(); self.viewer.mark_instances_dirty(); self.viewer.update()

    def _save_label(self):
        if self.viewer.image is None: return
        if not self.labels_dir:
            self.labels_dir = os.path.join(os.path.dirname(self.image_dir), "labels"); self.ed_labels.setText(self.labels_dir)
        os.makedirs(self.labels_dir, exist_ok=True)
        save_classes(self.labels_dir, self.classes)
        it = self.list_images.currentItem()
        if not it: return
        base = os.path.splitext(it.text())[0]
        txtp = os.path.join(self.labels_dir, base+".txt")
        H, W = self.viewer.image.shape[:2]
        with open(txtp, "w", encoding="utf-8", newline="\n") as f:
            for (m, cid) in self.viewer.instances:
                polys = self.viewer.mask_to_polygons(m, self.min_area_frac, 400)
                for P in polys:
                    xs = np.clip(P[:,0]/float(W), 0.0, 1.0); ys = np.clip(P[:,1]/float(H), 0.0, 1.0)
                    coords = []
                    for x,y in zip(xs,ys): coords.extend([f"{x:.6f}", f"{y:.6f}"])
                    line = f"{cid} " + " ".join(coords); f.write(line+"\n")
        self.statusBar().showMessage(f"Saved: {txtp}")

    # ===================== EXPORT DATASET =====================
    def _export_dataset(self):
        if not self.image_dir or not os.path.isdir(self.image_dir):
            QMessageBox.warning(self, "Export", "Folder gambar belum dipilih."); return
        if not self.labels_dir or not os.path.isdir(self.labels_dir):
            res = QMessageBox.question(self, "Export",
                "Folder labels belum dipilih.\nTetap lanjut? (gambar tanpa label akan dibuatkan file label kosong)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if res != QMessageBox.Yes: return
        outdir = self.ed_outdir.text().strip() or self.out_dir
        if not outdir:
            QMessageBox.warning(self, "Export", "Output dataset dir belum diisi."); return
        self.train_pct = int(self.sb_train.value()); self.split_seed = int(self.sb_seed.value())
        self.make_yaml = bool(self.cb_yaml.isChecked())
        imgs = list_images_in(self.image_dir)
        if not imgs:
            QMessageBox.warning(self, "Export", "Tidak ada gambar di folder terpilih."); return
        random.seed(self.split_seed)
        imgs_shuf = imgs[:]; random.shuffle(imgs_shuf)
        n_train = max(1, int(len(imgs_shuf)*self.train_pct/100.0))
        train_list = imgs_shuf[:n_train]
        val_list = imgs_shuf[n_train:] or imgs_shuf[-max(1, len(imgs_shuf)//5):]
        img_train_dir = os.path.join(outdir, "images", "train")
        img_val_dir   = os.path.join(outdir, "images", "val")
        lbl_train_dir = os.path.join(outdir, "labels", "train")
        lbl_val_dir   = os.path.join(outdir, "labels", "val")
        for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]: os.makedirs(d, exist_ok=True)
        def _label_path_for(img_name: str) -> str:
            base = os.path.splitext(img_name)[0] + ".txt"
            return os.path.join(self.labels_dir, base)
        ok_train = ok_val = 0
        for name in train_list:
            src_img = os.path.join(self.image_dir, name)
            if copy_image(src_img, os.path.join(img_train_dir, name)): ok_train += 1
            copy_text(_label_path_for(name), os.path.join(lbl_train_dir, os.path.splitext(name)[0]+".txt"))
        for name in val_list:
            src_img = os.path.join(self.image_dir, name)
            if copy_image(src_img, os.path.join(img_val_dir, name)): ok_val += 1
            copy_text(_label_path_for(name), os.path.join(lbl_val_dir, os.path.splitext(name)[0]+".txt"))
        if self.classes:
            with open(os.path.join(outdir, "classes.txt"), "w", encoding="utf-8", newline="\n") as f:
                for c in self.classes: f.write(str(c).strip()+"\n")
        if self.make_yaml:
            yaml_path = os.path.join(outdir, "data.yaml")
            names_yaml = "[" + ", ".join(repr(c) for c in self.classes) + "]"
            content = ("path: .\ntrain: images/train\nval: images/val\n"
                       f"names: {names_yaml}\n"
                       f"nc: {len(self.classes)}\n"
                       "task: segment\n")
            with open(yaml_path, "w", encoding="utf-8", newline="\n") as f: f.write(content)
        self.statusBar().showMessage(f"Export selesai → {outdir} | train: {ok_train} img, val: {ok_val} img")
        QMessageBox.information(self, "Export",
            f"Dataset siap!\n\n- Train: {ok_train} gambar\n- Val: {ok_val} gambar\n\nFolder: {outdir}\n"
            f"{'data.yaml dibuat.' if self.make_yaml else ''}")

    def _export_mask(self, sel_is_white: bool):
        if self.viewer.image is None:
            QMessageBox.information(self, "Mask", "Tidak ada gambar yang aktif.")
            return
        m = self.viewer.current_selection_mask()
        if m is None or not np.any(m):
            QMessageBox.information(self, "Mask", "Tidak ada seleksi dari Magic Wand / Quick Select / Instances.")
            return
        mask_out = m.copy() if sel_is_white else (255 - m)
        mask_out = mask_out.astype(np.uint8)
        it = self.list_images.currentItem()
        base = os.path.splitext(it.text())[0] if it else "mask"
        default_dir = self.labels_dir or self.image_dir or "."
        os.makedirs(default_dir, exist_ok=True)
        default_name = f"{base}_mask.png"
        path, _ = QFileDialog.getSaveFileName(self, "Simpan Mask Binary", os.path.join(default_dir, default_name), "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if imwrite(path, mask_out):
            area_px = int((m > 0).sum())
            H, W = mask_out.shape[:2]
            frac = (area_px / float(H * W)) * 100.0
            mode = "Sel=Putih" if sel_is_white else "Sel=Hitam"
            self.statusBar().showMessage(
                f"Mask tersimpan: {path} | {mode} | area seleksi: {area_px} px ({frac:.2f}% dari gambar)"
            )
            QMessageBox.information(
                self, "Mask",
                f"Mask berhasil disimpan:\n{path}\n\n"
                f"Mode: {mode}\n"
                f"Area seleksi: {area_px} piksel ({frac:.2f}% dari gambar)"
            )
        else:
            QMessageBox.warning(self, "Mask", "Gagal menyimpan mask.")

    def _copy_fingerprint(self):
        try:
            # kalau modul lisensi tersedia
            from leaf_labeler.licensing.license_client import hardware_fingerprint
            fp = hardware_fingerprint()
        except Exception:
            fp = "N/A (module licensing not found)"
        QApplication.clipboard().setText(fp)
        QMessageBox.information(self, "Hardware Fingerprint",
                                f"Fingerprint disalin ke clipboard:\n\n{fp}")

# ===================== main =====================
def main():
    # 1) Taskbar grouping + ikon benar
    _set_win_app_user_model_id("OSAI.LeafLabeler")

    # 2) HiDPI icon
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtWidgets import QApplication as _QApp
    # Enable proper HiDPI scaling (Qt >= 6.6)
    if hasattr(_Qt, "AA_EnableHighDpiScaling"):
        _QApp.setAttribute(_Qt.AA_EnableHighDpiScaling, True)

    app = _QApp(sys.argv)

    # 3) Set application-level icon (berlaku ke semua window & taskbar)
    ico_path = app_asset("logo_os_ai_corp.ico") or app_asset("logo_os_ai_corp.png")
    if ico_path:
        app.setWindowIcon(QIcon(ico_path))

    w = MainWindow()
    # (opsional) set ulang untuk window spesifik
    if ico_path:
        w.setWindowIcon(QIcon(ico_path))

    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
