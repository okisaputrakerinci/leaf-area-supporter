# LeafLabeler — Manual Segmentation & YOLO Dataset Builder

> Fast manual labeling with **Smart Wand (edge-aware)**, **Quick Select (GrabCut)**, multi-class instances, and **one-click YOLO dataset export**.

![status-badge](https://img.shields.io/badge/status-active-success) ![python](https://img.shields.io/badge/Python-3.10%2B-blue) ![qt](https://img.shields.io/badge/Qt-PySide6-brightgreen) ![opencv](https://img.shields.io/badge/OpenCV-4.8%2B-orange)

## ✨ Fitur Utama
- **Smart Wand**: seleksi cepat berbasis CIE-Lab + bilateral smoothing (edge-aware).
- **Quick Select**: seed FG/BG → GrabCut refinement.
- **Instance & Classes**: multi-instance per gambar + kelas dinamis (`classes.txt`).
- **Overlay kaya**: marching-ants, seeds, wand paint, instances (toggle cepat).
- **Export YOLO** (segmentasi): auto split **train/val**, tulis **`data.yaml`**, salin **images/labels**.
- **Undo/Redo**, zoom, pan, brush size & tolerance on the fly.
- **Lisensi**: dukung trial & full license (dengan hardware fingerprint).

## 🖥️ Persyaratan
- Windows 10/11 (disarankan), Python **3.10–3.11**
- GPU opsional (untuk inference lain; labeling berjalan di CPU)
- Dependensi utama: `opencv-contrib-python`, `PySide6`, `numpy`, `matplotlib`

## 🚀 Instalasi
```bash
# 1) Clone
git clone https://github.com/<username>/leaf-area-project.git
cd leaf-area-project

# 2) Buat venv & install deps
python -m venv .venv
.venv\Scripts\activate     # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
