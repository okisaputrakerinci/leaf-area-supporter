# Runtime hook: paksa Qt pakai ANGLE (Direct3D) supaya rendering ringan.
import os
os.environ.setdefault("QT_OPENGL", "angle")
# Jika GPU bermasalah, aktifkan WARP:
# os.environ.setdefault("QT_ANGLE_PLATFORM", "warp")
