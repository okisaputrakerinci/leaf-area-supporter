@echo off
setlocal
set QT_OPENGL=angle
REM (opsional) pilih backend ANGLE:
REM set QT_ANGLE_PLATFORM=d3d11
REM set QT_ANGLE_PLATFORM=warp

cd /d "D:\leaf-area\Model-Latih-YOLO"
".\.venv311\Scripts\python.exe" gui_labeler.py
