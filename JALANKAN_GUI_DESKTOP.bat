@echo off
title GUI Desktop Crop Gambar
cd /d "%~dp0"

echo =======================================================
echo    MEMULAI GUI DESKTOP CROP OTOMATIS GAMBAR
echo =======================================================
echo.

python gui_crop.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Terjadi kesalahan saat menjalankan GUI Desktop.
    pause
)
