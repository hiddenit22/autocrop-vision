@echo off
title Web App Crop Gambar - Flask Server
cd /d "%~dp0"

echo =======================================================
echo    MEMULAI SERVER WEB APP CROP OTOMATIS GAMBAR
echo =======================================================
echo.
echo Direktori: %CD%
echo Membuka browser di http://localhost:5050 ...
echo.

start "" "http://localhost:5050"
python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Terjadi kesalahan saat menjalankan server.
    pause
)
