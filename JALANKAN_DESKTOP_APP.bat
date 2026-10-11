@echo off
title Crop Otomatis Box Gambar (Desktop App)
cd /d "%~dp0"

echo =======================================================
echo    MEMULAI APLIKASI DESKTOP CROP OTOMATIS GAMBAR
echo =======================================================
echo.
echo Menyiapkan backend server dan membuka jendela aplikasi...
echo.

:: 1. Cek & jalankan server background jika belum aktif di port 5050
powershell -Command "if (!(Get-NetTCPConnection -LocalPort 5050 -State Listen -ErrorAction SilentlyContinue)) { Start-Process -WindowStyle Hidden python -ArgumentList 'app.py' }"
timeout /t 2 /nobreak >nul

:: 2. Buka jendela aplikasi desktop mandiri (App Mode tanpa address bar / tabs)
set "EDGE_1=%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"
set "EDGE_2=%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"
set "CHROME_1=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
set "CHROME_2=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"

if exist "%EDGE_1%" (
    start "" "%EDGE_1%" --app="http://localhost:5050"
    goto SELESAI
)
if exist "%EDGE_2%" (
    start "" "%EDGE_2%" --app="http://localhost:5050"
    goto SELESAI
)
if exist "%CHROME_1%" (
    start "" "%CHROME_1%" --app="http://localhost:5050"
    goto SELESAI
)
if exist "%CHROME_2%" (
    start "" "%CHROME_2%" --app="http://localhost:5050"
    goto SELESAI
)

:: Fallback jika path tidak ditemukan di lokasi standar
start msedge --app="http://localhost:5050" 2>nul || start chrome --app="http://localhost:5050" 2>nul || start "" "http://localhost:5050"

:SELESAI
exit
