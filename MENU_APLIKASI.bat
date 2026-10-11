@echo off
title Launcher Crop Otomatis Box Gambar
cd /d "%~dp0"

:MENU
cls
echo =======================================================
echo     PILIH MODE APLIKASI CROP OTOMATIS GAMBAR
echo =======================================================
echo.
echo  [1] Jendela Desktop Mandiri (App Mode - Tanpa Tab/URL) [TERBAIK]
echo  [2] Web App (Buka di Tab Browser - http://localhost:5050)
echo  [3] GUI Tkinter Klasik (Desktop Sederhana)
echo  [4] Install / Update Library (requirements.txt)
echo  [5] Keluar
echo.
echo =======================================================
set /p pilihan="Masukkan nomor pilihan (1/2/3/4/5): "

if "%pilihan%"=="1" goto RUN_DESKTOP_APP
if "%pilihan%"=="2" goto RUN_WEB
if "%pilihan%"=="3" goto RUN_GUI
if "%pilihan%"=="4" goto INSTALL_REQ
if "%pilihan%"=="5" goto KELUAR

echo Pilihan tidak valid. Silakan coba lagi.
timeout /t 2 >nul
goto MENU

:RUN_DESKTOP_APP
call JALANKAN_DESKTOP_APP.bat
goto MENU

:RUN_WEB
cls
echo Membuka browser di http://localhost:5050 ...
start "" "http://localhost:5050"
python app.py
pause
goto MENU

:RUN_GUI
cls
echo Menjalankan GUI Tkinter...
python gui_crop.py
pause
goto MENU

:INSTALL_REQ
cls
echo Memeriksa dan menginstall dependensi...
pip install -r requirements.txt
echo.
echo Selesai!
pause
goto MENU

:KELUAR
exit
