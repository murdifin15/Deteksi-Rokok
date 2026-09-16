@echo off
title Live Webcam Deteksi Rokok (YOLO11)
cls
echo =========================================================
echo       APLIKASI DETEKSI ROKOK REAL-TIME (YOLO11)
echo =========================================================
echo.
echo Kamera yang terdeteksi:
echo   [1] Webcam Eksternal (V380 FHD Camera) - REKOMENDASI
echo   [0] Webcam Internal Laptop (HD Webcam)
echo.
set /p cam_choice="Pilih nomor kamera [Tekan ENTER untuk Webcam Eksternal (1)]: "
if "%cam_choice%"=="" set cam_choice=1

echo.
echo Menjalankan aplikasi dengan Kamera Source %cam_choice% ...
echo =========================================================
python detect_webcam.py --source %cam_choice%
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Terjadi kesalahan saat menjalankan aplikasi.
    pause
)
