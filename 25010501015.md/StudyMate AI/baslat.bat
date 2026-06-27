@echo off
title StudyMate AI Baslatici
echo ===================================================
echo             StudyMate AI Baslatici
echo ===================================================
echo.

cd /d "%~dp0"

:: 1. Python Kontrolu
python --version >nul 2>&1
if %errorlevel% equ 0 goto :python_ok

echo [HATA] Bilgisayarinizda Python yuklu bulunamadi!
echo.
echo Bu sistemi calistirmak icin bilgisayarinizda Python yuklu olmalidir.
echo Lutfen asagidaki adimlari takip edin:
echo 1. https://www.python.org/downloads/ adresinden Python indirin.
echo 2. Kurulumu baslatin.
echo 3. Kurulum ekraninin altindaki Add Python to PATH secenegini mutlaka isaretleyin!
echo 4. Kurulum bittikten sonra bu dosyayi baslat.bat tekrar acin.
echo.
pause
exit /b

:python_ok

:: 2. Sanal Ortam Gecerlilik Kontrolu
if not exist .venv\Scripts\python.exe goto :setup_venv

.venv\Scripts\python.exe -c "import streamlit" >nul 2>&1
if %errorlevel% equ 0 goto :start_system

:setup_venv
echo.
echo [BILGI] Sanal ortam kuruluyor veya yenileniyor...
echo Bu islem ilk calistirmada veya baska bilgisayara tasindiginda yapilir, 1-2 dakika surebilir...
echo.

if not exist .venv goto :create_venv
echo Eski sanal ortam dosyalari temizleniyor...
rd /s /q .venv

:create_venv
echo Yeni sanal ortam olusturuluyor...
python -m venv .venv
if %errorlevel% neq 0 goto :venv_error

echo Gerekli kutuphaneler yukleniyor...
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
if %errorlevel% neq 0 goto :pip_error

echo [BASARILI] Kurulum tamamlandi!
echo.

:start_system
:: 3. Ortam degiskenlerini ata
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set USE_HASH_EMBEDDINGS=1
set STUDYMATE_DATA_DIR=%CD%\data

:: 4. Tarayiciyi otomatik ac
start http://localhost:8501

echo.
echo ===================================================
echo [BASARILI] Sistem Baslatildi!
echo ===================================================
echo Tarayiciniz otomatik olarak acilacaktir...
echo Eger acilmazsa tarayicinizin adres cubuguna su adresi yazin:
echo http://localhost:8501
echo.
echo Sistemi kapatmak icin bu siyah pencereyi kapatabilirsiniz.
echo ===================================================
echo.

".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.fileWatcherType none
pause
exit /b

:venv_error
echo [HATA] Sanal ortam olusturulamadi!
pause
exit /b

:pip_error
echo [HATA] Kutuphaneler yuklenirken hata olustu!
pause
exit /b
