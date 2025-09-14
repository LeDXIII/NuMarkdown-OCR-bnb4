@echo off

chcp 65001 >nul

echo ==========================================

echo NuMarkdown BNB4 OCR (Windows) - Universal Installer

echo ==========================================

:: 0. Check for Python

echo [0/5] Checking Python installation...

python --version >nul 2>&1

if %errorlevel% neq 0 (
    echo ERROR: Python not found.
    echo Please install Python 3.8 or later from https://www.python.org/downloads/
    echo Make sure Python is added to your PATH environment variable.
    pause
    exit /b 1
)

:: Get Python version string (e.g., "Python 3.10.11")
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v

:: Extract major and minor versions
for /f "tokens=1,2 delims=." %%m in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%m
    set PYTHON_MINOR=%%n
)

:: Check version: must be >= 3.8
if %PYTHON_MAJOR% LSS 3 (
    echo ERROR: Python 3.8 or later is required. Your version is %PYTHON_VERSION%.
    pause
    exit /b 1
)
if %PYTHON_MAJOR% EQU 3 (
    if %PYTHON_MINOR% LSS 8 (
        echo ERROR: Python 3.8 or later is required. Your version is %PYTHON_VERSION%.
        pause
        exit /b 1
    )
)
echo Compatible Python version %PYTHON_VERSION% detected.

:: 1. Create virtual environment
echo [1/5] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: 2. Upgrade pip
echo [2/5] Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

:: 3. Detect CUDA and install PyTorch FOR CUDA ONLY (CPU option disabled)
echo [3/5] Detecting CUDA and installing PyTorch...

:: Check if nvidia-smi is available (NVIDIA GPU present)
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: NVIDIA GPU not detected. CUDA support required for this project.
    echo Please install NVIDIA drivers and ensure nvidia-smi is available in PATH.
    pause
    exit /b 1
)

:: Try to get NVIDIA driver version using more compatible command
set DRIVER_VERSION=
for /f "tokens=3" %%i in ('nvidia-smi 2^>nul ^| findstr /C:"Driver Version"') do set DRIVER_VERSION=%%i
echo DEBUG: DRIVER_VERSION=[%DRIVER_VERSION%]

if not defined DRIVER_VERSION (
    echo ERROR: Could not read NVIDIA driver version.
    echo Please ensure NVIDIA drivers are properly installed.
    pause
    exit /b 1
)

echo Detected NVIDIA Driver Version: %DRIVER_VERSION%

:: Determine compatible CUDA Toolkit version with ACTUAL available PyTorch versions
if "%DRIVER_VERSION%" LSS "525.60.13" (
    set CUDA_VERSION=cu118
    set TORCH_VERSION=2.6.0+cu118
    set TORCHVISION_VERSION=0.21.0+cu118
    set TORCHAUDIO_VERSION=2.6.0+cu118
    set PYTORCH_INDEX=https://download.pytorch.org/whl/cu118
    echo Selected CUDA version: 11.8
) else if "%DRIVER_VERSION%" LSS "535.104.05" (
    set CUDA_VERSION=cu121
    set TORCH_VERSION=2.6.0+cu121
    set TORCHVISION_VERSION=0.21.0+cu121
    set TORCHAUDIO_VERSION=2.6.0+cu121
    set PYTORCH_INDEX=https://download.pytorch.org/whl/cu121
    echo Selected CUDA version: 12.1
) else (
    set CUDA_VERSION=cu124
    set TORCH_VERSION=2.6.0+cu124
    set TORCHVISION_VERSION=0.21.0+cu124
    set TORCHAUDIO_VERSION=2.6.0+cu124
    set PYTORCH_INDEX=https://download.pytorch.org/whl/cu124
    echo Selected CUDA version: 12.4
)

echo Installing PyTorch for CUDA from: %PYTORCH_INDEX%
pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url %PYTORCH_INDEX%

:: 4. Install PySide6
echo [4/5] Installing PySide6...
pip install PySide6>=6.5.0

:: 5. Install other dependencies
echo [5/5] Installing other dependencies...

:: First, install compatible versions of conflicting packages
pip install "dill>=0.3.0,<0.3.9" "fsspec[http]>=2023.1.0,<=2025.3.0" "multiprocess<0.70.17" "tzdata>=2022.7"

:: Now install the rest of the dependencies without recursive deps to avoid breaking PyTorch
pip install --no-deps -r requirements.txt

:: Install xformers and optimum separately
pip install xformers>=0.0.26 --no-deps
pip install optimum

:: Install qwen-vl-utils
pip install qwen-vl-utils

:: Final message
echo.
echo Installation completed successfully!
echo To run the application, execute: run.bat
echo.

pause
