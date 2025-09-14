@echo off
setlocal

:: Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please run install.bat first.
    pause
    exit /b 1
)

:: Run the main Python script
python gui.py

pause