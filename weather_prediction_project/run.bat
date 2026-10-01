@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================
echo Weather Prediction Project
echo Working directory: %CD%
echo ============================================

where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python was not found on PATH.
        echo Activate your Python or conda environment and try again.
        pause
        exit /b 1
    ) else (
        set "PY=py -3"
    )
) else (
    set "PY=python"
)

echo Using Python:
%PY% --version
if errorlevel 1 (
    echo [ERROR] Could not run Python.
    pause
    exit /b 1
)

echo.
echo [1/3] Checking dependencies...
%PY% -c "import numpy, pandas, sklearn, matplotlib, PIL, joblib" >nul 2>nul
if errorlevel 1 (
    echo Missing dependencies detected. Installing from requirements.txt...
    %PY% -m pip install --disable-pip-version-check -r "requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
) else (
    echo Dependencies are already installed.
)

if /i "%~1"=="--check" (
    echo Dependency check completed.
    exit /b 0
)

echo.
echo [2/3] Running experiment...
%PY% run_experiment.py
if errorlevel 1 (
    echo [ERROR] The experiment failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Finished. Results are in the outputs folder.
pause
exit /b 0
