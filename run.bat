@echo off
SETLOCAL EnableDelayedExpansion

echo ==============================================================================
echo  Typhon Intrusion Detection System (IDS) — Windows Launcher
echo ==============================================================================

set VENV_DIR=venv

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    py -0 >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python 3.9+ is not found in PATH. Please install Python.
        pause
        exit /b 1
    )
    set PY_CMD=py
) else (
    set PY_CMD=python
)

:: Create virtual environment if missing
if not exist "%VENV_DIR%" (
    echo [INFO] Creating Python virtual environment in %VENV_DIR%...
    %PY_CMD% -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    
    echo [INFO] Installing required packages from requirements.txt...
    call %VENV_DIR%\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo [INFO] Environment successfully prepared.
) else (
    call %VENV_DIR%\Scripts\activate.bat
)

:: Run Detection on sample data or passed arguments
if "%~1"=="" (
    echo [INFO] Running baseline detection on sample flows (data/sample_flows.csv)...
    python main.py --mode detect --file data/sample_flows.csv
) else (
    echo [INFO] Executing Typhon with custom arguments: %*
    python main.py %*
)

if %errorlevel% neq 0 (
    echo [ERROR] Typhon execution returned an error.
) else (
    echo [SUCCESS] Execution finished successfully. Check outputs/ for generated reports.
)

pause
