@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "MODE=%~1"
set "SHOULD_PAUSE=1"
if "%MODE%"=="--post-install" set "SHOULD_PAUSE=0"

set "EXIT_CODE=0"
set "VENV_PY=.venv\Scripts\python.exe"
set "PY_CMD="
set "PY_LABEL="
set "PY_VERSION="
set "IMPORT_FAILURE="

echo === AI Gold Research System Environment Check ===
echo.

if exist "%VENV_PY%" (
    set "PY_CMD=%VENV_PY%"
    set "PY_LABEL=existing virtual environment"
    goto :have_python
)

echo [INFO] No local .venv found yet. Checking the best available interpreter on this machine.
echo.

where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PY_CMD=py -3.11"
        set "PY_LABEL=Python 3.11 via py launcher"
        goto :have_python
    )
)

where python >nul 2>nul
if errorlevel 1 (
    echo [FAIL] Python was not found on PATH.
    echo        Install Python 3.11 or 3.12 and try again.
    set "EXIT_CODE=1"
    goto :end
)

set "PY_CMD=python"
set "PY_LABEL=python from PATH"

:have_python
for /f %%v in ('%PY_CMD% -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul') do set "PY_VERSION=%%v"
if not defined PY_VERSION (
    echo [FAIL] Failed to query the selected Python interpreter.
    set "EXIT_CODE=1"
    goto :end
)

echo [ OK ] Selected interpreter: %PY_LABEL%
echo [ OK ] Python version: %PY_VERSION%

if "%PY_VERSION%"=="3.11" (
    echo [ OK ] Python 3.11 is the preferred version for this project.
) else if "%PY_VERSION%"=="3.12" (
    echo [WARN] Python 3.12 is supported on a best-effort basis.
    echo        Python 3.11 remains the preferred version for the smoothest setup.
) else (
    echo [FAIL] Python %PY_VERSION% is not supported by this launcher.
    echo        Use Python 3.11 or 3.12.
    set "EXIT_CODE=1"
    goto :end
)

call :check_import streamlit "GUI"
call :check_import talib "feature engineering"
call :check_import MetaTrader5 "Exness broker features"
call :check_import pandas_ta "feature engineering"
call :check_import xgboost "ensemble model training"

echo.
if defined IMPORT_FAILURE (
    echo [FAIL] One or more required imports failed.
    if "%MODE%"=="--post-install" (
        echo        Fresh environment setup is incomplete.
    ) else (
        echo        This can be normal before the first install. Try running start.bat to create or repair the environment.
    )
    set "EXIT_CODE=1"
    goto :end
)

echo Environment looks ready for the current app workflow.

:end
if "%SHOULD_PAUSE%"=="1" pause
exit /b %EXIT_CODE%

:check_import
%PY_CMD% -c "import %~1" >nul 2>nul
if errorlevel 1 (
    echo [FAIL] %~1 import failed. %~2 will not work correctly.
    set "IMPORT_FAILURE=1"
) else (
    echo [ OK ] %~1 import succeeded
)
exit /b 0
