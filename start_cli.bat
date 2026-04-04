@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "SHOULD_PAUSE=1"
set "EXIT_CODE=0"
set "VENV_PY=.venv\Scripts\python.exe"
set "VENV_ACTIVATE=.venv\Scripts\activate.bat"
set "PY_CMD="
set "PY_LABEL="

if exist "%VENV_PY%" (
    echo Reusing existing virtual environment...
    goto :activate_venv
)

call :select_python
if errorlevel 1 (
    set "EXIT_CODE=1"
    goto :end
)

echo Creating virtual environment with %PY_LABEL%...
call %PY_CMD% -m venv .venv
if errorlevel 1 (
    echo [FAIL] Failed to create virtual environment.
    set "EXIT_CODE=1"
    goto :end
)

:activate_venv
call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo [FAIL] Failed to activate virtual environment.
    set "EXIT_CODE=1"
    goto :end
)

for /f %%v in ('python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul') do set "ACTIVE_PY_VERSION=%%v"
if "%ACTIVE_PY_VERSION%"=="3.11" (
    echo [ OK ] Active environment is using Python 3.11.
) else if "%ACTIVE_PY_VERSION%"=="3.12" (
    echo [WARN] Active environment is using Python 3.12.
    echo        This is supported on a best-effort basis. Python 3.11 remains the preferred version.
) else (
    echo [FAIL] Active environment is using unsupported Python %ACTIVE_PY_VERSION%.
    echo        Please recreate the environment with Python 3.11 or 3.12.
    set "EXIT_CODE=1"
    goto :end
)

python -m pip install --upgrade pip
if errorlevel 1 (
    echo [FAIL] Failed to upgrade pip.
    set "EXIT_CODE=1"
    goto :end
)

pip install -r requirements.txt
if errorlevel 1 (
    echo [FAIL] Failed to install requirements.
    echo        Run check_env.bat for a clearer diagnosis.
    set "EXIT_CODE=1"
    goto :end
)

call check_env.bat --post-install
if errorlevel 1 (
    echo [FAIL] Environment verification failed. The CLI will not be started.
    set "EXIT_CODE=1"
    goto :end
)

python main.py
set "EXIT_CODE=%ERRORLEVEL%"

:end
if not "%EXIT_CODE%"=="0" echo.
if "%SHOULD_PAUSE%"=="1" pause
exit /b %EXIT_CODE%

:select_python
where py >nul 2>nul
if not errorlevel 1 (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PY_CMD=py -3.11"
        set "PY_LABEL=Python 3.11"
        exit /b 0
    )
)

where python >nul 2>nul
if errorlevel 1 (
    echo [FAIL] Python was not found on PATH.
    echo        Install Python 3.11 or 3.12 and try again.
    exit /b 1
)

for /f %%v in ('python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2^>nul') do set "SYS_PY_VERSION=%%v"
if "%SYS_PY_VERSION%"=="3.11" (
    set "PY_CMD=python"
    set "PY_LABEL=Python 3.11"
    exit /b 0
)
if "%SYS_PY_VERSION%"=="3.12" (
    set "PY_CMD=python"
    set "PY_LABEL=Python 3.12"
    echo [WARN] Python 3.11 was not found. Falling back to Python 3.12.
    echo        This is supported on a best-effort basis.
    exit /b 0
)

echo [FAIL] Detected Python %SYS_PY_VERSION% on PATH.
echo        Supported versions for this launcher are Python 3.11 and 3.12.
exit /b 1
