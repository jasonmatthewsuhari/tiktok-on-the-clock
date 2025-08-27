@echo off
setlocal enabledelayedexpansion

rem --- config, pls do not change :) ---
set "ENV_NAME=tiktok-env"
set "PY_VERSION=3.10"
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

rem Helper function for logging (defined as a subroutine at the end)
rem Usage: call :log "message"

rem Ensure conda exists, warn the user if it is not found.
rem if this fails here, pls follow: https://docs.anaconda.com/miniconda/install/
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda not found in PATH.
    echo Install Miniconda/Anaconda and run: conda init cmd.exe ^(then restart your shell^)
    exit /b 1
)

rem Create the conda env if it currently does not exist for the user
call conda shell.cmd.exe hook > "%TEMP%\conda_hook.bat"
call "%TEMP%\conda_hook.bat"
del "%TEMP%\conda_hook.bat"

conda env list | findstr /x "%ENV_NAME%" >nul 2>&1
if errorlevel 1 (
    call :log "Creating conda env '%ENV_NAME%' (Python %PY_VERSION%)..."
    conda create -y -n "%ENV_NAME%" "python=%PY_VERSION%"
) else (
    call :log "Conda env '%ENV_NAME%' already exists."
)

rem Activate only if not already active
if "%CONDA_DEFAULT_ENV%"=="%ENV_NAME%" (
    call :log "Already inside '%ENV_NAME%'; skipping activation."
) else (
    call :log "Activating '%ENV_NAME%'..."
    conda activate "%ENV_NAME%"
)

rem Upgrade pip toolchain & install pip-tools for the pipcompile tool :)
call :log "Upgrading pip/setuptools/wheel and installing pip-tools..."
python -m pip install --upgrade pip setuptools wheel pip-tools

rem Compile the reqs like usual
if exist "%PROJECT_ROOT%\requirements.in" (
    call :log "Compiling requirements.in -> requirements.txt ..."
    pushd "%PROJECT_ROOT%"
    python -m piptools compile requirements.in --resolver=backtracking
    popd
)

rem Install from requirements.txt AFTER compiling the reqs
if exist "%PROJECT_ROOT%\requirements.txt" (
    call :log "Installing from requirements.txt with pip-sync..."
    pushd "%PROJECT_ROOT%"
    python -m piptools sync requirements.txt
    popd
) else (
    call :log "No requirements.txt found; skipping install."
)

call :log "DONE. Active env: '%CONDA_DEFAULT_ENV%'"
goto :eof

rem Helper function for logging with timestamp
:log
for /f "tokens=1-3 delims=:." %%a in ('echo %time%') do (
    set "hour=%%a"
    set "minute=%%b"
    set "second=%%c"
)
rem Remove leading spaces from hour
set "hour=!hour: =!"
echo.
echo [!hour!:!minute!:!second!] %~1
goto :eof
