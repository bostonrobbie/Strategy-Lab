@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   MARCUS AUTONOMOUS AGENT - Windows Service Installer
echo   Installs Marcus as a 24/7 Windows Service via NSSM
echo ============================================================
echo.

:: === Configuration ===
set SERVICE_NAME=MarcusAgent
set PYTHON_SCRIPT=%~dp0src\backtesting\marcus_daemon.py
set WORKING_DIR=%~dp0src\backtesting
set LOG_DIR=%~dp0..\Marcus_Research\logs

:: === Find Python ===
echo [1/5] Locating Python...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH.
    echo         Please install Python 3.9+ and add to PATH.
    goto :error
)
for /f "tokens=*" %%i in ('where python') do (
    set PYTHON_EXE=%%i
    goto :found_python
)
:found_python
echo       Found: %PYTHON_EXE%

:: Verify Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
echo       Version: %PYTHON_VER%
echo.

:: === Find or Download NSSM ===
echo [2/5] Checking for NSSM (Non-Sucking Service Manager)...
where nssm >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%i in ('where nssm') do set NSSM=%%i
    echo       Found: !NSSM!
) else (
    echo       NSSM not found in PATH.
    echo       Please install NSSM:
    echo         1. Download from https://nssm.cc/download
    echo         2. Extract nssm.exe to a folder in your PATH
    echo         3. Or place nssm.exe next to this script
    echo.

    :: Check if nssm.exe is in script directory
    if exist "%~dp0nssm.exe" (
        set NSSM=%~dp0nssm.exe
        echo       Found local: !NSSM!
    ) else (
        echo [ERROR] NSSM not available. Cannot install service.
        goto :error
    )
)
echo.

:: === Create log directory ===
echo [3/5] Creating directories...
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
echo       Logs: %LOG_DIR%
echo.

:: === Check if service already exists ===
echo [4/5] Checking existing service...
sc query %SERVICE_NAME% >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo       Service '%SERVICE_NAME%' already exists.
    echo       Stopping and removing old service...
    net stop %SERVICE_NAME% >nul 2>&1
    "%NSSM%" remove %SERVICE_NAME% confirm >nul 2>&1
    echo       Old service removed.
)
echo.

:: === Install Service ===
echo [5/5] Installing Marcus service...

"%NSSM%" install %SERVICE_NAME% "%PYTHON_EXE%" "%PYTHON_SCRIPT%"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install service. Try running as Administrator.
    goto :error
)

:: Configure service parameters
"%NSSM%" set %SERVICE_NAME% AppDirectory "%WORKING_DIR%"
"%NSSM%" set %SERVICE_NAME% DisplayName "Marcus Autonomous Research Agent"
"%NSSM%" set %SERVICE_NAME% Description "24/7 autonomous quant research agent for NQ futures strategy discovery"
"%NSSM%" set %SERVICE_NAME% Start SERVICE_AUTO_START

:: Logging
"%NSSM%" set %SERVICE_NAME% AppStdout "%LOG_DIR%\marcus_stdout.log"
"%NSSM%" set %SERVICE_NAME% AppStderr "%LOG_DIR%\marcus_stderr.log"
"%NSSM%" set %SERVICE_NAME% AppStdoutCreationDisposition 4
"%NSSM%" set %SERVICE_NAME% AppStderrCreationDisposition 4
"%NSSM%" set %SERVICE_NAME% AppRotateFiles 1
"%NSSM%" set %SERVICE_NAME% AppRotateOnline 1
"%NSSM%" set %SERVICE_NAME% AppRotateBytes 10485760

:: Restart on failure
"%NSSM%" set %SERVICE_NAME% AppRestartDelay 30000
"%NSSM%" set %SERVICE_NAME% AppThrottle 60000

:: Environment
"%NSSM%" set %SERVICE_NAME% AppEnvironmentExtra PYTHONUNBUFFERED=1

echo.
echo ============================================================
echo   INSTALLATION COMPLETE
echo ============================================================
echo.
echo   Service Name:  %SERVICE_NAME%
echo   Python:        %PYTHON_EXE%
echo   Script:        %PYTHON_SCRIPT%
echo   Log Dir:       %LOG_DIR%
echo.
echo   Commands:
echo     Start:   net start %SERVICE_NAME%
echo     Stop:    net stop %SERVICE_NAME%
echo     Status:  sc query %SERVICE_NAME%
echo     Remove:  nssm remove %SERVICE_NAME% confirm
echo     Edit:    nssm edit %SERVICE_NAME%
echo.
echo   Quick Test (single cycle, no service):
echo     python "%PYTHON_SCRIPT%" --once
echo.
echo   Dashboard Location:
echo     %~dp0..\Marcus_Research\dashboard\marcus_live.html
echo.
echo ============================================================
goto :end

:error
echo.
echo Installation failed. See errors above.
echo.

:end
endlocal
pause
