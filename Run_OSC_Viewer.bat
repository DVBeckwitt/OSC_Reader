@echo off
setlocal

REM Always run from the project directory so local source works too.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "PY_CMD="
where py >nul 2>&1
if not errorlevel 1 set "PY_CMD=py"
if not defined PY_CMD (
    where python >nul 2>&1
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo Python was not found on PATH.
    echo Install Python 3 and ensure it is available as "py" or "python".
    pause
    popd >nul
    exit /b 1
)

if "%~1"=="" (
    %PY_CMD% -m OSC_Reader.OSC_Viewer
) else (
    %PY_CMD% -m OSC_Reader.OSC_Viewer "%~1"
)

set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
