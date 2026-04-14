@echo off
setlocal EnableExtensions

REM Always run from the project directory so local source works too.
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "PY_CMD="
call :resolve_python

if not defined PY_CMD (
    echo Python was not found.
    echo Checked PATH, local virtual environments, the Windows launcher,
    echo and common per-user install locations.
    echo Install Python 3, or update this script to point at an existing python.exe.
    pause
    popd >nul
    exit /b 1
)

if "%~1"=="" (
    call :run_python -m OSC_Reader.OSC_Viewer
) else (
    call :run_python -m OSC_Reader.OSC_Viewer "%~1"
)

set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%

:resolve_python
if defined PYTHON if exist "%PYTHON%" (
    set "PY_CMD=%PYTHON%"
    exit /b 0
)

for %%P in (
    "%SCRIPT_DIR%.venv\Scripts\python.exe"
    "%SCRIPT_DIR%venv\Scripts\python.exe"
) do (
    if exist "%%~fP" (
        set "PY_CMD=%%~fP"
        exit /b 0
    )
)

where py >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py"
    exit /b 0
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=python"
    exit /b 0
)

if exist "%SystemRoot%\py.exe" (
    set "PY_CMD=%SystemRoot%\py.exe"
    exit /b 0
)

if exist "%LocalAppData%\Programs\Python\Launcher\py.exe" (
    set "PY_CMD=%LocalAppData%\Programs\Python\Launcher\py.exe"
    exit /b 0
)

for %%P in (
    "%LocalAppData%\Programs\Python\Python313\python.exe"
    "%LocalAppData%\Programs\Python\Python312\python.exe"
    "%LocalAppData%\Programs\Python\Python311\python.exe"
    "%LocalAppData%\Programs\Python\Python310\python.exe"
    "%LocalAppData%\Programs\Python\Python39\python.exe"
    "%ProgramFiles%\Python313\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles%\Python310\python.exe"
    "%ProgramFiles%\Python39\python.exe"
    "%ProgramFiles(x86)%\Python313\python.exe"
    "%ProgramFiles(x86)%\Python312\python.exe"
    "%ProgramFiles(x86)%\Python311\python.exe"
    "%ProgramFiles(x86)%\Python310\python.exe"
    "%ProgramFiles(x86)%\Python39\python.exe"
) do (
    if exist "%%~fP" (
        set "PY_CMD=%%~fP"
        exit /b 0
    )
)

call :try_registry_path "HKCU\Software\Python\PythonCore\3.13\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKCU\Software\Python\PythonCore\3.12\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKCU\Software\Python\PythonCore\3.11\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKCU\Software\Python\PythonCore\3.10\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKCU\Software\Python\PythonCore\3.9\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKLM\Software\Python\PythonCore\3.13\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKLM\Software\Python\PythonCore\3.12\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKLM\Software\Python\PythonCore\3.11\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKLM\Software\Python\PythonCore\3.10\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKLM\Software\Python\PythonCore\3.9\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKLM\Software\WOW6432Node\Python\PythonCore\3.13\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKLM\Software\WOW6432Node\Python\PythonCore\3.12\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKLM\Software\WOW6432Node\Python\PythonCore\3.11\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKLM\Software\WOW6432Node\Python\PythonCore\3.10\InstallPath"
if defined PY_CMD exit /b 0
call :try_registry_path "HKLM\Software\WOW6432Node\Python\PythonCore\3.9\InstallPath"
exit /b 0

:try_registry_path
for /f "skip=2 tokens=2,*" %%A in ('reg query %~1 /ve 2^>nul') do (
    if exist "%%Bpython.exe" (
        set "PY_CMD=%%Bpython.exe"
        exit /b 0
    )
)
exit /b 0

:run_python
if /I "%PY_CMD%"=="py" (
    py %*
    exit /b %ERRORLEVEL%
)
if /I "%PY_CMD%"=="python" (
    python %*
    exit /b %ERRORLEVEL%
)
"%PY_CMD%" %*
exit /b %ERRORLEVEL%
