@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "LAUNCHER=%SCRIPT_DIR%Run_OSC_Viewer.bat"
set "PROG_ID=OSCReader.osc"

if not exist "%LAUNCHER%" (
    echo Could not find launcher:
    echo %LAUNCHER%
    pause
    exit /b 1
)

echo Registering .osc file association for the current user...

reg add "HKCU\Software\Classes\.osc" /ve /d "%PROG_ID%" /f >nul
if errorlevel 1 goto :reg_fail

reg add "HKCU\Software\Classes\%PROG_ID%" /ve /d "OSC Reader File" /f >nul
if errorlevel 1 goto :reg_fail

reg add "HKCU\Software\Classes\%PROG_ID%\shell" /ve /d "open" /f >nul
if errorlevel 1 goto :reg_fail

reg add "HKCU\Software\Classes\%PROG_ID%\shell\open\command" /ve /d "\"%LAUNCHER%\" \"%%1\"" /f >nul
if errorlevel 1 goto :reg_fail

REM Remove explicit per-extension UserChoice so Windows can fall back to our ProgID.
reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\.osc\UserChoice" /f >nul 2>&1

echo Done. Double-click an .osc file to open it in OSC_Reader.
echo If Windows still opens a different app, sign out/in or restart Explorer once.
exit /b 0

:reg_fail
echo Failed to write registry keys. Try running this script from a normal user shell.
pause
exit /b 1
