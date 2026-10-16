@echo off
REM Double-click this file to start the web app (Windows).
REM It starts the Flask server and opens http://localhost:5000 in the browser.
REM Close this window (or press Ctrl+C) to stop the server.

title Procurement Risk Agent - server
cd /d "%~dp0"
set URL=http://localhost:5000

REM Already running? Then just open the browser.
netstat -ano | findstr ":5000 " | findstr "LISTENING" >nul
if %errorlevel%==0 (
    echo The web app is already running. Opening %URL% ...
    start "" %URL%
    ping -n 3 127.0.0.1 >nul
    exit /b 0
)

REM Use the project's virtual environment if there is one, otherwise the global Python.
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

%PY% --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ and try again.
    echo Press any key to close this window.
    pause >nul
    exit /b 1
)

REM Install the dependencies the first time.
%PY% -c "import flask, pandas" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installation failed.
        echo Press any key to close this window.
        pause >nul
        exit /b 1
    )
)

REM Generate and process the data the first time.
if not exist "data\processed\open_orders.csv" (
    echo First start: generating and processing the data...
    %PY% -m src.main setup
    if errorlevel 1 (
        echo Setup failed.
        echo Press any key to close this window.
        pause >nul
        exit /b 1
    )
)

echo.
echo Starting the web app at %URL%
echo Keep this window open. Close it to stop the server.
echo.
REM Open the browser after a short delay, once the server is listening.
start "" cmd /c "ping -n 3 127.0.0.1 >nul & start "" %URL%"
%PY% app.py
echo Press any key to close this window.
pause >nul
