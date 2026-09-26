@echo off
cd /d "%~dp0"
echo Pivot Scanner — %DATE% %TIME%
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python from https://python.org
    pause
    exit /b 1
)

python -c "import yfinance" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    python -m pip install -r requirements.txt
    echo.
)

python scanner.py
if %errorlevel% neq 0 (
    echo.
    echo Scanner failed. Check the error above.
    pause
)
