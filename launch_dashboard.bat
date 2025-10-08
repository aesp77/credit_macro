@echo off
REM CDS Monitor - Update and Launch Script
REM This script updates the databases and launches the Streamlit dashboard

echo ======================================================================
echo CDS Monitor - Update and Launch
echo ======================================================================
echo.

REM Step 1: Update databases
echo [Step 1/2] Updating databases with latest Bloomberg data...
echo ----------------------------------------------------------------------
poetry run python update_databases.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Database update failed!
    echo Please check Bloomberg connection and try again.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Database update complete!
echo ======================================================================
echo.

REM Step 2: Launch Streamlit
echo [Step 2/2] Launching Streamlit dashboard...
echo ----------------------------------------------------------------------
echo.
echo The dashboard will open in your browser shortly...
echo Press Ctrl+C to stop the server when done.
echo.

cd src\apps
poetry run streamlit run streamlit_app.py

REM If streamlit exits
echo.
echo Streamlit server stopped.
pause
