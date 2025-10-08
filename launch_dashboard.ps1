# CDS Monitor - Update and Launch Script
# This script updates the databases and launches the Streamlit dashboard

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "CDS Monitor - Update and Launch" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Update databases
Write-Host "[Step 1/2] Updating databases with latest Bloomberg data..." -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"

try {
    poetry run python update_databases.py
    
    if ($LASTEXITCODE -ne 0) {
        throw "Database update failed with exit code $LASTEXITCODE"
    }
    
    Write-Host ""
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "Database update complete!" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host ""
}
catch {
    Write-Host ""
    Write-Host "ERROR: Database update failed!" -ForegroundColor Red
    Write-Host "Please check Bloomberg connection and try again." -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 2: Launch Streamlit
Write-Host "[Step 2/2] Launching Streamlit dashboard..." -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------"
Write-Host ""
Write-Host "The dashboard will open in your browser shortly..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server when done." -ForegroundColor Cyan
Write-Host ""

# Navigate to src/apps directory for proper module imports
Set-Location -Path "$PSScriptRoot\src\apps"
try {
    poetry run streamlit run streamlit_app.py
}
finally {
    Set-Location -Path $PSScriptRoot
    Write-Host ""
    Write-Host "Streamlit server stopped." -ForegroundColor Yellow
}
