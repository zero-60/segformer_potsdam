@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo [MERIS] Preparing TLSNMSP data protocol...
python tools\prepare_meris_tlsnmsp.py --data-root "D:\MRISE\dataset\TLSNMSP"
echo.
echo If preparation passed, inspect data\processed\meris_tlsnmsp\preparation_report.txt
pause
