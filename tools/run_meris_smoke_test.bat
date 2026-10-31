@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo [MERIS] Running one-batch model/data smoke test...
python tools\train_model_b2_rgbnir_meris.py --smoke-test --data-root "D:\MRISE\dataset\TLSNMSP"
echo.
pause
