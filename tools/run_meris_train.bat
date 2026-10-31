@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo [MERIS] Starting TLSNMSP training. Make sure smoke test passed first.
python tools\train_model_b2_rgbnir_meris.py --data-root "D:\MRISE\dataset\TLSNMSP" --epochs 40 --batch-size 1 --grad-accum-steps 4 --num-workers 0
echo.
pause
