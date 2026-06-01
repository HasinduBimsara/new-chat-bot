@echo off
echo ========================================================
echo Starting SLM Retraining on Global Suggestions Dataset...
echo ========================================================
echo This process will train the SLM exclusively on the new
echo dataset of 634 JSON suggestions + Thousands of Gemini
echo Generated Worldwide Clinical Suggestions.
echo.
echo Press CTRL+C at any time to stop training.
echo The model will automatically save as best_model_params_sinhala.pt
echo ========================================================
pause

call .venv_gpu\Scripts\activate
uv run --python .venv_gpu python train_slm.py
pause
