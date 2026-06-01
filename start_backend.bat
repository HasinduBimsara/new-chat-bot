@echo off
echo Starting Backend with GPU Support...
cd backend
call ..\.venv_gpu\Scripts\activate
python app.py
pause
