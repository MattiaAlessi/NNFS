@echo off
setlocal

pyinstaller --clean --noconfirm --onefile --name fashion-mnist-trainer Train.py
if errorlevel 1 exit /b %errorlevel%

echo Built dist\fashion-mnist-trainer.exe
