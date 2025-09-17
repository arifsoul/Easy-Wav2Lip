@echo off
:run_loop
python GUI.py

if exist "run.txt" (
    echo starting Easy-Wav2Lip...
    python run.py
    del run.txt
    goto run_loop
)
