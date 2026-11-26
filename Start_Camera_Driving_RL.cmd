@echo off
rem Camera driving with the learned speed governor on (docs/RL_SPEED_DESIGN_20261112.md)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Camera_Driving.ps1" -RLSpeed %*
