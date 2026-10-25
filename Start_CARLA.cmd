@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_CARLA.ps1" %*
if errorlevel 1 pause
