@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Indian_Traffic.ps1" %*
if errorlevel 1 pause
