@echo off
if not "%~1"=="" goto arguments
echo.
echo Choose an Indian road setting:
echo   1. Bazaar market street
echo   2. Residential colony
echo   3. Divided main road
echo   4. Rural village road
echo   5. Irregular junction
echo   6. Tour all five
echo.
choice /C 123456 /N /M "Choose 1-6: "
if errorlevel 6 goto tour
if errorlevel 5 goto junction
if errorlevel 4 goto rural
if errorlevel 3 goto mainroad
if errorlevel 2 goto colony
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Indian_Worlds.ps1" -Map market_street
goto end
:colony
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Indian_Worlds.ps1" -Map residential_colony
goto end
:mainroad
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Indian_Worlds.ps1" -Map divided_main_road
goto end
:rural
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Indian_Worlds.ps1" -Map rural_road
goto end
:junction
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Indian_Worlds.ps1" -Map irregular_junction
goto end
:tour
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Indian_Worlds.ps1" -Map all
goto end
:arguments
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start_Indian_Worlds.ps1" %*
:end
