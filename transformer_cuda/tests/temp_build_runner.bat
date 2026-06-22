@echo off
setlocal
cd /d "%~dp0"
set "PATH=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\System32\OpenSSH\"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" x64
if errorlevel 1 (
  echo vcvars64.bat failed with error %errorlevel%
  exit /b %errorlevel%
)
if not exist build mkdir build
cd /d build
cmake .. -G Ninja
if errorlevel 1 (
  echo cmake configuration failed with error %errorlevel%
  exit /b %errorlevel%
)
cmake --build . --config Release
if errorlevel 1 (
  echo build failed with error %errorlevel%
  exit /b %errorlevel%
)
echo BUILD COMPLETE
endlocal
