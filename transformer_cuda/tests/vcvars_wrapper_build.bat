@echo off
setlocal
cd /d "%~dp0"
rem Temporarily shorten PATH so vcvars64.bat doesn't fail on long env
set "OLDPATH=%PATH%"
set "PATH=C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Windows\System32\OpenSSH\"
rem Call Visual Studio environment setup
call "%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" x64
if errorlevel 1 (
  echo vcvars64.bat failed with error %errorlevel%
  endlocal
  exit /b %errorlevel%
)
if not exist build mkdir build
cd /d build
cmake -S .. -B . -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
if errorlevel 1 (
  echo cmake configure failed with error %errorlevel%
  endlocal
  exit /b %errorlevel%
)
cmake --build . --config Release
if errorlevel 1 (
  echo build failed with error %errorlevel%
  endlocal
  exit /b %errorlevel%
)
echo BUILD COMPLETE
endlocal
