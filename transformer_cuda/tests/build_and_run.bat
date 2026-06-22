@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" x64
cd /d C:\Users\undre\Desktop\CUDA\LLM\transformer_cuda\tests\build
cmake -G "Ninja" ..
cmake --build . --config Release
tests.exe
pause
