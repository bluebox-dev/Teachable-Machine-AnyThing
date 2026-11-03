@echo off
rem ดับเบิลคลิกไฟล์นี้บน Windows เพื่อเปิดแอป ปิดหน้าต่างนี้เมื่อเลิกใช้งาน
chcp 65001 >nul
cd /d "%~dp0"
set PORT=8002
start "" "http://localhost:%PORT%/"
if exist "..\..\.venv\Scripts\python.exe" (
  "..\..\.venv\Scripts\python.exe" -m http.server %PORT% --bind 127.0.0.1
) else (
  py -3 -m http.server %PORT% --bind 127.0.0.1 || python -m http.server %PORT% --bind 127.0.0.1
)
pause
