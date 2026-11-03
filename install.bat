@echo off
rem ติดตั้งสิ่งที่ต้องใช้บน Windows สำหรับเครื่องที่ไม่มี Git Bash (ดับเบิลคลิกไฟล์นี้ได้เลย)
rem ระบุเวอร์ชัน Python เองได้ เช่น install.bat 3.10
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
if not "%~1"=="" (
  py -%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1 && set "PY=py -%~1"
  if not defined PY (
    echo ไม่พบ Python %~1 ให้ติดตั้งก่อนด้วยคำสั่ง: winget install Python.Python.%~1
    pause
    exit /b 1
  )
  goto :create
)
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY (
  echo ไม่พบ Python 3.9 ขึ้นไป ให้ติดตั้งก่อนด้วยคำสั่ง: winget install Python.Python.3.12
  echo หรือดาวน์โหลดจาก https://www.python.org/downloads/
  pause
  exit /b 1
)

:create
echo [1/3] สร้าง environment ที่ .venv
rem ใช้ --clear เพราะ venv จะไม่เปลี่ยนเวอร์ชันของ environment เดิมให้
%PY% -m venv --clear .venv || goto :failed
echo [2/3] ติดตั้งไลบรารี (numpy, opencv-python, sounddevice)
.venv\Scripts\python.exe -m pip install --quiet --upgrade pip || goto :failed
.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt || goto :failed
echo [3/3] ตรวจสอบการติดตั้ง
.venv\Scripts\python.exe -c "import numpy, cv2, sounddevice" || goto :failed

echo.
echo ติดตั้งเสร็จแล้ว
echo เปิดแอปบนเว็บ: ดับเบิลคลิก run.bat ในโฟลเดอร์ application ของแต่ละโปรเจกต์
echo รันโมเดลด้วย Python: .venv\Scripts\python.exe Image-Project\application\run_model.py
pause
exit /b 0

:failed
echo ติดตั้งไม่สำเร็จ ตรวจสอบข้อความด้านบนและการเชื่อมต่ออินเทอร์เน็ต
pause
exit /b 1
