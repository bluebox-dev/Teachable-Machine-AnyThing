#!/usr/bin/env bash
# ติดตั้งสิ่งที่ต้องใช้สำหรับทั้ง 3 โปรเจกต์ ใช้ได้บน macOS, Linux และ Windows (ผ่าน Git Bash หรือ WSL)
#   bash install.sh                 ใช้ Python ตัวแรกที่พบ
#   bash install.sh python3.10      ระบุ Python เอง (Windows: bash install.sh "py -3.10")
set -u

cd "$(dirname "$0")" || exit 1
VENV=".venv"
MIN_MINOR=9

case "$(uname -s)" in
  Darwin) OS=mac ;;
  Linux) OS=linux ;;
  MINGW*|MSYS*|CYGWIN*) OS=windows ;;
  *) OS=unknown ;;
esac

python_help() {
  echo "ไม่พบ Python 3.$MIN_MINOR ขึ้นไป ให้ติดตั้งก่อนแล้วรันสคริปต์นี้อีกครั้ง:"
  case "$OS" in
    mac) echo "  brew install python   หรือดาวน์โหลดจาก https://www.python.org/downloads/" ;;
    linux) echo "  sudo apt install python3 python3-venv   (Fedora: sudo dnf install python3)" ;;
    windows) echo "  winget install Python.Python.3.12   หรือดาวน์โหลดจาก https://www.python.org/downloads/" ;;
    *) echo "  https://www.python.org/downloads/" ;;
  esac
}

# หา Python ที่ใช้งานได้จริง (บน Windows คำสั่ง python3 อาจเป็นตัวเปิด Microsoft Store ที่รันไม่ได้)
PYTHON=()
if [ "$#" -ge 1 ]; then
  CANDIDATES=("$1")
else
  CANDIDATES=("python3" "python" "py -3")
fi
for candidate in "${CANDIDATES[@]}"; do
  # shellcheck disable=SC2086
  if $candidate -c "import sys; sys.exit(0 if sys.version_info >= (3, $MIN_MINOR) else 1)" >/dev/null 2>&1; then
    read -r -a PYTHON <<< "$candidate"
    break
  fi
done
if [ "${#PYTHON[@]}" -eq 0 ]; then
  [ "$#" -ge 1 ] && echo "ใช้ '$1' ไม่ได้ หรือเวอร์ชันต่ำกว่า 3.$MIN_MINOR"
  python_help
  exit 1
fi
echo "[1/4] พบ $("${PYTHON[@]}" --version) ระบบ $OS"

VERSION_CODE="import sys; print('%d.%d' % sys.version_info[:2])"
WANTED="$("${PYTHON[@]}" -c "$VERSION_CODE" | tr -d '\r')"
VENV_ARGS=()
if [ -d "$VENV" ]; then
  # venv จะไม่เปลี่ยนเวอร์ชันของ environment เดิมให้ จึงต้องล้างเมื่อไม่ตรงหรือใช้ไม่ได้
  CURRENT=""
  for existing in "$VENV/Scripts/python.exe" "$VENV/bin/python"; do
    if [ -x "$existing" ]; then
      CURRENT="$("$existing" -c "$VERSION_CODE" 2>/dev/null | tr -d '\r')"
      break
    fi
  done
  if [ "$CURRENT" != "$WANTED" ]; then
    echo "      environment เดิมเป็น Python ${CURRENT:-ที่ใช้ไม่ได้} จะสร้างใหม่ด้วย $WANTED"
    VENV_ARGS=(--clear)
  fi
fi

echo "[2/4] สร้าง environment ที่ $VENV"
if ! "${PYTHON[@]}" -m venv ${VENV_ARGS[@]+"${VENV_ARGS[@]}"} "$VENV"; then
  echo "สร้าง environment ไม่สำเร็จ"
  [ "$OS" = linux ] && echo "บน Debian/Ubuntu ให้ติดตั้งเพิ่ม: sudo apt install python3-venv"
  exit 1
fi
if [ -x "$VENV/Scripts/python.exe" ]; then
  VENV_PYTHON="$VENV/Scripts/python.exe"
else
  VENV_PYTHON="$VENV/bin/python"
fi

echo "[3/4] ติดตั้งไลบรารี (numpy, opencv-python, sounddevice)"
if ! "$VENV_PYTHON" -m pip install --quiet --upgrade pip || ! "$VENV_PYTHON" -m pip install --quiet -r requirements.txt; then
  echo "ติดตั้งไลบรารีไม่สำเร็จ ตรวจสอบอินเทอร์เน็ตแล้วลองใหม่"
  exit 1
fi

echo "[4/4] ตรวจสอบการติดตั้ง"
FAILED=0
if ! "$VENV_PYTHON" -c "import numpy" >/dev/null 2>&1; then
  echo "  ใช้งาน numpy ไม่ได้"
  FAILED=1
fi
if ! "$VENV_PYTHON" -c "import cv2" >/dev/null 2>&1; then
  echo "  ใช้งาน OpenCV ไม่ได้ (จำเป็นสำหรับ Image และ Pose)"
  [ "$OS" = linux ] && echo "  บน Debian/Ubuntu ให้ติดตั้งเพิ่ม: sudo apt install libgl1 libglib2.0-0"
  FAILED=1
fi
if ! "$VENV_PYTHON" -c "import sounddevice" >/dev/null 2>&1; then
  echo "  ใช้งานไมโครโฟนไม่ได้ (ไฟล์ .wav ยังใช้ได้ปกติ)"
  [ "$OS" = linux ] && echo "  บน Debian/Ubuntu ให้ติดตั้งเพิ่ม: sudo apt install libportaudio2"
fi
if [ "$FAILED" -ne 0 ]; then
  exit 1
fi

cat <<'EOF'

ติดตั้งเสร็จแล้ว

เปิดแอปบนเว็บ:
  bash run.sh image
  bash run.sh audio
  bash run.sh pose

รันโมเดลด้วย Python:
  bash run.sh image python
  bash run.sh audio python --source Audio-Project/Data-Test
  bash run.sh pose python --source Pose-Project/Data-Test
EOF
