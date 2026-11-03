#!/usr/bin/env bash
# เปิดแอปหรือรันโมเดล ใช้ได้บน macOS, Linux และ Windows (ผ่าน Git Bash หรือ WSL)
#   bash run.sh image            เปิดแอปบนเว็บ
#   bash run.sh image python     รันโมเดลด้วย Python (เพิ่มตัวเลือกต่อท้ายได้ เช่น --source photo.jpg)
set -u

cd "$(dirname "$0")" || exit 1

usage() {
  echo "วิธีใช้: bash run.sh <image|audio|pose> [web|python] [ตัวเลือกของ run_model.py]"
  exit 1
}

[ "$#" -ge 1 ] || usage
case "$1" in
  image) PROJECT=Image-Project; PORT=8001 ;;
  audio) PROJECT=Audio-Project; PORT=8002 ;;
  pose) PROJECT=Pose-Project; PORT=8003 ;;
  *) usage ;;
esac
MODE="${2:-web}"
shift
[ "$#" -gt 0 ] && shift

if [ -x .venv/Scripts/python.exe ]; then
  PYTHON=(.venv/Scripts/python.exe)
elif [ -x .venv/bin/python ]; then
  PYTHON=(.venv/bin/python)
elif python3 -c "" >/dev/null 2>&1; then
  PYTHON=(python3)
elif python -c "" >/dev/null 2>&1; then
  PYTHON=(python)
elif py -3 -c "" >/dev/null 2>&1; then
  PYTHON=(py -3)
else
  echo "ไม่พบ Python ให้รัน: bash install.sh"
  exit 1
fi

open_browser() {
  sleep 1
  case "$(uname -s)" in
    Darwin) open "$1" ;;
    MINGW*|MSYS*|CYGWIN*) cmd.exe //c start "" "$1" ;;
    *)
      if command -v wslview >/dev/null 2>&1; then
        wslview "$1"
      elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$1"
      fi
      ;;
  esac
}

case "$MODE" in
  web)
    URL="http://localhost:$PORT/"
    echo "เปิดแอปที่ $URL  กด Ctrl+C เพื่อปิด"
    open_browser "$URL" >/dev/null 2>&1 &
    cd "$PROJECT/application" || exit 1
    case "${PYTHON[0]}" in
      .venv/*) PYTHON[0]="../../${PYTHON[0]}" ;;
    esac
    exec "${PYTHON[@]}" -m http.server "$PORT" --bind 127.0.0.1
    ;;
  python)
    if ! "${PYTHON[@]}" -c "import numpy" >/dev/null 2>&1; then
      echo "ยังไม่ได้ติดตั้งไลบรารี ให้รัน: bash install.sh"
      exit 1
    fi
    exec "${PYTHON[@]}" "$PROJECT/application/run_model.py" "$@"
    ;;
  *) usage ;;
esac
