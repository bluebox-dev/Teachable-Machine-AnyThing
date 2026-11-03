#!/bin/bash
# ดับเบิลคลิกไฟล์นี้บน macOS เพื่อเปิดแอป กด Ctrl+C ในหน้าต่าง Terminal เพื่อปิด
cd "$(dirname "$0")" || exit 1
PORT=8003
(sleep 1 && open "http://localhost:$PORT/") &
python3 -m http.server "$PORT" --bind 127.0.0.1
