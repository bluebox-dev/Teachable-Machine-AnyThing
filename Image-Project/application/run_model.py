#!/usr/bin/env python3
"""รันโมเดล Image จาก Teachable Machine ด้วย Python

วิธีใช้:
    python run_model.py                              เปิดกล้อง
    python run_model.py --source ../Data-Test        ทำนายทุกรูปในโฟลเดอร์
    python run_model.py --source photo.jpg           ทำนายรูปเดียว
    python run_model.py --model tm-my-model.zip      เปลี่ยนโมเดล

ต้องการแค่ numpy และ opencv-python (ติดตั้งด้วย install.sh หรือ install.bat ที่โฟลเดอร์บนสุด)
"""
import argparse
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

# ============================== ตั้งค่า: แก้ตรงนี้ได้เลย ==============================

MODEL = "my-model"        # โฟลเดอร์โมเดล, ไฟล์ .zip ที่ export มา หรือลิงก์จาก Teachable Machine
SOURCE = "webcam"         # "webcam", ไฟล์รูป หรือโฟลเดอร์รูป
CAMERA_INDEX = 0          # เลขกล้อง ถ้ามีหลายตัวลอง 1, 2, ...
FLIP_CAMERA = True        # กลับภาพซ้าย-ขวา ให้เหมือนตอนเก็บข้อมูลบน Teachable Machine
MIN_CONFIDENCE = 0.60     # ความมั่นใจขั้นต่ำ (0-1) ถ้าต่ำกว่านี้จะถือว่า "ไม่แน่ใจ"
SHOW_WINDOW = True        # False = ไม่เปิดหน้าต่างกล้อง พิมพ์ผลใน Terminal อย่างเดียว
SAVE_DIR = ""             # ใส่ชื่อโฟลเดอร์ถ้าต้องการบันทึกรูปพร้อมผลทำนาย เช่น "output"


def on_result(label, confidence):
    """ถูกเรียกเมื่อผลเปลี่ยนและมั่นใจเกิน MIN_CONFIDENCE ใส่โค้ดของคุณที่นี่ได้เลย

    เช่น ส่งแจ้งเตือน, สั่งงาน Arduino หรือบันทึกลงไฟล์
    """
    print(f"ตรวจพบ: {label} ({confidence:.0%})")


# ==================== ส่วนด้านล่างคือการทำงานของโมเดล ปกติไม่ต้องแก้ ====================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HERE = Path(__file__).resolve().parent


def resolve_path(text):
    """หาไฟล์จากโฟลเดอร์ปัจจุบันก่อน ถ้าไม่เจอให้หาเทียบกับโฟลเดอร์ของสคริปต์นี้"""
    path = Path(text)
    return path if path.exists() else HERE / path


def read_model_files(model):
    """คืนฟังก์ชันอ่านไฟล์ในโมเดล รองรับโฟลเดอร์, ไฟล์ .zip และลิงก์"""
    text = str(model)
    if text.startswith(("http://", "https://")):
        base = text.split("?")[0].split("#")[0].rstrip("/")
        return lambda name: urllib.request.urlopen(f"{base}/{name}").read()
    path = resolve_path(text)
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        return archive.read
    return lambda name: (path / name).read_bytes()


def conv2d(x, kernel, strides=(1, 1), same=False, depthwise=False):
    """Convolution ของภาพ [สูง, กว้าง, ช่อง] ด้วย NumPy"""
    kernel_h, kernel_w = kernel.shape[:2]
    if same:
        pads = []
        for size, k, s in zip(x.shape[:2], (kernel_h, kernel_w), strides):
            total = max((int(np.ceil(size / s)) - 1) * s + k - size, 0)
            pads.append((total // 2, total - total // 2))
        x = np.pad(x, (*pads, (0, 0)))
    height, width = x.shape[:2]
    last_h, last_w = height - kernel_h, width - kernel_w
    out = 0
    for i in range(kernel_h):
        for j in range(kernel_w):
            patch = x[i:last_h + i + 1:strides[0], j:last_w + j + 1:strides[1]]
            out = out + (patch * kernel[i, j, :, 0] if depthwise else patch @ kernel[i, j])
    return out


class TeachableModel:
    """อ่าน model.json กับ weights.bin ของ TensorFlow.js แล้วคำนวณตามโครงสร้างในไฟล์"""

    def __init__(self, source):
        read = read_model_files(source)
        spec = json.loads(read("model.json"))
        self.metadata = json.loads(read("metadata.json"))
        kind = "Audio" if "wordLabels" in self.metadata else self.metadata.get("packageName", "")
        if kind and kind != "@teachablemachine/image":
            sys.exit(f"โมเดลนี้เป็นชนิด {kind} สคริปต์นี้ใช้ได้เฉพาะโมเดล Image")
        self.labels = self.metadata["labels"]
        self.size = self.metadata.get("imageSize", 224)
        self.topology = spec["modelTopology"]
        manifest = spec["weightsManifest"][0]
        data = b"".join(read(name) for name in manifest["paths"])
        self.weights = {}
        offset = 0
        for item in manifest["weights"]:
            count = int(np.prod(item["shape"], dtype=int))
            values = np.frombuffer(data, np.float32, count, offset)
            self.weights[item["name"]] = values.reshape(item["shape"])
            offset += count * 4

    def forward(self, layer, x):
        kind, cfg = layer["class_name"], layer["config"]
        name = cfg.get("name") if isinstance(cfg, dict) else None
        w = self.weights
        if kind == "Sequential":
            for child in cfg["layers"] if isinstance(cfg, dict) else cfg:
                x = self.forward(child, x)
            return x
        if kind == "Model":
            outputs = {}
            for child in cfg["layers"]:
                if child["class_name"] == "InputLayer":
                    outputs[child["config"]["name"]] = x
                    continue
                inputs = [outputs[node[0]] for node in child["inbound_nodes"][0]]
                outputs[child["config"]["name"]] = self.forward(child, inputs if len(inputs) > 1 else inputs[0])
            return outputs[cfg["output_layers"][0][0]]
        if kind == "ZeroPadding2D":
            return np.pad(x, (*map(tuple, cfg["padding"]), (0, 0)))
        if kind == "Conv2D":
            out = conv2d(x, w[f"{name}/kernel"], cfg["strides"], cfg["padding"] == "same")
            return out + w[f"{name}/bias"] if cfg["use_bias"] else out
        if kind == "DepthwiseConv2D":
            out = conv2d(x, w[f"{name}/depthwise_kernel"], cfg["strides"], cfg["padding"] == "same", depthwise=True)
            return out + w[f"{name}/bias"] if cfg["use_bias"] else out
        if kind == "BatchNormalization":
            scale = np.sqrt(w[f"{name}/moving_variance"] + cfg["epsilon"])
            return (x - w[f"{name}/moving_mean"]) / scale * w[f"{name}/gamma"] + w[f"{name}/beta"]
        if kind == "ReLU":
            return np.clip(x, 0, cfg.get("max_value") or None)
        if kind == "Add":
            return sum(x)
        if kind == "GlobalAveragePooling2D":
            return x.mean(axis=(0, 1))
        if kind == "Dense":
            out = x @ w[f"{name}/kernel"]
            if cfg["use_bias"]:
                out = out + w[f"{name}/bias"]
            if cfg["activation"] == "relu":
                return np.maximum(out, 0)
            if cfg["activation"] == "softmax":
                out = np.exp(out - out.max())
                return out / out.sum()
            return out
        if kind == "Dropout":
            return x
        raise NotImplementedError(f"ยังไม่รองรับเลเยอร์ {kind}")

    def predict(self, bgr):
        """รับภาพจาก OpenCV แล้วคืนความน่าจะเป็นของทุกคลาส"""
        height, width = bgr.shape[:2]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        square = bgr[top:top + side, left:left + side]
        if side != self.size:
            method = cv2.INTER_AREA if side > self.size else cv2.INTER_LINEAR
            square = cv2.resize(square, (self.size, self.size), interpolation=method)
        rgb = square[:, :, ::-1].astype(np.float32) / 127 - 1
        return self.forward(self.topology, rgb)


def describe(model, scores):
    best = int(np.argmax(scores))
    return model.labels[best], float(scores[best])


def draw_result(frame, model, scores, min_confidence):
    """วาดผลทำนายลงบนภาพ (ถ้าชื่อคลาสเป็นภาษาไทย OpenCV จะแสดงเป็น ???)"""
    label, confidence = describe(model, scores)
    title = f"{label} {confidence:.0%}" if confidence >= min_confidence else "not sure"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (32, 32, 32), -1)
    cv2.putText(frame, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    for i, (name, score) in enumerate(zip(model.labels, scores)):
        y = 60 + i * 26
        cv2.rectangle(frame, (10, y - 14), (10 + int(160 * score), y + 4), (210, 120, 40), -1)
        cv2.rectangle(frame, (10, y - 14), (170, y + 4), (255, 255, 255), 1)
        cv2.putText(frame, f"{name} {score:.0%}", (178, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def save_image(path, frame):
    # ใช้ tofile และ fromfile เพื่อให้เปิดพาธภาษาไทยบน Windows ได้
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix or ".jpg", frame)[1].tofile(str(path))


def run_files(model, source, args):
    path = resolve_path(source)
    if path.is_dir():
        files = sorted(
            (f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda f: (len(f.stem), f.stem),
        )
    else:
        files = [path]
    if not files or not files[0].exists():
        sys.exit(f"ไม่พบรูปภาพที่ {path}")
    for file in files:
        frame = cv2.imdecode(np.fromfile(str(file), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"{file.name}: อ่านไฟล์นี้ไม่ได้")
            continue
        scores = model.predict(frame)
        label, confidence = describe(model, scores)
        print(f"{file.name}: {label} {confidence:.1%}")
        if confidence >= args.min_confidence:
            on_result(label, confidence)
        if SAVE_DIR:
            draw_result(frame, model, scores, args.min_confidence)
            save_image(resolve_path(SAVE_DIR) / file.name, frame)


def run_webcam(model, args):
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        sys.exit(f"เปิดกล้องหมายเลข {args.camera} ไม่ได้ ลองเปลี่ยน CAMERA_INDEX หรืออนุญาตให้ Terminal ใช้กล้อง")
    print("เปิดกล้องแล้ว กด q ที่หน้าต่างกล้องเพื่อออก" if args.window else "เปิดกล้องแล้ว กด Ctrl+C เพื่อออก")
    last_label = None
    last_time = time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                sys.exit("อ่านภาพจากกล้องไม่ได้")
            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)
            scores = model.predict(frame)
            label, confidence = describe(model, scores)
            if confidence >= args.min_confidence and label != last_label:
                on_result(label, confidence)
                last_label = label
            if args.window:
                now = time.perf_counter()
                fps = 1 / max(now - last_time, 1e-6)
                last_time = now
                draw_result(frame, model, scores, args.min_confidence)
                cv2.putText(frame, f"{fps:.0f} FPS", (frame.shape[1] - 90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Image Classifier", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="รันโมเดล Image จาก Teachable Machine")
    parser.add_argument("--model", default=MODEL, help="โฟลเดอร์, ไฟล์ .zip หรือลิงก์ของโมเดล")
    parser.add_argument("--source", default=SOURCE, help='"webcam", ไฟล์รูป หรือโฟลเดอร์รูป')
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="เลขกล้อง")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE, help="ความมั่นใจขั้นต่ำ 0-1")
    parser.add_argument("--no-window", dest="window", action="store_false", default=SHOW_WINDOW, help="ไม่เปิดหน้าต่างกล้อง")
    args = parser.parse_args()

    model = TeachableModel(args.model)
    print(f"โหลดโมเดลแล้ว: {', '.join(model.labels)}")
    if args.source == "webcam":
        run_webcam(model, args)
    else:
        run_files(model, args.source, args)


if __name__ == "__main__":
    main()
