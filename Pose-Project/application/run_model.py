#!/usr/bin/env python3
"""รันโมเดล Pose จาก Teachable Machine ด้วย Python

วิธีใช้:
    python run_model.py                              เปิดกล้อง
    python run_model.py --source ../Data-Test        ทำนายทุกรูปในโฟลเดอร์
    python run_model.py --source photo.jpg           ทำนายรูปเดียว
    python run_model.py --model tm-my-model.zip      เปลี่ยนโมเดล

ต้องการแค่ numpy และ opencv-python (ติดตั้งด้วย install.sh หรือ install.bat ที่โฟลเดอร์บนสุด)
ครั้งแรกจะดาวน์โหลด PoseNet จาก Google (ราว 5 MB) มาเก็บไว้ในโฟลเดอร์ posenet
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
MIN_PART_CONFIDENCE = 0.5 # แสดงจุดบนร่างกายเฉพาะจุดที่มั่นใจเกินค่านี้
SHOW_SKELETON = True      # วาดโครงร่างบนภาพ
SHOW_WINDOW = True        # False = ไม่เปิดหน้าต่างกล้อง พิมพ์ผลใน Terminal อย่างเดียว
SAVE_DIR = ""             # ใส่ชื่อโฟลเดอร์ถ้าต้องการบันทึกรูปพร้อมผลทำนาย เช่น "output"


def on_result(label, confidence, keypoints):
    """ถูกเรียกเมื่อท่าเปลี่ยนและมั่นใจเกิน MIN_CONFIDENCE ใส่โค้ดของคุณที่นี่ได้เลย

    keypoints คือ dict เช่น keypoints["leftWrist"] = (x, y, ความมั่นใจ) ใช้นับท่าหรือวัดมุมได้
    """
    print(f"ตรวจพบท่า: {label} ({confidence:.0%})")


# ==================== ส่วนด้านล่างคือการทำงานของโมเดล ปกติไม่ต้องแก้ ====================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HERE = Path(__file__).resolve().parent
POSENET_URL = "https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/075/"
POSENET_DIR = HERE / "posenet"
INPUT_SIZE = 257
SAMPLE_SIZE = 224
STRIDE = 16
PARTS = ["nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder", "rightShoulder", "leftElbow",
         "rightElbow", "leftWrist", "rightWrist", "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]
BONES = [("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"), ("leftElbow", "leftWrist"),
         ("leftHip", "leftKnee"), ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
         ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"), ("rightHip", "rightKnee"),
         ("rightKnee", "rightAnkle"), ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")]


def resolve_path(text):
    """หาไฟล์จากโฟลเดอร์ปัจจุบันก่อน ถ้าไม่เจอให้หาเทียบกับโฟลเดอร์ของสคริปต์นี้"""
    path = Path(text)
    return path if path.exists() else HERE / path


def read_model_files(model):
    """คืนฟังก์ชันอ่านไฟล์ในโมเดล รองรับโฟลเดอร์, ไฟล์ .zip และลิงก์"""
    text = str(model)
    if text.startswith(("http://", "https://")):
        base = text.split("?")[0].split("#")[0].rstrip("/")
        for name in ("model.json", "metadata.json", "weights.bin"):
            base = base[:-len(name) - 1] if base.endswith("/" + name) else base
        return lambda name: urllib.request.urlopen(f"{base}/{name}").read()
    path = resolve_path(text)
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        # ไฟล์โมเดลรุ่นใหม่อ้างชื่อแบบ ./model.weights.bin
        return lambda name: archive.read(name[2:] if name.startswith("./") else name)
    return lambda name: (path / name).read_bytes()


def load_weights(manifest, read):
    data = b"".join(read(name) for name in manifest["paths"])
    weights = {}
    offset = 0
    for item in manifest["weights"]:
        count = int(np.prod(item["shape"], dtype=int))
        values = np.frombuffer(data, np.float32, count, offset)
        weights[item["name"]] = values.reshape(item["shape"])
        offset += count * 4
    return weights


def conv2d_same(x, kernel, strides, depthwise=False):
    """Convolution แบบ SAME ของภาพ [สูง, กว้าง, ช่อง] ด้วย NumPy"""
    kernel_h, kernel_w = kernel.shape[:2]
    pads = []
    for size, k, s in zip(x.shape[:2], (kernel_h, kernel_w), strides):
        total = max((int(np.ceil(size / s)) - 1) * s + k - size, 0)
        pads.append((total // 2, total - total // 2))
    x = np.pad(x, (*pads, (0, 0)))
    last_h, last_w = x.shape[0] - kernel_h, x.shape[1] - kernel_w
    out = 0
    for i in range(kernel_h):
        for j in range(kernel_w):
            patch = x[i:last_h + i + 1:strides[0], j:last_w + j + 1:strides[1]]
            out = out + (patch * kernel[i, j, :, 0] if depthwise else patch @ kernel[i, j])
    return out


class PoseNet:
    """PoseNet MobileNetV1 0.75 stride 16 ชุดเดียวกับที่ Teachable Machine ใช้"""

    OUTPUTS = {"heatmap": "MobilenetV1/heatmap_2/BiasAdd", "offset": "MobilenetV1/offset_2/BiasAdd"}

    def __init__(self):
        spec_path = POSENET_DIR / "model-stride16.json"
        if not spec_path.exists():
            self.download(spec_path)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.nodes = {node["name"]: node for node in spec["modelTopology"]["node"]}
        self.weights = load_weights(spec["weightsManifest"][0], lambda name: (POSENET_DIR / name).read_bytes())

    @staticmethod
    def download(spec_path):
        print("กำลังดาวน์โหลด PoseNet ครั้งแรก...")
        POSENET_DIR.mkdir(exist_ok=True)
        try:
            spec = urllib.request.urlopen(POSENET_URL + spec_path.name).read()
            for name in json.loads(spec)["weightsManifest"][0]["paths"]:
                (POSENET_DIR / name).write_bytes(urllib.request.urlopen(POSENET_URL + name).read())
        except OSError as err:
            sys.exit(f"ดาวน์โหลด PoseNet ไม่สำเร็จ ({err}) ตรวจสอบอินเทอร์เน็ตแล้วลองใหม่")
        spec_path.write_bytes(spec)  # เขียนไฟล์นี้ท้ายสุด เพื่อให้รู้ว่าดาวน์โหลดครบแล้ว

    def run(self, rgb_square):
        """คืน heatmap [17, 17, 17] และ offset [17, 17, 34]"""
        results = {"sub_2": self.prepare(rgb_square)}

        def evaluate(name):
            if name in self.weights:
                return self.weights[name]
            if name in results:
                return results[name]
            node = self.nodes[name]
            op = node["op"]
            inputs = [evaluate(i) for i in node["input"]]
            if op == "Conv2D" or op == "DepthwiseConv2dNative":
                strides = [int(s) for s in node["attr"]["strides"]["list"]["i"][1:3]]
                out = conv2d_same(inputs[0], inputs[1], strides, depthwise=op != "Conv2D")
            elif op == "BiasAdd":
                out = inputs[0] + inputs[1]
            elif op == "Relu6":
                out = np.clip(inputs[0], 0, 6)
            else:
                raise NotImplementedError(f"ยังไม่รองรับ {op}")
            results[name] = out
            return out

        heatmap = 1 / (1 + np.exp(-evaluate(self.OUTPUTS["heatmap"])))
        return heatmap, evaluate(self.OUTPUTS["offset"])

    @staticmethod
    def prepare(rgb_square):
        """ย่อขยายแบบเดียวกับ tf.image.resizeBilinear แล้วปรับค่าเป็นช่วง -1 ถึง 1"""
        size = rgb_square.shape[0]
        positions = np.arange(INPUT_SIZE) * size / INPUT_SIZE
        low = positions.astype(int)
        high = np.minimum(low + 1, size - 1)
        weight = (positions - low).astype(np.float32)
        image = rgb_square.astype(np.float32)
        rows = image[low] * (1 - weight)[:, None, None] + image[high] * weight[:, None, None]
        resized = rows[:, low] * (1 - weight)[None, :, None] + rows[:, high] * weight[None, :, None]
        return resized / 127.5 - 1


class TeachableModel:
    """PoseNet สกัดท่าทาง แล้วส่งให้โมเดลที่เทรนจาก Teachable Machine แยกประเภท"""

    def __init__(self, source):
        read = read_model_files(source)
        spec = json.loads(read("model.json"))
        metadata = json.loads(read("metadata.json"))
        kind = "Audio" if "wordLabels" in metadata else metadata.get("packageName", "")
        if kind and kind != "@teachablemachine/pose":
            sys.exit(f"โมเดลนี้เป็นชนิด {kind} สคริปต์นี้ใช้ได้เฉพาะโมเดล Pose")
        posenet = metadata.get("modelSettings", {}).get("posenet", {})
        if posenet.get("multiplier", 0.75) != 0.75 or posenet.get("outputStride", STRIDE) != STRIDE or posenet.get("inputResolution", INPUT_SIZE) != INPUT_SIZE:
            sys.exit(f"สคริปต์นี้รองรับเฉพาะ PoseNet multiplier 0.75, stride 16, ขนาด 257 แต่โมเดลนี้ใช้ {posenet}")
        self.labels = metadata["labels"]
        config = spec["modelTopology"]["config"]
        self.layers = config["layers"] if isinstance(config, dict) else config
        self.weights = load_weights(spec["weightsManifest"][0], read)
        self.posenet = PoseNet()

    def classify(self, features):
        x = features
        for layer in self.layers:
            if layer["class_name"] != "Dense":
                continue
            cfg = layer["config"]
            x = x @ self.weights[f"{cfg['name']}/kernel"]
            if cfg["use_bias"]:
                x = x + self.weights[f"{cfg['name']}/bias"]
            if cfg["activation"] == "relu":
                x = np.maximum(x, 0)
            elif cfg["activation"] == "softmax":
                x = np.exp(x - x.max())
                x = x / x.sum()
        return x

    def predict(self, bgr):
        """คืนความน่าจะเป็นของทุกคลาส และตำแหน่งจุดบนร่างกายเทียบกับภาพต้นฉบับ"""
        height, width = bgr.shape[:2]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        square = bgr[top:top + side, left:left + side]
        if side != SAMPLE_SIZE:
            method = cv2.INTER_AREA if side > SAMPLE_SIZE else cv2.INTER_LINEAR
            square = cv2.resize(square, (SAMPLE_SIZE, SAMPLE_SIZE), interpolation=method)
        heatmap, offset = self.posenet.run(square[:, :, ::-1])
        scores = self.classify(np.concatenate([heatmap, offset], axis=2).reshape(-1))

        keypoints = {}
        scale = side / INPUT_SIZE
        grid_h, grid_w = heatmap.shape[:2]
        for i, part in enumerate(PARTS):
            y, x = np.unravel_index(np.argmax(heatmap[:, :, i]), (grid_h, grid_w))
            py = y * STRIDE + offset[y, x, i]
            px = x * STRIDE + offset[y, x, i + len(PARTS)]
            keypoints[part] = (left + px * scale, top + py * scale, float(heatmap[y, x, i]))
        return scores, keypoints


def describe(model, scores):
    best = int(np.argmax(scores))
    return model.labels[best], float(scores[best])


def draw_result(frame, model, scores, keypoints, args):
    """วาดโครงร่างและผลทำนายลงบนภาพ (ถ้าชื่อคลาสเป็นภาษาไทย OpenCV จะแสดงเป็น ???)"""
    if SHOW_SKELETON:
        thick = max(2, frame.shape[0] // 200)
        for a, b in BONES:
            if keypoints[a][2] >= MIN_PART_CONFIDENCE and keypoints[b][2] >= MIN_PART_CONFIDENCE:
                cv2.line(frame, tuple(map(int, keypoints[a][:2])), tuple(map(int, keypoints[b][:2])), (120, 220, 90), thick, cv2.LINE_AA)
        for x, y, score in keypoints.values():
            if score >= MIN_PART_CONFIDENCE:
                cv2.circle(frame, (int(x), int(y)), thick + 2, (60, 200, 255), -1, cv2.LINE_AA)
    label, confidence = describe(model, scores)
    title = f"{label} {confidence:.0%}" if confidence >= args.min_confidence else "not sure"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (32, 32, 32), -1)
    cv2.putText(frame, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def save_image(path, frame):
    # ใช้ tofile และ fromfile เพื่อให้เปิดพาธภาษาไทยบน Windows ได้
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix or ".jpg", frame)[1].tofile(str(path))


def read_image(file):
    # ใช้ fromfile เพื่อให้เปิดพาธภาษาไทยบน Windows ได้ และไฟล์ว่างจะคืน None
    data = np.fromfile(str(file), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


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
        frame = read_image(file)
        if frame is None:
            print(f"{file.name}: อ่านไฟล์นี้ไม่ได้")
            continue
        scores, keypoints = model.predict(frame)
        label, confidence = describe(model, scores)
        print(f"{file.name}: {label} {confidence:.1%}")
        if confidence >= args.min_confidence:
            on_result(label, confidence, keypoints)
        if SAVE_DIR:
            draw_result(frame, model, scores, keypoints, args)
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
            scores, keypoints = model.predict(frame)
            label, confidence = describe(model, scores)
            if confidence >= args.min_confidence and label != last_label:
                on_result(label, confidence, keypoints)
                last_label = label
            if args.window:
                now = time.perf_counter()
                fps = 1 / max(now - last_time, 1e-6)
                last_time = now
                draw_result(frame, model, scores, keypoints, args)
                cv2.putText(frame, f"{fps:.0f} FPS", (frame.shape[1] - 90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Pose Classifier", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()


def load_model(source):
    """โหลดโมเดล ถ้าไม่สำเร็จให้แจ้งสาเหตุแทนการแสดง Traceback"""
    try:
        return TeachableModel(source)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as err:
        sys.exit(f"โหลดโมเดลจาก {source} ไม่สำเร็จ ตรวจสอบว่าลิงก์ถูกต้อง และมี model.json, metadata.json, weights.bin ครบ ({err})")


def main():
    parser = argparse.ArgumentParser(description="รันโมเดล Pose จาก Teachable Machine")
    parser.add_argument("--model", default=MODEL, help="โฟลเดอร์, ไฟล์ .zip หรือลิงก์ของโมเดล")
    parser.add_argument("--source", default=SOURCE, help='"webcam", ไฟล์รูป หรือโฟลเดอร์รูป')
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="เลขกล้อง")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE, help="ความมั่นใจขั้นต่ำ 0-1")
    parser.add_argument("--no-window", dest="window", action="store_false", default=SHOW_WINDOW, help="ไม่เปิดหน้าต่างกล้อง")
    args = parser.parse_args()

    model = load_model(args.model)
    print(f"โหลดโมเดลแล้ว: {', '.join(model.labels)}")
    if args.source == "webcam":
        run_webcam(model, args)
    else:
        run_files(model, args.source, args)


if __name__ == "__main__":
    main()
