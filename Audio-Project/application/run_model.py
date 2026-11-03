#!/usr/bin/env python3
"""รันโมเดล Audio จาก Teachable Machine ด้วย Python

วิธีใช้:
    python run_model.py                              ฟังจากไมโครโฟน
    python run_model.py --source ../Data-Test        ทำนายทุกไฟล์ .wav ในโฟลเดอร์
    python run_model.py --source voice.wav           ทำนายไฟล์เดียว
    python run_model.py --model tm-my-model.zip      เปลี่ยนโมเดล
    python run_model.py --list-mics                  ดูรายชื่อไมโครโฟน

ต้องการแค่ numpy และ sounddevice (ติดตั้งด้วย install.sh หรือ install.bat ที่โฟลเดอร์บนสุด)
"""
import argparse
import json
import queue
import sys
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

import numpy as np

# ============================== ตั้งค่า: แก้ตรงนี้ได้เลย ==============================

MODEL = "my-model"        # โฟลเดอร์โมเดล, ไฟล์ .zip ที่ export มา หรือลิงก์จาก Teachable Machine
SOURCE = "mic"            # "mic", ไฟล์ .wav หรือโฟลเดอร์ที่มีไฟล์ .wav
MIC_DEVICE = None         # None = ไมค์เริ่มต้น หรือใส่เลขจาก --list-mics
MIN_CONFIDENCE = 0.75     # ความมั่นใจขั้นต่ำ (0-1) ต่ำกว่านี้จะไม่นับว่าได้ยินคำนั้น
OVERLAP = 0.5             # 0-0.75 ยิ่งมากยิ่งทำนายถี่ แต่ใช้เครื่องมากขึ้น
COOLDOWN_SECONDS = 1.0    # ไม่แจ้งคำเดิมซ้ำภายในเวลานี้ เพราะหนึ่งคำอาจถูกจับได้หลายรอบ
IGNORE_LABELS = ["Background Noise", "_background_noise_"]  # คลาสที่ไม่ต้องแจ้ง


def on_result(label, confidence):
    """ถูกเรียกเมื่อได้ยินคำที่มั่นใจเกิน MIN_CONFIDENCE ใส่โค้ดของคุณที่นี่ได้เลย

    เช่น สั่งเปิดไฟเมื่อได้ยิน "hello" หรือบันทึกลงไฟล์
    """
    print(f"ได้ยิน: {label} ({confidence:.0%})")


# ==================== ส่วนด้านล่างคือการทำงานของโมเดล ปกติไม่ต้องแก้ ====================

# ค่าเดียวกับที่ Teachable Machine ใช้สร้างสเปกโตรแกรม
SAMPLE_RATE = 44100
FFT_SIZE = 2048
HOP = 1024
FRAMES = 43
BINS = 232
SILENCE_DB = -160.0
HERE = Path(__file__).resolve().parent
_n = np.arange(FFT_SIZE)
BLACKMAN = (0.42 - 0.5 * np.cos(2 * np.pi * _n / FFT_SIZE) + 0.08 * np.cos(4 * np.pi * _n / FFT_SIZE)).astype(np.float32)


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


def conv2d(x, kernel):
    """Convolution แบบ valid ของ [เวลา, ความถี่, ช่อง] ด้วย NumPy"""
    kernel_h, kernel_w = kernel.shape[:2]
    last_h, last_w = x.shape[0] - kernel_h, x.shape[1] - kernel_w
    out = 0
    for i in range(kernel_h):
        for j in range(kernel_w):
            out = out + x[i:last_h + i + 1, j:last_w + j + 1] @ kernel[i, j]
    return out


class TeachableModel:
    """อ่าน model.json กับ weights.bin ของ TensorFlow.js แล้วคำนวณตามโครงสร้างในไฟล์"""

    def __init__(self, source):
        read = read_model_files(source)
        spec = json.loads(read("model.json"))
        metadata = json.loads(read("metadata.json"))
        if "wordLabels" not in metadata:
            kind = metadata.get("packageName", "ที่ไม่รู้จัก")
            sys.exit(f"โมเดลนี้เป็นชนิด {kind} สคริปต์นี้ใช้ได้เฉพาะโมเดล Audio")
        self.labels = metadata["wordLabels"]
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
                source = outputs[child["inbound_nodes"][0][0][0]]
                outputs[child["config"]["name"]] = self.forward(child, source)
            return outputs[cfg["output_layers"][0][0]]
        if kind == "Conv2D":
            return np.maximum(conv2d(x, w[f"{name}/kernel"]) + w[f"{name}/bias"], 0)
        if kind == "MaxPooling2D":
            stride_h, stride_w = cfg["strides"]
            windows = np.lib.stride_tricks.sliding_window_view(x, cfg["pool_size"], axis=(0, 1))
            return windows[::stride_h, ::stride_w].max(axis=(-2, -1))
        if kind == "Flatten":
            return x.reshape(-1)
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

    def predict(self, spectrogram):
        """รับสเปกโตรแกรม [43, 232] แล้วคืนความน่าจะเป็นของทุกคลาส"""
        normalized = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
        return self.forward(self.topology, normalized[:, :, None])


def spectrogram(samples):
    """แปลงเสียง 44100 Hz เป็นสเปกโตรแกรม dB ได้ค่าเดียวกับ AnalyserNode ของเบราว์เซอร์

    แต่ละเฟรมใช้เสียง 2048 ตัวอย่าง และเลื่อนไปทีละ 1024 ตัวอย่าง
    """
    count = (len(samples) - FFT_SIZE) // HOP + 1
    windows = np.stack([samples[f * HOP:f * HOP + FFT_SIZE] for f in range(count)])
    spectrum = np.fft.rfft(windows * BLACKMAN, axis=1)[:, :BINS] / FFT_SIZE
    return np.maximum(20 * np.log10(np.abs(spectrum) + 1e-12), SILENCE_DB).astype(np.float32)


def to_frames(samples):
    """สเปกโตรแกรมของไฟล์เสียง ก่อนเริ่มไฟล์ถือว่าเงียบ เหมือนตอนเล่นไฟล์ในเบราว์เซอร์"""
    padded = np.concatenate([np.zeros(HOP, np.float32), samples])
    if len(padded) < FFT_SIZE:
        padded = np.pad(padded, (0, FFT_SIZE - len(padded)))
    return spectrogram(padded)


def loudest_window(frames):
    """ไฟล์ยาวกว่า 1 วินาที: เลือกช่วง 43 เฟรมที่มีพลังงานมากที่สุด"""
    if len(frames) < FRAMES:
        frames = np.vstack([frames, np.full((FRAMES - len(frames), BINS), SILENCE_DB, np.float32)])
    energy = (10 ** (frames / 10)).sum(axis=1)
    totals = np.convolve(energy, np.ones(FRAMES), mode="valid")
    start = int(np.argmax(totals))
    return frames[start:start + FRAMES]


def read_wav(path):
    """อ่านไฟล์ .wav เป็นเสียงช่องเดียว 44100 Hz"""
    with wave.open(str(path), "rb") as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width == 3:
        raw = b"".join(raw[i:i + 3] + (b"\xff" if raw[i + 2] & 0x80 else b"\0") for i in range(0, len(raw), 3))
        width = 4
    if width == 1:
        samples = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    else:
        samples = np.frombuffer(raw, {2: np.int16, 4: np.int32}[width]).astype(np.float32) / (2 ** (8 * width - 1))
    samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        count = int(round(len(samples) * SAMPLE_RATE / rate))
        positions = np.arange(count) * rate / SAMPLE_RATE
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    return samples


def run_files(model, source, args):
    path = resolve_path(source)
    if path.is_dir():
        files = sorted(path.glob("*.wav"), key=lambda f: (len(f.stem), f.stem))
    else:
        files = [path]
    if not files or not files[0].exists():
        sys.exit(f"ไม่พบไฟล์เสียงที่ {path}")
    for file in files:
        try:
            samples = read_wav(file)
        except (wave.Error, EOFError):
            print(f"{file.name}: อ่านไม่ได้ สคริปต์นี้รองรับเฉพาะไฟล์ .wav แบบ PCM")
            continue
        if not len(samples):
            print(f"{file.name}: ไฟล์นี้ไม่มีเสียง")
            continue
        scores = model.predict(loudest_window(to_frames(samples)))
        best = int(np.argmax(scores))
        print(f"{file.name}: {model.labels[best]} {scores[best]:.1%}")
        if scores[best] >= args.min_confidence and model.labels[best] not in IGNORE_LABELS:
            on_result(model.labels[best], float(scores[best]))


class Listener:
    """เก็บเสียงต่อเนื่องแล้วทำนายทุกช่วงเวลาตาม OVERLAP"""

    def __init__(self, model, min_confidence, overlap):
        self.model = model
        self.min_confidence = min_confidence
        self.step = max(1, round(FRAMES * (1 - overlap))) * HOP
        self.needed = (FRAMES + 1) * HOP
        self.buffer = np.zeros(0, np.float32)
        self.pending = 0
        self.last_seen = {}

    def feed(self, block):
        """รับเสียงชุดใหม่ และคืนผล (ป้ายชื่อ, ความมั่นใจ) เมื่อได้ยินคำ"""
        self.buffer = np.concatenate([self.buffer, block])[-self.needed:]
        self.pending += len(block)
        if len(self.buffer) < self.needed or self.pending < self.step:
            return None
        self.pending = 0
        scores = self.model.predict(spectrogram(self.buffer))
        best = int(np.argmax(scores))
        label, confidence = self.model.labels[best], float(scores[best])
        now = time.monotonic()
        if confidence < self.min_confidence or label in IGNORE_LABELS:
            return None
        if now - self.last_seen.get(label, -1e9) < COOLDOWN_SECONDS:
            return None
        self.last_seen[label] = now
        return label, confidence


def import_sounddevice():
    try:
        import sounddevice
    except OSError:
        sys.exit("ไม่พบไลบรารีเสียงของระบบ บน Linux ให้ติดตั้งด้วย: sudo apt install libportaudio2")
    except ImportError:
        sys.exit("ยังไม่ได้ติดตั้ง sounddevice ให้รัน install.sh หรือ install.bat ก่อน")
    return sounddevice


def run_mic(model, args):
    sounddevice = import_sounddevice()
    blocks = queue.Queue()
    listener = Listener(model, args.min_confidence, args.overlap)

    def callback(data, frames, timing, status):
        blocks.put(data[:, 0].copy())

    try:
        stream = sounddevice.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=HOP, device=args.mic,
                                         dtype="float32", callback=callback)
    except (sounddevice.PortAudioError, ValueError) as err:
        sys.exit(f"เปิดไมโครโฟนไม่ได้ ({err}) ลองดูเลขไมค์ด้วย --list-mics หรืออนุญาตให้ Terminal ใช้ไมโครโฟน")
    with stream:
        print("กำลังฟัง... กด Ctrl+C เพื่อออก")
        try:
            while True:
                result = listener.feed(blocks.get())
                if result:
                    on_result(*result)
        except KeyboardInterrupt:
            pass


def load_model(source):
    """โหลดโมเดล ถ้าไม่สำเร็จให้แจ้งสาเหตุแทนการแสดง Traceback"""
    try:
        return TeachableModel(source)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as err:
        sys.exit(f"โหลดโมเดลจาก {source} ไม่สำเร็จ ตรวจสอบว่าลิงก์ถูกต้อง และมี model.json, metadata.json, weights.bin ครบ ({err})")


def main():
    parser = argparse.ArgumentParser(description="รันโมเดล Audio จาก Teachable Machine")
    parser.add_argument("--model", default=MODEL, help="โฟลเดอร์, ไฟล์ .zip หรือลิงก์ของโมเดล")
    parser.add_argument("--source", default=SOURCE, help='"mic", ไฟล์ .wav หรือโฟลเดอร์')
    parser.add_argument("--mic", type=int, default=MIC_DEVICE, help="เลขไมโครโฟน")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE, help="ความมั่นใจขั้นต่ำ 0-1")
    parser.add_argument("--overlap", type=float, default=OVERLAP, help="0-0.75")
    parser.add_argument("--list-mics", action="store_true", help="แสดงรายชื่อไมโครโฟนแล้วออก")
    args = parser.parse_args()

    if args.list_mics:
        print(import_sounddevice().query_devices())
        return
    model = load_model(args.model)
    print(f"โหลดโมเดลแล้ว: {', '.join(model.labels)}")
    if args.source == "mic":
        run_mic(model, args)
    else:
        run_files(model, args.source, args)


if __name__ == "__main__":
    main()
