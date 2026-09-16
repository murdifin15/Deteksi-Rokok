"""
Sistem Deteksi Rokok Real-Time
YOLO11 + ByteTrack (sinkron, tanpa threading inferensi)

Setiap frame:
  1. Baca frame terbaru dari kamera
  2. Jalankan model.track() langsung (sinkron)
  3. Gambar bounding box dari hasil inferensi
  4. Tampilkan

Karena setiap frame diproses langsung, bounding box
SELALU tepat di posisi objek — tidak pernah nyangkut.
"""

import os
import sys
import time
import signal
import argparse
import threading
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistem Deteksi Rokok Real-Time (YOLO11)"
    )
    parser.add_argument("--model",   type=str,   default="best model.pt")
    parser.add_argument("--source",  type=str,   default="1")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--label",   type=str,   default="Rokok")
    parser.add_argument("--imgsz",   type=int,   default=320,
                        help="Ukuran inference. 320 = cepat di CPU.")
    parser.add_argument("--test",    action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Camera thread — membaca frame terus-menerus agar buffer
# kamera selalu terkuras dan kita selalu dapat frame terbaru
# ─────────────────────────────────────────────────────────────
class CameraReader(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap     = cap
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, f = self.cap.read()
            if ret and f is not None and f.size > 0:
                with self.lock:
                    self.frame = f

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, conf, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 230), 3)

    cl = max(4, min(22, (x2 - x1) // 5, (y2 - y1) // 5))
    for p1, p2 in [
        ((x1,y1),(x1+cl,y1)), ((x1,y1),(x1,y1+cl)),
        ((x2,y1),(x2-cl,y1)), ((x2,y1),(x2,y1+cl)),
        ((x1,y2),(x1+cl,y2)), ((x1,y2),(x1,y2-cl)),
        ((x2,y2),(x2-cl,y2)), ((x2,y2),(x2,y2-cl)),
    ]:
        cv2.line(frame, p1, p2, (0, 240, 255), 3)

    text = f"{label}  {conf*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ly = max(0, y1 - th - 10)
    cv2.rectangle(frame, (x1, ly), (x1+tw+12, ly+th+10), (0, 0, 180), -1)
    cv2.putText(frame, text, (x1+6, ly+th+3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)


def draw_ui(frame, boxes, conf_thr, fps, label):
    h, w = frame.shape[:2]
    det  = len(boxes) > 0

    cv2.rectangle(frame, (0, 0), (w, 50),
                  (0, 0, 160) if det else (25, 110, 25), -1)
    status = (f"ROKOK TERDETEKSI!  [JUMLAH: {len(boxes)}]"
              if det else "STATUS: AMAN  (Tidak ada rokok)")
    cv2.putText(frame, status, (16, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h-32), (w, h), (10, 10, 10), -1)
    info = (f"FPS: {fps:.1f}  |  Conf: {conf_thr*100:.0f}%  |  "
            "[Q] Keluar  [S] Simpan  [+/-] Sensitivitas")
    cv2.putText(frame, info, (12, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190,190,190), 1, cv2.LINE_AA)

    for b in boxes:
        draw_box(frame, int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4], label)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Model
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}"); sys.exit(1)

    print(f"[INFO] Memuat model: {model_path} ...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}"); sys.exit(1)
    print("[INFO] Model berhasil dimuat!")

    if hasattr(model.model, 'names'):
        model.model.names = {0: args.label}
    print(f"[INFO] Label: {model.names}")

    # Test mode
    if args.test:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(dummy, (200, 180), (440, 240), (200, 200, 200), -1)
        r = model.predict(dummy, conf=args.conf, verbose=False)
        print(f"[SUCCESS] Test OK. Deteksi: {len(r[0].boxes)}"); return

    # Camera
    src = int(args.source) if args.source.isdigit() else args.source
    cap = None
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    def try_open_cam(cam_src):
        for backend in backends:
            c = cv2.VideoCapture(cam_src, backend)
            if c.isOpened():
                return c
        return None

    cap = try_open_cam(src)
    if not cap or not cap.isOpened():
        if isinstance(src, int) and src != 0:
            print(f"[WARN] Kamera source {src} tidak ditemukan, mencoba source 0...")
            cap = try_open_cam(0)
            if cap and cap.isOpened():
                args.source = "0"
        elif isinstance(src, int) and src == 0:
            print(f"[WARN] Kamera source 0 tidak ditemukan, mencoba source 1...")
            cap = try_open_cam(1)
            if cap and cap.isOpened():
                args.source = "1"

    if not cap or not cap.isOpened():
        print("[ERROR] Kamera tidak bisa dibuka. Pastikan webcam terhubung dan tidak digunakan aplikasi lain.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FPS,          30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    win = "DETEKSI ROKOK REAL-TIME (YOLO11)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    print(f"\n{'='*60}")
    print(f"       SISTEM DETEKSI ROKOK REAL-TIME")
    print(f"{'='*60}")
    print(f"  Model  : {os.path.basename(model_path)}")
    print(f"  Kamera : source={args.source} ({actual_w}x{actual_h})")
    print(f"  imgsz  : {args.imgsz}  |  conf : {args.conf}")
    print(f"  Mode   : Sinkron (setiap frame diproses)")
    print(f"{'-'*60}")
    print(f"  [Q/ESC] Keluar  [S] Simpan  [+/-] Sensitivitas")
    print(f"{'='*60}\n")

    captures = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
    os.makedirs(captures, exist_ok=True)

    # Camera reader thread (hanya untuk drain buffer)
    cam = CameraReader(cap)
    cam.start()

    conf     = args.conf
    fps      = 0.0
    prev_t   = time.time()

    # Persistence: saat model miss sesaat, tetap tampilkan box terakhir
    PERSIST_MAX  = 3        # tahan box selama max 3 frame saat miss
    last_boxes   = []       # box terakhir yang valid
    miss_count   = 0        # berapa frame berturut-turut model miss

    # Graceful shutdown
    shutdown = threading.Event()
    def sig_handler(sig, f):
        print("\n[INFO] Ctrl+C diterima...")
        shutdown.set()
    signal.signal(signal.SIGINT, sig_handler)

    # ─────────────────────────────────────────────────────────
    # MAIN LOOP — sinkron, sederhana, pasti benar
    #
    # Setiap iterasi:
    #   frame → model.track() → boxes → draw → display
    #
    # Karena inferensi dan display terjadi pada FRAME YANG SAMA,
    # bounding box SELALU di posisi yang benar.
    # ByteTrack (persist=True) menangani tracking antar frame.
    # Persistence 3 frame mencegah flicker saat miss sesaat.
    # ─────────────────────────────────────────────────────────
    while not shutdown.is_set():
        frame = cam.get()
        if frame is None:
            time.sleep(0.002)
            continue

        # ── Inference + tracking langsung pada frame ini ──
        results = model.track(
            frame,
            conf=conf,
            iou=args.iou,
            imgsz=args.imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        # ── Parse boxes ──
        boxes = []
        if results[0].boxes is not None:
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                c = float(b.conf[0].cpu().numpy())
                boxes.append((x1, y1, x2, y2, c))

        # ── Persistence: tahan box saat miss sesaat ──
        if boxes:
            last_boxes = boxes
            miss_count = 0
        else:
            miss_count += 1
            if miss_count <= PERSIST_MAX:
                boxes = last_boxes   # tampilkan box terakhir
            else:
                last_boxes = []      # sudah terlalu lama miss, benar-benar hilang

        # ── FPS ──
        now  = time.time()
        dt   = now - prev_t
        if dt > 0:
            fps = 0.8 * fps + 0.2 / dt
        prev_t = now

        # ── Draw & display ──
        draw_ui(frame, boxes, conf, fps, args.label)
        cv2.imshow(win, frame)

        # ── Keyboard ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("[INFO] Menghentikan program..."); break
        elif key in (ord('s'), ord('S')):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(captures, f"deteksi_rokok_{ts}.jpg")
            cv2.imwrite(fn, frame)
            print(f"[TERSIMPAN] {fn}")
        elif key in (ord('+'), ord('=')):
            conf = min(0.95, conf + 0.05)
            print(f"[KONTROL] Conf naik → {conf*100:.0f}%")
        elif key in (ord('-'), ord('_')):
            conf = max(0.05, conf - 0.05)
            print(f"[KONTROL] Conf turun → {conf*100:.0f}%")

    cam.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program selesai.")


if __name__ == "__main__":
    main()
