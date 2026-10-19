"""
5.1 YOLO ile nesne tespiti — karedeki araç, yaya gibi nesnelerin bulunması.

Ne işe yarar:
    Önümüzde ne olduğunu bilmek için nesneleri bulur. Renk ya da kenar yöntemleri "bu bir araç"
    diyemez; bunu derin öğrenme modeli yapar.

Nasıl çalışır:
    Hazır YOLO modeli (yolo11n.pt) bir kez yüklenir. Her kare düzeltilip modele verilir; her nesne için kutu
    (x1, y1, x2, y2), sınıf ve güven kaydedilir. Cihaz kendiliğinden seçilir: GPU varsa model kısa bir ölçümle
    GPU'da ve CPU'da denenir, hızlı olan kullanılır; GPU yoksa CPU.

Bağlantılar:
    Alır : 1.1'in kalibrasyon.npz dosyası; kutular şerit hattıyla aynı düzeltilmiş görüntüde olsun
           diye.
    Verir: <ad>_tespit.npz (boxes, class_name, confidence, frame) -> 5.2 ve 5.3.
    Ana hat: main.py modeli load_model() ile bir kez yükler, her karede detect() çağırır.

Parametreler:
    --input       Gerçek veri: video ya da görüntü (varsayılan kolay video). [DEĞİŞTİR]
    --model       Hazır model dosyası: data/hazir_veri/yolo_modeli/yolo11n.pt.
    --calibration Başka gereksinimden: 1.1'in çıktısı.
    --iou         Başka gereksinimden: 5.4'ün analiziyle seçilen değer (0.7).
    --device      Ayar: cihaz; varsayılan auto (GPU ve CPU kısa bir ölçümle karşılaştırılır, hızlı olan seçilir).
    --choose-device   Yalnız cihaz seçimini yapar ve çıkar (docker/baslat.sh kurulumda çalıştırır).
    --conf, --classes   Ayar (asıl güven ve sınıf filtresi 5.3'te).
"""

import argparse
import json
import math
import os
import platform
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# AYARLAR
# Komut satırı argümanları bu varsayılanları ezer (python ... --help).
# [DEĞİŞTİR] işaretli olanları kendi verine göre ayarla.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / Path(__file__).resolve().parent.name

# [DEĞİŞTİR] Renkli kare: video (önerilen, İnşa), tek görüntü ya da görüntü klasörü.
INPUT_PATH = DATA_DIR / "ortam_verisi" / "videolar" / "kolay_video" / "kolay_video_iyi_isik.mp4"

# [DEĞİŞTİR] Hazır COCO ağırlıklı nano model. yolov8n.pt de kullanılabilir.
MODEL_PATH = DATA_DIR / "hazir_veri" / "yolo_modeli" / "yolo11n.pt"

# [DEĞİŞTİR] Kamera kalibrasyonu (.npz). None ise undistort atlanır.
CALIBRATION_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "kalibrasyon.npz"

CONF = 0.25                   # güven eşiği (ultralytics varsayılanı)
IOU = 0.7                     # NMS IoU eşiği (ultralytics varsayılanı)
IMGSZ = 640                   # ağ giriş boyutu (px); kutular girdinin kendi piksellerinde döner
DEVICE = "auto"               # "auto": GPU ve CPU ölçülür, hızlı olan; ya da "cpu" / "0" (GPU numarası)
CLASSES = None                # None: tüm COCO sınıfları; ör. ("car", "person", "truck")

# Video: her FRAME_STEP karede bir, en çok MAX_FRAMES kare (None = tümü).
FRAME_STEP = 1
MAX_FRAMES = None
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


def import_ultralytics(offline):
    """[EK] (kütüphane) ultralytics; yoksa imajın yeniden kurulması gerektiğini söyler.

    offline: True ise ultralytics ağa çıkmaz (YOLO_OFFLINE, içe aktarmadan önce okunur);
    model indirmesi istendiğinde False verilir.
    """
    # [EK] Kullanım verisi gönderimini ve gereksiz ağ denetimini kapat (tespit için gerekmez).
    if offline:
        os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    try:
        from ultralytics import YOLO, settings
    except ImportError:
        raise InputError(
            "ultralytics kurulu değil.\n"
            "  -> İmajı yeniden kur (host'ta): docker compose -f docker/docker-compose.yml build\n"
            "     ultralytics docker/requirements.txt'e 5.1 için eklendi.") from None
    try:
        if settings.get("sync", False):
            settings.update({"sync": False})
    except Exception:  # ayar dosyası yazılamazsa tespit yine çalışır
        pass
    return YOLO


# [EK] 2.2'deki kalibrasyon yükleyicisinin kopyası (klasörler arası import yapılmaz).
class CameraCalibration:
    """Kamera kalibrasyonunu yükler ve kareleri düzeltir (undistort)."""

    REQUIRED_KEYS = ("camera_matrix", "dist_coeffs", "image_size")

    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise InputError(
                f"Kalibrasyon dosyası bulunamadı: {path}\n"
                "  -> --calibration ile doğru yolu ver ya da undistort'suz çalıştır "
                "(--calibration vermeden).")
        try:
            with np.load(path) as data:
                missing = [k for k in self.REQUIRED_KEYS if k not in data.files]
                if missing:
                    raise ValueError(f"eksik alanlar: {', '.join(missing)}")
                self.camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
                self.dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).ravel()
                self.image_size = tuple(int(v) for v in data["image_size"])
            if self.camera_matrix.shape != (3, 3):
                raise ValueError(f"camera_matrix 3x3 olmalı, verilen {self.camera_matrix.shape}")
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise InputError(
                f"Kalibrasyon dosyası okunamadı: {path} ({exc})\n"
                "  -> Beklenen alanlar: camera_matrix (3x3), dist_coeffs, image_size.") from None
        self.path = path

    def check_size(self, width, height):
        if (int(width), int(height)) != self.image_size:
            raise InputError(
                f"Görüntü boyutu {width}x{height}, ancak kalibrasyon "
                f"{self.image_size[0]}x{self.image_size[1]} için yapıldı ({self.path}).\n"
                "  -> Kalibrasyonu bu çözünürlükte yap ya da kareleri aynı boyutta ver.")

    def undistort(self, frame):
        self.check_size(frame.shape[1], frame.shape[0])
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)


# [EK] Cihaz seçimi. "auto": NVIDIA GPU yoksa CPU. GPU varsa YOLO önce GPU'da, sonra CPU'da kısa bir
# ölçümle denenir ve hızlı olan seçilir; zayıf GPU ile güçlü CPU'su olan bir bilgisayarda CPU seçilebilir.
# Ana hatta YOLO, CPU'yu şerit koluyla paylaştığı için CPU süresi CPU_SHARE_FACTOR ile çarpılıp karşılaştırılır
# (ölçüm: tek başına 13.6 ms/kare, ana hatta ~22 ms). Sonuç her donanım ve model için ayrı, geçici klasöre
# yazılır ve konteyner yeniden kurulana kadar tekrar ölçülmez (büyük bir model başka bir sonuç verebilir).
# Proje klasörüne yazılmaz: başka bir bilgisayarda yeniden ölçülür.
DEVICE_CACHE = Path(tempfile.gettempdir()) / "ads_cihaz_secimi.json"
BENCH_WARMUP, BENCH_FRAMES = 3, 10
CPU_SHARE_FACTOR = 1.5


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_device(device):
    """Elle verilen cihazı denetler; GPU istenip de yoksa anlaşılır hata verir."""
    device = str(device).strip().lower()
    if device not in ("auto", "cpu") and not cuda_available():
        raise InputError(
            f"GPU istendi (--device {device}) ama kullanılabilir bir NVIDIA GPU yok.\n"
            "  -> --device auto ya da --device cpu ver; GPU için konteyneri bash docker/baslat.sh ile başlat.")
    return device


def hardware_key(imgsz, model_name=""):
    """Ölçümün geçerli olduğu donanım ve model: torch sürümü, GPU, CPU, çekirdek sayısı, model dosyası."""
    import torch
    cpu = platform.processor() or "?"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as info:
            cpu = next((line.split(":", 1)[1].strip() for line in info if line.startswith("model name")), cpu)
    except OSError:
        pass
    return (f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | {cpu} x{os.cpu_count()} | imgsz {imgsz}"
            f" | model {model_name}")


def time_per_frame_ms(model_loader, device, imgsz):
    """YOLO'nun bu cihazda 1280x720 bir karede ortalama süresi (ms); ilk kareler ısınma sayılmaz."""
    import torch
    model = model_loader()
    frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    for i in range(BENCH_WARMUP + BENCH_FRAMES):
        if i == BENCH_WARMUP:
            if device != "cpu":
                torch.cuda.synchronize()
            start = time.perf_counter()
        model.predict(source=frame, imgsz=imgsz, device=device, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / BENCH_FRAMES


def read_device_cache():
    """Kayıtlı ölçümler: {anahtar: {"device", "gpu_ms", "cpu_ms", "model"}}; dosya yoksa, bozuksa ya da eski
    (tek kayıtlı) biçimdeyse boş (yeniden ölçülür)."""
    try:
        saved = json.loads(DEVICE_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) and "key" not in saved else {}


def resolve_device(device, model_loader=None, imgsz=IMGSZ, remeasure=False, model_name=""):
    """Kullanılacak cihaz: "cpu" ya da GPU numarası. model_loader: her çağrıda yeni bir YOLO modeli döndürür;
    model_name: ölçümün hangi model için saklanacağı (her model ayrı ölçülür)."""
    device = check_device(device)
    if device != "auto":
        return device
    if not cuda_available():
        return "cpu"
    if model_loader is None:          # ölçüm yapılamıyorsa GPU
        return "0"
    key = hardware_key(imgsz, model_name)
    cache = read_device_cache()
    saved = cache.get(key)
    if isinstance(saved, dict) and not remeasure:
        how = "bu bilgisayar ve model için önceki ölçüm"
    else:
        cpu_ms = time_per_frame_ms(model_loader, "cpu", imgsz)
        try:
            gpu_ms = time_per_frame_ms(model_loader, "0", imgsz)
        except Exception as exc:      # GPU çalışmıyorsa CPU
            print(f"UYARI: GPU denemesi başarısız ({type(exc).__name__}: {exc}); CPU kullanılacak.")
            gpu_ms = float("inf")
        choice = "0" if gpu_ms <= CPU_SHARE_FACTOR * cpu_ms else "cpu"
        saved, how = {"device": choice, "gpu_ms": gpu_ms, "cpu_ms": cpu_ms, "model": model_name}, "ölçüldü"
        cache[key] = saved
        try:
            DEVICE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        except (OSError, ValueError):
            pass
    print(f"Cihaz seçimi ({how}{', ' + model_name if model_name else ''}): GPU {saved['gpu_ms']:.1f} ms/kare, "
          f"CPU {saved['cpu_ms']:.1f} ms/kare "
          f"(CPU şerit koluyla paylaşıldığı için x{CPU_SHARE_FACTOR:g} sayılır) -> {device_label(saved['device'])}")
    return saved["device"]


def device_label(device):
    """Ekrana yazılacak cihaz adı: "cpu" ya da "GPU 0 (kartın adı)"."""
    if device == "cpu":
        return "cpu"
    try:
        import torch
        return f"GPU {device} ({torch.cuda.get_device_name(int(device))})"
    except Exception:
        return f"GPU {device}"


# --------------------------------------------------------------------------- model
def load_model(args):
    path = Path(args.model)
    YOLO = import_ultralytics(offline=not (args.download_model and not path.is_file()))
    if not path.is_file():
        if not args.download_model:
            raise InputError(
                f"Model dosyası bulunamadı: {path}\n"
                "  -> Bir kez indir: python yolo_tespit.py --download-model\n"
                "     ya da --model ile var olan bir .pt dosyası ver.")
        # [EK] Açıkça istendiğinde ultralytics'in indiricisiyle GitHub'dan indir.
        from ultralytics.utils.downloads import attempt_download_asset
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            attempt_download_asset(str(path))
        except Exception as exc:
            raise InputError(f"Model indirilemedi: {path.name} ({exc})\n"
                             "  -> Ağ bağlantısını ve model adını (ör. yolo11n.pt, yolov8n.pt) "
                             "kontrol et.") from None
        if not path.is_file():
            raise InputError(f"Model indirilemedi: {path}\n"
                             "  -> Model adı ultralytics'in hazır modellerinden biri olmalı "
                             "(ör. yolo11n.pt, yolov8n.pt).")
        print(f"Model indirildi: {path}")
    try:
        model = YOLO(str(path))
    except Exception as exc:
        raise InputError(f"Model yüklenemedi: {path} ({type(exc).__name__}: {exc})\n"
                         "  -> Dosyanın bozuk olmadığını ve bir YOLO .pt dosyası olduğunu "
                         "kontrol et.") from None
    return model


def class_ids(model, names):
    """[EK] Sınıf adlarını numaraya çevirir; yanlış ad için geçerli adları listeler."""
    if not names:
        return None
    lookup = {n: i for i, n in model.names.items()}
    unknown = [n for n in names if n not in lookup]
    if unknown:
        raise InputError(f"Bilinmeyen sınıf adı: {', '.join(unknown)}\n"
                         f"  -> Geçerli adlar: {', '.join(model.names.values())}")
    return [lookup[n] for n in names]


def detect(model, frame, args, classes):
    """Tek kare -> (kutular float32[K, 4], sınıf int64[K], güven float32[K])."""
    try:
        result = model.predict(source=frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                               device=args.device, classes=classes, verbose=False)[0]
    except Exception as exc:
        raise InputError(f"YOLO tahmini başarısız ({type(exc).__name__}: {exc})\n"
                         "  -> --device'ı kontrol et (auto, cpu ya da GPU numarası); "
                         "GPU'da sorun varsa --device cpu dene.") from None
    boxes = result.boxes
    return (boxes.xyxy.cpu().numpy().astype(np.float32), boxes.cls.cpu().numpy().astype(np.int64),
            boxes.conf.cpu().numpy().astype(np.float32))


# --------------------------------------------------------------------------- girdi / çıktı
def read_image(path):
    img = cv2.imread(str(path), flags=cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(
            f"Görüntü okunamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(IMAGE_EXTS))}) olduğunu kontrol et.")
    return img


def iter_frames(path, args):
    """(ad, kare) çiftleri: tek görüntü, klasör ya da video."""
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise InputError(f"Klasörde görüntü yok: {path}\n"
                             f"  -> Desteklenen uzantılar: {', '.join(sorted(IMAGE_EXTS))}")
        for file in files:
            yield file.stem, read_image(file)
        return
    if path.suffix.lower() not in VIDEO_EXTS:
        yield path.stem, read_image(path)
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputError(
            f"Video açılamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(VIDEO_EXTS))}) olduğunu kontrol et.")
    index = used = 0
    try:
        while args.max_frames is None or used < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % args.frame_step == 0:
                used += 1
                yield f"kare{index:05d}", frame
            index += 1
    finally:
        cap.release()
    if used == 0:
        raise InputError(f"Videodan hiç kare okunamadı: {path}")


def video_fps(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if math.isfinite(fps) and fps > 0 else 25.0


def class_color(cid):
    """Sınıfa göre sabit renk (BGR)."""
    rng = np.random.default_rng(cid * 7919 + 17)
    return tuple(int(v) for v in rng.integers(60, 256, 3))


def draw_compare(frame, boxes, cls, conf, names, undistorted):
    width = frame.shape[1]
    fs = 0.6 * max(width, 640) / 1280.0 + 0.2
    th = max(1, round(2 * width / 1280.0))
    left, right = frame.copy(), frame.copy()
    for (x1, y1, x2, y2), c, p in zip(boxes, cls, conf):
        color = class_color(int(c))
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(right, p1, p2, color, thickness=th + 1)
        label = f"{names[int(c)]} {p:.2f}"         # COCO adları ASCII
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y_text = max(tht + 4, p1[1] - 4)
        cv2.rectangle(right, (p1[0], y_text - tht - 4), (p1[0] + tw + 4, y_text + base - 2), color,
                      thickness=-1)
        cv2.putText(right, label, (p1[0] + 2, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    thickness=th, lineType=cv2.LINE_AA)
    for img, text in ((left, "Girdi (duzeltilmis)" if undistorted else "Girdi"),
                      (right, f"YOLO: {len(boxes)} nesne")):
        org = (10, round(30 * fs / 0.6))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thickness=th + 2,
                    lineType=cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255), thickness=th,
                    lineType=cv2.LINE_AA)
    return cv2.hconcat([left, right])


def summarize(cls, conf, names):
    if len(cls) == 0:
        return "nesne yok"
    parts = []
    for c in sorted(set(cls.tolist()), key=lambda c: -int((cls == c).sum())):
        sel = conf[cls == c]
        parts.append(f"{names[c]} x{len(sel)} (güven {sel.min():.2f}-{sel.max():.2f})")
    return ", ".join(parts)


# --------------------------------------------------------------------------- ana akış
def run(args):
    path = Path(args.input)
    if args.download_model and not path.exists():
        load_model(args)          # yalnızca indir ve yüklenebildiğini doğrula
        print("Model hazır. Tespit için --input ile girdi ver.")
        return
    if args.choose_device:
        load_model(args)          # model var mı, yüklenebiliyor mu
        if not cuda_available():
            print("NVIDIA GPU bulunamadı; YOLO CPU'da çalışacak.")
        device = resolve_device("auto", lambda: load_model(args), args.imgsz, remeasure=True,
                                model_name=Path(str(args.model)).name)
        print(f"Seçilen cihaz: {device_label(device)}")
        return
    if not path.exists():
        raise InputError(
            f"Girdi bulunamadı: {path}\n"
            "  -> --input ile doğru yolu ver ya da AYARLAR'daki INPUT_PATH'i düzelt.")
    calibration = CameraCalibration(args.calibration) if args.calibration is not None else None
    model = load_model(args)
    args.device = resolve_device(args.device, lambda: load_model(args), args.imgsz,
                                 model_name=Path(str(args.model)).name)
    names = model.names
    classes = class_ids(model, args.classes)
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"Çıktı klasörü oluşturulamadı: {args.out_dir} ({exc})\n"
                         "  -> --out-dir'i kontrol et.") from None
    is_video = path.is_file() and path.suffix.lower() in VIDEO_EXTS
    compare_dir = args.out_dir / path.name if path.is_dir() else args.out_dir
    if args.compare and path.is_dir():
        compare_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {Path(args.model).name}, cihaz: {device_label(args.device)}, güven >= {args.conf:g}, "
          f"NMS IoU {args.iou:g}, imgsz {args.imgsz}; undistort: {'var' if calibration else 'yok'}; "
          f"sınıflar: {', '.join(args.classes) if args.classes else 'tümü'}")
    frames, det_frame, all_boxes, all_cls, all_conf = [], [], [], [], []
    image_size, writer, timings = None, None, []
    try:
        for name, frame in iter_frames(path, args):
            if calibration is not None:
                frame = calibration.undistort(frame)
            size = (frame.shape[1], frame.shape[0])
            if image_size is None:
                image_size = size
            elif size != image_size:
                raise InputError(
                    f"Farklı çözünürlükte kare: {name} ({size[0]}x{size[1]}), ilk kare "
                    f"{image_size[0]}x{image_size[1]}.\n"
                    "  -> Kutular piksel cinsindendir; aynı boyutta kareler ver.")
            start = time.perf_counter()
            boxes, cls, conf = detect(model, frame, args, classes)
            timings.append(time.perf_counter() - start)
            index = len(frames)
            frames.append(name)
            det_frame += [index] * len(boxes)
            all_boxes.append(boxes)
            all_cls.append(cls)
            all_conf.append(conf)
            if not is_video or len(frames) <= 3:
                print(f"{name}: {summarize(cls, conf, names)}")
            if args.compare:
                img = draw_compare(frame, boxes, cls, conf, names, calibration is not None)
                if is_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(args.out_dir / f"{path.stem}_tespit_karsilastirma.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                                 video_fps(path) / args.frame_step,
                                                 (img.shape[1], img.shape[0]), isColor=True)
                        if not writer.isOpened():
                            raise InputError("Karşılaştırma videosu yazılamadı.\n"
                                             "  -> --out-dir'i ve yazma iznini kontrol et.")
                    writer.write(img)
                elif not cv2.imwrite(str(compare_dir / f"{name}_tespit_karsilastirma.png"), img):
                    raise InputError(f"Görsel yazılamadı: {compare_dir}\n  -> --out-dir'i kontrol et.")
    finally:
        if writer is not None:
            writer.release()

    cls_all = np.concatenate(all_cls) if all_cls else np.empty(0, np.int64)
    conf_all = np.concatenate(all_conf) if all_conf else np.empty(0, np.float32)
    n = len(frames)
    print(f"{n} kare, {len(cls_all)} tespit: {summarize(cls_all, conf_all, names)}")
    # [EK] Hız: ilk kare modelin ısınmasını içerir, ayrı yazılır.
    if n > 1:
        steady = float(np.mean(timings[1:]))
        print(f"Hız ({device_label(args.device)}): ilk kare (ısınma) {1000 * timings[0]:.0f} ms; sonraki kareler "
              f"ortalama {1000 * steady:.1f} ms / kare ({1 / steady:.1f} fps)")
    else:
        print(f"Hız ({device_label(args.device)}): {1000 * timings[0]:.0f} ms (tek kare; modelin ısınmasını içerir)")
    out_path = args.out_dir / f"{path.stem if path.is_file() else path.name}_tespit.npz"
    try:
        np.savez(out_path, frame=np.array(frames, dtype=np.str_),
                 det_frame=np.array(det_frame, dtype=np.int64),
                 boxes=(np.concatenate(all_boxes) if all_boxes else np.empty((0, 4))).astype(np.float32).reshape(-1, 4),
                 class_id=cls_all, class_name=np.array([names[int(c)] for c in cls_all], dtype=np.str_),
                 confidence=conf_all,
                 class_names=np.array([names[i] for i in range(len(names))], dtype=np.str_),
                 image_size=np.array(image_size, dtype=np.int64), undistorted=np.bool_(calibration is not None),
                 model=np.str_(Path(args.model).name), conf=np.float64(args.conf), iou=np.float64(args.iou))
    except OSError as exc:
        raise InputError(f"Sonuç yazılamadı: {out_path} ({exc})") from None
    print(f"Kaydedildi: {out_path}")


class TurkishArgumentParser(argparse.ArgumentParser):
    """Argüman hatasında "HATA: ..." yazar ve 1 koduyla çıkar."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"HATA: Geçersiz komut satırı argümanı: {message}\n"
              "  -> Seçenekler için --help kullan.", file=sys.stderr)
        sys.exit(1)


def parse_args(argv=None):
    p = TurkishArgumentParser(
        description="YOLO (ultralytics, COCO nano) ile renkli karede nesne tespiti: kutu, sınıf, güven.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Video, görüntü ya da görüntü klasörü (varsayılan: {INPUT_PATH})")
    p.add_argument("--model", type=Path, default=MODEL_PATH,
                   help=f"YOLO ağırlıkları .pt (varsayılan: {MODEL_PATH})")
    p.add_argument("--choose-device", action="store_true",
                   help="Yalnız cihaz seçimini yap: YOLO'yu GPU ve CPU'da ölç, hızlı olanı kaydet ve çık")
    p.add_argument("--download-model", action="store_true",
                   help="Model dosyası yoksa ultralytics'in hazır modelini indir")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--calibration", type=Path, default=CALIBRATION_PATH,
                   help="Kamera kalibrasyonu .npz; verilirse YOLO düzeltilmiş karede çalışır")
    p.add_argument("--conf", type=float, default=CONF, help=f"Güven eşiği, 0-1 (varsayılan: {CONF})")
    p.add_argument("--iou", type=float, default=IOU, help=f"NMS IoU eşiği, 0-1 (varsayılan: {IOU})")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help=f"Ağ giriş boyutu, px (varsayılan: {IMGSZ})")
    p.add_argument("--device", default=DEVICE,
                   help=f"Cihaz: auto (GPU ve CPU ölçülür, hızlı olan), cpu ya da GPU numarası (varsayılan: {DEVICE})")
    p.add_argument("--classes", nargs="+", default=list(CLASSES) if CLASSES else None,
                   help="Yalnız bu sınıflar (COCO adları, ör. car person truck); varsayılan: tümü")
    p.add_argument("--compare", action="store_true", help="Girdi ve kutulu kareyi yan yana kaydet")
    p.add_argument("--frame-step", type=int, default=FRAME_STEP,
                   help=f"Videoda her N karede bir (varsayılan: {FRAME_STEP})")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                   help="Videoda en çok kaç kare (varsayılan: tümü)")
    return p.parse_args(argv)


def validate_args(args):
    for name, value in (("--conf", args.conf), ("--iou", args.iou)):
        if not (math.isfinite(value) and 0 < value <= 1):
            raise InputError(f"{name} 0 < değer <= 1 olmalı, verilen: {value}")
    if args.imgsz < 32 or args.imgsz % 32:
        raise InputError(f"--imgsz 32'nin pozitif katı olmalı (YOLO adımı), verilen: {args.imgsz}")
    if args.frame_step <= 0:
        raise InputError(f"--frame-step pozitif olmalı, verilen: {args.frame_step}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise InputError(f"--max-frames pozitif olmalı, verilen: {args.max_frames}")
    args.device = check_device(args.device)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    run(args)


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
