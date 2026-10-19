"""
5.2 Bounding Box koordinatları — kutuların piksel koordinatları ve alt-orta nokta.

Ne işe yarar:
    Nesneyi çevreleyen kutuyu sayılarla verir. Kutunun alt-orta noktası nesnenin yola değdiği yerdir;
    şeritle ilişki ve mesafe bu noktayla hesaplanır.

Nasıl çalışır:
    5.1'in kutularından (x1, y1, x2, y2), (merkez, genişlik, yükseklik) ve alt-orta nokta
    ((x1 + x2) / 2, y2) hesaplanır. Görüntü kenarına değen kutular işaretlenir; alt kenara değiyorsa
    temas noktası görünmüyor demektir.

Bağlantılar:
    Alır : 5.1'in <ad>_tespit.npz dosyası.
    Verir: <ad>_kutular.npz (xyxy, xywh, bottom_center) -> 5.3 ve 5.5.
    Ana hat: main.py her karede 5.3'ün filtresinden sonra box_coordinates() çağırır.

Parametreler:
    --input       Başka gereksinimden: 5.1'in çıktısı (varsayılan kolay videonun tespitleri).
    --source, --calibration   Yalnız çizim için: kaynak video (gerçek veri) ve 1.1'in kalibrasyonu.
    --edge-margin, --csv, --compare   Ayar.
"""

import argparse
import csv
import math
import sys
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
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUT_DIR = OUTPUTS / Path(__file__).resolve().parent.name

# [DEĞİŞTİR] 5.1'in tespit dosyası.
INPUT_PATH = OUTPUTS / "5.1 YOLO ile nesne tespiti" / "kolay_video_iyi_isik_tespit.npz"

# [DEĞİŞTİR] 5.1'e verilen kaynak (yalnız --compare çizimi için): görüntü, klasör ya da video.
SOURCE_PATH = DATA_DIR / "ortam_verisi" / "videolar" / "kolay_video" / "kolay_video_iyi_isik.mp4"

# [DEĞİŞTİR] 5.1 kalibrasyonla çalıştıysa aynı kalibrasyon (çizim düzeltilmiş kareye yapılır).
CALIBRATION_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "kalibrasyon.npz"

# Kutu bir kenara bu kadar px'ten yakınsa "kenara değiyor" sayılır.
EDGE_MARGIN_PX = 2.0
# Görüntü dışına bu kadar px'ten fazla taşan kutu geçersiz sayılır (çözünürlük uyuşmazlığı).
OUT_OF_BOUNDS_PX = 2.0
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
REQUIRED_KEYS = ("frame", "det_frame", "boxes", "class_id", "class_name", "confidence", "image_size")


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


# --------------------------------------------------------------------------- koordinatlar
def box_coordinates(xyxy, image_size, edge_margin):
    """xyxy (M x 4, px) -> tüm biçimler. Gereksinimin özü: merkez biçimi ve alt-orta nokta."""
    width, height = image_size
    x1, y1, x2, y2 = (xyxy[:, i].astype(np.float64) for i in range(4))
    xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)
    bottom_center = np.stack([(x1 + x2) / 2, y2], axis=1)
    # [EK] Oranlı koordinatlar (ultralytics xyxyn / xywhn ile aynı tanım).
    norm = np.array([width, height, width, height], dtype=np.float64)
    # [EK] Kenar işaretleri: kesik kutunun alt-orta noktası gerçek temas noktası değildir.
    left, top = x1 <= edge_margin, y1 <= edge_margin
    right, bottom = x2 >= width - edge_margin, y2 >= height - edge_margin
    return {
        "xyxy": xyxy.astype(np.float32),
        "xywh": xywh.astype(np.float32),
        "xyxyn": (xyxy / norm).astype(np.float32),
        "xywhn": (xywh / norm).astype(np.float32),
        "bottom_center": bottom_center.astype(np.float32),
        "area": ((x2 - x1) * (y2 - y1)).astype(np.float32),     # [EK] kaba yakınlık ipucu
        "truncated": left | top | right | bottom,
        "bottom_truncated": bottom,
    }


# --------------------------------------------------------------------------- girdi
def load_detections(path, out_of_bounds):
    path = Path(path)
    if not path.is_file():
        raise InputError(
            f"Tespit dosyası bulunamadı: {path}\n"
            "  -> Önce 5.1 yolo_tespit.py'yi çalıştır ya da --input ile <ad>_tespit.npz ver.")
    try:
        with np.load(path) as data:
            missing = [k for k in REQUIRED_KEYS if k not in data.files]
            if missing:
                raise ValueError(f"eksik alanlar: {', '.join(missing)}")
            det = {k: data[k] for k in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise InputError(f"Tespit dosyası okunamadı: {path} ({exc})\n"
                         "  -> 5.1'in ürettiği <ad>_tespit.npz dosyasını ver.") from None
    boxes = np.asarray(det["boxes"], dtype=np.float64).reshape(-1, 4)
    m, n = len(boxes), len(det["frame"])
    lengths = {k: len(det[k]) for k in ("det_frame", "class_id", "class_name", "confidence")}
    if any(v != m for v in lengths.values()):
        raise InputError(f"Tespit dizilerinin uzunlukları tutarsız (kutu {m}, {lengths}): {path}")
    if m and (det["det_frame"].min() < 0 or det["det_frame"].max() >= n):
        raise InputError(f"det_frame kare listesinin ({n} kare) dışında: {path}")
    size = tuple(int(v) for v in np.asarray(det["image_size"]).ravel())
    if len(size) != 2 or min(size) <= 0:
        raise InputError(f"image_size geçersiz: {size} ({path})")
    # [EK] Geçerlilik: ters kutu ya da görüntü dışına taşma çözünürlük uyuşmazlığı demektir.
    if m:
        if (boxes[:, 2] < boxes[:, 0]).any() or (boxes[:, 3] < boxes[:, 1]).any():
            raise InputError(f"Geçersiz kutu (x2 < x1 ya da y2 < y1): {path}")
        lo, hi = boxes.min(axis=0), boxes.max(axis=0)
        if (lo[[0, 1]] < -out_of_bounds).any() or hi[2] > size[0] + out_of_bounds or hi[3] > size[1] + out_of_bounds:
            raise InputError(
                f"Kutular görüntünün ({size[0]}x{size[1]}) dışına taşıyor: {path}\n"
                "  -> image_size kutuların alındığı kareyle aynı olmalı (5.1 çıktısını değiştirme).")
    det["boxes"] = boxes
    det["image_size"] = size
    det["undistorted"] = bool(det.get("undistorted", False))
    return det


# [EK] 2.2'deki kalibrasyon yükleyicisinin kopyası (klasörler arası import yapılmaz).
class CameraCalibration:
    """Kamera kalibrasyonunu yükler ve kareleri düzeltir (undistort)."""

    REQUIRED_KEYS = ("camera_matrix", "dist_coeffs", "image_size")

    def __init__(self, path):
        path = Path(path)
        if not path.is_file():
            raise InputError(f"Kalibrasyon dosyası bulunamadı: {path}\n"
                             "  -> 5.1'de kullanılan kalibrasyon dosyasını --calibration ile ver.")
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

    def undistort(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.image_size:
            raise InputError(
                f"Görüntü boyutu {frame.shape[1]}x{frame.shape[0]}, ancak kalibrasyon "
                f"{self.image_size[0]}x{self.image_size[1]} için yapıldı ({self.path}).")
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)


def source_frames(source, names):
    """5.1'in kare adlarına karşılık gelen kaynak kareler (yalnız çizim için)."""
    source = Path(source)
    if not source.exists():
        raise InputError(f"Kaynak bulunamadı: {source}\n"
                         "  -> --source ile 5.1'e verilen görüntü / klasör / videoyu ver.")
    if source.is_dir():
        files = {p.stem: p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        for name in names:
            if name not in files:
                raise InputError(f"Kaynak klasörde '{name}' karesi yok: {source}")
            yield name, cv2.imread(str(files[name]), flags=cv2.IMREAD_COLOR)
        return
    if source.suffix.lower() not in VIDEO_EXTS:
        if list(names) != [source.stem]:
            raise InputError(f"Kaynak tek görüntü ({source.name}) ama tespit dosyasında "
                             f"{len(names)} kare var: {', '.join(names[:3])}...")
        img = cv2.imread(str(source), flags=cv2.IMREAD_COLOR)
        if img is None:
            raise InputError(f"Kaynak görüntü okunamadı: {source}")
        yield source.stem, img
        return
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise InputError(f"Kaynak video açılamadı: {source}")
    try:
        wanted = {int(n[4:]): n for n in names if n.startswith("kare") and n[4:].isdigit()}
        if len(wanted) != len(names):
            raise InputError("Kaynak video ama kare adları 'kareNNNNN' biçiminde değil.")
        index, last = 0, max(wanted)
        while index <= last:
            ok, frame = cap.read()
            if not ok:
                raise InputError(f"Kaynak videoda {index}. kare yok (tespit dosyası {last}. kareye kadar).")
            if index in wanted:
                yield wanted[index], frame
            index += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------- görsel / çıktı
def class_color(cid):
    rng = np.random.default_rng(int(cid) * 7919 + 17)
    return tuple(int(v) for v in rng.integers(60, 256, 3))


def draw_compare(frame, rows, label_left):
    width = frame.shape[1]
    fs = 0.6 * max(width, 640) / 1280.0 + 0.2
    th = max(1, round(2 * width / 1280.0))
    left, right = frame.copy(), frame.copy()
    for r in rows:
        color = class_color(r["class_id"])
        x1, y1, x2, y2 = (int(round(v)) for v in r["xyxy"])
        cv2.rectangle(right, (x1, y1), (x2, y2), color, thickness=th + 1)
        cx, cy = (int(round(v)) for v in r["xywh"][:2])
        cv2.drawMarker(right, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=th)
        bx, by = (int(round(v)) for v in r["bottom_center"])
        # Alt-orta nokta: kesikse içi boş (güvenilmez), değilse dolu daire.
        cv2.circle(right, (bx, by), 7, (0, 0, 255), thickness=-1 if not r["bottom_truncated"] else 2)
        text = f"{r['class_name']} ({bx},{by})"
        cv2.putText(right, text, (x1 + 2, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    thickness=th + 2, lineType=cv2.LINE_AA)
        cv2.putText(right, text, (x1 + 2, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, fs, color,
                    thickness=th, lineType=cv2.LINE_AA)
    for img, text in ((left, label_left), (right, "Kutu, merkez (+), alt-orta nokta (kirmizi)")):
        org = (10, round(30 * fs / 0.6))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thickness=th + 2,
                    lineType=cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255), thickness=th,
                    lineType=cv2.LINE_AA)
    return cv2.hconcat([left, right])


def write_csv(path, det, coords):
    """[EK] Okunabilir tablo: kare başına bir satır değil, tespit başına bir satır."""
    header = ["kare", "sinif", "guven", "x1", "y1", "x2", "y2", "merkez_x", "merkez_y", "genislik",
              "yukseklik", "alt_orta_x", "alt_orta_y", "alan", "kenara_degiyor", "alt_kenara_degiyor"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(det["boxes"])):
            w.writerow([str(det["frame"][det["det_frame"][i]]), str(det["class_name"][i]),
                        f"{float(det['confidence'][i]):.4f}",
                        *(f"{v:.2f}" for v in coords["xyxy"][i]), *(f"{v:.2f}" for v in coords["xywh"][i]),
                        *(f"{v:.2f}" for v in coords["bottom_center"][i]), f"{coords['area'][i]:.1f}",
                        int(coords["truncated"][i]), int(coords["bottom_truncated"][i])])


# --------------------------------------------------------------------------- ana akış
def run(args):
    det = load_detections(args.input, args.out_of_bounds)
    coords = box_coordinates(det["boxes"], det["image_size"], args.edge_margin)
    names = [str(n) for n in det["frame"]]
    calibration = None
    if args.compare:
        if det["undistorted"] and args.calibration is None:
            raise InputError("5.1 düzeltilmiş (undistort) karede çalışmış; kutular o karenin koordinatlarında.\n"
                             "  -> Çizim için aynı kalibrasyonu --calibration ile ver.")
        if args.calibration is not None:
            calibration = CameraCalibration(args.calibration)
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"Çıktı klasörü oluşturulamadı: {args.out_dir} ({exc})") from None

    stem = Path(args.input).stem
    stem = stem[:-len("_tespit")] if stem.endswith("_tespit") else stem
    width, height = det["image_size"]
    m = len(det["boxes"])
    print(f"{len(names)} kare, {m} kutu ({width}x{height}); kenara değen {int(coords['truncated'].sum())}, "
          f"alt kenara değen {int(coords['bottom_truncated'].sum())}")
    for i in range(min(m, 8)):
        x1, y1, x2, y2 = coords["xyxy"][i]
        cx, cy, bw, bh = coords["xywh"][i]
        bx, by = coords["bottom_center"][i]
        flag = " [alt kenar: temas noktası görünmüyor]" if coords["bottom_truncated"][i] else (
            " [kenara değiyor]" if coords["truncated"][i] else "")
        print(f"  {names[det['det_frame'][i]]}: {det['class_name'][i]} {float(det['confidence'][i]):.2f} | "
              f"xyxy ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) | xywh ({cx:.0f}, {cy:.0f}, {bw:.0f}, {bh:.0f}) | "
              f"alt-orta ({bx:.0f}, {by:.0f}){flag}")
    if m > 8:
        print(f"  ... ({m - 8} kutu daha)")

    out_path = args.out_dir / f"{stem}_kutular.npz"
    try:
        np.savez(out_path, frame=np.asarray(det["frame"], dtype=np.str_),
                 det_frame=np.asarray(det["det_frame"], dtype=np.int64),
                 class_id=np.asarray(det["class_id"], dtype=np.int64),
                 class_name=np.asarray(det["class_name"], dtype=np.str_),
                 confidence=np.asarray(det["confidence"], dtype=np.float32), **coords,
                 image_size=np.array(det["image_size"], dtype=np.int64),
                 undistorted=np.bool_(det["undistorted"]),
                 **({"class_names": np.asarray(det["class_names"], dtype=np.str_)}
                    if "class_names" in det else {}))
        print(f"Kaydedildi: {out_path}")
        if args.csv:
            csv_path = args.out_dir / f"{stem}_kutular.csv"
            write_csv(csv_path, det, coords)
            print(f"Kaydedildi: {csv_path}")
    except OSError as exc:
        raise InputError(f"Sonuç yazılamadı: {args.out_dir} ({exc})") from None

    if not args.compare:
        return
    source = Path(args.source)
    is_video = source.is_file() and source.suffix.lower() in VIDEO_EXTS
    label = "Girdi (duzeltilmis)" if calibration else "Girdi"
    writer = None
    try:
        for index, (name, frame) in enumerate(source_frames(source, names)):
            if frame is None:
                raise InputError(f"Kaynak kare okunamadı: {name}")
            if calibration is not None:
                frame = calibration.undistort(frame)
            if (frame.shape[1], frame.shape[0]) != (width, height):
                raise InputError(
                    f"Kaynak kare {frame.shape[1]}x{frame.shape[0]}, tespitler {width}x{height} için.\n"
                    "  -> 5.1'e verilen kaynağı ver.")
            frame_index = names.index(name)
            sel = np.flatnonzero(det["det_frame"] == frame_index)
            rows = [{"class_id": int(det["class_id"][i]), "class_name": str(det["class_name"][i]),
                     **{k: coords[k][i] for k in ("xyxy", "xywh", "bottom_center", "bottom_truncated")}}
                    for i in sel]
            img = draw_compare(frame, rows, label)
            if is_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.out_dir / f"{stem}_kutular_karsilastirma.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                                             (img.shape[1], img.shape[0]), isColor=True)
                writer.write(img)
            else:
                out_img = args.out_dir / (f"{stem}_kutular_karsilastirma.png" if len(names) == 1
                                          else f"{name}_kutular_karsilastirma.png")
                if not cv2.imwrite(str(out_img), img):
                    raise InputError(f"Görsel yazılamadı: {out_img}")
    finally:
        if writer is not None:
            writer.release()
    print(f"Karşılaştırma görseli kaydedildi: {args.out_dir}")


class TurkishArgumentParser(argparse.ArgumentParser):
    """Argüman hatasında "HATA: ..." yazar ve 1 koduyla çıkar."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"HATA: Geçersiz komut satırı argümanı: {message}\n"
              "  -> Seçenekler için --help kullan.", file=sys.stderr)
        sys.exit(1)


def parse_args(argv=None):
    p = TurkishArgumentParser(
        description="5.1 tespitlerinden kutu koordinatları: xyxy, xywh, oranlı biçimler ve alt-orta nokta.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"5.1'in tespit dosyası (varsayılan: {INPUT_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--csv", action="store_true", help="Tabloyu ayrıca CSV olarak kaydet")
    p.add_argument("--compare", action="store_true",
                   help="Kaynak karede kutu, merkez ve alt-orta noktayı çiz (--source gerekir)")
    p.add_argument("--source", type=Path, default=SOURCE_PATH,
                   help=f"5.1'e verilen kaynak, yalnız çizim için (varsayılan: {SOURCE_PATH})")
    p.add_argument("--calibration", type=Path, default=CALIBRATION_PATH,
                   help="5.1 kalibrasyonla çalıştıysa aynı kalibrasyon dosyası (çizim için)")
    p.add_argument("--edge-margin", type=float, default=EDGE_MARGIN_PX,
                   help=f"Kenara değme payı, px (varsayılan: {EDGE_MARGIN_PX})")
    p.add_argument("--out-of-bounds", type=float, default=OUT_OF_BOUNDS_PX,
                   help=f"Görüntü dışına taşma payı, px (varsayılan: {OUT_OF_BOUNDS_PX})")
    return p.parse_args(argv)


def validate_args(args):
    for name, value in (("--edge-margin", args.edge_margin), ("--out-of-bounds", args.out_of_bounds)):
        if not (math.isfinite(value) and value >= 0):
            raise InputError(f"{name} negatif olmayan bir sayı olmalı, verilen: {value}")


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
