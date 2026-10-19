"""
5.3 Confidence Score ve sınıf bilgisi — güvene ve sınıfa göre tespit filtresi.

Ne işe yarar:
    Modelin emin olmadığı tespitleri ve sürüşle ilgisi olmayan sınıfları eler; gereksiz alarmı önler.

Nasıl çalışır:
    Güveni eşiğin (0.45) altında kalan tespitler atılır; yalnız ilgili sınıflar (yaya, araba, motosiklet,
    otobüs, kamyon) kalır. Kalan tespitlere risk grubu yazılır.

Bağlantılar:
    Alır : 5.2'nin <ad>_kutular.npz dosyası (ya da doğrudan 5.1'in tespitleri).
    Verir: <ad>_filtreli.npz -> 5.5.
    Ana hat: main.py her karede filter_mask() uygular; eşiği ve sınıfları bu modülden okur.

Parametreler:
    --input       Başka gereksinimden: 5.2'nin çıktısı (varsayılan kolay videonun kutuları).
    --conf, --classes, --class-conf   Ayar: güven eşiği ve sınıflar (doküman değerleri). [DEĞİŞTİR]
    --source, --calibration   Yalnız çizim için: kaynak video (gerçek veri) ve 1.1'in kalibrasyonu.
"""

import argparse
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

# [DEĞİŞTİR] Ham tespitler: 5.2'nin kutu dosyası (önerilen) ya da 5.1'in tespit dosyası.
INPUT_PATH = OUTPUTS / "5.2 Bounding Box koordinatları" / "kolay_video_iyi_isik_kutular.npz"

# [DEĞİŞTİR] Güven eşiği (İnşa: ~0.4-0.5).
CONF_THRESHOLD = 0.45

# [DEĞİŞTİR] Tutulacak COCO sınıfları: person, car, motorcycle, bus, truck (İnşa).
CLASSES = (0, 2, 3, 5, 7)

# Sınıfa özel eşik (ör. {"person": 0.35}); boş ise herkes CONF_THRESHOLD.
CLASS_CONF = {}

# [EK] Risk grupları (sınıf adına göre); listede olmayan "diger".
RISK_GROUPS = {"person": "savunmasiz", "bicycle": "savunmasiz", "motorcycle": "savunmasiz",
               "car": "arac", "bus": "buyuk_arac", "truck": "buyuk_arac"}

# [DEĞİŞTİR] Yalnız --compare çizimi için: 5.1'e verilen kaynak ve (varsa) kalibrasyon.
SOURCE_PATH = DATA_DIR / "ortam_verisi" / "videolar" / "kolay_video" / "kolay_video_iyi_isik.mp4"
CALIBRATION_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "kalibrasyon.npz"
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
REQUIRED_KEYS = ("frame", "det_frame", "class_id", "class_name", "confidence", "image_size")
# Tespit başına olan alanlar (5.1 ve 5.2 biçimleri): hepsi aynı maskeyle süzülür.
DETECTION_KEYS = ("det_frame", "boxes", "xyxy", "xywh", "xyxyn", "xywhn", "bottom_center", "area",
                  "truncated", "bottom_truncated", "class_id", "class_name", "confidence")


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


# --------------------------------------------------------------------------- filtre
def filter_mask(conf, cls, conf_threshold, keep_classes, class_conf):
    """Gereksinimin özü: güven eşiği ve ilgili sınıflar. Dönüş: (tut, düşük güven, ilgisiz sınıf)."""
    class_ok = np.isin(cls, list(keep_classes))
    # [EK] Sınıfa özel eşik: yoksa genel eşik.
    thresholds = np.array([class_conf.get(int(c), conf_threshold) for c in cls], dtype=np.float64)
    conf_ok = conf >= thresholds
    return class_ok & conf_ok, class_ok & ~conf_ok, ~class_ok


def risk_group(names):
    return np.array([RISK_GROUPS.get(str(n), "diger") for n in names], dtype=np.str_)


# --------------------------------------------------------------------------- girdi
def load_detections(path):
    path = Path(path)
    if not path.is_file():
        raise InputError(
            f"Tespit dosyası bulunamadı: {path}\n"
            "  -> Önce 5.1 (ve 5.2) çalıştır ya da --input ile <ad>_kutular.npz / <ad>_tespit.npz ver.")
    try:
        with np.load(path) as data:
            missing = [k for k in REQUIRED_KEYS if k not in data.files]
            if missing:
                raise ValueError(f"eksik alanlar: {', '.join(missing)}")
            det = {k: data[k] for k in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise InputError(f"Tespit dosyası okunamadı: {path} ({exc})\n"
                         "  -> 5.1 ya da 5.2'nin ürettiği dosyayı ver.") from None
    m = len(det["class_id"])
    bad = {k: len(det[k]) for k in DETECTION_KEYS if k in det and len(det[k]) != m}
    if bad:
        raise InputError(f"Tespit dizilerinin uzunlukları tutarsız (tespit {m}, {bad}): {path}")
    conf = np.asarray(det["confidence"], dtype=np.float64)
    if m and (not np.isfinite(conf).all() or conf.min() < 0 or conf.max() > 1):
        raise InputError(f"confidence 0-1 aralığında olmalı (en az {conf.min():.3f}, en çok {conf.max():.3f}): {path}")
    return det


def parse_classes(tokens, det):
    """Ad ya da numara -> numara kümesi. 80 sınıflık ad listesi varsa onunla doğrular."""
    names = [str(n) for n in det["class_names"]] if "class_names" in det else None
    lookup = {n: i for i, n in enumerate(names)} if names else {}
    if not names:  # ad listesi yoksa girdideki tespitlerden öğren
        lookup = {str(n): int(i) for n, i in zip(det["class_name"], det["class_id"])}
    ids, unknown = [], []
    for t in tokens:
        t = str(t)
        if t.isdigit():
            if names and int(t) >= len(names):
                unknown.append(t)
            ids.append(int(t))
        elif t in lookup:
            ids.append(lookup[t])
        else:
            unknown.append(t)
    if unknown:
        valid = ", ".join(names) if names else ", ".join(sorted(lookup)) + " (ad listesi yok; girdideki adlar)"
        raise InputError(f"Bilinmeyen sınıf: {', '.join(unknown)}\n  -> Geçerli adlar: {valid}")
    return sorted(set(ids))


def parse_class_conf(items, det):
    """[EK] "ad=eşik" listesi -> {numara: eşik}."""
    result = {}
    for item in items:
        if "=" not in item:
            raise InputError(f"--class-conf biçimi 'sinif=esik' olmalı (ör. person=0.35), verilen: {item}")
        name, value = item.split("=", 1)
        try:
            v = float(value)
        except ValueError:
            raise InputError(f"--class-conf eşiği sayı olmalı, verilen: {item}") from None
        if not (math.isfinite(v) and 0 < v <= 1):
            raise InputError(f"--class-conf eşiği 0 < değer <= 1 olmalı, verilen: {item}")
        result[parse_classes([name], det)[0]] = v
    return result


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
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise InputError(f"Kalibrasyon dosyası okunamadı: {path} ({exc})") from None
        self.path = path

    def undistort(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.image_size:
            raise InputError(
                f"Görüntü boyutu {frame.shape[1]}x{frame.shape[0]}, ancak kalibrasyon "
                f"{self.image_size[0]}x{self.image_size[1]} için yapıldı ({self.path}).")
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)


def source_frames(source, names):
    """5.1'in kare adlarına karşılık gelen kaynak kareler (yalnız çizim için; 5.2 ile aynı)."""
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
            raise InputError(f"Kaynak tek görüntü ({source.name}) ama tespit dosyasındaki kareler: "
                             f"{', '.join(names[:3])}")
        yield source.stem, cv2.imread(str(source), flags=cv2.IMREAD_COLOR)
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
                raise InputError(f"Kaynak videoda {index}. kare yok.")
            if index in wanted:
                yield wanted[index], frame
            index += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------- görsel
def draw_panel(frame, boxes, names, conf, keep, title, show_removed):
    width = frame.shape[1]
    fs = 0.6 * max(width, 640) / 1280.0 + 0.2
    th = max(1, round(2 * width / 1280.0))
    img = frame.copy()
    for (x1, y1, x2, y2), n, c, k in zip(boxes, names, conf, keep):
        if not k and not show_removed:
            continue
        color = (0, 200, 0) if k else (0, 0, 255)
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, color, thickness=th + (1 if k else 0))
        text = f"{n} {c:.2f}"
        org = (p1[0] + 2, max(15, p1[1] - 6))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thickness=th + 2, lineType=cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, thickness=th, lineType=cv2.LINE_AA)
    org = (10, round(30 * fs / 0.6))
    cv2.putText(img, title, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thickness=th + 2, lineType=cv2.LINE_AA)
    cv2.putText(img, title, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255), thickness=th, lineType=cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- ana akış
def run(args):
    det = load_detections(args.input)
    keep_classes = parse_classes(args.classes, det)
    class_conf = parse_class_conf(args.class_conf, det)
    outside = [c for c in class_conf if c not in keep_classes]
    if outside:
        print(f"UYARI: --class-conf'taki sınıf(lar) {outside} tutulan sınıflarda değil; etkisiz.", file=sys.stderr)

    conf = np.asarray(det["confidence"], dtype=np.float64)
    cls = np.asarray(det["class_id"], dtype=np.int64)
    keep, low_conf, other_class = filter_mask(conf, cls, args.conf, keep_classes, class_conf)
    idx = np.flatnonzero(keep)

    out = {k: v for k, v in det.items() if k not in DETECTION_KEYS}
    out.update({k: det[k][idx] for k in DETECTION_KEYS if k in det})
    out["kept_index"] = idx.astype(np.int64)
    out["risk_group"] = risk_group(det["class_name"][idx])
    out["conf_threshold"] = np.float64(args.conf)
    out["kept_classes"] = np.array(keep_classes, dtype=np.int64)
    out["class_conf_ids"] = np.array(list(class_conf), dtype=np.int64)
    out["class_conf_values"] = np.array(list(class_conf.values()), dtype=np.float64)

    names_all = det["class_name"]
    class_names = {int(c): str(n) for c, n in zip(cls, names_all)}
    print(f"Güven >= {args.conf:g}" + (f" (sınıfa özel: {', '.join(f'{class_names.get(c, c)}={v:g}' for c, v in class_conf.items())})"
                                        if class_conf else "")
          + f"; sınıflar: {', '.join(str(class_names.get(c, c)) for c in keep_classes)}")
    m = len(cls)
    print(f"{len(det['frame'])} kare, {m} ham tespit -> {len(idx)} kaldı; elenen: düşük güven "
          f"{int(low_conf.sum())}, ilgisiz sınıf {int(other_class.sum())}")
    # [EK] Sınıf başına sayım ve elenme nedenleri.
    for c in sorted(set(cls.tolist())):
        sel = cls == c
        kept_c = int((keep & sel).sum())
        reason = "ilgisiz sınıf" if not np.isin(c, keep_classes) else f"{int((low_conf & sel).sum())} düşük güven"
        print(f"  {class_names[c]:14s} ham {int(sel.sum()):4d} -> kalan {kept_c:4d} "
              f"(güven {conf[sel].min():.2f}-{conf[sel].max():.2f}; elenen: {reason if kept_c < sel.sum() else '-'})")

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"Çıktı klasörü oluşturulamadı: {args.out_dir} ({exc})") from None
    stem = Path(args.input).stem
    for suffix in ("_kutular", "_tespit"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    out_path = args.out_dir / f"{stem}_filtreli.npz"
    try:
        np.savez(out_path, **out)
    except OSError as exc:
        raise InputError(f"Sonuç yazılamadı: {out_path} ({exc})") from None
    print(f"Kaydedildi: {out_path}")

    if args.compare:
        compare(args, det, keep, stem)


def compare(args, det, keep, stem):
    if bool(det.get("undistorted", False)) and args.calibration is None:
        raise InputError("Tespitler düzeltilmiş (undistort) karede; çizim için 5.1'deki kalibrasyonu "
                         "--calibration ile ver.")
    calibration = CameraCalibration(args.calibration) if args.calibration is not None else None
    boxes = det["xyxy"] if "xyxy" in det else det["boxes"]
    names = [str(n) for n in det["frame"]]
    width, height = (int(v) for v in np.asarray(det["image_size"]).ravel())
    source = Path(args.source)
    is_video = source.is_file() and source.suffix.lower() in VIDEO_EXTS
    writer = None
    try:
        for name, frame in source_frames(source, names):
            if frame is None:
                raise InputError(f"Kaynak kare okunamadı: {name}")
            if calibration is not None:
                frame = calibration.undistort(frame)
            if (frame.shape[1], frame.shape[0]) != (width, height):
                raise InputError(f"Kaynak kare {frame.shape[1]}x{frame.shape[0]}, tespitler {width}x{height} için.")
            sel = np.flatnonzero(det["det_frame"] == names.index(name))
            args_panel = (boxes[sel], det["class_name"][sel], det["confidence"][sel], keep[sel])
            left = draw_panel(frame, *args_panel, f"Ham: {len(sel)} tespit", show_removed=True)
            right = draw_panel(frame, *args_panel, f"Filtreli: {int(keep[sel].sum())} (guven >= {args.conf:g})",
                               show_removed=False)
            img = cv2.hconcat([left, right])
            if is_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.out_dir / f"{stem}_filtreli_karsilastirma.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (img.shape[1], img.shape[0]),
                                             isColor=True)
                writer.write(img)
            else:
                out_img = args.out_dir / (f"{stem}_filtreli_karsilastirma.png" if len(names) == 1
                                          else f"{name}_filtreli_karsilastirma.png")
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
        description="Ham tespitleri güven eşiği ve ilgili COCO sınıflarıyla filtreler.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"5.2 kutu dosyası ya da 5.1 tespit dosyası (varsayılan: {INPUT_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD,
                   help=f"Güven eşiği, 0-1 (varsayılan: {CONF_THRESHOLD})")
    p.add_argument("--classes", nargs="+", default=[str(c) for c in CLASSES],
                   help=f"Tutulacak sınıflar, ad ya da COCO numarası (varsayılan: {' '.join(map(str, CLASSES))})")
    p.add_argument("--class-conf", nargs="+", default=[f"{k}={v}" for k, v in CLASS_CONF.items()],
                   metavar="SINIF=ESIK", help="Sınıfa özel güven eşiği, ör. person=0.35 (varsayılan: yok)")
    p.add_argument("--compare", action="store_true",
                   help="Ham ve filtrelenmiş tespitleri kaynak karede yan yana çiz (--source gerekir)")
    p.add_argument("--source", type=Path, default=SOURCE_PATH,
                   help=f"5.1'e verilen kaynak, yalnız çizim için (varsayılan: {SOURCE_PATH})")
    p.add_argument("--calibration", type=Path, default=CALIBRATION_PATH,
                   help="5.1 kalibrasyonla çalıştıysa aynı kalibrasyon (çizim için)")
    return p.parse_args(argv)


def validate_args(args):
    if not (math.isfinite(args.conf) and 0 < args.conf <= 1):
        raise InputError(f"--conf 0 < değer <= 1 olmalı, verilen: {args.conf}")


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
