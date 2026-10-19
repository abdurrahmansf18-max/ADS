"""
4.1 Gaussian Blur ve Bilateral Filter — kareyi yumuşatıp sahte kenarları azaltır.

Ne işe yarar:
    Asfalt dokusu ve kamera gürültüsü kenar algılamada sahte kenar üretir. Yumuşatma bu gürültüyü
    bastırır, şerit çizgisinin kenarı kalır.

Nasıl çalışır:
    Kare düzeltilir; Gaussian (komşu piksellerin ağırlıklı ortalaması) ya da Bilateral (yalnız benzer
    renkli komşuların ortalaması, kenarı daha iyi korur) uygulanır. Etkiyi göstermek için filtreden önce
    ve sonra kenar oranı ölçülür.

Bağlantılar:
    Alır : 1.1'in kalibrasyon.npz dosyası; kareyi kendisi düzeltir.
    Verir: yumuşatılmış kare -> 4.3 gradyan maskesi; gürültü kenar sanılmasın diye gradyandan önce.
    Ana hat: main.py her karede smooth() ile yalnız gradyan dalını yumuşatır; renk dalında yumuşatma
             soluk çizgiyi silebiliyordu.

Parametreler:
    --input       Gerçek veri: görüntü, klasör ya da video (varsayılan data/ortam_verisi/test_kareleri/kolay_video).
                  [DEĞİŞTİR]
    --calibration Başka gereksinimden: 1.1'in çıktısı.
    --filter, --gaussian-ksize, --bilateral-*   Ayar: filtre türü ve boyutu (varsayılanlar uygun).
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / Path(__file__).resolve().parent.name

# [DEĞİŞTİR] Görüntü, görüntü klasörü ya da video.
INPUT_PATH = DATA_DIR / "ortam_verisi" / "test_kareleri" / "kolay_video"

# [DEĞİŞTİR] Kamera kalibrasyonu (.npz). None ise undistort atlanır.
CALIBRATION_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "kalibrasyon.npz"

FILTER = "gaussian"           # "gaussian" ya da "bilateral"

# [DEĞİŞTİR] Gaussian: çekirdek boyutu (tek sayı, px) ve sigma (0: boyuttan hesaplanır).
GAUSSIAN_KSIZE = 5
GAUSSIAN_SIGMA = 0.0

# [DEĞİŞTİR] Bilateral: komşuluk çapı (px), renk ve uzay sigmaları.
# Renkli görüntüde renk farkı 3 kanalın mutlak farkları toplamıdır (gride göre ~3 kat).
# 45: gürültüyü bastırır, ~25 gri seviyelik soluk çizgi kenarını korur; 75 onu siler.
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 45.0
BILATERAL_SIGMA_SPACE = 75.0

# Etki ölçümü ve karşılaştırma görselinin alt satırı (yalnızca değerlendirme için).
EDGE_METHOD = "sobel"         # alt satırda gösterilen kenar: "sobel" (x yönü) ya da "canny"
SOBEL_THRESHOLD = 12          # sabit ölçekli |gx| eşiği (2.2 ile aynı)
CANNY_THRESHOLDS = (30, 60)   # 2.2 ile aynı

# Video çıktısı: "mp4v" (.mp4, kayıplı, küçük) ya da "FFV1" (.mkv, kayıpsız, büyük).
VIDEO_FOURCC = "mp4v"
# Video: en çok kaç kare işlensin (None = tümü).
MAX_FRAMES = None
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
SOBEL_X_MAX = 4.0 * 255.0     # ksize=3 için |gx|'in alabileceği en büyük değer


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


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


# --------------------------------------------------------------------------- yumuşatma
def smooth(frame, args):
    """Gereksinimin özü: Gaussian ya da Bilateral yumuşatma."""
    if args.filter == "gaussian":
        return cv2.GaussianBlur(frame, (args.gaussian_ksize, args.gaussian_ksize),
                                args.gaussian_sigma)
    return cv2.bilateralFilter(frame, args.bilateral_d, args.bilateral_sigma_color,
                               args.bilateral_sigma_space)


def filter_label(args):
    if args.filter == "gaussian":
        sigma = f", sigma {args.gaussian_sigma:g}" if args.gaussian_sigma > 0 else ""
        return f"Gaussian {args.gaussian_ksize}x{args.gaussian_ksize}{sigma}"
    return (f"Bilateral d={args.bilateral_d}, sc={args.bilateral_sigma_color:g}, "
            f"ss={args.bilateral_sigma_space:g}")


# --------------------------------------------------------------------------- etki ölçümü
def sobel_x_mask(gray, threshold):
    """x yönünde Sobel -> mutlak değer -> sabit ölçek (0-255) -> eşik."""
    abs_gx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    # [EK] Sabit ölçek: blur öncesi ve sonrası aynı ölçekte karşılaştırılır.
    scaled = np.clip(255.0 * abs_gx / SOBEL_X_MAX, 0, 255).astype(np.uint8)
    return cv2.inRange(scaled, threshold, 255)


def canny_mask(gray, thresholds):
    return cv2.Canny(gray, thresholds[0], thresholds[1], L2gradient=True)


# [EK] Kenar oranı (blur öncesi / sonrası): yumuşatmanın sahte kenarları ne kadar azalttığı.
def edge_stats(before_bgr, after_bgr, args):
    """Dönüş: ({"sobel": (önce, sonra), "canny": (önce, sonra)} oranlar, gösterilecek maskeler)."""
    gray_before = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)
    masks = {"sobel": (sobel_x_mask(gray_before, args.sobel_threshold),
                       sobel_x_mask(gray_after, args.sobel_threshold)),
             "canny": (canny_mask(gray_before, args.canny_thresholds),
                       canny_mask(gray_after, args.canny_thresholds))}
    ratios = {k: tuple(cv2.countNonZero(m) / m.size for m in pair) for k, pair in masks.items()}
    return ratios, masks[args.edge_method]


def format_ratios(ratios):
    return (f"Sobel-x kenar %{100 * ratios['sobel'][0]:.2f} -> %{100 * ratios['sobel'][1]:.2f}, "
            f"Canny kenar %{100 * ratios['canny'][0]:.2f} -> %{100 * ratios['canny'][1]:.2f}")


# --------------------------------------------------------------------------- görsel
def put_label(img, text, fs, th):
    org = (10, round(35 * fs / 0.8))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                thickness=th + 2, lineType=cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255),
                thickness=th, lineType=cv2.LINE_AA)


# [EK] 2x2: girdi | yumuşatılmış, kenar blur'suz | kenar blur'lu.
def draw_compare(before, after, edge_masks, ratios, args, undistorted):
    width = before.shape[1]
    fs = 0.8 * width / 1280.0
    th = max(1, round(2 * width / 1280.0))
    name = "Sobel-x" if args.edge_method == "sobel" else "Canny"
    panels = [before.copy(), after.copy(),
              cv2.cvtColor(edge_masks[0], cv2.COLOR_GRAY2BGR),
              cv2.cvtColor(edge_masks[1], cv2.COLOR_GRAY2BGR)]
    labels = ["Girdi (duzeltilmis)" if undistorted else "Girdi", filter_label(args),
              f"{name} kenar, blur yok: %{100 * ratios[args.edge_method][0]:.2f}",
              f"{name} kenar, {args.filter}: %{100 * ratios[args.edge_method][1]:.2f}"]
    for img, text in zip(panels, labels):
        put_label(img, text, fs, th)
    return cv2.vconcat([cv2.hconcat(panels[:2]), cv2.hconcat(panels[2:])])


# --------------------------------------------------------------------------- girdi / çıktı
def read_image(path):
    img = cv2.imread(str(path), flags=cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(
            f"Görüntü okunamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(IMAGE_EXTS))}) olduğunu kontrol et.")
    return img


def write_image(path, img):
    if not cv2.imwrite(str(path), img):
        raise InputError(f"Görüntü yazılamadı: {path}\n  -> --out-dir'i ve yazma iznini kontrol et.")


def open_writer(path, fourcc, fps, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size, isColor=True)
    if not writer.isOpened():
        raise InputError(f"Video yazılamadı: {path} ({fourcc})\n"
                         "  -> --out-dir'i ve yazma iznini kontrol et.")
    return writer


def process_frame(frame, args, calibration):
    before = calibration.undistort(frame) if calibration is not None else frame
    after = smooth(before, args)
    ratios, masks = edge_stats(before, after, args)
    return before, after, ratios, masks


def process_image(path, out_dir, args, calibration):
    before, after, ratios, masks = process_frame(read_image(path), args, calibration)
    stem = f"{path.stem}_{args.filter}"
    out_path = out_dir / f"{stem}.png"
    write_image(out_path, after)
    if args.compare:
        write_image(out_dir / f"{stem}_karsilastirma.png",
                    draw_compare(before, after, masks, ratios, args, calibration is not None))
    print(f"{path.name}: {format_ratios(ratios)} -> {out_path}")


def process_folder(path, out_dir, args, calibration):
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise InputError(f"Klasörde görüntü yok: {path}\n"
                         f"  -> Desteklenen uzantılar: {', '.join(sorted(IMAGE_EXTS))}")
    sub_dir = out_dir / path.name
    sub_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        process_image(file, sub_dir, args, calibration)
    print(f"{len(files)} görüntü işlendi -> {sub_dir}")


def process_video(path, out_dir, args, calibration):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputError(
            f"Video açılamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(VIDEO_EXTS))}) olduğunu kontrol et.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (math.isfinite(fps) and fps > 0):
        fps = 25.0
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise InputError(f"Videodan hiç kare okunamadı: {path}")
    height, width = frame.shape[:2]
    if calibration is not None:
        calibration.check_size(width, height)
    ext = ".mkv" if args.video_fourcc == "FFV1" else ".mp4"
    out_path = out_dir / f"{path.stem}_{args.filter}{ext}"
    writers = []
    try:
        writer = open_writer(out_path, args.video_fourcc, fps, (width, height))
        writers.append(writer)
        compare_writer = None
        if args.compare:
            compare_writer = open_writer(out_dir / f"{path.stem}_{args.filter}_karsilastirma.mp4",
                                         "mp4v", fps, (2 * width, 2 * height))
            writers.append(compare_writer)
        count, totals = 0, np.zeros(4)
        while ok and (args.max_frames is None or count < args.max_frames):
            if frame.shape[:2] != (height, width):
                raise InputError(
                    f"Videonun {count}. karesinin boyutu değişti "
                    f"({frame.shape[1]}x{frame.shape[0]}, ilk kare {width}x{height}).")
            before, after, ratios, masks = process_frame(frame, args, calibration)
            writer.write(after)
            if compare_writer is not None:
                compare_writer.write(draw_compare(before, after, masks, ratios, args,
                                                  calibration is not None))
            totals += (*ratios["sobel"], *ratios["canny"])
            count += 1
            ok, frame = cap.read()
    finally:
        cap.release()
        for w in writers:
            w.release()
    mean = totals / count
    print(f"{path.name}: {count} kare, ortalama "
          + format_ratios({"sobel": mean[:2], "canny": mean[2:]}) + f" -> {out_path}")


# --------------------------------------------------------------------------- argümanlar
class TurkishArgumentParser(argparse.ArgumentParser):
    """Argüman hatasında "HATA: ..." yazar ve 1 koduyla çıkar."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"HATA: Geçersiz komut satırı argümanı: {message}\n"
              "  -> Seçenekler için --help kullan.", file=sys.stderr)
        sys.exit(1)


def parse_args(argv=None):
    p = TurkishArgumentParser(
        description="Görüntüyü Gaussian ya da Bilateral filtreyle yumuşatır; sahte kenarların "
                    "azalmasını Sobel-x ve Canny ile ölçer.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Görüntü, görüntü klasörü ya da video (varsayılan: {INPUT_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--calibration", type=Path, default=CALIBRATION_PATH,
                   help="Kamera kalibrasyonu .npz; verilirse önce undistort (varsayılan: yok)")
    p.add_argument("--filter", choices=("gaussian", "bilateral"), default=FILTER,
                   help=f"Yumuşatma filtresi (varsayılan: {FILTER})")
    p.add_argument("--gaussian-ksize", type=int, default=GAUSSIAN_KSIZE,
                   help=f"Gaussian çekirdek boyutu, tek sayı (varsayılan: {GAUSSIAN_KSIZE})")
    p.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA,
                   help=f"Gaussian sigma; 0 ise boyuttan hesaplanır (varsayılan: {GAUSSIAN_SIGMA})")
    p.add_argument("--bilateral-d", type=int, default=BILATERAL_D,
                   help=f"Bilateral komşuluk çapı (varsayılan: {BILATERAL_D})")
    p.add_argument("--bilateral-sigma-color", type=float, default=BILATERAL_SIGMA_COLOR,
                   help=f"Bilateral renk sigması (varsayılan: {BILATERAL_SIGMA_COLOR})")
    p.add_argument("--bilateral-sigma-space", type=float, default=BILATERAL_SIGMA_SPACE,
                   help=f"Bilateral uzay sigması (varsayılan: {BILATERAL_SIGMA_SPACE})")
    p.add_argument("--edge-method", choices=("sobel", "canny"), default=EDGE_METHOD,
                   help=f"Karşılaştırmada gösterilen kenar yöntemi (varsayılan: {EDGE_METHOD})")
    p.add_argument("--sobel-threshold", type=int, default=SOBEL_THRESHOLD,
                   help=f"Sabit ölçekli |gx| eşiği, 0-255 (varsayılan: {SOBEL_THRESHOLD})")
    p.add_argument("--canny-thresholds", type=float, nargs=2, default=CANNY_THRESHOLDS,
                   metavar=("ALT", "UST"), help=f"Canny çift eşiği (varsayılan: {CANNY_THRESHOLDS})")
    p.add_argument("--video-fourcc", choices=("mp4v", "FFV1"), default=VIDEO_FOURCC,
                   help=f"Video çıktısı kodeği (varsayılan: {VIDEO_FOURCC})")
    p.add_argument("--compare", action="store_true",
                   help="Girdi, yumuşatılmış ve kenar maskelerini 2x2 kaydet")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                   help="Videoda en çok kaç kare işlensin (varsayılan: tümü)")
    return p.parse_args(argv)


def validate_args(args):
    if args.gaussian_ksize <= 0 or args.gaussian_ksize % 2 == 0:
        raise InputError(f"--gaussian-ksize pozitif tek sayı olmalı, verilen: {args.gaussian_ksize}")
    if not (math.isfinite(args.gaussian_sigma) and args.gaussian_sigma >= 0):
        raise InputError(f"--gaussian-sigma negatif olmayan bir sayı olmalı, verilen: {args.gaussian_sigma}")
    if args.bilateral_d <= 0:
        raise InputError(f"--bilateral-d pozitif olmalı, verilen: {args.bilateral_d}")
    for name, value in (("--bilateral-sigma-color", args.bilateral_sigma_color),
                        ("--bilateral-sigma-space", args.bilateral_sigma_space)):
        if not (math.isfinite(value) and value > 0):
            raise InputError(f"{name} pozitif bir sayı olmalı, verilen: {value}")
    if not 0 <= args.sobel_threshold <= 255:
        raise InputError(f"--sobel-threshold 0-255 aralığında olmalı, verilen: {args.sobel_threshold}")
    low, high = args.canny_thresholds
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
        raise InputError(f"--canny-thresholds 0 <= ALT <= UST olmalı, verilen: {low} {high}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise InputError(f"--max-frames pozitif olmalı, verilen: {args.max_frames}")


# --------------------------------------------------------------------------- ana akış
def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    calibration = CameraCalibration(args.calibration) if args.calibration is not None else None
    path = Path(args.input)
    if not path.exists():
        raise InputError(
            f"Girdi bulunamadı: {path}\n"
            "  -> --input ile doğru yolu ver ya da AYARLAR'daki INPUT_PATH'i düzelt.")
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"Çıktı klasörü oluşturulamadı: {args.out_dir} ({exc})\n"
                         "  -> --out-dir'i kontrol et.") from None

    print(f"Filtre: {filter_label(args)}; undistort: "
          f"{'var (' + str(args.calibration) + ')' if calibration else 'yok'}; "
          f"kenar oranı: blur'suz -> blur'lu")
    if path.is_dir():
        process_folder(path, args.out_dir, args, calibration)
    elif path.suffix.lower() in VIDEO_EXTS:
        process_video(path, args.out_dir, args, calibration)
    else:
        process_image(path, args.out_dir, args, calibration)


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
