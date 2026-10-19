"""
2.1 HLS/HSV/LAB ve eşikleme — renkten şerit maskesi (ilk renk maskesi; ana hatta yerini 4.2 aldı).

Ne işe yarar:
    Beyaz ve sarı şerit çizgilerini renklerinden ayırıp siyah-beyaz bir maske çıkarır. RGB ışıkla çok
    değiştiği için renk, parlaklıktan ayrı tutan HLS ve LAB uzaylarında aranır.

Nasıl çalışır:
    Kare HLS ve LAB'a çevrilir. Beyaz çizgi HLS-L, sarı çizgi LAB-B kanalında eşiklenir (cv2.inRange);
    iki maske birleştirilir (OR). Sonuç: şerit 255, arka plan 0.

Bağlantılar:
    Alır : 1.1'in düzeltilmiş kareleri; eşikler bozulmasız karede ayarlandı.
    Verir: <ad>_maske.png -> 1.2 ile kuş bakışına çevrilip 2.3'e; ayrıca 4.4'e renk maskesi olarak.
    Ana hat: kullanılmaz; ana hatta renk maskesini 4.2 çıkarır (bu modülün geliştirilmiş hâli).

Parametreler:
    --input       Başka gereksinimden: 1.1'in düzeltilmiş kareleri (ya da düzeltilmiş video).
    --white-l-range, --yellow-b-range
                  Gerçek veriye göre ayarlanan eşikler (200-255 ve 155-255). [DEĞİŞTİR]
    --white-channel, --compare   Ayar.
"""

import argparse
import math
import sys
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

# [DEĞİŞTİR] Renkli kare: tek görüntü, görüntü klasörü ya da video.
INPUT_PATH = PROJECT_ROOT / "outputs" / "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme" / "duzeltilmis_kareler"

# [DEĞİŞTİR] Eşikler (OpenCV 8 bit ölçeği, 0-255). esik_ayarlama.ipynb'de kendi
# verinde deneyip buraya yaz.
WHITE_CHANNEL = "L"           # beyaz çizgi kanalı: "L" (HLS açıklık) ya da "S" (HLS doygunluk)
WHITE_L_RANGE = (200, 255)    # HLS-L aralığı (WHITE_CHANNEL = "L" ise)
WHITE_S_RANGE = (170, 255)    # HLS-S aralığı (WHITE_CHANNEL = "S" ise)
YELLOW_B_RANGE = (155, 255)   # LAB-B aralığı: 128 nötr, büyüdükçe sarı

# Video: en çok kaç kare işlensin (None = tümü).
MAX_FRAMES = None
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


# --------------------------------------------------------------------------- eşikleme
def color_masks(frame_bgr, white_channel, white_range, yellow_b_range):
    """Renk eşikleme. Dönüş: (beyaz maske, sarı maske, birleşik maske), hepsi 0/255 uint8."""
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    _, light, sat = cv2.split(hls)
    _, _, lab_b = cv2.split(lab)

    white_src = light if white_channel == "L" else sat
    white = cv2.inRange(white_src, white_range[0], white_range[1])
    yellow = cv2.inRange(lab_b, yellow_b_range[0], yellow_b_range[1])
    return white, yellow, cv2.bitwise_or(white, yellow)


# [EK] Maske oranı: eşiğin gevşek ya da sıkı olduğunu sayıyla gösterir.
def mask_ratio(mask):
    return cv2.countNonZero(mask) / mask.size


def format_ratios(white, yellow, mask):
    return (f"maske %{100 * mask_ratio(mask):.2f} (beyaz %{100 * mask_ratio(white):.2f}, "
            f"sarı %{100 * mask_ratio(yellow):.2f})")


# --------------------------------------------------------------------------- girdi / çıktı
def read_image(path):
    img = cv2.imread(str(path), flags=cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(
            f"Görüntü okunamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(IMAGE_EXTS))}) olduğunu kontrol et.")
    return img


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputError(
            f"Video açılamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(VIDEO_EXTS))}) olduğunu kontrol et.")
    return cap


def load_frame(path, frame_index=0):
    """Tek kare (notebook için): görüntü ya da videonun frame_index. karesi."""
    path = Path(path)
    if not path.is_file():
        raise InputError(
            f"Girdi bulunamadı: {path}\n"
            "  -> Doğru yolu ver ya da AYARLAR'daki yolu düzelt.")
    if path.suffix.lower() not in VIDEO_EXTS:
        return read_image(path)
    cap = open_video(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise InputError(f"Videonun {frame_index}. karesi okunamadı: {path}\n"
                         "  -> Daha küçük bir kare numarası ver.")
    return frame


def write_image(path, img):
    if not cv2.imwrite(str(path), img):
        raise InputError(f"Görüntü yazılamadı: {path}\n  -> --out-dir'i ve yazma iznini kontrol et.")


def open_writer(path, fourcc, fps, size, is_color):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size,
                             isColor=is_color)
    if not writer.isOpened():
        raise InputError(f"Video yazılamadı: {path} ({fourcc})\n"
                         "  -> --out-dir'i ve yazma iznini kontrol et.")
    return writer


# [EK] Kanal katkısını renkle gösteren karşılaştırma görseli.
def draw_compare(frame, white, yellow, params):
    """Solda girdi, sağda maske: beyaz = HLS-L/S, sarı = LAB-B kanalının bulduğu."""
    white_channel, white_range, yellow_b_range = params
    width = frame.shape[1]
    fs = 0.8 * width / 1280.0
    th = max(1, round(2 * width / 1280.0))
    y0 = round(35 * fs / 0.8)

    view = np.zeros_like(frame)
    view[yellow > 0] = (0, 255, 255)
    view[white > 0] = (255, 255, 255)
    left = frame.copy()
    for img, text in (
            (left, "Girdi"),
            (view, f"Maske - beyaz: HLS-{white_channel} {white_range[0]}-{white_range[1]}, "
                   f"sari: LAB-B {yellow_b_range[0]}-{yellow_b_range[1]}")):
        cv2.putText(img, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    thickness=th + 2, lineType=cv2.LINE_AA)
        cv2.putText(img, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255),
                    thickness=th, lineType=cv2.LINE_AA)
    return cv2.hconcat([left, view])


# --------------------------------------------------------------------------- işleme
def process_image(path, out_dir, params, compare):
    frame = read_image(path)
    white, yellow, mask = color_masks(frame, *params)
    mask_path = out_dir / f"{path.stem}_maske.png"
    write_image(mask_path, mask)
    if compare:
        write_image(out_dir / f"{path.stem}_karsilastirma.png",
                    draw_compare(frame, white, yellow, params))
    print(f"{path.name}: {format_ratios(white, yellow, mask)} -> {mask_path}")


def process_folder(path, out_dir, params, compare):
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise InputError(f"Klasörde görüntü yok: {path}\n"
                         f"  -> Desteklenen uzantılar: {', '.join(sorted(IMAGE_EXTS))}")
    sub_dir = out_dir / path.name
    sub_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        process_image(file, sub_dir, params, compare)
    print(f"{len(files)} görüntü işlendi -> {sub_dir}")


def process_video(path, out_dir, params, compare, max_frames):
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (math.isfinite(fps) and fps > 0):
        fps = 25.0
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise InputError(f"Videodan hiç kare okunamadı: {path}")

    height, width = frame.shape[:2]
    mask_path = out_dir / f"{path.stem}_maske.mkv"
    writers = []
    try:
        # [EK] FFV1 kayıpsızdır; mp4 sıkıştırması maskenin 0/255 değerlerini bozar.
        mask_writer = open_writer(mask_path, "FFV1", fps, (width, height), is_color=False)
        writers.append(mask_writer)
        compare_writer = None
        if compare:
            compare_writer = open_writer(out_dir / f"{path.stem}_karsilastirma.mp4", "mp4v",
                                         fps, (2 * width, height), is_color=True)
            writers.append(compare_writer)

        count = 0
        totals = np.zeros(3)
        while ok and (max_frames is None or count < max_frames):
            if frame.shape[:2] != (height, width):
                raise InputError(
                    f"Videonun {count}. karesinin boyutu değişti "
                    f"({frame.shape[1]}x{frame.shape[0]}, ilk kare {width}x{height}).")
            white, yellow, mask = color_masks(frame, *params)
            mask_writer.write(mask)
            if compare_writer is not None:
                compare_writer.write(draw_compare(frame, white, yellow, params))
            totals += (mask_ratio(white), mask_ratio(yellow), mask_ratio(mask))
            count += 1
            ok, frame = cap.read()
    finally:
        cap.release()
        for writer in writers:
            writer.release()

    white_avg, yellow_avg, mask_avg = 100 * totals / count
    print(f"{path.name}: {count} kare, ortalama maske %{mask_avg:.2f} "
          f"(beyaz %{white_avg:.2f}, sarı %{yellow_avg:.2f}) -> {mask_path}")


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
        description="Renkli kareden HLS-L (ya da S) ve LAB-B eşiklemesiyle ikili şerit maskesi üretir.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Görüntü, görüntü klasörü ya da video (varsayılan: {INPUT_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--white-channel", choices=("L", "S"), default=WHITE_CHANNEL,
                   help=f"Beyaz çizgi için HLS kanalı (varsayılan: {WHITE_CHANNEL})")
    p.add_argument("--white-l-range", type=int, nargs=2, default=WHITE_L_RANGE,
                   metavar=("ALT", "UST"), help=f"HLS-L aralığı (varsayılan: {WHITE_L_RANGE})")
    p.add_argument("--white-s-range", type=int, nargs=2, default=WHITE_S_RANGE,
                   metavar=("ALT", "UST"), help=f"HLS-S aralığı (varsayılan: {WHITE_S_RANGE})")
    p.add_argument("--yellow-b-range", type=int, nargs=2, default=YELLOW_B_RANGE,
                   metavar=("ALT", "UST"), help=f"LAB-B aralığı (varsayılan: {YELLOW_B_RANGE})")
    p.add_argument("--compare", action="store_true",
                   help="Girdi ve maskeyi yan yana kaydet")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                   help="Videoda en çok kaç kare işlensin (varsayılan: tümü)")
    return p.parse_args(argv)


def validate_args(args):
    for name, (low, high) in (("--white-l-range", args.white_l_range),
                              ("--white-s-range", args.white_s_range),
                              ("--yellow-b-range", args.yellow_b_range)):
        if not 0 <= low <= high <= 255:
            raise InputError(f"{name} 0 <= ALT <= UST <= 255 olmalı, verilen: {low} {high}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise InputError(f"--max-frames pozitif olmalı, verilen: {args.max_frames}")


# --------------------------------------------------------------------------- ana akış
def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    white_range = args.white_l_range if args.white_channel == "L" else args.white_s_range
    params = (args.white_channel, tuple(white_range), tuple(args.yellow_b_range))

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

    print(f"Eşikler: beyaz HLS-{params[0]} {params[1][0]}-{params[1][1]}, "
          f"sarı LAB-B {params[2][0]}-{params[2][1]}")
    if path.is_dir():
        process_folder(path, args.out_dir, params, args.compare)
    elif path.suffix.lower() in VIDEO_EXTS:
        process_video(path, args.out_dir, params, args.compare, args.max_frames)
    else:
        process_image(path, args.out_dir, params, args.compare)


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
