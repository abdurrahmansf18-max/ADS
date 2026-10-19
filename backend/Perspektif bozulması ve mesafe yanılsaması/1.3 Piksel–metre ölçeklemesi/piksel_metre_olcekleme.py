"""
1.3 Piksel–metre ölçeklemesi — kuş bakışında 1 pikselin kaç metre olduğu (kurulum, bir kez).

Ne işe yarar:
    Sapma, eğrilik ve engel mesafesinin metre cinsinden verilebilmesi için piksel–metre oranını bulur:
    xm (yatay) ve ym (dikey). "80 piksel saptın" bir anlam taşımaz, "0.46 m saptın" taşır.

Nasıl çalışır:
    Kuş bakışı düz yol karesinde şerit çizgileri renkle bulunur. İki çizgi arası ve kesik çizgi boyu
    piksel olarak ölçülür ve gerçek ölçülere bölünür: xm = 3.66 m / şerit genişliği (px),
    ym = 3.05 m / kesik çizgi boyu (px). Kesikler yalnız kesik çizgide aranır: satırların çoğunu kaplayan
    düz çizgi atlanır, kopan bir parçası kesik sayılmaz. Birden çok karenin ortancası alınır.

Bağlantılar:
    Alır : 1.2'nin kuş bakışına çevirdiği düz yol kareleri; ölçü yalnız kuş bakışında doğrudur.
    Verir: data/uretilen_veri/kurulum_dosyalari/<kurulum>/olcek.npz (xm_per_pix, ym_per_pix,
           lane_width_px) -> 2.5 (xm), 3.1-3.3 (xm, ym), 5.5 (ym) ve ana hat.

Parametreler:
    --input       Başka gereksinimden: 1.2'nin kuş bakışı düz yol kareleri.
    --lane-width-m, --dash-length-m
                  Gerçek veri: yolun gerçek ölçüleri (ABD otoyolu: 3.66 m ve 3.05 m). [DEĞİŞTİR]
    --lane-px, --dash-px   İsteğe bağlı: pikseller elle ölçüldüyse.
    --out ve diğerleri     Ayar: kayıt yeri, renk eşikleri.
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

# [DEĞİŞTİR] Düz yol kesitinin kuş bakışı (düzleştirilmiş) hali:
# tek görüntü, görüntü klasörü ya da video.
INPUT_PATH = PROJECT_ROOT / "outputs" / "1.2 Perspektif dönüşümü" / "kus_bakisi_duz_yol"

# [DEĞİŞTİR] Gerçek şerit genişliği (m), iki çizginin merkezleri arası.
# Yol sınıfına göre kabaca 3.0 - 3.75 m.
LANE_WIDTH_M = 3.66

# [DEĞİŞTİR] Gerçek kesik çizgi boyu (m). Ör. ABD otoyollarında ~3.0 m (10 ft).
DASH_LENGTH_M = 3.05

# Elle ölçülmüş piksel değerleri. None ise görüntüden ölçülür.
LANE_WIDTH_PX = None
DASH_LENGTH_PX = None
# Elle girişte ölçeğin geçerli olduğu kuş bakışı görüntü boyutu (genişlik, yükseklik).
# None ise görüntüden okunur.
IMAGE_SIZE = None

# Çıktılar.
# ölçek (diğer modüller okur)
SCALE_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "olcek.npz"
COMPARE_DIR = OUTPUT_DIR                           # --compare görselleri

# Video girişi: her FRAME_STEP karede bir ölçüm, en çok MAX_FRAMES kare.
FRAME_STEP = 10
MAX_FRAMES = 30

# Şerit maskesi eşikleri (HLS, OpenCV ölçeği: H 0-180, L ve S 0-255).
WHITE_L_MIN = 200
YELLOW_S_MIN = 100
YELLOW_H_RANGE = (15, 35)

# Piksel parametreleri: ~1280x720 kuş bakışı görüntüye göre seçildi;
# farklı çözünürlükte orantılı değiştir.
LINE_BAND_PX = 40  # çizgi merkezinin iki yanındaki arama bandı
MIN_DASH_PX = 10   # bundan kısa parçalar gürültü sayılır
# [EK] Çizgi bandında çizgi pikseli olan satırların oranı bunun üstündeyse çizgi düz sayılır ve kesik aranmaz.
# Düz çizgi satırların ~%90-100'ünü kaplar (bir yerden kopsa da); kesik çizgi aralıklarıyla ~%25-40'ını
# (ABD: 3.05 m çizgi, 9.14 m boşluk). Böylece düz çizginin kopan parçası (ör. düzeltilmiş görüntünün kenarındaki
# boş bölge yüzünden) kesik çizgi sayılıp ölçeği bozmaz.
SOLID_COVERAGE = 0.6

# Sonuçta gösterilen örnek: "merkezden N piksel sapma" kaç metre?
EXAMPLE_OFFSET_PX = 80
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


class MeasureError(InputError):
    """Bir karede ölçüm yapılamadı; klasör / video girişinde o kare atlanır."""


# [EK] Kaydedilen ölçeği bir kez yükleyen sınıf. Diğer modüller kopyalayarak kullanır.
class PixelMeterScale:
    """Kaydedilmiş piksel–metre ölçeğini yükler ve piksel -> metre çevirir."""

    REQUIRED_KEYS = ("xm_per_pix", "ym_per_pix", "image_size")

    def __init__(self, path=SCALE_PATH):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Ölçek dosyası bulunamadı: {path}\n"
                "  -> Önce 1.3 piksel_metre_olcekleme.py ile ölçeği hesapla.")
        try:
            with np.load(path) as data:
                missing = [k for k in self.REQUIRED_KEYS if k not in data.files]
                if missing:
                    raise ValueError(f"eksik alanlar: {', '.join(missing)}")
                self.xm_per_pix = float(data["xm_per_pix"])
                self.ym_per_pix = float(data["ym_per_pix"])
                self.image_size = tuple(int(v) for v in data["image_size"])
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(
                f"Ölçek dosyası okunamadı: {path} ({exc})\n"
                "  -> Dosyayı 1.3 piksel_metre_olcekleme.py ile yeniden üret.") from None
        self.path = path

    def check_size(self, width, height):
        """Ölçek yalnızca hesaplandığı kuş bakışı görüntü boyutunda geçerlidir."""
        if (int(width), int(height)) != self.image_size:
            raise ValueError(
                f"Görüntü boyutu {width}x{height}, ancak ölçek "
                f"{self.image_size[0]}x{self.image_size[1]} kuş bakışı görüntü için "
                f"hesaplandı ({self.path}).\n"
                "  -> Ölçeği bu çözünürlükte yeniden hesapla ya da görüntüyü aynı boyutta düzleştir.")

    def x_to_m(self, px):
        return px * self.xm_per_pix

    def y_to_m(self, px):
        return px * self.ym_per_pix


def compute_scale(real_m, pixels):
    """Gereksinimin özü: metre / piksel."""
    return real_m / pixels


# --------------------------------------------------------------------------- girdi
def read_image(path):
    img = cv2.imread(str(path), flags=cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(
            f"Görüntü okunamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(IMAGE_EXTS))}) olduğunu kontrol et.")
    return img


def iter_video(path, frame_step, max_frames):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputError(
            f"Video açılamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(VIDEO_EXTS))}) olduğunu kontrol et.")
    index = used = 0
    try:
        while used < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % frame_step == 0:
                used += 1
                yield f"{path.stem}_kare{index:05d}", frame
            index += 1
    finally:
        cap.release()
    if used == 0:
        raise InputError(f"Videodan hiç kare okunamadı: {path}")


def iter_frames(path, frame_step, max_frames):
    """(ad, görüntü) çiftleri: tek görüntü, görüntü klasörü ya da video."""
    path = Path(path)
    if not path.exists():
        raise InputError(
            f"Girdi bulunamadı: {path}\n"
            "  -> --input ile doğru yolu ver ya da AYARLAR'daki INPUT_PATH'i düzelt.\n"
            "  -> Görüntü yoksa piksel değerlerini elle ver: "
            "--lane-px ... --dash-px ... --image-size GENISLIK YUKSEKLIK")
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise InputError(
                f"Klasörde görüntü yok: {path}\n"
                f"  -> Desteklenen uzantılar: {', '.join(sorted(IMAGE_EXTS))}")
        for file in files:
            yield file.stem, read_image(file)
    elif path.suffix.lower() in VIDEO_EXTS:
        yield from iter_video(path, frame_step, max_frames)
    else:
        yield path.stem, read_image(path)


# --------------------------------------------------------------------------- ölçüm
def lane_mask(img_bgr, white_l_min, yellow_s_min, yellow_h_range):
    """Beyaz ve sarı şerit pikselleri için ikili maske."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Girdi zaten ikili maskeyse (ör. eşikleme modülünün çıktısı) aynen kullan.
    if np.isin(gray, (0, 255)).all():
        return gray > 0

    # [EK] HLS: beyaz çizgi yüksek L, sarı çizgi belirli H ve yüksek S ile ayrılır.
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    hue, light, sat = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    white = light >= white_l_min
    yellow = (sat >= yellow_s_min) & (hue >= yellow_h_range[0]) & (hue <= yellow_h_range[1])
    return white | yellow


def find_line_centers(mask, band_px, min_line_px):
    """Sol ve sağ şerit çizgisinin x merkezi (alt-piksel)."""
    width = mask.shape[1]
    # [EK] Sütun histogramı: düz yolda çizgiler dikey olduğundan sütunlarda tepe yapar.
    histogram = mask.sum(axis=0).astype(np.float64)
    mid = width // 2
    centers = []
    for lo, hi, side in ((0, mid, "Sol"), (mid, width, "Sağ")):
        peak = lo + int(np.argmax(histogram[lo:hi]))
        if histogram[peak] < min_line_px:
            raise MeasureError(
                f"{side} şerit çizgisi bulunamadı (görüntünün {side.lower()} yarısında "
                "yeterli çizgi pikseli yok).\n"
                "  -> Girdinin düz yolun kuş bakışı görüntüsü olduğunu kontrol et ya da "
                "eşikleri (--white-l-min, --yellow-s-min, --yellow-h-range) ayarla.")
        # [EK] Tepe çevresinde ağırlıklı merkez: kalın çizgide argmax çizginin
        # herhangi bir sütununa düşebilir; ağırlıklı ortalama merkezi verir.
        a, b = max(lo, peak - band_px), min(hi, peak + band_px + 1)
        weights = histogram[a:b]
        centers.append(float(np.dot(np.arange(a, b), weights) / weights.sum()))
    return centers[0], centers[1]


def band_coverage(mask, center_x, band_px):
    """Çizgi bandında çizgi pikseli bulunan satırların oranı (düz çizgi ~1, kesik çizgi ~0.25-0.4)."""
    x0 = max(0, int(round(center_x)) - band_px)
    x1 = min(mask.shape[1], int(round(center_x)) + band_px + 1)
    return float(np.count_nonzero(mask[:, x0:x1].any(axis=1)) / mask.shape[0])


def find_dashes(mask, center_x, band_px, min_dash_px):
    """Çizgi bandındaki tam kesik çizgi parçaları: [(x, y, genişlik, yükseklik), ...]."""
    height, width = mask.shape
    x0 = max(0, int(round(center_x)) - band_px)
    x1 = min(width, int(round(center_x)) + band_px + 1)
    band = mask[:, x0:x1].astype(np.uint8)
    # [EK] Bağlı bileşenler: her parça ayrı bulunur; aradaki gürültü pikselleri
    # iki parçayı birleştirmez (satır izdüşümünde birleştirirdi).
    _, _, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    dashes = []
    for left, top, w, h, _area in stats[1:]:
        if top == 0 or top + h == height:
            continue  # görüntü kenarında kesilmiş parça, boyu güvenilmez
        if h < min_dash_px:
            continue  # gürültü
        dashes.append((int(left) + x0, int(top), int(w), int(h)))
    return dashes


def measure_frame(img, args):
    mask = lane_mask(img, args.white_l_min, args.yellow_s_min, args.yellow_h_range)
    left_x, right_x = find_line_centers(mask, args.band_px, args.min_dash_px)
    lane_px = right_x - left_x
    if lane_px <= 2 * args.band_px:
        raise MeasureError(
            f"Çizgiler arası {lane_px:.0f} px, arama bandına (2 x {args.band_px} px) "
            "göre çok dar.\n  -> --band-px'i küçült ya da görüntüyü kontrol et.")
    # İki taraf da taranır (orta şeritte iki çizgi de kesik olabilir); düz çizgi tarafı atlanır.
    solid = [band_coverage(mask, x, args.band_px) >= SOLID_COVERAGE for x in (left_x, right_x)]
    dashes = [dash for x, is_solid in zip((left_x, right_x), solid) if not is_solid
              for dash in find_dashes(mask, x, args.band_px, args.min_dash_px)]
    return {"mask": mask, "left_x": left_x, "right_x": right_x,
            "lane_px": lane_px, "dashes": dashes, "solid": solid}


def measure_input(args):
    """Tüm karelerde ölçer; şerit genişliği ve kesik çizgi boyu için ortanca verir."""
    path = Path(args.input)
    multi = path.is_dir() or path.suffix.lower() in VIDEO_EXTS
    lane_values, dash_values = [], []
    frame_size = None
    skipped = 0

    for name, img in iter_frames(path, args.frame_step, args.max_frames):
        size = (img.shape[1], img.shape[0])
        if frame_size is None:
            frame_size = size
        elif size != frame_size:
            raise InputError(
                f"Farklı çözünürlükte kare: {name} ({size[0]}x{size[1]}), ilk kare "
                f"{frame_size[0]}x{frame_size[1]}.\n"
                "  -> Piksel ölçeği tek bir çözünürlükte geçerlidir; aynı boyutta kuş "
                "bakışı görüntüler ver.")
        try:
            meas = measure_frame(img, args)
        except MeasureError as exc:
            if not multi:
                raise
            skipped += 1
            print(f"UYARI: {name} atlandı: {str(exc).splitlines()[0]}", file=sys.stderr)
            continue

        lane_values.append(meas["lane_px"])
        dash_values.extend(d[3] for d in meas["dashes"])
        if args.compare:
            save_compare(img, meas, args, name)

    if not lane_values:
        raise MeasureError(
            f"Hiçbir karede iki şerit çizgisi bulunamadı ({skipped} kare atlandı).\n"
            "  -> Girdinin düz yolun kuş bakışı görüntüsü olduğunu kontrol et ya da "
            "eşikleri ayarla.")
    # [EK] Ortanca: tek tük hatalı karelere ve kısmi parçalara karşı dayanıklı.
    return {
        "image_size": frame_size,
        "n_frames": len(lane_values),
        "skipped": skipped,
        "lane_px": float(np.median(lane_values)),
        "lane_range": (min(lane_values), max(lane_values)),
        "dash_px": float(np.median(dash_values)) if dash_values else None,
        "dash_count": len(dash_values),
        "dash_range": (min(dash_values), max(dash_values)) if dash_values else None,
    }


# --------------------------------------------------------------------------- görsel
def put_label(img, text, org, color, font_scale, thickness):
    """Okunaklı olsun diye siyah kenarlı yazı (Türkçe karakter kullanma)."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                thickness=thickness + 2, lineType=cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                thickness=thickness, lineType=cv2.LINE_AA)


def draw_compare(img, meas, lane_width_m, dash_length_m):
    """Solda girdi, sağda maske üzerinde ölçüm."""
    height, width = img.shape[:2]
    fs = 0.8 * width / 1280.0
    th = max(1, round(2 * width / 1280.0))
    red, green, yellow, white = (0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 255, 255)

    view = cv2.cvtColor(meas["mask"].astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    left_x, right_x = round(meas["left_x"]), round(meas["right_x"])
    y = height * 3 // 4
    cv2.arrowedLine(view, (left_x, y), (right_x, y), red, thickness=th,
                    line_type=cv2.LINE_AA, tipLength=0.03)
    cv2.arrowedLine(view, (right_x, y), (left_x, y), red, thickness=th,
                    line_type=cv2.LINE_AA, tipLength=0.03)
    text = f"Serit genisligi: {meas['lane_px']:.1f} px = {lane_width_m:g} m"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    put_label(view, text, ((left_x + right_x - tw) // 2, y - 12), red, fs, th)

    for x, top, w, h in meas["dashes"]:
        cv2.rectangle(view, (x, top), (x + w - 1, top + h - 1), green, thickness=th)
        label = f"{h} px"
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = x - lw - 6 if x > width // 2 else x + w + 6
        put_label(view, label, (tx, top + h // 2), green, fs, th)

    put_label(view, "Olcum", (10, round(35 * fs / 0.8)), white, fs, th)
    lines = [f"xm_per_pix = {lane_width_m / meas['lane_px']:.6f} m/px"]
    if meas["dashes"]:
        dash_px = float(np.median([d[3] for d in meas["dashes"]]))
        lines.append(f"ym_per_pix = {dash_length_m / dash_px:.6f} m/px "
                     f"(kesik cizgi {dash_px:.0f} px = {dash_length_m:g} m)")
    else:
        lines.append("Tam kesik cizgi parcasi yok")
    for i, line in enumerate(lines):
        put_label(view, line, (10, round((75 + 35 * i) * fs / 0.8)), yellow, fs, th)

    left = img.copy()
    put_label(left, "Girdi (kus bakisi)", (10, round(35 * fs / 0.8)), white, fs, th)
    return cv2.hconcat([left, view])


def save_compare(img, meas, args, name):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{name}_karsilastirma.png"
    if not cv2.imwrite(str(out_path), draw_compare(img, meas, args.lane_width_m,
                                                   args.dash_length_m)):
        raise InputError(f"Görsel yazılamadı: {out_path}\n  -> --out-dir'i kontrol et.")
    print(f"Kaydedildi: {out_path}")


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
        description="Bilinen gerçek ölçülerden xm_per_pix ve ym_per_pix (m/px) hesaplar.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Kuş bakışı düz yol: görüntü, klasör ya da video (varsayılan: {INPUT_PATH})")
    p.add_argument("--lane-width-m", type=float, default=LANE_WIDTH_M,
                   help=f"Gerçek şerit genişliği, çizgi merkezleri arası, m (varsayılan: {LANE_WIDTH_M})")
    p.add_argument("--dash-length-m", type=float, default=DASH_LENGTH_M,
                   help=f"Gerçek kesik çizgi boyu, m (varsayılan: {DASH_LENGTH_M})")
    p.add_argument("--lane-px", type=float, default=LANE_WIDTH_PX,
                   help="Şerit genişliği, px (verilmezse görüntüden ölçülür)")
    p.add_argument("--dash-px", type=float, default=DASH_LENGTH_PX,
                   help="Kesik çizgi boyu, px (verilmezse görüntüden ölçülür)")
    p.add_argument("--image-size", type=int, nargs=2, default=IMAGE_SIZE,
                   metavar=("GENISLIK", "YUKSEKLIK"),
                   help="Piksel değerlerinin ölçüldüğü kuş bakışı görüntü boyutu "
                        "(görüntü kullanılmıyorsa zorunlu)")
    p.add_argument("--out", type=Path, default=SCALE_PATH,
                   help=f"Ölçek dosyası, .npz (varsayılan: {SCALE_PATH})")
    p.add_argument("--compare", action="store_true",
                   help="Girdi ve ölçüm görselini yan yana kaydet")
    p.add_argument("--out-dir", type=Path, default=COMPARE_DIR,
                   help=f"--compare görsellerinin klasörü (varsayılan: {COMPARE_DIR})")
    p.add_argument("--frame-step", type=int, default=FRAME_STEP,
                   help=f"Videoda her N karede bir ölç (varsayılan: {FRAME_STEP})")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                   help=f"Videoda en çok kaç kare ölçülsün (varsayılan: {MAX_FRAMES})")
    p.add_argument("--white-l-min", type=int, default=WHITE_L_MIN,
                   help=f"Beyaz çizgi için HLS L alt eşiği, 0-255 (varsayılan: {WHITE_L_MIN})")
    p.add_argument("--yellow-s-min", type=int, default=YELLOW_S_MIN,
                   help=f"Sarı çizgi için HLS S alt eşiği, 0-255 (varsayılan: {YELLOW_S_MIN})")
    p.add_argument("--yellow-h-range", type=int, nargs=2, default=YELLOW_H_RANGE,
                   metavar=("ALT", "UST"),
                   help=f"Sarı ton aralığı, HLS H 0-180 (varsayılan: {YELLOW_H_RANGE})")
    p.add_argument("--band-px", type=int, default=LINE_BAND_PX,
                   help=f"Çizgi merkezinin iki yanındaki arama bandı, px (varsayılan: {LINE_BAND_PX})")
    p.add_argument("--min-dash-px", type=int, default=MIN_DASH_PX,
                   help=f"Bundan kısa parçalar gürültü sayılır, px (varsayılan: {MIN_DASH_PX})")
    p.add_argument("--example-px", type=float, default=EXAMPLE_OFFSET_PX,
                   help=f"Sonuçta metreye çevrilecek örnek sapma, px (varsayılan: {EXAMPLE_OFFSET_PX})")
    return p.parse_args(argv)


def validate_args(args):
    def require_positive(name, value):
        if value is not None and not (math.isfinite(value) and value > 0):
            raise InputError(f"{name} pozitif bir sayı olmalı, verilen: {value}")

    for name, value in (("--lane-width-m", args.lane_width_m),
                        ("--dash-length-m", args.dash_length_m),
                        ("--lane-px", args.lane_px), ("--dash-px", args.dash_px),
                        ("--frame-step", args.frame_step), ("--max-frames", args.max_frames),
                        ("--band-px", args.band_px), ("--min-dash-px", args.min_dash_px),
                        ("--example-px", args.example_px)):
        require_positive(name, value)
    if args.image_size is not None:
        for value in args.image_size:
            require_positive("--image-size", value)
    for name, value in (("--white-l-min", args.white_l_min),
                        ("--yellow-s-min", args.yellow_s_min)):
        if not 0 <= value <= 255:
            raise InputError(f"{name} 0-255 aralığında olmalı, verilen: {value}")
    h_low, h_high = args.yellow_h_range
    if not 0 <= h_low <= h_high <= 180:
        raise InputError(
            f"--yellow-h-range 0 <= ALT <= UST <= 180 olmalı, verilen: {h_low} {h_high}")
    if args.out.suffix != ".npz":
        raise InputError(f"--out .npz uzantılı olmalı, verilen: {args.out}")
    if args.lane_px is not None and args.dash_px is not None and args.image_size is None:
        raise InputError(
            "Piksel değerleri elle verildiğinde ölçüldükleri kuş bakışı görüntünün "
            "boyutu da gerekli.\n"
            "  -> --image-size GENISLIK YUKSEKLIK ekle (ör. --image-size 1280 720). "
            "Piksel ölçeği yalnızca ölçüldüğü çözünürlükte geçerlidir.")


# --------------------------------------------------------------------------- ana akış
def main(argv=None):
    args = parse_args(argv)
    validate_args(args)

    lane_px, dash_px = args.lane_px, args.dash_px
    image_size = tuple(args.image_size) if args.image_size is not None else None
    meas = None
    if lane_px is None or dash_px is None:
        meas = measure_input(args)
        if image_size is not None and image_size != meas["image_size"]:
            raise InputError(
                f"--image-size {image_size[0]}x{image_size[1]}, ancak girdi "
                f"{meas['image_size'][0]}x{meas['image_size'][1]}.\n"
                "  -> Elle verilen piksel değerleri ile girdi aynı çözünürlükte olmalı.")
        image_size = meas["image_size"]
        if lane_px is None:
            lane_px = meas["lane_px"]
        if dash_px is None:
            dash_px = meas["dash_px"]
            if dash_px is None:
                raise InputError(
                    "Girdide tam (üst ve alt kenara değmeyen) kesik çizgi parçası bulunamadı; düz çizgiler "
                    "sayılmaz.\n"
                    "  -> Kesik çizginin en az bir parçası tamamen görünen bir kare kullan, "
                    "klasör ya da video ver veya --dash-px ile elle gir.")
    elif args.compare:
        print("UYARI: Görüntü kullanılmadığı için --compare yok sayıldı.", file=sys.stderr)

    width, height = image_size
    if lane_px >= width or dash_px >= height:
        raise InputError(
            f"Piksel değerleri görüntü boyutuyla uyuşmuyor: şerit {lane_px:g} px "
            f"(genişlik {width}), kesik çizgi {dash_px:g} px (yükseklik {height}).\n"
            "  -> --lane-px / --dash-px / --image-size değerlerini kontrol et.")

    xm_per_pix = compute_scale(args.lane_width_m, lane_px)
    ym_per_pix = compute_scale(args.dash_length_m, dash_px)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez(args.out,
                 xm_per_pix=np.float64(xm_per_pix), ym_per_pix=np.float64(ym_per_pix),
                 image_size=np.array(image_size, dtype=np.int64),
                 lane_width_m=np.float64(args.lane_width_m), lane_width_px=np.float64(lane_px),
                 dash_length_m=np.float64(args.dash_length_m), dash_length_px=np.float64(dash_px),
                 n_frames=np.int64(meas["n_frames"] if meas else 0),
                 source=np.str_(str(args.input) if meas else "elle"))
    except OSError as exc:
        raise InputError(f"Ölçek dosyası yazılamadı: {args.out} ({exc})\n"
                         "  -> --out yolunu ve yazma iznini kontrol et.") from None

    # Kaydedileni diğer modüllerin yapacağı gibi geri yükle (yuvarlak tur denetimi).
    scale = PixelMeterScale(args.out)
    scale.check_size(width, height)

    lane_src = "elle" if args.lane_px is not None else "ölçüldü"
    dash_src = "elle" if args.dash_px is not None else "ölçüldü"
    if meas:
        print(f"Kaynak          : {args.input} ({meas['n_frames']} kare ölçüldü, "
              f"{meas['skipped']} atlandı)")
    print(f"Görüntü boyutu  : {width}x{height} (ölçek yalnızca bu boyutta geçerli)")
    print(f"Şerit genişliği : {args.lane_width_m:g} m / {lane_px:.1f} px ({lane_src})", end="")
    if meas and args.lane_px is None and meas["n_frames"] > 1:
        print(f"  [karelerde {meas['lane_range'][0]:.1f} - {meas['lane_range'][1]:.1f} px]", end="")
    print()
    print(f"Kesik çizgi boyu: {args.dash_length_m:g} m / {dash_px:.1f} px ({dash_src})", end="")
    if meas and args.dash_px is None:
        print(f"  [{meas['dash_count']} parça, {meas['dash_range'][0]} - "
              f"{meas['dash_range'][1]} px]", end="")
    print()
    print(f"xm_per_pix = {scale.xm_per_pix:.6f} m/px")
    print(f"ym_per_pix = {scale.ym_per_pix:.6f} m/px")
    print(f"Örnek: merkezden {args.example_px:g} px yanal sapma = "
          f"{scale.x_to_m(args.example_px):.3f} m")
    print(f"Kaydedildi: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (InputError, FileNotFoundError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
