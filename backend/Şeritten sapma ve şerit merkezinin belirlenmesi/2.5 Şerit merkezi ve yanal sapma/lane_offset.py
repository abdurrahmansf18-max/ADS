"""
2.5 Şerit merkezi ve yanal sapma — aracın şerit ortasına göre ne kadar kaydığı.

Ne işe yarar:
    Aracın şeridin ortasında mı, sağa ya da sola mı kaydığını metre cinsinden söyler; şeritten çıkmaya
    yaklaşınca uyarı verir.

Nasıl çalışır:
    Görüntünün alt satırında sol ve sağ polinomun x değeri bulunur; ikisinin ortası şerit merkezidir.
    Görüntü merkezi (kamera) ile şerit merkezi arasındaki fark xm ile metreye çevrilir.
    Pozitif sapma = araç merkezin sağında.

Bağlantılar:
    Alır : 2.4'ün polinomları, şerit çizgilerinin yeri için; 1.3'ün ölçek dosyası, piksel farkını metreye
           çevirmek için.
    Verir: sapma (m), yön ve uyarı -> ekrana ve lane_offset.csv'ye; ana hatta ekrana yazılır.
    Ana hat: main.py her karede LaneOffsetCalculator.compute() çağırır.

Parametreler:
    --input       Başka gereksinimden: 2.4'ün polinom dosyaları (varsayılan outputs/2.4 Polinom uydurma).
    --scale       Başka gereksinimden: 1.3'ün olcek.npz dosyası (kurulumun klasöründe).
    --camera-offset   Gerçek veri: kameranın araç ortasına uzaklığı (ölçülmeli; 0 = ortada). [DEĞİŞTİR]
    --warn-offset     Gerçek veriye göre: uyarı sınırı, araç genişliğine bağlıdır (0.5 m). [DEĞİŞTİR]
    --left-fit, --right-fit, --xm-per-pix   İsteğe bağlı: değerleri elle vermek için.
"""

import argparse
import csv
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# AYARLAR
# Komut satırından argüman verilmezse bu değerler kullanılır.
# [DEĞİŞTİR] ile işaretlenenleri kendi verine göre ayarlaman gerekir.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# [DEĞİŞTİR] Girdi: şerit polinomlarını içeren .npz dosyası ya da bu dosyaların klasörü
# (polinom uydurmanın çıktısı). Polinomlar komut satırından da verilebilir (--left-fit/--right-fit).
INPUT_PATH = PROJECT_ROOT / "outputs" / "2.4 Polinom uydurma"

# [DEĞİŞTİR] Kuş bakışı görüntüde yatay metre/piksel oranı (piksel–metre ölçeklemesinin çıktısı).
# None: SCALE_PATH'e bakılır; o da yoksa ölçek standart şerit genişliğinden tahmin edilir.
XM_PER_PIX = None

# xm_per_pix'i içeren .npz dosyası (XM_PER_PIX verilmemişse kullanılır). None: kullanma.
SCALE_PATH = PROJECT_ROOT / "data" / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "olcek.npz"

# Görüntü boyutu (genişlik, yükseklik). None: polinom dosyasındaki image_size kullanılır.
IMAGE_SIZE = None

# Girdi dosyalarındaki anahtar adları; polinom ya da ölçek dosyası farklı ad kullanıyorsa değiştir.
LEFT_FIT_KEY = "left_fit"
RIGHT_FIT_KEY = "right_fit"
IMAGE_SIZE_KEY = "image_size"
XM_PER_PIX_KEY = "xm_per_pix"
FRAME_KEY = "frame"

# xm_per_pix verilmezse ölçek tahmini için kullanılan standart şerit genişliği (metre).
# Türkiye'de şerit genişliği genellikle 3.5-3.75 m'dir.
LANE_WIDTH_M = 3.66

# Kameranın aracın ortasına göre yatay konumu, metre (sağ +, sol -). 0: kamera tam ortada.
CAMERA_OFFSET_M = 0.0

# Bundan küçük sapmalar "ortada" sayılır (metre).
CENTER_TOLERANCE_M = 0.1

# Sapma bu değere ulaşınca şeritten çıkma uyarısı verilir (metre). 3.5 m şeritte 1.8 m
# genişliğindeki bir aracın tekerleği ~0.85 m sapmada çizgiye değer; 0.5 m erken uyarı payı bırakır.
DEPARTURE_WARNING_M = 0.5

# Sonuç tablosunun ve kontrol görüntülerinin yazılacağı klasör.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / Path(__file__).resolve().parent.name

# True: polinomları, şerit merkezini ve sapmayı gösteren kontrol görüntüsü kaydedilir.
VISUALIZE = False
# =============================================================================

# [EK] Bu aralığın dışındaki şerit genişliği yanlış polinom ya da xm_per_pix belirtisidir (metre).
LANE_WIDTH_RANGE_M = (2.5, 5.0)

DIRECTION_ASCII = {"sağa": "saga", "sola": "sola", "ortada": "ortada"}
CSV_COLUMNS = [
    "kaynak", "kare", "sol_x_px", "sag_x_px", "serit_merkezi_px", "goruntu_merkezi_px", "sapma_m", "yon",
    "serit_genisligi_m", "xm_per_pix", "xm_tahmini", "cikis_uyarisi",
]

# Kontrol görüntüsü renkleri (BGR).
LEFT_COLOR = (0, 0, 255)
RIGHT_COLOR = (255, 0, 0)
CENTER_COLOR = (0, 255, 0)
VEHICLE_COLOR = (255, 255, 255)
ARROW_COLOR = (0, 255, 255)
WARNING_COLOR = (0, 0, 255)


@dataclass
class LaneOffset:
    left_x_px: float
    right_x_px: float
    lane_center_px: float
    image_center_px: float
    offset_m: float
    direction: str
    lane_width_m: float
    xm_per_pix: float
    xm_estimated: bool
    departure_warning: bool
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.direction == "ortada":
            return f"Araç şerit merkezinde (sapma {self.offset_m:+.3f} m)"
        side = "sağında" if self.offset_m > 0 else "solunda"
        return f"Araç şerit merkezinin {abs(self.offset_m):.3f} m {side} ({self.direction} kaymış)"


def to_coefficients(values, name: str) -> np.ndarray:
    """Polinom katsayılarını 1B float diziye çevirir (np.polyfit biçimi)."""
    try:
        coefficients = np.asarray(values, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: katsayılar sayı olmalı (np.polyfit biçimi, ör. A B C).") from exc
    if coefficients.size == 0 or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{name}: geçerli polinom katsayıları gerekli (np.polyfit biçimi, ör. A B C).")
    return coefficients


def to_image_size(values) -> tuple[int, int]:
    size = np.asarray(values, dtype=np.float64).ravel()
    if size.size != 2 or not np.all(np.isfinite(size)) or np.any(size < 1):
        raise ValueError("Görüntü boyutu iki pozitif sayı olmalı (genişlik yükseklik).")
    return int(size[0]), int(size[1])


class LaneOffsetCalculator:
    """Ölçek ve ayarları bir kez alır; her karede compute() ile yanal sapmayı hesaplar."""

    def __init__(
        self,
        image_size,
        xm_per_pix: float | None = XM_PER_PIX,
        lane_width_m: float = LANE_WIDTH_M,
        camera_offset_m: float = CAMERA_OFFSET_M,
        center_tolerance_m: float = CENTER_TOLERANCE_M,
        departure_warning_m: float = DEPARTURE_WARNING_M,
    ):
        self.image_size = to_image_size(image_size)
        if xm_per_pix is not None and not (np.isfinite(xm_per_pix) and xm_per_pix > 0):
            raise ValueError("xm_per_pix pozitif bir sayı olmalı (metre/piksel).")
        if lane_width_m <= 0:
            raise ValueError("Standart şerit genişliği pozitif olmalı (--lane-width).")
        if center_tolerance_m < 0:
            raise ValueError("Ortada toleransı negatif olamaz (--center-tolerance).")
        if departure_warning_m <= 0:
            raise ValueError("Çıkma uyarısı eşiği pozitif olmalı (--warn-offset).")
        self.xm_per_pix = xm_per_pix
        self.lane_width_m = lane_width_m
        self.camera_offset_m = camera_offset_m
        self.center_tolerance_m = center_tolerance_m
        self.departure_warning_m = departure_warning_m

    def compute(self, left_fit, right_fit) -> LaneOffset:
        width, height = self.image_size
        left = to_coefficients(left_fit, "Sol polinom")
        right = to_coefficients(right_fit, "Sağ polinom")

        # [EK] Alt satır için görüntü yüksekliği gerekir: y = yükseklik - 1 (aracın hemen önü).
        y_bottom = height - 1
        left_x = float(np.polyval(left, y_bottom))
        right_x = float(np.polyval(right, y_bottom))
        # [EK] Tutarlılık: sağ şerit solda kalıyorsa polinomlar yer değiştirmiştir.
        if right_x <= left_x:
            raise ValueError(
                f"Alt satırda sağ şerit (x={right_x:.1f}) sol şeridin (x={left_x:.1f}) solunda kalıyor; "
                "sol ve sağ polinom yer değiştirmiş olabilir."
            )

        lane_center = (left_x + right_x) / 2
        image_center = (width - 1) / 2  # piksel indekslerinde görüntünün tam ortası
        warnings = []

        xm_per_pix, estimated = self.xm_per_pix, False
        if xm_per_pix is None:
            # [EK] xm_per_pix tahmini: alt satırdaki şerit genişliği = standart şerit genişliği.
            xm_per_pix, estimated = self.lane_width_m / (right_x - left_x), True
            warnings.append(
                f"xm_per_pix verilmedi; şerit genişliği {self.lane_width_m} m varsayılarak "
                f"{xm_per_pix:.6f} m/piksel tahmin edildi. Ölçülmüş değeri --xm-per-pix ya da --scale ile ver."
            )
        lane_width_m = (right_x - left_x) * xm_per_pix

        # [EK] Tutarlılık: makul olmayan şerit genişliği ya da görüntü dışı şerit.
        low, high = LANE_WIDTH_RANGE_M
        if not estimated and not low <= lane_width_m <= high:
            warnings.append(
                f"Ölçülen şerit genişliği {lane_width_m:.2f} m, makul aralığın ({low}-{high} m) dışında; "
                "polinomları ya da xm_per_pix değerini kontrol et."
            )
        for name, x in (("Sol", left_x), ("Sağ", right_x)):
            if not 0 <= x <= width - 1:
                warnings.append(f"{name} şerit alt satırda görüntünün dışında (x={x:.1f}); sonuç güvenilmez olabilir.")

        # Kamera görüntünün ortasında kabul edilir. [EK] Kamera montaj kayması araç merkezine göre düzeltilir.
        offset_m = (image_center - lane_center) * xm_per_pix - self.camera_offset_m

        # [EK] "Ortada" toleransı.
        if abs(offset_m) <= self.center_tolerance_m:
            direction = "ortada"
        elif offset_m > 0:
            direction = "sağa"
        else:
            direction = "sola"

        return LaneOffset(
            left_x_px=left_x,
            right_x_px=right_x,
            lane_center_px=lane_center,
            image_center_px=image_center,
            offset_m=offset_m,
            direction=direction,
            lane_width_m=lane_width_m,
            xm_per_pix=xm_per_pix,
            xm_estimated=estimated,
            # [EK] Şeritten çıkma uyarısı.
            departure_warning=abs(offset_m) >= self.departure_warning_m,
            warnings=warnings,
        )


def load_npz(path: Path, description: str, fix_hint: str) -> dict[str, np.ndarray]:
    """.npz dosyasını okur; bozuk ya da .npz olmayan dosyada anlaşılır hata verir."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} bulunamadı: {path}\n{fix_hint}")
    try:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    # Bozuk zip, düz metin ya da .npz uzantılı tek dizi (.npy) dosyası buraya düşer.
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{description} okunamadı (bozuk ya da .npz biçiminde değil): {path}\n{fix_hint}") from exc


def load_fits(path: Path, left_key: str, right_key: str, size_key: str, frame_key: str):
    """Polinom dosyasını [(kare adı, sol, sağ), ...] listesine çevirir; görüntü boyutunu da döndürür.

    Tek polinom (3,) ya da kare başına bir satır (N, 3; 2.4'ün çıktısı) kabul edilir.
    """
    data = load_npz(path, "Polinom dosyası",
                    "Polinom dosyasını yeniden oluştur ya da --input ile doğru dosyayı ver.")
    missing = [key for key in (left_key, right_key) if key not in data]
    if missing:
        raise ValueError(
            f"Polinom dosyasında {', '.join(missing)} yok: {path}\n"
            f"Dosyadaki anahtarlar: {', '.join(data) or '(boş)'}. "
            "Farklı ad kullanılıyorsa --fit-keys SOL SAG ile belirt."
        )
    # [EK] Kare kare girdi: 1B dizi tek polinomdur, 2B dizide her satır bir karedir.
    left, right = np.atleast_2d(data[left_key]), np.atleast_2d(data[right_key])
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"Polinom dosyasında sol ({left.shape[0]}) ve sağ ({right.shape[0]}) satır sayısı farklı: {path}\n"
            "Her kare için bir sol ve bir sağ polinom olmalı."
        )
    names = np.atleast_1d(data[frame_key]) if frame_key in data else []
    if len(names) == left.shape[0]:
        names = [str(name) for name in names]
    else:
        names = [f"kare{i:05d}" for i in range(left.shape[0])] if left.shape[0] > 1 else [""]
    return list(zip(names, left, right)), data.get(size_key)


def load_xm_per_pix(path: Path, key: str) -> float:
    data = load_npz(path, "Ölçek dosyası",
                    "--scale ile doğru dosyayı ver ya da değeri --xm-per-pix ile doğrudan gir.")
    if key not in data:
        raise ValueError(
            f"Ölçek dosyasında '{key}' yok: {path}\n"
            f"Dosyadaki anahtarlar: {', '.join(data) or '(boş)'}. Farklı ad kullanılıyorsa --xm-key ile belirt."
        )
    return float(np.asarray(data[key]).ravel()[0])


# [EK] Kontrol görüntüsü: polinomlar, şerit merkezi, araç konumu ve sapma.
def draw_offset(result: LaneOffset, left_fit, right_fit, image_size, camera_offset_m: float) -> np.ndarray:
    width, height = image_size
    canvas = np.zeros((height, width, 3), np.uint8)
    scale = max(width / 1000, 0.6)
    thickness = max(round(2 * scale), 1)

    ys = np.arange(height, dtype=np.float64)
    left_xs, right_xs = np.polyval(left_fit, ys), np.polyval(right_fit, ys)
    for xs, color in ((left_xs, LEFT_COLOR), (right_xs, RIGHT_COLOR), ((left_xs + right_xs) / 2, CENTER_COLOR)):
        points = np.column_stack([xs, ys]).round().astype(np.int32)
        cv2.polylines(canvas, [points], False, color, thickness=thickness, lineType=cv2.LINE_AA)

    bottom = height - 1
    marker = round(80 * scale)
    vehicle_x = round(result.image_center_px - camera_offset_m / result.xm_per_pix)
    lane_x = round(result.lane_center_px)
    cv2.line(canvas, (vehicle_x, bottom), (vehicle_x, bottom - marker), VEHICLE_COLOR,
             thickness=thickness + 1, lineType=cv2.LINE_AA)
    if lane_x != vehicle_x:
        cv2.arrowedLine(canvas, (lane_x, bottom - marker // 2), (vehicle_x, bottom - marker // 2), ARROW_COLOR,
                        thickness=thickness, line_type=cv2.LINE_AA, tipLength=0.2)

    lines = [
        (f"Sapma: {abs(result.offset_m):.2f} m {DIRECTION_ASCII[result.direction]}", ARROW_COLOR),
        (f"Serit genisligi: {result.lane_width_m:.2f} m" + (" (tahmini olcek)" if result.xm_estimated else ""),
         (255, 255, 255)),
        ("Kirmizi: sol  Mavi: sag  Yesil: serit merkezi  Beyaz: arac", (255, 255, 255)),
    ]
    if result.departure_warning:
        lines.append(("UYARI: seritten cikma riski", WARNING_COLOR))
    for row, (text, color) in enumerate(lines):
        origin = (round(15 * scale), round((35 + row * 32) * scale))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, color,
                    thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


def result_row(source: str, frame: str, result: LaneOffset) -> list:
    return [
        source, frame, f"{result.left_x_px:.2f}", f"{result.right_x_px:.2f}", f"{result.lane_center_px:.2f}",
        f"{result.image_center_px:.2f}", f"{result.offset_m:.4f}", result.direction,
        f"{result.lane_width_m:.3f}", f"{result.xm_per_pix:.6f}",
        "evet" if result.xm_estimated else "hayır", "evet" if result.departure_warning else "hayır",
    ]


def empty_row(source: str, frame: str, status: str) -> list:
    """Hesaplanamayan kare: yon sütununda durum ("yok" ya da "hata"), sayısal alanlar boş."""
    return [source, frame, "", "", "", "", "", status, "", "", "", ""]


def print_summary(source: str, frame_count: int, results: list[LaneOffset], departure_warning_m: float) -> None:
    """Çok kareli dosyanın özeti; kare kare değerler tabloda."""
    if not results:
        print(f"  {source}: {frame_count} kare, hiçbirinde iki şerit birden yok; sapma hesaplanamadı.")
        return
    offsets = np.array([r.offset_m for r in results])
    counts = {d: sum(r.direction == d for r in results) for d in ("sağa", "sola", "ortada")}
    print(f"  {source}: {frame_count} kare, {len(results)} karede hesaplandı "
          f"({frame_count - len(results)} karede hesaplanamadı)")
    print(f"      sapma ort. {offsets.mean():+.3f} m (en küçük {offsets.min():+.3f}, en büyük {offsets.max():+.3f}) | "
          f"sağa {counts['sağa']}, sola {counts['sola']}, ortada {counts['ortada']} kare | "
          f"çıkma uyarısı {sum(r.departure_warning for r in results)} kare (eşik {departure_warning_m} m)")
    warnings = sorted({w for r in results for w in r.warnings})
    for warning in warnings[:3]:
        print(f"      UYARI: {warning}")
    if len(warnings) > 3:
        print(f"      (+{len(warnings) - 3} benzer uyarı; kare kare değerler tabloda)")


def visualization_path(output_dir: Path, source: str, frame: str, multi: bool) -> Path:
    stem = Path(source).stem
    if not multi:
        return output_dir / f"{stem}_offset.png"
    folder = output_dir / f"{stem}_offset"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{frame.replace('/', '_').replace(chr(92), '_')}_offset.png"


def print_result(source: str, result: LaneOffset, departure_warning_m: float) -> None:
    print(f"  {source}: {result.describe()}")
    print(f"      sol x={result.left_x_px:.1f}  sağ x={result.right_x_px:.1f}  "
          f"şerit merkezi x={result.lane_center_px:.1f}  görüntü merkezi x={result.image_center_px:.1f}  "
          f"şerit genişliği={result.lane_width_m:.2f} m  xm_per_pix={result.xm_per_pix:.6f}")
    if result.departure_warning:
        print(f"      ŞERİTTEN ÇIKMA UYARISI: sapma {abs(result.offset_m):.2f} m ≥ {departure_warning_m} m")
    for warning in result.warnings:
        print(f"      UYARI: {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Şerit polinomlarından şerit merkezini ve aracın yanal sapmasını (metre, yön) hesaplar."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH,
                        help=f"Polinom .npz dosyası ya da klasörü (varsayılan: {INPUT_PATH})")
    parser.add_argument("--left-fit", type=float, nargs="+", metavar="KATSAYI",
                        help="Sol polinom katsayıları, en yüksek dereceden (ör. A B C); --input yerine kullanılır")
    parser.add_argument("--right-fit", type=float, nargs="+", metavar="KATSAYI",
                        help="Sağ polinom katsayıları, en yüksek dereceden (ör. A B C)")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("GENISLIK", "YUKSEKLIK"), default=IMAGE_SIZE,
                        help="Görüntü boyutu (varsayılan: polinom dosyasındaki image_size)")
    parser.add_argument("--xm-per-pix", type=float, default=XM_PER_PIX,
                        help="Yatay metre/piksel (varsayılan: --scale ya da şerit genişliğinden tahmin)")
    parser.add_argument("--scale", type=Path, default=SCALE_PATH,
                        help="xm_per_pix içeren .npz dosyası")
    parser.add_argument("--fit-keys", nargs=2, metavar=("SOL", "SAG"), default=(LEFT_FIT_KEY, RIGHT_FIT_KEY),
                        help=f"Polinom dosyasındaki anahtarlar (varsayılan: {LEFT_FIT_KEY} {RIGHT_FIT_KEY})")
    parser.add_argument("--size-key", default=IMAGE_SIZE_KEY,
                        help=f"Polinom dosyasındaki görüntü boyutu anahtarı (varsayılan: {IMAGE_SIZE_KEY})")
    parser.add_argument("--frame-key", default=FRAME_KEY,
                        help=f"Polinom dosyasındaki kare adları anahtarı (varsayılan: {FRAME_KEY})")
    parser.add_argument("--xm-key", default=XM_PER_PIX_KEY,
                        help=f"Ölçek dosyasındaki xm_per_pix anahtarı (varsayılan: {XM_PER_PIX_KEY})")
    parser.add_argument("--lane-width", type=float, default=LANE_WIDTH_M,
                        help=f"Ölçek tahmini için standart şerit genişliği, m (varsayılan: {LANE_WIDTH_M})")
    parser.add_argument("--camera-offset", type=float, default=CAMERA_OFFSET_M,
                        help=f"Kameranın araç ortasına göre konumu, m, sağ + (varsayılan: {CAMERA_OFFSET_M})")
    parser.add_argument("--center-tolerance", type=float, default=CENTER_TOLERANCE_M,
                        help=f"Bundan küçük sapma 'ortada' sayılır, m (varsayılan: {CENTER_TOLERANCE_M})")
    parser.add_argument("--warn-offset", type=float, default=DEPARTURE_WARNING_M,
                        help=f"Şeritten çıkma uyarısı eşiği, m (varsayılan: {DEPARTURE_WARNING_M})")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    parser.add_argument("--visualize", action="store_true", default=VISUALIZE,
                        help="Polinomları, şerit merkezini ve sapmayı gösteren kontrol görüntüsü kaydet")
    args = parser.parse_args()
    if (args.left_fit is None) != (args.right_fit is None):
        parser.error("--left-fit ve --right-fit birlikte verilmeli.")
    if args.left_fit is not None and args.image_size is None:
        parser.error("Polinomlar komut satırından verildiğinde --image-size GENISLIK YUKSEKLIK de gerekli.")
    return args


def collect_inputs(args: argparse.Namespace) -> list[tuple[str, Path | None]]:
    """(kaynak adı, dosya yolu) listesi; komut satırı polinomları için yol None'dır."""
    if args.left_fit is not None:
        return [("komut_satiri", None)]
    if not args.input.exists():
        raise FileNotFoundError(
            f"Girdi bulunamadı: {args.input}\n"
            "Şerit polinomlarının .npz dosyasını ya da klasörünü bu yola koy, --input ile başka bir yol ver "
            "ya da polinomları --left-fit / --right-fit ile gir."
        )
    if args.input.is_dir():
        paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() == ".npz")
        if not paths:
            raise FileNotFoundError(f"Klasörde .npz dosyası yok: {args.input}")
        return [(p.name, p) for p in paths]
    if args.input.suffix.lower() != ".npz":
        raise ValueError(f"Desteklenmeyen dosya türü: {args.input.name}\nPolinomlar .npz dosyasında olmalı.")
    return [(args.input.name, args.input)]


def main() -> int:
    args = parse_args()
    try:
        inputs = collect_inputs(args)
        xm_per_pix = args.xm_per_pix
        if xm_per_pix is None and args.scale is not None:
            xm_per_pix = load_xm_per_pix(args.scale, args.xm_key)
        args.output.mkdir(parents=True, exist_ok=True)

        rows = []
        computed = total = 0
        print("Yanal sapma:")
        for source, path in inputs:
            try:
                if path is None:
                    frames, size = [("", args.left_fit, args.right_fit)], None
                else:
                    frames, size = load_fits(path, *args.fit_keys, args.size_key, args.frame_key)
                image_size = args.image_size if args.image_size is not None else size
                if image_size is None:
                    raise ValueError(
                        f"Görüntü boyutu yok: {source}\n"
                        f"Dosyada '{args.size_key}' anahtarı bulunmuyor; --image-size GENISLIK YUKSEKLIK ile ver."
                    )
                calculator = LaneOffsetCalculator(
                    image_size, xm_per_pix, lane_width_m=args.lane_width, camera_offset_m=args.camera_offset,
                    center_tolerance_m=args.center_tolerance, departure_warning_m=args.warn_offset,
                )
            except ValueError as exc:
                if len(inputs) == 1:
                    raise
                print(f"  UYARI: {source} atlandı. {exc}", file=sys.stderr)
                continue

            multi = len(frames) > 1
            results = []
            for frame, left_fit, right_fit in frames:
                total += 1
                label = f"{source} {frame}".strip()
                # [EK] Şeridin bulunamadığı kare (2.4'te NaN satırı) hesaplanmaz; tabloya "yok" yazılır.
                if not (np.all(np.isfinite(left_fit)) and np.all(np.isfinite(right_fit))):
                    rows.append(empty_row(source, frame, "yok"))
                    if not multi:
                        print(f"  {label}: şerit bulunamadı (polinomda NaN); sapma hesaplanmadı.")
                    continue
                try:
                    result = calculator.compute(left_fit, right_fit)
                except ValueError as exc:
                    if not multi and len(inputs) == 1:
                        raise
                    print(f"  UYARI: {label} hesaplanamadı. {exc}", file=sys.stderr)
                    rows.append(empty_row(source, frame, "hata"))
                    continue

                computed += 1
                results.append(result)
                rows.append(result_row(source, frame, result))
                if not multi:
                    print_result(label, result, args.warn_offset)
                if args.visualize:
                    image_path = visualization_path(args.output, source, frame, multi)
                    image = draw_offset(result, to_coefficients(left_fit, "Sol polinom"),
                                        to_coefficients(right_fit, "Sağ polinom"), calculator.image_size,
                                        args.camera_offset)
                    if not cv2.imwrite(str(image_path), image):
                        raise OSError(
                            f"Görüntü yazılamadı: {image_path}\n"
                            "Çıktı klasörünün yazma iznini kontrol et ya da --output ile başka bir klasör ver."
                        )
            if multi:
                print_summary(source, len(frames), results, args.warn_offset)

        csv_path = args.output / "lane_offset.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)
    except PermissionError as exc:
        print(
            f"HATA: Dosyaya erişim izni yok: {exc.filename}\n"
            "Klasörün izinlerini kontrol et ya da --input / --scale / --output ile başka bir yol ver.",
            file=sys.stderr,
        )
        return 1
    except cv2.error as exc:
        print(f"HATA: OpenCV kontrol görüntüsünü çizemedi: {getattr(exc, 'err', exc)}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print(f"{computed} / {total} karede sapma hesaplandı ({len(inputs)} girdi); tablo: {csv_path}")
    return 0 if computed else 1


if __name__ == "__main__":
    sys.exit(main())
