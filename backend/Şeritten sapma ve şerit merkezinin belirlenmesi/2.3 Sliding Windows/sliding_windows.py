"""
2.3 Sliding Windows — kuş bakışı maskede sol ve sağ şerit piksellerinin bulunması.

Ne işe yarar:
    Maskedeki beyaz piksellerden hangisinin sol şeride, hangisinin sağ şeride ait olduğunu ayırır;
    gürültüyü dışarıda bırakır. Virajlı şeritte de çalışır.

Nasıl çalışır:
    Görüntünün alt yarısının sütun histogramındaki iki tepe şeritlerin başladığı yerdir. Oradan yukarı
    doğru küçük pencerelerle şerit izlenir; her pencere içindeki piksellerin ortasına kaydırılır.
    Pencerelere düşen pikseller o şeride yazılır.

Bağlantılar:
    Alır : kuş bakışı maske, yani 2.1 (ya da 4.4) maskesinin 1.2 ile çevrilmiş hâli.
    Verir: <ad>_lane_pixels.npz (left_x, left_y, right_x, right_y) -> 2.4 polinom uydurma ve 3.2.
    Ana hat: main.py her karede SlidingWindowSearch.find() çağırır.

Parametreler:
    --input       Başka gereksinimden: kuş bakışı maskeler (varsayılan outputs/1.2 .../kus_bakisi_maskeler;
                  1.2 warp_perspective.py ile 2.1'in maskelerinden üretilir).
    --windows, --margin, --min-pixels, --min-peak   Ayar: pencere sayısı ve boyutu (varsayılanlar uygun).
"""

import argparse
import sys
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

# [DEĞİŞTİR] Girdi: kuş bakışı ikili maske (tek görüntü), maske klasörü veya maske videosu.
# Maske tek kanallı olmalı: 0 = arka plan, 0'dan büyük = şerit adayı (0/1 ve 0/255 ikisi de olur).
INPUT_PATH = PROJECT_ROOT / "outputs" / "1.2 Perspektif dönüşümü" / "kus_bakisi_maskeler"

# Koordinat dosyalarının ve kontrol görüntülerinin yazılacağı klasör.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / Path(__file__).resolve().parent.name

# Pencere sayısı (gereksinim: yaklaşık 9-10).
N_WINDOWS = 9

# Pencerenin yarı genişliği, görüntü genişliğine oran olarak (0.08 -> 1280 px'de ±102 px).
# Şerit çizgisi pencereden taşıyorsa büyüt, komşu gürültü pencereye giriyorsa küçült.
MARGIN_RATIO = 0.08

# Merkezin kaydırılması için pencerede olması gereken en az piksel, pencere alanına oran olarak
# (0.02 -> 1280x720 ve 9 pencerede ~326 piksel). Çok düşük değerde kesikli şeridin boşluğundaki
# pencere gürültüye kayar; çok yüksek değerde ince/soluk çizgiler pencereyi kaydıramaz. 0.02,
# sentetik testlerde (kalın, ince, seyrek ve gürültülü çizgiler) en dengeli sonucu verdi.
MIN_PIXELS_RATIO = 0.02

# Bir yarıdaki histogram tepesi alt yarı yüksekliğinin bu oranından düşükse o şerit bulunamadı
# sayılır (o yarıda yalnızca gürültü var demektir). 0.07 -> 720 px'de ~25 piksel. Yükseltmek
# sahte şeridi azaltır ama seyrek/soluk kesikli şeritleri kaçırmaya başlar.
MIN_PEAK_RATIO = 0.07

# Yetersiz pikselli pencerede son kaymayla devam et (kesikli şeritler için).
USE_MOMENTUM = True

# 0/255 maskelerde şerit sayılacak en düşük değer; kayıplı biçimlerdeki ara değerleri eler.
MASK_THRESHOLD = 127

# True: girdi maskesi ve kontrol görüntüsü yan yana kaydedilir.
SAVE_COMPARISON = False
# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Kontrol görüntüsü renkleri (BGR).
LEFT_COLOR = (0, 0, 255)
RIGHT_COLOR = (255, 0, 0)
RECENTERED_COLOR = (0, 255, 0)
NOT_RECENTERED_COLOR = (0, 165, 255)
HISTOGRAM_COLOR = (0, 255, 255)

# [EK] Bu oranın üzerinde ara değer (0 ve 255'ten uzak) içeren girdi ikili maske sayılmaz.
NON_BINARY_WARN_RATIO = 0.02


@dataclass
class Window:
    side: str
    x_low: int
    x_high: int
    y_low: int
    y_high: int
    pixel_count: int
    recentered: bool


@dataclass
class LanePixels:
    """Sliding windows sonucu. Koordinatlar görüntü koordinatlarıdır (x sütun, y satır, y aşağı artar)."""

    leftx: np.ndarray
    lefty: np.ndarray
    rightx: np.ndarray
    righty: np.ndarray
    left_base: int | None
    right_base: int | None
    histogram: np.ndarray
    binary: np.ndarray
    windows: list[Window] = field(default_factory=list)

    @property
    def image_size(self) -> tuple[int, int]:
        height, width = self.binary.shape
        return width, height

    def noise_count(self) -> int:
        """Hiçbir pencereye girmeyen (gürültü sayılan) maske pikseli sayısı."""
        assigned = np.zeros(self.binary.shape, bool)
        assigned[self.lefty, self.leftx] = True
        assigned[self.righty, self.rightx] = True
        return int(np.count_nonzero(self.binary.astype(bool) & ~assigned))


# [EK] İkili maske dönüşümü: 0/1 ve 0/255 maskeleri aynı şekilde ele alır.
def to_binary(mask: np.ndarray, threshold: int = MASK_THRESHOLD) -> np.ndarray:
    """Maskeyi 0/1 değerli uint8 diziye çevirir."""
    gray = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if gray.max() <= 1:
        return (gray > 0).astype(np.uint8)
    return (gray > threshold).astype(np.uint8)


# [EK] Girdinin gerçekten ikili maske olup olmadığını kontrol eder.
def looks_binary(mask: np.ndarray) -> bool:
    gray = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if gray.max() <= 1:
        return True
    intermediate = np.count_nonzero((gray > 30) & (gray < 225))
    return intermediate / gray.size <= NON_BINARY_WARN_RATIO


class SlidingWindowSearch:
    """Parametreleri bir kez alır; her karede find() ile şerit piksellerini bulur."""

    def __init__(
        self,
        n_windows: int = N_WINDOWS,
        margin_ratio: float = MARGIN_RATIO,
        min_pixels_ratio: float = MIN_PIXELS_RATIO,
        min_peak_ratio: float = MIN_PEAK_RATIO,
        use_momentum: bool = USE_MOMENTUM,
        mask_threshold: int = MASK_THRESHOLD,
    ):
        if n_windows < 1:
            raise ValueError("Pencere sayısı en az 1 olmalı (--windows).")
        if not 0 < margin_ratio <= 0.5:
            raise ValueError("Pencere yarı genişliği oranı 0 ile 0.5 arasında olmalı (--margin).")
        if not 0 <= min_pixels_ratio <= 1:
            raise ValueError("En az piksel oranı 0 ile 1 arasında olmalı (--min-pixels).")
        if not 0 <= min_peak_ratio <= 1:
            raise ValueError("Histogram tepe oranı 0 ile 1 arasında olmalı (--min-peak).")
        self.n_windows = n_windows
        self.margin_ratio = margin_ratio
        self.min_pixels_ratio = min_pixels_ratio
        self.min_peak_ratio = min_peak_ratio
        self.use_momentum = use_momentum
        self.mask_threshold = mask_threshold

    def window_geometry(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Verilen boyut için pencere yarı genişliği ve en az piksel sayısı (piksel cinsinden)."""
        # [EK] Oransal pencere boyutu: her çözünürlükte aynı davranış.
        width, height = image_size
        margin = max(1, round(self.margin_ratio * width))
        window_area = 2 * margin * height / self.n_windows
        min_pixels = max(1, round(self.min_pixels_ratio * window_area))
        return margin, min_pixels

    def find(self, mask: np.ndarray) -> LanePixels:
        binary = to_binary(mask, self.mask_threshold)
        height, width = binary.shape
        if height < self.n_windows:
            raise ValueError(f"Görüntü yüksekliği ({height} px) pencere sayısından ({self.n_windows}) küçük.")

        # Alt yarının histogramı: her sütundaki şerit pikseli sayısı.
        histogram = np.sum(binary[height // 2:, :], axis=0)
        midpoint = width // 2
        left_half, right_half = histogram[:midpoint], histogram[midpoint:]
        # [EK] Zayıf tepe kontrolü: tepe yalnızca gürültü kadarsa (ya da yarı boşsa) şerit bulunamadı sayılır.
        min_peak = max(self.min_peak_ratio * (height - height // 2), 1)
        left_base = int(np.argmax(left_half)) if left_half.max() >= min_peak else None
        right_base = int(np.argmax(right_half)) + midpoint if right_half.max() >= min_peak else None

        nonzeroy, nonzerox = binary.nonzero()
        margin, min_pixels = self.window_geometry((width, height))
        # Pencere sınırları alttan yukarı; linspace tüm yüksekliği kapsar.
        edges = np.linspace(height, 0, self.n_windows + 1).round().astype(int)

        windows: list[Window] = []
        lane_indices = {}
        for side, base in (("left", left_base), ("right", right_base)):
            if base is None:
                lane_indices[side] = np.array([], dtype=np.int64)
                continue
            indices, side_windows = self._track(
                nonzerox, nonzeroy, base, edges, margin, min_pixels, width, side
            )
            lane_indices[side] = indices
            windows.extend(side_windows)

        left, right = lane_indices["left"], lane_indices["right"]
        return LanePixels(
            leftx=nonzerox[left].astype(np.int32),
            lefty=nonzeroy[left].astype(np.int32),
            rightx=nonzerox[right].astype(np.int32),
            righty=nonzeroy[right].astype(np.int32),
            left_base=left_base,
            right_base=right_base,
            histogram=histogram,
            binary=binary,
            windows=windows,
        )

    def _track(self, nonzerox, nonzeroy, base, edges, margin, min_pixels, width, side):
        """Tek bir şeridi alttan yukarı pencerelerle takip eder."""
        current = base
        last_shift = 0
        collected, windows = [], []
        for i in range(self.n_windows):
            y_high, y_low = int(edges[i]), int(edges[i + 1])
            x_low, x_high = current - margin, current + margin
            inside = (
                (nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= x_low) & (nonzerox < x_high)
            ).nonzero()[0]
            collected.append(inside)

            recentered = len(inside) >= min_pixels
            if recentered:
                new_center = int(np.mean(nonzerox[inside]))
                last_shift = new_center - current
                current = new_center
            elif self.use_momentum:
                # [EK] Momentum: yeterli piksel yoksa son kaymayla devam et.
                current += last_shift
            current = int(np.clip(current, 0, width - 1))

            windows.append(Window(side, x_low, x_high, y_low, y_high, len(inside), recentered))
        return np.concatenate(collected), windows


def draw_windows(result: LanePixels) -> np.ndarray:
    """Pencereleri, şerit piksellerini ve histogramı maske üzerine çizer (cv2.rectangle)."""
    binary = result.binary
    height, width = binary.shape
    output = np.dstack([binary * 255] * 3).astype(np.uint8)  # beyaz: gürültü sayılan pikseller
    output[result.lefty, result.leftx] = LEFT_COLOR
    output[result.righty, result.rightx] = RIGHT_COLOR

    scale = max(width / 1000, 0.6)
    thickness = max(round(2 * scale), 1)

    # [EK] Alt yarının histogramı: eğrinin yüksekliği o sütundaki piksel sayısına eşittir.
    if result.histogram.max() > 0:
        points = np.column_stack([np.arange(width), height - 1 - result.histogram]).astype(np.int32)
        cv2.polylines(output, [points], False, HISTOGRAM_COLOR, thickness=thickness, lineType=cv2.LINE_AA)

    for window in result.windows:
        color = RECENTERED_COLOR if window.recentered else NOT_RECENTERED_COLOR
        cv2.rectangle(output, (window.x_low, window.y_low), (window.x_high - 1, window.y_high - 1),
                      color, thickness=thickness)

    lines = [
        (f"Sol: {len(result.leftx)} px  Sag: {len(result.rightx)} px  Gurultu: {result.noise_count()} px",
         (255, 255, 255)),
        ("Yesil: merkez kaydirildi  Turuncu: yetersiz piksel", (255, 255, 255)),
        ("Sari: alt yari histogrami", HISTOGRAM_COLOR),
    ]
    for row, (text, color) in enumerate(lines):
        origin = (round(15 * scale), round((35 + row * 32) * scale))
        cv2.putText(output, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (0, 0, 0),
                    thickness=thickness + 3, lineType=cv2.LINE_AA)
        cv2.putText(output, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, color,
                    thickness=thickness, lineType=cv2.LINE_AA)
    return output


# [EK] --compare için yan yana karşılaştırma görüntüsü.
def side_by_side(mask: np.ndarray, debug: np.ndarray) -> np.ndarray:
    left = mask if mask.ndim == 3 else cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if left.max() <= 1:
        left = left * 255
    return np.hstack([left.astype(np.uint8), debug])


def summarize(result: LanePixels, search: SlidingWindowSearch) -> str:
    recentered = {"left": 0, "right": 0}
    for window in result.windows:
        recentered[window.side] += window.recentered
    _, min_pixels = search.window_geometry(result.image_size)
    parts = []
    for side, name, count in (("left", "sol", len(result.leftx)), ("right", "sağ", len(result.rightx))):
        base = result.left_base if side == "left" else result.right_base
        if base is None:
            parts.append(f"{name}: BULUNAMADI")
        else:
            parts.append(f"{name}: {count} px (başlangıç x={base}, kaydırılan pencere "
                         f"{recentered[side]}/{search.n_windows})")
        if base is not None and count < min_pixels:
            parts[-1] += " [çok az piksel]"
    parts.append(f"gürültü: {result.noise_count()} px")
    return " | ".join(parts)


def save_lane_pixels(output_path: Path, result: LanePixels) -> None:
    # Anahtar adları 2.4 ve 3.1–3.3'ün piksel listesi girdisiyle aynıdır.
    np.savez(
        output_path,
        left_x=result.leftx,
        left_y=result.lefty,
        right_x=result.rightx,
        right_y=result.righty,
        image_size=np.array(result.image_size),
    )


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(
            f"Görüntü yazılamadı: {path}\n"
            "Çıktı klasörünün yazma iznini kontrol et ya da --output ile başka bir klasör ver."
        )


def process_images(search: SlidingWindowSearch, image_paths: list[Path], output_dir: Path,
                   compare: bool) -> int:
    """Maskelerde şerit piksellerini bulur, koordinatları ve kontrol görüntüsünü kaydeder."""
    written = 0
    for path in image_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"UYARI: {path.name} okunamadı (bozuk ya da desteklenmeyen biçim), atlandı.",
                  file=sys.stderr)
            continue
        if not looks_binary(mask):
            print(f"UYARI: {path.name} ikili maskeye benzemiyor; girdi kuş bakışı ikili maske "
                  "(0 = arka plan, 255 = şerit) olmalı.", file=sys.stderr)

        result = search.find(mask)
        debug = draw_windows(result)
        save_lane_pixels(output_dir / f"{path.stem}_lane_pixels.npz", result)
        write_image(output_dir / f"{path.stem}_windows.png", side_by_side(mask, debug) if compare else debug)
        print(f"  {path.name}: {summarize(result, search)}")
        written += 1
    return written


# [EK] Video girdisi: VideoCapture ile kare kare okunur, VideoWriter ile yazılır.
def process_video(search: SlidingWindowSearch, video_path: Path, output_dir: Path,
                  compare: bool) -> tuple[Path, int, int, int]:
    """Maske videosunun her karesinde şerit piksellerini bulur, kontrol videosunu yazar."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(
            f"Video açılamadı: {video_path}\n"
            "Dosya bozuk ya da desteklenmeyen bir biçimde olabilir; başka bir video dene "
            "ya da videoyu mp4 (H.264) olarak dışa aktar."
        )

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output_path = output_dir / f"{video_path.stem}_windows.mp4"
    writer = None
    frame_count = missing_left = missing_right = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = search.find(mask)
            missing_left += result.left_base is None
            missing_right += result.right_base is None
            debug = draw_windows(result)
            output = side_by_side(mask, debug) if compare else debug

            if writer is None:
                height, width = output.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise OSError(
                        f"Çıktı videosu oluşturulamadı: {output_path}\n"
                        "Çıktı klasörünün yazma iznini kontrol et ya da --output ile başka bir klasör ver."
                    )
            writer.write(output)
            frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise ValueError(
            f"Videodan hiç kare okunamadı: {video_path}\n"
            "Video boş ya da bozuk olabilir; başka bir video dene."
        )
    return output_path, frame_count, missing_left, missing_right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kuş bakışı ikili maskede sol ve sağ şerit piksellerini kayan pencerelerle bulur."
    )
    parser.add_argument(
        "--input", type=Path, default=INPUT_PATH,
        help=f"Tek maske, maske klasörü veya maske videosu (varsayılan: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--windows", type=int, default=N_WINDOWS,
        help=f"Pencere sayısı (varsayılan: {N_WINDOWS})",
    )
    parser.add_argument(
        "--margin", type=float, default=MARGIN_RATIO,
        help=f"Pencere yarı genişliği, görüntü genişliğine oran (varsayılan: {MARGIN_RATIO})",
    )
    parser.add_argument(
        "--min-pixels", type=float, default=MIN_PIXELS_RATIO,
        help=f"Merkez kaydırma için en az piksel, pencere alanına oran (varsayılan: {MIN_PIXELS_RATIO})",
    )
    parser.add_argument(
        "--min-peak", type=float, default=MIN_PEAK_RATIO,
        help=f"Şerit sayılacak en düşük histogram tepesi, alt yarı yüksekliğine oran (varsayılan: {MIN_PEAK_RATIO})",
    )
    parser.add_argument(
        "--no-momentum", dest="momentum", action="store_false", default=USE_MOMENTUM,
        help="Yetersiz pikselli pencerede son kaymayla devam etmeyi kapat",
    )
    parser.add_argument(
        "--threshold", type=int, default=MASK_THRESHOLD,
        help=f"0/255 maskelerde şerit sayılacak en düşük değer (varsayılan: {MASK_THRESHOLD})",
    )
    parser.add_argument(
        "--compare", action="store_true", default=SAVE_COMPARISON,
        help="Girdi maskesi ve kontrol görüntüsünü yan yana kaydet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input
    suffix = input_path.suffix.lower()

    try:
        search = SlidingWindowSearch(
            n_windows=args.windows, margin_ratio=args.margin, min_pixels_ratio=args.min_pixels,
            min_peak_ratio=args.min_peak, use_momentum=args.momentum, mask_threshold=args.threshold,
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"Girdi bulunamadı: {input_path}\n"
                "Kuş bakışı ikili maskeleri bu yola koy ya da --input ile bir maske, "
                "maske klasörü veya maske videosu ver."
            )
        if input_path.is_file() and suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise ValueError(
                f"Desteklenmeyen dosya türü: {input_path.name}\n"
                f"Görüntü: {', '.join(sorted(IMAGE_EXTENSIONS))} | "
                f"Video: {', '.join(sorted(VIDEO_EXTENSIONS))}"
            )

        args.output.mkdir(parents=True, exist_ok=True)

        if suffix in VIDEO_EXTENSIONS and input_path.is_file():
            output_path, frames, missing_left, missing_right = process_video(
                search, input_path, args.output, args.compare
            )
            print(f"{frames} kare işlendi: {output_path}")
            print(f"  Sol şeridin bulunamadığı kare: {missing_left} | sağ şeridin: {missing_right}")
            return 0

        if input_path.is_dir():
            image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            if not image_paths:
                raise FileNotFoundError(
                    f"Klasörde görüntü dosyası yok: {input_path}\n"
                    f"Desteklenen uzantılar: {', '.join(sorted(IMAGE_EXTENSIONS))}"
                )
        else:
            image_paths = [input_path]

        print("Sliding windows:")
        written = process_images(search, image_paths, args.output, args.compare)
    except PermissionError as exc:
        print(
            f"HATA: Dosyaya erişim izni yok: {exc.filename}\n"
            "Klasörün izinlerini kontrol et ya da --input / --output ile başka bir yol ver.",
            file=sys.stderr,
        )
        return 1
    except cv2.error as exc:
        print(
            f"HATA: OpenCV işlemi tamamlanamadı: {getattr(exc, 'err', exc)}\n"
            "Girdinin tek kanallı bir ikili maske olduğunu kontrol et.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print(f"{written} / {len(image_paths)} maske işlendi: {args.output}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
