"""
1.2 Perspektif dönüşümü — kuş bakışı matrisinin hesabı (kurulum, her kamera montajı için bir kez).

Ne işe yarar:
    Yola eğik bakan görüntüyü yukarıdan bakıyormuş gibi düzleştiren matrisi (M) ve tersini (Minv) bulur.
    Kuş bakışında şeritler paralel görünür; şerit arama ve metre ölçümü bu görüntüde yapılır.

Nasıl çalışır:
    Düz yol karesinde yolu çevreleyen 4 nokta (yamuk) ve hedef dikdörtgenin 4 köşesinden
    cv2.getPerspectiveTransform ile M ve Minv hesaplanır ve kaydedilir. Bir kontrol görüntüsü çizilir:
    kuş bakışında şeritler paralel çıkmalıdır.

Bağlantılar:
    Alır : 1.1'in düzelttiği düz yol karesi; noktalar bozulmasız karede seçilmelidir.
    Verir: data/uretilen_veri/kurulum_dosyalari/<kurulum>/perspektif.npz (M, Minv, image_size, warped_size)
           -> 1.2 warp_perspective.py, 5.5 (engelin yerini kuş bakışına taşır) ve ana hat.

Parametreler:
    --image       Başka gereksinimden: 1.1'in düzeltilmiş düz yol karesi (duz_yol_1_undistorted.jpg).
    --src         Gerçek veriden ölçülür: karedeki 4 yol noktası. Kamera montajına özeldir; zor video
                  için ayrı ölçüldü (kurulum_dosyalari/zor_video/perspektif.npz). [DEĞİŞTİR]
    --dst, --output, --verify-dir   Ayar: hedef dikdörtgen (varsayılan uygun), kayıt yerleri.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# AYARLAR
# Komut satırından argüman verilmezse bu değerler kullanılır.
# [DEĞİŞTİR] ile işaretlenenleri kendi verine göre ayarlaman gerekir.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# [DEĞİŞTİR] Düz yol görüntüsü (1.1'deki undistort.py ile düzeltilmiş olmalı).
IMAGE_PATH = PROJECT_ROOT / "outputs" / "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme" / "duzeltilmis_kareler" / "duz_yol_1_undistorted.jpg"

# [DEĞİŞTİR] Kaynak noktalar: yol yamuğunun köşeleri, piksel (x, y).
# Sıra: sol-üst, sağ-üst, sağ-alt, sol-alt. Yamuğun sol ve sağ kenarı
# şerit çizgilerinin üzerine oturmalı.
# None: görüntü boyutuna göre varsayılan yamuk (DEFAULT_SRC_RATIOS) kullanılır.
# Örnek (1280x720): [(585, 460), (695, 460), (1127, 720), (203, 720)]
# Kolay video için ölçülen noktalar (düzeltilmiş düz yol karelerinde şerit çizgilerine uydurularak).
SRC_POINTS = [(581, 460), (702, 460), (1106, 720), (212, 720)]

# Hedef noktalar: kuş bakışı görüntüdeki dikdörtgen, aynı sırayla, piksel (x, y).
# None: görüntü boyutuna göre varsayılan dikdörtgen (DEFAULT_DST_RATIOS) kullanılır.
DST_POINTS = None

# M ve Minv'in kaydedileceği dosya (warp_perspective.py bu dosyayı okur).
OUTPUT_PATH = PROJECT_ROOT / "data" / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "perspektif.npz"

# Kuş bakışı görüntünün ve doğrulama görüntüsünün yazılacağı klasör.
VERIFY_DIR = PROJECT_ROOT / "outputs" / Path(__file__).resolve().parent.name
# =============================================================================

# [EK] Varsayılan noktalar, görüntü boyutuna oran olarak (x / genişlik, y / yükseklik).
# Kaputu görünmeyen, ufku görüntü yüksekliğinin ~%64'ünde olan bir ön kamera için tipik değerler.
DEFAULT_SRC_RATIOS = [(0.457, 0.639), (0.543, 0.639), (0.880, 1.0), (0.159, 1.0)]
DEFAULT_DST_RATIOS = [(0.25, 0.0), (0.75, 0.0), (0.75, 1.0), (0.25, 1.0)]

POINT_ORDER = "sol-üst, sağ-üst, sağ-alt, sol-alt"
DRAW_COLOR = (0, 0, 255)


def load_image(image_path: Path) -> np.ndarray:
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Görüntü bulunamadı: {image_path}\n"
            "Düz yol görüntüsünü bu yola koy ya da --image ile başka bir dosya ver."
        )
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(
            f"Görüntü okunamadı (bozuk ya da desteklenmeyen biçim): {image_path}\n"
            "Dosyayı bir görüntü programıyla açıp kontrol et ya da --image ile başka bir görüntü ver."
        )
    return image


def to_points(values, name: str) -> np.ndarray:
    """Ayarlardaki [(x, y), ...] listesini ya da komut satırındaki 8 sayıyı 4x2 diziye çevirir."""
    flat = np.asarray(values, np.float32).ravel()
    if flat.size != 8:
        raise ValueError(f"{name}: tam 4 nokta (8 sayı: x1 y1 ... x4 y4) gerekli, {flat.size} sayı verildi.")
    return flat.reshape(4, 2)


def points_from_ratios(ratios, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return np.array([(x * width, y * height) for x, y in ratios], np.float32)


# [EK] Nokta kontrolü: yanlış sıralı ya da bozuk dörtgen görüntüyü ters çevirir veya bozar.
def validate_quad(points: np.ndarray, name: str, image_size: tuple[int, int]) -> None:
    """Noktaların görüntü içinde, doğru sırada ve dışbükey bir dörtgen oluşturduğunu kontrol eder."""
    width, height = image_size
    outside = [str(i) for i, (x, y) in enumerate(points, start=1) if not (0 <= x <= width and 0 <= y <= height)]
    if outside:
        raise ValueError(f"{name}: {', '.join(outside)}. nokta görüntünün dışında (görüntü {width}x{height}).")

    top_left, top_right, bottom_right, bottom_left = points
    if not (
        top_left[0] < top_right[0]
        and bottom_left[0] < bottom_right[0]
        and top_left[1] < bottom_left[1]
        and top_right[1] < bottom_right[1]
    ):
        raise ValueError(f"{name}: noktaların sırası {POINT_ORDER} olmalı.")

    # Ardışık kenarların vektörel çarpımı hep pozitif olmalı (dışbükey, saat yönünde dörtgen).
    corners = points.astype(np.float64)
    for i in range(4):
        a, b, c = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross <= 0:
            raise ValueError(
                f"{name}: noktalar dışbükey bir dörtgen oluşturmuyor "
                "(üç nokta aynı doğru üzerinde ya da bir köşe içeri göçük)."
            )


def compute_transform(src_points: np.ndarray, dst_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M: kaynak -> hedef, Minv: hedef -> kaynak."""
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    return M, Minv


# [EK] Doğrulama görüntüsü için çizim (polylines, circle, putText).
def draw_quad(image: np.ndarray, points: np.ndarray, label: str) -> np.ndarray:
    """Dörtgeni ve numaralı köşelerini görüntünün üzerine çizer."""
    output = image.copy()
    scale = max(output.shape[1] / 1000, 0.6)
    thickness = max(round(2 * scale), 1)
    corners = np.round(points).astype(np.int32)

    cv2.polylines(output, [corners], True, DRAW_COLOR, thickness=thickness, lineType=cv2.LINE_AA)
    for number, (x, y) in enumerate(corners, start=1):
        cv2.circle(output, (int(x), int(y)), round(6 * scale), DRAW_COLOR,
                   thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(output, str(number), (int(x) + round(10 * scale), int(y) - round(10 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, DRAW_COLOR,
                    thickness=thickness, lineType=cv2.LINE_AA)
    cv2.putText(output, label, (round(20 * scale), round(50 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, DRAW_COLOR,
                thickness=thickness, lineType=cv2.LINE_AA)
    return output


# [EK] M ve Minv'i sabit tutmak için dosyaya kaydetme (np.savez).
def save_transform(output_path: Path, M, Minv, src_points, dst_points, image_size) -> Path:
    """M ve Minv'i np.savez ile kaydeder ve yazılan dosyanın yolunu döndürür."""
    # np.savez uzantı yoksa kendisi ".npz" ekler; ekrana doğru yolu yazmak için burada ekliyoruz.
    if output_path.suffix != ".npz":
        output_path = output_path.with_name(output_path.name + ".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        M=M,
        Minv=Minv,
        src_points=src_points,
        dst_points=dst_points,
        image_size=np.array(image_size),
        # Kuş bakışı görüntü orijinalle aynı boyutta üretilir (warpPerspective dsize = image_size).
        warped_size=np.array(image_size),
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Yol yamuğu ve hedef dikdörtgenden perspektif dönüşüm matrisi M ve Minv'i hesaplar."
    )
    point_metavar = ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")
    parser.add_argument(
        "--image", type=Path, default=IMAGE_PATH,
        help=f"Düz yol görüntüsü (varsayılan: {IMAGE_PATH})",
    )
    parser.add_argument(
        "--src", type=float, nargs=8, metavar=point_metavar, default=SRC_POINTS,
        help=f"Kaynak noktalar, piksel ({POINT_ORDER}). Varsayılan: görüntü boyutuna göre tipik yamuk",
    )
    parser.add_argument(
        "--dst", type=float, nargs=8, metavar=point_metavar, default=DST_POINTS,
        help=f"Hedef noktalar, piksel ({POINT_ORDER}). Varsayılan: genişliğin %%25-%%75'i arası dikdörtgen",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"M ve Minv'in yazılacağı .npz dosyası (varsayılan: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--verify-dir", type=Path, default=VERIFY_DIR,
        help=f"Kuş bakışı ve doğrulama görüntülerinin klasörü (varsayılan: {VERIFY_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        image = load_image(args.image)
        height, width = image.shape[:2]
        image_size = (width, height)

        src_points = (
            to_points(args.src, "Kaynak noktalar") if args.src is not None
            else points_from_ratios(DEFAULT_SRC_RATIOS, image_size)
        )
        dst_points = (
            to_points(args.dst, "Hedef noktalar") if args.dst is not None
            else points_from_ratios(DEFAULT_DST_RATIOS, image_size)
        )
        validate_quad(src_points, "Kaynak noktalar", image_size)
        validate_quad(dst_points, "Hedef noktalar", image_size)

        M, Minv = compute_transform(src_points, dst_points)
        warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)

        output_path = save_transform(args.output, M, Minv, src_points, dst_points, image_size)

        args.verify_dir.mkdir(parents=True, exist_ok=True)
        stem, suffix = args.image.stem, args.image.suffix
        warped_path = args.verify_dir / f"{stem}_warped{suffix}"
        verify_path = args.verify_dir / f"{stem}_verification{suffix}"
        verification = np.hstack([
            draw_quad(image, src_points, "Orijinal + kaynak noktalar"),
            draw_quad(warped, dst_points, "Kus bakisi + hedef dikdortgen"),
        ])
        for path, output in ((warped_path, warped), (verify_path, verification)):
            if not cv2.imwrite(str(path), output):
                raise OSError(
                    f"Görüntü yazılamadı: {path}\n"
                    "--verify-dir klasörünün yazma iznini kontrol et ya da başka bir klasör ver."
                )
    except PermissionError as exc:
        print(
            f"HATA: Dosyaya erişim izni yok: {exc.filename}\n"
            "Klasörün izinlerini kontrol et ya da --image / --output / --verify-dir ile başka bir yol ver.",
            file=sys.stderr,
        )
        return 1
    except cv2.error as exc:
        print(
            f"HATA: OpenCV dönüşümü hesaplayamadı: {getattr(exc, 'err', exc)}\n"
            "Kaynak ve hedef noktalarını kontrol et.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    np.set_printoptions(precision=4, suppress=True)
    print("Perspektif dönüşümü hesaplandı.")
    print(f"  Görüntü boyutu : {width}x{height}")
    print(f"  Kaynak noktalar ({POINT_ORDER}):\n   " + str(src_points).replace("\n", "\n   "))
    print(f"  Hedef noktalar:\n   " + str(dst_points).replace("\n", "\n   "))
    print("  M:\n   " + str(M).replace("\n", "\n   "))
    print("  Minv:\n   " + str(Minv).replace("\n", "\n   "))
    print(f"  Kaydedildi     : {output_path}")
    print(f"  Kuş bakışı     : {warped_path}")
    print(f"  Doğrulama      : {verify_path}")
    print(
        "\nDoğrulama görüntüsünde şeritler kuş bakışında birbirine paralel olmalı (aralarındaki\n"
        "mesafe her yükseklikte aynı) ve dikdörtgenin dikey kenarlarıyla hizalı görünmeli.\n"
        "Yukarı doğru açılıyor ya da daralıyorsa --src'deki üst iki noktayı (1 ve 2) şerit\n"
        "çizgileri üzerinde kaydırıp tekrar çalıştır."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
