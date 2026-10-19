"""
1.1 Kamera kalibrasyonu ve distorsiyon düzeltme — kamera kalibrasyonu (kurulum, bir kez çalıştırılır).

Ne işe yarar:
    Kamera lensinin görüntüyü nasıl büktüğünü ölçer. Lens düz çizgileri eğri gösterir; bu düzeltilmezse
    kuş bakışı, şerit ölçümü ve eğrilik hatalı çıkar.

Nasıl çalışır:
    Satranç tahtası fotoğraflarında tahtanın iç köşeleri bulunur. Köşelerin bilinen düzeninden kamera
    matrisi ve bozulma katsayıları hesaplanır (cv2.calibrateCamera) ve dosyaya kaydedilir.
    Lens modeli kendiliğinden seçilir: normal (5 katsayı) ve geniş açılı (8 katsayı) model ikisi de denenir.
    Geniş açılı model hatayı en az yarıya indiriyorsa ya da normal model güvenilmezse o seçilir; yoksa normal
    model kalır (fazla katsayı, az veriyle görüntü kenarlarında yanlış düzeltme yapabilir).
    Köşesi bulunamayan fotoğraflar atlanır. Boyut ölçütü, köşesi bulunan fotoğraflarda en çok görülen
    boyuttur; farklı boyuttakiler de atlanır. Sonuç güvenilir değilse (10'dan az fotoğraf ya da hata
    1 pikselden büyük) dosya kaydedilmez, var olan dosyaya dokunulmaz ve ne yapılacağını söyleyen bir
    HATA verilir.

Bağlantılar:
    Alır : Hiçbir gereksinimden almaz; zincirin ilk adımıdır.
    Verir: data/uretilen_veri/kurulum_dosyalari/<kurulum>/kalibrasyon.npz (camera_matrix, dist_coeffs, image_size);
           her kurulumun (kameranın) kendi kalibrasyonu.
           1.1 undistort.py, 2.2, 4.1, 4.3, 5.x ve ana hat bunu okur; çünkü bütün ölçümler
           bozulması giderilmiş karede yapılmalıdır.

Parametreler:
    --images      Gerçek veri: satranç tahtası fotoğrafları (videoyu çeken kameradan;
                  data/hazir_veri/kalibrasyon_fotograflari/<kurulum>/). [DEĞİŞTİR]
    --pattern     Gerçek veri: tahtanın iç köşe sayısı (9 6); her kenar 3–20, önerilen 5x4 – 18x12. [DEĞİŞTİR]
    --lens-model  Ayar: otomatik (varsayılan), normal ya da genis; elle sabitlemek için.
    --output, --corners-dir
                  Ayar: kayıt yeri, köşe kontrol görüntüleri.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# AYARLAR
# Komut satırından argüman verilmezse bu değerler kullanılır.
# [DEĞİŞTİR] ile işaretlenenleri kendi verine göre ayarlaman gerekir.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# [DEĞİŞTİR] Satranç tahtası fotoğraflarının bulunduğu klasör (15-20 fotoğraf): kurulumun (kameranın) adıyla.
# Fotoğrafların hepsi aynı kamerayla ve aynı çözünürlükte çekilmiş olmalı.
IMAGES_DIR = PROJECT_ROOT / "data" / "hazir_veri" / "kalibrasyon_fotograflari" / "kolay_video"

# [DEĞİŞTİR] Tahtadaki İÇ köşe sayısı (sütun, satır). Kare sayısı değildir:
# 10x7 kareli bir tahtanın iç köşe sayısı 9x6'dır.
PATTERN_SIZE = (9, 6)

# Kalibrasyon sonucunun kaydedileceği dosya, kurulumun klasöründe (undistort.py bu dosyayı okur).
OUTPUT_PATH = PROJECT_ROOT / "data" / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "kalibrasyon.npz"

# Köşeleri çizilmiş kontrol görüntülerinin kaydedileceği klasör (None: kaydetme).
CORNERS_DIR = None

# Lens modeli: "otomatik" normal (5 katsayı) ve geniş açılı (8 katsayı) modeli dener, uygun olanı seçer
# (bkz. WIDE_GAIN). "normal" ya da "genis" ile elle sabitlenir.
LENS_MODEL = "otomatik"
# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# [EK] Bundan az geçerli fotoğrafla kalibrasyon yapılmaz; önerilen sayının altında uyarı verilir.
MIN_IMAGES = 3
RECOMMENDED_MIN_IMAGES = 10
# [EK] Bu değerin üzerindeki RMS hatası kalibrasyonun güvenilmez olduğunu gösterir.
MAX_GOOD_RMS = 1.0
# [EK] Geniş açılı model, hatayı en az bu orana indiriyorsa seçilir (0.5: yarıya). Gerçek lens farkı hatayı çok
# düşürür; küçük düşüş yalnız fazla katsayının gürültüye uymasıdır.
WIDE_GAIN = 0.5

# [EK] İç köşe sayısı sınırı (her kenar). 1280x720'de ölçüldü (sentetik, projenin kamerasıyla): 3'ün altında hesap
# belirsiz; 5x4 – 18x12 en doğru aralık; 20x14'te kareler küçülüp köşe inceltme penceresine (aşağıda) sığmadığı için
# hata artar, 25x17'de sonuç güvenilmez çıkar.
PATTERN_LIMITS = (3, 20)

# [EK] cornerSubPix arama penceresi ve durma ölçütü.
SUBPIX_WINDOW = (11, 11)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

DIST_COEFF_NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
LENS_MODELS = {"normal": "normal (5 katsayı)", "genis": "geniş açılı (8 katsayı)"}


def list_images(images_dir: Path) -> list[Path]:
    """Klasördeki görüntü dosyalarını sıralı olarak döndürür."""
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Fotoğraf klasörü bulunamadı: {images_dir}\n"
            "Satranç tahtası fotoğraflarını bu klasöre koy ya da --images ile başka bir klasör ver."
        )
    if not images_dir.is_dir():
        raise NotADirectoryError(f"--images bir klasör olmalı: {images_dir}")

    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(
            f"Klasörde görüntü dosyası yok: {images_dir}\n"
            f"Desteklenen uzantılar: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    return paths


def chessboard_object_points(pattern_size: tuple[int, int]) -> np.ndarray:
    """Tahtanın iç köşelerinin düzlem üzerindeki 3B koordinatları (Z = 0).

    Kare kenarı 1 birim kabul edilir; bu seçim kamera matrisini ve
    distorsiyon katsayılarını etkilemez.
    """
    cols, rows = pattern_size
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return points


def find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    """Satranç tahtası köşelerini bulur ve alt-piksel hassasiyetine iyileştirir."""
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return None
    # [EK] cornerSubPix: köşeleri alt-piksel hassasiyetine iyileştirir, kalibrasyon hatasını düşürür.
    return cv2.cornerSubPix(gray, corners, SUBPIX_WINDOW, (-1, -1), SUBPIX_CRITERIA)


def calibrate(
    image_paths: list[Path],
    pattern_size: tuple[int, int],
    corners_dir: Path | None = None,
    rational_model: bool = False,
) -> dict:
    """Fotoğraflardaki köşelerden kamera matrisini ve distorsiyon katsayılarını hesaplar."""
    object_template = chessboard_object_points(pattern_size)
    object_points, image_points = [], []
    used, rejected = [], []

    # 1. tur: her fotoğrafta köşeleri bul.
    found = []  # (yol, görüntü ya da None, köşeler, (genişlik, yükseklik))
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            rejected.append((path, "dosya okunamadı"))
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, pattern_size)
        if corners is None:
            rejected.append((path, "köşeler bulunamadı"))
            continue
        keep = image if corners_dir is not None else None  # kontrol görüntüsü için
        found.append((path, keep, corners, (gray.shape[1], gray.shape[0])))

    # [EK] Tüm fotoğraflar aynı çözünürlükte olmalı. Ölçüt, köşesi bulunan fotoğraflarda en çok görülen
    # boyuttur; fotoğrafların sırasına bağlı değildir. Farklı boyuttakiler atlanır.
    image_size = None
    sizes = Counter(size for *_, size in found).most_common()
    if sizes:
        if len(sizes) > 1 and sizes[0][1] == sizes[1][1]:
            listed = ", ".join(f"{w}x{h} ({n} fotoğraf)" for (w, h), n in sizes)
            raise ValueError(
                f"Fotoğraflar farklı boyutlarda ve hiçbiri çoğunlukta değil: {listed}.\n"
                "Yalnız videoyu çeken kameranın, videoyla aynı çözünürlükteki fotoğraflarını kullan."
            )
        image_size = sizes[0][0]

    # 2. tur: çoğunluk boyutundaki fotoğrafları kullan.
    for path, image, corners, size in found:
        if size != image_size:
            rejected.append(
                (path, f"boyut farklı ({size[0]}x{size[1]}, çoğunluk {image_size[0]}x{image_size[1]})")
            )
            continue

        object_points.append(object_template)
        image_points.append(corners)
        used.append(path)

        # [EK] İsteğe bağlı kontrol görüntüsü: köşelerin doğru bulunduğunu gözle görmek için.
        if corners_dir is not None:
            cv2.drawChessboardCorners(image, pattern_size, corners, True)
            corners_path = corners_dir / f"{path.stem}_corners.jpg"
            if not cv2.imwrite(str(corners_path), image):
                raise OSError(
                    f"Kontrol görüntüsü yazılamadı: {corners_path}\n"
                    "--corners-dir klasörünün yazma iznini kontrol et ya da başka bir klasör ver."
                )

    rejected.sort(key=lambda item: item[0])  # rapor fotoğraf sırasıyla yazılsın

    if len(used) < MIN_IMAGES:
        cols, rows = pattern_size
        raise ValueError(
            f"Köşeleri bulunan fotoğraf sayısı {len(used)}; kalibrasyon için en az {MIN_IMAGES} gerekli.\n"
            f"--pattern değerinin tahtadaki İÇ köşe sayısıyla (şu an {cols}x{rows}) "
            "eşleştiğini ve tahtanın fotoğraflarda tamamen göründüğünü kontrol et."
        )

    # [EK] İsteğe bağlı rasyonel model: geniş açılı lensler için 8 katsayılı distorsiyon.
    flags = cv2.CALIB_RATIONAL_MODEL if rational_model else 0
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None, flags=flags
    )
    # OpenCV 5 katsayı dizisini 14 elemana tamamlayabiliyor; modelin kullanmadıkları sıfırdır, atılır.
    dist_coeffs = dist_coeffs[:, : len(DIST_COEFF_NAMES) if rational_model else 5]

    # [EK] projectPoints ile her fotoğraf için ayrı RMS hatası: hatalı fotoğrafları bulmaya yarar.
    per_view_errors = []
    for corners, rvec, tvec in zip(image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(object_template, rvec, tvec, camera_matrix, dist_coeffs)
        residuals = corners.reshape(-1, 2) - projected.reshape(-1, 2)
        per_view_errors.append(float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))))

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_size": image_size,
        "rms": rms,
        "pattern_size": pattern_size,
        "used": used,
        "rejected": rejected,
        "per_view_errors": per_view_errors,
    }


def save_calibration(output_path: Path, result: dict) -> Path:
    """Kalibrasyon sonucunu np.savez ile kaydeder ve yazılan dosyanın yolunu döndürür."""
    # np.savez uzantı yoksa kendisi ".npz" ekler; ekrana doğru yolu yazmak için burada ekliyoruz.
    if output_path.suffix != ".npz":
        output_path = output_path.with_name(output_path.name + ".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        camera_matrix=result["camera_matrix"],
        dist_coeffs=result["dist_coeffs"],
        image_size=np.array(result["image_size"]),
        rms=result["rms"],
        pattern_size=np.array(result["pattern_size"]),
    )
    return output_path


# [EK] Güvenilirlik denetimi: sonuç bu sınırların dışındaysa dosya kaydedilmez.
def reliability_problems(result: dict) -> list[str]:
    problems = []
    if len(result["used"]) < RECOMMENDED_MIN_IMAGES:
        problems.append(
            f"{len(result['used'])} fotoğraf kullanılabildi; güvenilir sonuç için en az "
            f"{RECOMMENDED_MIN_IMAGES} gerekir."
        )
    if result["rms"] > MAX_GOOD_RMS:
        problems.append(f"RMS hatası {result['rms']:.3f} piksel; en çok {MAX_GOOD_RMS} piksel olmalı.")
    return problems


# [EK] Lens modelinin kendiliğinden seçilmesi: gerekmedikçe az katsayılı model (fazla katsayı, az veriyle
# görüntü kenarlarında yanlış düzeltme yapabilir).
def calibrate_auto(image_paths: list[Path], pattern_size: tuple[int, int], corners_dir: Path | None = None,
                   lens_model: str = "otomatik") -> dict:
    """lens_model "normal" ya da "genis" ise o model. "otomatik" ise ikisi de denenir: geniş açılı model güvenilirse
    ve hatayı en az WIDE_GAIN oranına indiriyorsa (ya da normal model güvenilmezse) o, yoksa normal seçilir.
    Sonuçta "lens_model" (seçilen) ve "tried" (denenip seçilmeyen model ve hatası) yazar."""
    if lens_model != "otomatik":
        result = calibrate(image_paths, pattern_size, corners_dir, lens_model == "genis")
        result.update(lens_model=lens_model, tried=[])
        return result
    normal = calibrate(image_paths, pattern_size, corners_dir, False)
    normal.update(lens_model="normal", tried=[])
    try:
        wide = calibrate(image_paths, pattern_size, None, True)   # köşe görüntüleri ilk turda yazıldı
    except cv2.error:                                     # 8 katsayı bu veriyle çözülemedi: normal sonuç kalır
        return normal
    wide.update(lens_model="genis", tried=[("normal", normal["rms"])])
    normal["tried"] = [("genis", wide["rms"])]
    if not reliability_problems(wide) and (reliability_problems(normal) or wide["rms"] <= WIDE_GAIN * normal["rms"]):
        return wide
    return normal


def print_report(result: dict, total_images: int, output_path: Path | None) -> None:
    width, height = result["image_size"]
    np.set_printoptions(precision=4, suppress=True)

    print("Kalibrasyon tamamlandı.")
    print(f"  Kullanılan fotoğraf : {len(result['used'])} / {total_images}")
    print(f"  Görüntü boyutu      : {width}x{height}")
    print(f"  RMS hatası          : {result['rms']:.4f} piksel")
    print(f"  Lens modeli         : {LENS_MODELS[result['lens_model']]}")
    for model, rms in result["tried"]:
        print(f"                        ({LENS_MODELS[model]} de denendi: RMS {rms:.4f} piksel; seçilmedi)")
    print("  Kamera matrisi:")
    print("   " + str(result["camera_matrix"]).replace("\n", "\n   "))
    coeff_names = ", ".join(DIST_COEFF_NAMES[: result["dist_coeffs"].size])
    print(f"  Distorsiyon katsayıları ({coeff_names}): {result['dist_coeffs'].ravel()}")
    if output_path:
        print(f"  Kaydedildi          : {output_path}")
    else:
        print("  Kaydedilmedi        : (aşağıdaki HATA'ya bak)")

    print("\nFotoğraf başına RMS hatası (piksel):")
    for path, error in zip(result["used"], result["per_view_errors"]):
        print(f"  {path.name}: {error:.4f}")

    if result["rejected"]:
        print("\nAtlanan fotoğraflar:")
        for path, reason in result["rejected"]:
            print(f"  {path.name}: {reason}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Satranç tahtası fotoğraflarından kamera matrisi ve distorsiyon katsayılarını hesaplar."
    )
    parser.add_argument(
        "--images", type=Path, default=IMAGES_DIR,
        help=f"Satranç tahtası fotoğraflarının klasörü (varsayılan: {IMAGES_DIR})",
    )
    parser.add_argument(
        "--pattern", type=int, nargs=2, metavar=("SUTUN", "SATIR"), default=PATTERN_SIZE,
        help=f"Tahtadaki iç köşe sayısı (varsayılan: {PATTERN_SIZE[0]} {PATTERN_SIZE[1]})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Kalibrasyon sonucunun yazılacağı .npz dosyası (varsayılan: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--corners-dir", type=Path, default=CORNERS_DIR,
        help="Köşeleri çizilmiş kontrol görüntülerinin kaydedileceği klasör (varsayılan: kaydetme)",
    )
    parser.add_argument(
        "--lens-model", choices=("otomatik", "normal", "genis"), default=LENS_MODEL,
        help="Lens modeli: otomatik (önce normal, hata yüksekse geniş açılı), normal (5 katsayı), "
             f"genis (8 katsayı) (varsayılan: {LENS_MODEL})",
    )
    args = parser.parse_args()
    low, high = PATTERN_LIMITS
    if min(args.pattern) < low or max(args.pattern) > high:
        parser.error(f"--pattern değerleri {low}-{high} arasında olmalı (tahtanın iç köşe sayısı), verilen: "
                     f"{args.pattern[0]} {args.pattern[1]}. Önerilen aralık 5x4 – 18x12.")
    return args


def main() -> int:
    args = parse_args()
    pattern_size = tuple(args.pattern)

    try:
        image_paths = list_images(args.images)
        if args.corners_dir is not None:
            args.corners_dir.mkdir(parents=True, exist_ok=True)
        result = calibrate_auto(image_paths, pattern_size, args.corners_dir, args.lens_model)
        problems = reliability_problems(result)
        output_path = save_calibration(args.output, result) if not problems else None
    except PermissionError as exc:
        print(
            f"HATA: Dosyaya erişim izni yok: {exc.filename}\n"
            "Klasörün izinlerini kontrol et ya da --images / --output / --corners-dir ile başka bir yol ver.",
            file=sys.stderr,
        )
        return 1
    except cv2.error as exc:
        print(
            f"HATA: OpenCV kalibrasyonu tamamlayamadı: {getattr(exc, 'err', exc)}\n"
            "Fotoğrafları ve --pattern değerini kontrol et.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print_report(result, len(image_paths), output_path)
    advice = ("Hatası yüksek, bulanık ya da tahtası tam görünmeyen fotoğrafları çıkar; tahtanın farklı "
              "açılardan tamamen göründüğü yeni fotoğraflar ekleyip tekrar çalıştır.")
    if problems:
        print("\nHATA: Kalibrasyon güvenilir değil, dosya kaydedilmedi (var olan dosyaya dokunulmadı):\n"
              + "".join(f"  - {p}\n" for p in problems) + f"  -> {advice}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
