"""
1.1 Kamera kalibrasyonu ve distorsiyon düzeltme — kareleri düzeltme (ana hatta her kare).

Ne işe yarar:
    Her kareden lens bozulmasını kaldırır; düz çizgiler düz görünür. Ana hattın ilk adımıdır, sonraki
    bütün adımlar düzeltilmiş kareyle çalışır.

Nasıl çalışır:
    Kalibrasyon dosyası bir kez yüklenir, her kareye cv2.undistort uygulanır. Kare boyutu kalibrasyonun
    yapıldığı boyuttan farklıysa hata verir.

Bağlantılar:
    Alır : 1.1 calibrate_camera.py'nin kalibrasyon.npz dosyası; düzeltme bu katsayılarla yapılır.
    Verir: düzeltilmiş kareler (outputs/1.1 .../duzeltilmis_kareler) -> 1.2 perspektif kurulumu ve kuş
           bakışı, 2.1 ve 4.2 renk maskesi; çünkü bu adımlar bozulmasız kare ister.
    Ana hat: main.py her karede Undistorter.undistort() çağırır.

Parametreler:
    --input       Gerçek veri: görüntü, klasör ya da video (varsayılan data/ortam_verisi/test_kareleri/kolay_video).
                  [DEĞİŞTİR]
    --calib       Başka gereksinimden: 1.1 calibrate_camera.py'nin çıktısı.
    --output, --compare   Ayar: kayıt klasörü, önce / sonra görseli.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# AYARLAR
# Komut satırından argüman verilmezse bu değerler kullanılır.
# [DEĞİŞTİR] ile işaretlenenleri kendi verine göre ayarlaman gerekir.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# calibrate_camera.py'nin ürettiği kalibrasyon dosyası.
CALIBRATION_PATH = PROJECT_ROOT / "data" / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "kalibrasyon.npz"

# [DEĞİŞTİR] Düzeltilecek girdi: tek görüntü, görüntü klasörü veya video.
# Varsayılan olarak kalibrasyon fotoğrafları düzeltilir (sonucu gözle kontrol etmek için);
# yol görüntüsü/videosu için bunu değiştir.
INPUT_PATH = PROJECT_ROOT / "data" / "ortam_verisi" / "test_kareleri" / "kolay_video"

# Düzeltilmiş görüntülerin/videonun yazılacağı klasör.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / Path(__file__).resolve().parent.name / "duzeltilmis_kareler"

# True: orijinal ve düzeltilmiş görüntü yan yana kaydedilir (karşılaştırma için).
SAVE_COMPARISON = False
# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def load_calibration(calibration_path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Kalibrasyon dosyasından kamera matrisi, distorsiyon katsayıları ve görüntü boyutunu okur."""
    calibration_path = Path(calibration_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(
            f"Kalibrasyon dosyası bulunamadı: {calibration_path}\n"
            "Önce calibrate_camera.py'yi çalıştır ya da --calib ile dosya yolunu ver."
        )

    try:
        with np.load(calibration_path) as data:
            missing = {"camera_matrix", "dist_coeffs", "image_size"} - set(data.files)
            if not missing:
                camera_matrix = data["camera_matrix"]
                dist_coeffs = data["dist_coeffs"]
                width, height = (int(v) for v in data["image_size"])
    # Bozuk zip, düz metin ya da .npz uzantılı tek dizi (.npy) dosyası buraya düşer.
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"Kalibrasyon dosyası okunamadı (bozuk ya da .npz biçiminde değil): {calibration_path}\n"
            "calibrate_camera.py'yi çalıştırıp dosyayı yeniden oluştur ya da --calib ile doğru dosyayı ver."
        ) from exc
    if missing:
        raise ValueError(
            f"Geçersiz kalibrasyon dosyası: {calibration_path}\n"
            f"Eksik alanlar: {', '.join(sorted(missing))}. calibrate_camera.py ile yeniden oluştur."
        )

    return camera_matrix, dist_coeffs, (width, height)


class Undistorter:
    """Kalibrasyonu bir kez yükler; her karede yalnızca cv2.undistort çağırır."""

    def __init__(self, calibration_path: str | Path):
        self.camera_matrix, self.dist_coeffs, self.image_size = load_calibration(calibration_path)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        # [EK] Çözünürlük kontrolü: kamera matrisi yalnızca kalibre edildiği boyutta geçerlidir.
        height, width = frame.shape[:2]
        if (width, height) != self.image_size:
            raise ValueError(
                f"Görüntü boyutu {width}x{height}, kalibrasyon ise "
                f"{self.image_size[0]}x{self.image_size[1]} için yapılmış. Kamera matrisi yalnızca "
                "kalibre edildiği çözünürlükte geçerlidir; aynı çözünürlükte tekrar kalibre et."
            )
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)


# [EK] --compare için yan yana karşılaştırma görüntüsü.
def side_by_side(original: np.ndarray, corrected: np.ndarray) -> np.ndarray:
    """Orijinal ve düzeltilmiş görüntüyü etiketleyerek yan yana birleştirir."""
    panels = []
    for image, label in ((original, "Orijinal"), (corrected, "Duzeltilmis")):
        panel = image.copy()
        scale = max(panel.shape[1] / 1000, 0.6)
        thickness = max(int(2 * scale), 1)
        cv2.putText(panel, label, (int(20 * scale), int(50 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2 * scale, (0, 0, 255), thickness=thickness, lineType=cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def process_images(
    undistorter: Undistorter, image_paths: list[Path], output_dir: Path, compare: bool
) -> int:
    """Görüntüleri düzeltip kaydeder; yazılan görüntü sayısını döndürür."""
    written = 0
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"UYARI: {path.name} okunamadı, atlandı.", file=sys.stderr)
            continue
        try:
            corrected = undistorter.undistort(image)
        except ValueError as exc:
            print(f"UYARI: {path.name} atlandı. {exc}", file=sys.stderr)
            continue

        output = side_by_side(image, corrected) if compare else corrected
        output_path = output_dir / f"{path.stem}_undistorted{path.suffix}"
        if not cv2.imwrite(str(output_path), output):
            raise OSError(
                f"Görüntü yazılamadı: {output_path}\n"
                "Çıktı klasörünün yazma iznini kontrol et ya da --output ile başka bir klasör ver."
            )
        written += 1
    return written


# [EK] Video girdisi: VideoCapture ile kare kare okunur, VideoWriter ile yazılır.
def process_video(
    undistorter: Undistorter, video_path: Path, output_dir: Path, compare: bool
) -> tuple[Path, int]:
    """Videonun her karesini düzeltip yeni bir .mp4 dosyasına yazar."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(
            f"Video açılamadı: {video_path}\n"
            "Dosya bozuk ya da desteklenmeyen bir biçimde olabilir; başka bir video dene "
            "ya da videoyu mp4 (H.264) olarak dışa aktar."
        )

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output_path = output_dir / f"{video_path.stem}_undistorted.mp4"
    writer = None
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            corrected = undistorter.undistort(frame)
            output = side_by_side(frame, corrected) if compare else corrected

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
    return output_path, frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kalibrasyon sonucuyla görüntü veya videodaki lens distorsiyonunu düzeltir."
    )
    parser.add_argument(
        "--calib", type=Path, default=CALIBRATION_PATH,
        help=f"calibrate_camera.py'nin ürettiği .npz dosyası (varsayılan: {CALIBRATION_PATH})",
    )
    parser.add_argument(
        "--input", type=Path, default=INPUT_PATH,
        help=f"Tek görüntü, görüntü klasörü veya video (varsayılan: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--compare", action="store_true", default=SAVE_COMPARISON,
        help="Orijinal ve düzeltilmiş görüntüyü yan yana kaydet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input
    suffix = input_path.suffix.lower()

    try:
        undistorter = Undistorter(args.calib)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Girdi bulunamadı: {input_path}\n"
                "--input ile bir görüntü, görüntü klasörü veya video ver."
            )
        if input_path.is_file() and suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise ValueError(
                f"Desteklenmeyen dosya türü: {input_path.name}\n"
                f"Görüntü: {', '.join(sorted(IMAGE_EXTENSIONS))} | "
                f"Video: {', '.join(sorted(VIDEO_EXTENSIONS))}"
            )

        args.output.mkdir(parents=True, exist_ok=True)

        if suffix in VIDEO_EXTENSIONS and input_path.is_file():
            output_path, frame_count = process_video(undistorter, input_path, args.output, args.compare)
            print(f"{frame_count} kare düzeltildi: {output_path}")
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

        written = process_images(undistorter, image_paths, args.output, args.compare)
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
            "Girdi dosyasını ve kalibrasyon dosyasını kontrol et.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print(f"{written} / {len(image_paths)} görüntü düzeltildi: {args.output}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
