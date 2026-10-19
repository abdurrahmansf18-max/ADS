"""
1.2 Perspektif dönüşümü — kuş bakışına çevirme ve geri çevirme (ana hatta her kare).

Ne işe yarar:
    Kareyi ya da maskeyi M ile kuş bakışına çevirir; kuş bakışında çizileni Minv ile kamera görünümüne
    geri taşır.

Nasıl çalışır:
    M ve Minv bir kez yüklenir, cv2.warpPerspective ile çevrilir. Kare boyutu M'nin hesaplandığı boyutla
    aynı olmalıdır; değilse hata verir.

Bağlantılar:
    Alır : 1.2 compute_perspective.py'nin perspektif dosyası; 1.1'in düzelttiği kare ya da 2.1 / 4.4'ün
           maskesi.
    Verir: kuş bakışı maske -> 2.3 şerit arama; kuş bakışı düz yol karesi -> 1.3 ölçek. Geri çevirme ile
           ana hatta bulunan şerit alanı kameradaki kareye çizilir.
    Ana hat: main.py maskeyi warp(), şerit alanını unwarp() ile çevirir.

Parametreler:
    --input       Başka gereksinimden: 1.1'in düzeltilmiş kareleri (varsayılan) ya da 2.1'in maskeleri.
    --transform   Başka gereksinimden: 1.2 compute_perspective.py'nin çıktısı.
    --output, --inverse, --compare   Ayar.
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

# compute_perspective.py'nin ürettiği dönüşüm dosyası (M ve Minv).
TRANSFORM_PATH = PROJECT_ROOT / "data" / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "perspektif.npz"

# [DEĞİŞTİR] Dönüştürülecek girdi: tek görüntü, görüntü klasörü veya video.
# Görüntülerin distorsiyonu 1.1'deki undistort.py ile düzeltilmiş olmalı.
INPUT_PATH = PROJECT_ROOT / "outputs" / "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme" / "duzeltilmis_kareler"

# Dönüştürülmüş görüntülerin/videonun yazılacağı klasör.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / Path(__file__).resolve().parent.name / "kus_bakisi"

# True: Minv ile ters dönüşüm uygulanır (kuş bakışı -> orijinal görünüm).
INVERSE = False

# True: girdi ve çıktı yan yana kaydedilir (karşılaştırma için).
SAVE_COMPARISON = False
# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# [EK] M ve Minv'i dosyadan okuma (np.load).
def load_transform(transform_path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Dönüşüm dosyasından M, Minv ve görüntü boyutunu okur."""
    transform_path = Path(transform_path)
    if not transform_path.is_file():
        raise FileNotFoundError(
            f"Dönüşüm dosyası bulunamadı: {transform_path}\n"
            "Önce compute_perspective.py'yi çalıştır ya da --transform ile dosya yolunu ver."
        )

    try:
        with np.load(transform_path) as data:
            missing = {"M", "Minv", "image_size"} - set(data.files)
            if not missing:
                M = data["M"]
                Minv = data["Minv"]
                width, height = (int(v) for v in data["image_size"])
    # Bozuk zip, düz metin ya da .npz uzantılı tek dizi (.npy) dosyası buraya düşer.
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"Dönüşüm dosyası okunamadı (bozuk ya da .npz biçiminde değil): {transform_path}\n"
            "compute_perspective.py'yi çalıştırıp dosyayı yeniden oluştur ya da --transform ile doğru dosyayı ver."
        ) from exc
    if missing:
        raise ValueError(
            f"Geçersiz dönüşüm dosyası: {transform_path}\n"
            f"Eksik alanlar: {', '.join(sorted(missing))}. compute_perspective.py ile yeniden oluştur."
        )

    return M, Minv, (width, height)


class PerspectiveTransformer:
    """M ve Minv'i bir kez yükler, sabit tutar; her karede yalnızca cv2.warpPerspective çağırır."""

    def __init__(self, transform_path: str | Path):
        self.M, self.Minv, self.image_size = load_transform(transform_path)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Orijinal görüntü -> kuş bakışı."""
        return self._apply(frame, self.M)

    def unwarp(self, frame: np.ndarray) -> np.ndarray:
        """Kuş bakışı -> orijinal görünüm."""
        return self._apply(frame, self.Minv)

    def _apply(self, frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        # [EK] Çözünürlük kontrolü: dönüşüm yalnızca hesaplandığı boyutta geçerlidir.
        height, width = frame.shape[:2]
        if (width, height) != self.image_size:
            raise ValueError(
                f"Görüntü boyutu {width}x{height}, dönüşüm ise "
                f"{self.image_size[0]}x{self.image_size[1]} için hesaplanmış. Noktalar piksel "
                "cinsinden olduğundan dönüşüm yalnızca bu çözünürlükte geçerlidir."
            )
        return cv2.warpPerspective(frame, matrix, self.image_size, flags=cv2.INTER_LINEAR)


# [EK] --compare için yan yana karşılaştırma görüntüsü.
def side_by_side(left: np.ndarray, right: np.ndarray, labels: tuple[str, str]) -> np.ndarray:
    """İki görüntüyü etiketleyerek yan yana birleştirir."""
    panels = []
    for image, label in zip((left, right), labels):
        panel = image.copy()
        scale = max(panel.shape[1] / 1000, 0.6)
        thickness = max(round(2 * scale), 1)
        cv2.putText(panel, label, (round(20 * scale), round(50 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2 * scale, (0, 0, 255), thickness=thickness, lineType=cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def process_images(transform, image_paths: list[Path], output_dir: Path, tag: str,
                   labels: tuple[str, str] | None) -> int:
    """Görüntüleri dönüştürüp kaydeder; yazılan görüntü sayısını döndürür."""
    written = 0
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"UYARI: {path.name} okunamadı, atlandı.", file=sys.stderr)
            continue
        try:
            result = transform(image)
        except ValueError as exc:
            print(f"UYARI: {path.name} atlandı. {exc}", file=sys.stderr)
            continue

        output = side_by_side(image, result, labels) if labels else result
        output_path = output_dir / f"{path.stem}_{tag}{path.suffix}"
        if not cv2.imwrite(str(output_path), output):
            raise OSError(
                f"Görüntü yazılamadı: {output_path}\n"
                "Çıktı klasörünün yazma iznini kontrol et ya da --output ile başka bir klasör ver."
            )
        written += 1
    return written


# [EK] Video girdisi: VideoCapture ile kare kare okunur, VideoWriter ile yazılır.
def process_video(transform, video_path: Path, output_dir: Path, tag: str,
                  labels: tuple[str, str] | None) -> tuple[Path, int]:
    """Videonun her karesini dönüştürüp yeni bir .mp4 dosyasına yazar."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(
            f"Video açılamadı: {video_path}\n"
            "Dosya bozuk ya da desteklenmeyen bir biçimde olabilir; başka bir video dene "
            "ya da videoyu mp4 (H.264) olarak dışa aktar."
        )

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output_path = output_dir / f"{video_path.stem}_{tag}.mp4"
    writer = None
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = transform(frame)
            output = side_by_side(frame, result, labels) if labels else result

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
        description="Kaydedilmiş M / Minv ile görüntü veya videoya perspektif dönüşümü uygular."
    )
    parser.add_argument(
        "--transform", type=Path, default=TRANSFORM_PATH,
        help=f"compute_perspective.py'nin ürettiği .npz dosyası (varsayılan: {TRANSFORM_PATH})",
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
        "--inverse", action="store_true", default=INVERSE,
        help="Minv ile ters dönüşüm uygula (kuş bakışı -> orijinal görünüm)",
    )
    parser.add_argument(
        "--compare", action="store_true", default=SAVE_COMPARISON,
        help="Girdi ve çıktıyı yan yana kaydet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input
    suffix = input_path.suffix.lower()

    try:
        transformer = PerspectiveTransformer(args.transform)
        if args.inverse:
            transform, tag, labels = transformer.unwarp, "unwarped", ("Kus bakisi", "Orijinal gorunum")
        else:
            transform, tag, labels = transformer.warp, "warped", ("Orijinal", "Kus bakisi")
        if not args.compare:
            labels = None

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
            output_path, frame_count = process_video(transform, input_path, args.output, tag, labels)
            print(f"{frame_count} kare dönüştürüldü: {output_path}")
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

        written = process_images(transform, image_paths, args.output, tag, labels)
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
            "Girdi dosyasını ve dönüşüm dosyasını kontrol et.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1

    print(f"{written} / {len(image_paths)} görüntü dönüştürüldü: {args.output}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
