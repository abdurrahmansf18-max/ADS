"""
Arayüz — ana hat servisi: ana hattı (backend/entegrasyon/main.py) arayüzün içinden çalıştırır.

Canlı önizleme gösterebilmek için ana hat ayrı bir süreç değil, arayüzün içinde bir iş parçacığında çalışır;
her kare için LanePipeline.process_frame çağrılır. İşlenmiş video tarayıcının oynatabildiği H.264 (mp4)
olarak ffmpeg (imageio-ffmpeg) ile yazılır. Terminal kutusu için eşdeğer komut ve ilerleme satırları tutulur.
Sonuçlar kurulumun klasörüne, komut satırındaki adlarla yazılır: outputs/arayuz/<kurulum>/<video>_islenmis.mp4 ve
.csv (farklı kurulumlardaki aynı adlı videolar birbirinin üzerine yazmasın).
"""

import contextlib
import importlib.util
import io
import shlex
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import katalog
from katalog import Lock

PREVIEW_EVERY = 5          # canlı önizleme kaç karede bir yenilensin
PREVIEW_WIDTH = 720        # önizleme genişliği (px)
LOG_EVERY_S = 2.0          # terminal kutusuna kaç saniyede bir ilerleme satırı yazılsın
RISK_LEVEL = {"": 0, "yesil": 1, "sari": 2, "kirmizi": 3}
KURULUM_PAGE = "sayfalar/kurulum.py"
SETUP_CHOICE = "ku_ad"                 # kurulum sayfasında seçili kurulum (kilit düğmesi o kurulumu açar)

_MAIN = None


def load_main():
    """backend/entegrasyon/main.py'yi bir kez yükler (gereksinim modüllerini o yükler)."""
    global _MAIN
    if _MAIN is None:
        if str(katalog.INTEGRATION_DIR) not in sys.path:
            sys.path.insert(0, str(katalog.INTEGRATION_DIR))
        spec = importlib.util.spec_from_file_location("ads_ana_hat", katalog.INTEGRATION_DIR / "main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["ads_ana_hat"] = module
        spec.loader.exec_module(module)
        _MAIN = module
    return _MAIN


# --------------------------------------------------------------------------- girdi bilgisi ve kilitler
def video_info(path: Path | None) -> dict | None:
    """Videonun boyutu, hızı ve kare sayısı; açılamıyorsa None."""
    if path is None or not Path(path).is_file():
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        info = {"width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": capture.get(cv2.CAP_PROP_FPS) or 25.0, "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}
    finally:
        capture.release()
    return info if info["width"] > 0 else None


def setup_size(name: str) -> tuple[int, int] | None:
    """Kurulumun hangi çözünürlük için yapıldığı (perspektif dosyasındaki image_size)."""
    path = katalog.setup_files(name)["perspektif"]
    try:
        with np.load(path) as data:
            width, height = (int(v) for v in data["image_size"])
            return width, height
    except (OSError, KeyError, ValueError):
        return None


def run_locks(video: Path | None, setup: str | None, obstacles: bool, info: dict | None,
              model: Path = katalog.MODEL_PATH) -> tuple[list, list]:
    """Ana hattın kilitleri: (kurulum eksikleri, diğer eksikler). İkisi de boşsa çalıştırılabilir."""
    setup_locks, other_locks = [], []
    if setup is None:
        setup_locks.append(Lock("1.1 → 1.3", "", "Hiç kurulum yok: önce bu kamera için kurulum yapılmalı "
                                "(kalibrasyon, perspektif, ölçek).", KURULUM_PAGE))
    else:
        missing = katalog.missing_setup(setup)          # kalibrasyon da kurulumun kendi dosyası
        if missing:
            setup_locks.append(Lock(" → ".join(missing), f"({setup})",
                                    f"{setup} kurulumunun dosyaları eksik: " + ", ".join(
                                        katalog.SETUP_PARTS[p][0] for p, ok in katalog.setup_status(setup).items()
                                        if not ok) + ".", KURULUM_PAGE, setup, SETUP_CHOICE))
    if video is not None and info is None:
        other_locks.append(Lock("", "Video açılamıyor", f"{Path(video).name} açılamadı; dosya bozuk ya da "
                                "desteklenmeyen bir biçimde olabilir."))
    elif info is not None and setup is not None:
        size = setup_size(setup)
        if size is not None and size != (info["width"], info["height"]):
            other_locks.append(Lock("", f"Video {info['width']}x{info['height']}, kurulum {size[0]}x{size[1]} için",
                                    "Bu çözünürlük için yeni bir kurulum gerekir.", KURULUM_PAGE))
    if obstacles and not Path(model).is_file():
        other_locks.append(Lock("", "Model yok; engel tespitini kapat ya da modeli ekle",
                                f"YOLO modeli bulunamadı: {Path(model).name} (`data/hazir_veri/yolo_modeli/`)."))
    return setup_locks, other_locks


# --------------------------------------------------------------------------- ayarlar ve komut
@dataclass
class Settings:
    video: Path
    setup: str
    obstacles: bool
    max_frames: int | None
    smooth: int
    lane_width_range: tuple[float, float] | None    # None: kurulumdaki gerçek şerit genişliğinin ±%20'si
    combine: str
    device: str
    model: Path = katalog.MODEL_PATH

    @property
    def files(self) -> dict[str, Path]:
        return katalog.setup_files(self.setup)

    def args(self) -> list:
        """Bu ayarların main.py argümanları (komut satırındaki gibi)."""
        files = self.files
        args = ["--input", self.video, "--calibration", files["kalibrasyon"], "--perspective", files["perspektif"],
                "--scale", files["olcek"], "--smooth", self.smooth, "--combine", self.combine]
        if self.lane_width_range is not None:
            args += ["--lane-width-range", f"{self.lane_width_range[0]:g}", f"{self.lane_width_range[1]:g}"]
        args += ["--device", self.device, "--model", self.model] if self.obstacles else ["--no-obstacles"]
        if self.max_frames:
            args += ["--max-frames", self.max_frames]
        return args + ["--output", katalog.UI_OUTPUT_DIR / self.setup]

    def command(self) -> str:
        """Bu ayarların komut satırındaki karşılığı (terminal kutusunda gösterilir)."""
        return (f"$ cd {shlex.quote(str(katalog.INTEGRATION_DIR))}\n"
                f"$ python main.py {' '.join(shlex.quote(str(a)) for a in self.args())}")


ADJUSTABLE = {"combine", "smooth", "obstacles", "model", "max_frames"}     # Ana hat sayfasında ayarlananlar
MAIN_GROUP = katalog.SCRIPT_NAMES["main"]


def _range_text(args) -> str:
    if args.lane_width_range is not None:
        return f"{args.lane_width_range[0]:g}–{args.lane_width_range[1]:g} m (elle)"
    try:
        low, high = load_main().lane_width_range(args.scale)
        real = katalog._scale_value(args.scale, "lane_width_m")
        return f"{low:.2f}–{high:.2f} m (kurulumdaki {real:.2f} m ±%20)"
    except (OSError, ValueError):
        return "kurulumdaki gerçek genişliğin ±%20'si"


def fixed_rows(settings: Settings | None, obstacles: bool) -> list:
    """Ana hattın sabit ayarları: main.py'nin kendi ayarları (Ana hat sayfasında ayarlananlar dışında), kurulumdan
    gelen ölçek ve her karede kullandığı modüllerin ayarları, gerçekte kullanılan değerleriyle."""
    main = load_main()
    args = (katalog.parse_command(main, settings.args()) if settings is not None else None) \
        or katalog.parse_command(main, [])
    display = {("main", "device"): lambda a, v: katalog.device_text(v, a.model) if a.obstacles
               else "engel tespiti kapalı",
               ("main", "lane_width_range"): lambda a, v: _range_text(a)}
    rows = katalog.fixed_rows("main", katalog.parser_of(main), args, skip=ADJUSTABLE, group=MAIN_GROUP,
                              display=display)
    for key, name in (("xm_per_pix", "Metre/piksel, yatay (xm)"), ("ym_per_pix", "Metre/piksel, dikey (ym)")):
        value = katalog._scale_value(args.scale, key) if args is not None else None
        rows.append(katalog.FixedRow(MAIN_GROUP, name, f"{value:.6f} m/px" if value else "kurulum yok",
                                     katalog.FROM_SETUP,
                                     f"1.3'ün ölçek dosyasından (`{key}`): yanal sapma ve şerit genişliği xm ile, "
                                     f"eğrilik ve engel mesafesi ym ile hesaplanır.\n\n"
                                     f"{katalog.SOURCE_HELP[katalog.FROM_SETUP]}"))
    return rows + katalog.pipeline_module_rows(obstacles)


# --------------------------------------------------------------------------- video yazıcı
class H264Writer:
    """Kareleri ffmpeg'e verip tarayıcının oynatabildiği H.264 mp4 yazar."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        try:
            import imageio_ffmpeg
        except ImportError:
            raise OSError("imageio-ffmpeg kurulu değil; tarayıcıda oynayacak video yazılamıyor.\n"
                          "İmajı yeniden kur: bash docker/baslat.sh --build") from None
        self.path = Path(path)
        self._process = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(self.path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self._process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self._process.stdin.close()
        error = self._process.stderr.read().decode(errors="replace").strip()
        if self._process.wait() != 0:
            raise OSError(f"Video yazılamadı: {self.path}\n{error or 'ffmpeg hata verdi.'}")


# --------------------------------------------------------------------------- çalıştırma
class PipelineRun:
    """Arka planda çalışan tek bir ana hat çalıştırması; sayfa bunun durumunu okuyup gösterir."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = "starting"            # starting, running, done, stopped, error
        self.error = ""
        self.device = ""
        self.total = 0
        self.done = 0
        self.fps = 0.0
        self.video_fps = 25.0
        self.started = time.time()
        self.finished: float | None = None
        self.preview: bytes | None = None
        self.last = None                   # son karenin FrameResult'ı
        self.results: list = []
        self.rows: list = []
        stem = Path(settings.video).stem
        self.video_path = katalog.UI_OUTPUT_DIR / settings.setup / f"{stem}_islenmis.mp4"
        self.csv_path = katalog.UI_OUTPUT_DIR / settings.setup / f"{stem}_islenmis.csv"
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self.state in ("starting", "running")

    @property
    def command_text(self) -> str:
        return self.settings.command()

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    def log(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def start(self) -> "PipelineRun":
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            self._process()
        except Exception as exc:          # anlaşılır mesajla göster (modüllerin kendi Türkçe mesajları dahil)
            message = str(exc).strip() or type(exc).__name__
            self.error = message
            first, *rest = message.splitlines()
            self.log(f"HATA: {first}")
            for line in rest:
                self.log(f"  {line}")
            self.state = "error"
        finally:
            self.finished = time.time()

    def _process(self) -> None:
        main = load_main()
        s = self.settings
        files = s.files
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):       # cihaz seçimi mesajı terminal kutusuna gelsin
            pipeline = main.LanePipeline(
                files["kalibrasyon"], files["perspektif"], files["olcek"], smooth_frames=s.smooth,
                lane_width_range_m=s.lane_width_range, combine_mode=s.combine, obstacles=s.obstacles,
                model_path=s.model, yolo_iou=main.YOLO_IOU, yolo_device=s.device)
        for line in captured.getvalue().splitlines():
            self.log(line)
        low, high = pipeline.lane_width_range_m
        self.log(f"Kabul edilen şerit genişliği: {low:.2f}–{high:.2f} m")
        if pipeline.obstacles:
            self.device = pipeline.m["5.1"].device_label(pipeline.args51.device)
            self.log(f"YOLO cihazı: {self.device}")

        capture = cv2.VideoCapture(str(s.video))
        if not capture.isOpened():
            raise OSError(f"Video açılamadı: {s.video}\nDosya bozuk ya da desteklenmeyen bir biçimde olabilir.")
        self.video_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total = min(count, s.max_frames) if s.max_frames else count
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        writer, loop_start, last_log = None, time.time(), time.time()
        self.state = "running"
        try:
            index = 0
            while not self._stop.is_set() and (s.max_frames is None or index < s.max_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                output, result = pipeline.process_frame(frame)
                if writer is None:
                    writer = H264Writer(self.video_path, output.shape[1], output.shape[0], self.video_fps)
                writer.write(output)
                self.rows.append(main.result_row(f"kare{index:05d}", result))
                self.results.append(result)
                self.last = result
                index += 1
                self.done = index
                self.fps = index / max(time.time() - loop_start, 1e-6)
                if index == 1 or index % PREVIEW_EVERY == 0:
                    self.preview = _jpeg(output)
                if time.time() - last_log >= LOG_EVERY_S:
                    last_log = time.time()
                    self.log(f"kare {index}/{self.total} · {self.fps:.1f} kare/sn")
        finally:
            capture.release()
            if writer is not None:
                writer.close()
        if not self.results:
            raise ValueError(f"Videodan hiç kare okunamadı: {s.video}\nVideo boş ya da bozuk olabilir.")
        main.write_csv(self.csv_path, self.rows)
        self.log(f"{Path(s.video).name}: {self.done} kare, işleme hızı {self.fps:.1f} FPS")
        self.log(f"  Video: {self.video_path}")
        self.log(f"  Tablo: {self.csv_path}")
        self.state = "stopped" if self._stop.is_set() else "done"

    # ------------------------------------------------------------------ sonuçlar
    def summary(self) -> dict:
        results = self.results
        statuses = Counter(r.status for r in results)
        measured = [r for r in results if r.offset is not None]
        curved = [r.radius_m for r in measured if r.curve_direction in ("sag", "sol")]
        nearest = [r.nearest_in_lane() for r in results]
        return {
            "frames": len(results),
            "measured": statuses["olculdu"], "history": statuses["gecmisten"], "none": statuses["yok"],
            "width": float(np.mean([r.lane_width_m for r in measured])) if measured else float("nan"),
            "offset": float(np.mean([r.offset.offset_m for r in measured])) if measured else float("nan"),
            "radius": float(np.median(curved)) if curved else float("nan"),
            "straight": sum(r.curve_direction == "duz" for r in measured),
            "detections": sum(len(r.obstacles) for r in results),
            "in_lane": sum(n is not None for n in nearest),
            "risks": Counter(n["risk"] for n in nearest if n is not None),
        }

    def frame_table(self) -> pd.DataFrame:
        """Kare kare sonuç tablosu (CSV ile aynı sütunlar) + zaman ve sayısal risk düzeyi."""
        main = load_main()
        frame = pd.DataFrame(self.rows, columns=main.CSV_COLUMNS)
        frame.insert(0, "zaman_sn", np.round(np.arange(len(frame)) / self.video_fps, 2))
        for column in ("serit_genisligi_m", "sapma_m", "egrilik_r_m", "en_yakin_engel_m"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["risk_duzeyi"] = frame["risk"].map(RISK_LEVEL).fillna(0).astype(int)
        return frame


def _jpeg(image: np.ndarray) -> bytes:
    height, width = image.shape[:2]
    if width > PREVIEW_WIDTH:
        image = cv2.resize(image, (PREVIEW_WIDTH, round(height * PREVIEW_WIDTH / width)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes() if ok else b""
