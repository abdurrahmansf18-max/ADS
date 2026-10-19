"""
Entegrasyon — bütün gereksinimlerin tek hatta (process_frame) bağlanması.

Ne işe yarar:
    Videoyu kare kare işler: şerit alanını boyar; yanal sapmayı, eğrilik yarıçapını ve viraj yönünü
    yazar; engelleri şeritle ilişkilendirip çizer (şeritteki kırmızı, dışındaki yeşil) ve FPS ölçer.

Nasıl çalışır:
    1. Kare düzeltilir (1.1).
    2. Şerit maskesi çıkarılır: 4.1 yumuşatma + 4.2 renk + 4.3 gradyan -> 4.4 OR.
    3. Maske kuş bakışına çevrilir (1.2), şerit pikselleri bulunur (2.3), polinom uydurulur (2.4).
       Son 8 karenin ortalaması alınır; şerit genişliği makul değilse (kurulumdaki gerçek genişliğin ±%20'si
       dışındaysa) o karenin ölçümü reddedilir.
    4. Sapma (2.5), eğrilik (3.3) ve viraj sınıfı (3.1) hesaplanır.
    5. Engeller bulunur ve şeritle ilişkilendirilir (5.1 -> 5.3 -> 5.2 -> 5.5).
    6. Şerit alanı Minv ile kareye geri çizilir (1.2), yazılar eklenir.

Bağlantılar:
    Gereksinim modüllerini değiştirmeden kendi klasörlerinden yükler (moduller.py) ve çekirdek
    fonksiyonlarını çağırır; böylece her modül tek başına da çalışmaya devam eder.
    Alır : 1.1, 1.2 ve 1.3'ün kurulum dosyaları (data/uretilen_veri/kurulum_dosyalari/<kurulum>/) ve YOLO modeli.
    Verir: outputs/entegrasyon/<ad>_islenmis.mp4 ve kare başına sonuç tablosu <ad>_islenmis.csv.

Parametreler:
    --input       Gerçek veri: video, görüntü ya da klasör (varsayılan kolay video). [DEĞİŞTİR]
    --calibration, --perspective, --scale
                  Başka gereksinimlerden: 1.1, 1.2 ve 1.3'ün kurulum dosyaları. Videonun kamerasına
                  uygun kurulumun klasöründekiler verilir; zor video için kurulum_dosyalari/zor_video/
                  içindeki kalibrasyon.npz, perspektif.npz ve olcek.npz. [DEĞİŞTİR]
    --model, --iou, --device
                  5.1'in modeli, 5.4'ün seçtiği iou ve YOLO'nun cihazı (auto: GPU ve CPU kısa bir ölçümle
                  karşılaştırılır, hızlı olan seçilir; GPU yoksa CPU).
    --smooth, --lane-width-range, --combine, --blur-target, --no-obstacles, --debug
                  Ayar: kararlılık ve maske ayarları, engel kolunu kapatma (varsayılanlar uygun).
                  --lane-width-range verilmezse ölçek dosyasındaki gerçek şerit genişliğinin ±%20'si.
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.dont_write_bytecode = True  # kod klasörlerinde yalnız kod kalsın (kural 1); moduller'den önce
import moduller  # noqa: E402

# =============================================================================
# AYARLAR
# Komut satırından argüman verilmezse bu değerler kullanılır.
# [DEĞİŞTİR] ile işaretlenenleri kendi verine göre ayarlaman gerekir.
# =============================================================================
PROJECT_ROOT = moduller.PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"

# [DEĞİŞTİR] Girdi: video, tek görüntü ya da görüntü klasörü.
INPUT_PATH = DATA_DIR / "ortam_verisi" / "videolar" / "kolay_video" / "kolay_video_iyi_isik.mp4"

# Kurulum dosyaları: 1.1 calibrate_camera.py, 1.2 compute_perspective.py, 1.3 piksel_metre_olcekleme.py üretir.
CALIBRATION_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "kalibrasyon.npz"
PERSPECTIVE_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "perspektif.npz"
SCALE_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "olcek.npz"

# 4.4 birleştirme modu: "or" (doküman: buradan başla) ya da "and" (gölgede sahte piksel çoğalırsa).
COMBINE_MODE = "or"

# 4.1 bulanıklaştırmasının uygulandığı dal: "gradient" (yalnız 4.3'e) ya da "both" (4.2 ve 4.3'e).
BLUR_TARGET = "gradient"

# Ortalaması alınacak kare sayısı (doküman: son 5-10 kare).
SMOOTH_FRAMES = 8

# Kabul edilen şerit genişliği (m): None ise kurulumda (1.3) girilen gerçek genişliğin ±LANE_WIDTH_TOLERANCE'ı
# (3.66 m için 2.93–4.39 m); böylece başka ölçüdeki yolda da doğru çalışır. Sabit aralık için (en az, en çok) yaz.
LANE_WIDTH_RANGE_M = None
LANE_WIDTH_TOLERANCE = 0.20

# Engel hattı (5.1–5.5): YOLO ile tespit, şeritle ilişkilendirme, mesafe ve risk.
OBSTACLES = True
MODEL_PATH = DATA_DIR / "hazir_veri" / "yolo_modeli" / "yolo11n.pt"
# NMS IoU eşiği (5.4'ün analiziyle seçilir; ultralytics varsayılanı 0.7).
YOLO_IOU = 0.7
# YOLO'nun çalışacağı cihaz: "auto" (5.1 GPU ile CPU'yu ölçüp hızlı olanı seçer), "cpu" ya da GPU no ("0").
YOLO_DEVICE = "auto"

# Çıktı klasörü ve video için en çok kaç kare işleneceği (None: tümü).
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "entegrasyon"
MAX_FRAMES = None

# DEBUG: köşede şerit maskesi (4.4) ve kayan pencere görüntüsü; SAVE_COMPARISON: girdi ve çıktı yan yana.
DEBUG = False
SAVE_COMPARISON = False
# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
# Maske adımı 4.1 blur -> 4.2 renk + 4.3 gradyan -> 4.4 birleştirme ile yapılır.
REQUIREMENTS = ("1.1", "1.2", "1.3", "2.3", "2.4", "2.5", "3.1", "3.3", "4.1", "4.2", "4.3", "4.4")
OBSTACLE_REQUIREMENTS = ("5.1", "5.2", "5.3", "5.5")
OBSTACLE_COLORS = {"seritte": (0, 0, 255), "disinda": (0, 200, 0)}   # doküman: şeritteki kırmızı, diğerleri yeşil
OTHER_OBSTACLE_COLOR = (160, 160, 160)                               # bölge dışı / belirsiz

LANE_COLOR = (0, 200, 0)
LEFT_COLOR = (0, 0, 255)
RIGHT_COLOR = (255, 0, 0)
WARNING_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
OFFSET_ASCII = {"sağa": "saga", "sola": "sola", "ortada": "ortada"}
CURVE_ASCII = {"sag": "saga viraj", "sol": "sola viraj", "duz": "duz yol", "belirsiz": "belirsiz", "yok": "-"}
CSV_COLUMNS = ["kare", "durum", "neden", "sol_piksel", "sag_piksel", "serit_genisligi_m", "sapma_m", "yon",
               "cikis_uyarisi", "egrilik_r_m", "viraj_yonu", "viraj_sinifi", "engel_sayisi", "seritteki_engel",
               "en_yakin_engel_m", "risk"]


@dataclass
class FrameResult:
    status: str = "yok"            # "olculdu", "gecmisten" ya da "yok"
    reason: str = ""
    left_count: int = 0
    right_count: int = 0
    lane_width_m: float = float("nan")
    offset: object = None          # 2.5'in LaneOffset sonucu
    radius_m: float = float("nan")
    curve_direction: str = "yok"   # 3.3: "sag", "sol", "duz", "belirsiz"
    curve_class: str = "yok"       # 3.1: "duz", "hafif", "sert", "belirsiz"
    obstacles: tuple = ()          # 5.x: her engel için sözlük (kutu, sınıf, güven, durum, mesafe, risk)

    def nearest_in_lane(self):
        """Şeritteki en yakın engel (yoksa None)."""
        in_lane = [o for o in self.obstacles if o["status"] == "seritte" and np.isfinite(o["distance_m"])]
        return min(in_lane, key=lambda o: o["distance_m"]) if in_lane else None


def lane_width_range(scale_path, tolerance=LANE_WIDTH_TOLERANCE):
    """[EK] Kabul edilen şerit genişliği: ölçek dosyasındaki gerçek genişliğin (1.3'te girilen) ±tolerance'ı."""
    with np.load(scale_path) as data:
        if "lane_width_m" not in data.files:
            raise ValueError(f"Ölçek dosyasında gerçek şerit genişliği (lane_width_m) yok: {scale_path}\n"
                             "1.3'ü yeniden çalıştır ya da --lane-width-range ile aralığı ver.")
        width = float(data["lane_width_m"])
    return round(width * (1 - tolerance), 3), round(width * (1 + tolerance), 3)


class LanePipeline:
    """Ana hat: kurulum dosyalarını bir kez yükler; her karede process_frame çağrılır."""

    def __init__(self, calibration_path, perspective_path, scale_path, smooth_frames=SMOOTH_FRAMES,
                 lane_width_range_m=LANE_WIDTH_RANGE_M, combine_mode=COMBINE_MODE, blur_target=BLUR_TARGET,
                 obstacles=OBSTACLES, model_path=MODEL_PATH, yolo_iou=YOLO_IOU, yolo_device=YOLO_DEVICE):
        if combine_mode not in ("or", "and"):
            raise ValueError(f"Birleştirme modu or ya da and olmalı, verilen: {combine_mode}")
        needed = REQUIREMENTS + (OBSTACLE_REQUIREMENTS if obstacles else ())
        self.m = {requirement: moduller.load(requirement) for requirement in needed}
        if blur_target not in ("gradient", "both"):
            raise ValueError(f"Blur dalı gradient ya da both olmalı, verilen: {blur_target}")
        self.combine_mode, self.blur_target = combine_mode, blur_target
        # 4.1–4.4 kendi komut satırı varsayılanlarıyla kullanılır.
        self.args41, self.args42 = moduller.default_args("4.1"), moduller.default_args("4.2")
        self.args43, self.args44 = moduller.default_args("4.3"), moduller.default_args("4.4")
        self.args43.blur = "none"   # doküman: undistort -> blur (4.1, bir kez) -> renk/gradyan
        self.clahe = (cv2.createCLAHE(clipLimit=self.args42.clahe_clip,
                                      tileGridSize=(self.args42.clahe_tiles, self.args42.clahe_tiles))
                      if self.args42.clahe else None)
        self.undistorter = self.m["1.1"].Undistorter(calibration_path)          # 1.1: K, distorsiyon
        self.perspective = self.m["1.2"].PerspectiveTransformer(perspective_path)  # 1.2: M, Minv
        self.scale = self.m["1.3"].PixelMeterScale(scale_path)                  # 1.3: xm, ym

        # [EK] Kurulum dosyaları aynı çözünürlük için üretilmiş olmalı.
        width, height = self.perspective.image_size
        if self.undistorter.image_size != (width, height):
            raise ValueError(
                f"Kalibrasyon {self.undistorter.image_size[0]}x{self.undistorter.image_size[1]}, perspektif "
                f"{width}x{height} için üretilmiş.\nİkisini de videonun çözünürlüğünde yeniden üret."
            )
        self.scale.check_size(width, height)  # kuş bakışı görüntü orijinalle aynı boyutta
        self.size = (width, height)
        self.xm, self.ym = self.scale.xm_per_pix, self.scale.ym_per_pix

        self.window_search = self.m["2.3"].SlidingWindowSearch()
        self.offset_calculator = self.m["2.5"].LaneOffsetCalculator(self.size, self.xm)
        self.lane_width_range_m = (tuple(lane_width_range_m) if lane_width_range_m is not None
                                   else lane_width_range(scale_path))
        self.history = deque(maxlen=smooth_frames)     # doküman: son 5-10 karenin polinomları
        self.max_misses = 2 * smooth_frames
        self.misses = 0
        self.debug_views = None
        self.obstacles = obstacles
        if obstacles:
            self._init_obstacles(perspective_path, scale_path, model_path, yolo_iou, yolo_device)

    def _init_obstacles(self, perspective_path, scale_path, model_path, yolo_iou, yolo_device):
        """5.1 modeli ve 5.5'in yükleyicileri bir kez kurulur."""
        # Konteyner kullanıcısının ev klasörü yazılamaz; ultralytics ayarlarını geçici klasöre yazsın.
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        self.args51 = moduller.default_args("5.1")
        self.args51.model, self.args51.iou = Path(model_path), float(yolo_iou)
        self.args55 = moduller.default_args("5.5")
        try:
            self.model = self.m["5.1"].load_model(self.args51)                          # 5.1 (ultralytics)
            self.args51.device = self.m["5.1"].resolve_device(                          # GPU / CPU (ölçerek)
                yolo_device, lambda: self.m["5.1"].load_model(self.args51), self.args51.imgsz,
                model_name=Path(model_path).name)
            self.birdseye = self.m["5.5"].BirdsEyeTransform(perspective_path)           # 5.5: M
            self.scale55 = self.m["5.5"].PixelMeterScale(scale_path)                    # 5.5: ym, lane_width_px
        except Exception as exc:  # modüllerin kendi InputError'ı dahil: anlaşılır mesaj korunur
            raise ValueError(f"Engel hattı kurulamadı: {exc}") from exc
        if self.birdseye.image_size != self.size or self.birdseye.warped_size != self.size:
            raise ValueError("5.5'in perspektif boyutları ana hatla aynı olmalı.")
        filters = self.m["5.3"]
        self.keep_classes, self.class_conf = set(filters.CLASSES), dict(filters.CLASS_CONF)
        self.conf_threshold = filters.CONF_THRESHOLD

    def _detect_obstacles(self, undistorted, lane_fits):
        """5.1 tespit -> 5.3 filtre -> 5.2 alt-orta nokta -> 5.5 şeritte mi + mesafe + risk."""
        try:
            boxes, classes, confidences = self.m["5.1"].detect(self.model, undistorted, self.args51, None)  # 5.1
        except Exception as exc:
            raise ValueError(f"YOLO tespiti başarısız: {exc}") from exc
        keep, _, _ = self.m["5.3"].filter_mask(confidences, classes, self.conf_threshold,
                                               self.keep_classes, self.class_conf)         # 5.3
        boxes, classes, confidences = boxes[keep], classes[keep], confidences[keep]
        if len(boxes) == 0:
            return ()
        coords = self.m["5.2"].box_coordinates(boxes, self.size, self.m["5.2"].EDGE_MARGIN_PX)  # 5.2
        detections = {"bottom_center": coords["bottom_center"], "xyxy": coords["xyxy"],
                      "det_frame": np.zeros(len(boxes), np.int64), "bottom_truncated": coords["bottom_truncated"]}
        lanes = {0: lane_fits} if lane_fits is not None else {}
        relation = self.m["5.5"].relate(detections, lanes, self.birdseye, self.scale55, self.args55)  # 5.5
        return tuple(
            {"box": coords["xyxy"][i], "class_name": self.model.names[int(classes[i])],
             "confidence": float(confidences[i]), "status": str(relation["status"][i]),
             "distance_m": float(relation["distance_m"][i]), "risk": str(relation["risk"][i]),
             "reliable": bool(relation["distance_reliable"][i])}
            for i in range(len(boxes)))

    def reset(self):
        """Birbirinden bağımsız görüntüler arasında geçmişi sıfırlar."""
        self.history.clear()
        self.misses = 0

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, FrameResult]:
        undistorted = self.undistorter.undistort(frame)                                    # 1.1
        mask = self._mask(undistorted)                                                     # 4.1–4.4
        birds_eye_mask = self.perspective.warp(mask)                                       # 1.2 (M)
        lanes = self.window_search.find(birds_eye_mask)                                    # 2.3
        result = FrameResult(left_count=len(lanes.leftx), right_count=len(lanes.rightx))

        fits, reason = self._fit(lanes)                                                    # 2.4
        if fits is not None:
            accepted, reason = self._check_width(fits)
            if accepted:
                self.history.append(fits)
                self.misses = 0
                result.status = "olculdu"
        if result.status != "olculdu":
            result.reason = reason
            self.misses += 1
            # [EK] Uzun süre ölçüm yoksa eski şerit çizilmez.
            if self.misses > self.max_misses:
                self.history.clear()
            if self.history:
                result.status = "gecmisten"

        self.debug_views = (mask, lanes)
        lane_fits = None
        if self.history:
            # Doküman: son karelerin polinomlarının ortalaması titremeyi azaltır.
            lane_fits = (np.mean([f[0] for f in self.history], axis=0),
                         np.mean([f[1] for f in self.history], axis=0))
            self._measure(*lane_fits, result)                                              # 2.5, 3.3, 3.1
        if self.obstacles:
            result.obstacles = self._detect_obstacles(undistorted, lane_fits)              # 5.1–5.5

        if lane_fits is None:
            output = undistorted.copy()
            lines = [("Serit bulunamadi", WARNING_COLOR)]
        else:
            output, lines = self._draw_lane(undistorted, *lane_fits, result)
        if self.obstacles:
            lines.append(self._draw_obstacles(output, result))
        draw_lines(output, lines)
        return output, result

    def _mask(self, undistorted):
        """İkili şerit maskesi (0/255, kamera görüntüsünde). Doküman: maske undistort edilmiş
        karede üretilir, sonra M ile kuş bakışına çevrilir."""
        smooth = self.m["4.1"].smooth(undistorted, self.args41)                            # 4.1
        # [EK] Blur yalnız gradyan dalında (varsayılan): ince/soluk çizginin renk tepesini eşiğin
        # altına itiyor (zor videoda ölçülen kare %61'e karşı %79), gradyanı ise temizliyor.
        color_input = smooth if self.blur_target == "both" else undistorted
        channels = self.m["4.2"].channel_images(color_input, self.clahe)
        _, color = self.m["4.2"].channel_masks(channels, self.args42)                      # 4.2 (L, S, B)
        _, _, gradient = self.m["4.3"].edge_mask(smooth, self.args43)                      # 4.3 (2.2 + çift kenar)
        combined = self.m["4.4"].combine(color > 0, gradient > 0, self.combine_mode)      # 4.4
        combined = self.m["4.4"].clean(combined, self.args44.morph, self.args44.morph_ksize, self.args44.morph_shape)
        return combined.astype(np.uint8) * 255

    def _fit(self, lanes):
        """2.4: her şerit için np.polyfit(y, x, 2); uydurulamazsa nedeni döner."""
        polinom = self.m["2.4"]
        fits = []
        for name, x, y in (("sol", lanes.leftx, lanes.lefty), ("sağ", lanes.rightx, lanes.righty)):
            fit = polinom.fit_lane(x.astype(np.float64), y.astype(np.float64), self.size[1],
                                   polinom.MIN_FIT_PIXELS, polinom.MIN_Y_SPAN)
            if fit["fit"] is None:
                return None, f"{name} şerit uydurulamadı: {fit['reason']}"
            fits.append(np.asarray(fit["fit"], dtype=np.float64))
        return tuple(fits), ""

    def _check_width(self, fits):
        """Doküman: genişlik makul mü. [EK] Alt, orta ve üst satırda bakılır (paralellik)."""
        height = self.size[1]
        ys = np.array([height - 1, height / 2, 0.0])
        widths = (np.polyval(fits[1], ys) - np.polyval(fits[0], ys)) * self.xm
        low, high = self.lane_width_range_m
        if np.all((widths >= low) & (widths <= high)):
            return True, ""
        return False, (f"şerit genişliği makul değil (alt/orta/üst {widths[0]:.2f}/{widths[1]:.2f}/"
                       f"{widths[2]:.2f} m; kabul {low}-{high} m)")

    def _measure(self, left, right, result):
        height = self.size[1]
        result.lane_width_m = float((np.polyval(right, height - 1) - np.polyval(left, height - 1)) * self.xm)
        try:
            result.offset = self.offset_calculator.compute(left, right)                     # 2.5
        except ValueError as exc:
            result.reason = str(exc)

        egrilik, katsayi = self.m["3.3"], self.m["3.1"]
        fits_m = {"left": egrilik.pixel_fit_to_meters(left, self.xm, self.ym),
                  "right": egrilik.pixel_fit_to_meters(right, self.xm, self.ym)}
        y_eval_m = (height - 1) * self.ym                                                  # araca en yakın satır
        radii = {side: float(egrilik.radius_of_curvature(fit[0], fit[1], y_eval_m)) for side, fit in fits_m.items()}
        direction, radius, _ = egrilik.classify(fits_m, radii, egrilik.STRAIGHT_MAX_A, egrilik.LR_RATIO_WARN)  # 3.3
        a_m = {"left": float(katsayi.a_to_meters(left[0], self.xm, self.ym)),
               "right": float(katsayi.a_to_meters(right[0], self.xm, self.ym))}
        category, _, _, _ = katsayi.interpret(a_m, katsayi.STRAIGHT_MAX_A, katsayi.SHARP_MIN_A,
                                              katsayi.LR_RATIO_WARN)                       # 3.1
        result.radius_m, result.curve_direction, result.curve_class = float(radius), direction, category

    def _draw_obstacles(self, output, result):
        """Doküman (Sıra 16): şeritteki engeller kırmızı, dışındakiler yeşil. Dönüş: özet satırı."""
        scale = output.shape[1] / 1280
        for obstacle in result.obstacles:
            x1, y1, x2, y2 = (int(round(v)) for v in obstacle["box"])
            color = OBSTACLE_COLORS.get(obstacle["status"], OTHER_OBSTACLE_COLOR)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness=max(round(3 * scale), 2))
            label = obstacle["class_name"]
            if np.isfinite(obstacle["distance_m"]):
                label += f" {obstacle['distance_m']:.0f} m"
            cv2.putText(output, label, (x1, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, color,
                        thickness=max(round(2 * scale), 1), lineType=cv2.LINE_AA)
        nearest = result.nearest_in_lane()
        if nearest is None and result.status == "yok" and result.obstacles:
            # Şerit bilinmeden "şeritte yok" denemez; 5.5 bu nesneleri "belirsiz" sayar.
            return (f"Engel: {len(result.obstacles)} nesne, serit bilinmiyor (konum belirsiz)", (0, 200, 255))
        if nearest is None:
            return (f"Engel: seritte yok ({len(result.obstacles)} nesne)", TEXT_COLOR)
        risk_color = {"kirmizi": WARNING_COLOR, "sari": (0, 220, 255)}.get(nearest["risk"], (0, 200, 0))
        return (f"Engel: seritte {nearest['class_name']} {nearest['distance_m']:.0f} m (risk: {nearest['risk']})",
                risk_color)

    def _draw_lane(self, undistorted, left, right, result):
        """Doküman: şerit alanını yeşil boya, Minv ile orijinale taşı, addWeighted ile bindir, yaz."""
        height = self.size[1]
        ys = np.linspace(0, height - 1, 72)
        left_xs, right_xs = np.polyval(left, ys), np.polyval(right, ys)
        layer = np.zeros_like(undistorted)
        polygon = np.vstack([np.column_stack([left_xs, ys]), np.column_stack([right_xs, ys])[::-1]])
        cv2.fillPoly(layer, [polygon.round().astype(np.int32)], LANE_COLOR)
        for xs, color in ((left_xs, LEFT_COLOR), (right_xs, RIGHT_COLOR)):
            cv2.polylines(layer, [np.column_stack([xs, ys]).round().astype(np.int32)], False, color,
                          thickness=14, lineType=cv2.LINE_AA)
        output = cv2.addWeighted(undistorted, 1.0, self.perspective.unwarp(layer), 0.4, 0)  # 1.2 (Minv)

        lines = []
        offset = result.offset
        if offset is not None:
            text = ("Sapma: ortada" if offset.direction == "ortada"
                    else f"Sapma: {abs(offset.offset_m):.2f} m {OFFSET_ASCII[offset.direction]}")
            lines.append((f"{text} ({offset.offset_m:+.2f} m)", TEXT_COLOR))
        if result.curve_direction in ("sag", "sol"):
            lines.append((f"Egrilik: R = {result.radius_m:.0f} m, {CURVE_ASCII[result.curve_direction]} "
                          f"({result.curve_class})", TEXT_COLOR))
        else:
            lines.append((f"Egrilik: {CURVE_ASCII.get(result.curve_direction, '-')}", TEXT_COLOR))
        lines.append((f"Serit genisligi: {result.lane_width_m:.2f} m", TEXT_COLOR))
        if result.status == "gecmisten":
            lines.append(("Olcum: onceki kareler (bu kare reddedildi)", (0, 200, 255)))
        if offset is not None and offset.departure_warning:
            lines.append(("UYARI: seritten cikma riski", WARNING_COLOR))
        return output, lines


def draw_lines(image, lines):
    """Sol üstte yarı saydam zemin üzerine yazı (cv2.putText; ASCII, kural 21)."""
    scale = image.shape[1] / 1280
    height = round((24 + 38 * len(lines)) * scale)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (round(760 * scale), height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, dst=image)
    for row, (text, color) in enumerate(lines):
        cv2.putText(image, text, (round(18 * scale), round((42 + 38 * row) * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9 * scale, color, thickness=max(round(2 * scale), 1), lineType=cv2.LINE_AA)


# [EK] --debug: sağ üst köşede şerit maskesi (4.4) ve kayan pencere görüntüsü.
def add_debug_views(output, pipeline):
    mask, lanes = pipeline.debug_views
    height, width = output.shape[:2]
    inset_w, inset_h = width // 4, height // 4
    windows = pipeline.m["2.3"].draw_windows(lanes)
    for row, view in enumerate((cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), windows)):
        small = cv2.resize(view, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
        y0 = row * inset_h
        output[y0:y0 + inset_h, width - inset_w:] = small
        cv2.rectangle(output, (width - inset_w, y0), (width - 1, y0 + inset_h - 1), (255, 255, 255), thickness=2)
    return output


def finish_frame(frame, output, pipeline, args):
    if args.debug:
        output = add_debug_views(output, pipeline)
    if args.compare:
        output = np.hstack([frame, output])
    return output


def result_row(name, result):
    offset, nearest = result.offset, result.nearest_in_lane()
    return [
        name, result.status, result.reason, result.left_count, result.right_count,
        "" if np.isnan(result.lane_width_m) else f"{result.lane_width_m:.3f}",
        "" if offset is None else f"{offset.offset_m:.4f}",
        "" if offset is None else offset.direction,
        "" if offset is None else ("evet" if offset.departure_warning else "hayır"),
        "" if np.isnan(result.radius_m) or np.isinf(result.radius_m) else f"{result.radius_m:.1f}",
        result.curve_direction, result.curve_class, len(result.obstacles),
        sum(o["status"] == "seritte" for o in result.obstacles),
        "" if nearest is None else f"{nearest['distance_m']:.1f}", "" if nearest is None else nearest["risk"],
    ]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Görüntü yazılamadı: {path}\nÇıktı klasörünün yazma iznini kontrol et ya da --output ver.")


def process_video(pipeline, path, args):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(
            f"Video açılamadı: {path}\n"
            "Dosya bozuk ya da desteklenmeyen bir biçimde olabilir; başka bir video dene."
        )
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    video_path = args.output / f"{path.stem}_islenmis.mp4"
    writer, rows, results, busy = None, [], [], 0.0
    try:
        index = 0
        while args.max_frames is None or index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            output, result = pipeline.process_frame(frame)
            busy += time.perf_counter() - start
            output = finish_frame(frame, output, pipeline, args)
            if writer is None:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (output.shape[1], output.shape[0]))
                if not writer.isOpened():
                    raise OSError(f"Çıktı videosu oluşturulamadı: {video_path}\nÇıktı klasörünün iznini kontrol et.")
            writer.write(output)
            rows.append(result_row(f"kare{index:05d}", result))
            results.append(result)
            index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not results:
        raise ValueError(f"Videodan hiç kare okunamadı: {path}\nVideo boş ya da bozuk olabilir.")
    csv_path = args.output / f"{path.stem}_islenmis.csv"
    write_csv(csv_path, rows)
    print_summary(path.name, results, busy, video_path, csv_path)


def process_images(pipeline, paths, args):
    rows, results, busy = [], [], 0.0
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"UYARI: {path.name} okunamadı, atlandı.", file=sys.stderr)
            continue
        pipeline.reset()  # bağımsız görüntüler arasında ortalama alınmaz
        start = time.perf_counter()
        output, result = pipeline.process_frame(frame)
        busy += time.perf_counter() - start
        write_image(args.output / f"{path.stem}_islenmis.jpg", finish_frame(frame, output, pipeline, args))
        rows.append(result_row(path.name, result))
        results.append(result)
        offset = result.offset
        print(f"  {path.name}: {result.status}"
              + (f" | sapma {offset.offset_m:+.2f} m ({offset.direction})" if offset else "")
              + (f" | R {result.radius_m:.0f} m {CURVE_ASCII[result.curve_direction]} ({result.curve_class})"
                 if result.curve_direction in ("sag", "sol") else f" | {CURVE_ASCII.get(result.curve_direction, '-')}")
              + (f" | genişlik {result.lane_width_m:.2f} m" if not np.isnan(result.lane_width_m) else "")
              + (f" | engel {len(result.obstacles)}, şeritte {sum(o['status'] == 'seritte' for o in result.obstacles)}"
                 if pipeline.obstacles else "")
              + (f" | {result.reason}" if result.reason else ""))
    if not results:
        raise ValueError("Hiç görüntü işlenemedi.")
    csv_path = args.output / "goruntuler_islenmis.csv"
    write_csv(csv_path, rows)
    print(f"{len(results)} görüntü işlendi: {args.output} | tablo: {csv_path.name}")


def print_summary(name, results, busy, video_path, csv_path):
    statuses = Counter(r.status for r in results)
    measured = [r for r in results if r.offset is not None]
    print(f"{name}: {len(results)} kare, işleme hızı {len(results) / busy:.1f} FPS")
    print(f"  Durum: bu karede ölçüldü {statuses['olculdu']}, önceki karelerden {statuses['gecmisten']}, "
          f"şerit yok {statuses['yok']}")
    if measured:
        offsets = np.array([r.offset.offset_m for r in measured])
        widths = np.array([r.lane_width_m for r in measured])
        curved = [r.radius_m for r in measured if r.curve_direction in ("sag", "sol")]
        directions = Counter(r.offset.direction for r in measured)
        print(f"  Yanal sapma: ort. {offsets.mean():+.2f} m, en küçük {offsets.min():+.2f}, en büyük "
              f"{offsets.max():+.2f} | sağa {directions['sağa']}, sola {directions['sola']}, ortada "
              f"{directions['ortada']} | çıkma uyarısı {sum(r.offset.departure_warning for r in measured)} kare")
        print(f"  Şerit genişliği: ort. {widths.mean():.2f} m (std {widths.std():.2f})")
        curve_counts = Counter(r.curve_direction for r in measured)
        print(f"  Eğrilik: virajlı {len(curved)} kare (ortanca R {np.median(curved):.0f} m)" if curved else
              "  Eğrilik: virajlı kare yok", end="")
        print(f" | düz {curve_counts['duz']}, sağa {curve_counts['sag']}, sola {curve_counts['sol']}, "
              f"belirsiz {curve_counts['belirsiz']}")
    if any(r.obstacles for r in results):
        nearest = [r.nearest_in_lane() for r in results]
        risks = Counter(n["risk"] for n in nearest if n is not None)
        print(f"  Engel: {sum(len(r.obstacles) for r in results)} tespit, şeritte engel olan kare "
              f"{sum(n is not None for n in nearest)} | risk kırmızı {risks['kirmizi']}, sarı {risks['sari']}, "
              f"yeşil {risks['yesil']} kare")
    reasons = Counter(r.reason.split(" (")[0].split(":")[0] for r in results if r.reason)
    for reason, count in reasons.most_common(3):
        print(f"  Reddedilme nedeni: {reason} ({count} kare)")
    print(f"  Video: {video_path}\n  Tablo: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Şerit ve engel hattını uçtan uca çalıştırır (process_frame).")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help=f"Video, görüntü ya da klasör (varsayılan: {INPUT_PATH})")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH, help="1.1 kalibrasyon dosyası")
    parser.add_argument("--perspective", type=Path, default=PERSPECTIVE_PATH, help="1.2 perspektif dosyası (M, Minv)")
    parser.add_argument("--scale", type=Path, default=SCALE_PATH, help="1.3 ölçek dosyası (xm, ym)")
    parser.add_argument("--combine", choices=("or", "and"), default=COMBINE_MODE,
                        help=f"4.4 birleştirme modu (varsayılan: {COMBINE_MODE})")
    parser.add_argument("--blur-target", choices=("gradient", "both"), default=BLUR_TARGET,
                        help=f"4.1 bulanıklaştırmasının uygulandığı dal (varsayılan: {BLUR_TARGET})")
    parser.add_argument("--no-obstacles", dest="obstacles", action="store_false", default=OBSTACLES,
                        help="Engel hattını (5.1–5.5) kapat")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help=f"YOLO ağırlıkları (varsayılan: {MODEL_PATH})")
    parser.add_argument("--iou", type=float, default=YOLO_IOU, help=f"NMS IoU eşiği, 5.4 (varsayılan: {YOLO_IOU})")
    parser.add_argument("--device", default=YOLO_DEVICE,
                        help=f"YOLO cihazı: auto (GPU ve CPU ölçülür, hızlı olan), cpu ya da GPU numarası "
                             f"(varsayılan: {YOLO_DEVICE})")
    parser.add_argument("--smooth", type=int, default=SMOOTH_FRAMES, help="Ortalaması alınacak kare sayısı")
    parser.add_argument("--lane-width-range", type=float, nargs=2, default=LANE_WIDTH_RANGE_M,
                        help="Kabul edilen şerit genişliği, m (en az, en çok); varsayılan: ölçek dosyasındaki "
                             f"gerçek genişliğin ±%%{LANE_WIDTH_TOLERANCE * 100:.0f}'si")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="Videoda en çok kaç kare işlensin")
    parser.add_argument("--debug", action="store_true", default=DEBUG, help="Köşede maske ve kayan pencere görüntüsü")
    parser.add_argument("--compare", action="store_true", default=SAVE_COMPARISON, help="Girdi ve çıktıyı yan yana kaydet")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth en az 1 olmalı.")
    if args.lane_width_range is not None and not 0 < args.lane_width_range[0] < args.lane_width_range[1]:
        parser.error("--lane-width-range: 0 < en az < en çok olmalı.")
    return args


def main():
    args = parse_args()
    try:
        if not args.input.exists():
            raise FileNotFoundError(f"Girdi bulunamadı: {args.input}\n--input ile bir video, görüntü ya da klasör ver.")
        suffix = args.input.suffix.lower()
        if args.input.is_file() and suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise ValueError(f"Desteklenmeyen dosya türü: {args.input.name}")
        if args.input.is_dir():
            paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
            if not paths:
                raise FileNotFoundError(f"Klasörde görüntü yok: {args.input}")
        pipeline = LanePipeline(args.calibration, args.perspective, args.scale, smooth_frames=args.smooth,
                                lane_width_range_m=args.lane_width_range, combine_mode=args.combine,
                                blur_target=args.blur_target, obstacles=args.obstacles, model_path=args.model,
                                yolo_iou=args.iou, yolo_device=args.device)
        low, high = pipeline.lane_width_range_m
        print(f"Kabul edilen şerit genişliği: {low:.2f}–{high:.2f} m")
        if pipeline.obstacles:
            print(f"YOLO cihazı: {pipeline.m['5.1'].device_label(pipeline.args51.device)}")
        args.output.mkdir(parents=True, exist_ok=True)
        if args.input.is_dir():
            process_images(pipeline, paths, args)
        elif suffix in VIDEO_EXTENSIONS:
            process_video(pipeline, args.input, args)
        else:
            process_images(pipeline, [args.input], args)
    except PermissionError as exc:
        print(f"HATA: Dosyaya erişim izni yok: {exc.filename}\nİzinleri kontrol et ya da --output ile başka bir klasör ver.",
              file=sys.stderr)
        return 1
    except cv2.error as exc:
        print(f"HATA: OpenCV işlemi tamamlanamadı: {getattr(exc, 'err', exc)}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
