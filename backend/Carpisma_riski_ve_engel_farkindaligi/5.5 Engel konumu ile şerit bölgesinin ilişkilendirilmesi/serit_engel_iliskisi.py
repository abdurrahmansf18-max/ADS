"""
5.5 Engel konumu ile şerit bölgesinin ilişkilendirilmesi — nesne bizim şeridimizde mi, ne kadar uzakta?

Ne işe yarar:
    Yan şeritteki araçla önümüzdeki engeli ayırır; yalnız kendi şeridimizdeki engel çarpışma riskidir.
    Şerit hattını (2.x) ve engel hattını (5.x) birleştiren adımdır.

Nasıl çalışır:
    Kutunun alt-orta noktası M ile kuş bakışına taşınır. Aynı yükseklikte sol ve sağ şeridin x'i
    polinomdan bulunur; nokta ikisinin arasındaysa nesne şeritte sayılır. Mesafe = (alt kenar - nokta)
    x ym; 10 m'den yakını kırmızı, 25 m'den yakını sarı risktir.

Bağlantılar:
    Alır : 5.3'ün filtreli tespitleri (alt-orta nokta), 2.4'ün polinomları (şerit sınırları),
           1.2'nin perspektif dosyası (M) ve 1.3'ün ölçeği (ym). Tespitler ve şeritler aynı videonun aynı
           karelerinden olmalıdır.
    Verir: <ad>_serit_iliskisi.npz (durum, mesafe, risk); ana hatta kutu rengi (şeritte kırmızı, dışında
           yeşil) ve risk yazısı.
    Ana hat: main.py her karede relate() çağırır.

Parametreler:
    --detections  Başka gereksinimden: 5.3'ün çıktısı (varsayılan kolay video).
    --lanes       Başka gereksinimden: 2.4'ün aynı video için polinomları.
    --perspective, --scale   Başka gereksinimden: 1.2 ve 1.3'ün kurulum dosyaları.
    --offset-m    Gerçek veri: kuş bakışının alt kenarı ile aracın önü arasındaki mesafe (ölçülmeli).
                  [DEĞİŞTİR]
    --red-m, --yellow-m   Ayar: risk sınırları (doküman örneği). [DEĞİŞTİR]
    --source, --calibration   Yalnız çizim için: kaynak video ve 1.1'in kalibrasyonu.
"""

import argparse
import csv
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
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUT_DIR = OUTPUTS / Path(__file__).resolve().parent.name

# [DEĞİŞTİR] Filtrelenmiş tespitler (5.3) ve şerit polinomları (2.4), aynı videodan.
DETECTIONS_PATH = OUTPUTS / "5.3 Confidence Score ve sınıf bilgisi" / "kolay_video_iyi_isik_filtreli.npz"
LANES_PATH = OUTPUTS / "2.4 Polinom uydurma" / "kolay_video_iyi_isik_undistorted_maske_warped_polinom.npz"

# [DEĞİŞTİR] Perspektif matrisi M (kamera -> kuş bakışı) ve 1.3'ün ölçek dosyası.
PERSPECTIVE_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "perspektif.npz"
SCALE_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "olcek.npz"

# [DEĞİŞTİR] Risk eşikleri (m), yalnız kendi şeridimizdeki nesne için (İnşa: örnek 10 / 25).
RED_M = 10.0
YELLOW_M = 25.0
OFFSET_M = 0.0                # [EK] kuş bakışı alt kenarından aracın önüne sabit uzaklık

LANE_WIDTH_FALLBACK = True    # [EK] tek şerit yoksa 1.3'teki şerit genişliğiyle tahmin et

# [DEĞİŞTİR] Yalnız --compare çizimi için: 5.1'e verilen kaynak ve (varsa) kalibrasyon.
SOURCE_PATH = DATA_DIR / "ortam_verisi" / "videolar" / "kolay_video" / "kolay_video_iyi_isik.mp4"
CALIBRATION_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "kalibrasyon.npz"
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
DETECTION_KEYS = ("det_frame", "boxes", "xyxy", "xywh", "xyxyn", "xywhn", "bottom_center", "area",
                  "truncated", "bottom_truncated", "class_id", "class_name", "confidence", "kept_index",
                  "risk_group")
RISK_ORDER = {"yok": 0, "yesil": 1, "sari": 2, "kirmizi": 3}
RISK_BGR = {"kirmizi": (0, 0, 255), "sari": (0, 220, 255), "yesil": (0, 200, 0), "yok": (200, 200, 200)}


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


def load_npz(path, required, what, hint):
    path = Path(path)
    if not path.is_file():
        raise InputError(f"{what} bulunamadı: {path}\n  -> {hint}")
    try:
        with np.load(path) as data:
            missing = [k for k in required if k not in data.files]
            if missing:
                raise ValueError(f"eksik alanlar: {', '.join(missing)}")
            return {k: data[k] for k in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise InputError(f"{what} okunamadı: {path} ({exc})\n  -> {hint}") from None


def size_of(arr):
    return tuple(int(v) for v in np.asarray(arr).ravel())


# [EK] M'yi bir kez yükleyen sınıf (kural 18); kaynak / hedef boyutlarını bilir.
class BirdsEyeTransform:
    """Kamera görüntüsündeki noktaları kuş bakışına taşır (cv2.perspectiveTransform)."""

    def __init__(self, path):
        data = load_npz(path, ("M", "image_size", "warped_size"), "Perspektif dosyası",
                        "Perspektif matrisini M, image_size, warped_size alanlarıyla .npz olarak ver "
                        "(--perspective).")
        self.M = np.asarray(data["M"], dtype=np.float64)
        if self.M.shape != (3, 3) or not np.isfinite(self.M).all():
            raise InputError(f"M 3x3 sonlu bir matris olmalı, verilen {self.M.shape}: {path}")
        self.image_size, self.warped_size = size_of(data["image_size"]), size_of(data["warped_size"])
        self.path = Path(path)

    def to_birdseye(self, points):
        """Gereksinimin özü: cv2.perspectiveTransform (M x 2 px -> M x 2 px)."""
        if len(points) == 0:
            return np.zeros((0, 2))
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)


# [EK] 1.3'teki yükleyicinin kopyası (klasörler arası import yapılmaz).
class PixelMeterScale:
    """Kaydedilmiş piksel–metre ölçeğini yükler."""

    def __init__(self, path):
        data = load_npz(path, ("xm_per_pix", "ym_per_pix", "image_size"), "Ölçek dosyası",
                        "Önce 1.3 piksel_metre_olcekleme.py ile ölçeği hesapla (--scale).")
        self.xm_per_pix, self.ym_per_pix = float(data["xm_per_pix"]), float(data["ym_per_pix"])
        self.image_size = size_of(data["image_size"])
        self.lane_width_px = float(data["lane_width_px"]) if "lane_width_px" in data else math.nan


# [EK] 2.2'deki kalibrasyon yükleyicisinin kopyası (yalnız çizim için).
class CameraCalibration:
    def __init__(self, path):
        data = load_npz(path, ("camera_matrix", "dist_coeffs", "image_size"), "Kalibrasyon dosyası",
                        "5.1'de kullanılan kalibrasyonu --calibration ile ver.")
        self.camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).ravel()
        self.image_size = size_of(data["image_size"])

    def undistort(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.image_size:
            raise InputError(f"Kalibrasyon {self.image_size[0]}x{self.image_size[1]} için, kare "
                             f"{frame.shape[1]}x{frame.shape[0]}.")
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)


# --------------------------------------------------------------------------- ilişkilendirme
def lane_bounds(left_fit, right_fit, y, lane_width_px, fallback):
    """Aynı y'de sol / sağ şerit x'i (np.polyval). [EK] Tek şerit yoksa genişlikten tahmin."""
    xl = np.polyval(left_fit, y) if np.isfinite(left_fit).all() else np.nan
    xr = np.polyval(right_fit, y) if np.isfinite(right_fit).all() else np.nan
    estimated = False
    if fallback and math.isfinite(lane_width_px):
        if math.isnan(xl) and not math.isnan(xr):
            xl, estimated = xr - lane_width_px, True
        elif math.isnan(xr) and not math.isnan(xl):
            xr, estimated = xl + lane_width_px, True
    return xl, xr, estimated


def relate(det, lanes_by_frame, birdseye, scale, args):
    """Her tespit için kuş bakışı nokta, şerit durumu, örtüşme, mesafe ve risk."""
    bottom = det["bottom_center"].astype(np.float64)
    xyxy = det["xyxy"].astype(np.float64)
    bev = birdseye.to_birdseye(bottom)
    # [EK] Alt kenarın iki ucu (şerit örtüşmesi için).
    bev_l = birdseye.to_birdseye(np.stack([xyxy[:, 0], xyxy[:, 3]], axis=1))
    bev_r = birdseye.to_birdseye(np.stack([xyxy[:, 2], xyxy[:, 3]], axis=1))
    wb, hb = birdseye.warped_size
    m = len(bottom)
    out = {"bev_point": bev, "lane_left_x": np.full(m, np.nan), "lane_right_x": np.full(m, np.nan),
           "lane_estimated": np.zeros(m, bool), "status": np.empty(m, dtype="<U10"),
           "lane_overlap": np.zeros(m), "distance_m": np.full(m, np.nan),
           "distance_reliable": np.zeros(m, bool), "risk": np.full(m, "yok", dtype="<U7")}
    for i in range(m):
        x, y = bev[i]
        inside_region = 0 <= x < wb and 0 <= y < hb
        if 0 <= y < hb:
            # Mesafe: (kuş bakışı yüksekliği - y) x ym_per_pix (+ sabit pay).
            out["distance_m"][i] = (hb - y) * scale.ym_per_pix + args.offset_m
        fits = lanes_by_frame.get(int(det["det_frame"][i]))
        if not (0 <= y < hb):
            out["status"][i] = "bolge_disi"           # [EK] polinom ve mesafe dış değerleme olurdu
            continue
        if fits is None:
            out["status"][i] = "belirsiz"
            continue
        xl, xr, est = lane_bounds(fits[0], fits[1], y, scale.lane_width_px, args.lane_width_fallback)
        out["lane_left_x"][i], out["lane_right_x"][i], out["lane_estimated"][i] = xl, xr, est
        if math.isnan(xl) or math.isnan(xr):
            out["status"][i] = "belirsiz"
            continue
        # Gereksinimin özü: aynı y'de x_sol < x < x_sağ ise şeritte.
        out["status"][i] = "seritte" if xl < x < xr else "disinda"
        a, b = sorted((bev_l[i, 0], bev_r[i, 0]))
        out["lane_overlap"][i] = max(0.0, min(b, xr) - max(a, xl)) / (b - a) if b > a else float(xl < x < xr)
        truncated = bool(det["bottom_truncated"][i]) if "bottom_truncated" in det else False
        out["distance_reliable"][i] = inside_region and not truncated
        if out["status"][i] == "seritte":
            d = out["distance_m"][i]
            out["risk"][i] = "kirmizi" if d < args.red_m else ("sari" if d < args.yellow_m else "yesil")
    return out


# --------------------------------------------------------------------------- girdi hazırlığı
def load_inputs(args):
    det = load_npz(args.detections, ("frame", "det_frame", "class_name", "confidence", "image_size"),
                   "Tespit dosyası", "Önce 5.1-5.3'ü çalıştır ya da --detections ile <ad>_filtreli.npz ver.")
    m = len(det["class_name"])
    if "xyxy" not in det:
        if "boxes" not in det:
            raise InputError(f"Tespit dosyasında kutu yok (xyxy / boxes): {args.detections}")
        det["xyxy"] = np.asarray(det["boxes"], dtype=np.float64).reshape(-1, 4)
    if "bottom_center" not in det:          # 5.2'siz girdi: alt-orta noktayı burada hesapla
        b = det["xyxy"]
        det["bottom_center"] = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)
    bad = {k: len(det[k]) for k in DETECTION_KEYS if k in det and len(det[k]) != m}
    if bad:
        raise InputError(f"Tespit dizilerinin uzunlukları tutarsız (tespit {m}, {bad}): {args.detections}")
    lanes = load_npz(args.lanes, ("left_fit", "right_fit", "image_size"), "Şerit dosyası",
                     "Önce 2.4 polinom_uydurma.py'yi çalıştır ya da --lanes ile <ad>_polinom.npz ver.")
    birdseye = BirdsEyeTransform(args.perspective)
    scale = PixelMeterScale(args.scale)
    # [EK] Boyut denetimleri (kural 17): pikseller ancak aynı çözünürlükte anlamlıdır.
    checks = (("tespit görüntüsü", size_of(det["image_size"]), "M'nin kaynağı (image_size)", birdseye.image_size),
              ("şerit polinomlarının görüntüsü", size_of(lanes["image_size"]), "M'nin hedefi (warped_size)",
               birdseye.warped_size),
              ("ölçek (1.3)", scale.image_size, "M'nin hedefi (warped_size)", birdseye.warped_size))
    for a_name, a, b_name, b in checks:
        if a != b:
            raise InputError(f"Boyut uyuşmuyor: {a_name} {a[0]}x{a[1]}, {b_name} {b[0]}x{b[1]}.\n"
                             "  -> Tespitler kamera (undistort) karesinde, şeritler ve ölçek kuş bakışında, "
                             "M ikisi arasında olmalı.")
    return det, lanes, birdseye, scale


def match_frames(det, lanes):
    """Tespit karesi -> (sol, sağ) polinom. Adla; tek kare - tek kare ise doğrudan."""
    det_names = [str(n) for n in det["frame"]]
    left, right = np.atleast_2d(lanes["left_fit"]), np.atleast_2d(lanes["right_fit"])
    lane_names = [str(n) for n in lanes["frame"]] if "frame" in lanes else [str(i) for i in range(len(left))]
    by_name = {n: i for i, n in enumerate(lane_names)}
    result = {}
    for i, n in enumerate(det_names):
        if n in by_name:
            result[i] = (left[by_name[n]], right[by_name[n]])
    if not result and len(det_names) == 1 and len(lane_names) == 1:
        result[0] = (left[0], right[0])
    if not result:
        raise InputError(
            "Tespit ve şerit dosyalarının kare adları eşleşmiyor "
            f"(tespit: {', '.join(det_names[:3])}..., şerit: {', '.join(lane_names[:3])}...).\n"
            "  -> 5.1 ve 2.4 aynı videodan (aynı kare numaralarıyla) üretilmeli.")
    return result, det_names


# --------------------------------------------------------------------------- görsel
def source_frames(source, names):
    source = Path(source)
    if not source.exists():
        raise InputError(f"Kaynak bulunamadı: {source}\n  -> --source ile 5.1'e verilen kaynağı ver.")
    if source.is_dir():
        files = {p.stem: p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        for name in names:
            if name not in files:
                raise InputError(f"Kaynak klasörde '{name}' karesi yok: {source}")
            yield name, cv2.imread(str(files[name]), flags=cv2.IMREAD_COLOR)
        return
    if source.suffix.lower() not in VIDEO_EXTS:
        if len(names) != 1:
            raise InputError(f"Kaynak tek görüntü ama tespit dosyasında {len(names)} kare var.")
        yield names[0], cv2.imread(str(source), flags=cv2.IMREAD_COLOR)
        return
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise InputError(f"Kaynak video açılamadı: {source}")
    try:
        wanted = {int(n[4:]): n for n in names if n.startswith("kare") and n[4:].isdigit()}
        if len(wanted) != len(names):
            raise InputError("Kaynak video ama kare adları 'kareNNNNN' biçiminde değil.")
        index, last = 0, max(wanted)
        while index <= last:
            ok, frame = cap.read()
            if not ok:
                raise InputError(f"Kaynak videoda {index}. kare yok.")
            if index in wanted:
                yield wanted[index], frame
            index += 1
    finally:
        cap.release()


def put(img, text, org, color, fs, th):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thickness=th + 2, lineType=cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, thickness=th, lineType=cv2.LINE_AA)


def draw_compare(frame, sel, det, rel, fits, birdseye, scale, args):
    """Solda kamera karesi (kutular risk renginde), sağda kuş bakışı (şerit eğrileri, noktalar)."""
    width = frame.shape[1]
    fs = 0.6 * max(width, 640) / 1280.0 + 0.2
    th = max(2, round(2 * width / 1280.0))
    cam = frame.copy()
    wb, hb = birdseye.warped_size
    bev = cv2.warpPerspective(frame, birdseye.M, (wb, hb), flags=cv2.INTER_LINEAR)   # [EK] arka plan
    bev = (bev * 0.6).astype(np.uint8)
    for d_m, label in ((args.red_m, "kirmizi"), (args.yellow_m, "sari")):           # risk sınırları
        y = int(round(hb - (d_m - args.offset_m) / scale.ym_per_pix))
        if 0 <= y < hb:
            cv2.line(bev, (0, y), (wb - 1, y), RISK_BGR[label], thickness=1, lineType=cv2.LINE_AA)
            put(bev, f"{d_m:g} m", (5, max(15, y - 5)), RISK_BGR[label], fs * 0.8, max(1, th - 1))
    if fits is not None:
        ys = np.arange(hb, dtype=np.float64)
        for fit in fits:
            if np.isfinite(fit).all():
                pts = np.column_stack([np.polyval(fit, ys), ys]).round().astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(bev, [pts], False, (255, 200, 0), thickness=th, lineType=cv2.LINE_AA)
    for i in sel:
        status, risk = str(rel["status"][i]), str(rel["risk"][i])
        color = RISK_BGR[risk] if status == "seritte" else (200, 200, 200)
        x1, y1, x2, y2 = (int(round(v)) for v in det["xyxy"][i])
        cv2.rectangle(cam, (x1, y1), (x2, y2), color, thickness=th)
        d = rel["distance_m"][i]
        text = f"{det['class_name'][i]} {status}" + (f" {d:.1f}m" if math.isfinite(d) else "")
        put(cam, text, (x1 + 2, max(15, y1 - 6)), color, fs, th)
        bx, by = (int(round(v)) for v in det["bottom_center"][i])
        cv2.circle(cam, (bx, by), 6, color, thickness=-1)
        qx, qy = rel["bev_point"][i]
        if 0 <= qx < wb and 0 <= qy < hb:
            cv2.circle(bev, (int(round(qx)), int(round(qy))), 8, color, thickness=-1)
            put(bev, f"{d:.1f}m", (int(qx) + 10, int(qy)), color, fs, th)
    put(cam, "Kamera: kutu rengi = risk (gri: seritte degil)", (10, round(30 * fs / 0.6)), (0, 165, 255), fs, th)
    put(bev, "Kus bakisi: serit egrileri, alt-orta noktalar", (10, round(30 * fs / 0.6)), (0, 165, 255), fs, th)
    if bev.shape[0] != cam.shape[0]:
        scale_f = cam.shape[0] / bev.shape[0]
        bev = cv2.resize(bev, (max(1, round(bev.shape[1] * scale_f)), cam.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([cam, bev])


# --------------------------------------------------------------------------- ana akış
def run(args):
    det, lanes, birdseye, scale = load_inputs(args)
    lanes_by_frame, names = match_frames(det, lanes)
    rel = relate(det, lanes_by_frame, birdseye, scale, args)

    n = len(names)
    frame_risk, nearest = np.full(n, "yok", dtype="<U7"), np.full(n, np.nan)
    for f in range(n):
        sel = np.flatnonzero((det["det_frame"] == f) & (rel["status"] == "seritte"))
        if len(sel):
            frame_risk[f] = max((str(rel["risk"][i]) for i in sel), key=RISK_ORDER.get)
            nearest[f] = np.nanmin(rel["distance_m"][sel])

    m = len(det["class_name"])
    counts = {s: int((rel["status"] == s).sum()) for s in ("seritte", "disinda", "bolge_disi", "belirsiz")}
    print(f"{n} kare ({len(lanes_by_frame)} karede şerit var), {m} tespit: " +
          ", ".join(f"{k} {v}" for k, v in counts.items()) +
          f"; risk eşikleri < {args.red_m:g} m kırmızı, < {args.yellow_m:g} m sarı")
    for i in range(min(m, 10)):
        d = rel["distance_m"][i]
        extra = []
        if rel["lane_estimated"][i]:
            extra.append("bir sınır tahmini")
        if rel["status"][i] == "disinda" and rel["lane_overlap"][i] > 0:
            extra.append(f"alt kenarın %{100 * rel['lane_overlap'][i]:.0f}'i şeritte")
        if math.isfinite(d) and not rel["distance_reliable"][i]:
            extra.append("mesafe güvenilmez")
        print(f"  {names[det['det_frame'][i]]}: {det['class_name'][i]} {float(det['confidence'][i]):.2f} -> "
              f"{rel['status'][i]}" + (f", {d:.1f} m" if math.isfinite(d) else "") +
              f", risk {rel['risk'][i]}" + (f" ({'; '.join(extra)})" if extra else ""))
    if m > 10:
        print(f"  ... ({m - 10} tespit daha)")
    risky = {r: int((frame_risk == r).sum()) for r in ("kirmizi", "sari", "yesil")}
    print("Kare riski: " + ", ".join(f"{k} {v}" for k, v in risky.items()) + f", yok {int((frame_risk == 'yok').sum())}")

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"Çıktı klasörü oluşturulamadı: {args.out_dir} ({exc})") from None
    stem = Path(args.detections).stem
    for suffix in ("_filtreli", "_kutular", "_tespit"):
        stem = stem[:-len(suffix)] if stem.endswith(suffix) else stem
    out_path = args.out_dir / f"{stem}_serit_iliskisi.npz"
    payload = {**det, **rel, "frame_risk": frame_risk, "nearest_m": nearest,
               "lanes_found": np.array([f in lanes_by_frame for f in range(n)]),
               "red_m": np.float64(args.red_m), "yellow_m": np.float64(args.yellow_m), "offset_m": np.float64(args.offset_m)}
    try:
        np.savez(out_path, **payload)
        print(f"Kaydedildi: {out_path}")
        if args.csv:
            with open(args.out_dir / f"{stem}_serit_iliskisi.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["kare", "sinif", "guven", "alt_orta_x", "alt_orta_y", "kus_x", "kus_y", "serit_sol_x",
                            "serit_sag_x", "durum", "serit_ortusme", "mesafe_m", "mesafe_guvenilir", "risk"])
                for i in range(m):
                    w.writerow([names[det["det_frame"][i]], str(det["class_name"][i]), f"{float(det['confidence'][i]):.3f}",
                                *(f"{v:.1f}" for v in det["bottom_center"][i]), *(f"{v:.1f}" for v in rel["bev_point"][i]),
                                f"{rel['lane_left_x'][i]:.1f}", f"{rel['lane_right_x'][i]:.1f}", rel["status"][i],
                                f"{rel['lane_overlap'][i]:.2f}", f"{rel['distance_m'][i]:.2f}",
                                int(rel["distance_reliable"][i]), rel["risk"][i]])
            print(f"Kaydedildi: {args.out_dir / f'{stem}_serit_iliskisi.csv'}")
    except OSError as exc:
        raise InputError(f"Sonuç yazılamadı: {args.out_dir} ({exc})") from None

    if args.compare:
        if bool(det.get("undistorted", False)) and args.calibration is None:
            raise InputError("Tespitler düzeltilmiş (undistort) karede; çizim için kalibrasyonu --calibration ile ver.")
        calibration = CameraCalibration(args.calibration) if args.calibration is not None else None
        source = Path(args.source)
        is_video = source.is_file() and source.suffix.lower() in VIDEO_EXTS
        writer = None
        try:
            for name, frame in source_frames(source, names):
                if frame is None:
                    raise InputError(f"Kaynak kare okunamadı: {name}")
                if calibration is not None:
                    frame = calibration.undistort(frame)
                if (frame.shape[1], frame.shape[0]) != birdseye.image_size:
                    raise InputError(f"Kaynak kare {frame.shape[1]}x{frame.shape[0]}, M "
                                     f"{birdseye.image_size[0]}x{birdseye.image_size[1]} için.")
                f = names.index(name)
                img = draw_compare(frame, np.flatnonzero(det["det_frame"] == f), det, rel,
                                   lanes_by_frame.get(f), birdseye, scale, args)
                if is_video:
                    if writer is None:
                        writer = cv2.VideoWriter(str(args.out_dir / f"{stem}_serit_karsilastirma.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (img.shape[1], img.shape[0]),
                                                 isColor=True)
                    writer.write(img)
                else:
                    out_img = args.out_dir / (f"{stem}_serit_karsilastirma.png" if n == 1 else f"{name}_serit_karsilastirma.png")
                    if not cv2.imwrite(str(out_img), img):
                        raise InputError(f"Görsel yazılamadı: {out_img}")
        finally:
            if writer is not None:
                writer.release()
        print(f"Karşılaştırma görseli kaydedildi: {args.out_dir}")


class TurkishArgumentParser(argparse.ArgumentParser):
    """Argüman hatasında "HATA: ..." yazar ve 1 koduyla çıkar."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"HATA: Geçersiz komut satırı argümanı: {message}\n"
              "  -> Seçenekler için --help kullan.", file=sys.stderr)
        sys.exit(1)


def parse_args(argv=None):
    p = TurkishArgumentParser(
        description="Tespitlerin alt-orta noktasını kuş bakışına taşır; kendi şeridimizde mi, ne kadar uzakta, risk.")
    p.add_argument("--detections", type=Path, default=DETECTIONS_PATH,
                   help=f"5.3 filtreli tespitler (varsayılan: {DETECTIONS_PATH})")
    p.add_argument("--lanes", type=Path, default=LANES_PATH, help=f"2.4 şerit polinomları (varsayılan: {LANES_PATH})")
    p.add_argument("--perspective", type=Path, default=PERSPECTIVE_PATH,
                   help=f"Perspektif matrisi M (.npz) (varsayılan: {PERSPECTIVE_PATH})")
    p.add_argument("--scale", type=Path, default=SCALE_PATH, help=f"1.3 ölçek dosyası (varsayılan: {SCALE_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--red-m", type=float, default=RED_M, help=f"Kırmızı risk sınırı, m (varsayılan: {RED_M})")
    p.add_argument("--yellow-m", type=float, default=YELLOW_M, help=f"Sarı risk sınırı, m (varsayılan: {YELLOW_M})")
    p.add_argument("--offset-m", type=float, default=OFFSET_M,
                   help=f"Kuş bakışı alt kenarından aracın önüne sabit uzaklık, m (varsayılan: {OFFSET_M})")
    p.add_argument("--no-lane-width-fallback", dest="lane_width_fallback", action="store_false",
                   default=LANE_WIDTH_FALLBACK, help="Tek şerit yoksa genişlikten tahmin etme")
    p.add_argument("--csv", action="store_true", help="Tabloyu ayrıca CSV olarak kaydet")
    p.add_argument("--compare", action="store_true", help="Kamera karesi ve kuş bakışını yan yana çiz (--source gerekir)")
    p.add_argument("--source", type=Path, default=SOURCE_PATH, help=f"5.1'e verilen kaynak (varsayılan: {SOURCE_PATH})")
    p.add_argument("--calibration", type=Path, default=CALIBRATION_PATH, help="5.1 kalibrasyonla çalıştıysa aynı dosya")
    return p.parse_args(argv)


def validate_args(args):
    if not (math.isfinite(args.red_m) and math.isfinite(args.yellow_m) and 0 < args.red_m < args.yellow_m):
        raise InputError(f"0 < --red-m < --yellow-m olmalı, verilen: {args.red_m} / {args.yellow_m}")
    if not (math.isfinite(args.offset_m) and args.offset_m >= 0):
        raise InputError(f"--offset-m negatif olmayan bir sayı olmalı, verilen: {args.offset_m}")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    run(args)


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
