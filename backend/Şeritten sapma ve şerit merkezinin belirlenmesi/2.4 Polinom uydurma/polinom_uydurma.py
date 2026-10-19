"""
2.4 Polinom uydurma — şerit piksellerinden x = A·y² + B·y + C eğrisi.

Ne işe yarar:
    Dağınık şerit piksellerini tek, düzgün bir eğriye çevirir. Şerit merkezi (2.5), eğrilik (3.x) ve
    engelin şeritte olup olmadığı (5.5) bu formülle hesaplanır.

Nasıl çalışır:
    Sol ve sağ şerit pikselleri için np.polyfit(y, x, 2) ile en küçük kareler uydurulur. Çok az pikselli
    ya da çok kısa şeritte sonuç güvenilmez sayılır ve boş (NaN) bırakılır. Video ya da klasörde her kare
    ayrı uydurulur.

Bağlantılar:
    Alır : 2.3'ün <ad>_lane_pixels.npz dosyası (önerilen) ya da kuş bakışı maske; maske verilirse
           pikselleri kendi basit kayan penceresiyle ayırır.
    Verir: <ad>_polinom.npz (left_fit, right_fit = [A, B, C], frame) -> 2.5, 3.1, 3.3 ve 5.5.
    Ana hat: main.py her karede 2.3'ün pikselleriyle fit_lane() çağırır; son karelerin ortalaması alınır.

Parametreler:
    --input       Başka gereksinimden: 2.3'ün çıktısı (varsayılan sol_viraj test karesi) ya da 1.2'nin
                  kuş bakışı maske videosu.
    --min-fit-pixels, --min-y-span ve diğerleri   Ayar: güvenilirlik sınırları (varsayılanlar uygun).
"""

import argparse
import math
import sys
import warnings
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

# [DEĞİŞTİR] Kuş bakışı ikili maske (görüntü, klasör ya da video) ya da piksel listesi .npz.
# Varsayılan: 2.3'ün sol_viraj test karesinde bulduğu şerit pikselleri.
INPUT_PATH = PROJECT_ROOT / "outputs" / "2.3 Sliding Windows" / "sol_viraj_undistorted_maske_warped_lane_pixels.npz"

# Maske girdisi: gri değeri bundan büyük piksel şerit sayılır (sıkıştırılmış maskeler için).
MASK_THRESHOLD = 127
# İkili maske denetimi: ara tondaki (33-222) piksel oranı ya da doluluk bunu aşarsa
# girdi ikili maske sayılmaz (fotoğraf, ters çevrilmiş maske).
MAX_MIDTONE = 0.05
MAX_MASK_FILL = 0.5

# Kayan pencere (yalnızca maske girdisi).
N_WINDOWS = 9                 # dikeyde pencere sayısı
WINDOW_MARGIN = 0.08          # pencere yarı genişliği, görüntü genişliğinin oranı (1280'de ~100 px)
MIN_RECENTER_PX = 50          # pencerede bu kadar piksel varsa sonraki pencere ortalamaya kayar

# Uydurma güvenilirliği.
MIN_FIT_PIXELS = 100          # şerit başına en az piksel
MIN_Y_SPAN = 0.25             # piksellerin dikey açıklığı, görüntü yüksekliğinin oranı

# Video: en çok kaç kare işlensin (None = tümü).
MAX_FRAMES = None
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
SIDES = ("left", "right")
SIDE_TR = {"left": "sol", "right": "sağ"}


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


class FrameError(InputError):
    """Bir kare kullanılamadı; klasör / video girişinde o kare NaN olarak kaydedilir."""


# --------------------------------------------------------------------------- uydurma
def fit_lane(x, y, height, min_pixels, min_y_span):
    """np.polyfit(y, x, 2). Dönüş: {"fit": [A, B, C] ya da None, "rmse", "count", "reason"}."""
    result = {"fit": None, "rmse": math.nan, "count": int(len(x)), "reason": None}
    # [EK] Güvenilirlik denetimi: az piksel ya da kısa dikey açıklıkta A anlamsızdır.
    if len(x) < min_pixels:
        result["reason"] = f"yetersiz piksel ({len(x)} < {min_pixels})"
        return result
    span = float(y.max() - y.min())
    if span < min_y_span * height:
        result["reason"] = (f"dikey açıklık az ({span:.0f} px < yüksekliğin "
                            f"%{100 * min_y_span:.0f}'i)")
        return result
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.RankWarning)
        try:
            fit = np.polyfit(y, x, 2)
        except np.exceptions.RankWarning:
            result["reason"] = "kötü koşullu uydurma (RankWarning)"
            return result
    result["fit"] = fit
    # [EK] RMSE: piksellerin uydurulan eğriye ortalama yatay uzaklığı.
    result["rmse"] = float(np.sqrt(np.mean((np.polyval(fit, y) - x) ** 2)))
    return result


# --------------------------------------------------------------------------- piksel ayırma
def binarize(frame, threshold, max_midtone, max_fill):
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # [EK] İkili maske denetimi: maske neredeyse yalnız 0 ve 255'tir, fotoğrafta ara ton çoktur.
    midtone = np.count_nonzero((gray > 32) & (gray < 223)) / gray.size
    mask = gray > threshold
    fill = mask.mean()
    if midtone > max_midtone or fill > max_fill:
        raise FrameError(
            f"Girdi ikili maske gibi görünmüyor (ara ton %{100 * midtone:.1f}, "
            f"doluluk %{100 * fill:.1f}).\n"
            "  -> Bu modül kuş bakışı ikili maske ya da piksel listesi (.npz) bekler.")
    return mask


def find_lane_pixels(mask, n_windows, margin_frac, min_recenter):
    """[EK] Kayan pencere: alt yarının sütun histogramından başlayıp yukarı doğru izler.

    Dönüş: ({"left": (x, y), "right": (x, y)}, [(x0, y0, x1, y1), ...] pencereler).
    """
    height, width = mask.shape
    margin = max(1, round(margin_frac * width))
    ys, xs = np.nonzero(mask)
    bottom_hist = np.count_nonzero(mask[height // 2:], axis=0)
    full_hist = np.count_nonzero(mask, axis=0)
    edges = np.linspace(height, 0, n_windows + 1).round().astype(int)  # aşağıdan yukarı
    mid = width // 2
    lanes, windows = {}, []
    for side, lo, hi in (("left", 0, mid), ("right", mid, width)):
        # Alt yarıda piksel yoksa (ör. kesik çizginin boşluğu) tüm yükseklik kullanılır.
        hist = bottom_hist if bottom_hist[lo:hi].any() else full_hist
        if not hist[lo:hi].any():
            lanes[side] = (np.empty(0), np.empty(0))
            continue
        x_cur = lo + int(np.argmax(hist[lo:hi]))
        picked = []
        for y_bottom, y_top in zip(edges[:-1], edges[1:]):
            x0, x1 = x_cur - margin, x_cur + margin
            idx = np.flatnonzero((ys >= y_top) & (ys < y_bottom) & (xs >= x0) & (xs < x1))
            picked.append(idx)
            windows.append((x0, int(y_top), x1, int(y_bottom)))
            if len(idx) >= min_recenter:
                x_cur = int(round(xs[idx].mean()))
        idx = np.concatenate(picked)
        lanes[side] = (xs[idx].astype(np.float64), ys[idx].astype(np.float64))
    return lanes, windows


def fit_frame(lanes, height, args):
    results = {side: fit_lane(*lanes[side], height, args.min_fit_pixels, args.min_y_span)
               for side in SIDES}
    if all(results[side]["fit"] is None for side in SIDES):
        raise FrameError(
            "Hiçbir şerit uydurulamadı: "
            + "; ".join(f"{SIDE_TR[s]}: {results[s]['reason']}" for s in SIDES)
            + "\n  -> Girdinin kuş bakışı şerit maskesi olduğunu kontrol et ya da "
              "--min-fit-pixels / --min-y-span / pencere ayarlarını gevşet.")
    return results


# --------------------------------------------------------------------------- girdi
def read_image(path):
    img = cv2.imread(str(path), flags=cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(
            f"Görüntü okunamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(IMAGE_EXTS))}) olduğunu kontrol et.")
    return img


def load_pixel_lists(path):
    """Piksel listesi .npz -> ({"left": (x, y), "right": (x, y)}, (genişlik, yükseklik))."""
    keys = ("left_x", "left_y", "right_x", "right_y", "image_size")
    try:
        with np.load(path) as data:
            missing = [k for k in keys if k not in data.files]
            if missing:
                raise ValueError(f"eksik alanlar: {', '.join(missing)}")
            arrays = {k: np.asarray(data[k], dtype=np.float64).ravel() for k in keys}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise InputError(
            f"Piksel listesi dosyası okunamadı: {path} ({exc})\n"
            f"  -> Beklenen alanlar: {', '.join(keys)}.") from None
    if arrays["image_size"].size != 2 or (arrays["image_size"] <= 0).any():
        raise InputError(f"image_size iki pozitif sayı olmalı: {arrays['image_size'].tolist()} ({path})")
    width, height = (int(v) for v in arrays["image_size"])
    lanes = {}
    for side in SIDES:
        x, y = arrays[f"{side}_x"], arrays[f"{side}_y"]
        if len(x) != len(y):
            raise InputError(f"{side}_x ve {side}_y uzunlukları farklı ({len(x)} / {len(y)}): {path}")
        if len(x) and (x.min() < 0 or x.max() >= width or y.min() < 0 or y.max() >= height):
            raise InputError(
                f"{side} piksel koordinatları görüntü boyutunun ({width}x{height}) dışında: {path}\n"
                "  -> image_size'ın piksellerin alındığı kuş bakışı görüntüyle aynı olduğunu kontrol et.")
        lanes[side] = (x, y)
    return lanes, (width, height)


# --------------------------------------------------------------------------- görsel
def put_label(img, text, row, color, fs, th):
    org = (10, round((35 + 35 * row) * fs / 0.8))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                thickness=th + 2, lineType=cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color,
                thickness=th, lineType=cv2.LINE_AA)


def lanes_to_mask(lanes, size):
    width, height = size
    mask = np.zeros((height, width), bool)
    for x, y in lanes.values():
        mask[y.astype(int), x.astype(int)] = True
    return mask


# [EK] Pencereler, renkli pikseller ve uydurulan eğri.
def draw_compare(mask, lanes, windows, results):
    height, width = mask.shape
    fs = 0.8 * width / 1280.0
    th = max(1, round(2 * width / 1280.0))
    left = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    put_label(left, "Girdi (kus bakisi maske)", 0, (0, 165, 255), fs, th)

    view = np.zeros((height, width, 3), np.uint8)
    view[mask] = (70, 70, 70)
    colors = {"left": (0, 0, 255), "right": (255, 0, 0)}
    for side in SIDES:
        x, y = lanes[side]
        view[y.astype(int), x.astype(int)] = colors[side]
    for x0, y0, x1, y1 in windows:
        cv2.rectangle(view, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), thickness=th)
    plot_y = np.arange(height, dtype=np.float64)
    for row, side in enumerate(SIDES):
        res = results[side]
        name = "Sol" if side == "left" else "Sag"
        if res["fit"] is None:
            put_label(view, f"{name}: uydurulamadi", row, (0, 165, 255), fs, th)
            continue
        # np.polyval: eğriyi tüm yükseklik boyunca çiz.
        plot_x = np.polyval(res["fit"], plot_y)
        pts = np.column_stack([plot_x, plot_y]).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(view, [pts], False, (0, 255, 255), thickness=th + 1, lineType=cv2.LINE_AA)
        a, b, c = res["fit"]
        put_label(view, f"{name}: A={a:.2e} B={b:.3f} C={c:.1f} RMSE={res['rmse']:.1f}px",
                  row, (0, 255, 255), fs, th)
    return cv2.hconcat([left, view])


# --------------------------------------------------------------------------- kayıt
class FitTable:
    """Kare başına katsayıları toplar ve .npz olarak kaydeder."""

    def __init__(self):
        self.rows = []
        self.image_size = None

    def add(self, name, size, results):
        if self.image_size is None:
            self.image_size = size
        elif size != self.image_size:
            raise InputError(
                f"Farklı çözünürlükte kare: {name} ({size[0]}x{size[1]}), ilk kare "
                f"{self.image_size[0]}x{self.image_size[1]}.\n"
                "  -> Katsayılar piksel cinsindendir ve tek bir çözünürlükte anlamlıdır; "
                "aynı boyutta kareler ver.")
        self.rows.append((name, results))

    def save(self, path):
        def column(side, key, empty):
            return [r[side][key] if r is not None else empty for _, r in self.rows]

        nan3 = np.full(3, np.nan)
        fits = {side: np.array([r[side]["fit"] if r is not None and r[side]["fit"] is not None
                                else nan3 for _, r in self.rows], dtype=np.float64)
                for side in SIDES}
        np.savez(path,
                 left_fit=fits["left"], right_fit=fits["right"],
                 left_rmse=np.array(column("left", "rmse", math.nan), dtype=np.float64),
                 right_rmse=np.array(column("right", "rmse", math.nan), dtype=np.float64),
                 left_count=np.array(column("left", "count", 0), dtype=np.int64),
                 right_count=np.array(column("right", "count", 0), dtype=np.int64),
                 frame=np.array([name for name, _ in self.rows], dtype=np.str_),
                 image_size=np.array(self.image_size, dtype=np.int64))


def describe(results):
    parts = []
    for side in SIDES:
        res = results[side]
        if res["fit"] is None:
            parts.append(f"{SIDE_TR[side]}: uydurulamadı ({res['reason']})")
        else:
            a, b, c = res["fit"]
            parts.append(f"{SIDE_TR[side]} [A, B, C] = [{a:.3e}, {b:.4f}, {c:.2f}] "
                         f"(n={res['count']}, RMSE {res['rmse']:.2f} px)")
    return "\n  ".join(parts)


def handle_frame(name, mask, lanes, windows, table, args, multi, compare_sink):
    """Bir kareyi uydurur, tabloya ekler; çok kareli girişte hatalı kare NaN olur."""
    size = (mask.shape[1], mask.shape[0])
    try:
        results = fit_frame(lanes, mask.shape[0], args)
    except FrameError as exc:
        if not multi:
            raise
        print(f"UYARI: {name}: {str(exc).splitlines()[0]}", file=sys.stderr)
        table.add(name, size, None)
        return None
    table.add(name, size, results)
    for side in SIDES:
        if results[side]["fit"] is None:
            print(f"UYARI: {name}: {SIDE_TR[side]} şerit uydurulamadı ({results[side]['reason']})",
                  file=sys.stderr)
    if compare_sink is not None:
        compare_sink(name, draw_compare(mask, lanes, windows, results))
    return results


# --------------------------------------------------------------------------- işleme
def process_input(path, args):
    table = FitTable()
    out_dir = args.out_dir
    suffix = path.suffix.lower()

    if path.is_dir() or suffix not in VIDEO_EXTS | {".npz"}:
        # Tek görüntü ya da klasör.
        multi = path.is_dir()
        if multi:
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not files:
                raise InputError(f"Klasörde görüntü yok: {path}\n"
                                 f"  -> Desteklenen uzantılar: {', '.join(sorted(IMAGE_EXTS))}")
            compare_dir = out_dir / path.name
        else:
            files, compare_dir = [path], out_dir
        sink = None
        if args.compare:
            compare_dir.mkdir(parents=True, exist_ok=True)
            sink = lambda name, img: write_image(compare_dir / f"{name}_karsilastirma.png", img)
        for file in files:
            frame = read_image(file)
            try:
                mask = binarize(frame, args.mask_threshold, args.max_midtone, args.max_mask_fill)
            except FrameError as exc:
                if not multi:
                    raise
                print(f"UYARI: {file.name}: {str(exc).splitlines()[0]}", file=sys.stderr)
                table.add(file.stem, (frame.shape[1], frame.shape[0]), None)
                continue
            lanes, windows = find_lane_pixels(mask, args.windows, args.window_margin,
                                              args.min_recenter_px)
            results = handle_frame(file.stem, mask, lanes, windows, table, args, multi, sink)
            if results is not None and (not multi or len(files) <= 20):
                print(f"{file.name}:\n  {describe(results)}")
        return table

    if suffix == ".npz":
        lanes, size = load_pixel_lists(path)
        mask = lanes_to_mask(lanes, size)
        sink = None
        if args.compare:
            sink = lambda name, img: write_image(out_dir / f"{name}_karsilastirma.png", img)
        results = handle_frame(path.stem, mask, lanes, [], table, args, False, sink)
        print(f"{path.name}:\n  {describe(results)}")
        return table

    # Video.
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InputError(
            f"Video açılamadı: {path}\n"
            "  -> Dosyanın bozuk olmadığını ve desteklenen bir biçimde "
            f"({', '.join(sorted(VIDEO_EXTS))}) olduğunu kontrol et.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (math.isfinite(fps) and fps > 0):
        fps = 25.0
    writer = None
    try:
        index = 0
        while args.max_frames is None or index < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            name = f"kare{index:05d}"
            index += 1
            try:
                mask = binarize(frame, args.mask_threshold, args.max_midtone, args.max_mask_fill)
            except FrameError as exc:
                print(f"UYARI: {name}: {str(exc).splitlines()[0]}", file=sys.stderr)
                table.add(name, (frame.shape[1], frame.shape[0]), None)
                continue
            lanes, windows = find_lane_pixels(mask, args.windows, args.window_margin,
                                              args.min_recenter_px)
            sink = None
            if args.compare:
                if writer is None:
                    writer = open_writer(out_dir / f"{path.stem}_karsilastirma.mp4", "mp4v", fps,
                                         (2 * frame.shape[1], frame.shape[0]), is_color=True)
                sink = lambda _name, img: writer.write(img)
            handle_frame(name, mask, lanes, windows, table, args, True, sink)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not table.rows:
        raise InputError(f"Videodan hiç kare okunamadı: {path}")
    return table


def write_image(path, img):
    if not cv2.imwrite(str(path), img):
        raise InputError(f"Görüntü yazılamadı: {path}\n  -> --out-dir'i ve yazma iznini kontrol et.")


def open_writer(path, fourcc, fps, size, is_color):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size,
                             isColor=is_color)
    if not writer.isOpened():
        raise InputError(f"Video yazılamadı: {path} ({fourcc})\n"
                         "  -> --out-dir'i ve yazma iznini kontrol et.")
    return writer


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
        description="Şerit piksellerine np.polyfit(y, x, 2) ile x = A*y^2 + B*y + C uydurur.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Kuş bakışı ikili maske (görüntü, klasör, video) ya da piksel listesi "
                        f".npz (varsayılan: {INPUT_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--mask-threshold", type=int, default=MASK_THRESHOLD,
                   help=f"Maske eşiği, 0-254 (varsayılan: {MASK_THRESHOLD})")
    p.add_argument("--max-midtone", type=float, default=MAX_MIDTONE,
                   help=f"İkili maske sayılacak en büyük ara ton oranı (varsayılan: {MAX_MIDTONE})")
    p.add_argument("--max-mask-fill", type=float, default=MAX_MASK_FILL,
                   help=f"İkili maske sayılacak en büyük doluluk oranı (varsayılan: {MAX_MASK_FILL})")
    p.add_argument("--windows", type=int, default=N_WINDOWS,
                   help=f"Kayan pencere sayısı (varsayılan: {N_WINDOWS})")
    p.add_argument("--window-margin", type=float, default=WINDOW_MARGIN,
                   help=f"Pencere yarı genişliği, genişliğin oranı (varsayılan: {WINDOW_MARGIN})")
    p.add_argument("--min-recenter-px", type=int, default=MIN_RECENTER_PX,
                   help=f"Pencereyi kaydırmak için en az piksel (varsayılan: {MIN_RECENTER_PX})")
    p.add_argument("--min-fit-pixels", type=int, default=MIN_FIT_PIXELS,
                   help=f"Şerit başına en az piksel (varsayılan: {MIN_FIT_PIXELS})")
    p.add_argument("--min-y-span", type=float, default=MIN_Y_SPAN,
                   help=f"En az dikey açıklık, yüksekliğin oranı (varsayılan: {MIN_Y_SPAN})")
    p.add_argument("--compare", action="store_true",
                   help="Girdi maskesi ve uydurulan eğriyi yan yana kaydet")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                   help="Videoda en çok kaç kare işlensin (varsayılan: tümü)")
    return p.parse_args(argv)


def validate_args(args):
    checks = (
        ("--mask-threshold", args.mask_threshold, 0 <= args.mask_threshold <= 254, "0-254 aralığında"),
        ("--max-midtone", args.max_midtone, 0 <= args.max_midtone <= 1, "0 <= değer <= 1"),
        ("--max-mask-fill", args.max_mask_fill, 0 < args.max_mask_fill <= 1, "0 < değer <= 1"),
        ("--windows", args.windows, args.windows >= 1, "en az 1"),
        ("--window-margin", args.window_margin, 0 < args.window_margin <= 0.5, "0 < değer <= 0.5"),
        ("--min-recenter-px", args.min_recenter_px, args.min_recenter_px >= 1, "en az 1"),
        ("--min-fit-pixels", args.min_fit_pixels, args.min_fit_pixels >= 3,
         "en az 3 (2. derece için)"),
        ("--min-y-span", args.min_y_span, 0 < args.min_y_span <= 1, "0 < değer <= 1"),
    )
    for name, value, ok, rule in checks:
        if not ok or (isinstance(value, float) and not math.isfinite(value)):
            raise InputError(f"{name} {rule} olmalı, verilen: {value}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise InputError(f"--max-frames pozitif olmalı, verilen: {args.max_frames}")


# --------------------------------------------------------------------------- ana akış
def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    path = Path(args.input)
    if not path.exists():
        raise InputError(
            f"Girdi bulunamadı: {path}\n"
            "  -> --input ile doğru yolu ver ya da AYARLAR'daki INPUT_PATH'i düzelt.")
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"Çıktı klasörü oluşturulamadı: {args.out_dir} ({exc})\n"
                         "  -> --out-dir'i kontrol et.") from None

    table = process_input(path, args)
    fitted = [r for _, r in table.rows if r is not None]
    if not fitted:
        raise InputError(
            f"Hiçbir karede şerit uydurulamadı ({len(table.rows)} kare).\n"
            "  -> Girdinin kuş bakışı şerit maskesi olduğunu kontrol et.")

    out_path = args.out_dir / f"{path.stem}_polinom.npz"
    try:
        table.save(out_path)
    except OSError as exc:
        raise InputError(f"Sonuç yazılamadı: {out_path} ({exc})") from None
    if len(table.rows) > 1:
        counts = {s: sum(r[s]["fit"] is not None for r in fitted) for s in SIDES}
        print(f"{len(table.rows)} kare: sol {counts['left']}, sağ {counts['right']} karede uyduruldu; "
              f"{len(table.rows) - len(fitted)} kare kullanılamadı.")
    print(f"Kaydedildi: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
