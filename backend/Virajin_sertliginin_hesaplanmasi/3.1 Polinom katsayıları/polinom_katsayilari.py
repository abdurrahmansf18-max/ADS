"""
3.1 Polinom katsayıları — A katsayısından virajın yönü ve sertliği.

Ne işe yarar:
    Şerit polinomunun A katsayısını yorumlar: A ≈ 0 düz yol, |A| büyüdükçe viraj sertleşir, işareti
    yönü verir (kuş bakışında A > 0 sağa, A < 0 sola viraj).

Nasıl çalışır:
    Polinom alınır (ya da piksellerden yeniden uydurulur). A metre cinsine çevrilir (A·xm/ym²) ve
    eşiklerle düz / hafif / sert diye sınıflanır. Sol ve sağ şerit zıt yöne bükülüyorsa "belirsiz" der.

Bağlantılar:
    Alır : 2.4'ün polinomu (ya da 2.3'ün pikselleri), katsayılar için; 1.3'ün ölçeği, A'yı her kurulumda
           aynı anlamı taşıyan metre birimine çevirmek için.
    Verir: <ad>_katsayi.npz (yön, sınıf); ana hatta viraj sınıfı ekrana yazılır.
    Ana hat: main.py her karede a_to_meters() ve interpret() çağırır.

Parametreler:
    --input       Başka gereksinimden: 2.4'ün çıktısı (varsayılan sol_viraj test karesinin polinomu).
    --scale       Başka gereksinimden: 1.3'ün ölçek dosyası.
    --straight-max-a, --sharp-min-a   Ayar: düz ve sert viraj sınırları (gerçek virajlarda denenebilir).
"""

import argparse
import math
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np

# =============================================================================
# AYARLAR
# Komut satırı argümanları bu varsayılanları ezer (python ... --help).
# [DEĞİŞTİR] işaretli olanları kendi verine göre ayarla.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / Path(__file__).resolve().parent.name

# [DEĞİŞTİR] Şerit verisi: piksel listesi .npz ya da 2.4'ün <ad>_polinom.npz çıktısı.
# Varsayılan: 2.4'ün sol_viraj test karesi için uydurduğu polinom.
INPUT_PATH = (PROJECT_ROOT / "outputs" / "2.4 Polinom uydurma"
              / "sol_viraj_undistorted_maske_warped_lane_pixels_polinom.npz")

# [DEĞİŞTİR] 1.3'ün ölçek dosyası (xm_per_pix, ym_per_pix, image_size).
SCALE_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "olcek.npz"

# Sertlik eşikleri, metre cinsinden |A| (1/m). Tepe noktasında R ≈ 1 / (2|A|):
STRAIGHT_MAX_A = 1e-4         # bundan küçük: düz      (R > ~5000 m)
SHARP_MIN_A = 1e-3            # bundan büyük: sert     (R < ~500 m); arası hafif viraj

# Sol / sağ |A| oranı bunu aşarsa tutarsızlık uyarısı.
LR_RATIO_WARN = 2.0

# Piksel listesi girdisinde uydurma güvenilirliği.
MIN_FIT_PIXELS = 100          # şerit başına en az piksel
MIN_Y_SPAN = 0.25             # piksellerin dikey açıklığı, görüntü yüksekliğinin oranı
# =============================================================================

SIDES = ("left", "right")
SIDE_TR = {"left": "Sol", "right": "Sağ"}
CATEGORY_TR = {"duz": "düz", "hafif": "hafif viraj", "sert": "sert viraj",
               "belirsiz": "belirsiz (şeritler zıt yöne bükülüyor)", "yok": "şerit yok"}
DIRECTION_TR = {"sag": "sağa", "sol": "sola", "-": "-"}


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


# [EK] 1.3'teki yükleyicinin kopyası (klasörler arası import yapılmaz).
class PixelMeterScale:
    """Kaydedilmiş piksel–metre ölçeğini yükler ve piksel -> metre çevirir."""

    REQUIRED_KEYS = ("xm_per_pix", "ym_per_pix", "image_size")

    def __init__(self, path=SCALE_PATH):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Ölçek dosyası bulunamadı: {path}\n"
                "  -> Önce 1.3 piksel_metre_olcekleme.py ile ölçeği hesapla.")
        try:
            with np.load(path) as data:
                missing = [k for k in self.REQUIRED_KEYS if k not in data.files]
                if missing:
                    raise ValueError(f"eksik alanlar: {', '.join(missing)}")
                self.xm_per_pix = float(data["xm_per_pix"])
                self.ym_per_pix = float(data["ym_per_pix"])
                self.image_size = tuple(int(v) for v in data["image_size"])
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(
                f"Ölçek dosyası okunamadı: {path} ({exc})\n"
                "  -> Dosyayı 1.3 piksel_metre_olcekleme.py ile yeniden üret.") from None
        self.path = path

    def check_size(self, width, height):
        """Ölçek yalnızca hesaplandığı kuş bakışı görüntü boyutunda geçerlidir."""
        if (int(width), int(height)) != self.image_size:
            raise ValueError(
                f"Görüntü boyutu {width}x{height}, ancak ölçek "
                f"{self.image_size[0]}x{self.image_size[1]} kuş bakışı görüntü için "
                f"hesaplandı ({self.path}).\n"
                "  -> Ölçeği bu çözünürlükte yeniden hesapla ya da görüntüyü aynı boyutta düzleştir.")

    def x_to_m(self, px):
        return px * self.xm_per_pix

    def y_to_m(self, px):
        return px * self.ym_per_pix


# --------------------------------------------------------------------------- hesap
def fit_lane(x, y, height, min_pixels, min_y_span):
    """np.polyfit(y, x, 2) güvenilirlik denetimiyle. Dönüş: ([A, B, C] ya da None, neden)."""
    # [EK] Güvenilirlik denetimi: kısa parçadan çıkan A anlamsızdır.
    if len(x) < min_pixels:
        return None, f"yetersiz piksel ({len(x)} < {min_pixels})"
    span = float(y.max() - y.min())
    if span < min_y_span * height:
        return None, f"dikey açıklık az ({span:.0f} px < yüksekliğin %{100 * min_y_span:.0f}'i)"
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.RankWarning)
        try:
            return np.polyfit(y, x, 2), None
        except np.exceptions.RankWarning:
            return None, "kötü koşullu uydurma (RankWarning)"


# [EK] A'yı metreye çevirme: x_m = xm·x, y_m = ym·y  =>  A_m = A·xm / ym².
def a_to_meters(a_px, xm, ym):
    return np.asarray(a_px, dtype=np.float64) * xm / ym ** 2


def interpret(a_m, straight_max_a, sharp_min_a, ratio_warn):
    """Sol / sağ A_m (yoksa NaN) -> (sınıf, yön, ortalama A_m, tutarlı mı)."""
    available = {s: float(a_m[s]) for s in SIDES if not math.isnan(a_m[s])}
    if not available:
        return "yok", "-", math.nan, True
    mean_a = float(np.mean(list(available.values())))
    consistent = True
    # [EK] Tutarlılık: virajda iki şerit aynı yöne bükülmeli, |A|'ları yakın olmalı.
    curved = [s for s, a in available.items() if abs(a) >= straight_max_a]
    if len(available) == 2 and curved:
        a_left, a_right = available["left"], available["right"]
        if len(curved) == 2 and np.sign(a_left) != np.sign(a_right):
            return "belirsiz", "-", mean_a, False
        small, large = sorted((abs(a_left), abs(a_right)))
        consistent = small > 0 and large / small <= ratio_warn
    # Gereksinimin özü: A ≈ 0 düz, |A| büyüdükçe sert, işaret yön.
    # [EK] Sertlik sınıfları.
    magnitude = abs(mean_a)
    if magnitude < straight_max_a:
        return "duz", "-", mean_a, consistent
    category = "sert" if magnitude >= sharp_min_a else "hafif"
    return category, ("sag" if mean_a > 0 else "sol"), mean_a, consistent


# --------------------------------------------------------------------------- girdi
def load_input(path):
    """Girdi .npz -> ("pixels" | "fits", veri sözlüğü, (genişlik, yükseklik))."""
    try:
        with np.load(path) as data:
            files = set(data.files)
            if {"left_x", "left_y", "right_x", "right_y", "image_size"} <= files:
                kind = "pixels"
                arrays = {k: np.asarray(data[k], dtype=np.float64).ravel()
                          for k in ("left_x", "left_y", "right_x", "right_y")}
            elif {"left_fit", "right_fit", "image_size"} <= files:
                kind = "fits"
                arrays = {k: np.atleast_2d(np.asarray(data[k], dtype=np.float64))
                          for k in ("left_fit", "right_fit")}
                n = arrays["left_fit"].shape[0]
                arrays["frame"] = (np.asarray(data["frame"]).astype(str) if "frame" in files
                                   else np.array([str(i) for i in range(n)]))
            else:
                raise ValueError(
                    "tanınan alanlar yok; beklenen: left_x, left_y, right_x, right_y, image_size "
                    "(piksel listesi) ya da left_fit, right_fit, image_size (2.4 çıktısı)")
            size = np.asarray(data["image_size"]).ravel()
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise InputError(f"Girdi dosyası okunamadı: {path} ({exc})") from None
    if size.size != 2 or (size <= 0).any():
        raise InputError(f"image_size iki pozitif sayı olmalı: {size.tolist()} ({path})")
    width, height = (int(v) for v in size)

    if kind == "pixels":
        for side in SIDES:
            x, y = arrays[f"{side}_x"], arrays[f"{side}_y"]
            if len(x) != len(y):
                raise InputError(f"{side}_x ve {side}_y uzunlukları farklı ({len(x)} / {len(y)}): {path}")
            if len(x) and (x.min() < 0 or x.max() >= width or y.min() < 0 or y.max() >= height):
                raise InputError(
                    f"{side} piksel koordinatları görüntü boyutunun ({width}x{height}) dışında: {path}")
    else:
        for side in SIDES:
            if arrays[f"{side}_fit"].ndim != 2 or arrays[f"{side}_fit"].shape[1] != 3:
                raise InputError(f"{side}_fit N x 3 olmalı, verilen {arrays[f'{side}_fit'].shape}: {path}")
        if arrays["left_fit"].shape != arrays["right_fit"].shape:
            raise InputError(f"left_fit ve right_fit boyutları farklı: {path}")
    return kind, arrays, (width, height)


# --------------------------------------------------------------------------- görsel
def draw_bands(ax, straight, sharp, limit, vertical):
    """A sayı doğrusunda sertlik bölgeleri (sola sert ... sağa sert)."""
    bands = ((-limit, -sharp, "#f4a6a6", "sola sert"), (-sharp, -straight, "#fbd5a5", "sola hafif"),
             (-straight, straight, "#cfe8cf", "düz"), (straight, sharp, "#fbd5a5", "sağa hafif"),
             (sharp, limit, "#f4a6a6", "sağa sert"))
    for lo, hi, color, label in bands:
        (ax.axhspan if vertical else ax.axvspan)(lo, hi, color=color, alpha=0.6, lw=0)
        if not vertical:
            ax.text((lo + hi) / 2, 0.92, label, ha="center", va="top", fontsize=9,
                    transform=ax.get_xaxis_transform())


# [EK] Şeritler (girdi) | A'nın sertlik bölgelerindeki yeri (çıktı).
def plot_single(path, arrays, fits, a_m, result, size, args):
    import matplotlib
    matplotlib.use("Agg")  # konteynerde ekran yok
    import matplotlib.pyplot as plt

    width, height = size
    colors = {"left": "tab:red", "right": "tab:blue"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.4]})
    plot_y = np.linspace(0, height - 1, 200)
    for side in SIDES:
        if arrays is not None:
            ax0.scatter(arrays[f"{side}_x"], arrays[f"{side}_y"], s=1, color=colors[side], alpha=0.3)
        if not np.isnan(fits[side][0]):
            a, b, c = fits[side]
            ax0.plot(np.polyval(fits[side], plot_y), plot_y, color=colors[side], lw=2,
                     label=f"{SIDE_TR[side]}: A={a:.2e} B={b:.3f} C={c:.0f}")
    ax0.set_xlim(0, width)
    ax0.set_ylim(height, 0)
    ax0.set_title("Girdi: şeritler ve piksel katsayıları")
    ax0.set_xlabel("x (px)")
    ax0.set_ylabel("y (px, 0 = üst)")
    ax0.legend(loc="lower center", fontsize=9)

    finite = [abs(v) for v in a_m.values() if not math.isnan(v)]
    limit = max([2.5 * args.sharp_min_a] + [1.2 * v for v in finite])
    draw_bands(ax1, args.straight_max_a, args.sharp_min_a, limit, vertical=False)
    for row, side in enumerate(SIDES):
        if not math.isnan(a_m[side]):
            ax1.plot(a_m[side], 0.25 + 0.15 * row, "o", color=colors[side], ms=10,
                     label=f"{SIDE_TR[side]} A = {a_m[side]:.2e} 1/m")
    if not math.isnan(result["mean_a"]):
        ax1.plot(result["mean_a"], 0.6, "D", color="black", ms=10,
                 label=f"ortalama A = {result['mean_a']:.2e} 1/m")
    ax1.axvline(0, color="gray", lw=1)
    ax1.set_xlim(-limit, limit)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xlabel("A (1/m)   — işaret: yön, büyüklük: sertlik")
    direction = DIRECTION_TR[result["direction"]]
    ax1.set_title(f"Çıktı: {CATEGORY_TR[result['category']]}"
                  + (f", {direction}" if direction != "-" else ""))
    ax1.legend(loc="lower center", fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# [EK] Çok kareli girdi: piksel A (girdi) | metre A ve sertlik bölgeleri (çıktı).
def plot_series(path, fits, a_m, mean_a, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.arange(len(mean_a))
    colors = {"left": "tab:red", "right": "tab:blue"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
    for side in SIDES:
        ax0.plot(frames, fits[side][:, 0], ".-", color=colors[side], label=f"{SIDE_TR[side]} A (1/px)")
        ax1.plot(frames, a_m[side], ".", color=colors[side], alpha=0.6, label=f"{SIDE_TR[side]} A (1/m)")
    ax1.plot(frames, mean_a, "k-", lw=1.5, label="ortalama A")
    finite = np.abs(mean_a[np.isfinite(mean_a)])
    limit = max(2.5 * args.sharp_min_a, 1.2 * finite.max() if finite.size else 0)
    draw_bands(ax1, args.straight_max_a, args.sharp_min_a, limit, vertical=True)
    ax1.set_ylim(-limit, limit)
    ax0.set_title("Girdi: piksel katsayısı A (1/px)")
    ax1.set_title("Çıktı: metre A (1/m); yeşil düz, turuncu hafif, kırmızı sert")
    for ax in (ax0, ax1):
        ax.axhline(0, color="gray", lw=1)
        ax.set_xlabel("kare")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- ana akış
def run(args):
    path = Path(args.input)
    if not path.is_file():
        raise InputError(
            f"Girdi bulunamadı: {path}\n"
            "  -> --input ile piksel listesi ya da 2.4 çıktısı (.npz) ver veya AYARLAR'daki "
            "INPUT_PATH'i düzelt.")
    try:
        scale = PixelMeterScale(args.scale)
    except (FileNotFoundError, ValueError) as exc:
        raise InputError(str(exc)) from None
    kind, arrays, size = load_input(path)
    width, height = size
    try:
        scale.check_size(width, height)
    except ValueError as exc:
        raise InputError(str(exc)) from None
    xm, ym = scale.xm_per_pix, scale.ym_per_pix

    if kind == "pixels":
        fits = {}
        for side in SIDES:
            fit, reason = fit_lane(arrays[f"{side}_x"], arrays[f"{side}_y"], height,
                                   args.min_fit_pixels, args.min_y_span)
            if fit is None:
                print(f"UYARI: {SIDE_TR[side].lower()} şerit uydurulamadı ({reason})", file=sys.stderr)
                fit = np.full(3, np.nan)
            fits[side] = fit[None, :]
        frames = np.array([path.stem])
    else:
        fits = {s: arrays[f"{s}_fit"] for s in SIDES}
        frames = arrays["frame"]

    n = len(frames)
    a_m = {s: a_to_meters(fits[s][:, 0], xm, ym) for s in SIDES}
    categories, directions = [], []
    mean_a, consistent = np.full(n, np.nan), np.ones(n, bool)
    for i in range(n):
        cat, direc, mean_a[i], consistent[i] = interpret(
            {s: float(a_m[s][i]) for s in SIDES}, args.straight_max_a, args.sharp_min_a,
            args.lr_ratio_warn)
        categories.append(cat)
        directions.append(direc)
    if all(c == "yok" for c in categories):
        raise InputError(f"Hiçbir şerit uydurulamadı: {path}\n"
                         "  -> Girdide en az bir şerit için yeterli piksel / katsayı olmalı.")

    print(f"Ölçek (1.3): xm_per_pix = {xm:.6f}, ym_per_pix = {ym:.6f} m/px ({width}x{height}); "
          f"eşikler: düz < {args.straight_max_a:g} <= hafif < {args.sharp_min_a:g} <= sert (1/m)")
    if n == 1:
        for side in SIDES:
            fit = fits[side][0]
            if np.isnan(fit[0]):
                print(f"{SIDE_TR[side]}: şerit yok")
            else:
                print(f"{SIDE_TR[side]}: [A, B, C] = [{fit[0]:.3e}, {fit[1]:.4f}, {fit[2]:.2f}] (px) "
                      f"-> A = {a_m[side][0]:.3e} 1/m")
        direction = DIRECTION_TR[directions[0]]
        print(f"Yorum: ortalama A = {mean_a[0]:.3e} 1/m -> {CATEGORY_TR[categories[0]]}"
              + (f", {direction}" if direction != "-" else ""))
        if not consistent[0]:
            print("UYARI: sol ve sağ şerit uyuşmuyor (zıt yön ya da |A|'lar 2 kattan fazla "
                  "farklı); sonuç güvenilmez.")
    else:
        counts = {c: categories.count(c) for c in CATEGORY_TR}
        print(f"{n} kare: " + ", ".join(f"{CATEGORY_TR[c]} {k}" for c, k in counts.items() if k)
              + f"; sağa {directions.count('sag')}, sola {directions.count('sol')}; "
                f"tutarsız kare {int((~consistent).sum())}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{path.stem}_katsayi.npz"
    try:
        np.savez(out_path,
                 left_fit=fits["left"], right_fit=fits["right"],
                 left_a_m=a_m["left"], right_a_m=a_m["right"], mean_a_m=mean_a,
                 category=np.array(categories, dtype=np.str_),
                 direction=np.array(directions, dtype=np.str_), consistent=consistent,
                 straight_max_a=np.float64(args.straight_max_a),
                 sharp_min_a=np.float64(args.sharp_min_a),
                 frame=np.asarray(frames, dtype=np.str_),
                 xm_per_pix=np.float64(xm), ym_per_pix=np.float64(ym),
                 image_size=np.array(size, dtype=np.int64))
        print(f"Kaydedildi: {out_path}")
        if args.compare:
            plot_path = args.out_dir / f"{path.stem}_katsayi_karsilastirma.png"
            if n == 1:
                plot_single(plot_path, arrays if kind == "pixels" else None,
                            {s: fits[s][0] for s in SIDES}, {s: float(a_m[s][0]) for s in SIDES},
                            {"category": categories[0], "direction": directions[0],
                             "mean_a": mean_a[0]}, size, args)
            else:
                plot_series(plot_path, fits, a_m, mean_a, args)
            print(f"Kaydedildi: {plot_path}")
    except OSError as exc:
        raise InputError(f"Sonuç yazılamadı: {args.out_dir} ({exc})\n"
                         "  -> --out-dir'i ve yazma iznini kontrol et.") from None


class TurkishArgumentParser(argparse.ArgumentParser):
    """Argüman hatasında "HATA: ..." yazar ve 1 koduyla çıkar."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"HATA: Geçersiz komut satırı argümanı: {message}\n"
              "  -> Seçenekler için --help kullan.", file=sys.stderr)
        sys.exit(1)


def parse_args(argv=None):
    p = TurkishArgumentParser(
        description="Şerit polinomunun [A, B, C] katsayılarını bulur; A'dan viraj sertliği ve yönü.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Piksel listesi ya da 2.4 çıktısı .npz (varsayılan: {INPUT_PATH})")
    p.add_argument("--scale", type=Path, default=SCALE_PATH,
                   help=f"1.3'ün ölçek dosyası (varsayılan: {SCALE_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--straight-max-a", type=float, default=STRAIGHT_MAX_A,
                   help=f"|A| bundan küçükse düz, 1/m (varsayılan: {STRAIGHT_MAX_A})")
    p.add_argument("--sharp-min-a", type=float, default=SHARP_MIN_A,
                   help=f"|A| bundan büyükse sert viraj, 1/m (varsayılan: {SHARP_MIN_A})")
    p.add_argument("--lr-ratio-warn", type=float, default=LR_RATIO_WARN,
                   help=f"Sol / sağ |A| oranı uyarı sınırı (varsayılan: {LR_RATIO_WARN})")
    p.add_argument("--min-fit-pixels", type=int, default=MIN_FIT_PIXELS,
                   help=f"Şerit başına en az piksel (varsayılan: {MIN_FIT_PIXELS})")
    p.add_argument("--min-y-span", type=float, default=MIN_Y_SPAN,
                   help=f"En az dikey açıklık, yüksekliğin oranı (varsayılan: {MIN_Y_SPAN})")
    p.add_argument("--compare", action="store_true",
                   help="Şeritleri ve A'nın sertlik bölgelerindeki yerini grafik olarak kaydet")
    return p.parse_args(argv)


def validate_args(args):
    straight, sharp = args.straight_max_a, args.sharp_min_a
    if not (math.isfinite(straight) and math.isfinite(sharp) and 0 <= straight < sharp):
        raise InputError(f"0 <= --straight-max-a < --sharp-min-a olmalı, verilen: {straight} / {sharp}")
    if not (math.isfinite(args.lr_ratio_warn) and args.lr_ratio_warn >= 1):
        raise InputError(f"--lr-ratio-warn en az 1 olmalı, verilen: {args.lr_ratio_warn}")
    if args.min_fit_pixels < 3:
        raise InputError(f"--min-fit-pixels en az 3 (2. derece için) olmalı, verilen: {args.min_fit_pixels}")
    if not (math.isfinite(args.min_y_span) and 0 < args.min_y_span <= 1):
        raise InputError(f"--min-y-span 0 < değer <= 1 olmalı, verilen: {args.min_y_span}")


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
