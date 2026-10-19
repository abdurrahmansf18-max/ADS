"""
3.3 Eğrilik yarıçapı — virajın sertliği metre cinsinden (R) ve yönü.

Ne işe yarar:
    Yolun araca en yakın noktada hangi yarıçaplı daireye uyduğunu bulur: büyük R hafif viraj ya da düz
    yol, küçük R sert viraj demektir.

Nasıl çalışır:
    Polinom metre cinsine çevrilir (3.2'deki gibi). Araca en yakın satırda
    R = (1 + (2Ay + B)²)^(3/2) / |2A| hesaplanır; sol ve sağ şeridin R'si ortalanır. A çok küçükse
    "düz" der.

Bağlantılar:
    Alır : 2.4'ün polinomu (ya da 2.3'ün pikselleri); 1.3'ün ölçeği, R metre cinsinden olsun diye.
    Verir: R (m) ve yön -> ekrana ve <ad>_egrilik.npz'ye; ana hatta ekrana yazılır.
    Ana hat: main.py her karede pixel_fit_to_meters(), radius_of_curvature() ve classify() çağırır.

Parametreler:
    --input       Başka gereksinimden: 2.4'ün çıktısı (varsayılan sol_viraj test karesinin polinomu).
    --scale       Başka gereksinimden: 1.3'ün ölçek dosyası.
    --left-m, --right-m, --y-m   İsteğe bağlı: metre cinsinden katsayıları elle vermek için.
    --straight-max-a, --y-eval   Ayar (varsayılanlar uygun).
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

# R'nin hesaplandığı satır (px). None ise görüntünün en alt satırı (araca en yakın).
Y_EVAL_PX = None

# Metre uzayında |A| bundan küçükse "düz" (1/m). B ≈ 0 iken R > 1 / (2 * 1e-4) = 5000 m.
STRAIGHT_MAX_A = 1e-4

# Sol / sağ R oranı bunu aşarsa (ya da yönler farklıysa) tutarsızlık uyarısı.
LR_RATIO_WARN = 2.0

# Piksel listesi girdisinde uydurma güvenilirliği.
MIN_FIT_PIXELS = 100          # şerit başına en az piksel
MIN_Y_SPAN = 0.25             # piksellerin dikey açıklığı, görüntü yüksekliğinin oranı
# =============================================================================

SIDES = ("left", "right")
SIDE_TR = {"left": "Sol", "right": "Sağ"}
DIRECTION_TR = {"sag": "sağa viraj", "sol": "sola viraj", "duz": "düz",
                "belirsiz": "belirsiz (şeritler zıt yöne bükülüyor)", "yok": "şerit yok"}


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


# --------------------------------------------------------------------------- eğrilik
def radius_of_curvature(a, b, y):
    """R = (1 + (2Ay + B)²)^(3/2) / |2A|. Metre katsayılar ve metre y ile R metre çıkar.

    A = 0 ise inf, A NaN ise NaN döner.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return (1.0 + (2.0 * a * y + b) ** 2) ** 1.5 / np.abs(2.0 * a)


def refit_in_meters(x_px, y_px, xm, ym, height, min_pixels, min_y_span):
    """Pikselleri metreye çevirip np.polyfit ile yeniden uydurur. Dönüş: ([A, B, C], neden)."""
    # [EK] Güvenilirlik denetimi: kısa parçadan çıkan A (dolayısıyla R) anlamsızdır.
    if len(x_px) < min_pixels:
        return None, f"yetersiz piksel ({len(x_px)} < {min_pixels})"
    span = float(y_px.max() - y_px.min())
    if span < min_y_span * height:
        return None, f"dikey açıklık az ({span:.0f} px < yüksekliğin %{100 * min_y_span:.0f}'i)"
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.RankWarning)
        try:
            return np.polyfit(y_px * ym, x_px * xm, 2), None
        except np.exceptions.RankWarning:
            return None, "kötü koşullu uydurma (RankWarning)"


# [EK] Piksel katsayılarını metreye çevirme; yeniden uydurmayla aynı sonucu verir:
# x_m = xm·x, y_m = ym·y  =>  x_m = (xm·A/ym²)·y_m² + (xm·B/ym)·y_m + xm·C
def pixel_fit_to_meters(fit_px, xm, ym):
    fit_px = np.asarray(fit_px, dtype=np.float64)
    return fit_px * np.array([xm / ym ** 2, xm / ym, xm])


def classify(fits_m, radii, straight_max_a, ratio_warn):
    """Yön ("sag", "sol", "duz", "belirsiz", "yok"), ortalama R ve sol / sağ tutarlılığı."""
    available = [s for s in SIDES if not np.isnan(fits_m[s][0])]
    if not available:
        return "yok", math.nan, True
    radius = float(np.mean([radii[s] for s in available]))
    a = {s: float(fits_m[s][0]) for s in available}
    consistent = True
    # [EK] Tutarlılık: virajda iki şerit aynı yöne bükülmeli ve R'leri yakın olmalı.
    curved = [s for s in available if abs(a[s]) >= straight_max_a]
    if len(available) == 2 and curved:
        if len(curved) == 2 and np.sign(a["left"]) != np.sign(a["right"]):
            # Zıt yöne bükülen şeritlerin ortalama A'sı ~0 olur; "düz" demek yanıltır.
            return "belirsiz", radius, False
        r_left, r_right = radii["left"], radii["right"]
        consistent = bool(max(r_left, r_right) / min(r_left, r_right) <= ratio_warn)
    mean_a = float(np.mean(list(a.values())))
    if abs(mean_a) < straight_max_a:
        direction = "duz"
    else:
        direction = "sag" if mean_a > 0 else "sol"
    return direction, radius, consistent


# --------------------------------------------------------------------------- girdi
def load_scale(path):
    try:
        return PixelMeterScale(path)
    except (FileNotFoundError, ValueError) as exc:
        raise InputError(str(exc)) from None


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
def osculating_circle(fit_m, y):
    """x = f(y) eğrisinin y'deki teğet dairesinin merkezi: (x + (1+f'²)/f'', y - f'(1+f'²)/f'')."""
    a, b, c = fit_m
    x0 = a * y * y + b * y + c
    d1, d2 = 2 * a * y + b, 2 * a
    k = (1 + d1 * d1) / d2
    return (x0 + k, y - d1 * k), (x0, y)


# [EK] Piksel uzayında girdi | metre uzayında eğri ve teğet daire.
def plot_single(path, arrays, fits_px, fits_m, radii, result, scale, y_eval_m, size):
    import matplotlib
    matplotlib.use("Agg")  # konteynerde ekran yok
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    width, height = size
    colors = {"left": "tab:red", "right": "tab:blue"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 8))
    plot_y = np.linspace(0, height - 1, 200)
    for side in SIDES:
        ax0.scatter(arrays[f"{side}_x"], arrays[f"{side}_y"], s=1, color=colors[side],
                    alpha=0.4, label=f"{SIDE_TR[side]} pikseller")
        if fits_px[side] is not None:
            ax0.plot(np.polyval(fits_px[side], plot_y), plot_y, color="gold", lw=2)
    ax0.set_xlim(0, width)
    ax0.set_ylim(height, 0)
    ax0.set_title("Girdi: şerit pikselleri (px), piksel uzayında uydurma")
    ax0.set_xlabel("x (px)")
    ax0.set_ylabel("y (px, 0 = üst)")
    ax0.legend(loc="upper right", markerscale=8)

    y_m = plot_y * scale.ym_per_pix
    for side in SIDES:
        fit = fits_m[side]
        if np.isnan(fit[0]):
            continue
        ax1.plot(np.polyval(fit, y_m), y_m, color=colors[side], lw=5, alpha=0.5,
                 label=f"{SIDE_TR[side]}: R = {radii[side]:.0f} m")
        if fit[0] != 0:
            # Teğet daire araç yakınında şeridin üstüne oturur; görünsün diye siyah, en üstte.
            center, point = osculating_circle(fit, y_eval_m)
            ax1.add_patch(Circle(center, radii[side], fill=False, ls=(0, (4, 3)), lw=1.3,
                                 color="black", zorder=5))
            ax1.plot(*point, "o", color="black", ms=5, zorder=6)
    ax1.set_xlim(-0.05 * width * scale.xm_per_pix, 1.05 * width * scale.xm_per_pix)
    ax1.set_ylim(height * scale.ym_per_pix, 0)
    ax1.set_aspect("equal")
    ax1.set_title(f"Çıktı: metre uzayı, ortalama R = {result['radius']:.0f} m "
                  f"({DIRECTION_TR[result['direction']]})\nsiyah kesikli: y = {y_eval_m:.1f} m'deki teğet daire")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m, 0 = görüntü üstü)")
    ax1.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# [EK] Çok kareli girdi: kare kare A (girdi) | R (çıktı).
def plot_series(path, fits_m, radii, radius, straight_max_a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(radius)
    frames = np.arange(n)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
    for side, color in (("left", "tab:red"), ("right", "tab:blue")):
        ax0.plot(frames, fits_m[side][:, 0], ".-", color=color, label=f"{SIDE_TR[side]} A")
        ax1.plot(frames, radii[side], ".", color=color, alpha=0.5, label=f"{SIDE_TR[side]} R")
    for limit in (straight_max_a, -straight_max_a):
        ax0.axhline(limit, color="gray", ls="--", lw=1)
    ax0.set_title("Girdi: metre uzayında A (1/m); kesikli: düz sınırı")
    ax0.set_xlabel("kare")
    ax0.legend()
    ax1.plot(frames, radius, "k-", lw=1.5, label="ortalama R")
    ax1.axhline(1 / (2 * straight_max_a), color="gray", ls="--", lw=1, label="düz sınırı (B=0)")
    ax1.set_yscale("log")
    ax1.set_title("Çıktı: eğrilik yarıçapı R (m, log)")
    ax1.set_xlabel("kare")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- işleme
def describe(fits_m, radii, direction, radius, consistent):
    lines = []
    for side in SIDES:
        a, b, _ = fits_m[side]
        if np.isnan(a):
            lines.append(f"{SIDE_TR[side]}: şerit yok")
        else:
            lines.append(f"{SIDE_TR[side]}: A = {a:.3e} 1/m, B = {b:.4f} -> R = {radii[side]:.1f} m")
    lines.append(f"Ortalama R = {radius:.1f} m, yön: {DIRECTION_TR[direction]}")
    if not consistent:
        lines.append("UYARI: sol ve sağ şerit uyuşmuyor (yön farklı ya da R'ler 2 kattan fazla "
                     "ayrışıyor); sonuç güvenilmez.")
    return "\n".join(lines)


def run_direct(args):
    """Metre cinsinden [A, B] + y doğrudan verildi; dosya ve ölçek gerekmez."""
    fits_m = {side: (np.array([*coeffs, 0.0]) if coeffs is not None else np.full(3, np.nan))
              for side, coeffs in (("left", args.left_m), ("right", args.right_m))}
    radii = {s: float(radius_of_curvature(fits_m[s][0], fits_m[s][1], args.y_m)) for s in SIDES}
    direction, radius, consistent = classify(fits_m, radii, args.straight_max_a, args.lr_ratio_warn)
    print(f"y = {args.y_m:g} m")
    print(describe(fits_m, radii, direction, radius, consistent))


def run_file(args):
    path = Path(args.input)
    if not path.is_file():
        raise InputError(
            f"Girdi bulunamadı: {path}\n"
            "  -> --input ile piksel listesi ya da 2.4 çıktısı (.npz) ver veya AYARLAR'daki "
            "INPUT_PATH'i düzelt. Katsayıları doğrudan vermek için: --left-m A B --right-m A B --y-m Y")
    scale = load_scale(args.scale)
    kind, arrays, size = load_input(path)
    width, height = size
    try:
        scale.check_size(width, height)
    except ValueError as exc:
        raise InputError(str(exc)) from None
    y_eval_px = height - 1 if args.y_eval is None else args.y_eval
    if not 0 <= y_eval_px <= height - 1:
        raise InputError(f"--y-eval 0 ile {height - 1} arasında olmalı (görüntü yüksekliği {height}), "
                         f"verilen: {y_eval_px}")
    xm, ym = scale.xm_per_pix, scale.ym_per_pix
    y_eval_m = y_eval_px * ym

    if kind == "pixels":
        fits_px, fits_m_single = {}, {}
        for side in SIDES:
            x, y = arrays[f"{side}_x"], arrays[f"{side}_y"]
            fit_m, reason = refit_in_meters(x, y, xm, ym, height, args.min_fit_pixels, args.min_y_span)
            if fit_m is None:
                print(f"UYARI: {SIDE_TR[side].lower()} şerit uydurulamadı ({reason})", file=sys.stderr)
                fits_m_single[side] = np.full(3, np.nan)
                fits_px[side] = None
            else:
                fits_m_single[side] = fit_m
                fits_px[side] = np.polyfit(y, x, 2)   # yalnızca girdi grafiği için
        fits_m = {s: fits_m_single[s][None, :] for s in SIDES}
        frames = np.array([path.stem])
    else:
        fits_m = {s: pixel_fit_to_meters(arrays[f"{s}_fit"], xm, ym) for s in SIDES}
        frames = arrays["frame"]

    n = len(frames)
    radii = {s: radius_of_curvature(fits_m[s][:, 0], fits_m[s][:, 1], y_eval_m) for s in SIDES}
    directions, radius, consistent = [], np.full(n, np.nan), np.ones(n, bool)
    for i in range(n):
        d, r, c = classify({s: fits_m[s][i] for s in SIDES}, {s: float(radii[s][i]) for s in SIDES},
                           args.straight_max_a, args.lr_ratio_warn)
        directions.append(d)
        radius[i], consistent[i] = r, c
    if all(d == "yok" for d in directions):
        raise InputError(f"Hiçbir karede şerit yok; R hesaplanamadı: {path}\n"
                         "  -> Girdide en az bir şerit için yeterli piksel / katsayı olmalı.")

    print(f"Ölçek: xm_per_pix = {xm:.6f}, ym_per_pix = {ym:.6f} m/px ({width}x{height})")
    print(f"y = {y_eval_px:g} px (araca en yakın: {height - 1}) = {y_eval_m:.2f} m")
    if n == 1:
        print(describe({s: fits_m[s][0] for s in SIDES}, {s: float(radii[s][0]) for s in SIDES},
                       directions[0], radius[0], consistent[0]))
    else:
        counts = {d: directions.count(d) for d in DIRECTION_TR}
        valid = radius[~np.isnan(radius)]
        print(f"{n} kare: " + ", ".join(f"{DIRECTION_TR[d]} {c}" for d, c in counts.items() if c)
              + f"; ortanca R = {np.median(valid):.1f} m; tutarsız kare {int((~consistent).sum())}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{path.stem}_egrilik.npz"
    try:
        np.savez(out_path,
                 left_fit_m=fits_m["left"], right_fit_m=fits_m["right"],
                 left_radius_m=radii["left"], right_radius_m=radii["right"], radius_m=radius,
                 direction=np.array(directions, dtype=np.str_), consistent=consistent,
                 y_eval_m=np.float64(y_eval_m), frame=np.asarray(frames, dtype=np.str_),
                 xm_per_pix=np.float64(xm), ym_per_pix=np.float64(ym),
                 image_size=np.array(size, dtype=np.int64))
        print(f"Kaydedildi: {out_path}")
        if args.compare:
            plot_path = args.out_dir / f"{path.stem}_egrilik_karsilastirma.png"
            if kind == "pixels":
                plot_single(plot_path, arrays, fits_px, {s: fits_m[s][0] for s in SIDES},
                            {s: float(radii[s][0]) for s in SIDES},
                            {"radius": radius[0], "direction": directions[0]}, scale, y_eval_m, size)
            else:
                plot_series(plot_path, fits_m, radii, radius, args.straight_max_a)
            print(f"Kaydedildi: {plot_path}")
    except OSError as exc:
        raise InputError(f"Sonuç yazılamadı: {args.out_dir} ({exc})\n"
                         "  -> --out-dir'i ve yazma iznini kontrol et.") from None


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
        description="Metre cinsinden polinomdan eğrilik yarıçapı R = (1+(2Ay+B)^2)^1.5/|2A| ve yön.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Piksel listesi ya da 2.4 çıktısı .npz (varsayılan: {INPUT_PATH})")
    p.add_argument("--scale", type=Path, default=SCALE_PATH,
                   help=f"1.3'ün ölçek dosyası (varsayılan: {SCALE_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--y-eval", type=float, default=Y_EVAL_PX,
                   help="R'nin hesaplanacağı satır, px (varsayılan: en alt satır)")
    p.add_argument("--straight-max-a", type=float, default=STRAIGHT_MAX_A,
                   help=f"|A| bundan küçükse düz, 1/m (varsayılan: {STRAIGHT_MAX_A})")
    p.add_argument("--lr-ratio-warn", type=float, default=LR_RATIO_WARN,
                   help=f"Sol / sağ R oranı uyarı sınırı (varsayılan: {LR_RATIO_WARN})")
    p.add_argument("--min-fit-pixels", type=int, default=MIN_FIT_PIXELS,
                   help=f"Şerit başına en az piksel (varsayılan: {MIN_FIT_PIXELS})")
    p.add_argument("--min-y-span", type=float, default=MIN_Y_SPAN,
                   help=f"En az dikey açıklık, yüksekliğin oranı (varsayılan: {MIN_Y_SPAN})")
    p.add_argument("--compare", action="store_true",
                   help="Girdi ve çıktıyı yan yana grafik olarak kaydet")
    p.add_argument("--left-m", type=float, nargs=2, metavar=("A", "B"),
                   help="Doğrudan giriş: sol şeridin metre katsayıları")
    p.add_argument("--right-m", type=float, nargs=2, metavar=("A", "B"),
                   help="Doğrudan giriş: sağ şeridin metre katsayıları")
    p.add_argument("--y-m", type=float, help="Doğrudan giriş: R'nin hesaplanacağı y (m)")
    return p.parse_args(argv)


def validate_args(args):
    def finite(*values):
        return all(math.isfinite(v) for v in values)

    if not (finite(args.straight_max_a) and args.straight_max_a >= 0):
        raise InputError(f"--straight-max-a negatif olmayan bir sayı olmalı, verilen: {args.straight_max_a}")
    if not (finite(args.lr_ratio_warn) and args.lr_ratio_warn >= 1):
        raise InputError(f"--lr-ratio-warn en az 1 olmalı, verilen: {args.lr_ratio_warn}")
    if args.min_fit_pixels < 3:
        raise InputError(f"--min-fit-pixels en az 3 (2. derece için) olmalı, verilen: {args.min_fit_pixels}")
    if not (finite(args.min_y_span) and 0 < args.min_y_span <= 1):
        raise InputError(f"--min-y-span 0 < değer <= 1 olmalı, verilen: {args.min_y_span}")
    if args.y_eval is not None and not finite(args.y_eval):
        raise InputError(f"--y-eval sonlu bir sayı olmalı, verilen: {args.y_eval}")
    direct = args.left_m is not None or args.right_m is not None
    if direct:
        if args.y_m is None:
            raise InputError("Doğrudan girişte --y-m (metre) de gerekli.\n"
                             "  -> Ör.: --left-m 1e-3 -0.02 --right-m 9.8e-4 -0.02 --y-m 30")
        for coeffs in (args.left_m, args.right_m):
            if coeffs is not None and not finite(*coeffs):
                raise InputError(f"Katsayılar sonlu sayı olmalı, verilen: {coeffs}")
        if not finite(args.y_m):
            raise InputError(f"--y-m sonlu bir sayı olmalı, verilen: {args.y_m}")
    elif args.y_m is not None:
        raise InputError("--y-m yalnızca --left-m / --right-m ile kullanılır.")
    return direct


# --------------------------------------------------------------------------- ana akış
def main(argv=None):
    args = parse_args(argv)
    if validate_args(args):
        run_direct(args)
    else:
        run_file(args)


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
