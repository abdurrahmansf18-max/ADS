"""
3.2 Piksel–metre ölçeklemesi — şerit polinomunun metre cinsinden yeniden uydurulması.

Ne işe yarar:
    Polinomu piksel yerine metre cinsine çevirir; yoksa eğrilik yarıçapı piksel biriminde çıkar. Yatay ve
    dikey ölçek (xm, ym) farklı olduğu için tek bir çarpanla çevrilemez.

Nasıl çalışır:
    Pikseller xm ve ym ile metreye çevrilip yeniden uydurulur; katsayı verilirse formülle çevrilir
    (A·xm/ym², B·xm/ym, C·xm). Piksel ve metre cinsinden yarıçap yan yana gösterilir.

Bağlantılar:
    Alır : 2.3'ün pikselleri (ya da 2.4'ün polinomu); 1.3'ün ölçeği, çeviri xm ve ym ile yapıldığı için.
    Verir: <ad>_metre_polinom.npz (açıklayıcı çıktı); aynı çeviriyi 3.3 kendi içinde yapar.
    Ana hat: doğrudan çağrılmaz.

Parametreler:
    --input       Başka gereksinimden: 2.3'ün çıktısı (varsayılan sol_viraj test karesinin pikselleri).
    --scale       Başka gereksinimden: 1.3'ün ölçek dosyası.
    --y-eval      Ayar: yarıçapın hesaplandığı satır (varsayılan görüntünün altı, araca en yakın yer).
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
# Varsayılan: 2.3'ün sol_viraj test karesinde bulduğu şerit pikselleri.
INPUT_PATH = PROJECT_ROOT / "outputs" / "2.3 Sliding Windows" / "sol_viraj_undistorted_maske_warped_lane_pixels.npz"

# [DEĞİŞTİR] 1.3'ün ölçek dosyası (xm_per_pix, ym_per_pix, image_size).
SCALE_PATH = DATA_DIR / "uretilen_veri" / "kurulum_dosyalari" / "kolay_video" / "olcek.npz"

# R'nin hesaplandığı satır (px). None ise görüntünün en alt satırı (araca en yakın).
Y_EVAL_PX = None

# Piksel listesi girdisinde uydurma güvenilirliği.
MIN_FIT_PIXELS = 100          # şerit başına en az piksel
MIN_Y_SPAN = 0.25             # piksellerin dikey açıklığı, görüntü yüksekliğinin oranı
# =============================================================================

SIDES = ("left", "right")
SIDE_TR = {"left": "Sol", "right": "Sağ"}


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
def radius_of_curvature(a, b, y):
    """R = (1 + (2Ay + B)²)^(3/2) / |2A|. Birimi a, b, y'nin birimine bağlıdır."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return (1.0 + (2.0 * a * y + b) ** 2) ** 1.5 / np.abs(2.0 * a)


def fit_checked(x, y, height_px, min_pixels, min_y_span, y_px):
    """np.polyfit(y, x, 2) güvenilirlik denetimiyle. y_px: dikey açıklık denetimi için."""
    # [EK] Güvenilirlik denetimi: kısa parçadan çıkan A anlamsızdır.
    if len(x) < min_pixels:
        return None, f"yetersiz piksel ({len(x)} < {min_pixels})"
    span = float(y_px.max() - y_px.min())
    if span < min_y_span * height_px:
        return None, f"dikey açıklık az ({span:.0f} px < yüksekliğin %{100 * min_y_span:.0f}'i)"
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.RankWarning)
        try:
            return np.polyfit(y, x, 2), None
        except np.exceptions.RankWarning:
            return None, "kötü koşullu uydurma (RankWarning)"


# [EK] Piksel katsayılarını metreye çevirme; yeniden uydurmayla aynı sonucu verir:
# x_m = xm·x, y_m = ym·y  =>  x_m = (xm·A/ym²)·y_m² + (xm·B/ym)·y_m + xm·C
def pixel_fit_to_meters(fit_px, xm, ym):
    return np.asarray(fit_px, dtype=np.float64) * np.array([xm / ym ** 2, xm / ym, xm])


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
# [EK] Piksel uzayı | metre uzayı (eşit eksen): aynı şerit iki birimde.
def plot_compare(path, arrays, fits_px, fits_m, radii_px, radii_m, scale, size, y_eval_px):
    import matplotlib
    matplotlib.use("Agg")  # konteynerde ekran yok
    import matplotlib.pyplot as plt

    width, height = size
    xm, ym = scale.xm_per_pix, scale.ym_per_pix
    colors = {"left": "tab:red", "right": "tab:blue"}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.6, 1]})
    plot_y = np.linspace(0, height - 1, 200)
    for side in SIDES:
        if arrays is not None:
            ax0.scatter(arrays[f"{side}_x"], arrays[f"{side}_y"], s=1, color=colors[side], alpha=0.3)
        fit_px, fit_m = fits_px[side][0], fits_m[side][0]
        if np.isnan(fit_px[0]):
            continue
        ax0.plot(np.polyval(fit_px, plot_y), plot_y, color=colors[side], lw=2,
                 label=f"{SIDE_TR[side]}: A={fit_px[0]:.2e} 1/px, R={radii_px[side][0]:.0f} 'px'")
        ax1.plot(np.polyval(fit_m, plot_y * ym), plot_y * ym, color=colors[side], lw=2,
                 label=f"{SIDE_TR[side]}: A={fit_m[0]:.2e} 1/m, R={radii_m[side][0]:.0f} m")
    for ax, y_line in ((ax0, y_eval_px), (ax1, y_eval_px * ym)):
        ax.axhline(y_line, color="gray", ls="--", lw=1)
        ax.legend(loc="upper right")
    ax0.set_xlim(0, width)
    ax0.set_ylim(height, 0)
    ax0.set_title("Girdi: piksel uzayı (R piksel biriminde, anlamsız)")
    ax0.set_xlabel("x (px)")
    ax0.set_ylabel("y (px, 0 = üst)")
    ax1.set_xlim(0, width * xm)
    ax1.set_ylim(height * ym, 0)
    ax1.set_aspect("equal")
    ax1.set_title(f"Çıktı: metre uzayı (eşit eksen)\nxm = {xm:.5f}, ym = {ym:.5f} m/px")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m, 0 = görüntü üstü)")
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
    y_eval_px = height - 1 if args.y_eval is None else args.y_eval
    if not 0 <= y_eval_px <= height - 1:
        raise InputError(f"--y-eval 0 ile {height - 1} arasında olmalı (görüntü yüksekliği {height}), "
                         f"verilen: {y_eval_px}")
    xm, ym = scale.xm_per_pix, scale.ym_per_pix
    y_eval_m = y_eval_px * ym
    nan3 = np.full(3, np.nan)

    if kind == "pixels":
        fits_px, fits_m, max_diff = {}, {}, 0.0
        for side in SIDES:
            x, y = arrays[f"{side}_x"], arrays[f"{side}_y"]
            fit_px, reason = fit_checked(x, y, height, args.min_fit_pixels, args.min_y_span, y)
            # Gereksinimin özü: koordinatları metreye çevirip yeniden uydur.
            fit_m, _ = fit_checked(x * xm, y * ym, height, args.min_fit_pixels, args.min_y_span, y)
            if fit_px is None or fit_m is None:
                print(f"UYARI: {SIDE_TR[side].lower()} şerit uydurulamadı ({reason})", file=sys.stderr)
                fit_px = fit_m = nan3
            else:
                # [EK] Doğrulama: formülle çevirme yeniden uydurmayla aynı eğriyi vermeli.
                y_m = np.linspace(0, (height - 1) * ym, 200)
                conv = pixel_fit_to_meters(fit_px, xm, ym)
                max_diff = max(max_diff, float(np.abs(np.polyval(conv, y_m) - np.polyval(fit_m, y_m)).max()))
            fits_px[side], fits_m[side] = fit_px[None, :], fit_m[None, :]
        frames = np.array([path.stem])
    else:
        fits_px = {s: arrays[f"{s}_fit"] for s in SIDES}
        fits_m = {s: pixel_fit_to_meters(fits_px[s], xm, ym) for s in SIDES}
        frames = arrays["frame"]

    if all(np.isnan(fits_m[s][:, 0]).all() for s in SIDES):
        raise InputError(f"Hiçbir şerit uydurulamadı: {path}\n"
                         "  -> Girdide en az bir şerit için yeterli piksel / katsayı olmalı.")
    radii_px = {s: radius_of_curvature(fits_px[s][:, 0], fits_px[s][:, 1], y_eval_px) for s in SIDES}
    radii_m = {s: radius_of_curvature(fits_m[s][:, 0], fits_m[s][:, 1], y_eval_m) for s in SIDES}

    print(f"Ölçek (1.3): xm_per_pix = {xm:.6f}, ym_per_pix = {ym:.6f} m/px, xm/ym = {xm / ym:.3f} "
          f"({width}x{height})")
    print(f"y = {y_eval_px:g} px = {y_eval_m:.2f} m (araca en yakın satır: {height - 1})")
    n = len(frames)
    if n == 1:
        for side in SIDES:
            fp, fm = fits_px[side][0], fits_m[side][0]
            if np.isnan(fm[0]):
                print(f"{SIDE_TR[side]}: şerit yok")
                continue
            r_px, r_m = float(radii_px[side][0]), float(radii_m[side][0])
            print(f"{SIDE_TR[side]}:\n"
                  f"  piksel [A, B, C] = [{fp[0]:.3e}, {fp[1]:.4f}, {fp[2]:.2f}] -> R = {r_px:.1f} 'px'\n"
                  f"  metre  [A, B, C] = [{fm[0]:.3e}, {fm[1]:.4f}, {fm[2]:.3f}] -> R = {r_m:.1f} m\n"
                  # [EK] Tek çarpanla çevirmek yanlış: xm != ym.
                  f"  tek çarpanla (yanlış): R_px·xm = {r_px * xm:.1f} m, R_px·ym = {r_px * ym:.1f} m")
        if kind == "pixels":
            print(f"Doğrulama: yeniden uydurma ile formülle çevirme eğrileri arasındaki en büyük fark "
                  f"{max_diff:.1e} m")
    else:
        valid = {s: radii_m[s][np.isfinite(radii_m[s])] for s in SIDES}
        print(f"{n} kare çevrildi; ortanca R sol {np.median(valid['left']) if valid['left'].size else math.nan:.1f} m, "
              f"sağ {np.median(valid['right']) if valid['right'].size else math.nan:.1f} m")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{path.stem}_metre_polinom.npz"
    try:
        np.savez(out_path,
                 left_fit_m=fits_m["left"], right_fit_m=fits_m["right"],
                 left_fit_px=fits_px["left"], right_fit_px=fits_px["right"],
                 left_radius_m=radii_m["left"], right_radius_m=radii_m["right"],
                 left_radius_px=radii_px["left"], right_radius_px=radii_px["right"],
                 y_eval_px=np.float64(y_eval_px), y_eval_m=np.float64(y_eval_m),
                 frame=np.asarray(frames, dtype=np.str_),
                 xm_per_pix=np.float64(xm), ym_per_pix=np.float64(ym),
                 image_size=np.array(size, dtype=np.int64))
        print(f"Kaydedildi: {out_path}")
        if args.compare:
            if n > 1:
                print("UYARI: --compare yalnızca tek kareli girdide çizilir; çok kareli girdide "
                      "atlandı.", file=sys.stderr)
            else:
                plot_path = args.out_dir / f"{path.stem}_metre_polinom_karsilastirma.png"
                plot_compare(plot_path, arrays if kind == "pixels" else None, fits_px, fits_m,
                             radii_px, radii_m, scale, size, y_eval_px)
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
        description="Şerit polinomunu x*xm_per_pix, y*ym_per_pix ile metre uzayında yeniden uydurur.")
    p.add_argument("--input", type=Path, default=INPUT_PATH,
                   help=f"Piksel listesi ya da 2.4 çıktısı .npz (varsayılan: {INPUT_PATH})")
    p.add_argument("--scale", type=Path, default=SCALE_PATH,
                   help=f"1.3'ün ölçek dosyası (varsayılan: {SCALE_PATH})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--y-eval", type=float, default=Y_EVAL_PX,
                   help="R'nin hesaplanacağı satır, px (varsayılan: en alt satır)")
    p.add_argument("--min-fit-pixels", type=int, default=MIN_FIT_PIXELS,
                   help=f"Şerit başına en az piksel (varsayılan: {MIN_FIT_PIXELS})")
    p.add_argument("--min-y-span", type=float, default=MIN_Y_SPAN,
                   help=f"En az dikey açıklık, yüksekliğin oranı (varsayılan: {MIN_Y_SPAN})")
    p.add_argument("--compare", action="store_true",
                   help="Piksel ve metre uzayını yan yana grafik olarak kaydet")
    return p.parse_args(argv)


def validate_args(args):
    if args.min_fit_pixels < 3:
        raise InputError(f"--min-fit-pixels en az 3 (2. derece için) olmalı, verilen: {args.min_fit_pixels}")
    if not (math.isfinite(args.min_y_span) and 0 < args.min_y_span <= 1):
        raise InputError(f"--min-y-span 0 < değer <= 1 olmalı, verilen: {args.min_y_span}")
    if args.y_eval is not None and not math.isfinite(args.y_eval):
        raise InputError(f"--y-eval sonlu bir sayı olmalı, verilen: {args.y_eval}")


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
