"""
4.4 Renk ve gradyan maskelerinin birleştirilmesi — iki maskeden tek şerit maskesi.

Ne işe yarar:
    Renk maskesi soluk çizgide, gradyan maskesi gölgede yanılır. İkisini birleştirmek birbirinin açığını
    kapatır; sonuç şerit aramaya (2.3) giden maskedir.

Nasıl çalışır:
    Renk ve gradyan maskeleri OR (biri yeterli) ya da AND (ikisi de gerekli) ile birleştirilir; istenirse
    morfolojiyle küçük gürültü temizlenir. Piksellerin hangi maskeden geldiği oran olarak yazılır.

Bağlantılar:
    Alır : renk maskesi 4.2'den (ya da 2.1'den), gradyan maskesi 4.3'ten (ya da 2.2'den); maskeler aynı
           karelerden, aynı sırada olmalıdır.
    Verir: <ad>_<mod>_maske.png -> 1.2 ile kuş bakışına -> 2.3.
    Ana hat: main.py OR ile birleştirir (combine, clean); AND zor videoda soluk sarı çizgiyi
             kaybettiği için seçilmedi.

Parametreler:
    --color-masks     Başka gereksinimden: 4.2'nin maskeleri (varsayılan) ya da 2.1'inkiler.
    --gradient-mask   Başka gereksinimden: 4.3'ün maskeleri (varsayılan) ya da 2.2'ninkiler.
    --mode, --morph   Ayar: or / and / both (ikisini karşılaştırır), morfoloji (varsayılan yok).
"""

import argparse
import math
import sys
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

# [DEĞİŞTİR] Renk maske(ler)i (2.1 / 4.2 çıktısı) ve gradyan maskesi (2.2 / 4.3 çıktısı):
# tek görüntü, klasör ya da video; hepsi aynı türde olmalı.
# Varsayılan: ana hattaki gibi 4.2 renk + 4.3 gradyan maskeleri (kolay videonun test kareleri).
COLOR_MASKS = [OUTPUTS / "4.2 HLS∕HSV∕LAB renk kanalları" / "duzeltilmis_kareler"]
GRADIENT_MASK = OUTPUTS / "4.3 Sobel gradyanı ve Canny" / "kolay_video"

MODE = "or"                   # "or", "and" ya da "both"

# İsteğe bağlı morfolojik temizlik: "none", "open" (küçük gürültüyü sil), "close" (boşluk doldur).
MORPH = "none"
MORPH_KSIZE = 3               # çekirdek boyutu, px (~1280x720 için)
# Çekirdek şekli: "auto" (açma: dikey 1 x k, kapama: kare k x k), "rect" (k x k), "vertical" (1 x k).
# Kare açma, ~2 px ince gradyan kenarlarını (soluk çizgi) siler; şeritler dikeye yakın olduğu
# için açmada dikey çekirdek kullanılır.
MORPH_SHAPE = "auto"

MASK_THRESHOLD = 127          # kayıplı maskeler için eşik
MAX_MIDTONE = 0.05            # ara ton (33-222) oranı bunu aşarsa girdi ikili maske sayılmaz

# Video: en çok kaç kare işlensin (None = tümü).
MAX_FRAMES = None
# =============================================================================

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
MORPH_OPS = {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
STAT_KEYS = ("color", "gradient", "both", "color_only", "gradient_only", "or", "and")


class InputError(Exception):
    """Kullanıcının düzeltebileceği hata; "HATA: ..." olarak yazılır, çıkış kodu 1."""


# --------------------------------------------------------------------------- birleştirme
def combine(color, gradient, mode):
    """Gereksinimin özü: NumPy mantıksal işlemleri."""
    return (color | gradient) if mode == "or" else (color & gradient)


def morph_kernel_size(op, ksize, shape):
    """(genişlik, yükseklik). [EK] auto: açmada dikey (ince şerit kenarlarını korur)."""
    if shape == "vertical" or (shape == "auto" and op == "open"):
        return (1, ksize)
    return (ksize, ksize)


def clean(mask, op, ksize, shape="auto"):
    """İsteğe bağlı cv2.morphologyEx temizliği (bool -> bool)."""
    if op == "none":
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size(op, ksize, shape))
    return cv2.morphologyEx(mask.astype(np.uint8) * 255, MORPH_OPS[op], kernel) > 0


# [EK] Kaynak oranları: OR / AND farkının nereden geldiğini gösterir.
def source_stats(color, gradient):
    n = color.size
    both = np.count_nonzero(color & gradient)
    c, g = np.count_nonzero(color), np.count_nonzero(gradient)
    return {"color": c / n, "gradient": g / n, "both": both / n, "color_only": (c - both) / n,
            "gradient_only": (g - both) / n, "or": (c + g - both) / n, "and": both / n}


def format_stats(s):
    return (f"renk %{100 * s['color']:.2f}, gradyan %{100 * s['gradient']:.2f}, ikisi "
            f"%{100 * s['both']:.2f} | OR %{100 * s['or']:.2f}, AND %{100 * s['and']:.2f} "
            f"(yalnız gradyan %{100 * s['gradient_only']:.2f}, yalnız renk %{100 * s['color_only']:.2f})")


# --------------------------------------------------------------------------- girdi
def binarize(frame, name, args):
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # [EK] İkili maske denetimi: maske neredeyse yalnız 0 ve 255'tir.
    midtone = np.count_nonzero((gray > 32) & (gray < 223)) / gray.size
    if midtone > args.max_midtone:
        raise InputError(
            f"{name} ikili maske gibi görünmüyor (ara ton %{100 * midtone:.1f}).\n"
            "  -> 2.1 / 2.2 gibi modüllerin ürettiği maskeleri ver, renkli kare değil.")
    return gray > args.mask_threshold


def source_kind(path):
    path = Path(path)
    if not path.exists():
        raise InputError(
            f"Maske bulunamadı: {path}\n"
            "  -> --color-masks / --gradient-mask ile doğru yolu ver ya da AYARLAR'daki yolları "
            "düzelt (önce 2.1 ve 2.2'yi çalıştır).")
    if path.is_dir():
        return "folder"
    return "video" if path.suffix.lower() in VIDEO_EXTS else "image"


def read_image(path):
    img = cv2.imread(str(path), flags=cv2.IMREAD_UNCHANGED)
    if img is None:
        raise InputError(f"Maske okunamadı: {path}\n"
                         "  -> Dosyanın bozuk olmadığını kontrol et.")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def folder_files(path):
    files = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise InputError(f"Klasörde maske yok: {path}\n"
                         f"  -> Desteklenen uzantılar: {', '.join(sorted(IMAGE_EXTS))}")
    return files


def base_name(path):
    stem = Path(path).stem
    return stem[:-len("_maske")] if stem.endswith("_maske") else stem


def load_set(color_frames, gradient_frame, names, args):
    """Bir karenin renk(ler) ve gradyan maskelerini okur, hizayı denetler."""
    masks = [binarize(f, n, args) for f, n in zip(color_frames, names[:-1])]
    gradient = binarize(gradient_frame, names[-1], args)
    for m, n in zip(masks, names[:-1]):
        if m.shape != gradient.shape:
            raise InputError(
                f"Maske boyutları farklı: {n} {m.shape[1]}x{m.shape[0]}, {names[-1]} "
                f"{gradient.shape[1]}x{gradient.shape[0]}.\n"
                "  -> Renk ve gradyan maskeleri aynı kareden, aynı çözünürlükte üretilmeli.")
    color = masks[0]
    for m in masks[1:]:
        color = color | m           # birden çok renk maskesi önce kendi arasında OR'lanır
    return color, gradient


# --------------------------------------------------------------------------- görsel / çıktı
def put_label(img, text, fs, th):
    org = (10, round(35 * fs / 0.8))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                thickness=th + 2, lineType=cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 165, 255),
                thickness=th, lineType=cv2.LINE_AA)


# [EK] Kaynağa göre renklendirilmiş karşılaştırma: renk | gradyan | sonuç(lar).
def draw_compare(color, gradient, results, args):
    height, width = color.shape
    fs = 0.8 * width / 1280.0
    th = max(1, round(2 * width / 1280.0))
    panels = []
    for mask, text in ((color, "Renk maskesi"), (gradient, "Gradyan maskesi")):
        img = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        put_label(img, text, fs, th)
        panels.append(img)
    for mode, mask in results.items():
        img = np.zeros((height, width, 3), np.uint8)
        img[mask & ~color & ~gradient] = (255, 0, 255)     # morfolojinin eklediği: mor
        img[mask & color & ~gradient] = (0, 255, 255)      # yalnız renk: sarı
        img[mask & gradient & ~color] = (255, 255, 0)      # yalnız gradyan: camgöbeği
        img[mask & color & gradient] = (255, 255, 255)     # ikisi: beyaz
        kw, kh = morph_kernel_size(args.morph, args.morph_ksize, args.morph_shape)
        morph = f", {args.morph} {kw}x{kh}" if args.morph != "none" else ""
        put_label(img, f"{mode.upper()}{morph}: %{100 * mask.mean():.2f}", fs, th)
        panels.append(img)
    return cv2.hconcat(panels)


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


def modes_of(args):
    return ("or", "and") if args.mode == "both" else (args.mode,)


def process_pair(color, gradient, args):
    results = {m: clean(combine(color, gradient, m), args.morph, args.morph_ksize, args.morph_shape)
               for m in modes_of(args)}
    return results, source_stats(color, gradient)


# --------------------------------------------------------------------------- işleme
def process_images(color_paths, gradient_path, out_dir, args, name):
    frames = [read_image(p) for p in color_paths]
    color, gradient = load_set(frames, read_image(gradient_path), [*map(str, color_paths), str(gradient_path)], args)
    results, stats = process_pair(color, gradient, args)
    for mode, mask in results.items():
        write_image(out_dir / f"{name}_{mode}_maske.png", mask.astype(np.uint8) * 255)
    if args.compare:
        write_image(out_dir / f"{name}_karsilastirma.png", draw_compare(color, gradient, results, args))
    return stats, {m: float(r.mean()) for m, r in results.items()}


def process_folders(color_dirs, gradient_dir, out_dir, args):
    lists = [folder_files(d) for d in color_dirs] + [folder_files(gradient_dir)]
    counts = [len(f) for f in lists]
    if len(set(counts)) != 1:
        raise InputError(
            "Klasörlerdeki maske sayıları farklı: "
            + ", ".join(f"{d} ({c})" for d, c in zip([*color_dirs, gradient_dir], counts))
            + "\n  -> Renk ve gradyan maskeleri aynı karelerden üretilmeli (sıralı adla eşleşir).")
    sub_dir = out_dir / Path(color_dirs[0]).name
    sub_dir.mkdir(parents=True, exist_ok=True)
    totals, result_totals = dict.fromkeys(STAT_KEYS, 0.0), {}
    for group in zip(*lists):
        stats, ratios = process_images(group[:-1], group[-1], sub_dir, args, base_name(group[0]))
        print(f"{group[0].name} + {group[-1].name}: {format_stats(stats)}")
        for k in STAT_KEYS:
            totals[k] += stats[k]
        for m, v in ratios.items():
            result_totals[m] = result_totals.get(m, 0.0) + v
    n = counts[0]
    return {k: v / n for k, v in totals.items()}, {m: v / n for m, v in result_totals.items()}, n, sub_dir


def process_videos(color_paths, gradient_path, out_dir, args):
    caps = []
    for p in [*color_paths, gradient_path]:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            for c in caps:
                c.release()
            raise InputError(f"Video açılamadı: {p}\n  -> Dosyanın bozuk olmadığını kontrol et.")
        caps.append(cap)
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    if not (math.isfinite(fps) and fps > 0):
        fps = 25.0
    name = base_name(color_paths[0])
    names = [*map(str, color_paths), str(gradient_path)]
    writers, totals, result_totals, count = {}, dict.fromkeys(STAT_KEYS, 0.0), {}, 0
    try:
        while args.max_frames is None or count < args.max_frames:
            reads = [c.read() for c in caps]
            ok = [r[0] for r in reads]
            if not any(ok):
                break
            if not all(ok):
                ended = [n for n, o in zip(names, ok) if not o]
                print(f"UYARI: kare sayıları farklı; {', '.join(ended)} {count}. karede bitti, "
                      "orada durduruldu.", file=sys.stderr)
                break
            frames = [r[1] for r in reads]
            color, gradient = load_set(frames[:-1], frames[-1], names, args)
            results, stats = process_pair(color, gradient, args)
            height, width = color.shape
            if not writers:
                for mode in results:
                    # [EK] FFV1 kayıpsızdır; mp4 maskenin 0/255 değerlerini bozar.
                    writers[mode] = open_writer(out_dir / f"{name}_{mode}_maske.mkv", "FFV1", fps,
                                                (width, height), is_color=False)
                if args.compare:
                    writers["compare"] = open_writer(
                        out_dir / f"{name}_karsilastirma.mp4", "mp4v", fps,
                        ((2 + len(results)) * width, height), is_color=True)
            for mode, mask in results.items():
                writers[mode].write(mask.astype(np.uint8) * 255)
            if args.compare:
                writers["compare"].write(draw_compare(color, gradient, results, args))
            for k in STAT_KEYS:
                totals[k] += stats[k]
            for m, r in results.items():
                result_totals[m] = result_totals.get(m, 0.0) + float(r.mean())
            count += 1
    finally:
        for c in caps:
            c.release()
        for w in writers.values():
            w.release()
    if count == 0:
        raise InputError(f"Videolardan hiç kare okunamadı: {', '.join(names)}")
    return {k: v / count for k, v in totals.items()}, {m: v / count for m, v in result_totals.items()}, count


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
        description="Renk ve gradyan maskelerini OR / AND ile tek ikili maskede birleştirir.")
    p.add_argument("--color-masks", type=Path, nargs="+", default=COLOR_MASKS,
                   help="Renk maske(ler)i: görüntü, klasör ya da video; birden çoksa OR'lanır "
                        f"(varsayılan: {', '.join(map(str, COLOR_MASKS))})")
    p.add_argument("--gradient-mask", type=Path, default=GRADIENT_MASK,
                   help=f"Gradyan maskesi (varsayılan: {GRADIENT_MASK})")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Çıktı klasörü (varsayılan: {OUTPUT_DIR})")
    p.add_argument("--mode", choices=("or", "and", "both"), default=MODE,
                   help=f"Birleştirme: or, and ya da ikisi (varsayılan: {MODE})")
    p.add_argument("--morph", choices=("none", "open", "close"), default=MORPH,
                   help=f"Morfolojik temizlik (varsayılan: {MORPH})")
    p.add_argument("--morph-ksize", type=int, default=MORPH_KSIZE,
                   help=f"Morfoloji çekirdek boyutu, px (varsayılan: {MORPH_KSIZE})")
    p.add_argument("--morph-shape", choices=("auto", "rect", "vertical"), default=MORPH_SHAPE,
                   help=f"Çekirdek şekli; auto: açmada 1 x k, kapamada k x k (varsayılan: {MORPH_SHAPE})")
    p.add_argument("--mask-threshold", type=int, default=MASK_THRESHOLD,
                   help=f"Maske eşiği, 0-254 (varsayılan: {MASK_THRESHOLD})")
    p.add_argument("--max-midtone", type=float, default=MAX_MIDTONE,
                   help=f"İkili maske sayılacak en büyük ara ton oranı (varsayılan: {MAX_MIDTONE})")
    p.add_argument("--compare", action="store_true",
                   help="Renk, gradyan ve sonuç(lar)ı yan yana kaydet")
    p.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                   help="Videoda en çok kaç kare işlensin (varsayılan: tümü)")
    return p.parse_args(argv)


def validate_args(args):
    if args.morph_ksize <= 0:
        raise InputError(f"--morph-ksize pozitif olmalı, verilen: {args.morph_ksize}")
    if not 0 <= args.mask_threshold <= 254:
        raise InputError(f"--mask-threshold 0-254 aralığında olmalı, verilen: {args.mask_threshold}")
    if not (math.isfinite(args.max_midtone) and 0 <= args.max_midtone <= 1):
        raise InputError(f"--max-midtone 0-1 aralığında olmalı, verilen: {args.max_midtone}")
    if args.max_frames is not None and args.max_frames <= 0:
        raise InputError(f"--max-frames pozitif olmalı, verilen: {args.max_frames}")


# --------------------------------------------------------------------------- ana akış
def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    paths = [*args.color_masks, args.gradient_mask]
    kinds = {source_kind(p) for p in paths}
    if len(kinds) != 1:
        raise InputError(
            "Renk ve gradyan maskeleri aynı türde olmalı (hepsi görüntü, hepsi klasör ya da hepsi "
            f"video); verilen: {', '.join(f'{p} ({source_kind(p)})' for p in paths)}")
    kind = kinds.pop()
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"Çıktı klasörü oluşturulamadı: {args.out_dir} ({exc})\n"
                         "  -> --out-dir'i kontrol et.") from None

    kw, kh = morph_kernel_size(args.morph, args.morph_ksize, args.morph_shape)
    morph = f", morfoloji {args.morph} {kw}x{kh} (genişlik x yükseklik)" if args.morph != "none" else ""
    print(f"Mod: {args.mode}{morph}; renk maskesi {len(args.color_masks)} adet")
    if kind == "image":
        stats, ratios = process_images(args.color_masks, args.gradient_mask, args.out_dir, args,
                                       base_name(args.color_masks[0]))
        print(f"Kaynaklar: {format_stats(stats)}")
        where = args.out_dir
    elif kind == "folder":
        stats, ratios, n, where = process_folders(args.color_masks, args.gradient_mask, args.out_dir, args)
        print(f"{n} kare, ortalama kaynaklar: {format_stats(stats)}")
    else:
        stats, ratios, n = process_videos(args.color_masks, args.gradient_mask, args.out_dir, args)
        print(f"{n} kare, ortalama kaynaklar: {format_stats(stats)}")
        where = args.out_dir
    print("Sonuç maskesi doluluğu: " + ", ".join(f"{m.upper()} %{100 * v:.2f}" for m, v in ratios.items())
          + f" -> {where}")


if __name__ == "__main__":
    try:
        main()
    except InputError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        sys.exit(1)
