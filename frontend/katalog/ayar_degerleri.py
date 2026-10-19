"""
Katalog — ayar değerleri: ayarların başlangıç değerleri ve "Sabit ayarlar" satırları.

Başlangıç değerleri (setting_defaults): kurulum dosyasında kayıtlıysa o değer, yoksa modülün varsayılanı; Kurulum ve
Tek işlem sayfası aynı değeri kullanır. Sabit ayarlar (fixed_rows): modülün ayarlanamayan bütün ayarları, gerçekte
kullanılan değerleriyle ve nereden geldikleriyle (kendiliğinden, kurulumdan, sabit); ana hattın her karede
kullandığı modül ayarları da (PIPELINE_SETTINGS).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .islemler import Operation, setting_default
from .kurulumlar import setup_files
from .moduller import load_module, module_defaults, module_parser, parse_command, script_key
from .yollar import DEFAULT_SETUP, MODEL_PATH, SCRIPTS


# --------------------------------------------------------------------------- kurulum ayarlarının başlangıcı
# Kurulum işlemlerinin ayarları kurulum dosyasına da yazılır. Kurulum ve Tek işlem sayfası bu ayarları aynı
# değerle başlatır: dosyada kayıtlı değer (yeniden yapınca aynı ayarla başlar, dosya yanlışlıkla değişmez), dosya
# yoksa modülün varsayılanı.
SAVED_SETTINGS = {
    ("1.1-kal", "pattern"): ("kalibrasyon", lambda d: tuple(int(v) for v in d["pattern_size"])),
    ("1.2-persp", "src"): ("perspektif", lambda d: ([(int(round(x)), int(round(y))) for x, y in d["src_points"]],
                                                    tuple(int(v) for v in d["image_size"]))),
    ("1.3", "lane_width_m"): ("olcek", lambda d: float(d["lane_width_m"])),
    ("1.3", "dash_length_m"): ("olcek", lambda d: float(d["dash_length_m"])),
}
SETTING_NOTES = {
    "1.1-kal": ["**İç köşe sayısı** kullandığın tahtanınkiyle aynı girilir; kare sayısı değildir (10x7 kareli bir "
                "tahtanın iç köşe sayısı 9x6).",
                "**Lens modeli kendiliğinden seçilir:** normal (5 katsayı) ve geniş açılı (8 katsayı) model ikisi de "
                "denenir; geniş açılı model hatayı en az yarıya indiriyorsa o, yoksa normal model kullanılır.",
                "**Güvenilirlik:** Köşesi bulunamayan ya da boyutu farklı fotoğraflar atlanır. Geriye 10'dan az "
                "fotoğraf kalırsa ya da hata 1 pikselden büyükse dosya kaydedilmez; var olan kalibrasyon korunur."],
    "1.2-persp": ["Noktalar düz yolda şerit çizgilerinin üzerinde, yolu çevreleyen bir yamuk oluşturur (sol üst, sağ "
                  "üst, sağ alt, sol alt). Anlık önizlemede kuş bakışındaki iki çizgi dikey ve paralel çıkmalı.",
                  "Her nokta görüntünün içinde olmalı; sıra yanlışsa ya da yamuk bozuksa Çalıştır kilitlenir."],
    "1.3": ["Yolun gerçek ölçüleri girilir; başka bir ülkenin yolunda o ülkenin standardı (şerit 2–5 m, kesik çizgi "
            "0.5–10 m)."],
}
SAVED_NOTE = ("Başlangıç değerleri kurulum dosyasında kayıtlı olanlardır (yeniden çalıştırınca aynı ayarla başlar); "
              "dosya yoksa modülün varsayılanı. Kurulum ve Tek işlem sayfasında aynıdır.")


def _saved_setting(op_key: str, dest: str, setup: str):
    entry = SAVED_SETTINGS.get((op_key, dest))
    if entry is None:
        return None
    part, read = entry
    try:
        with np.load(setup_files(setup)[part], allow_pickle=False) as data:
            return read(data)
    except (OSError, KeyError, ValueError, TypeError):
        return None


def default_points(image_size: tuple[int, int]) -> list[tuple[int, int]]:
    """1.2'nin varsayılan noktaları: 1280x720'de modülün noktaları, başka boyutta modülün oranlarıyla."""
    persp = load_module(SCRIPTS["1.2-persp"])
    if tuple(image_size) == (1280, 720):
        return [tuple(int(v) for v in point) for point in persp.SRC_POINTS]
    return [(int(round(x)), int(round(y))) for x, y in persp.points_from_ratios(persp.DEFAULT_SRC_RATIOS, image_size)]


def setting_defaults(op: Operation, setup: str = DEFAULT_SETUP, image_size: tuple[int, int] | None = None) -> dict:
    """Ayarların başlangıç değerleri: kurulum dosyasında kayıtlıysa o değer, yoksa modülün varsayılanı.
    Noktalar yalnız aynı boyuttaki görüntü için kayıtlı olandır."""
    values = {}
    for item in op.settings:
        saved = _saved_setting(op.key, item.dest, setup)
        if item.kind == "points":
            size = tuple(image_size or (1280, 720))
            values[item.dest] = saved[0] if saved is not None and saved[1] == size else default_points(size)
        else:
            values[item.dest] = saved if saved is not None else setting_default(op, item)
    return values


# =========================================================================== sabit ayarlar
# Ayarlar bölümünde ayarlanabilir olanların yanında modülün bütün diğer ayarları da görünür ("Sabit ayarlar"): adı,
# gerçekte kullanılan değeri ve nereden geldiği. Liste modülün kendi komut satırı tanımından okunur (elle yazılmaz);
# değer, arayüzün kuracağı komut modülün kendi parse_args'ıyla çözülerek bulunur. Dosya yolları burada yoktur
# (onlar Girdi bölümünde ve soru işaretlerindedir).
AUTO, FROM_SETUP, FIXED = "kendiliğinden", "kurulumdan", "sabit"
SOURCE_HELP = {
    AUTO: "Kendiliğinden: sistem kendisi belirler (ölçerek ya da girdiden okuyarak).",
    FROM_SETUP: "Kurulumdan: kurulum dosyasından gelir; kurulum değişince kendisi de değişir.",
    FIXED: "Sabit: modülün kendi değeri ya da arayüzün her zaman verdiği değer; arayüzden değiştirilmez.",
}
SOURCE_COLORS = {AUTO: "blue", FROM_SETUP: "green", FIXED: "gray"}
SOURCES = {                         # (betik, ayar) -> kaynak; listede olmayanlar sabit
    ("1.1-kal", "lens_model"): AUTO,
    ("1.3", "lane_px"): AUTO, ("1.3", "dash_px"): AUTO, ("1.3", "image_size"): AUTO,
    ("2.5", "image_size"): AUTO, ("2.5", "xm_per_pix"): FROM_SETUP,
    ("3.2", "y_eval"): AUTO, ("3.3", "y_eval"): AUTO,
    ("5.1", "device"): AUTO, ("5.4", "device"): AUTO,
    ("main", "device"): AUTO, ("main", "lane_width_range"): FROM_SETUP,
}
NAME_OVERRIDES = {                  # "…'yı kapat" biçimindeki bayraklar: değer açık / kapalı okunabilsin
    ("2.3", "momentum"): "Yetersiz pikselli pencerede son kaymayla devam",
    ("4.3", "pair_filter"): "Çift kenar filtresi (gölge sınırlarını atar)",
    ("5.5", "lane_width_fallback"): "Tek şerit varken genişlikten tahmin",
    ("main", "obstacles"): "Engel hattı (5.1–5.5)",
    ("2.1", "white_l_range"): "Beyaz: HLS-L aralığı",            # Tek işlem'deki ayarlanabilir adlarla aynı
    ("2.1", "white_s_range"): "Beyaz: HLS-S aralığı (kanal S seçilirse)",
    ("2.1", "yellow_b_range"): "Sarı: LAB-B aralığı",
}
SCRIPT_NAMES = {
    "1.1-kal": "1.1 Kalibrasyon", "1.1-duz": "1.1 Düzeltme", "1.2-persp": "1.2 Perspektif hesabı",
    "1.2-warp": "1.2 Kuş bakışı", "1.3": "1.3 Ölçek", "2.1": "2.1 Renk eşikleme", "2.2": "2.2 Kenar algılama",
    "2.3": "2.3 Sliding Windows", "2.4": "2.4 Polinom uydurma", "2.5": "2.5 Yanal sapma",
    "3.1": "3.1 Polinom katsayıları", "3.2": "3.2 Metre polinomu", "3.3": "3.3 Eğrilik yarıçapı",
    "4.1": "4.1 Yumuşatma", "4.2": "4.2 Renk kanalları", "4.3": "4.3 Gradyan", "4.4": "4.4 Birleştirme",
    "5.1": "5.1 YOLO tespiti", "5.2": "5.2 Kutu koordinatları", "5.3": "5.3 Güven / sınıf filtresi",
    "5.4": "5.4 NMS analizi", "5.5": "5.5 Şerit–engel ilişkisi", "main": "Ana hat (main.py)",
}
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck",
              8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign"}


@dataclass(frozen=True)
class FixedRow:
    group: str                      # başlık (birden çok modül gösteriliyorsa, ör. "4.1 Yumuşatma")
    name: str
    value: str
    source: str                     # AUTO, FROM_SETUP ya da FIXED
    help: str


def _is_path(action) -> bool:
    default = action.default
    if isinstance(default, (list, tuple)) and default:
        default = default[0]
    return action.type is Path or isinstance(default, Path)


def _help_text(action) -> str:
    text = action.help or ""
    try:
        params = dict(vars(action), prog="")
        if params.get("choices") is not None:
            params["choices"] = ", ".join(map(str, params["choices"]))
        return text % params
    except (TypeError, ValueError, KeyError):
        return text


def _setting_name(key: str, action, short: bool = True) -> str:
    """Ayarın adı, modülün kendi açıklamasından: parantezler çıkarılır, ilk cümle (kısa: ilk ":" ya da ";" öncesi)."""
    if (key, action.dest) in NAME_OVERRIDES:
        return NAME_OVERRIDES[(key, action.dest)]
    text, previous = _help_text(action), None
    while text != previous:                             # iç içe parantezler: (varsayılan: (15, 35))
        previous, text = text, re.sub(r"\s*\([^()]*\)", "", text)
    for separator in ((": ", "; ", ", ör.", ". ") if short else ("; ", ", ör.", ". ")):
        text = text.split(separator)[0]
    text = text.strip().rstrip(",").strip()
    return text[:1].upper() + text[1:] if text else action.option_strings[0]


def _number(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def format_setting(value, action=None) -> str:
    """Ayar değerinin okunur hali: açık / kapalı, aralık (200–255), boyut (1280x720), noktalar ((x, y) …)."""
    metavar = tuple(action.metavar) if action is not None and isinstance(action.metavar, tuple) else ()
    if value is None:
        return "verilmedi"
    if isinstance(value, bool):
        return "açık" if value else "kapalı"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_number(v)}" for k, v in value.items()) or "yok"
    if isinstance(value, (list, tuple)):
        if not value:
            return "yok"
        if all(isinstance(v, (list, tuple)) for v in value):
            return " ".join("(" + ", ".join(_number(x) for x in point) + ")" for point in value)
        if len(value) == 8 and len(metavar) == 8:
            return " ".join(f"({_number(value[i])}, {_number(value[i + 1])})" for i in range(0, 8, 2))
        if len(value) == 2 and metavar == ("ALT", "UST"):
            return f"{_number(value[0])}–{_number(value[1])}"
        if len(value) == 2 and metavar == ("GENISLIK", "YUKSEKLIK"):
            return f"{value[0]}x{value[1]}"
        return " ".join(_number(v) for v in value)
    return _number(value)


def device_text(value, model: Path | None) -> str:
    """YOLO cihazı: "auto" ise bu bilgisayarda ve bu modelde ölçülüp seçilen cihaz."""
    if value != "auto":
        return f"{value} (elle)"
    try:
        detector = load_module(SCRIPTS["5.1"])
        name = Path(model).name if model else MODEL_PATH.name
        entry = next((e for e in detector.read_device_cache().values() if e.get("model") == name), None)
    except Exception:
        return "auto"
    return f"auto → {detector.device_label(entry['device'])}" if entry else "auto (ilk çalıştırmada ölçülür)"


def _scale_value(path, key: str) -> float | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            return float(data[key])
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _classes_text(value) -> str:
    if not value:
        return "tümü"
    return " · ".join(f"{v} {COCO_NAMES[int(v)]}" if str(v).isdigit() and int(v) in COCO_NAMES else str(v)
                      for v in value)


def _dst_text(args, value) -> str:
    """1.2'nin hedef noktaları: verilmezse görüntü genişliğinin %25–75'i, tam yükseklik (modülün kuralı)."""
    if value is not None:
        return format_setting(value)
    persp = load_module(SCRIPTS["1.2-persp"])
    size = None
    try:
        import cv2
        image = cv2.imread(str(args.image)) if getattr(args, "image", None) else None
        size = (image.shape[1], image.shape[0]) if image is not None else None
    except Exception:
        pass
    if size is None:
        return "genişliğin %25–75'i, tam yükseklik"
    points = persp.points_from_ratios(persp.DEFAULT_DST_RATIOS, size)
    return " ".join(f"({x:g}, {y:g})" for x, y in points) + " (genişliğin %25–75'i)"


DISPLAY = {                         # (betik, ayar) -> değerin gösterimi (args, değer) -> yazı
    ("1.2-persp", "dst"): _dst_text,
    ("1.3", "lane_px"): lambda a, v: "görüntüden ölçülür" if v is None else _number(v),
    ("1.3", "dash_px"): lambda a, v: "görüntüden ölçülür" if v is None else _number(v),
    ("1.3", "image_size"): lambda a, v: "görüntüden" if v is None else format_setting(v),
    ("2.5", "image_size"): lambda a, v: "polinom dosyasından" if v is None else format_setting(v),
    ("2.5", "xm_per_pix"): lambda a, v: (f"{_scale_value(a.scale, 'xm_per_pix'):.6f} m/px (ölçek dosyasından)"
                                         if v is None and a.scale and _scale_value(a.scale, "xm_per_pix")
                                         else format_setting(v)),
    ("3.2", "y_eval"): lambda a, v: "en alt satır" if v is None else _number(v),
    ("3.3", "y_eval"): lambda a, v: "en alt satır" if v is None else _number(v),
    ("5.1", "device"): lambda a, v: device_text(v, a.model),
    ("5.4", "device"): lambda a, v: device_text(v, a.model),
    ("5.1", "classes"): lambda a, v: _classes_text(v),
    ("5.3", "classes"): lambda a, v: _classes_text(v),
}


def fixed_rows(key: str, parser, args, skip=(), group: str = "", only=None, values: dict | None = None,
               display: dict | None = None, sources: dict | None = None, notes: dict | None = None) -> list[FixedRow]:
    """Bir modülün sabit ayar satırları: parser'daki her ayar (yardım, dosya yolları ve skip dışındakiler).
    only: yalnız bu ayarlar, bu sırayla; values: args yerine gösterilecek değerler (ör. modülün sabiti);
    notes: ayarın açıklamasına eklenecek not."""
    if parser is None:
        return []
    actions = [a for a in parser._actions if a.dest != "help" and a.dest not in skip and not _is_path(a)]
    if only is not None:
        by_dest = {a.dest: a for a in actions}
        actions = [by_dest[dest] for dest in only if dest in by_dest]
    display = {**DISPLAY, **(display or {})}
    names = [_setting_name(key, action) for action in actions]
    names = [_setting_name(key, action, short=False) if names.count(name) > 1 else name   # aynı ad: uzun hali
             for name, action in zip(names, actions)]
    rows = []
    for action, name in zip(actions, names):
        value = values[action.dest] if values and action.dest in values else getattr(args, action.dest, action.default)
        source = (sources or {}).get(action.dest) or SOURCES.get((key, action.dest), FIXED)
        hook = display.get((key, action.dest))
        shown = hook(args, value) if hook else format_setting(value, action)
        flag = action.option_strings[0] if action.option_strings else action.dest
        note = (notes or {}).get(action.dest)
        rows.append(FixedRow(group, name, shown, source,
                             f"`{flag}` — {_help_text(action)}" + (f"\n\n{note}" if note else "")
                             + f"\n\n{SOURCE_HELP[source]}"))
    return rows


def operation_fixed_rows(op: Operation, commands: list[tuple[Path, list]] | None) -> list[FixedRow]:
    """İşlemin sabit ayarları: komut(lar)ındaki her modül için, ayarlanabilir ayarlar ve girdiler dışında kalanlar,
    gerçekte kullanılacak değerleriyle. Komut kurulamıyorsa (ör. girdi yok) modülün varsayılanları ve arayüzün
    sabit argümanları gösterilir."""
    if not commands:
        commands = [(op.script, list(op.fixed))]
    first: dict[Path, list] = {}
    for script, args in commands:                       # aynı modül birden çok karede: biri yeter
        first.setdefault(Path(script), args)
    adjustable = {item.dest for item in op.settings} | {item.dest for item in op.inputs if item.dest}
    rows = []
    for script, args in first.items():
        key = script_key(script)
        module = load_module(script)
        parsed = parse_command(module, args) or module_defaults(script)
        rows += fixed_rows(key, module_parser(script), parsed, adjustable if script == Path(op.script) else (),
                           group=SCRIPT_NAMES.get(key, key) if len(first) > 1 else "")
    return rows


# Ana hattın her karede kullandığı modüller ve ayarları (main.py ile aynı): (betik, ayarlar, modül sabitleri,
# ana hattın verdiği değerler, engel kolundan mı). Sabit verilen ayarın değeri main.py'nin kullandığı modül
# sabitidir (ör. 2.3'ün SlidingWindowSearch() varsayılanları); verilmeyenler modülün komut satırı varsayılanı.
PIPELINE_SETTINGS = (
    ("4.1", ("filter", "gaussian_ksize", "gaussian_sigma", "bilateral_d", "bilateral_sigma_color",
             "bilateral_sigma_space"), {}, {}, False),
    ("4.2", ("channels", "white_l_range", "yellow_s_min", "yellow_h_range", "yellow_b_range", "clahe", "clahe_clip",
             "clahe_tiles"), {}, {}, False),
    ("4.3", ("blur", "method", "sobel_ksize", "sobel_range", "canny_thresholds", "pair_filter", "max_line_width"), {},
     {"blur": "none"}, False),
    ("4.4", ("morph", "morph_ksize", "morph_shape"), {}, {}, False),
    ("2.3", ("windows", "margin", "min_pixels", "min_peak", "momentum", "threshold"),
     {"windows": "N_WINDOWS", "margin": "MARGIN_RATIO", "min_pixels": "MIN_PIXELS_RATIO",
      "min_peak": "MIN_PEAK_RATIO", "momentum": "USE_MOMENTUM", "threshold": "MASK_THRESHOLD"}, {}, False),
    ("2.4", ("min_fit_pixels", "min_y_span"), {"min_fit_pixels": "MIN_FIT_PIXELS", "min_y_span": "MIN_Y_SPAN"}, {},
     False),
    ("2.5", ("camera_offset", "center_tolerance", "warn_offset"),
     {"camera_offset": "CAMERA_OFFSET_M", "center_tolerance": "CENTER_TOLERANCE_M",
      "warn_offset": "DEPARTURE_WARNING_M"}, {}, False),
    ("3.1", ("straight_max_a", "sharp_min_a", "lr_ratio_warn"),
     {"straight_max_a": "STRAIGHT_MAX_A", "sharp_min_a": "SHARP_MIN_A", "lr_ratio_warn": "LR_RATIO_WARN"}, {},
     False),
    ("3.3", ("straight_max_a", "lr_ratio_warn", "y_eval"),
     {"straight_max_a": "STRAIGHT_MAX_A", "lr_ratio_warn": "LR_RATIO_WARN"}, {"y_eval": None}, False),
    ("5.1", ("conf", "imgsz", "classes"), {}, {}, True),
    ("5.2", ("edge_margin",), {"edge_margin": "EDGE_MARGIN_PX"}, {}, True),
    ("5.3", ("conf", "classes", "class_conf"),
     {"conf": "CONF_THRESHOLD", "classes": "CLASSES", "class_conf": "CLASS_CONF"}, {}, True),
    ("5.5", ("red_m", "yellow_m", "offset_m", "lane_width_fallback"), {}, {}, True),
)
PIPELINE_NOTES = {
    ("4.3", "blur"): "Ana hat bulanıklaştırmayı 4.1'de bir kez yapar; 4.3 kendisi yapmaz.",
    ("3.3", "y_eval"): "Ana hat R'yi aracın en yakın satırında (en alt satır) hesaplar.",
}


def pipeline_module_rows(obstacles: bool) -> list[FixedRow]:
    """Ana hattın kullandığı modül ayarları, main.py'nin gerçekte kullandığı değerlerle."""
    rows = []
    for key, dests, constants, given, obstacle in PIPELINE_SETTINGS:
        if obstacle and not obstacles:
            continue
        script = SCRIPTS[key]
        module = load_module(script)
        values = {dest: getattr(module, name) for dest, name in constants.items()}
        values.update(given)
        notes = {dest: note for (k, dest), note in PIPELINE_NOTES.items() if k == key}
        rows += fixed_rows(key, module_parser(script), module_defaults(script), group=SCRIPT_NAMES[key],
                           only=dests, values=values, notes=notes)
    return rows
