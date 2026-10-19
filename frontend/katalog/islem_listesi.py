"""
Katalog — işlemlerin listesi: Tek işlem sayfasındaki 24 işlem (Kurulum adımları, Şerit kolu, Engel kolu).

Her işlem bir Operation'dır: gereksinim modülü, girdileri (varsayılanları modülün kendi AYARLAR bölümünden), arayüzde
ayarlanabilen ayarları, her zaman verilen argümanları ve Sonuç'ta gösterilecek çıktıları. Yeni bir işlem eklemek ya
da bir işlemin girdisini, ayarını, çıktısını değiştirmek için yalnız bu dosya güncellenir.
"""

from pathlib import Path

from .degerler import (calibration_metrics, coefficient_metrics, curvature_metrics, detection_metrics,
                       offset_metrics, polynomial_metrics, relation_metrics, scale_metrics)
from .islemler import Input, Operation, Setting, View
from .kurulumlar import setup_files, yolo_models
from .moduller import load_module, module_defaults
from .yollar import (DEFAULT_SETUP, GROUPS, IMAGE_EXTENSIONS, MAX_FRAMES_LIMIT, OUTPUTS_DIR, SCRIPTS,
                     VIDEO_EXTENSIONS, VIDEOS_DIR)


# --------------------------------------------------------------------------- yardımcılar
def _stem(path: Path, strip: tuple = ()) -> str:
    stem = Path(path).stem
    for suffix in strip:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _out(op_key: str, attr: str) -> Path:
    return Path(getattr(module_defaults(SCRIPTS[op_key]), attr))


def is_media_file(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + (".npz", ".csv")


def _sub(out_dir: Path, path: Path) -> Path:
    """Klasör girdisinde modüller çıktıyı girdi klasörünün adıyla bir alt klasöre yazar."""
    path = Path(path)
    return out_dir / path.name if path.is_dir() or not (path.is_file() or is_media_file(path)) else out_dir


METHOD_NAMES = {"sobel": "Sobel-x", "canny": "Canny"}
I = Input
S = Setting

OPERATIONS: dict[str, Operation] = {}


def _add(op: Operation) -> None:
    OPERATIONS[op.key] = op


# ---------------------------------------------------------------- Kurulum adımları
try:                                   # iç köşe sınırı 1.1'in kendi kuralı (modül yüklenemezse sayfalar yine açılsın)
    _PATTERN_LIMITS = tuple(load_module(SCRIPTS["1.1-kal"]).PATTERN_LIMITS)
except Exception:
    _PATTERN_LIMITS = (3, 20)
_add(Operation(
    "1.1-kal", "1.1", "Kalibrasyon", GROUPS[0],
    "Satranç tahtası fotoğraflarından lensin görüntüyü nasıl büktüğünü ölçer.",
    SCRIPTS["1.1-kal"],
    inputs=(I("--images", "Satranç tahtası fotoğrafları", dest="images", real_data=True),),
    settings=(S("--pattern", "İç köşe", "pattern", "pair_int", low=_PATTERN_LIMITS[0], high=_PATTERN_LIMITS[1],
                parts=("sütun", "satır"),
                help=f"Tahtanın iç köşe sayısı (kare sayısının bir eksiği: 10x7 kareli tahta 9x6). Her kenar "
                     f"{_PATTERN_LIMITS[0]}–{_PATTERN_LIMITS[1]}; en doğru sonuç 5x4 ile 18x12 arasında."),),
    fixed=("--corners-dir",
           OUTPUTS_DIR / "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme" / "koseler" / DEFAULT_SETUP),
    views=lambda inp, val: [
        View("metrics", _out("1.1-kal", "output"), metrics=calibration_metrics),
        View("npz", _out("1.1-kal", "output"), label="Kalibrasyon dosyası"),
        View("images", OUTPUTS_DIR / "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme" / "koseler" / DEFAULT_SETUP,
             "*_corners.*", "Köşe kontrol görüntüleri")],
    protected=lambda inp, val: [_out("1.1-kal", "output")],
))
_add(Operation(
    "1.1-duz", "1.1", "Düzeltme", GROUPS[0],
    "Kalibrasyonla karelerden lens bozulmasını kaldırır; düz çizgiler düz görünür.",
    SCRIPTS["1.1-duz"],
    inputs=(I("--input", "Kareler (görüntü, klasör ya da video)", dest="input", real_data=True),
            I("--calib", "Kalibrasyon dosyası", dest="calib", producer="1.1-kal")),
    views=lambda inp, val: [View("images", _out("1.1-duz", "output"), "*_undistorted.*", "Düzeltilmiş kareler",
                                 pairs=inp[0])],
))
_add(Operation(
    "1.2-persp", "1.2", "Perspektif hesabı", GROUPS[0],
    "Düz yol karesindeki 4 noktadan kuş bakışı matrisini (M, Minv) hesaplar.",
    SCRIPTS["1.2-persp"],
    inputs=(I("--image", "Düzeltilmiş düz yol karesi", dest="image", producer="1.1-duz"),),
    settings=(S("--src", "4 kaynak nokta", "src", "points",
                help="Düz yolda şerit çizgilerinin üzerinde, yolu çevreleyen yamuğun köşeleri (piksel): sol üst, "
                     "sağ üst, sağ alt, sol alt."),),
    views=lambda inp, val: [
        View("npz", _out("1.2-persp", "output"), label="Perspektif dosyası"),
        View("images", _out("1.2-persp", "verify_dir"), f"{Path(inp[0]).stem}_*", "Kontrol görüntüsü ve kuş bakışı")],
    protected=lambda inp, val: [_out("1.2-persp", "output")],
))
_KB_DUZ = OUTPUTS_DIR / "1.2 Perspektif dönüşümü" / "kus_bakisi_duz_yol"
_add(Operation(
    "1.2-duzyol", "1.2", "Kuş bakışı (düz yol)", GROUPS[0],
    "Düz yol karelerini kuş bakışına çevirir; 1.3 ölçeği bu karelerde ölçer.",
    SCRIPTS["1.2-warp"],
    inputs=(I("", "Düzeltilmiş düz yol kareleri",
              default=OUTPUTS_DIR / "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme" / "duzeltilmis_kareler",
              producer="1.1-duz", pattern="duz_yol_*_undistorted.*"),
            I("--transform", "Perspektif dosyası", dest="transform", producer="1.2-persp")),
    commands=lambda inp, val: [
        (SCRIPTS["1.2-warp"], ["--transform", inp[1], "--input", frame, "--output", _KB_DUZ])
        for frame in sorted(Path(inp[0]).glob("duz_yol_*_undistorted.*"))],
    views=lambda inp, val: [View("images", _KB_DUZ, "*_warped.*", "Kuş bakışı düz yol kareleri", pairs=inp[0])],
))
_add(Operation(
    "1.3", "1.3", "Ölçek", GROUPS[0],
    "Kuş bakışında şerit genişliğini ve kesik çizgi boyunu ölçüp 1 pikselin kaç metre olduğunu bulur.",
    SCRIPTS["1.3"],
    inputs=(I("--input", "Kuş bakışı düz yol kareleri", dest="input", producer="1.2-duzyol"),),
    settings=(S("--lane-width-m", "Şerit genişliği (m)", "lane_width_m", "float", low=2.0, high=5.0, step=0.01,
                help="Yolun gerçek ölçüsü (ABD otoyolu 3.66 m)."),
              S("--dash-length-m", "Kesik çizgi boyu (m)", "dash_length_m", "float", low=0.5, high=10.0, step=0.01,
                help="Yolun gerçek ölçüsü (ABD otoyolu 3.05 m).")),
    fixed=("--compare",),
    views=lambda inp, val: [
        View("metrics", _out("1.3", "out"), metrics=scale_metrics),
        View("npz", _out("1.3", "out"), label="Ölçek dosyası"),
        View("images", _out("1.3", "out_dir"), "*_karsilastirma.png", "Karşılaştırma görüntüsü")],
    protected=lambda inp, val: [_out("1.3", "out")],
))

# ---------------------------------------------------------------- Şerit kolu (test kareleri)
_add(Operation(
    "2.1", "2.1", "Renk eşikleme", GROUPS[1],
    "Beyaz ve sarı şerit çizgilerini renklerinden (HLS-L, LAB-B) ayırıp maske çıkarır.",
    SCRIPTS["2.1"],
    inputs=(I("--input", "Düzeltilmiş kareler", dest="input", producer="1.1-duz", pattern="*_undistorted.*"),),
    settings=(S("--white-l-range", "Beyaz: HLS-L aralığı", "white_l_range", "pair_int", low=0, high=255),
              S("--yellow-b-range", "Sarı: LAB-B aralığı", "yellow_b_range", "pair_int", low=0, high=255)),
    views=lambda inp, val: [View("images", _sub(_out("2.1", "out_dir"), inp[0]), "*_maske.png", "Renk maskeleri",
                                 pairs=inp[0])],
))
_add(Operation(
    "2.2", "2.2", "Kenar algılama", GROUPS[1],
    "Parlaklığın birden değiştiği yerlerden (Sobel-x ya da Canny) kenar maskesi çıkarır.",
    SCRIPTS["2.2"],
    inputs=(I("--input", "Kareler", dest="input", real_data=True),
            I("--calibration", "Kalibrasyon dosyası", dest="calibration", producer="1.1-kal")),
    settings=(S("--method", "Yöntem", "method", "choice", ("sobel", "canny"), METHOD_NAMES),),
    views=lambda inp, val: [View("images", _sub(_out("2.2", "out_dir"), inp[0]),
                                 f"*_{'canny' if val.get('method') == 'canny' else 'sobelx'}_maske.png",
                                 "Kenar maskeleri", pairs=inp[0])],
))
_add(Operation(
    "4.1", "4.1", "Yumuşatma", GROUPS[1],
    "Kareyi Gaussian ya da Bilateral filtreyle yumuşatıp sahte kenarları azaltır.",
    SCRIPTS["4.1"],
    inputs=(I("--input", "Kareler", dest="input", real_data=True),
            I("--calibration", "Kalibrasyon dosyası", dest="calibration", producer="1.1-kal")),
    settings=(S("--filter", "Filtre", "filter", "choice", ("gaussian", "bilateral"),
                {"gaussian": "Gaussian", "bilateral": "Bilateral"}),),
    views=lambda inp, val: [View("images", _sub(_out("4.1", "out_dir"), inp[0]), f"*_{val.get('filter', 'gaussian')}.png",
                                 "Yumuşatılmış kareler", pairs=inp[0])],
))
_add(Operation(
    "4.2", "4.2", "Renk kanalları", GROUPS[1],
    "Gölgede de çalışan renk maskesi: beyaz için L, sarı için HLS-S ve LAB-B birlikte.",
    SCRIPTS["4.2"],
    inputs=(I("--input", "Düzeltilmiş kareler", dest="input", producer="1.1-duz", pattern="*_undistorted.*"),),
    settings=(S("--clahe", "Parlaklığa yerel kontrast artırma (CLAHE)", "clahe", "flag"),),
    views=lambda inp, val: [View("images", _sub(_out("4.2", "out_dir"), inp[0]), "*_kanal_maske.png", "Renk maskeleri",
                                 pairs=inp[0])],
))
_add(Operation(
    "4.3", "4.3", "Gradyan", GROUPS[1],
    "Kenar maskesinden gölge sınırlarını ayıklar (çift kenar filtresi).",
    SCRIPTS["4.3"],
    inputs=(I("--input", "Kareler", dest="input", real_data=True),
            I("--calibration", "Kalibrasyon dosyası", dest="calibration", producer="1.1-kal")),
    settings=(S("--method", "Yöntem", "method", "choice", ("sobel", "canny"), METHOD_NAMES),
              S("--no-pair-filter", "Çift kenar filtresi (gölge sınırlarını atar)", "pair_filter", "noflag")),
    views=lambda inp, val: [View("images", _sub(_out("4.3", "out_dir"), inp[0]),
                                 f"*_{'canny' if val.get('method') == 'canny' else 'sobelx'}_maske.png",
                                 "Kenar maskeleri", pairs=inp[0])],
))
_add(Operation(
    "4.4", "4.4", "Birleştirme", GROUPS[1],
    "Renk ve gradyan maskelerini OR / AND ile tek şerit maskesinde birleştirir.",
    SCRIPTS["4.4"],
    inputs=(I("--color-masks", "Renk maskeleri", dest="color_masks", producer="4.2", pattern="*_kanal_maske.png"),
            I("--gradient-mask", "Gradyan maskeleri", dest="gradient_mask", producer="4.3", pattern="*_maske.png")),
    settings=(S("--mode", "Birleştirme", "mode", "choice", ("or", "and", "both"),
                {"or": "OR — biri yeterli", "and": "AND — ikisi de", "both": "İkisi de (karşılaştırma)"}),
              S("--morph", "Morfolojik temizlik", "morph", "choice", ("none", "open", "close"),
                {"none": "Yok", "open": "Açma (küçük gürültü)", "close": "Kapama (boşluk doldur)"})),
    views=lambda inp, val: [View("images", _sub(_out("4.4", "out_dir"), inp[0]),
                                 "*_maske.png" if val.get("mode") == "both" else f"*_{val.get('mode', 'or')}_maske.png",
                                 "Birleşik maskeler")],
))
_KB_MASKE = OUTPUTS_DIR / "1.2 Perspektif dönüşümü" / "kus_bakisi_maskeler"
_add(Operation(
    "1.2-maske", "1.2", "Kuş bakışı (maskeler)", GROUPS[1],
    "2.1'in renk maskelerini kuş bakışına çevirir; 2.3 şerit piksellerini bu maskelerde arar.",
    SCRIPTS["1.2-warp"],
    inputs=(I("--input", "Renk maskeleri (2.1)",
              default=OUTPUTS_DIR / "2.1 HLS∕HSV∕LAB ve eşikleme" / "duzeltilmis_kareler", producer="2.1",
              pattern="*_maske.png"),
            I("--transform", "Perspektif dosyası", dest="transform", producer="1.2-persp")),
    fixed=("--output", _KB_MASKE),
    views=lambda inp, val: [View("images", _KB_MASKE, "*_maske_warped.png", "Kuş bakışı maskeler", pairs=inp[0])],
))
_add(Operation(
    "2.3", "2.3", "Sliding Windows", GROUPS[1],
    "Kuş bakışı maskede sol ve sağ şerit piksellerini kayan pencerelerle ayırır.",
    SCRIPTS["2.3"],
    inputs=(I("--input", "Kuş bakışı maskeler", dest="input", producer="1.2-maske", pattern="*.png"),),
    settings=(S("--windows", "Pencere sayısı", "windows", "int", low=3, high=30),
              S("--margin", "Pencere yarı genişliği (görüntü genişliğine oran)", "margin", "float", low=0.02,
                high=0.3, step=0.01)),
    views=lambda inp, val: [View("images", _out("2.3", "output"), "*_windows.png", "Kayan pencereler", pairs=inp[0]),
                            View("file", _out("2.3", "output"), "*_lane_pixels.npz", "Şerit pikselleri (2.4 okur)")],
))
_add(Operation(
    "2.4", "2.4", "Polinom uydurma", GROUPS[1],
    "Şerit piksellerinden x = A·y² + B·y + C eğrisini uydurur.",
    SCRIPTS["2.4"],
    inputs=(I("--input", "Şerit pikselleri (2.3)", dest="input", producer="2.3"),),
    views=lambda inp, val: [
        View("metrics", _out("2.4", "out_dir") / f"{Path(inp[0]).stem}_polinom.npz", metrics=polynomial_metrics),
        View("npz", _out("2.4", "out_dir") / f"{Path(inp[0]).stem}_polinom.npz", label="Polinom dosyası")],
))
_add(Operation(
    "2.5", "2.5", "Yanal sapma", GROUPS[1],
    "Aracın şerit ortasına göre ne kadar sağa ya da sola kaydığını metre cinsinden bulur.",
    SCRIPTS["2.5"],
    inputs=(I("--input", "Polinomlar (2.4)", dest="input", producer="2.4", pattern="*_polinom.npz"),
            I("--scale", "Ölçek dosyası", dest="scale", producer="1.3")),
    settings=(S("--warn-offset", "Şeritten çıkma uyarı sınırı (m)", "warn_offset", "float", low=0.1, high=2.0, step=0.05,
                help="Sapma bundan büyükse o kare için şeritten çıkma uyarısı verilir."),),
    fixed=("--visualize",),
    views=lambda inp, val: [
        View("metrics", _out("2.5", "output") / "lane_offset.csv", metrics=offset_metrics),
        View("csv", _out("2.5", "output") / "lane_offset.csv", label="Sapma tablosu"),
        View("images", _out("2.5", "output"), "*_offset.png", "Kontrol görüntüsü")],
))
for key, name, purpose, producer, out_name, metric in (
        ("3.1", "Polinom katsayıları", "A katsayısından virajın yönünü ve sertliğini (düz / hafif / sert) bulur.",
         "2.4", "katsayi", coefficient_metrics),
        ("3.2", "Metre polinomu", "Şerit polinomunu metre cinsinden yeniden uydurur (piksel ve metre yarıçapı).",
         "2.3", "metre_polinom", None),
        ("3.3", "Eğrilik yarıçapı", "Virajın yarıçapını (R, metre) ve yönünü hesaplar.", "2.4", "egrilik",
         curvature_metrics)):
    _add(Operation(
        key, key, name, GROUPS[1], purpose, SCRIPTS[key],
        inputs=(I("--input", "Polinom (2.4)" if producer == "2.4" else "Şerit pikselleri (2.3)", dest="input",
                  producer=producer),
                I("--scale", "Ölçek dosyası", dest="scale", producer="1.3")),
        fixed=("--compare",),
        views=(lambda out_name, metric, key: lambda inp, val: [
            *([View("metrics", _out(key, "out_dir") / f"{Path(inp[0]).stem}_{out_name}.npz", metrics=metric)]
              if metric else []),
            View("npz", _out(key, "out_dir") / f"{Path(inp[0]).stem}_{out_name}.npz", label="Sonuç dosyası"),
            View("images", _out(key, "out_dir"), f"{Path(inp[0]).stem}_{out_name}_karsilastirma.png",
                 "Karşılaştırma grafiği")])(out_name, metric, key),
    ))

# ---------------------------------------------------------------- Engel kolu (kolay video)
try:                                   # 5.1'in güven eşiği: 5.3'ün eşiği bunun altında etkisizdir
    _YOLO_CONF = float(module_defaults(SCRIPTS["5.1"]).conf)
except Exception:
    _YOLO_CONF = 0.25
_add(Operation(
    "5.1", "5.1", "YOLO tespiti", GROUPS[2],
    "Karedeki araç, yaya gibi nesneleri YOLO ile bulur (kutu, sınıf, güven).",
    SCRIPTS["5.1"],
    inputs=(I("--input", "Video ya da görüntü", dest="input", real_data=True),
            I("--model", "YOLO modeli", dest="model", options=yolo_models),
            I("--calibration", "Kalibrasyon dosyası", dest="calibration", producer="1.1-kal")),
    settings=(S("--max-frames", "En çok kaç kare (boş: tümü)", "max_frames", "optional_int", low=1, high=MAX_FRAMES_LIMIT,
                help="Videonun yalnız başını işlemek için (daha hızlı); boşsa bütün video."),),
    views=lambda inp, val: [
        View("metrics", _out("5.1", "out_dir") / f"{Path(inp[0]).stem}_tespit.npz", metrics=detection_metrics),
        View("npz", _out("5.1", "out_dir") / f"{Path(inp[0]).stem}_tespit.npz", label="Tespit dosyası")],
))
_add(Operation(
    "5.2", "5.2", "Kutu koordinatları", GROUPS[2],
    "Kutuların koordinatlarını ve nesnenin yola değdiği alt-orta noktayı çıkarır.",
    SCRIPTS["5.2"],
    inputs=(I("--input", "Tespitler (5.1)", dest="input", producer="5.1"),),
    fixed=("--csv",),
    views=lambda inp, val: [
        View("csv", _out("5.2", "out_dir") / f"{_stem(inp[0], ('_tespit',))}_kutular.csv", label="Kutu tablosu"),
        View("file", _out("5.2", "out_dir") / f"{_stem(inp[0], ('_tespit',))}_kutular.npz", label="Kutu dosyası")],
))
_add(Operation(
    "5.3", "5.3", "Güven / sınıf filtresi", GROUPS[2],
    "Emin olunmayan ve sürüşle ilgisiz tespitleri eler.",
    SCRIPTS["5.3"],
    inputs=(I("--input", "Kutular (5.2)", dest="input", producer="5.2"),),
    settings=(S("--conf", "Güven eşiği", "conf", "float", low=_YOLO_CONF, high=0.95, step=0.05,
                help=f"Bundan emin olunmayan tespitler elenir. 5.1 {_YOLO_CONF:g}'in altındakileri zaten bulmaz; "
                     "bu yüzden en az o değer."),),
    views=lambda inp, val: [
        View("metrics", _out("5.3", "out_dir") / f"{_stem(inp[0], ('_kutular', '_tespit'))}_filtreli.npz",
             metrics=detection_metrics),
        View("npz", _out("5.3", "out_dir") / f"{_stem(inp[0], ('_kutular', '_tespit'))}_filtreli.npz",
             label="Filtrelenmiş tespitler")],
))
_add(Operation(
    "5.4", "5.4", "NMS analizi", GROUPS[2],
    "Farklı IoU eşiklerinde kutu sayısını ve çift sayımı ölçüp NMS eşiğini seçtirir.",
    SCRIPTS["5.4"],
    inputs=(I("--input", "Video ya da görüntü", dest="input", real_data=True),
            I("--model", "YOLO modeli", dest="model", options=yolo_models)),
    settings=(S("--iou-values", "Denenecek IoU eşikleri", "iou_values", "floats",
                help="Her eşikte kalan kutu sayısı ve çift sayım karşılaştırılır."),),
    fixed=("--compare",),
    views=lambda inp, val: [
        View("csv", _out("5.4", "out_dir") / f"{Path(inp[0]).stem}_nms_analiz.csv", label="IoU tablosu"),
        View("images", _out("5.4", "out_dir"), f"{Path(inp[0]).stem}_nms_karsilastirma.png", "Karşılaştırma")],
))
_O11 = OUTPUTS_DIR / "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme"
_O21 = OUTPUTS_DIR / "2.1 HLS∕HSV∕LAB ve eşikleme"
_O24 = OUTPUTS_DIR / "2.4 Polinom uydurma"


def _video_lane_commands(inp, val):
    video, calibration, transform = (Path(p) for p in inp)
    stem = video.stem
    corrected = _O11 / "video_duzeltilmis"
    return [
        (SCRIPTS["1.1-duz"], ["--input", video, "--calib", calibration, "--output", corrected]),
        (SCRIPTS["2.1"], ["--input", corrected / f"{stem}_undistorted.mp4"]),
        (SCRIPTS["1.2-warp"], ["--transform", transform, "--input", _O21 / f"{stem}_undistorted_maske.mkv",
                               "--output", _KB_MASKE]),
        (SCRIPTS["2.4"], ["--input", _KB_MASKE / f"{stem}_undistorted_maske_warped.mp4"]),
    ]


_add(Operation(
    "video-serit", "1.1 → 2.1 → 1.2 → 2.4", "Video şerit hazırlığı", GROUPS[2],
    "Videonun her karesi için şerit polinomlarını üretir (düzeltme → renk maskesi → kuş bakışı → polinom); "
    "5.5 tespitleri bu şeritlerle eşleştirir.",
    SCRIPTS["2.4"],
    inputs=(I("", "Video", default=VIDEOS_DIR / DEFAULT_SETUP / "kolay_video_iyi_isik.mp4", real_data=True),
            I("", "Kalibrasyon dosyası", default=setup_files(DEFAULT_SETUP)["kalibrasyon"], producer="1.1-kal"),
            I("", "Perspektif dosyası", default=setup_files(DEFAULT_SETUP)["perspektif"], producer="1.2-persp")),
    commands=_video_lane_commands,
    views=lambda inp, val: [
        View("metrics", _O24 / f"{Path(inp[0]).stem}_undistorted_maske_warped_polinom.npz", metrics=polynomial_metrics),
        View("npz", _O24 / f"{Path(inp[0]).stem}_undistorted_maske_warped_polinom.npz", label="Kare kare polinomlar")],
))
_add(Operation(
    "5.5", "5.5", "Şerit–engel ilişkisi", GROUPS[2],
    "Nesnenin bizim şeridimizde olup olmadığını, mesafesini ve riskini bulur.",
    SCRIPTS["5.5"],
    inputs=(I("--detections", "Filtrelenmiş tespitler (5.3)", dest="detections", producer="5.3"),
            I("--lanes", "Videonun şerit polinomları", dest="lanes", producer="video-serit"),
            I("--perspective", "Perspektif dosyası", dest="perspective", producer="1.2-persp"),
            I("--scale", "Ölçek dosyası", dest="scale", producer="1.3")),
    settings=(S("--red-m", "Kırmızı risk sınırı (m)", "red_m", "float", low=1.0, high=100.0, step=1.0,
                help="Şeritteki engel bundan yakınsa risk kırmızı."),
              S("--yellow-m", "Sarı risk sınırı (m)", "yellow_m", "float", low=1.0, high=200.0, step=1.0,
                help="Şeritteki engel bundan yakınsa (kırmızı değilse) risk sarı; kırmızı sınırdan büyük olmalı.")),
    fixed=("--csv",),
    views=lambda inp, val: [
        View("metrics", _out("5.5", "out_dir") / f"{_stem(inp[0], ('_filtreli', '_kutular', '_tespit'))}_serit_iliskisi.npz",
             metrics=relation_metrics),
        View("csv", _out("5.5", "out_dir") / f"{_stem(inp[0], ('_filtreli', '_kutular', '_tespit'))}_serit_iliskisi.csv",
             label="Tespit tablosu")],
))
