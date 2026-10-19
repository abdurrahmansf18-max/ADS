"""
Entegrasyon — gereksinim modüllerini klasörlerinden yükleyen yardımcı.

Ne işe yarar:
    main.py'nin gereksinim dosyalarını kullanabilmesini sağlar. Klasör adlarında boşluk ve nokta olduğu
    için Python'un normal import'u bu dosyaları yükleyemez.

Nasıl çalışır:
    Gereksinim numarasından dosya yolunu bulur (MODULE_FILES), dosyayı bir kez yükler ve sonraki
    çağrılarda aynı nesneyi döndürür. Kod klasörlerine __pycache__ yazdırmaz.

Bağlantılar:
    Yalnız main.py kullanır; gereksinim dosyalarının kendisine dokunmaz. Bir klasörün ya da dosyanın adı
    değişirse MODULE_FILES güncellenir.

Parametreler:
    load("1.1") gibi gereksinim numarası (komut satırı yoktur).
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # ADS/
BACKEND_DIR = Path(__file__).resolve().parents[1]    # ADS/backend/: gereksinim klasörleri burada

# Gereksinim -> (ana klasör, gereksinim klasörü, dosya). Ana hatta çağrılan dosyalar.
MODULE_FILES = {
    "1.1": ("Perspektif bozulması ve mesafe yanılsaması",
            "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme", "undistort.py"),
    "1.2": ("Perspektif bozulması ve mesafe yanılsaması", "1.2 Perspektif dönüşümü", "warp_perspective.py"),
    "1.3": ("Perspektif bozulması ve mesafe yanılsaması", "1.3 Piksel–metre ölçeklemesi",
            "piksel_metre_olcekleme.py"),
    "2.3": ("Şeritten sapma ve şerit merkezinin belirlenmesi", "2.3 Sliding Windows", "sliding_windows.py"),
    "2.4": ("Şeritten sapma ve şerit merkezinin belirlenmesi", "2.4 Polinom uydurma", "polinom_uydurma.py"),
    "2.5": ("Şeritten sapma ve şerit merkezinin belirlenmesi", "2.5 Şerit merkezi ve yanal sapma",
            "lane_offset.py"),
    "3.1": ("Virajin_sertliginin_hesaplanmasi", "3.1 Polinom katsayıları", "polinom_katsayilari.py"),
    "3.3": ("Virajin_sertliginin_hesaplanmasi", "3.3 Eğrilik yarıçapı", "egrilik_yaricapi.py"),
    "4.1": ("Degisen_isik_golge_ve_zemin_kosullari", "4.1 Gaussian Blur ve Bilateral Filter", "yumusatma.py"),
    "4.2": ("Degisen_isik_golge_ve_zemin_kosullari", "4.2 HLS∕HSV∕LAB renk kanalları", "renk_kanallari.py"),
    "4.3": ("Degisen_isik_golge_ve_zemin_kosullari", "4.3 Sobel gradyanı ve Canny", "gradyan_golge.py"),
    "4.4": ("Degisen_isik_golge_ve_zemin_kosullari", "4.4 Renk ve gradyan maskelerinin birleştirilmesi",
            "maske_birlestirme.py"),
    "5.1": ("Carpisma_riski_ve_engel_farkindaligi", "5.1 YOLO ile nesne tespiti", "yolo_tespit.py"),
    "5.2": ("Carpisma_riski_ve_engel_farkindaligi", "5.2 Bounding Box koordinatları", "kutu_koordinatlari.py"),
    "5.3": ("Carpisma_riski_ve_engel_farkindaligi", "5.3 Confidence Score ve sınıf bilgisi", "guven_sinif_filtresi.py"),
    "5.5": ("Carpisma_riski_ve_engel_farkindaligi", "5.5 Engel konumu ile şerit bölgesinin ilişkilendirilmesi",
            "serit_engel_iliskisi.py"),
}


def default_args(requirement: str):
    """Modülün komut satırı varsayılanları (kendi parse_args'ı ile); ana hat bunları ezebilir."""
    return load(requirement).parse_args([])

_loaded = {}


def module_path(requirement: str) -> Path:
    main_folder, requirement_folder, filename = MODULE_FILES[requirement]
    return BACKEND_DIR / main_folder / requirement_folder / filename


def load(requirement: str):
    """Gereksinim modülünü bir kez yükler ve aynı nesneyi döndürür."""
    if requirement in _loaded:
        return _loaded[requirement]
    if requirement not in MODULE_FILES:
        raise ValueError(f"Bilinmeyen gereksinim: {requirement} (bilinenler: {', '.join(MODULE_FILES)})")
    path = module_path(requirement)
    if not path.is_file():
        raise FileNotFoundError(
            f"Gereksinim modülü bulunamadı: {path}\n"
            "Klasör ya da dosya adı değiştiyse moduller.py içindeki MODULE_FILES'ı güncelle."
        )
    name = "ads_" + requirement.replace(".", "_")
    # Gereksinim klasörlerinde yalnız kod kalsın (kural 1): __pycache__ yazılmaz.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass gibi yapılar modülü sys.modules'te arar
    spec.loader.exec_module(module)
    _loaded[requirement] = module
    return module
