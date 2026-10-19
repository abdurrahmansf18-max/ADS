"""
Katalog — yollar ve sabitler: projenin klasörleri, kurulum dosyalarının adları, gereksinim modüllerinin dosyaları.

Bir klasörün yeri değişirse yalnız burası değişir; arayüzün başka hiçbir yeri yolu kendisi kurmaz. Kurulum adı
(ör. kolay_video) her şeyi bağlar: data/ altındaki dört klasör ve kurulum_dosyalari/<ad>/ (bkz. kurulumlar.py).
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
DATA_DIR = PROJECT_ROOT / "data"
ENV_DATA_DIR = DATA_DIR / "ortam_verisi"            # yolun kendisi: videolar ve onlardan seçilen kareler
READY_DATA_DIR = DATA_DIR / "hazir_veri"            # olduğu gibi kullanılan: satranç fotoğrafları, YOLO modeli
PRODUCED_DATA_DIR = DATA_DIR / "uretilen_veri"      # kurulumun ürettikleri (sıfırlamada yalnız bunlar silinir)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SETUP_DIR = PRODUCED_DATA_DIR / "kurulum_dosyalari"          # her kurulumun kendi klasörü: <ad>/
BACKUP_DIR = SETUP_DIR / "yedek"                            # üzerine yazılan kurulum dosyalarının yedekleri
VIDEOS_DIR = ENV_DATA_DIR / "videolar"                      # her kurulumun videoları: <ad>/
TEST_FRAMES_DIR = ENV_DATA_DIR / "test_kareleri"
PHOTOS_DIR = READY_DATA_DIR / "kalibrasyon_fotograflari"     # her kurulumun (kameranın) fotoğrafları: <ad>/
MODEL_DIR = READY_DATA_DIR / "yolo_modeli"                  # YOLO modelleri (.pt); arayüzde seçilir
MODEL_PATH = MODEL_DIR / "yolo11n.pt"                       # varsayılan model (en küçüğü, en hızlısı)
DEFAULT_SETUP = "kolay_video"                               # modüllerin argümansız çalıştığı kurulum
UI_OUTPUT_DIR = OUTPUTS_DIR / "arayuz"            # arayüzün kendi sonuçları (ana hat)
INTEGRATION_DIR = BACKEND_DIR / "entegrasyon"

MAX_FRAMES_LIMIT = 100_000                        # "en çok kaç kare" ayarlarının üst sınırı
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Kurulum klasöründeki dosyalar: arayüzdeki adı ve onları üreten gereksinim (kilit mesajlarında kullanılır).
SETUP_FILE_NAMES = {"kalibrasyon": "kalibrasyon.npz", "perspektif": "perspektif.npz", "olcek": "olcek.npz"}
SETUP_PARTS = {
    "kalibrasyon": ("Kamera kalibrasyonu", "1.1"),
    "perspektif": ("Perspektif dosyası", "1.2"),
    "olcek": ("Piksel–metre ölçeği", "1.3"),
}
INTEGRATION_OUTPUT_DIR = OUTPUTS_DIR / "entegrasyon"     # ana hattın komut satırı çıktıları (main.py)
SETUP_STEPS = ("Kalibrasyon", "Düzeltme", "Perspektif", "Kuş bakışı", "Ölçek")   # kurulumun 5 adımı

TEK_ISLEM_PAGE = "sayfalar/tek_islem.py"
UPLOAD_DIR = Path("/tmp") / "ads_yuklenenler"      # yüklenen dosyalar (proje klasörüne yazılmaz)

P1 = BACKEND_DIR / "Perspektif bozulması ve mesafe yanılsaması"
S2 = BACKEND_DIR / "Şeritten sapma ve şerit merkezinin belirlenmesi"
V3 = BACKEND_DIR / "Virajin_sertliginin_hesaplanmasi"
D4 = BACKEND_DIR / "Degisen_isik_golge_ve_zemin_kosullari"
C5 = BACKEND_DIR / "Carpisma_riski_ve_engel_farkindaligi"
SCRIPTS = {
    "1.1-kal": P1 / "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme" / "calibrate_camera.py",
    "1.1-duz": P1 / "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme" / "undistort.py",
    "1.2-persp": P1 / "1.2 Perspektif dönüşümü" / "compute_perspective.py",
    "1.2-warp": P1 / "1.2 Perspektif dönüşümü" / "warp_perspective.py",
    "1.3": P1 / "1.3 Piksel–metre ölçeklemesi" / "piksel_metre_olcekleme.py",
    "2.1": S2 / "2.1 HLS∕HSV∕LAB ve eşikleme" / "renk_esikleme.py",
    "2.2": S2 / "2.2 Sobel veya Canny kenar algılama" / "kenar_algilama.py",
    "2.3": S2 / "2.3 Sliding Windows" / "sliding_windows.py",
    "2.4": S2 / "2.4 Polinom uydurma" / "polinom_uydurma.py",
    "2.5": S2 / "2.5 Şerit merkezi ve yanal sapma" / "lane_offset.py",
    "3.1": V3 / "3.1 Polinom katsayıları" / "polinom_katsayilari.py",
    "3.2": V3 / "3.2 Piksel–metre ölçeklemesi" / "metre_polinom.py",
    "3.3": V3 / "3.3 Eğrilik yarıçapı" / "egrilik_yaricapi.py",
    "4.1": D4 / "4.1 Gaussian Blur ve Bilateral Filter" / "yumusatma.py",
    "4.2": D4 / "4.2 HLS∕HSV∕LAB renk kanalları" / "renk_kanallari.py",
    "4.3": D4 / "4.3 Sobel gradyanı ve Canny" / "gradyan_golge.py",
    "4.4": D4 / "4.4 Renk ve gradyan maskelerinin birleştirilmesi" / "maske_birlestirme.py",
    "5.1": C5 / "5.1 YOLO ile nesne tespiti" / "yolo_tespit.py",
    "5.2": C5 / "5.2 Bounding Box koordinatları" / "kutu_koordinatlari.py",
    "5.3": C5 / "5.3 Confidence Score ve sınıf bilgisi" / "guven_sinif_filtresi.py",
    "5.4": C5 / "5.4 Non-Maximum Suppression (NMS)" / "nms_iou_analizi.py",
    "5.5": C5 / "5.5 Engel konumu ile şerit bölgesinin ilişkilendirilmesi" / "serit_engel_iliskisi.py",
}
GROUPS = ("Kurulum adımları", "Şerit kolu (test kareleri)", "Engel kolu (kolay video)")
