# ADS — şerit ve engel tespiti (V1)

Görüntü işleme dersi projesi. Arabanın önüne takılı bir kameranın videosunda şeridi bulur; aracın şerit ortasından
ne kadar kaydığını, virajın yarıçapını ve yönünü ölçer; önündeki araçları ve yayaları (YOLO) şeritle ilişkilendirip
risklerini belirler. Sonuç, bu bilgilerin üzerine çizildiği bir video ve kare kare bir tablodur.

İş üç aşamadır:

1. **Kurulum** (her kamera için bir kez): lens bozulması (kalibrasyon), kuş bakışı dönüşümü (perspektif) ve
   1 pikselin kaç metre olduğu (ölçek) ölçülür.
2. **Ana hat**: video kare kare işlenir.
3. **Tek işlem**: 24 gereksinimden biri tek başına denenir.

Hepsi tarayıcıdaki arayüzden yapılır. Her şey Docker konteynerinde çalışır; bilgisayara Python ya da kütüphane
kurmak gerekmez.

## Klasörler

```
ADS/
├── backend/     görüntü işleme: 20 gereksinim (1.1–5.5) ve hepsini bağlayan entegrasyon/main.py   → backend/README.md
├── frontend/    arayüz (Streamlit): Kurulum, Ana hat, Tek işlem sayfaları                         → frontend/README.md
├── data/        veriler, kurulum adıyla ayrılmış üç grup                                         → data/*/README.md
│   ├── ortam_verisi/    videolar ve onlardan seçilen test kareleri
│   ├── hazir_veri/      satranç tahtası fotoğrafları, YOLO modeli
│   └── uretilen_veri/   kurulumun ürettiği dosyalar (kalibrasyon, perspektif, ölçek)
├── docker/      Dockerfile, docker-compose, başlatma betikleri, Python paketleri (requirements.txt)
└── outputs/     işlemlerin çıktıları (çalıştırınca oluşur; depoda yok)
```

Depoda iki hazır kurulumun verileri var, aynı kameradan (Udacity, bkz. [Kaynaklar](#kaynaklar-ve-lisanslar)):

| Kurulum | Video | Satranç fotoğrafı | Test karesi |
|---|---|---|---|
| `kolay_video` | `kolay_video_iyi_isik.mp4` (50 sn, iyi ışık) | 15 | 3 düz yol + 6 deneme |
| `zor_video` | `zor_video_golge_zemin.mp4` (16 sn, gölge, zemin değişimi) | 15 | 3 düz yol + 6 deneme |

## Gerekenler

- **Docker Engine ve Docker Compose v2.** Linux'ta denendi.
- **Disk:** yaklaşık 15 GB boş yer. Docker imajı 10.6 GB tutuyor.
- **İsteğe bağlı NVIDIA ekran kartı:** CUDA 13'ü destekleyen sürücü (580 ve sonrası) ve
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  gerekir. Ekran kartı yoksa proje CPU ile çalışır; engel tespiti daha yavaş olur.

## A'dan Z'ye kurulum

### 1. Depoyu indir

```bash
git clone https://github.com/abdurrahmansf18-max/ADS.git
cd ADS
```

### 2. (İsteğe bağlı) Diğer YOLO modellerini indir

Depoda yalnız en küçük model `yolo11n.pt` (5.6 MB) var; arayüzün varsayılanı odur. Daha isabetli ama daha yavaş
olan s, m, l, x modellerini (toplam 225 MB) istersen indir. Arayüz bu klasördeki modelleri kendisi listeler.

```bash
for m in s m l x; do
  curl -L -o data/hazir_veri/yolo_modeli/yolo11$m.pt \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11$m.pt
done
```

### 3. İmajı kur ve konteyneri başlat

```bash
bash docker/baslat.sh --build     # ilk kez: imajı kurar ve başlatır (paketler indirileceği için uzun sürer)
bash docker/baslat.sh             # sonraki seferler
```

Betik ekran kartını kendisi bulur (yoksa CPU ile başlatır). YOLO'nun GPU'da mı CPU'da mı daha hızlı olduğunu bir
kez ölçer. Bitince adresleri yazar:

```
Arayuz  : http://127.0.0.1:8501
Jupyter : http://127.0.0.1:8890/lab?token=...
Konsol  : docker exec -it ads-cv-lab bash
```

- **Arayüz** ve **Jupyter Lab** konteynerle birlikte kendiliğinden başlar. Jupyter, değer denemeleri için bir
  çalışma ortamıdır; Jupyter'in anahtarını sonradan görmek için: `docker exec ads-cv-lab jupyter server list`
- Konteynerdeki kullanıcının kimliği (UID) 1000'dir. Senin kimliğin farklıysa (`id -u`), oluşan dosyalar sana ait
  olsun diye imajı bir kez kendi kimliğinle kur:
  ```bash
  docker compose -f docker/docker-compose.yml build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)
  bash docker/baslat.sh
  ```

### 4. Kurulumu yap (her kamera için bir kez)

Tarayıcıda <http://127.0.0.1:8501> → **Kurulum** sayfası → üstten kurulumu seç (`kolay_video` ya da `zor_video`)
→ 5 adımı sırayla **▶ Çalıştır** ile yap. Fotoğraflar ve kareler kurulumun klasörlerinden kendiliğinden alınır.
Depodaki veriler için girilecek değerler:

| Adım | Ayar | `kolay_video` | `zor_video` |
|---|---|---|---|
| 1 Kalibrasyon | İç köşe | 9 x 6 | 9 x 6 |
| 3 Perspektif hesabı | 4 kaynak nokta | varsayılan noktalar | (614, 480) (721, 480) (1030, 720) (270, 720) |
| 5 Ölçek | Şerit genişliği / kesik çizgi boyu | 3.66 m / 3.05 m | 3.66 m / 3.05 m |

3. adımda noktaları seçerken anlık önizlemede kuş bakışındaki iki şerit çizgisi dikey ve paralel olmalı.
Beş adım bitince adım şeridinin altında **✓ Kurulum hazır** yazar. Sonuç dosyaları
`data/uretilen_veri/kurulum_dosyalari/<kurulum>/` klasörüne kaydedilir.

### 5. Videoyu işle

**Ana hat** sayfası → video seç (kurulum, videonun klasöründen kendiliğinden seçilir) → **▶ Çalıştır**.
- **Çalışırken:** canlı önizleme ve anlık değerler görünür.
- **Bitince:** işlenmiş video, özet değerler, grafikler ve kare kare tablo görünür (indirilebilir).
- **Sonuçların yeri:** `outputs/arayuz/<kurulum>/`

### 6. Tek tek gereksinimler

**Tek işlem** sayfası bir gereksinimi tek başına çalıştırır ve yalnız onun sonucunu gösterir; varsayılan olarak
`kolay_video` kurulumunun verileri ve dosyalarıyla. Bir işlemin girdisi başka bir işlemin çıktısıysa ve o çıktı
yoksa işlem kilitlidir; kilit kutusundaki düğme önce çalıştırılacak işleme götürür.

### 7. (İsteğe bağlı) Komut satırından

Her gereksinim ve ana hat komut satırından da çalışır (arayüzdeki "Komut" düğmesi aynı komutu gösterir):

```bash
docker exec -it ads-cv-lab bash
cd backend/entegrasyon
python main.py --help
python main.py                    # kolay video, kolay_video kurulumuyla (önce kurulum) → outputs/entegrasyon/
```

Bütün komutlar ve akış: [backend/README.md](backend/README.md).

## Kendi verinle

1. **Kurulumu aç:** Kurulum sayfasında **＋ Yeni kurulum** → kurulum adını yaz (ör. `kamera1`) → **＋ Oluştur**. Bu,
   dört klasörü açar.
2. **Verileri koy:**
   - satranç tahtası fotoğraflarını `data/hazir_veri/kalibrasyon_fotograflari/kamera1/` klasörüne;
   - videoları `data/ortam_verisi/videolar/kamera1/` klasörüne;
   - videodan seçtiğin düz yol karelerini (`duz_yol_1.png`, …) `data/ortam_verisi/test_kareleri/kamera1/` klasörüne.
   
   Fotoğrafları ve kareleri Kurulum sayfasındaki **＋ Ekle** ile de ekleyebilirsin.
3. **Kurulumu yap ve videoyu işle:** Kurulumun 5 adımını yap, sonra videoyu Ana hat'ta işle.

Ne çeşit fotoğraf ve video gerektiği, kaç kare seçileceği: [data/ortam_verisi/README.md](data/ortam_verisi/README.md),
[data/hazir_veri/README.md](data/hazir_veri/README.md). Aynı kamerayla, aynı takılışla çekilmiş başka bir video için
yeni kurulum gerekmez; video aynı kurulumun klasörüne konur.

## Sıfırlama ve silme

| Düğme | Siler | Dokunmaz |
|---|---|---|
| Kurulum → ⟲ Sıfırla | o kurulumun dosyaları, yedekleri ve ara çıktıları | toplanan veriler, öteki kurulumlar |
| Kurulum → 🗑 Sil | yukarıdakiler ve o kurulumun ana hat sonuçları ve toplanan verileri (onay için ad yazılır) | öteki kurulumlar |
| Ana hat → ⟲ Sıfırla | ana hat sonuçları | kurulumlar, Tek işlem çıktıları |
| Tek işlem → ⟲ Sıfırla | Tek işlem çıktıları | kurulumlar, ana hat sonuçları |
| Yan menü → ⟲ Her şeyi sıfırla | bütün üretilenler | toplanan veriler, YOLO modeli, kod |

Silmeden önce pencerede neyin silineceği tek tek yazar. Silinenler geri gelmez. Kod ve toplanan veriler
(Sil hariç) hiçbir zaman silinmez.

## Durdurma, yeniden başlatma, sorunlar

```bash
docker compose -f docker/docker-compose.yml down       # konteyneri durdur ve kaldır
bash docker/baslat.sh                                  # yeniden başlat
docker restart ads-cv-lab                              # konteyneri kaldırmadan yeniden başlat
docker logs ads-cv-lab                                 # arayüz ya da Jupyter açılmazsa nedeni
```

- **Bağlantı noktası dolu:** başka bir program 8501 ya da 8890'ı kullanıyorsa
  `UI_PORT=8502 JUPYTER_PORT=8891 bash docker/baslat.sh`
- **Ekran kartını kapatmak / zorlamak:** `ADS_GPU=0 bash docker/baslat.sh` (CPU), `ADS_GPU=1 …` (GPU zorla).
- **Arayüz kodu değişince:** sayfayı yenilemek (F5) yeterli; `frontend/.streamlit/config.toml` değişince
  `docker restart ads-cv-lab`.
- **`docker/` içindeki bir dosya değişince:** imaj yeniden kurulur: `bash docker/baslat.sh --build`

## Belgeler

| Dosya | İçinde |
|---|---|
| [backend/README.md](backend/README.md) | gereksinimlerin listesi, hangisinin ne zaman çalıştığı, kurulum ve ana hat akışı, komutlar |
| [frontend/README.md](frontend/README.md) | arayüzün katmanları, bir işlem kartının yapısı, neyin nerede değiştirileceği |
| [data/ortam_verisi/README.md](data/ortam_verisi/README.md) | video ve test karesi özellikleri, kurulum adı kuralı |
| [data/hazir_veri/README.md](data/hazir_veri/README.md) | satranç tahtası fotoğrafları, YOLO modelleri |
| [data/uretilen_veri/README.md](data/uretilen_veri/README.md) | kurulum dosyaları, sıfırlama ve silme |
| [data/KAYNAKLAR.md](data/KAYNAKLAR.md) | verilerin kaynakları ve lisansları |

## Sürüm

**V1** — ilk tam sürüm:
- 20 gereksinim ve onları bağlayan ana hat;
- arayüz (Kurulum, Ana hat, Tek işlem);
- iki hazır kurulumun verileri;
- Docker ile GPU ya da CPU üzerinde çalışma.

## Kaynaklar ve lisanslar

- **Videolar, satranç tahtası fotoğrafları ve test kareleri:**
  [Udacity CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines), MIT lisansı.
  Ayrıntı ve lisans metni [data/KAYNAKLAR.md](data/KAYNAKLAR.md) dosyasında.
- **YOLO11 modeli ve `ultralytics` kütüphanesi:** [Ultralytics](https://github.com/ultralytics/ultralytics),
  AGPL-3.0 lisansı.
