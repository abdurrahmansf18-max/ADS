# backend — görüntü işleme hattı

Projenin bütün işi bu klasördeki kodlarda yapılır. Her gereksinim kendi klasöründe, tek başına çalışan bir
modüldür. `entegrasyon/` bu modülleri tek bir hatta bağlar. Arayüz (`../frontend/`) de bu hattı ve modülleri kullanır.

```
backend/
├── Perspektif bozulması ve mesafe yanılsaması/        1.1, 1.2, 1.3
├── Şeritten sapma ve şerit merkezinin belirlenmesi/   2.1 – 2.5
├── Virajin_sertliginin_hesaplanmasi/                  3.1 – 3.3
├── Degisen_isik_golge_ve_zemin_kosullari/             4.1 – 4.4
├── Carpisma_riski_ve_engel_farkindaligi/              5.1 – 5.5
└── entegrasyon/                                       main.py (bütün hat), moduller.py (yükleyici)
```

Veriler `../data/` altında üç gruptadır: `ortam_verisi/` (videolar ve onlardan seçilen kareler), `hazir_veri/`
(satranç tahtası fotoğrafları, YOLO modeli) ve `uretilen_veri/` (kurulum dosyaları). Sonuçlar `../outputs/`
altındadır. Ayrıntı: her grubun kendi `README.md` dosyası.

## Hangi gereksinim ne zaman çalışır?

### Tek seferlik (ilk kurulum)

Her kurulum (kamera ve takılışı) için bir kez çalıştırılır; sonucu kurulumun kendi klasörüne,
`data/uretilen_veri/kurulum_dosyalari/<kurulum>/` altına kaydedilir. Kamera ya da kameranın arabaya takılışı
değişmedikçe tekrar çalıştırılmaz (ayrıntı: "Projenin çalışma akışı → İlk kurulum").

| No | Dosya | Ne üretir |
|---|---|---|
| 1.1 | `calibrate_camera.py` | Kamera kalibrasyonu (`<kurulum>/kalibrasyon.npz`) |
| 1.2 | `compute_perspective.py` | Kuş bakışı matrisi (`<kurulum>/perspektif.npz`) |
| 1.3 | `piksel_metre_olcekleme.py` | Piksel–metre ölçeği (`<kurulum>/olcek.npz`) |

Kurulum sırasında ana hattın iki modülü (`undistort.py`, `warp_perspective.py`) de ara dosya üretmek için
kullanılır.

**5.4** `nms_iou_analizi.py` da bir kez çalıştırılır, ama kurulumun parçası değildir. YOLO'nun IoU eşiğini
seçen bir analizdir; seçilen değer (0.7) ana hatta yazılıdır. Yalnız YOLO modeli değişirse yeniden
çalıştırılır.

### Her çalıştırmada (ana hat, her karede)

`entegrasyon/main.py` çalıştırıldığında her kare için çağrılır.

| No | Dosya | Görevi |
|---|---|---|
| 1.1 | `undistort.py` | Kareyi düzeltme |
| 4.1 | `yumusatma.py` | Yumuşatma |
| 4.2 | `renk_kanallari.py` | Renk maskesi |
| 4.3 | `gradyan_golge.py` | Kenar maskesi (gölge sınırları ayıklanmış) |
| 4.4 | `maske_birlestirme.py` | Maskeleri birleştirme |
| 1.2 | `warp_perspective.py` | Kuş bakışına çevirme ve geri çevirme |
| 2.3 | `sliding_windows.py` | Sol ve sağ şerit piksellerini bulma |
| 2.4 | `polinom_uydurma.py` | Şeride eğri uydurma |
| 2.5 | `lane_offset.py` | Yanal sapma |
| 3.3 | `egrilik_yaricapi.py` | Eğrilik yarıçapı ve viraj yönü |
| 3.1 | `polinom_katsayilari.py` | Viraj sınıfı (düz / hafif / sert) |
| 5.1 | `yolo_tespit.py` | Nesne tespiti |
| 5.3 | `guven_sinif_filtresi.py` | Güven ve sınıf filtresi |
| 5.2 | `kutu_koordinatlari.py` | Kutu koordinatları ve alt-orta nokta |
| 5.5 | `serit_engel_iliskisi.py` | Engel şeritte mi, ne kadar uzakta, risk |

### Ana hatta çağrılmayanlar

Tek başına çalıştırılıp sonucu gösterilir.

| No | Dosya | Neden ana hatta yok |
|---|---|---|
| 2.1 | `renk_esikleme.py` | İlk renk maskesi; ana hatta yerini geliştirilmiş hâli 4.2 aldı |
| 2.2 | `kenar_algilama.py` | Aynı kod 4.3'ün içinde; ana hat 4.3'ü kullanır |
| 3.2 | `metre_polinom.py` | Metreye çevirmeyi 3.3 kendi içinde yapar |

## Projenin çalışma akışı

Proje iki aşamada çalışır. Önce **ilk kurulum** bir kez, elle yapılır. Sonra **ana hat** her çalıştırmada
kendiliğinden işler. Ana hat kurulumun ürettiği üç dosyayı okur; kurulum yapılmadıysa çalışmaz, eksik
dosyayı ve çalıştırılacak kurulum modülünü söyleyip durur.

| | İlk kurulum | Ana hat |
|---|---|---|
| Ne zaman | Bir kez (aşağıdaki durumlarda yeniden) | Her çalıştırmada |
| Nasıl | Adımlar tek tek, sırayla, elle çalıştırılır | Tek komut; adımlar her kare için kendiliğinden sırayla çalışır |
| İnsan girdisi | Gerekir: satranç fotoğrafları, 4 yol noktası, yolun gerçek ölçüleri | Gerekmez; yalnız video seçilir |
| Sonuç | `data/uretilen_veri/kurulum_dosyalari/<kurulum>/` içinde 3 dosya | `outputs/entegrasyon/` içinde işlenmiş video ve tablo |

Komutlar konteynerin içinde çalıştırılır:

```
bash docker/baslat.sh --build    # proje kökünde (ADS/): ilk kez imajı kurar ve konteyneri başlatır
bash docker/baslat.sh            # sonraki seferlerde yalnız başlatır
docker exec -it ads-cv-lab bash  # konteynerin içine gir
```

Konteyner açılınca **arayüz** (http://127.0.0.1:8501) ve **Jupyter Lab** (http://127.0.0.1:8890) kendiliğinden
başlar; `baslat.sh` ikisi hazır olunca adreslerini (Jupyter'in token'ıyla) yazar. Arayüz kapanırsa birkaç saniye
içinde kendiliğinden yeniden açılır. Gereksinim dosyalarının (backend) ayrıca başlatılması gerekmez: arayüzden
**Çalıştır** ile ya da komut satırından çağrılınca çalışırlar.

Arayüzün üç sayfası bu iki aşamayı karşılar: **Kurulum** (ilk kurulumun 5 adımı), **Ana hat** (videoyu tek düğmeyle
işler) ve **Tek işlem** (herhangi bir gereksinimi tek başına çalıştırır). Arayüz modülleri komut satırındaki gibi
çalıştırır; aşağıdaki komutların sonuçlarıyla aynıdır.

`baslat.sh`, bilgisayarda NVIDIA ekran kartı ve Docker'ın NVIDIA eklentisi varsa konteyneri GPU ile, yoksa
CPU ile başlatır. Sonra YOLO'yu (5.1, 5.4, ana hat) GPU'da ve CPU'da kısa bir ölçümle dener ve hızlı olanı seçer
(`--device auto`); zayıf ekran kartı ile güçlü işlemcisi olan bir bilgisayarda CPU seçilebilir. Ölçüm bir kez
yapılır, konteyner yeniden kurulana kadar saklanır. GPU'suz bilgisayarda hiçbir ayar değiştirmeden çalışır,
yalnız engel tespiti daha yavaştır. `bash` olmayan
bir bilgisayarda (ör. Windows) elle: GPU'suz `docker compose -f docker/docker-compose.yml up -d`; GPU ile
sonuna `-f docker/docker-compose.gpu.yml` eklenir.

### 1. İlk kurulum

**Ne zaman gerekir?**

| Durum | Yapılacak adımlar |
|---|---|
| Kurulumun klasöründe (`kurulum_dosyalari/<kurulum>/`) dosyalar yok | Hepsi (1.1 → 1.2 → 1.3) |
| Yeni bir kamera ya da lens | Hepsi |
| Aynı kamera, ama arabaya farklı açıyla ya da yükseklikte takılmış | 1.2 → 1.3 |
| Farklı çözünürlükte video | Hepsi |
| Başka bir ülkenin yolu (şerit genişliği, kesik çizgi boyu farklı) | Yalnız 1.3 |

Kurulum adı kamerayı temsil eder ve her şeyi bağlar: satranç fotoğrafları
`data/hazir_veri/kalibrasyon_fotograflari/<kurulum>/`, düz yol kareleri `data/ortam_verisi/test_kareleri/<kurulum>/`,
sonuçlar `data/uretilen_veri/kurulum_dosyalari/<kurulum>/`, videolar `data/ortam_verisi/videolar/<kurulum>/`. Aynı kameranın videoları için kurulum bir kez yapılır. `outputs/` klasörünü silmek kurulum
gerektirmez.

**Adımlar:**

1. **1.1** — Satranç tahtası fotoğraflarından kamera kalibre edilir → `kalibrasyon.npz` (lens modeli kendiliğinden
   seçilir). Sonuç güvenilir değilse (10'dan az fotoğraf ya da hata 1 pikselden büyük) dosya yazılmaz, HATA verilir.
2. **1.1** — Düz yol kareleri düzeltilir (ara dosya, `outputs/`).
3. **1.2** — Düzeltilmiş düz yol karesinde 4 yol noktası seçilir, kuş bakışı matrisi hesaplanır →
   `perspektif.npz`. Kontrol görüntüsünde şeritler paralel çıkmalıdır.
4. **1.2** — Düz yol kareleri kuş bakışına çevrilir (ara dosya, `outputs/`).
5. **1.3** — Kuş bakışında şerit genişliği ve kesik çizgi boyu ölçülüp gerçek ölçülere bölünür →
   `olcek.npz`.

**Komutlar** (`kolay_video` kurulumu; modüllerin varsayılan ayarları bu kurulum içindir):

```
cd "/workspace/backend/Perspektif bozulması ve mesafe yanılsaması/1.1 Kamera kalibrasyonu ve distorsiyon düzeltme"
python calibrate_camera.py                 # 1. adım
python undistort.py                        # 2. adım
cd "../1.2 Perspektif dönüşümü"
python compute_perspective.py              # 3. adım (kolay videonun 4 noktası varsayılanda yazılı)
K="/workspace/outputs/1.1 Kamera kalibrasyonu ve distorsiyon düzeltme/duzeltilmis_kareler"
D="/workspace/outputs/1.2 Perspektif dönüşümü/kus_bakisi_duz_yol"
python warp_perspective.py --input "$K/duz_yol_1_undistorted.jpg" --output "$D"   # 4. adım
python warp_perspective.py --input "$K/duz_yol_2_undistorted.jpg" --output "$D"
cd "../1.3 Piksel–metre ölçeklemesi"
python piksel_metre_olcekleme.py           # 5. adım
```

**Başka bir kurulum (kamera) için** en kolayı arayüzün Kurulum sayfasıdır: kurulum adı seçilir, klasörler ve
dosya yolları kendiliğinden ayarlanır. Komut satırında aynı adımlar şu farklarla yapılır (`K` = kurulum adı):

- **1.1:** `--images data/hazir_veri/kalibrasyon_fotograflari/K` ve
  `--output data/uretilen_veri/kurulum_dosyalari/K/kalibrasyon.npz`; 2. adımda `--calib` ile bu dosya verilir.
- **1.2:** `--image` ile o kurulumun düzeltilmiş düz yol karesi, `--src` ile o karede ölçülen 4 nokta,
  `--output data/uretilen_veri/kurulum_dosyalari/K/perspektif.npz`.
- **1.3:** `--input` ile o kurulumun kuş bakışı düz yol kareleri,
  `--out data/uretilen_veri/kurulum_dosyalari/K/olcek.npz`; başka ülkede `--lane-width-m` ve `--dash-length-m` de
  verilir.

### 2. Ana hat

Tek komutla başlar. Video kare kare okunur ve her kare için aşağıdaki 15 adım kendiliğinden, sırayla
çalışır; adımlar arasında elle bir şey yapılmaz.

**Komutlar:**

```
cd /workspace/backend/entegrasyon
python main.py                                                    # kolay video
python main.py --input /workspace/data/ortam_verisi/test_kareleri/kolay_video  # test kareleri
S=/workspace/data/uretilen_veri/kurulum_dosyalari/zor_video                   # zor videonun kurulumu
python main.py --input /workspace/data/ortam_verisi/videolar/zor_video/zor_video_golge_zemin.mp4 \
    --calibration $S/kalibrasyon.npz --perspective $S/perspektif.npz --scale $S/olcek.npz
```

Her video kendi kurulumunun klasöründeki üç dosyayla verilir; varsayılanlar `kolay_video` kurulumununkilerdir. Sonuç
`outputs/entegrasyon/<ad>_islenmis.mp4` ve kare başına tablo `<ad>_islenmis.csv` olarak kaydedilir; aynı
video yeniden işlenirse üzerine yazılır. Diğer seçenekler: `--max-frames N` (yalnız ilk N kare), `--debug`
(köşede maske ve kayan pencereler), `--no-obstacles` (engel kolunu kapatır); hepsi için `python main.py --help`.

**Her karede:**

1. **1.1** — Kare düzeltilir.
2. **4.1** — Kenar araması için kare yumuşatılır.
3. **4.2** — Beyaz ve sarı çizgiler renkten bulunur.
4. **4.3** — Çizgi kenarları bulunur, gölge sınırları atılır.
5. **4.4** — İki maske birleştirilir.
6. **1.2** — Maske kuş bakışına çevrilir.
7. **2.3** — Sol ve sağ şeridin pikselleri ayrılır.
8. **2.4** — Her şeride eğri uydurulur; son karelerin ortalaması alınır, şerit genişliği makul değilse
   (kurulumda 1.3'e girilen gerçek genişliğin ±%20'si dışındaysa) o karenin ölçümü reddedilir.
9. **2.5** — Aracın şerit ortasından ne kadar kaydığı hesaplanır.
10. **3.3 / 3.1** — Virajın yarıçapı, yönü ve sertliği hesaplanır.
11. **5.1** — Karedeki araç, yaya gibi nesneler bulunur.
12. **5.3** — Emin olunmayan ve ilgisiz tespitler elenir.
13. **5.2** — Her nesnenin yola değdiği nokta (kutunun alt ortası) bulunur.
14. **5.5** — Nesnenin bizim şeridimizde olup olmadığı, mesafesi ve riski bulunur.
15. **1.2** — Şerit alanı kameradaki kareye geri çizilir; sapma, viraj ve engeller yazılır, kare kaydedilir.

## Bu proje ne yapar?

Bu proje, arabanın önüne takılmış bir kameranın çektiği yol videosunu izleyen bir sürüş yardımcısıdır.

Önce kameranın kendisini tanır: lensin görüntüyü nasıl büktüğünü satranç tahtası fotoğraflarından öğrenir
ve her kareyi düzeltir. Sonra yolu sanki yukarıdan bakılıyormuş gibi düzleştirir. Böylece şerit çizgileri
paralel görünür ve görüntüdeki uzaklıklar gerçek metrelere çevrilebilir.

Her karede beyaz ve sarı şerit çizgilerini hem renklerinden hem de kenarlarından bulur. Gölge düştüğünde ya
da yolun rengi değiştiğinde çizgiyi kaybetmemek için bu iki yolu birlikte kullanır. Bulduğu çizgilere düzgün
bir eğri oturtur ve iki şeyi söyler: araç şeridin ortasında mı, yoksa sağa ya da sola ne kadar kaymış; öndeki
viraj ne kadar keskin ve hangi yöne dönüyor.

Aynı anda yoldaki araçları, yayaları ve diğer nesneleri bir yapay zekâ modeliyle tanır. Bunlardan hangisinin
bizim şeridimizde olduğunu ve ne kadar uzakta durduğunu bulur. Şeridimizdeki bir engel yaklaştıkça uyarı
sarıdan kırmızıya döner; yan şeritlerdeki araçlar tehlike sayılmaz.

Sonunda hepsini videonun üzerine çizer: şerit yeşile boyanır, sapma ve viraj bilgisi yazılır; şeridimizdeki
nesneler kırmızı, diğerleri yeşil kutu içine alınır. Böylece aracın şeritteki yeri, yolun gidişi ve öndeki
tehlike tek bakışta görülür.
