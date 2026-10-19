# Hazır veri — olduğu gibi kullanılan

Yolun kendisi değil; işlem için gereken, olduğu gibi kullanılan dosyalar. Elle toplanır; hiçbir sıfırlama bu
klasöre dokunmaz.

## `kalibrasyon_fotograflari/<kurulum_adı>/`

Satranç tahtası fotoğrafları (`.jpg`, `.png`), **videoyu çeken kameradan** ve videoyla **aynı çözünürlükte**.
Her kamera (kurulum) için ayrı bir klasör; klasörün adı kurulum adıdır, videonun adı da bu adla başlar
(ör. `kalibrasyon_fotograflari/kamera1/` → `kamera1_aksam.mp4`). Kurulum sayfası bu klasörü kendisi kullanır;
oradan eklenen fotoğraflar da bu klasöre kaydedilir.

- **Neden:** Kurulumun 1. adımı (1.1 kalibrasyon), lensin görüntüyü ne kadar büktüğünü bu fotoğraflardan ölçer.
  Bu düzeltme olmadan kuş bakışı ve ölçümler görüntünün kenarlarında kayar.
- **Kaç tane:** 15–20 fotoğraf. En az 10'unda tahta bulunabilmeli (kodda güvenilirlik sınırı).
- **Adlar:** Serbest; klasördeki bütün görüntüler kullanılır.
- **Aynı kamera, farklı takılış:** Ayrı bir kurulum olur (ör. `kamera1_yan`); fotoğraflar aynıysa o klasöre de
  kopyalanır.

### Tahta

- Basılı satranç deseni; düz ve sert bir yüzeye (karton, tahta) yapıştırılmış, bükülmeyen, parlamayan.
  Desenin çevresinde beyaz bir kenar kalsın.
- **İç köşe sayısı sabit değildir.** Kullandığın tahtanın iç köşe sayısı (sütun x satır) Kurulum'un 1. adımında
  girilir. Kare sayısı değildir: 10x7 kareli bir tahtanın iç köşe sayısı 9x6'dır (varsayılan).
- **Sınır:** Her kenar 3–20. **Önerilen: 5x4 ile 18x12 arası** (A4'e basılmış 9x6 tahta uygundur).

Neden bu aralık (1280x720'de, projenin kamerasıyla sentetik ölçüm; düzeltilmiş görüntüde kalan bükülme):

| İç köşe | Kalan bükülme | Sonuç |
|---|---|---|
| 3x3, 4x3 | 0.15–0.21 px | Çalışır, ama köşe az olduğu için 2–3 kat daha az doğru |
| 5x4 … 18x12 | 0.02–0.07 px | En doğru aralık |
| 20x14 | 0.16 px | Kareler küçülür (~20–35 px), köşeler karışmaya başlar |
| 25x17 | 0.73 px | Kalibrasyonların çoğu güvenilmez çıkar, kaydedilmez |

Üst sınırı belirleyen, karenin fotoğraftaki boyudur: her kare yaklaşık 25 pikselden büyük görünmeli.

### Hangi fotoğraflar çekilmeli

| Çeşit | Neden |
|---|---|
| **Görüntünün her yeri:** ortada, dört köşede ve kenarlarda (özellikle **alt köşelerde**) | Bükülme en çok kenarlarda olur; şerit çizgileri de alt köşelerdedir. Ölçüm: tahta hep ortada çekilirse kenarlarda 16–60 px bükülme kalır, her yere dağılınca 0.1 px'in altına iner. |
| **Farklı açılar:** sola, sağa, yukarı, aşağı eğik (yaklaşık 20–35°) ve düz | Eğik çekimler odak uzaklığını doğru bulur. Ölçüm: yalnız düz çekimle odak uzaklığı %1 sapar. |
| **Farklı uzaklıklar:** tahta görüntü genişliğinin yaklaşık üçte biri ile üçte ikisi arası | Yakın çekim kenarları, uzak çekim bütün görüntüyü kapsar. |
| Tahta **tamamen kadrajda**, net (bulanık değil), iyi ve düzgün ışıkta | Köşeleri tam görünmeyen ya da bulanık fotoğraflar atlanır. |
| Aynı kamera, aynı çözünürlük, aynı yakınlaştırma ve odak (videodaki ayarlar) | Kalibrasyon yalnız o kamera ayarı için geçerlidir. |

Lens modeli (normal ya da geniş açılı) kalibrasyon sırasında kendiliğinden seçilir; elle bir şey ayarlanmaz.

## `yolo_modeli/`

YOLO modelleri (`.pt`) — hazır eğitilmiş nesne tespiti modelleri (Ultralytics YOLO11; COCO veri setiyle
eğitilmiş, araba, kamyon, insan, bisiklet gibi 80 sınıf). Varsayılan: `yolo11n.pt`.

- **Neden:** Engel tespiti (5.1, 5.4 ve ana hattın engel kolu) her karedeki nesneleri bu modelle bulur. Model
  projede eğitilmez, hazır indirilir.
- **Birden çok model:** Klasöre konan her `.pt` dosyası arayüzde seçilebilir (Ana hat → Ayarlar, Tek işlem →
  5.1 / 5.4). Klasörde YOLO11'in beş boyutu var; büyüdükçe uzaktaki ve küçük araçları daha iyi bulur ama yavaşlar.

| Model | Boyut | Parametre | İsabet (COCO mAP50-95) | GPU (RTX 4060) | CPU |
|---|---|---|---|---|---|
| `yolo11n.pt` (varsayılan) | 5.6 MB | 2.6 M | 39.5 | 3.3 ms | 16 ms |
| `yolo11s.pt` | 19.3 MB | 9.5 M | 47.0 | 4.4 ms | 26 ms |
| `yolo11m.pt` | 40.7 MB | 20.1 M | 51.5 | 8.7 ms | 63 ms |
| `yolo11l.pt` | 51.4 MB | 25.4 M | 53.4 | 11.0 ms | 78 ms |
| `yolo11x.pt` | 114.6 MB | 57.0 M | 54.7 | 21.1 ms | 162 ms |

  Süreler bu projede, kare başına ölçüldü (yalnız YOLO); isabet Ultralytics'in yayımladığı değerdir. 25 kare/sn
  bir videoda bir kareye 40 ms düşer ve şerit kolu bunun ~24 ms'sini kullanır. Bu yüzden CPU'da yalnız `n`
  gerçek zamana yakın çalışır (ana hat 21.7 kare/sn); GPU'da `l`'ye kadar hepsi yetişir, `x` biraz geride kalır.
  GPU/CPU seçimi her model için ayrı ölçülür ve saklanır.
- **Kaynak:** https://github.com/ultralytics/assets/releases/tag/v8.4.0 (`yolo11n.pt` … `yolo11x.pt`).
- **Yoksa:** Yalnız engel tespiti kilitlenir; şerit işlemleri çalışmaya devam eder.
