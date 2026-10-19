# Ortam verisi — yolun kendisi

Kameranın çektiği videolar ve bu videolardan seçilen kareler. Elle toplanır; sıfırlama bu klasöre dokunmaz
(yalnız bir kurulumu **🗑 Sil** ile tamamen kaldırınca o kurulumun klasörleri silinir).

**Her şeyi bağlayan kurulum adıdır.** Kurulum adı kamerayı (ve takılışını) temsil eder. Bir kez seçilir (küçük
harf, rakam, alt çizgi; ör. `sehir_video`) ve dört yerde aynen kullanılır:

| Ne | Nerede | Örnek |
|---|---|---|
| Videolar | `videolar/<ad>/` (adlar serbest) | `videolar/sehir_video/aksam.mp4` |
| Kareler | `test_kareleri/<ad>/` | `test_kareleri/sehir_video/duz_yol_1.png` |
| Satranç fotoğrafları | `../hazir_veri/kalibrasyon_fotograflari/<ad>/` | `kalibrasyon_fotograflari/sehir_video/` |
| Kurulum dosyaları | `../uretilen_veri/kurulum_dosyalari/<ad>/` | `kurulum_dosyalari/sehir_video/perspektif.npz` |

Yeni bir kurulum, Kurulum sayfasında **＋ Yeni kurulum** → ad → **＋ Oluştur** ile açılır; bu, dört klasörü boş
olarak oluşturur (klasörleri elle açmak da aynı işi görür). Kurulum sayfası bu klasörleri kendisi kullanır; ana hat
da videonun hangi kuruluma ait olduğunu bulunduğu klasörden anlar ve o kurulumun kalibrasyonunu, perspektifini ve
ölçeğini kullanır. Aynı kamerayla, aynı takılışla çekilmiş videolar aynı klasöre konur (`videolar/sehir_video/aksam.mp4`,
`…/yagmur.mp4`); farklı kamera ya da takılış, farklı kurulum. Klasör dışındaki (ya da arayüze yüklenen) bir video
için ad kuralı geçerlidir: adı kurulum adıyla başlarsa (`sehir_video_aksam.mp4`) o kurulum seçilir.

Kurulum sayfasındaki **🗑 Sil**, bir kurulumu bu klasörlerdeki verileriyle birlikte tamamen kaldırır (onay için adı
yazılır); **⟲ Sıfırla** ise yalnız üretilenleri siler, buradaki verilere dokunmaz.

**Aynı kameradan, aynı açıyla ikinci bir video** (ör. daha zor koşullarda) için yeni kurulum gerekmez: video aynı
`videolar/<ad>/` klasörüne konur. Satranç fotoğrafları ve düz yol kareleri yeniden toplanmaz; kurulum kameranın
geometrisini ölçer, videonun zorluğu onu değiştirmez. İstersen yeni videodan deneme kareleri `test_kareleri/<ad>/`
içine eklenir; adları `duz_yol_` ile başlamamalı (yeni bir düz yol karesi, kurulumun 2–5. adımlarını ↻ yapar).
Ana hattın sonuçları her videonun kendi adıyla `outputs/arayuz/<ad>/` altına yazılır.

## `videolar/<kurulum_adı>/`

Modüller ve `main.py` argümansız çalışınca `videolar/kolay_video/kolay_video_iyi_isik.mp4` dosyasını arar.

### Videonun özellikleri

| Özellik | Gereken | Neden |
|---|---|---|
| Kamera | Arabanın önüne sabit takılı, ileri bakan, mümkünse arabanın ortasında | Kurulum bu takılışa göre ölçülür; yanal sapma kameranın arabanın ortasında olduğunu varsayar |
| Hareket | Elde tutulmamış, sallanmayan; yakınlaştırma ve görüntü sabitleme kapalı | Görüntü geometrisi video boyunca aynı kalmalı |
| Çözünürlük | Satranç fotoğraflarıyla aynı; önerilen **1280x720** | Varsayılan ayarlar bu boyuta göre; büyük çözünürlük yavaşlatır |
| Kare hızı | 25–30 kare/sn | Engel ve şerit takibi ardışık karelere dayanır |
| Süre | **15–60 sn** (projedeki videolar 16–50 sn) | GPU'da ~25 kare/sn işlenir: 1 dakikalık video yaklaşık 1 dakika sürer |
| Yol | İki yanında şerit çizgisi açıkça görünen yol (otoyol, çevre yolu); bir yanda kesik çizgi | Şerit iki çizgiden bulunur; ölçek kesik çizginin boyundan ölçülür |
| Düz bölüm | En az bir yerde birkaç saniye dümdüz giden yol | Kurulumun düz yol kareleri buradan alınır |
| Ülke | Bir videodaki yol tek bir ülkenin standardında | Kurulumda gerçek şerit genişliği ve kesik çizgi boyu girilir |

### Karşılaştırma için neler içermeli

Farklı videolar yöntemin nerede iyi, nerede zayıf olduğunu gösterir. Mümkünse süre, çözünürlük ve kare hızı aynı
olsun; yalnız koşul değişsin.

| Video türü | İçinde olması gerekenler |
|---|---|
| Kolay | İyi ışık, net çizgiler, hafif virajlar, önde birkaç araç |
| Zor | Gölgeler (köprü, ağaç), zemin rengi değişimi, soluk ya da sarı çizgi |
| Çok zor | Sert virajlar, ani ışık değişimi, yokuş |
| Engel | Aynı şeritte yaklaşan araç (risk), yan şeritlerde araçlar, şerit değiştiren araç |

Kaçınılacaklar: gece (amaç bu değilse), yoğun yağmur ya da sis, çizgisiz yol, arabanın kaputunun ya da
silecek kolunun görüntünün büyük kısmını kapladığı çekimler.

### Yapay zekâ ile üretilmiş ya da hazır bulunmuş videolar

- **Kalibrasyon:** Kurulumun 1. adımı, videoyu çeken kameranın satranç tahtası fotoğraflarını ister. Yapay zekâ
  ile üretilen videoda satranç fotoğrafları da aynı "kamera" ayarlarıyla (aynı görüş açısı, çözünürlük, lens)
  üretilmeli; hazır bulunan videoda o kameranın fotoğrafları bulunmalı. Fotoğraflar
  `../hazir_veri/kalibrasyon_fotograflari/<ad>/` klasörüne konur.
- **Tutarlılık:** Yapay zekâ videolarında yol geometrisi kareden kareye kayabilir (şerit genişliği değişir,
  çizgiler eğilir). Kurulum tek bir geometri ölçtüğü için böyle videolarda sonuçlar dalgalanır. Seçerken kamera
  açısının ve yolun video boyunca tutarlı olduğuna bak.

## `test_kareleri/<kurulum_adı>/`

Videodan seçilen tek kareler (`.png` önerilir, `.jpg` de olur). Ham kare olmalı: düzeltilmemiş, kırpılmamış,
videoyla aynı çözünürlükte.

### 1. Düz yol kareleri — zorunlu, 3–5 tane

Adları `duz_yol_1.png`, `duz_yol_2.png`, … olur (Kurulum 2. adımında yüklenirse adlar kendiliğinden verilir).

- **Ne:** Yolun dümdüz ve düz zeminde gittiği, iki şerit çizgisinin de baştan sona net göründüğü, kesik çizgilerin
  belli olduğu, önde çizgileri kapatan araç olmayan kareler. Araba şeridin ortasında olsun.
- **Nereden:** Videonun farklı anlarından (ör. 8., 10., 12. saniye); aynı anın ardışık kareleri olmasın.
- **Neden:** Perspektif noktaları bu karelerde seçilir. Ölçek (1 piksel kaç metre) bu karelerin hepsinde ölçülüp
  ortalaması alınır. 3–5 kare ölçümü dengeler; tek kare bir çizgi bozukluğuna takılabilir.

### 2. Deneme kareleri — isteğe bağlı, 5–8 tane

- **Ne:** Zorlu anlar: sola viraj, sağa viraj, gölge, zemin rengi değişimi, önde araç, yan şeritte araç.
  Adlar serbest; ne olduğu anlaşılsın (ör. `sol_viraj.png`, `golge.png`, `onde_arac.png`).
- **Neden:** Tek işlem sayfasında şerit ve engel işlemleri (2.x, 4.x, 5.x) bütün videoyu işlemeden bu karelerde
  denenir. Ana hat bu kareleri kullanmaz; o videonun tamamını işler.
- **Varsayılan:** Modüller argümansız çalışınca `test_kareleri/kolay_video/` klasörünü ve 2.4 / 3.x için
  `sol_viraj.jpg` karesini arar.

### Kare nasıl çıkarılır

Bir video oynatıcının "kare kaydet" özelliğiyle ya da `ffmpeg` ile. Bu klasördeyken, 8. saniyedeki kare:

```
mkdir -p test_kareleri/sehir_video
ffmpeg -ss 8 -i videolar/sehir_video/aksam.mp4 -frames:v 1 test_kareleri/sehir_video/duz_yol_1.png
```

Konteynerde `ffmpeg` komutu yoktur; orada aynı komut imageio-ffmpeg'in kopyasıyla çalışır:
`$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())") -ss 8 -i …`
