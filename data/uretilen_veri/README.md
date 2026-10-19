# Üretilen veri — kurulumun ürettikleri

**Bu klasör toplanmaz, elle dosya konmaz.** İçindekiler arayüzdeki **Kurulum** sayfasından (ya da 1.1–1.3
modüllerinden) üretilir; ana hat ve diğer modüller onları okur.

## `kurulum_dosyalari/<kurulum_adı>/`

Her kurulumun (kamera ve takılışı) kendi klasörü; Kurulum sayfasındaki **＋ Oluştur** açar (ilk adım çalışınca da
kendiliğinden açılır).

| Dosya | Üreten | Ne tutar |
|---|---|---|
| `kalibrasyon.npz` | Kurulum 1. adım (1.1) | Lensin bükülmesi; bu kurulumun kamerasına ait |
| `perspektif.npz` | Kurulum 3. adım (1.2) | Kuş bakışı matrisi; kameranın takılışına özel |
| `olcek.npz` | Kurulum 5. adım (1.3) | 1 pikselin kaç metre olduğu |

`kurulum_dosyalari/yedek/`: arayüzün, üzerine yazılan kurulum dosyalarından aldığı yedekler
(`yedek/<tarih-saat>/<kurulum_adı>/…`). Bu yüzden `yedek` bir kurulum adı olamaz; `video`, `kus_bakisi`,
`kus_bakisi_duz_yol` ve `kus_bakisi_maskeler` de olamaz (ara çıktı klasörleri arayüzün başka klasörleriyle
çakışırdı).

- **Neden burada:** Kurulum bir kez yapılır; sonucu burada saklanır ve projeyle birlikte gönderilir. Ana hat her
  videoyu, bulunduğu klasörün kurulumunun bu üç dosyasıyla işler; dosyalar yoksa ana hat kilitlenir.
- **Nasıl üretilir:** Önce aynı adla `ortam_verisi/test_kareleri/<ad>/` ve `hazir_veri/kalibrasyon_fotograflari/<ad>/`
  doldurulur. Sonra arayüzde Kurulum sayfasında kurulum adı seçilir ve 5 adım sırayla yapılır.
- **Ne zaman yeniden:** Kamera ya da kameranın takılışı değişince. Bir adım yeniden yapılınca sonraki adımlar
  ↻ (güncel değil) olarak işaretlenir.
- **Silinebilir:** **⟲ Sıfırla** bir kurulumun bu klasörünü (boş da olsa), yedeklerini ve 5 adımın `outputs/`
  altındaki bütün ara çıktılarını siler; boş klasör kalmaz. Toplanan veriler kalır, kurulum yeniden yapılabilir
  (klasör 1. adımda yeniden açılır). **🗑 Sil** kurulumu tamamen kaldırır: bu klasör, ara çıktılar, videolarının ana hat sonuçları,
  yedekleri ve `ortam_verisi/`, `hazir_veri/` içindeki o kuruluma ait veriler. Diğer kurulumlar etkilenmez.
- **Varsayılan:** Modüller argümansız çalışınca `kurulum_dosyalari/kolay_video/` içindeki üç dosyayı arar.
- Kurulumun ara çıktıları (düzeltilmiş ve kuş bakışı kareler) burada değil, `outputs/` altındadır; onlar
  istenince yeniden üretilir.
