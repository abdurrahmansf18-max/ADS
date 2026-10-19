# frontend — arayüz

Projenin tarayıcıdan kullanılan arayüzü (Streamlit). Konteynerle birlikte kendiliğinden başlar:
<http://127.0.0.1:8501>. Arayüz görüntü işleme yapmaz; `../backend/` içindeki gereksinim modüllerini komut
satırındaki gibi çalıştırır ve sonuçlarını gösterir.

## Katmanlar

```
frontend/
├── app.py              giriş: sayfa menüsü, yan menüdeki durum paneli, "Her şeyi sıfırla"
├── .streamlit/         config.toml: ana renkler ve sunucu ayarları
├── sayfalar/           NE gösterilir — üç sayfa
│   ├── kurulum.py        ilk kurulumun 5 adımı (kalibrasyon → düzeltme → perspektif → kuş bakışı → ölçek)
│   ├── ana_hat.py        bir videoyu tek düğmeyle baştan sona işler
│   └── tek_islem.py      24 gereksinimden birini tek başına çalıştırır
├── bilesenler/         NASIL gösterilir — üç sayfada ortak görünüm parçaları
│   ├── tema.py           renkler, durum simgeleri, CSS
│   ├── kart.py           işlem kartı, numaralı bölümler, "?" açıklama düğmesi, başlıklar
│   ├── ayarlar.py        Ayarlar bölümü: ayarlanabilir ayarlar ve "Sabit ayarlar" kutusu
│   ├── sonuc.py          Sonuç bölümü: değer kartları, görüntüler, tablolar, indirme
│   ├── terminal.py       İlerleme bölümü: durum satırı, terminal çıktısı, "Komut" düğmesi
│   ├── kilit.py          kilit kutusu (neyin eksik olduğu ve oraya götüren düğme)
│   └── sifirla.py        "Sıfırla" ve "Sil" düğmeleri ve onay pencereleri
├── servisler/          işleri çalıştırır (Streamlit kullanmaz)
│   ├── calistirici.py    bir modülü ayrı süreçte, komut satırındaki gibi çalıştırır; yedek alır
│   └── ana_hat.py        ana hattı (backend/entegrasyon/main.py) arayüzün içinde çalıştırır
└── katalog/            veri ve kurallar (Streamlit kullanmaz); sayfalar "import katalog" ile kullanır
    ├── yollar.py         proje klasörleri, kurulum dosya adları, modül dosyaları (SCRIPTS)
    ├── moduller.py       modülü yükleme, varsayılanlarını ve ArgumentParser'ını okuma, komutu deneme
    ├── kurulumlar.py     kurulum adları ve klasörleri, videolar, modeller, 5 adımın planı (SetupPlan)
    ├── islemler.py       işlem tanımı (Input, Setting, View, Operation, Lock), kilit ve durum kuralları
    ├── islem_listesi.py  Tek işlem'deki 24 işlem (OPERATIONS)
    ├── degerler.py       Sonuç'taki değer kartları
    ├── ayar_degerleri.py ayarların başlangıç değerleri ve "Sabit ayarlar" satırları
    └── silme.py          sıfırlama ve silme
```

Bağımlılık tek yönlüdür: `sayfalar → bilesenler → servisler, katalog → backend`. `katalog/` ve `servisler/`
Streamlit'e bağlı değildir; bir kural (yol, kilit, varsayılan değer, silinecekler) yalnız orada yazılıdır.
Ayarların adı, sınırı ve varsayılanı backend modülünün kendi `AYARLAR` bölümünden ve komut satırı tanımından
okunur; arayüz modülü değiştirmez.

## Bir işlem kartı

Üç sayfadaki her işlem aynı düzendedir:

| Bölüm | İçinde |
|---|---|
| Başlık | gereksinim no, ad, "?" (ne işe yaradığı), durum rozeti |
| 1 Girdi | gerçek veri (varsayılan / yükle / yol) ya da başka işlemlerin çıktıları |
| 2 Ayarlar | başlıkta "N ayarlanabilir · M sabit"; ayarlanabilir ayarlar ve kapalı "Sabit ayarlar (M)" kutusu |
| 3 Çalıştır | kilitler, "▶ Çalıştır" / "■ Durdur", "Komut"; biten kurulum adımında "✓ Bitti" ve "↺ Yeniden yap" |
| 4 İlerleme | tek satır durum (çok komutluda çubuk), açılır "Terminal çıktısı" |
| 5 Sonuç | değer kartları, görüntüler, tablolar; teknik ayrıntı açılır kutularda; dosya yolları "?" içinde |

Kurallar:

- **Sayfada açıklama yazısı yoktur.** Her açıklama, başlığın yanındaki mor "?" düğmesindedir (`kart.note`,
  `kart.section(...)` içindeki `notes.add(...)`).
- **Renk tek başına bilgi taşımaz;** yanında simge ve yazı vardır: ✓ bitti · ● sırada / çalışıyor ·
  ▶ çalıştırılabilir · ↻ yeniden yapılmalı · 🔒 kilitli · ! uyarı · ✗ hata · ■ durduruldu (`tema.STATES`).
- Biten kurulum adımında Girdi ve Ayarlar **kilitli** görünür (kullanılan değerler).
- Sabit ayarların kaynağı: **kendiliğinden** (sistem belirler), **kurulumdan** (kurulum dosyası), **sabit**
  (modülün kendi değeri).

## Neyi değiştirmek için hangi dosya

| Değiştirilecek | Dosya |
|---|---|
| Ana renkler (düğmeler, zemin, yazı, durum renkleri) | `.streamlit/config.toml` (değişince arayüz yeniden başlatılır) |
| Özel parçaların renk ve stili (kart, bölüm başlığı, terminal, "?" düğmesi, yan menü) | `bilesenler/tema.py` (`COLORS`, `STATES`, `_CSS`) |
| Kartın ve bölümlerin yapısı, başlıklar | `bilesenler/kart.py` |
| Sonuçların gösterimi | `bilesenler/sonuc.py` |
| Ayar bileşenleri, sabit ayarlar tablosu | `bilesenler/ayarlar.py` |
| İlerleme ve terminal | `bilesenler/terminal.py` |
| Yan menüdeki durum paneli | `app.py` |
| Bir sayfanın düzeni | `sayfalar/<sayfa>.py` |
| Bir işlemin girdisi, ayarı, çıktısı; yeni işlem | `katalog/islem_listesi.py` |
| Klasör yolları | `katalog/yollar.py` |

### Tasarım için kancalar (CSS)

`tema.py`'deki CSS şu sınıflara ve anahtarlara bağlıdır; adlarını değiştirirken ikisi birlikte değişmeli:

| Kanca | Nerede |
|---|---|
| `.ads-baslik` | sayfa başlığı (`kart.page_title`) |
| `.ads-kart-baslik`, `.ads-no` | işlem kartının başlığı ve numarası |
| `[class*="st-key-ads_kart"]` | işlem kartının kutusu (`st.container(key="ads_kart_…")`) |
| `.ads-bolum`, `.ads-bolum-no`, `.ads-bolum-ek`, `[class*="st-key-ads_bolum_"]` | numaralı bölüm başlığı, numarası, yanındaki kısa not, başlık satırı |
| `.ads-step` | liste ve grup başlıkları (yan menü, adım şeridi, sabit ayar grupları) |
| `[class*="st-key-ads_not_"]` | mor "?" açıklama düğmesi |
| `.ads-terminal`, `.ads-cmd`, `.ads-error`, `.ads-warning` | terminal kutusu, komut, HATA ve UYARI satırları |
| `.ads-lock` | kilit kutusundaki neden yazısı |

## Geliştirirken

- Kod değişince sayfayı yenilemek (F5) yeterlidir: `app.py` bir dosyanın değiştiğini görünce arayüzün bütün
  modüllerini yeniden yükler. `.streamlit/config.toml` değişince arayüz yeniden başlatılır:
  `docker restart ads-cv-lab` ya da yalnız arayüz için konteynerdeki `streamlit run` sürecini durdurmak
  (`docker/servisler.sh` onu 3 sn içinde yeniden açar).
- Oturum anahtarları sayfaya göre önekli: `ku_` Kurulum, `ah_` Ana hat, `ti_` Tek işlem, `ads_` ortak.
- Arayüzün yazdıkları: `outputs/arayuz/<kurulum>/` (ana hat), yüklenen dosyalar `/tmp/ads_yuklenenler/`
  (konteynerin içinde, proje klasörüne yazılmaz).
