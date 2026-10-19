"""
Katalog — arayüzün veri ve kural katmanı (Streamlit kullanmaz): yollar, kurulumlar, işlemler, ayar değerleri, silme.

Sayfalar ve bileşenler "import katalog" ile her şeye katalog.<ad> olarak ulaşır; her ad aşağıdaki dosyalardan
birindedir (bir şeyi bulmak ya da değiştirmek için bu sıra izlenebilir):

    yollar.py          proje klasörleri, kurulum dosya adları, gereksinim modüllerinin dosyaları (SCRIPTS)
    moduller.py        gereksinim modülünü yükleme, varsayılanları ve ArgumentParser'ı okuma, komutu deneme
    kurulumlar.py      kurulum adları ve klasörleri, videolar, modeller, kurulumun 5 adımının planı (SetupPlan)
    islemler.py        işlem tanımı (Input, Setting, View, Operation, Lock) ve kuralları (kilit, durum, komut)
    degerler.py        Sonuç'taki değer kartları
    islem_listesi.py   Tek işlem'deki 24 işlem (OPERATIONS)
    ayar_degerleri.py  ayarların başlangıç değerleri ve "Sabit ayarlar" satırları
    silme.py           sıfırlama ve silme
"""

from . import ayar_degerleri, degerler, islem_listesi, islemler, kurulumlar, moduller, silme, yollar

for _module in (yollar, moduller, kurulumlar, islemler, degerler, islem_listesi, ayar_degerleri, silme):
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})
del _module
