"""
Arayüz — giriş noktası: sayfa menüsü ve yan menüdeki durum paneli.

Üç sayfa projenin iki aşamasını ve gereksinimleri tek tek test etme ihtiyacını karşılar: Kurulum (ilk
kurulumun 5 adımı), Ana hat (videoyu tek düğmeyle işler), Tek işlem (bir gereksinimi tek başına çalıştırır).
Yan menüdeki durum paneli seçili sayfanın ihtiyaç duyduğu durumu gösterir; altındaki "Her şeyi sıfırla"
bütün kurulum dosyalarını ve çıktıları siler (her sayfanın sağ üstünde de yalnız o bölümü sıfırlayan düğme var).

Arayüzün katmanları (sayfalar, bilesenler, servisler, katalog) ve neyin nerede değiştirileceği: README.md.
Başlatma: konteynerle birlikte kendiliğinden başlar (docker/servisler.sh)  ->  http://127.0.0.1:8501
"""

import sys
from pathlib import Path

import streamlit as st

FRONTEND_DIR = Path(__file__).resolve().parent


def fresh_code() -> None:
    """frontend/ altında bir .py dosyası değiştiyse arayüzün bütün modüllerini yeniden yükletir. Streamlit yalnız
    değişen dosyayı yeniler; ona bağlı olanlar (ör. kart.py'nin kullandığı tema.py) ve "from bilesenler import …"
    ile alınanlar eski kopyada kalırdı. Böylece kod değişince sunucuyu yeniden başlatmak gerekmez."""
    stamp = max(path.stat().st_mtime for path in FRONTEND_DIR.rglob("*.py"))
    previous = getattr(sys, "_ads_kod_zamani", None)
    sys._ads_kod_zamani = stamp
    if previous is None or stamp == previous:
        return
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if name != "__main__" and file and Path(file).resolve().is_relative_to(FRONTEND_DIR):
            del sys.modules[name]


fresh_code()

import katalog  # noqa: E402  (fresh_code'dan sonra)
from bilesenler import kart, sifirla, tema  # noqa: E402

st.set_page_config(page_title="ADS — Şerit ve engel", page_icon="🛣️", layout="wide")
tema.apply_theme()

PAGES = [
    st.Page("sayfalar/kurulum.py", title="Kurulum", icon=":material/build:"),
    st.Page("sayfalar/ana_hat.py", title="Ana hat", icon=":material/play_circle:", default=True),
    st.Page("sayfalar/tek_islem.py", title="Tek işlem", icon=":material/science:"),
]
STEP_TITLES = katalog.SETUP_STEPS


def _line(text: str) -> None:
    st.markdown(text)


def _group(title: str) -> None:
    st.markdown(f"<div class='ads-step'>{title}</div>", unsafe_allow_html=True)


def _running_job(key: str):
    """Bir sayfanın çalışan işi (anahtar, iş) ya da (None, None)."""
    for job_key, job in st.session_state.get(key, {}).items():
        if job.running:
            return job_key, job
    return None, None


def _setup_line(name: str) -> str:
    """Bir kurulumun tek satırlık durumu: hazır, güncel değil ya da kaç adımı bitti."""
    plan = katalog.setup_plan(name)
    done = plan.done()
    if any(plan.outdated()):
        return f"↻ `{name}` — güncel değil"
    if all(done):
        return f"✓ `{name}` — hazır"
    return f"🔒 `{name}` — {sum(done)}/5 adım"


def status_setup() -> None:
    """Kurulum sayfası: her kurulum tek satırda (hazır / güncel değil / kaç adımı bitti) ve çalışan adım."""
    names = katalog.setup_candidates()
    _group("Kurulumlar")
    if not names:
        _line("Henüz kurulum yok")
    for name in names:
        _line(_setup_line(name))
    key, _ = _running_job("kurulum_isler")
    if key:
        name, step = key.split(":")
        _line(f"● Çalışıyor: `{name}` — adım {int(step) + 1} ({STEP_TITLES[int(step)]})")


def _device_line(model: Path) -> str:
    """Seçili YOLO modelinin cihazı: ölçülmüşse GPU / CPU, değilse ilk çalıştırmada seçileceği."""
    try:
        detector = katalog.load_module(katalog.SCRIPTS["5.1"])
        entry = next((e for e in detector.read_device_cache().values() if e.get("model") == model.name), None)
    except Exception:                      # 5.1 yüklenemezse durum paneli yine açılsın
        return f"✓ `{model.name}`"
    if entry is None:
        return f"✓ `{model.name}` — cihaz ilk çalıştırmada seçilir"
    return f"✓ `{model.name}` — {detector.device_label(entry['device'])}"


def status_pipeline() -> None:
    """Ana hat sayfası: hangi kurulumlar hazır, seçili YOLO modeli ve cihazı, son çalıştırma."""
    names = katalog.setup_candidates()
    _group("Kurulumlar")
    if not names:
        _line("🔒 Kurulum yok — **Kurulum** sayfasından başla")
    for name in names:
        missing = katalog.missing_setup(name)
        if missing:
            _line(f"🔒 `{name}` — kurulum eksik")
        elif katalog.setup_outdated(name):
            _line(f"↻ `{name}` — güncel olmayabilir")
        else:
            _line(f"✓ `{name}` — hazır")
    _group("Engel tespiti")
    models = katalog.yolo_models()
    selected = st.session_state.get("ah_model")
    if not models:
        _line("✗ YOLO modeli yok")
    elif not st.session_state.get("ah_engel", True):
        _line("Kapalı (yalnız şerit)")
    else:
        _line(_device_line(selected if selected in models else models[0]))
    run = st.session_state.get("ana_hat_calismasi")
    if run is not None:
        _group("Son çalıştırma")
        video = Path(run.settings.video).name
        if run.running:
            _line(f"● `{video}` — {run.done}/{run.total} kare")
        elif run.state == "error":
            _line(f"✗ `{video}` — hata verdi")
        else:
            word = "durduruldu" if run.state == "stopped" else "bitti"
            _line(f"✓ `{video}` — {word}, {run.done} kare")


def status_single() -> None:
    """Tek işlem sayfası: kullandığı kurulum ve her grupta kaç işlemin çıktısı var, kaçı kilitli."""
    _group("Kurulum")
    _line(_setup_line(katalog.DEFAULT_SETUP))
    _group("İşlemler")
    states = {key: katalog.status(op) for key, op in katalog.OPERATIONS.items()}
    for group in katalog.GROUPS:
        keys = [k for k, op in katalog.OPERATIONS.items() if op.group == group]
        done = sum(states[k] == "done" for k in keys)
        locked = sum(states[k] == "locked" for k in keys)
        _line(f"{group.split(' (')[0]}: ✓ {done}/{len(keys)}" + (f" · 🔒 {locked}" if locked else ""))
    key, _ = _running_job("tek_islem_isler")
    if key:
        _line(f"● Çalışıyor: **{katalog.OPERATIONS[key].label}**")


PANELS = {"Kurulum": status_setup, "Ana hat": status_pipeline, "Tek işlem": status_single}
LEGENDS = {
    "Kurulum": ["✓ hazır (ana hat bu kurulumla çalışır) · 🔒 adımları eksik · ↻ bir adım güncel değil, yeniden "
                "yapılmalı",
                "Kurulum adı kamerayı ve takılışını temsil eder; ayrıntılar Kurulum sayfasının adım şeridinde."],
    "Ana hat": ["✓ kurulum hazır · 🔒 kurulum eksik (ana hat bu kurulumun videolarında kilitli) · ↻ kurulum güncel "
                "olmayabilir",
                "Engel tespiti: Ana hat sayfasında seçili YOLO modeli ve kullanacağı cihaz (GPU ya da CPU; her model "
                "için ilk çalıştırmada kısa bir ölçümle seçilir)."],
    "Tek işlem": [f"Tek işlem varsayılan olarak `{katalog.DEFAULT_SETUP}` kurulumunun dosyalarıyla ve onun test "
                  "kareleri ile videosuyla çalışır.",
                  "İşlemler: ✓ çıktısı olan / toplam · 🔒 girdisi eksik (kilitli) işlem sayısı."],
}

navigation = st.navigation(PAGES)
notice = st.session_state.pop(sifirla.NOTICE, None)
if notice:
    st.toast(f"✓ {notice}")
with st.sidebar:
    kart.heading(f"Durum — {navigation.title}", LEGENDS.get(navigation.title), key="durum")
    PANELS.get(navigation.title, status_pipeline)()
    st.divider()
    sifirla.reset_button("hepsi", label="⟲ Her şeyi sıfırla",
                         help="Bütün kurulum dosyalarını ve çıktıları siler; proje kurulumdan başlar.")
navigation.run()
