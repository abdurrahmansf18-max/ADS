"""
Arayüz — sıfırlama: bir bölümün (ya da bütün projenin) ürettiklerini kalıcı olarak siler.

Her sayfanın sağ üst köşesinde o bölümün, yan menüde bütün projenin sıfırlama düğmesi vardır. Silmeden önce
bir pencerede neyin silineceği tek tek listelenir ve onay istenir; silinenler geri alınamaz. Gerçek veriler
(videolar, satranç fotoğrafları, test kareleri, yüklenen dosyalar, YOLO modeli) ve kod hiçbir zaman silinmez.
Diğer bölümler kendiliğinden uyarlanır: kilitler diskten okunduğu için silinen dosyaya ihtiyaç duyan işlemler
(ör. kurulumu silinen videonun ana hattı) kilitlenir.

Bölümler çakışmaz: outputs/ altındaki her dosya ya bir kurulumun (Kurulum'un Sıfırla'sı), ya ana hattın (Ana hat'ın
Sıfırla'sı) ya da Tek işlem'indir (Tek işlem'in Sıfırla'sı); her şeyi sıfırla hepsini siler. Boşalan klasörler de
silinir. Kurulum'un Sıfırla'sı kurulumun klasörünü, yedeklerini ve 5 adımın ara çıktılarını siler.

Kurulum sayfasındaki "Sil" ise bir kurulumu tamamen kaldırır: Sıfırla'nın sildikleri, videolarının ana hat
sonuçları ve toplanan veriler (fotoğraflar, kareler, videolar). Toplanan veriler yalnız bu yolla ve yalnız o
kurulumun klasörlerinden silinebilir; onay için kurulumun adı yazılır.
"""

import re
from pathlib import Path

import streamlit as st

import katalog
from katalog import SELECTED_OPERATION
from bilesenler.sonuc import relative

NOTICE = "ads_bildirim"          # sıfırlamadan sonra gösterilecek mesaj (app.py gösterir)
EFFECTS = {
    "ana_hat": "Kurulum ve Tek işlem etkilenmez; ana hat yeniden çalıştırılınca çıktılar yeniden üretilir.",
    "tek_islem": "Kurulum ve ana hat etkilenmez. Tek işlem'de çıktısı silinen işlemler ▶ ya da 🔒 olur; bir "
                 "işlemin girdisi silindiyse önce onu üreten işleme yönlendirilir.",
    "hepsi": "Proje kurulumdan başlar: kurulum yapılana kadar ana hat ve kurulum dosyasına ihtiyaç duyan bütün "
             "işlemler kilitlenir.",
}
KEEP = ("Silinmeyenler: videolar, satranç fotoğrafları, test kareleri, arayüzden yüklediğin dosyalar, "
        "YOLO modeli ve kod.")


def running_job() -> str | None:
    """Çalışan bir iş varsa adı (sıfırlama o sırada kapalıdır: çalışan programın dosyaları silinmesin)."""
    for key in ("kurulum_isler", "tek_islem_isler"):
        for job in st.session_state.get(key, {}).values():
            if job.running:
                return job.title or "bir işlem"
    run = st.session_state.get("ana_hat_calismasi")
    return "ana hat" if run is not None and run.running else None


def reset_button(section: str, name: str | None = None, label: str = "⟲ Sıfırla", help: str | None = None) -> None:
    """Sıfırlama düğmesi; basılınca onay penceresi açılır."""
    busy = running_job()
    if st.button(label, key=f"ads_sifirla_{section}", width="stretch", disabled=busy is not None,
                 help=f"Önce çalışan işi durdur: {busy}" if busy else help):
        _dialog(section, name)


def delete_button(name: str) -> None:
    """Kurulumu silme düğmesi (Kurulum sayfası); basılınca onay penceresi açılır."""
    busy = running_job()
    if st.button("🗑 Sil", key="ads_kurulum_sil", width="stretch", disabled=busy is not None,
                 help=f"Önce çalışan işi durdur: {busy}" if busy else
                 f"`{name}` kurulumunu tamamen siler: kurulum dosyaları, sonuçlar ve toplanan veriler."):
        _delete_dialog(name)


# --------------------------------------------------------------------------- onay penceresi
def _files(path: Path) -> list[Path]:
    return [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]


def _size(files: list[Path]) -> str:
    total = sum(p.stat().st_size for p in files)
    return f"{total / 1e6:.1f} MB" if total >= 1e5 else f"{total / 1e3:.0f} KB"


def _lines(paths: list[Path]) -> list[str]:
    """Silinecekler, okunabilir: tek dosyalar adıyla, outputs/ altındakiler klasör klasör (dosya sayısıyla)."""
    folders: dict[Path, list[Path]] = {}
    lines = []
    for path in paths:
        if path.is_file() and not path.is_relative_to(katalog.OUTPUTS_DIR):
            lines.append(f"`{relative(path)}`")
            continue
        top = path
        if path.is_relative_to(katalog.OUTPUTS_DIR) and path != katalog.OUTPUTS_DIR:
            top = katalog.OUTPUTS_DIR / path.relative_to(katalog.OUTPUTS_DIR).parts[0]
        folders.setdefault(top, []).extend(_files(path))
    for top, files in folders.items():
        lines.append(f"`{relative(top)}/` — {len(files)} dosya, {_size(files)}")
    return lines


def _effect(section: str, name: str | None) -> str:
    if section != "kurulum":
        return EFFECTS[section]
    return (f"`{name}` kurulumuyla eşleşen videolarda ana hat kilitlenir; Tek işlem'de bu dosyalara ihtiyaç duyan "
            "işlemler kilitlenir. Kurulum sayfasında adımlar baştan yapılır. Diğer kurulumlar etkilenmez.")


@st.dialog("Sıfırla", width="medium")
def _dialog(section: str, name: str | None) -> None:
    title = f"{katalog.RESET_SECTIONS[section]}" + (f" — `{name}`" if name else "")
    st.markdown(f"**{title}** sıfırlanacak. Silinenler **kalıcıdır, geri alınamaz.**")
    groups = katalog.reset_targets(section, name)
    if groups:
        st.markdown("**Silinecekler**")
        st.markdown("\n".join(f"- {label}: " + ", ".join(_lines(paths)) for label, paths in groups.items()))
    else:
        st.caption("Diskte silinecek bir şey yok; yalnız bu bölümün oturum bilgileri (çalıştırma geçmişi, ayarlar) "
                   "sıfırlanır.")
    st.markdown("**Etkisi:** " + _effect(section, name))
    st.caption(KEEP)
    sure = st.checkbox("Anladım, kalıcı olarak sil", key="ads_sifirla_onay")
    if st.button("⟲ Sil ve sıfırla", type="primary", disabled=not sure, key="ads_sifirla_sil"):
        count = katalog.delete_targets([p for paths in groups.values() for p in paths])
        clear_session(section, name)
        st.session_state[NOTICE] = f"{katalog.RESET_SECTIONS[section]}" + (f" ({name})" if name else "") + \
            f" sıfırlandı: {count} dosya silindi."
        st.rerun()


@st.dialog("Kurulumu sil", width="medium")
def _delete_dialog(name: str) -> None:
    st.markdown(f"**`{name}`** kurulumu tamamen silinecek ve listeden kalkacak. Silinenler **kalıcıdır, geri "
                "alınamaz.**")
    groups = katalog.delete_setup_targets(name)
    if groups:
        st.markdown("**Silinecekler**")
        st.markdown("\n".join(f"- {label}: " + ", ".join(_lines(paths)) for label, paths in groups.items()))
    if any(label.startswith("Toplanan veriler") for label in groups):
        st.warning("! Topladığın veriler de silinir (satranç fotoğrafları, test kareleri, videolar). Başka bir yerde "
                   "kopyası yoksa geri gelmez.")
    st.markdown("**Etkisi:** Bu kurulumun videoları ana hatta işlenemez; diğer kurulumlar etkilenmez.")
    st.caption("Yalnız sonuçları silip kurulumu yeniden yapmak istiyorsan bunun yerine **⟲ Sıfırla**'yı kullan; "
               "o, toplanan verilere dokunmaz.")
    typed = st.text_input(f"Onaylamak için kurulumun adını yaz: {name}", key="ads_sil_onay")
    if st.button("🗑 Kalıcı olarak sil", type="primary", disabled=typed.strip() != name, key="ads_sil_kesin"):
        count = katalog.delete_targets([p for paths in groups.values() for p in paths], setup=name)
        clear_session("kurulum", name, deleted=True)
        st.session_state.pop("ku_ad", None)
        st.session_state[NOTICE] = f"Kurulum silindi ({name}): {count} dosya."
        st.rerun()


# --------------------------------------------------------------------------- oturum
def clear_session(section: str, name: str | None = None, deleted: bool = False) -> None:
    """Bölümün oturum bilgilerini (çalıştırma geçmişi, seçimler, ayarlar) siler; sayfa varsayılanlarla açılır.
    deleted: kurulum tamamen silindi (Sil); o kurulumun ana hat sonucu da ekrandan kalkar."""
    state = st.session_state
    if section == "hepsi":
        for key in list(state.keys()):
            del state[key]
        return
    if section == "ana_hat":
        keys = [k for k in state if k == "ana_hat_calismasi" or k.startswith("ah_")]
    elif section == "tek_islem":
        keys = [k for k in state if k in ("tek_islem_isler", SELECTED_OPERATION) or k.startswith("ti_")]
    else:
        own = re.compile(rf"^ku_{re.escape(name)}_\d")
        keys = [k for k in state if own.match(k) or k == f"kurulum_adim:{name}"]
        jobs = state.get("kurulum_isler", {})
        for job_key in list(jobs):
            if job_key.startswith(f"{name}:"):
                del jobs[job_key]
        state.get("kurulum_yeniden", {}).pop(name, None)
        if name == katalog.DEFAULT_SETUP:       # Tek işlem'in kurulum işlemleri aynı dosyalarla çalışır
            setup_ops = [key for key, op in katalog.OPERATIONS.items() if op.group == katalog.GROUPS[0]]
            for op_key in setup_ops:
                state.get("tek_islem_isler", {}).pop(op_key, None)
            keys += [k for k in state for op_key in setup_ops if k in (f"ti_{op_key}_onay", f"ti_{op_key}_yedek")]
        run = state.get("ana_hat_calismasi")
        if deleted and run is not None and not run.running and run.settings.setup == name:
            keys.append("ana_hat_calismasi")    # sonuç dosyaları silindi
    for key in set(keys):
        del state[key]
