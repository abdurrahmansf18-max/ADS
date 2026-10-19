"""
Arayüz — Tek işlem sayfası: kurulumdaki ya da ana hattaki herhangi bir gereksinimi tek başına çalıştırır.

Solda gruplara ayrılmış işlem listesi (✓ çıktısı var · ▶ çalıştırılabilir · 🔒 kilitli), sağda seçilen işlemin
kartı: Girdi -> Ayarlar -> Çalıştır -> İlerleme -> Sonuç ya da Hata. Gerçek veri girdileri seçilir (varsayılan,
yükleme ya da yol); başka işlemlerin çıktıları ve hazır dosyalar tek satırda birer düğmedir (✓ / ✗, açınca yolu).
Ayarlarda yalnız işlemin kendi ayarları vardır; cihaz gibi kendiliğinden seçilenler ve sonucu bozabilecek ölçüler
modülün varsayılanında kalır. İşlem, komut satırında çalıştırılmış gibi arka planda çalışır; ihtiyaç duyduğu dosya
yoksa ya da bir ayar modülün kabul etmeyeceği bir değerdeyse kilitlidir. Açıklamalar başlıkların yanındaki soru
işaretindedir; çalışacak komut "Komut" düğmesindedir.
"""

import shutil
from pathlib import Path

import streamlit as st

import katalog
from servisler import calistirici
from bilesenler import ayarlar, kart, sifirla, sonuc, terminal
from bilesenler.kilit import SELECTED_OPERATION, Lock, show_locks

JOBS_KEY = "tek_islem_isler"
LIST_ICONS = {"done": "✓", "ready": "▶", "locked": "🔒", "running": "●"}


# --------------------------------------------------------------------------- girdiler
def _resolve(text: str) -> Path:
    path = Path(text.strip())
    return path if path.is_absolute() else katalog.PROJECT_ROOT / path


def _save_uploads(op_key: str, index: int, uploads) -> Path:
    """Yüklenen dosyaları geçici klasöre yazar: tek dosya -> o dosya, birden çok dosya -> klasör."""
    target = katalog.UPLOAD_DIR / op_key / str(index)
    if target.exists():
        shutil.rmtree(target)                 # bu girdinin önceki yüklemesi (yalnız geçici klasör)
    target.mkdir(parents=True)
    for upload in uploads:
        (target / Path(upload.name).name).write_bytes(upload.getbuffer())
    return target / Path(uploads[0].name).name if len(uploads) == 1 else target


def _source(item: katalog.Input) -> str:
    if item.producer:
        return f"{katalog.OPERATIONS[item.producer].label} işleminin çıktısı"
    return "gerçek veri" if item.real_data else "hazır dosya"


def _real_data_input(op: katalog.Operation, index: int, item: katalog.Input, default: Path, busy: bool) -> Path:
    """Gerçek veri: varsayılan, yükleme ya da yol."""
    mode = st.segmented_control(item.label, ["Varsayılan", "Dosya yükle", "Yol yaz"], default="Varsayılan",
                                required=True, key=f"ti_{op.key}_girdi{index}_mod", disabled=busy)
    if mode == "Dosya yükle":
        uploads = st.file_uploader(f"{item.label} yükle", accept_multiple_files=True, disabled=busy,
                                   key=f"ti_{op.key}_girdi{index}_yukle", label_visibility="collapsed")
        return _save_uploads(op.key, index, uploads) if uploads else katalog.UPLOAD_DIR / op.key / str(index)
    if mode == "Yol yaz":
        text = st.text_input(f"{item.label} yolu", value=sonuc.relative(default), disabled=busy,
                             key=f"ti_{op.key}_girdi{index}_yol", label_visibility="collapsed",
                             help="Proje köküne göre (ör. data/ortam_verisi/test_kareleri/kolay_video) ya da "
                                  "/workspace/… ile.")
        return _resolve(text) if text.strip() else default
    return default


def _linked_input(op: katalog.Operation, index: int, item: katalog.Input, default: Path, busy: bool) -> Path:
    """Başka işlemin çıktısı ya da hazır dosya: tek düğme (✓ / ✗ ve adı); açınca kaynağı ve yolu, istenirse
    başka bir dosya yazılır."""
    key = f"ti_{op.key}_girdi{index}_yol"
    text = st.session_state.get(key, sonuc.relative(default))
    path = _resolve(text) if text.strip() else default
    ready = katalog.path_ready(path, item.pattern)
    with st.popover(f"{'✓' if ready else '✗'} {item.label}", key=f"ti_{op.key}_girdi{index}_kutu", disabled=busy,
                    help=f"{_source(item)}; açınca yolu görünür ve değiştirilebilir."):
        st.markdown(f"**{item.label}** — {_source(item)}")
        st.text_input("Yol", value=sonuc.relative(default), key=key,
                      help="Varsayılan, bu dosyayı üreten işlemin varsayılan çıktısıdır; başka bir dosya için yolu yaz.")
        st.markdown(f"{'✓ var' if ready else '✗ yok'}: `{sonuc.relative(path)}`")
    return path


def input_widgets(op: katalog.Operation, busy: bool, notes: kart.Notes) -> list[Path]:
    """Girdiler: gerçek veri seçimle (varsayılan / yükle / yol), model seçim kutusuyla; başka işlemlerin çıktıları
    ve hazır dosyalar tek satırda, birer düğmeyle."""
    paths: dict[int, Path] = {}
    linked = []
    for index, item in enumerate(op.inputs):
        default = katalog.input_default(op, item)
        if item.options is not None:                 # ör. YOLO modeli: klasördekilerden seçilir
            choices = item.options()
            paths[index] = st.selectbox(item.label, choices, index=choices.index(default) if default in choices else 0,
                                        format_func=katalog.model_label, key=f"ti_{op.key}_girdi{index}_secim",
                                        disabled=busy) if choices else default
            notes.add(f"**{item.label}:** {'✓' if choices else '✗'} `{sonuc.relative(paths[index])}` — hazır dosya; "
                      f"`{sonuc.relative(katalog.MODEL_DIR)}/` klasöründeki modellerden seçilir "
                      f"({len(choices)} model).")
        elif item.real_data:
            paths[index] = _real_data_input(op, index, item, default, busy)
        else:
            linked.append((index, item, default))
    if linked:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            for index, item, default in linked:
                paths[index] = _linked_input(op, index, item, default, busy)
    for index, item in enumerate(op.inputs):
        if item.options is None:
            ready = katalog.path_ready(paths[index], item.pattern)
            notes.add(f"**{item.label}:** {'✓' if ready else '✗'} `{sonuc.relative(paths[index])}` — {_source(item)}")
    return [paths[index] for index in range(len(op.inputs))]


# --------------------------------------------------------------------------- sonuç
def results(op: katalog.Operation, inputs: list[Path], values: dict, job) -> None:
    views = op.views(inputs, values)
    if job is None and not any(katalog.view_ready(v) for v in views):
        return                                    # gösterilecek bir şey yoksa Sonuç bölümü çıkmaz
    with kart.section("Sonuç") as notes:
        if job is not None:
            sonuc.show_job_outcome(job)
        else:
            notes.add("Diskteki çıktılar: önceki bir çalıştırmadan (bu sayfadan, ana hattan ya da komut satırından).")
        sonuc.show_views(views, key=f"ti_{op.key}_gorunum", notes=notes)


# --------------------------------------------------------------------------- işlem kartı
def operation(op: katalog.Operation, jobs: dict, running_key: str | None) -> None:
    job = jobs.get(op.key)
    busy = job is not None and job.running
    state = "running" if busy else katalog.status(op)
    with kart.operation_card(op.number, op.name, op.purpose, state=state):
        with kart.section("Girdi") as notes:
            inputs = input_widgets(op, busy, notes)

        with kart.section("Ayarlar") as notes:
            image = ayarlar.image_of(inputs) if any(item.kind == "points" for item in op.settings) else None
            size = (image.shape[1], image.shape[0]) if image is not None else None
            values, errors = ayarlar.setting_widgets(op, katalog.setting_defaults(op, image_size=size),
                                                     key=f"ti_{op.key}", busy=busy, image=image)
            commands = katalog.build_commands(op, inputs, values) if not errors else []
            fixed = katalog.operation_fixed_rows(op, commands)
            ayarlar.fixed_settings(fixed)
            notes.extra = ayarlar.counts(len(op.settings), len(fixed))
            notes.add(*katalog.SETTING_NOTES.get(op.key, []))
            if any((op.key, item.dest) in katalog.SAVED_SETTINGS for item in op.settings):
                notes.add(katalog.SAVED_NOTE)
            notes.add("Değerler sınırlıdır; modülün kabul etmeyeceği bir değer girilirse Çalıştır kilitlenir ve "
                      "nedeni Çalıştır bölümünde yazar." if op.settings
                      else "Bu işlemin ayarlanabilir ayarı yok; modülün kendi değerleri kullanılır.", ayarlar.FIXED_NOTE)

        with kart.section("Çalıştır") as notes:
            input_locks = katalog.locks(op, inputs)
            problems = errors + [ayarlar.readable(op, p) for p in katalog.check_commands(commands)]
            other = running_key is not None and running_key != op.key
            setting_locks = [Lock("", "Ayar", problem) for problem in problems]
            if other:
                setting_locks.append(Lock("", "Başka işlem çalışıyor",
                                          f"Önce bitmesini bekle: {katalog.OPERATIONS[running_key].label}."))
            if input_locks:
                show_locks(input_locks, key=f"ti_{op.key}", title="Önce çalıştır")
            if setting_locks:
                show_locks(setting_locks, key=f"ti_{op.key}_ayar", title="Düzelt")
            protected = [p for p in (op.protected(inputs, values) if op.protected else []) if Path(p).is_file()]
            confirmed = True
            if protected:
                notes.add("Bu işlem kurulum dosyasının üzerine yazar: " + ", ".join(
                    f"`{sonuc.relative(p)}`" for p in protected) + ". Ana hat bu dosyayı kullanır; farklı ayarla "
                    "çalıştırırsan ana hattın sonuçları değişir. Üzerine yazmadan önce yedeği alınır.")
                confirmed = st.checkbox("! Var olan kurulum dosyasının üzerine yaz", key=f"ti_{op.key}_onay",
                                        disabled=busy)
            backup_note = st.session_state.get(f"ti_{op.key}_yedek")
            if backup_note:
                notes.add(f"Son yedek: `{backup_note}`")
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                if busy:
                    if st.button("■ Durdur", key=f"ti_{op.key}_durdur", help="Komutu durdur"):
                        job.stop()
                        st.rerun()
                else:
                    disabled = bool(input_locks or setting_locks) or not confirmed or not commands
                    if st.button("▶ Çalıştır", type="primary", key=f"ti_{op.key}_calistir", disabled=disabled):
                        if protected:
                            backup = calistirici.backup_files(protected, katalog.BACKUP_DIR)
                            st.session_state[f"ti_{op.key}_yedek"] = sonuc.relative(backup) if backup else None
                        jobs[op.key] = calistirici.sequence_job(commands, op.label).start()
                        st.rerun()
                terminal.command_button(calistirici.sequence_job(commands).command_text if commands else None,
                                        key=f"ti_{op.key}_komut")
            notes.add("Komut, komut satırında çalıştırılmış gibi arka planda çalışır; **Komut** düğmesi çalışacak "
                      "komutu gösterir.")

        if job is not None:
            with kart.section("İlerleme"):
                terminal.show_job(job)
        if not busy:
            results(op, inputs, values, job)


# =========================================================================== sayfa
corner = kart.page_title("Tek işlem", [
    "Kurulumdaki ya da ana hattaki herhangi bir gereksinimi tek başına çalıştırır ve yalnız onun sonucunu "
    "gösterir. Komut, komut satırında çalıştırılmış gibi arka planda çalışır.",
    "Bir işlemin girdisi başka bir işlemin çıktısıysa ve o çıktı yoksa işlem kilitlidir; kilit kutusundaki "
    "düğme önce çalıştırılacak işleme götürür."])
with corner:
    sifirla.reset_button("tek_islem", help="Kurulum ve ana hat dışındaki bütün çıktıları siler.")

jobs = st.session_state.setdefault(JOBS_KEY, {})
running_key = next((key for key, job in jobs.items() if job.running), None)
selected = st.session_state.get(SELECTED_OPERATION)
if selected not in katalog.OPERATIONS:
    selected = next(iter(katalog.OPERATIONS))

menu, card = st.columns([1, 2.7], gap="medium")
with menu:
    for index, group in enumerate(katalog.GROUPS):
        kart.heading(group, "✓ çıktısı var · ▶ çalıştırılabilir · 🔒 kilitli · ● çalışıyor" if index == 0 else None,
                     key=f"grup_{group}")
        for op in (o for o in katalog.OPERATIONS.values() if o.group == group):
            state = "running" if op.key == running_key else katalog.status(op)
            number = op.number if len(op.number) <= 4 else ""
            label = f"{LIST_ICONS[state]} {number} {op.name}".replace("  ", " ")
            if st.button(label, key=f"ti_sec_{op.key}", width="stretch",
                         type="primary" if op.key == selected else "secondary"):
                st.session_state[SELECTED_OPERATION] = op.key
                st.rerun()

with card:
    operation(katalog.OPERATIONS[selected], jobs, running_key)
