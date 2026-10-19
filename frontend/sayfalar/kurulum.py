"""
Arayüz — Kurulum sayfası: ilk kurulumun 5 adımını sırayla, adım adım yaptırır.

En üstte kurulum adı (kamera) seçilir; her şey bu adın klasörlerindedir: satranç fotoğrafları
kalibrasyon_fotograflari/<ad>/, düz yol kareleri test_kareleri/<ad>/, sonuçlar kurulum_dosyalari/<ad>/
(kalibrasyon.npz, perspektif.npz, olcek.npz). Kalibrasyon kurulumun kendi kamerasınındır; lens modelini
kendisi seçer. Adımlar dikey bir ilerleme şeridinde görünür (✓ bitti · ● sırada · ↻ yeniden yapılmalı · 🔒 kilitli);
bir adım bitmeden sonraki kilitli kalır. Adım kartı her zaman Girdi -> Ayarlar -> Çalıştır -> İlerleme -> Sonuç
sırasındadır; biten adımın girdi ve ayarları kilitli görünür (kullanılan, kurulum dosyasında kayıtlı değerler) ve
Çalıştır bölümünde "✓ Bitti" ile "Yeniden yap" durur. Yeniden yapılan adımdan sonraki adımlar da sırayla yeniden
yapılır (bir adımın çıktısı kendinden önceki bir adımın çıktısından eskiyse güncel değildir). Yeni kurulum
"＋ Oluştur" ile dört klasörüyle açılır. Sağ üstteki "Sıfırla" seçili kurulumun ürettiklerini siler (yeniden
yapılabilir); "Sil" kurulumu toplanan verileriyle birlikte tamamen kaldırır.
Her adım komut satırındaki gibi arka planda çalışır. Var olan bir kurulum dosyasının üzerine yazılacaksa
onay istenir ve önce yedeği alınır. Açıklamalar başlıkların yanındaki soru işaretindedir.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

import katalog
from servisler import calistirici
from bilesenler import ayarlar, kart, sifirla, sonuc, terminal
from bilesenler.kilit import Lock, show_locks

PAGE = "sayfalar/kurulum.py"
JOBS_KEY = "kurulum_isler"
REDO_KEY = "kurulum_yeniden"      # {kurulum adı: "Yeniden yap" ile seçilen adımlar}
NEW = "＋ Yeni kurulum"
STEPS = [
    ("1.1", "Kalibrasyon", "Satranç tahtası fotoğraflarından lensin görüntüyü nasıl büktüğünü ölçer. "
                           "Kalibrasyon bu kurulumun kamerasınındır."),
    ("1.1", "Düzeltme", "Düz yol karelerinden lens bozulmasını kaldırır; noktalar bozulmasız karede seçilir."),
    ("1.2", "Perspektif hesabı", "Düz yol karesinde yolu çevreleyen 4 nokta seçilir, kuş bakışı matrisi "
                                 "(M, Minv) hesaplanır."),
    ("1.2", "Kuş bakışı (düz yol)", "Düz yol kareleri kuş bakışına çevrilir; ölçek bu karelerde ölçülür."),
    ("1.3", "Ölçek", "Kuş bakışında şerit genişliği ve kesik çizgi boyu ölçülüp gerçek ölçülere bölünür: "
                     "1 piksel kaç metre?"),
]
REQUIRES = {0: [], 1: [0], 2: [1], 3: [1, 2], 4: [3]}     # adım -> çıktısına ihtiyaç duyduğu adımlar
STRIP = {"done": "✓", "next": "●", "ready": "▶", "redo": "↻", "locked": "🔒", "running": "●"}


@dataclass
class Progress:
    """Adımların durumu: diskten (bitti mi, güncel mi) ve kullanıcının yeniden yapmak istedikleri."""
    done: list[bool]
    outdated: list[list[str]]       # adımın çıktısından sonra değişen girdiler (boşsa güncel)
    redo: set[int]                  # "Yeniden yap" ile seçilen adımlar
    other: str | None = None        # başka bir adım çalışıyorsa onun adı

    def changed(self, step: int) -> bool:
        """Bitti ama kendisi yeniden yapılacak: girdisi değişti ya da "Yeniden yap" seçildi."""
        return self.done[step] and (bool(self.outdated[step]) or step in self.redo)

    def needs(self, step: int) -> bool:
        """Bitti ama yeniden yapılmalı: kendisi ya da kendinden önceki bir adım yeniden yapılacak."""
        return self.done[step] and any(self.changed(k) for k in range(step + 1))

    def editing(self, step: int) -> bool:
        """Kart girdi, ayar ve Çalıştır ile açılır (bitmemiş ya da yeniden yapılacak adım)."""
        return not self.done[step] or self.needs(step)


# --------------------------------------------------------------------------- yeniden yapma
def redo_steps(name: str) -> set[int]:
    return st.session_state.setdefault(REDO_KEY, {}).setdefault(name, set())


def start_redo(name: str, step: int) -> None:
    """Biten adımı yeniden yapılacak diye işaretler; önceki çalıştırmanın sonucu ve onayı karttan kalkar."""
    st.session_state.setdefault(JOBS_KEY, {}).pop(f"{name}:{step}", None)
    for key in (f"ku_{name}_{step}_onay", f"ku_{name}_{step}_yedek"):
        st.session_state.pop(key, None)
    redo_steps(name).add(step)


def cancel_redo(name: str, step: int) -> None:
    redo_steps(name).discard(step)


def create_setup() -> None:
    """"＋ Oluştur": yazılan adın klasörlerini açar, kurulumu seçer ve ad alanını temizler (yenisi yazılabilsin)."""
    name = st.session_state.get("ku_yeni_ad", "").strip()
    katalog.create_setup(name)
    st.session_state["ku_ad"] = name
    st.session_state["ku_yeni_ad"] = ""


def finish_redo(name: str, jobs: dict) -> None:
    """Başarıyla biten adım "yeniden yapılacak" listesinden çıkar."""
    for step in range(5):
        job = jobs.get(f"{name}:{step}")
        if job is not None and not job.running and job.outcome() in ("success", "warning"):
            redo_steps(name).discard(step)


# --------------------------------------------------------------------------- yardımcılar
def _save_photos(plan: katalog.SetupPlan, uploads) -> int:
    """Yüklenen satranç fotoğrafları kurulumun klasörüne (data/hazir_veri/kalibrasyon_fotograflari/<ad>/) kendi
    adlarıyla yazılır; aynı ad ve boyuttaki dosya yeniden yazılmaz. Dönüş: yeni yazılan fotoğraf sayısı."""
    plan.photos_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for upload in uploads:
        target = plan.photos_dir / Path(upload.name).name
        if not target.is_file() or target.stat().st_size != upload.size:
            target.write_bytes(upload.getbuffer())
            count += 1
    return count


def _save_frames(plan: katalog.SetupPlan, uploads) -> int:
    """Yüklenen düz yol kareleri data/ortam_verisi/test_kareleri/<ad>/ içine duz_yol_<n> adıyla (bir kez) yazılır."""
    saved = st.session_state.setdefault("ku_kaydedilen_kareler", set())
    plan.frames_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for upload in uploads:
        digest = f"{plan.name}:{upload.name}:{upload.size}"
        if digest in saved:
            continue
        index = len(plan.straight_frames()) + 1
        while any(plan.frames_dir.glob(f"duz_yol_{index}.*")):
            index += 1
        (plan.frames_dir / f"duz_yol_{index}{Path(upload.name).suffix.lower()}").write_bytes(upload.getbuffer())
        saved.add(digest)
        count += 1
    return count


def calibration_table(lines: list[str]) -> pd.DataFrame:
    """1.1'in çıktısından kullanılan (RMS) ve atlanan (neden) fotoğraflar."""
    rows, mode = [], None
    for line in lines:
        if line.startswith("Fotoğraf başına RMS hatası"):
            mode = "used"
            continue
        if line.startswith("Atlanan fotoğraflar"):
            mode = "skipped"
            continue
        if mode and line.startswith("  ") and ":" in line:
            name, value = line.strip().split(":", 1)
            rows.append({"Fotoğraf": name, "Durum": "✓ kullanıldı" if mode == "used" else "✗ atlandı",
                         "RMS (px) / neden": value.strip()})
        elif not line.startswith("  "):
            mode = None
    return pd.DataFrame(sorted(rows, key=lambda r: r["Fotoğraf"]))


@st.cache_data(show_spinner=False, max_entries=500)
def _thumbnail(path: str, mtime: float, width: int = 240):
    """Küçük resim (RGB); dosya değişirse yeniden üretilir."""
    image = cv2.imread(path)
    if image is None:
        return None
    height = round(image.shape[0] * width / image.shape[1])
    return cv2.cvtColor(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def thumbnails(paths: list[Path], columns: int = 6) -> None:
    """Görüntülerin küçük resimleri, adlarıyla (doğru dosyalar mı diye bakmak için)."""
    for start in range(0, len(paths), columns):
        for column, path in zip(st.columns(columns), paths[start:start + columns]):
            image = _thumbnail(str(path), path.stat().st_mtime)
            if image is None:
                column.caption(f"✗ {path.name} okunamadı")
            else:
                column.image(image, caption=path.name, width="stretch")


def input_status(files: list[Path], what: str, key: str, save=None) -> None:
    """Kurulumun klasöründen kendiliğinden alınan dosyalar: kaç tane (Göster: küçük resimler) ve "＋ Ekle" (yükleme
    aynı klasöre yazılır). Yoksa "yok" yazar; adım kilitli kalır. save(yüklenenler) -> yazılan dosya sayısı;
    save yoksa (biten adım) yalnız durum görünür."""
    show = False
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        if files:
            st.markdown(f"✓ **{len(files)} {what}**", width="content")
            show = st.toggle("Göster", key=key)
        else:
            st.markdown(f"✗ **{what.capitalize()} yok**", width="content")
        if save is not None:
            with st.popover("＋ Ekle", key=f"{key}_ekle", help=f"Bu kuruluma {what} ekle (kurulumun klasörüne kaydedilir)."):
                uploads = st.file_uploader(f"Bu kuruluma {what} ekle",
                                           type=[e.lstrip(".") for e in katalog.IMAGE_EXTENSIONS],
                                           accept_multiple_files=True, key=f"{key}_yukle")
                if uploads and save(uploads):
                    st.rerun()
    if show:
        thumbnails(files)


def data_row(plan) -> None:
    """Kurulumun toplanan verileri: kaç satranç fotoğrafı, düz yol karesi ve video var (kendi klasörlerinden)."""
    folders = katalog.setup_folders(plan.name)
    counts = [("satranç fotoğrafı", len(plan.photos()), folders["fotograflar"]),
              ("düz yol karesi", len(plan.straight_frames()), folders["kareler"]),
              ("video", len(katalog.setup_videos(plan.name)), folders["videolar"])]
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown("<div class='ads-step'>Veriler</div>", unsafe_allow_html=True, width="content")
        for label, count, _ in counts:
            st.badge(f"{'✓' if count else '✗'} {count} {label}", color="green" if count else "orange")
        kart.note([f"**{label}:** `{sonuc.relative(folder)}/`" for label, _, folder in counts]
                  + ["Bu klasörlere koyduğun dosyalar kendiliğinden kullanılır; adımlarda buradan da eklenebilir."],
                  key=f"veriler_{plan.name}")


# --------------------------------------------------------------------------- ortak: çalıştır, durum notları
def run_section(plan, step: int, commands, locks: list, protected: list[Path], errors: list[str], other: str | None,
                info: tuple[str, ...] = (), op: katalog.Operation | None = None):
    """Çalıştır bölümü (kilitler, ayar kontrolü, dosya koruması, Çalıştır/Durdur, Komut) ve İlerleme.
    Dönüş: (iş, çalışıyor mu)."""
    jobs = st.session_state.setdefault(JOBS_KEY, {})
    key = f"{plan.name}:{step}"
    job = jobs.get(key)
    busy = job is not None and job.running
    widget = f"ku_{plan.name}_{step}"

    with kart.section("Çalıştır") as notes:
        notes.add(*info)
        problems = errors + ([ayarlar.readable(op, p) for p in katalog.check_commands(commands)]
                             if commands and not errors else [])
        setting_locks = [Lock("", "Ayar", problem) for problem in problems]
        if other:
            setting_locks.append(Lock("", "Başka adım çalışıyor", f"Önce bitmesini bekle: {other}."))
        if locks:
            show_locks(locks, key=widget, title="Önce")
        if setting_locks:
            show_locks(setting_locks, key=f"{widget}_ayar", title="Düzelt")
        existing = [p for p in protected if Path(p).is_file()]
        confirmed = True
        if existing:
            notes.add("Bu adım kurulum dosyasının üzerine yazar: " + ", ".join(f"`{sonuc.relative(p)}`" for p in existing)
                      + ". Ana hat bu dosyayı kullanır. Üzerine yazmadan önce yedeği alınır.")
            confirmed = st.checkbox("! Var olan kurulum dosyasının üzerine yaz", key=f"{widget}_onay", disabled=busy)
        note = st.session_state.get(f"{widget}_yedek")
        if note:
            notes.add(f"Son yedek: `{note}`")
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            if busy:
                if st.button("■ Durdur", key=f"{widget}_durdur"):
                    job.stop()
                    st.rerun()
            else:
                disabled = bool(locks) or bool(setting_locks) or not confirmed or not commands
                if st.button("▶ Çalıştır", type="primary", key=f"{widget}_calistir", disabled=disabled):
                    if existing:
                        backup = calistirici.backup_files(existing, katalog.BACKUP_DIR)
                        st.session_state[f"{widget}_yedek"] = sonuc.relative(backup) if backup else None
                    jobs[key] = calistirici.sequence_job(commands, f"Kurulum {plan.name}: adım {step + 1}").start()
                    st.rerun()
            terminal.command_button(calistirici.sequence_job(commands).command_text if commands else None,
                                    key=f"{widget}_komut")
        notes.add("Adım komut satırındaki gibi arka planda çalışır; **Komut** düğmesi çalışacak komutu gösterir.")
    if job is not None:
        with kart.section("İlerleme"):
            terminal.show_job(job)
    return job, busy


def previous_locks(plan, step: int, progress: Progress) -> list:
    """Önceki adım bitmemişse ya da yeniden yapılacaksa kilit; düğme o adıma götürür."""
    locks = []
    for k in REQUIRES[step]:
        if progress.done[k] and not progress.needs(k):
            continue
        what = "bitmeli" if not progress.done[k] else "yeniden yapılmalı"
        locks.append(Lock(f"Adım {k + 1}", f"{STEPS[k][0]} {STEPS[k][1]}",
                          f"Önce adım {k + 1} {what}: {STEPS[k][0]} {STEPS[k][1]}.", PAGE, k, f"kurulum_adim:{plan.name}"))
    return locks


def done_note(plan, step: int, progress: Progress) -> None:
    """Biten ve güncel adım: kısa özet ve "Yeniden yap" düğmesi (ne etkileneceğiyle)."""
    later = [str(k + 1) for k in range(step + 1, 5) if progress.done[k]]
    impact = []
    if later:
        impact.append(f"Yeniden yaparsan adım {', '.join(later)} de ondan sonra sırayla yeniden yapılır.")
    if step == 0:
        impact.insert(0, "Satranç fotoğraflarını değiştirdiysen ya da eklediysen kalibrasyonu yeniden yap.")
    if step in (0, 2, 4):
        impact.append("Var olan dosyanın üzerine yazılmadan önce yedeği alınır.")
    with st.container(border=True):
        text, button = st.columns([3, 1], vertical_alignment="center")
        with text, st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.markdown("✓ **Bitti** — " + ("bu kurulumun kalibrasyonu hazır." if step == 0
                                            else "çıktısı diskte ve güncel."), width="content")
            kart.note(impact, key=f"bitti_{plan.name}_{step}")
        button.button("↺ Yeniden yap", key=f"ku_{plan.name}_{step}_yeniden", on_click=start_redo,
                      args=(plan.name, step), width="stretch", help="Bu adımın girdi ve ayarlarını yeniden açar.")


def redo_note(plan, step: int, progress: Progress) -> None:
    """Yeniden yapılacak adımın kartında neden: seçildi, girdisi değişti ya da önceki adım yeniden yapılacak."""
    if not progress.done[step]:
        return
    if step in progress.redo:
        text, button = st.columns([3, 1], vertical_alignment="center")
        text.info("↻ **Yeniden yapılıyor** — çalıştırınca çıktı yeniden üretilir, sonraki adımlar da sırayla "
                  "yeniden yapılır.")
        button.button("✕ Vazgeç", key=f"ku_{plan.name}_{step}_vazgec", on_click=cancel_redo, args=(plan.name, step),
                      width="stretch", help="Adım eski haliyle kalır.")
    elif progress.outdated[step]:
        st.warning(f"↻ **Güncel değil:** {', '.join(progress.outdated[step])} bu adımın çıktısından sonra değişti; "
                   "adımı yeniden çalıştır.")
    else:
        first = next(k for k in range(step) if progress.changed(k))
        st.info(f"↻ Adım {first + 1} yeniden yapılacağı için bu adım da ondan sonra yeniden yapılmalı.")


LOCKED_NOTE = ("Adım bittiği için Girdi ve Ayarlar kilitli: bu adımın kullandığı, kurulum dosyasında kayıtlı değerler "
               "görünür. Değiştirmek için Çalıştır bölümündeki **↺ Yeniden yap**.")


def setup_settings(plan, op: katalog.Operation, step: int, build, image=None,
                   locked: bool = False) -> tuple[dict, list[str], list]:
    """Adımın Ayarlar bölümü: Tek işlem sayfasındaki aynı işlemle aynı ayarlar, sınırlar ve başlangıç değerleri
    (bu kurulumun dosyasında kayıtlı değer, yoksa modülün varsayılanı) ve sabit ayarları. build(değerler) adımın
    komut(lar)ını kurar. locked: biten adım; kayıtlı değerler değiştirilemez halde görünür.
    Dönüş: (değerler, hatalar, komutlar)."""
    key = f"ku_{plan.name}_{step}"
    if locked:                              # dosya değişince (yeniden yapılınca) kilitli görünüm de yenilensin
        key += f"_kayit{max((p.stat().st_mtime_ns for p in plan.outputs(step)), default=0)}"
    with kart.section("Ayarlar") as notes:
        size = (image.shape[1], image.shape[0]) if image is not None else None
        values, errors = ayarlar.setting_widgets(op, katalog.setting_defaults(op, plan.name, size), key=key,
                                                 busy=locked, image=image)
        commands = build(values) if not errors else []
        fixed = katalog.operation_fixed_rows(op, commands)
        ayarlar.fixed_settings(fixed)
        notes.extra = ayarlar.counts(len(op.settings), len(fixed)) + (" · kilitli" if locked else "")
        notes.add(*([LOCKED_NOTE] if locked else []), *katalog.SETTING_NOTES.get(op.key, []), katalog.SAVED_NOTE,
                  ayarlar.FIXED_NOTE)
    return values, errors, commands


def fixed_only(op: katalog.Operation, commands: list, note: str, waiting: str = "", locked: bool = False) -> None:
    """Ayarlanabilir ayarı olmayan (ya da henüz gösterilemeyen) adımın Ayarlar bölümü: yalnız sabit ayarlar.
    waiting: ayarlanabilir ayarlar ne zaman görünecek (ör. "adım 2'den sonra"); locked: biten adım."""
    with kart.section("Ayarlar") as notes:
        fixed = katalog.operation_fixed_rows(op, commands)
        ayarlar.fixed_settings(fixed)
        notes.extra = (f"{len(op.settings)} ayarlanabilir ({waiting}) · {len(fixed)} sabit" if waiting
                       else ayarlar.counts(len(op.settings), len(fixed))) + (" · kilitli" if locked else "")
        notes.add(*([LOCKED_NOTE] if locked else []), note, ayarlar.FIXED_NOTE)


# --------------------------------------------------------------------------- adım 1: kalibrasyon
def form_calibration(plan, progress, locked: bool = False):
    photos, folder = plan.photos(), plan.photos_dir
    with kart.section("Girdi", extra=f"{len(photos)} satranç fotoğrafı") as notes:
        input_status(photos, "satranç fotoğrafı", key=f"ku_{plan.name}_0_goster",
                     save=None if locked else lambda uploads: _save_photos(plan, uploads))
        notes.add(f"{'✓' if photos else '✗'} {len(photos)} fotoğraf — `{sonuc.relative(folder)}/` — gerçek veri",
                  "Videoyu çeken kameradan, videoyla aynı çözünürlükte 15–20 satranç tahtası fotoğrafı. Bu klasöre "
                  "kendin koyabilir ya da **＋ Ekle** ile ekleyebilirsin; eklenenler aynı klasöre kaydedilir.")

    op = katalog.OPERATIONS["1.1-kal"]
    values, errors, commands = setup_settings(
        plan, op, 0, lambda v: katalog.setup_commands(plan, 0, {"photos": folder, "pattern": v["pattern"]})
        if photos else [], locked=locked)
    if locked:
        return None
    locks = [] if photos else [Lock("", "Satranç fotoğrafları", f"`{sonuc.relative(folder)}/` içinde fotoğraf yok; "
                                                                 "fotoğraf ekle.")]
    return run_section(plan, 0, commands, locks, [plan.calibration], errors, progress.other,
                       info=(f"Kalibrasyon yalnız bu kurulumun (`{plan.name}`) videolarında kullanılır.",), op=op)


def result_calibration(plan, job, notes):
    table = calibration_table(job.snapshot()) if job is not None else pd.DataFrame()
    if job is not None:
        sonuc.show_job_outcome(job)
    if job is None and plan.calibration.is_file():
        notes.add("Kullanılan / atlanan fotoğraf listesi adım bu sayfadan çalıştırılınca gösterilir.")
    if plan.calibration.is_file():
        items = katalog.calibration_metrics(plan.calibration)
        if not table.empty:
            used = int((table["Durum"] == "✓ kullanıldı").sum())
            items.append(("Kullanılan fotoğraf", f"{used} / {len(table)}",
                          "Köşesi bulunamayan ya da boyutu farklı fotoğraflar atlanır."))
        sonuc.metric_cards(items, columns=4)
        sonuc.show_views([katalog.View("images", plan.corners_dir, "*_corners.*", "Köşe kontrol görüntüleri")],
                         key=f"ku_{plan.name}_0_g", notes=notes)
    if not table.empty:
        with st.expander("Fotoğraf başına sonuç: kullanılan ve atlanan"):
            st.dataframe(table, hide_index=True, width="stretch")
    if plan.calibration.is_file():
        sonuc.show_views([katalog.View("npz", plan.calibration, label="Kalibrasyon dosyası")],
                         key=f"ku_{plan.name}_0_d", notes=notes)


# --------------------------------------------------------------------------- adım 2: düzeltme
def form_undistort(plan, progress, locked: bool = False):
    frames = plan.straight_frames()
    with kart.section("Girdi", extra=f"{len(frames)} düz yol karesi") as notes:
        input_status(frames, "düz yol karesi", key=f"ku_{plan.name}_1_goster",
                     save=None if locked else lambda uploads: _save_frames(plan, uploads))
        notes.add(f"{'✓' if frames else '✗'} {len(frames)} düz yol karesi (`duz_yol_*`) — "
                  f"`{sonuc.relative(plan.frames_dir)}/` — gerçek veri",
                  "Şeridin düz gittiği, iki çizginin de göründüğü kareler (aynı kamera, aynı takılış). **＋ Ekle** ile "
                  "yüklenenler duz_yol_<n> adıyla kaydedilir.")
    commands = katalog.setup_commands(plan, 1, {}) if frames else []
    fixed_only(katalog.OPERATIONS["1.1-duz"], commands, "Bu adımın ayarlanabilir ayarı yok; kalibrasyon dosyası "
                                                        "kullanılır.", locked=locked)
    if locked:
        return None
    locks = previous_locks(plan, 1, progress)
    if not frames:
        locks.append(Lock("", "Düz yol kareleri", f"`{sonuc.relative(plan.frames_dir)}` içinde duz_yol_* kare yok; "
                                                 "kare ekle."))
    return run_section(plan, 1, commands, locks, [], [], progress.other)


def result_undistort(plan, job, notes):
    if job is not None:
        sonuc.show_job_outcome(job)
    sonuc.show_views([katalog.View("images", plan.corrected_dir, "duz_yol_*_undistorted.*", "Önce / sonra",
                                   pairs=plan.frames_dir)], key=f"ku_{plan.name}_1_g", notes=notes)


# --------------------------------------------------------------------------- adım 3: perspektif
def form_perspective(plan, progress, locked: bool = False):
    frames = plan.corrected_frames()
    image_path = None
    with kart.section("Girdi") as notes:
        if frames and locked:               # biten adım: son seçilen kare, değiştirilemez (noktaların önizlemesi için)
            chosen = st.session_state.get(f"ku_{plan.name}_2_kare")
            image_path = chosen if chosen in frames else frames[0]
            st.selectbox("Düzeltilmiş düz yol karesi", [image_path], format_func=lambda p: p.name, disabled=True,
                         key=f"ku_{plan.name}_2_kare_kayit")
            notes.add(f"✓ `{sonuc.relative(image_path)}` — adım 2'nin çıktısı (noktaların önizlemesi bu karede)")
        elif frames:
            image_path = st.selectbox("Düzeltilmiş düz yol karesi", frames, format_func=lambda p: p.name,
                                      key=f"ku_{plan.name}_2_kare")
            notes.add(f"✓ `{sonuc.relative(image_path)}` — adım 2'nin çıktısı")
        else:
            notes.add("✗ Düzeltilmiş düz yol karesi yok — adım 2'nin çıktısı")
    op = katalog.OPERATIONS["1.2-persp"]
    image = cv2.imread(str(image_path)) if image_path else None
    if image is not None:
        values, errors, commands = setup_settings(
            plan, op, 2, lambda v: katalog.setup_commands(plan, 2, {"image": image_path, "src": v["src"]}), image,
            locked=locked)
    else:
        fixed_only(op, [], "4 kaynak nokta düzeltilmiş düz yol karesinde seçilir; adım 2 bitince burada görünür.",
                   waiting="adım 2'den sonra", locked=locked)
        values, errors, commands = {}, [], []
    if locked:
        return None
    locks = previous_locks(plan, 2, progress)
    return run_section(plan, 2, commands, locks, [plan.perspective], errors, progress.other, op=op)


def result_perspective(plan, job, notes):
    if job is not None:
        sonuc.show_job_outcome(job)
    sonuc.show_views([katalog.View("images", plan.verify_dir, "*_verification.*", "Kontrol görüntüsü"),
                      katalog.View("npz", plan.perspective, label="Perspektif dosyası")],
                     key=f"ku_{plan.name}_2_g", notes=notes)


# --------------------------------------------------------------------------- adım 4: kuş bakışı
def form_birdseye(plan, progress, locked: bool = False):
    frames = plan.corrected_frames()
    with kart.section("Girdi", extra="adım 2 ve 3'ün çıktıları") as notes:
        notes.add(f"{'✓' if frames else '✗'} {len(frames)} düzeltilmiş düz yol karesi — adım 2'nin çıktısı",
                  f"{'✓' if plan.perspective.is_file() else '✗'} `{sonuc.relative(plan.perspective)}` — adım 3'ün çıktısı")
    commands = katalog.setup_commands(plan, 3, {}) if frames and plan.perspective.is_file() else []
    fixed_only(katalog.OPERATIONS["1.2-duzyol"], commands, "Ayarlanabilir ayar yok: her düz yol karesi perspektif "
                                                           "dosyasıyla kuş bakışına çevrilir.", locked=locked)
    if locked:
        return None
    locks = previous_locks(plan, 3, progress)
    return run_section(plan, 3, commands, locks, [], [], progress.other)


def result_birdseye(plan, job, notes):
    if job is not None:
        sonuc.show_job_outcome(job)
    sonuc.show_views([katalog.View("images", plan.birdseye_dir, "duz_yol_*_warped.*", "Kuş bakışı kareler",
                                   pairs=plan.corrected_dir)], key=f"ku_{plan.name}_3_g", notes=notes)


# --------------------------------------------------------------------------- adım 5: ölçek
def form_scale(plan, progress, locked: bool = False):
    birdseye = plan.birdseye_frames()
    with kart.section("Girdi", extra="adım 4'ün çıktısı") as notes:
        notes.add(f"{'✓' if birdseye else '✗'} {len(birdseye)} kuş bakışı düz yol karesi — adım 4'ün çıktısı")
    op = katalog.OPERATIONS["1.3"]
    values, errors, commands = setup_settings(plan, op, 4,
                                              lambda v: katalog.setup_commands(plan, 4, v) if birdseye else [],
                                              locked=locked)
    if locked:
        return None
    locks = previous_locks(plan, 4, progress)
    return run_section(plan, 4, commands, locks, [plan.scale], errors, progress.other, op=op)


def result_scale(plan, job, notes):
    if job is not None:
        sonuc.show_job_outcome(job)
    sonuc.show_views([katalog.View("metrics", plan.scale, metrics=katalog.scale_metrics),
                      katalog.View("images", plan.scale_dir, "*_karsilastirma.png", "Karşılaştırma görüntüsü"),
                      katalog.View("npz", plan.scale, label="Ölçek dosyası")], key=f"ku_{plan.name}_4_g", notes=notes)


FORMS = (form_calibration, form_undistort, form_perspective, form_birdseye, form_scale)
RESULTS = (result_calibration, result_undistort, result_perspective, result_birdseye, result_scale)


def step_card(plan, step: int, progress: Progress) -> None:
    """Adımın kartı, her zaman aynı sırayla: Girdi -> Ayarlar -> Çalıştır -> İlerleme -> Sonuç. Yapılacak adımda
    girdi ve ayarlar değiştirilebilir; biten adımda kilitlidir (kullanılan değerler görünür) ve Çalıştır bölümünde
    "✓ Bitti" ile "↺ Yeniden yap" durur."""
    job = st.session_state.setdefault(JOBS_KEY, {}).get(f"{plan.name}:{step}")
    if progress.editing(step):
        redo_note(plan, step, progress)
        job, busy = FORMS[step](plan, progress)
        if busy:
            return
    else:
        FORMS[step](plan, progress, locked=True)
        with kart.section("Çalıştır"):
            done_note(plan, step, progress)
        if job is not None:
            with kart.section("İlerleme"):
                terminal.show_job(job)
    if job is not None or plan.outputs(step):             # gösterilecek bir şey yoksa Sonuç bölümü çıkmaz
        with kart.section("Sonuç") as notes:
            RESULTS[step](plan, job, notes)


# =========================================================================== sayfa
corner = kart.page_title("Kurulum", [
    "İlk kurulumun 5 adımını sırayla yaptırır. Bir adım bitmeden sonraki kilitli kalır; biten bir adım **↺ Yeniden "
    "yap** ile yeniden yapılabilir, ondan sonraki adımlar da sırayla yeniden yapılır.",
    "Her adım komut satırındaki gibi arka planda çalışır. Kurulum yalnız bir kez (ya da kamera veya montaj "
    "değişince) yapılır; ana hat bu dosyaları kullanır."])

jobs = st.session_state.setdefault(JOBS_KEY, {})
running = next((key for key, job in jobs.items() if job.running), None)

options = [*katalog.setup_candidates(), NEW]
if st.session_state.get("ku_ad") not in options:
    st.session_state["ku_ad"] = "kolay_video" if "kolay_video" in options else options[0]
top_name = st.columns([2, 1.4])[0]
choice = top_name.selectbox("Kurulum adı", options, key="ku_ad", disabled=running is not None,
                            help="Kurulum adı kamerayı (ve takılışını) temsil eder: fotoğraflar, kareler, videolar ve "
                                 "kurulum dosyaları bu adın klasörlerindedir (ör. videolar/kamera1/aksam.mp4).")
name = choice
if choice == NEW:
    with top_name, st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        name = st.text_input("Yeni kurulumun adı", key="ku_yeni_ad", placeholder="ör. kamera1",
                             help=f"{katalog.SETUP_NAME_HELP} Oluştur'a basınca bu adla dört klasör açılır "
                                  "(satranç fotoğrafları, test kareleri, videolar, kurulum dosyaları); dosyaları oraya "
                                  "koyabilir ya da adımlarda ekleyebilirsin.").strip()
        valid = katalog.valid_setup_name(name)
        st.button("＋ Oluştur", key="ku_olustur", type="primary", disabled=not valid, on_click=create_setup)
    if name and not valid:
        st.error(f"✗ Geçersiz ad: `{name}`. {katalog.SETUP_NAME_HELP}")
    elif name in options:
        st.info(f"`{name}` adında bir kurulum zaten var; **＋ Oluştur** onu seçer.")
    st.stop()
with corner:
    sifirla.reset_button("kurulum", name, help=f"`{name}` kurulumunun ürettiklerini siler: kurulum klasörü "
                                               "(kalibrasyon, perspektif, ölçek), yedekleri ve 5 adımın outputs/ "
                                               "altındaki bütün ara çıktıları; boş klasör kalmaz. Toplanan veriler "
                                               "kalır, kurulum yeniden yapılabilir.")
    sifirla.delete_button(name)

finish_redo(name, jobs)
plan = katalog.setup_plan(name)
data_row(plan)

done = plan.done()
progress = Progress(done, plan.outdated(), set(redo_steps(name)))
data_missing = [not plan.photos(), not plan.straight_frames(), False, False, False]
locked = [data_missing[i] or any(not done[k] or progress.needs(k) for k in REQUIRES[i]) for i in range(5)]
first_open = next((i for i in range(5) if progress.editing(i) and not locked[i]), None)
states = []
for i in range(5):
    if running == f"{name}:{i}":
        states.append("running")
    elif i == first_open:
        states.append("next")
    elif progress.needs(i):
        states.append("redo")
    elif done[i]:
        states.append("done")
    else:
        states.append("locked" if locked[i] else "ready")
step_key = f"kurulum_adim:{name}"
selected = st.session_state.get(step_key, first_open if first_open is not None else 0)

strip, card = st.columns([1, 2.7], gap="medium")
with strip:
    kart.heading("Adımlar", ["✓ bitti · ● sırada · ▶ çalıştırılabilir · ↻ yeniden yapılmalı · 🔒 kilitli",
                             f"Girdiler: `{sonuc.relative(plan.photos_dir)}/` (satranç fotoğrafları), "
                             f"`{sonuc.relative(plan.frames_dir)}/` (düz yol kareleri).",
                             f"Sonuçlar: `{sonuc.relative(katalog.setup_dir(name))}/` (kalibrasyon, perspektif, "
                             "ölçek)."], key=f"adimlar_{name}")
    for i, (number, title, _) in enumerate(STEPS):
        suffix = " (mevcut)" if i == 0 and done[0] and not progress.needs(0) else ""
        if st.button(f"{STRIP[states[i]]} Adım {i + 1} · {number} {title}{suffix}", key=f"ku_adim_{name}_{i}",
                     width="stretch", type="primary" if i == selected else "secondary"):
            st.session_state[step_key] = i
            st.rerun()
    if all(done) and not any(progress.needs(i) for i in range(5)):
        st.success(f"✓ **Kurulum hazır:** `{name}`")
        st.page_link("sayfalar/ana_hat.py", label="Ana hatta git", icon=":material/play_circle:")
    elif any(progress.needs(i) for i in range(5)):
        st.warning("↻ **Kurulum güncel değil:** ↻ işaretli adımları sırayla yeniden yap.")

if running and running != f"{name}:{selected}":
    run_name, run_step = running.split(":")
    progress.other = f"{run_name} — adım {int(run_step) + 1}"
with card:
    number, title, purpose = STEPS[selected]
    with kart.operation_card(f"Adım {selected + 1} · {number}", title, purpose, state=states[selected]):
        step_card(plan, selected, progress)
