"""
Arayüz — Ana hat sayfası: seçilen videoyu tek düğmeyle baştan sona işler.

Girdi (video + kendiliğinden eşleşen kurulum) -> Ayarlar -> Çalıştır (kurulum ya da girdi eksikse kilitli)
-> İlerleme (çubuk; çalışırken canlı önizleme ve anlık değerler; açılır terminal çıktısı) -> Sonuç (video, dört
özet değer, indirme; grafikler ve kare kare tablo açılır kutuda).
"""

import math
import tempfile
from pathlib import Path

import streamlit as st

import katalog
from servisler import ana_hat as servis
from bilesenler import ayarlar, kart, sifirla, sonuc, terminal
from bilesenler.kilit import Lock, show_locks

RUN_KEY = "ana_hat_calismasi"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "ads_yuklenenler"   # yüklenen videolar (proje klasörüne yazılmaz)
OFFSET_TEXT = {"sağa": "sağa", "sola": "sola", "ortada": "ortada"}
CURVE_TEXT = {"sag": "sağa viraj", "sol": "sola viraj", "duz": "düz yol", "belirsiz": "belirsiz", "yok": "—"}
RISK_TEXT = {"kirmizi": "kırmızı", "sari": "sarı", "yesil": "yeşil"}


def _fmt(value: float, pattern: str, empty: str = "—") -> str:
    return empty if value is None or (isinstance(value, float) and not math.isfinite(value)) else pattern.format(value)


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "—"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} dk {secs} sn" if minutes else f"{secs} sn"


def _saved_upload(upload) -> Path:
    """Yüklenen videoyu geçici klasöre yazar (aynı ad ve boyutta tekrar yazmaz)."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / Path(upload.name).name
    if not path.is_file() or path.stat().st_size != upload.size:
        path.write_bytes(upload.getbuffer())
    return path


def live_values(result) -> None:
    """Anlık değer kartları: sapma, viraj, engel ve risk."""
    if result is None:
        sonuc.metric_cards([("Yanal sapma", "—"), ("Viraj", "—"), ("Engel", "—"), ("Risk", "—")], columns=2)
        return
    offset = result.offset
    nearest = result.nearest_in_lane()
    in_lane = sum(o["status"] == "seritte" for o in result.obstacles)
    sonuc.metric_cards([
        ("Yanal sapma", f"{offset.offset_m:+.2f} m" if offset else "—",
         OFFSET_TEXT.get(offset.direction, "") if offset else "şerit yok"),
        ("Viraj", _fmt(result.radius_m, "R {:.0f} m") if result.curve_direction in ("sag", "sol")
         else CURVE_TEXT.get(result.curve_direction, "—"),
         f"{CURVE_TEXT.get(result.curve_direction, '')} · {result.curve_class}"),
        ("Engel", f"{len(result.obstacles)} nesne", f"şeritte {in_lane}"),
        ("Risk", RISK_TEXT.get(nearest["risk"], nearest["risk"]) if nearest else "yok",
         f"en yakın engel {nearest['distance_m']:.0f} m" if nearest else "şeritte engel yok"),
    ], columns=2)


def _status(run: "servis.PipelineRun") -> str:
    """İlerleme çubuğunun yazısı: çalışırken kare, hız ve kalan süre; bitince nasıl bittiği ve toplam süre."""
    if run.state == "starting":
        return "Hazırlanıyor: modüller ve model yükleniyor, cihaz seçiliyor…"
    if run.running:
        remaining = (run.total - run.done) / run.fps if run.fps > 0 and run.total else None
        return f"● {run.done} / {run.total} kare · {run.fps:.1f} kare/sn · kalan {_duration(remaining)}"
    word = {"done": "✓ Bitti", "stopped": "■ Durduruldu", "error": "✗ Hata verdi"}.get(run.state, "")
    device = f" · YOLO: {run.device}" if run.device else ""
    return (f"{word} · {run.done} kare · {run.fps:.1f} kare/sn · {_duration(run.finished - run.started)}"
            f"{device}")


def progress_view(run: "servis.PipelineRun") -> None:
    """İlerleme çubuğu; çalışırken canlı önizleme ve anlık değerler; açılır terminal çıktısı."""
    st.progress(min(run.done / max(run.total, 1), 1.0) if run.state != "done" else 1.0, text=_status(run))
    if run.running:
        preview, values = st.columns([3, 2])
        with preview:
            if run.preview:
                st.image(run.preview, caption="Canlı önizleme", width="stretch")
        with values:
            live_values(run.last)
    footer = "● çalışıyor …" if run.running else {"done": "[bitti]", "stopped": "[durduruldu]",
                                                    "error": "[hata]"}.get(run.state, "")
    with st.expander("Terminal çıktısı"):
        terminal.terminal(run.command_text, run.snapshot(), footer, height=200)


@st.fragment(run_every=0.5)
def live_progress(run: "servis.PipelineRun") -> None:
    progress_view(run)
    if not run.running:
        st.rerun()                       # bitti: Çalıştır düğmesi ve Sonuç bölümü görünsün


def results_view(run: "servis.PipelineRun", notes: kart.Notes) -> None:
    """Sonuç: işlenmiş video, dört özet değer, indirme; grafikler ve kare kare tablo açılır kutuda."""
    if run.state == "error":
        lines = run.error.splitlines()
        sonuc.show_outcome("error", errors=[[lines[0], *(f"-> {line}" for line in lines[1:])]])
    elif run.state == "stopped":
        sonuc.show_outcome("stopped")
    if not run.results:
        return
    summary = run.summary()
    n = summary["frames"]
    if run.video_path.is_file():
        sonuc.video_player(run.video_path)

    risks = summary["risks"]
    shown = summary["measured"] + summary["history"]
    limits = katalog.module_defaults(katalog.SCRIPTS["5.5"])           # ana hat 5.5'i kendi varsayılanlarıyla kullanır
    width_help = "Ortalama; gerçek genişliğe yakın olmalı (değilse kurulumun ölçeği yanlış olabilir)."
    real = katalog.setup_lane_width(run.settings.setup)
    if real is not None:
        width_help = f"Ortalama; kurulumdaki gerçek genişlik {real:.2f} m, yakın olmalı (değilse ölçek yanlış olabilir)."
    items = [
        ("Şerit bulunan kare", f"%{100 * shown / n:.0f}",
         f"Şeridin çizildiği kareler. %{100 * summary['measured'] / n:.0f}'inde o karede ölçüldü, geri kalanında son "
         f"geçerli ölçüm gösterildi; {summary['none']} karede şerit yok."),
        ("Ortalama sapma", _fmt(summary["offset"], "{:+.2f} m"), "+ araç şerit ortasının sağında, − solunda."),
        ("Viraj (ortanca R)", _fmt(summary["radius"], "{:.0f} m", "düz yol"),
         f"Virajlı karelerin ortanca yarıçapı; büyük R hafif viraj. Düz yol {summary['straight']} kare."),
    ]
    if run.settings.obstacles:
        items.append(("Riskli kare", f"{risks['kirmizi'] + risks['sari']}",
                      f"Şeritte yakın engel olan kareler: kırmızı (< {limits.red_m:g} m) {risks['kirmizi']}, "
                      f"sarı (< {limits.yellow_m:g} m) {risks['sari']}."))
    else:
        items.append(("Şerit genişliği", _fmt(summary["width"], "{:.2f} m"), width_help))
    sonuc.metric_cards(items, columns=4)
    sonuc.downloads([(run.video_path, "Video"), (run.csv_path, "Tablo (CSV)")], key="ah_indir")
    sonuc.saved_note(notes, run.video_path, "İşlenmiş video")
    sonuc.saved_note(notes, run.csv_path, "Kare kare tablo")

    with st.expander("Ayrıntılar: grafikler ve kare kare tablo"):
        extra = [("Şerit genişliği", _fmt(summary["width"], "{:.2f} m"), width_help)] if run.settings.obstacles else []
        extra.append(("Hız", f"{run.fps:.1f} kare/sn", run.device or "engel tespiti kapalı"))
        if run.settings.obstacles:
            extra += [("Engel tespiti", f"{summary['detections']}", "Bütün karelerdeki filtrelenmiş tespitler (5.3)."),
                      ("Şeritte engel olan kare", f"{summary['in_lane']}", "Kendi şeridimizde engel bulunan kareler (5.5).")]
        sonuc.metric_cards(extra, columns=4)
        frame = run.frame_table()
        charts = [("Yanal sapma", "sapma_m", "sapma (m)"), ("Eğrilik yarıçapı", "egrilik_r_m", "R (m)")]
        if run.settings.obstacles:
            charts.append(("Risk", "risk_duzeyi", "0 yok · 1 yeşil · 2 sarı · 3 kırmızı"))
        for title, column, y_label in charts:
            if column in frame and frame[column].notna().any():
                st.markdown(f"**{title}**")
                st.line_chart(frame, x="zaman_sn", y=column, x_label="zaman (sn)", y_label=y_label, height=200)
        st.markdown("**Kare kare tablo**")
        st.dataframe(frame, hide_index=True, width="stretch", height=300)


# =========================================================================== sayfa
corner = kart.page_title("Ana hat", [
    "Seçilen videoyu tek düğmeyle baştan sona işler: şerit alanı, yanal sapma, viraj ve engeller.",
    "Önce kurulum gerekir (kalibrasyon, perspektif, ölçek); kurulum yoksa Çalıştır kilitlidir ve kilit düğmesi "
    "Kurulum sayfasına götürür."])
with corner:
    sifirla.reset_button("ana_hat", help="İşlenmiş videoları ve tabloları siler; kuruluma dokunmaz.")

try:
    main = servis.load_main()              # varsayılan ayarlar ana hattın kendi AYARLAR bölümünden
except Exception as exc:                   # backend yüklenemezse sayfa yine açılsın
    st.error(f"✗ **Ana hat yüklenemedi:** {exc}")
    st.stop()

run = st.session_state.get(RUN_KEY)
busy = run is not None and run.running

with kart.operation_card("", "Ana hat", "Her kare düzeltilir, şerit bulunur, sapma ve viraj ölçülür, engeller "
                         "şeritle ilişkilendirilir; sonuç videonun üzerine çizilir.",
                         state="running" if busy else None):
    # ------------------------------------------------------------------ 1. Girdi
    with kart.section("Girdi") as notes:
        source = st.segmented_control("Video kaynağı", ["Proje videoları", "Video yükle"], default="Proje videoları",
                                      required=True, key="ah_kaynak", disabled=busy)
        video = None
        if source == "Video yükle":
            upload = st.file_uploader("Video", type=[e.lstrip(".") for e in katalog.VIDEO_EXTENSIONS], disabled=busy,
                                      label_visibility="collapsed")
            notes.add("Yüklenen video geçici bir klasöre yazılır, proje klasörüne yazılmaz.")
            if upload is not None:
                video = _saved_upload(upload)
        else:
            options = katalog.videos()
            if options:
                video = st.selectbox("Video", options, format_func=katalog.video_label, key="ah_video", disabled=busy,
                                     help="data/ortam_verisi/videolar/<kurulum>/ klasörlerindeki videolar.")
            else:
                st.warning("! `data/ortam_verisi/videolar/` içinde video yok; bir video yükle.")

        setups = katalog.setup_candidates()
        matched = katalog.setup_for_video(video) if video else None
        setup = None
        if setups:
            setup = st.selectbox("Kurulum", setups, index=setups.index(matched) if matched in setups else 0,
                                 key=f"ah_kurulum_{video.name if video else ''}", disabled=busy,
                                 help="Videonun klasöründen kendiliğinden seçilir (videolar/<kurulum>/); değiştirilebilir.")
            if video and matched is None:
                st.warning("! Bu video bir kurulumun klasöründe değil; seçilen kurulum başka bir kamera ya da takılış "
                           "için olabilir.")
            outdated = katalog.setup_outdated(setup)
            if outdated:
                st.warning(f"↻ `{setup}` kurulumu güncel olmayabilir: {', '.join(outdated)}.")
                notes.add(f"`{setup}`: {', '.join(outdated)}, kendinden önceki kurulum adımları değiştikten sonra "
                          "yeniden üretilmedi. Ana hat çalışır; doğru sonuç için **Kurulum** sayfasında ↻ işaretli "
                          "adımları sırayla yeniden yap.")
        info = servis.video_info(video)
        if info:
            notes.add(f"**Video:** {info['width']}x{info['height']} · {info['fps']:.2f} kare/sn · {info['frames']} kare "
                      f"· {info['frames'] / info['fps']:.0f} sn")
        notes.add("Kurulum videonun klasöründen seçilir (`videolar/<kurulum>/`); klasör dışındaki ya da yüklenen "
                  "videoda adının başından (ör. `zor_video_aksam.mp4` → `zor_video`). Kalibrasyon kameraya, "
                  "perspektif ve ölçek kameranın takılışına özeldir.")

    # ------------------------------------------------------------------ 2. Ayarlar
    with kart.section("Ayarlar") as notes:
        left, right = st.columns(2)
        obstacles = left.toggle("Engel tespiti (YOLO)", value=main.OBSTACLES, key="ah_engel", disabled=busy,
                                help="Kapalıyken yalnız şerit kolu çalışır (daha hızlı).")
        models = katalog.yolo_models()
        model = katalog.MODEL_PATH
        if models:
            model = left.selectbox("YOLO modeli", models, format_func=katalog.model_label, key="ah_model",
                                   disabled=busy or not obstacles,
                                   help="n en hızlısı; s, m, l, x daha isabetli ama daha yavaş.")
        whole = right.toggle("Videonun tamamı", value=True, key="ah_tamami", disabled=busy)
        max_frames = None
        if not whole:
            limit = info["frames"] if info and info["frames"] > 0 else katalog.MAX_FRAMES_LIMIT
            max_frames = int(right.number_input("Kaç kare işlensin", min_value=1, max_value=limit, value=min(300, limit),
                                                step=50, key="ah_kare", disabled=busy))
        left, right = st.columns(2)
        smooth = left.slider("Ortalaması alınacak kare sayısı", 1, 20, main.SMOOTH_FRAMES, key="ah_smooth",
                             disabled=busy, help="Çizilen şerit son bu kadar karenin ortalamasıdır. Büyük değer "
                                                 "titremeyi azaltır ama virajlarda geç kalır; küçük değer hızlı tepki "
                                                 "verir ama titrer.")
        combine = right.segmented_control("Maske birleştirme (4.4)", ["or", "and"], default=main.COMBINE_MODE,
                                          required=True, key="ah_birlestir", disabled=busy,
                                          format_func={"or": "OR — biri yeterli", "and": "AND — ikisi de gerekli"}.get,
                                          help="Bir piksel şerit sayılsın mı: OR, renk ya da kenar maskesinden biri "
                                               "bulursa (daha çok çizgi, biraz gürültü); AND, ikisi de bulursa (daha "
                                               "temiz, ama soluk ya da gölgeli çizgi kaybolabilir).")
        adjustable = 4 + bool(models)          # engel tespiti, (model), işlenecek kare, ortalama, birleştirme
        settings = (servis.Settings(video, setup, obstacles, max_frames, smooth, main.LANE_WIDTH_RANGE_M, combine,
                                    main.YOLO_DEVICE, model)
                    if video is not None and setup is not None else None)
        fixed = servis.fixed_rows(settings, obstacles)
        ayarlar.fixed_settings(fixed)
        notes.extra = ayarlar.counts(adjustable, len(fixed))
        notes.add("Varsayılanlar ana hattın kendi AYARLAR bölümünden gelir (`backend/entegrasyon/main.py`); değerler "
                  "sınırlıdır.", ayarlar.FIXED_NOTE + " Ana hattın sabit ayarları, her karede kullandığı modüllere "
                  "göre gruplanmıştır (main.py'nin kullandığı değerlerle).",
                  "**YOLO modeli:** `data/hazir_veri/yolo_modeli/` klasörüne konan modeller (.pt) listelenir. "
                  "Boyut harfi: n nano, s küçük, m orta, l büyük, x çok büyük. Büyüdükçe uzaktaki ve küçük araçları "
                  "daha iyi bulur ama yavaşlar.",
                  "**Cihaz** kendiliğinden seçilir: her model için GPU ile CPU bir kez kısa ölçülür, hızlı olan "
                  "kullanılır (GPU yoksa CPU).",
                  "**Kabul edilen şerit genişliği** kendiliğinden: kurulumda girilen gerçek genişliğin ±%20'si "
                  "(3.66 m için 2.93–4.39 m). Bir karede ölçülen genişlik bunun dışındaysa o karede önceki iyi "
                  "ölçüm gösterilir.",
                  "**Ortalaması alınacak kare sayısı** ve **Maske birleştirme** çoğu zaman değiştirilmez: titreme "
                  "ile tepki hızı arasındaki denge ve şerit piksellerinin ne kadar seçici bulunacağı.")

    # ------------------------------------------------------------------ 3. Çalıştır
    with kart.section("Çalıştır") as notes:
        setup_locks, other_locks = servis.run_locks(video, setup, obstacles, info, model)
        if video is None:
            other_locks.append(Lock("", "Video", "Önce bir video seç ya da yükle."))
        if setup_locks:
            show_locks(setup_locks, key="ah_kurulum_kilit", title="Önce kurulum")
        if other_locks:
            show_locks(other_locks, key="ah_diger_kilit", title="Eksik")
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            if busy:
                if st.button("■ Durdur", key="ah_durdur", help="İşlemi durdur; o ana kadar işlenen kareler kaydedilir."):
                    run.stop()
                    st.rerun()
            else:
                locked = bool(setup_locks or other_locks) or settings is None
                if st.button("▶ Çalıştır", type="primary", key="ah_calistir", disabled=locked):
                    st.session_state[RUN_KEY] = servis.PipelineRun(settings).start()
                    st.rerun()
            terminal.command_button(run.command_text if busy else settings.command() if settings else None,
                                    key="ah_komut")
        notes.add("Ana hat bu sayfanın içinde çalışır; komut satırındaki `python main.py` ile aynı işlemdir. "
                  "**Komut** düğmesi komut satırındaki karşılığını gösterir.",
                  "Çıktılar: `outputs/arayuz/<kurulum>/<video>_islenmis.mp4` ve `.csv`.")

    # ------------------------------------------------------------------ 4. İlerleme ve 5. Sonuç
    if run is not None:
        with kart.section("İlerleme"):
            if run.running:
                live_progress(run)
            else:
                progress_view(run)
        if not run.running:
            with kart.section("Sonuç") as notes:
                results_view(run, notes)
