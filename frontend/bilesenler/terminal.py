"""
Arayüz — ilerleme ve terminal kutusu: işin durumu tek satırda, programın çıktısı açılır kutuda.

İlerleme bölümünde yalnız durum satırı görünür (çalışıyor / bitti / hata, süre; birden çok komutlu işte çubuk).
Programın çıktısı, komut satırındaki gibi, "Terminal çıktısı" açılır kutusundadır. Çalışırken yarım saniyede bir
yenilenir; iş bitince sayfanın tamamı yenilenir ki sonuç görünsün.
"Durdur" düğmesi işlem kartının Çalıştır bölümündedir (Çalıştır düğmesi çalışırken Durdur'a dönüşür).
HATA satırları kırmızı, UYARI satırları sarı yazılır.
"""

import html

import streamlit as st

from bilesenler import tema

SHOWN_LINES = 400


def _line_class(line: str, block: str | None) -> tuple[str, str | None]:
    if line.startswith("$ "):
        return "ads-cmd", None                  # çalıştırılan komut
    if line.startswith("HATA:"):
        return "ads-error", "ads-error"
    if line.startswith("UYARI:"):
        return "ads-warning", "ads-warning"
    if block and line.startswith(("  ", "\t")) and line.strip():
        return block, block                     # HATA / UYARI bloğunun açıklama satırları
    return "", None


def terminal(command: str, lines: list[str], footer: str = "", height: int = 300) -> None:
    """Terminal görünümlü kutu: komut (sarı) ve çıktı satırları."""
    parts = [f"<span class='ads-cmd'>{html.escape(command)}</span>"] if command else []
    block = None
    for line in lines[-SHOWN_LINES:]:
        css, block = _line_class(line, block)
        text = html.escape(line)
        parts.append(f"<span class='{css}'>{text}</span>" if css else text)
    if footer:
        parts.append(f"<span class='ads-cmd'>{html.escape(footer)}</span>")
    body = "\n".join(parts)
    st.markdown(f"<div class='ads-terminal' style='max-height:{height}px'><pre>{body}</pre></div>",
                unsafe_allow_html=True)


def command_button(commands_text: str | None, key: str) -> None:
    """Çalıştır'ın yanındaki "Komut" düğmesi: tıklayınca çalışacak komut (komut satırındaki karşılığı) açılır."""
    with st.popover("Komut", icon=":material/terminal:", key=key, help="Çalışacak komut (komut satırındaki karşılığı)"):
        if commands_text:
            terminal(commands_text, [], "", height=160)
        else:
            st.markdown("Komut, girdiler hazır ve ayarlar doğru olunca oluşur.")


def job_terminal(job, height: int = 300) -> None:
    """Bir işin (servisler.calistirici.Job) terminal kutusu ve alt satırı (süre, çıkış kodu)."""
    if job.running:
        footer = f"● çalışıyor … {job.elapsed:.1f} sn"
    else:
        footer = f"[çıkış kodu {job.returncode}] {job.elapsed:.1f} sn" + (" — durduruldu" if job.stopped else "")
    terminal("", job.snapshot(), footer, height)        # komutlar çıktının içinde, sırasıyla yazılı


FINISHED = {"success": "✓ Bitti", "warning": "✓ Bitti, uyarı var", "error": "✗ Hata verdi", "stopped": "■ Durduruldu"}


def job_status(job) -> str:
    """İşin tek satırlık durumu: çalışıyor (birden çok komutta kaçıncısı) ya da nasıl bittiği, süresiyle."""
    count = len(job.steps)
    part = f" · komut {getattr(job, 'current', 0) + 1}/{count}" if count > 1 else ""
    if job.running:
        return f"● Çalışıyor{part} · {job.elapsed:.0f} sn"
    return f"{FINISHED[job.outcome()]} · {job.elapsed:.1f} sn"


def job_progress(job, height: int = 300) -> None:
    """İlerleme: durum satırı (birden çok komutta çubuk) ve açılır terminal çıktısı."""
    count = len(job.steps)
    if count > 1:
        done = count if not job.running and job.outcome() in ("success", "warning") else getattr(job, "current", 0)
        st.progress(min(done / count, 1.0), text=job_status(job))
    else:
        st.markdown(f"**{job_status(job)}**")
    with st.expander("Terminal çıktısı"):
        job_terminal(job, height)


def live_job_progress(job, height: int = 300) -> None:
    """Çalışan iş için kendini yenileyen ilerleme; iş bitince sayfanın tamamı yenilenir."""

    @st.fragment(run_every=0.5)
    def _view():
        job_progress(job, height)
        if not job.running:
            st.rerun()            # iş bitti: sonuç bölümü görünsün

    _view()


def show_job(job, key: str = "", height: int = 300) -> None:
    """İş çalışıyorsa canlı, bittiyse durağan ilerleme."""
    if job.running:
        live_job_progress(job, height)
    else:
        job_progress(job, height)


def state_of(job) -> str:
    """İşin tema durumu: running / stopped / error / warning / success."""
    return job.outcome()


__all__ = ["terminal", "job_terminal", "job_progress", "show_job", "state_of", "tema"]
