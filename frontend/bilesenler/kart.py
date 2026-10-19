"""
Arayüz — işlem kartı: her işlem yukarıdan aşağı aynı düzende gösterilir.

Başlık (gereksinim no + ad) -> Girdi -> Ayarlar -> Çalıştır -> İlerleme -> Sonuç ya da Hata. Kullanıcı bir kez
öğrenince üç sayfada da aynı şeyi görür. Açıklamalar sayfada yazı olarak durmaz: her başlığın yanındaki
soru işaretine (tıklayınca açılır) yazılır; sayfada yalnız girdiler, ayarlar, düğmeler ve sonuçlar kalır.
"""

import contextvars
from contextlib import contextmanager

import streamlit as st

from bilesenler import tema

SECTIONS = ("Girdi", "Ayarlar", "Çalıştır", "İlerleme", "Sonuç")
_CARD = contextvars.ContextVar("ads_kart", default="")      # açık kartın adı (bölüm notlarının anahtarı için)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def note(lines, key: str) -> None:
    """Soru işareti düğmesi: tıklayınca not açılır. lines: yazı ya da yazı listesi (boşsa düğme çıkmaz)."""
    lines = [lines] if isinstance(lines, str) else [line for line in (lines or []) if line]
    if not lines:
        return
    with st.popover("?", key=f"ads_not_{_slug(key)}", type="tertiary", help="Açıklama"):
        st.markdown("\n\n".join(lines))


class Notes:
    """Bir bölümün açıklamaları; bölüm bitince başlığındaki soru işaretine yazılır. extra: başlığın yanındaki kısa
    not; blok içinde verilirse (ör. ayar sayıları, ancak ayarlar çizilince belli olur) başlık bölüm bitince yazılır."""

    def __init__(self):
        self.lines: list[str] = []
        self.extra: str | None = None

    def add(self, *lines: str) -> None:
        self.lines += [line for line in lines if line]


def heading(html: str, lines=None, key: str = "", css: str = "ads-step") -> None:
    """Başlık ve yanında soru işareti (liste ve şerit başlıkları için)."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"<div class='{css}'>{html}</div>", unsafe_allow_html=True, width="content")
        note(lines, key or html)


def page_title(title: str, lines=None):
    """Sayfa başlığı (belirgin, renkli) ve açıklaması soru işaretinde; dönüş: sağ üst köşenin yeri
    (sıfırlama düğmesi)."""
    text, corner = st.columns([5, 1], vertical_alignment="center")
    with text:
        heading(title, lines, key=f"sayfa_{title}", css="ads-baslik")
    return corner


@contextmanager
def operation_card(number: str, name: str, purpose: str, state: str | None = None):
    """Kenarlıklı işlem kartı; başlığın yanındaki soru işaretinde ne işe yaradığı yazar."""
    slug = _slug(f"{number}_{name}")
    box = st.container(border=True, key=f"ads_kart_{slug}")      # anahtar: kart arka plan rengi (tema.py)
    with box:
        head, badge = st.columns([5, 1], vertical_alignment="center")
        with head:
            title = f"<span class='ads-no'>{number}</span> {name}" if number else name
            heading(title, purpose, key=f"kart_{slug}", css="ads-kart-baslik")
        if state:
            with badge:
                tema.state_badge(state)
        token = _CARD.set(slug)
        try:
            yield box
        finally:
            _CARD.reset(token)


@contextmanager
def section(title: str, extra: str = ""):
    """Kartın içindeki numaralı bölüm (Girdi, Ayarlar, Çalıştır, İlerleme, Sonuç).

    Bloğun içinde notes.add(...) ile eklenen açıklamalar sayfaya yazılmaz; blok bitince başlığın yanındaki
    soru işaretine yazılır. extra: başlığın yanında kısa, soluk not (ör. "15 satranç fotoğrafı"); blok içinde
    notes.extra ile de verilebilir."""
    number = SECTIONS.index(title) + 1 if title in SECTIONS else None

    def header(extra: str) -> str:
        text = f"<span class='ads-bolum-no'>{number}</span>{title}" if number else title
        if extra:
            text += f"<span class='ads-bolum-ek'>· {extra}</span>"
        return f"<div class='ads-bolum'>{text}</div>"

    with st.container(horizontal=True, vertical_alignment="center", gap="small", key=f"ads_bolum_{_CARD.get()}_{_slug(title)}"):
        title_slot = st.empty()
        title_slot.markdown(header(extra), unsafe_allow_html=True, width="content")
        slot = st.empty()
    notes = Notes()
    yield notes
    if notes.extra is not None:
        title_slot.markdown(header(notes.extra), unsafe_allow_html=True, width="content")
    if notes.lines:
        with slot.container():
            note(notes.lines, key=f"bolum_{_CARD.get()}_{title}")
