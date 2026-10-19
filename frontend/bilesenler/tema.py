"""
Arayüz — tema: renkler ve durum simgeleri.

Renkler videodaki çizimlerle aynı anlamı taşır: yeşil başarılı, sarı uyarı, kırmızı hata ya da risk,
gri kilitli. Renk hiçbir yerde tek başına bilgi taşımaz; yanında her zaman simge ve yazı vardır.
Ana renkler .streamlit/config.toml'da; burada zeminin ışık geçişi, yan menü, kart ve terminal kutusu gibi
özel parçaların stili var. Tema: gece otoyolu (koyu gece mavisi zemin, şerit sarısı vurgu). Başlıklar sarı
(sayfa ve kart bölümleri) ya da açık mavi (liste grupları); açıklama düğmesi (?) her yerde mor.
"""

import streamlit as st

COLORS = {
    "background": "#0E1420",   # gece mavisi (gece otoyolu)
    "panel": "#182131",        # paneller ve girişler
    "card": "#131B28",         # işlem kartı
    "sidebar": "#0A0F18",      # yan menü (bir ton koyu)
    "accent": "#F2C230",       # şerit sarısı
    "success": "#2EB872",      # videodaki şerit alanı
    "warning": "#F5A524",      # videodaki sarı risk
    "error": "#E5484D",        # videodaki şeritteki engel
    "muted": "#8C97A8",        # kilitli / bekliyor
    "text": "#E6EAF0",
    "border": "#263246",
    "group": "#7DB9F0",        # liste ve şerit grup başlıkları
    "help": "#A78BFA",         # açıklama düğmesi (?)
}

# Durum -> (simge, yazı, st.badge rengi)
STATES = {
    "done": ("✓", "Bitti", "green"),
    "ready": ("▶", "Çalıştırılabilir", "primary"),
    "next": ("●", "Sırada", "primary"),
    "running": ("●", "Çalışıyor", "orange"),
    "locked": ("🔒", "Kilitli", "gray"),
    "redo": ("↻", "Yeniden yapılmalı", "orange"),
    "success": ("✓", "Başarılı", "green"),
    "warning": ("!", "Uyarı", "orange"),
    "error": ("✗", "Hata", "red"),
    "stopped": ("■", "Durduruldu", "gray"),
}

_CSS = f"""
<style>
/* Zemin: gece otoyolu — sol üstte hafif sarı (far), sağ üstte hafif mavi (sokak lambası) ışık. */
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(1100px 520px at 10% -10%, rgba(242, 194, 48, 0.07), transparent 60%),
        radial-gradient(900px 600px at 105% -5%, rgba(91, 155, 213, 0.09), transparent 55%),
        {COLORS['background']};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{
    background: {COLORS['sidebar']}; border-right: 1px solid {COLORS['border']};
}}
/* Yan paneli açma / kapama okları hep aynı yerde dursun (sayfa ya da yan panel kaydırılınca kaymasın). */
[data-testid="stSidebarHeader"] {{
    position: sticky; top: 0; z-index: 5; background: {COLORS['sidebar']};
}}
[data-testid="stExpandSidebarButton"] {{
    position: fixed; top: 0.75rem; left: 0.75rem; z-index: 1000;
}}
[class*="st-key-ads_kart"] {{ background: {COLORS['card']}; }}
[data-testid="stMetric"] {{ background: {COLORS['panel']}; }}
.ads-terminal {{
    background: #0A0F17; border: 1px solid {COLORS['border']}; border-radius: 8px;
    padding: 10px 12px; overflow: auto; display: flex; flex-direction: column-reverse;
}}
.ads-terminal pre {{
    margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 0.82rem; line-height: 1.35;
    color: {COLORS['text']}; background: transparent;
}}
.ads-terminal .ads-cmd {{ color: {COLORS['accent']}; }}
.ads-terminal .ads-error {{ color: {COLORS['error']}; font-weight: 600; }}
.ads-terminal .ads-warning {{ color: {COLORS['warning']}; font-weight: 600; }}
/* Başlıklar: sayfa (sarı, büyük), kart (beyaz, numara sarı), bölüm (sarı, numaralı), liste grubu (açık mavi). */
.ads-baslik {{ color: {COLORS['accent']}; font-size: 2.1rem; font-weight: 800; line-height: 1.2; }}
.ads-kart-baslik {{ color: #FFFFFF; font-size: 1.3rem; font-weight: 700; }}
.ads-kart-baslik .ads-no {{ color: {COLORS['accent']}; margin-right: 0.3rem; }}
[class*="st-key-ads_bolum_"] {{
    margin-top: 0.9rem; padding-bottom: 0.3rem; border-bottom: 1px solid {COLORS['border']};
}}
.ads-bolum {{
    color: {COLORS['accent']}; font-size: 0.95rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase;
}}
.ads-bolum-no {{
    display: inline-block; width: 1.35rem; height: 1.35rem; margin-right: 0.45rem; border-radius: 50%;
    background: {COLORS['accent']}; color: {COLORS['background']}; text-align: center; line-height: 1.35rem;
    font-size: 0.8rem;
}}
.ads-bolum-ek {{
    margin-left: 0.5rem; color: {COLORS['muted']}; font-weight: 600; text-transform: none; letter-spacing: 0;
}}
.ads-step {{
    margin: 0.9rem 0 0.2rem 0; color: {COLORS['group']}; font-size: 0.82rem; font-weight: 800;
    letter-spacing: 0.06em; text-transform: uppercase;
}}
/* Açıklama düğmesi (?): mor, yuvarlak, oksuz; tıklayınca not açılır. Bileşenlerin kendi ? simgesi de mor. */
[class*="st-key-ads_not_"] button {{
    min-height: 0; height: 1.45rem; width: 1.45rem; padding: 0; border: none; border-radius: 50%;
    background: {COLORS['help']}; color: {COLORS['background']}; box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.22);
}}
[class*="st-key-ads_not_"] button:hover {{ background: #C4B5FD; color: {COLORS['background']}; }}
[class*="st-key-ads_not_"] button p {{ margin: 0; font-size: 0.85rem; font-weight: 800; }}
[class*="st-key-ads_not_"] button [aria-hidden="true"] {{ display: none; }}
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipHoverTarget"] svg {{ color: {COLORS['help']}; }}
[data-testid="stPopoverBody"] {{ border: 1px solid {COLORS['border']}; }}
.ads-lock {{ color: {COLORS['muted']}; }}
</style>
"""


def apply_theme() -> None:
    """Özel parçaların stilini sayfaya ekler (her sayfa çalışmasında bir kez)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def state_label(state: str) -> str:
    """"✓ Bitti" gibi simgeli durum yazısı."""
    icon, text, _ = STATES[state]
    return f"{icon} {text}"


def state_badge(state: str) -> None:
    icon, text, color = STATES[state]
    st.badge(f"{icon} {text}", color=color)
