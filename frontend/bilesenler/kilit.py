"""
Arayüz — kilit: bir işlem ihtiyaç duyduğu dosya yoksa kilitlidir.

Kilidin üstünde önce çalıştırılacak işlem numarasıyla yazar; yanında o işleme götüren bir düğme vardır.
Arayüz eksik adımı kendiliğinden çalıştırmaz, yalnız yönlendirir.
"""

import streamlit as st

from katalog import SELECTED_OPERATION, Lock  # noqa: F401  (sayfalar Lock'u buradan da alır)


def show_locks(locks: list[Lock], key: str, title: str = "Önce çalıştır") -> None:
    """Kilit kutusu: neyin eksik olduğu ve o işleme götüren düğmeler."""
    with st.container(border=True):
        st.markdown(f"🔒 **Kilitli** — {title}: " + ", ".join(f"**{lock.label}**" for lock in locks))
        for i, lock in enumerate(locks):
            text, button = st.columns([4, 1], vertical_alignment="center")
            text.markdown(f"<span class='ads-lock'>{lock.reason or lock.label}</span>", unsafe_allow_html=True)
            if lock.page and button.button(f"→ {lock.requirement or lock.name}", key=f"{key}_lock_{i}",
                                           help=f"{lock.label} işlemine git", width="stretch"):
                if lock.target is not None:
                    st.session_state[lock.target_key] = lock.target
                st.switch_page(lock.page)
