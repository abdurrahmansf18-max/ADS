"""
Arayüz — ayar bileşenleri: bir işlemin ayarları Tek işlem ve Kurulum sayfasında aynı biçimde çizilir.

Ayarlar katalog'daki tanımlardan (ad, sınır, adım, açıklama) gelir; iki sayfa aynı tanımı ve aynı başlangıç
değerini (katalog.setting_defaults) kullanır. Değerler sınırlıdır; 4 nokta ayarında anlık önizleme (yamuk ve kuş
bakışı) vardır. Dönüş: (değerler, hatalar); hatalar Çalıştır'ı "Düzelt" kilidiyle kilitler.

Ayarlanabilir ayarların altında "Sabit ayarlar" açılır kutusu modülün diğer bütün ayarlarını gösterir: adı (soru
işaretinde açıklaması), gerçekte kullanılan değeri ve kaynağı (kendiliğinden, kurulumdan ya da sabit).
"""

import cv2
import numpy as np
import streamlit as st

import katalog
from bilesenler import sonuc

CORNERS = ("1 · Sol üst", "2 · Sağ üst", "3 · Sağ alt", "4 · Sol alt")
FIXED_NOTE = ("**Sabit ayarlar:** modülün diğer bütün ayarları, bu çalıştırmada gerçekte kullanılacak değerleriyle "
              "(modülün kendi tanımından okunur); arayüzden değiştirilmez. Kaynak: kendiliğinden (sistem belirler), "
              "kurulumdan (kurulum dosyasından), sabit (modülün kendi değeri). Dosya yolları Girdi bölümündedir.")
COLUMNS = (3, 2.2, 1.2)


def _number_format(step: float | None) -> str:
    if not step or step >= 1:
        return "%.1f"
    decimals = len(f"{step:.10f}".rstrip("0").split(".")[1])
    return f"%.{decimals}f"


def _bounds(item: katalog.Setting, cast) -> dict:
    return {"min_value": cast(item.low) if item.low is not None else None,
            "max_value": cast(item.high) if item.high is not None else None}


def _clamp(item: katalog.Setting, value, cast):
    """Başlangıç değeri sınırların içinde (ör. dosyada kayıtlı eski bir değer)."""
    value = cast(value)
    if item.low is not None:
        value = max(value, cast(item.low))
    if item.high is not None:
        value = min(value, cast(item.high))
    return value


def image_of(paths) -> np.ndarray | None:
    """Girdiler arasındaki ilk görüntü (noktalar bu görüntüde seçilir)."""
    for path in paths:
        if path is not None and str(path).lower().endswith(katalog.IMAGE_EXTENSIONS):
            image = cv2.imread(str(path))
            if image is not None:
                return image
    return None


def _points(item: katalog.Setting, default, image: np.ndarray | None, key: str, busy: bool,
            errors: list[str]) -> list[tuple[int, int]]:
    """4 nokta (görüntü sınırları içinde) ve anlık önizleme."""
    height, width = image.shape[:2] if image is not None else (720, 1280)
    st.markdown(f"<span style='font-size:0.875rem'>{item.label} ({width}x{height} görüntüde)</span>",
                unsafe_allow_html=True, help=item.help or None)
    points = []
    for i, label in enumerate(CORNERS):
        name, x_col, y_col = st.columns([1.2, 1, 1], vertical_alignment="center")
        name.markdown(label)
        x = x_col.number_input("x", min_value=0, max_value=width, value=min(max(int(default[i][0]), 0), width),
                               step=1, key=f"{key}_p{i}_x", label_visibility="collapsed", disabled=busy)
        y = y_col.number_input("y", min_value=0, max_value=height, value=min(max(int(default[i][1]), 0), height),
                               step=1, key=f"{key}_p{i}_y", label_visibility="collapsed", disabled=busy)
        points.append((int(x), int(y)))
    persp = katalog.load_module(katalog.SCRIPTS["1.2-persp"])
    src = persp.to_points(points, item.label)
    try:
        persp.validate_quad(src, item.label, (width, height))
    except ValueError as exc:
        errors.append(str(exc).splitlines()[0])
        return points
    if image is not None:
        dst = persp.points_from_ratios(persp.DEFAULT_DST_RATIOS, (width, height))
        matrix, _ = persp.compute_transform(src, dst)
        warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
        st.markdown("**Anlık önizleme**")
        sonuc.image_pair(persp.draw_quad(image, src, "Orijinal + kaynak noktalar"),
                         persp.draw_quad(warped, dst, "Kus bakisi + hedef dikdortgen"),
                         "Yamuk (kaynak noktalar)", "Kuş bakışı (şeritler paralel mi?)")
    return points


def setting_widgets(op: katalog.Operation, defaults: dict, key: str, busy: bool = False,
                    image: np.ndarray | None = None) -> tuple[dict, list[str]]:
    """İşlemin ayar bileşenleri; dönüş: (değerler, hatalar). defaults: katalog.setting_defaults(...)."""
    values, errors = {}, []
    for item in op.settings:
        default = defaults[item.dest]
        widget = f"{key}_{item.dest}"
        help_text = item.help or None
        if item.kind == "int":
            value = st.number_input(item.label, value=_clamp(item, default, int), step=1, key=widget, disabled=busy,
                                    help=help_text, **_bounds(item, int))
        elif item.kind == "optional_int":
            value = st.number_input(item.label, min_value=int(item.low or 1),
                                    max_value=int(item.high or katalog.MAX_FRAMES_LIMIT), value=default, step=1,
                                    key=widget, placeholder="tümü", disabled=busy, help=help_text)
        elif item.kind == "float":
            value = st.number_input(item.label, value=_clamp(item, default, float), step=item.step or 0.01,
                                    format=_number_format(item.step), key=widget, disabled=busy, help=help_text,
                                    **_bounds(item, float))
        elif item.kind == "choice":
            value = st.segmented_control(item.label, list(item.choices), default=default, required=True, key=widget,
                                         disabled=busy, help=help_text,
                                         format_func=(item.names or {}).get if item.names else str)
        elif item.kind in ("flag", "noflag"):
            value = st.toggle(item.label, value=bool(default), key=widget, disabled=busy, help=help_text)
        elif item.kind in ("pair_int", "pair_float"):
            cast = int if item.kind == "pair_int" else float
            ranged = "aralığı" in item.label
            names = item.parts or (("en az", "en çok") if ranged else ("1.", "2."))
            low_col, high_col = st.columns(2)
            first = low_col.number_input(f"{item.label}: {names[0]}", value=_clamp(item, default[0], cast),
                                         key=f"{widget}_0", disabled=busy, help=help_text, **_bounds(item, cast))
            second = high_col.number_input(f"{item.label}: {names[1]}", value=_clamp(item, default[1], cast),
                                           key=f"{widget}_1", disabled=busy, **_bounds(item, cast))
            if ranged and first > second:
                errors.append(f"{item.label}: en az değer en çok değerden büyük olamaz ({first} > {second}).")
            value = (first, second)
        elif item.kind == "points":
            value = _points(item, default, image, widget, busy, errors)
        elif item.kind == "floats":
            text = st.text_input(item.label, value=" ".join(f"{v:g}" for v in default), key=widget, disabled=busy,
                                 help=(help_text or "") + " Boşlukla ayırarak yaz.")
            try:
                value = [float(v) for v in text.replace(",", " ").split()]
                if not value:
                    raise ValueError
            except ValueError:
                value = None
                errors.append(f"{item.label}: boşlukla ayrılmış en az bir sayı gerekir.")
        else:
            value = default
        values[item.dest] = value
    return values, errors


def readable(op: katalog.Operation | None, problem: str) -> str:
    """Modül mesajındaki bayrak adlarını ayarın arayüzdeki adıyla değiştirir (ör. --white-l-range -> «Beyaz …»)."""
    for item in sorted(op.settings if op else (), key=lambda s: len(s.flag), reverse=True):
        problem = problem.replace(item.flag, f"«{item.label}»")
    return problem


def counts(adjustable: int, fixed: int) -> str:
    """Ayarlar başlığının yanındaki sayılar; her sayfada aynı biçim (ör. "2 ayarlanabilir · 11 sabit")."""
    return f"{adjustable} ayarlanabilir · {fixed} sabit"


def fixed_settings(rows: list[katalog.FixedRow]) -> None:
    """"Sabit ayarlar (N)" açılır kutusu: ayar (soru işaretinde açıklaması), değer, kaynak; modül başlıklarıyla."""
    if not rows:
        return
    with st.expander(f"Sabit ayarlar ({len(rows)})"):
        head = st.columns(COLUMNS)
        head[0].markdown("**Ayar**")
        head[1].markdown("**Değer**")
        head[2].markdown("**Kaynak**", help="\n\n".join(katalog.SOURCE_HELP.values()))
        group = None
        for row in rows:
            if row.group != group:
                group = row.group
                if group:
                    st.markdown(f"<div class='ads-step'>{group}</div>", unsafe_allow_html=True)
            name, value, source = st.columns(COLUMNS, vertical_alignment="center")
            name.markdown(row.name, help=row.help)
            value.markdown(f"`{row.value}`")
            source.badge(row.source, color=katalog.SOURCE_COLORS[row.source])
