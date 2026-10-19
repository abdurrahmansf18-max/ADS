"""
Arayüz — sonuç ve çıktı gösterimi.

Sonuç bölümünde önce önemli olan görünür: değer kartları ve görüntüler (çok karede kaydırıcı). Teknik
ayrıntılar (kurulum dosyasının içi, kare kare tablolar) açılır kutudadır. Dosyaların nereye kaydedildiği
bölüm başlığının yanındaki soru işaretine yazılır; video ve tablolar için indirme düğmeleri tek satırdadır.
Başarı İlerleme bölümünde tek satırla yazar; burada yalnız uyarı (sarı), hata (kırmızı, modülün çözüm
önerisiyle) ve durduruldu (gri) kutusu çıkar.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import katalog
from servisler.calistirici import message_blocks

MIME = {".mp4": "video/mp4", ".csv": "text/csv", ".npz": "application/octet-stream", ".png": "image/png",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".mkv": "video/x-matroska"}
DOWNLOAD_LABELS = {".mp4": "Video", ".mkv": "Video", ".csv": "Tablo (CSV)"}
VIEW_ORDER = {"metrics": 0, "images": 1, "csv": 2, "file": 3, "npz": 4}     # önemli olan önce, dosya içeriği en sonda


# --------------------------------------------------------------------------- durum kutuları
def _blocks_markdown(blocks: list[list[str]]) -> str:
    """HATA / UYARI blokları: ilk satır mesaj, girintili satırlar (ör. "-> ...") çözüm önerisi."""
    parts = []
    for block in blocks:
        parts.append(f"**{block[0]}**")
        parts.extend(f"- {line.lstrip('-> ').strip()}" for line in block[1:])
    return "\n\n".join(parts) if len(blocks) > 1 else "\n".join(parts)


def show_outcome(state: str, title: str = "", errors: list[list[str]] | None = None,
                 warnings: list[list[str]] | None = None) -> None:
    """Sonuç kutusu; renk yanında her zaman simge ve yazı. Başarıda (state "success") kutu çıkmaz."""
    if state == "warning":
        st.warning("! **Tamamlandı, uyarı var**\n\n" + _blocks_markdown(warnings or []))
    elif state == "error":
        body = _blocks_markdown(errors) if errors else "Program bir hata ile durdu; ayrıntı terminal çıktısında."
        st.error("✗ **Hata**\n\n" + body)
    elif state == "stopped":
        st.info("■ **Durduruldu** — işlem yarıda kesildi; o ana kadar üretilenler aşağıda.")


def show_job_outcome(job) -> None:
    """Bir işin (servisler.calistirici.Job) uyarı, hata ya da durduruldu kutusu, modülün HATA / UYARI mesajlarıyla."""
    lines = job.snapshot()
    show_outcome(job.outcome(), errors=message_blocks(lines, "HATA:"), warnings=message_blocks(lines, "UYARI:"))


# --------------------------------------------------------------------------- dosya bilgisi
def relative(path: Path) -> str:
    """Proje köküne göre yol (ör. outputs/arayuz/x.mp4)."""
    try:
        return str(Path(path).resolve().relative_to(katalog.PROJECT_ROOT))
    except ValueError:
        return str(path)


def _size(path: Path) -> str:
    size = path.stat().st_size
    return f"{size / 1e6:.1f} MB" if size >= 1e5 else f"{max(size / 1e3, 0.1):.1f} KB"


def saved_note(notes, path: Path, label: str = "") -> None:
    """Dosyanın (ya da klasörün) yeri bölümün soru işaretine: "Kaydedildi: <yol> (boyut)"."""
    if notes is None:
        return
    path = Path(path)
    size = f" ({_size(path)})" if path.is_file() else ""
    notes.add(f"**{label or 'Kaydedildi'}:** `{relative(path)}`{size}")


def downloads(files: list[tuple[Path, str]], key: str) -> None:
    """İndirme düğmeleri tek satırda: [(dosya, yazı), ...]; dosyanın yeri ve boyutu düğmenin ipucunda."""
    files = [(Path(path), label) for path, label in files if Path(path).is_file()]
    if not files:
        return
    with st.container(horizontal=True, gap="small"):
        for index, (path, label) in enumerate(files):
            with open(path, "rb") as handle:
                st.download_button(f"⬇ {label}", data=handle.read(), file_name=path.name,
                                   mime=MIME.get(path.suffix.lower(), "application/octet-stream"),
                                   key=f"{key}_{index}", on_click="ignore",
                                   help=f"{relative(path)} ({_size(path)})")


# --------------------------------------------------------------------------- görüntüleyiciler
def _image(column, image, caption: str) -> None:
    if isinstance(image, np.ndarray):
        column.image(image, caption=caption, channels="BGR" if image.ndim == 3 else "RGB", width="stretch")
    else:
        column.image(str(image), caption=caption, width="stretch")


def image_pair(left, right, left_caption: str = "Girdi", right_caption: str = "Çıktı") -> None:
    """İki görüntü yan yana (yol ya da BGR dizi)."""
    a, b = st.columns(2)
    _image(a, left, left_caption)
    _image(b, right, right_caption)


def metric_cards(items: list[tuple], columns: int = 4) -> None:
    """Değer kartları: [(etiket, değer, açıklama), ...]."""
    for start in range(0, len(items), columns):
        row = st.columns(columns)
        for column, item in zip(row, items[start:start + columns]):
            label, value, *rest = item
            column.metric(label, value, help=rest[0] if rest else None, border=True)


def _format_value(value: np.ndarray) -> str:
    if value.dtype.kind in "US":
        return ", ".join(map(str, value.ravel()[:6])) + (" …" if value.size > 6 else "")
    if value.size <= 16:
        return np.array2string(value, precision=4, suppress_small=True, separator=", ")
    return f"{value.size} değer (en küçük {value.min():.4g}, en büyük {value.max():.4g})"


def npz_table(path: Path) -> None:
    """Kurulum dosyasının (ya da herhangi bir .npz'nin) içindeki değerlerin tablosu."""
    try:
        with np.load(path, allow_pickle=False) as data:
            rows = [{"Anahtar": k, "Boyut": "x".join(map(str, data[k].shape)) or "tek değer",
                     "Tür": str(data[k].dtype), "Değer": _format_value(data[k])} for k in data.files]
    except (OSError, ValueError) as exc:
        st.error(f"✗ **Dosya okunamadı:** {relative(path)} ({exc})")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def video_player(path: Path) -> None:
    """Tarayıcıda oynatıcı (H.264 mp4)."""
    st.video(str(path))


# --------------------------------------------------------------------------- katalog görünümleri
def _pair_for(output: Path, source: Path | None) -> Path | None:
    """Çıktı görüntüsüne karşılık gelen girdi: adı çıktının adının başı olan görüntü (en uzun eşleşme)."""
    if source is None:
        return None
    source = Path(source)
    if source.is_file():
        return source if source.suffix.lower() in katalog.IMAGE_EXTENSIONS else None
    if not source.is_dir():
        return None
    candidates = [p for p in source.iterdir()
                  if p.suffix.lower() in katalog.IMAGE_EXTENSIONS and output.stem.startswith(p.stem)]
    return max(candidates, key=lambda p: len(p.stem)) if candidates else None


def _images_view(view, key: str, notes) -> None:
    files = sorted(p for p in Path(view.path).glob(view.pattern) if p.is_file())
    images = [p for p in files if p.suffix.lower() in katalog.IMAGE_EXTENSIONS]
    others = [p for p in files if p not in images]
    if images:
        label = view.label or "Kare"
        index = 0
        if len(images) > 1:
            index = st.select_slider(f"{label} ({len(images)})", options=list(range(len(images))),
                                     format_func=lambda i: images[i].name, key=f"{key}_kare")
        else:
            st.markdown(f"**{label}**")
        output = images[index]
        source = _pair_for(output, view.pairs)
        if source is not None:
            image_pair(source, output, f"Girdi: {source.name}", f"Çıktı: {output.name}")
        else:
            st.image(str(output), caption=output.name, width="stretch")
        saved_note(notes, view.path, label)
    downloads([(other, DOWNLOAD_LABELS[other.suffix.lower()]) for other in others
               if other.suffix.lower() in DOWNLOAD_LABELS], key=f"{key}_dosya")
    for other in others:                        # görüntü olmayan çıktılar (ör. video): yeri soru işaretinde
        saved_note(notes, other)


def show_view(view, key: str, notes=None) -> None:
    """Katalogdaki bir çıktı görünümünü türüne göre gösterir (katalog.View); dosyanın yeri notes'a."""
    try:
        if view.kind == "metrics":
            metric_cards(view.metrics(view.path), columns=4)
        elif view.kind == "npz":
            with st.expander(f"{view.label or 'Dosya'}: içindeki değerler"):
                npz_table(view.path)
            saved_note(notes, view.path, view.label)
        elif view.kind == "csv":
            frame = pd.read_csv(view.path, encoding="utf-8-sig")
            if view.label:
                st.markdown(f"**{view.label}**")
            if view.chart and len(frame) > 1:
                x, ys = view.chart
                st.line_chart(frame, x=x, y=ys, height=220)
            st.dataframe(frame, hide_index=True, width="stretch", height=min(38 + 35 * len(frame), 360))
            downloads([(view.path, "Tablo (CSV)")], key=f"{key}_indir")
            saved_note(notes, view.path, view.label)
        elif view.kind == "images":
            _images_view(view, key, notes)
        elif katalog.view_is_dir(view):
            files = sorted(p for p in Path(view.path).glob(view.pattern) if p.is_file())
            if notes is not None:
                notes.add(f"**{view.label or 'Dosyalar'}:** {len(files)} dosya, `{relative(view.path)}/{view.pattern}`")
        else:
            label = DOWNLOAD_LABELS.get(Path(view.path).suffix.lower())
            if label:
                downloads([(view.path, label)], key=f"{key}_indir")
            saved_note(notes, view.path, view.label)
    except Exception as exc:                    # bozuk ya da yarım dosya: sayfa çökmesin
        st.error(f"✗ **Çıktı gösterilemedi:** `{relative(view.path)}` ({type(exc).__name__}: {exc})")


def show_views(views, key: str, notes=None) -> None:
    """Hazır görünümler, önem sırasıyla: değer kartları, görüntüler, tablolar, dosya içerikleri (açılır)."""
    ready = [view for view in views if katalog.view_ready(view)]
    for index, view in sorted(enumerate(ready), key=lambda item: VIEW_ORDER.get(item[1].kind, 3)):
        show_view(view, key=f"{key}{index}", notes=notes)
