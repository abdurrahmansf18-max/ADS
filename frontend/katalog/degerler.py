"""
Katalog — değer kartları: bir çıktı dosyasından Sonuç bölümündeki kartların değerleri (etiket, değer, açıklama).

Her fonksiyon bir dosya alır ve [(etiket, değer, açıklama), ...] döndürür; işlemler View("metrics", ...) ile kullanır.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .kurulumlar import setup_files
from .yollar import DEFAULT_SETUP


def _npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def calibration_metrics(path: Path) -> list[tuple]:
    d = _npz(path)
    k, dist = d["camera_matrix"], d["dist_coeffs"].ravel()
    return [("RMS hatası", f"{float(d['rms']):.3f} px", "Köşelerin yeniden yansıtma hatası; 1 pikselden küçük olmalı "
             f"(odak uzaklığı fx {k[0, 0]:.0f} px)."),
            ("Lens modeli", "Normal" if dist.size <= 5 else "Geniş açılı",
             f"{dist.size} katsayı. İki model de denenir; geniş açılı hatayı en az yarıya indiriyorsa o seçilir."),
            ("Görüntü boyutu", "x".join(str(int(v)) for v in d["image_size"]),
             "Videolar da bu çözünürlükte olmalı.")]


def scale_metrics(path: Path) -> list[tuple]:
    d = _npz(path)
    return [("xm (yatay)", f"{float(d['xm_per_pix']):.6f} m/px", "Yanal sapma ve şerit genişliği için."),
            ("ym (dikey)", f"{float(d['ym_per_pix']):.6f} m/px", "Eğrilik ve engel mesafesi için."),
            ("Şerit genişliği", f"{float(d['lane_width_px']):.1f} px = {float(d['lane_width_m']):.2f} m"),
            ("Kesik çizgi boyu", f"{float(d['dash_length_px']):.1f} px = {float(d['dash_length_m']):.2f} m")]


def setup_lane_width(name: str = DEFAULT_SETUP) -> float | None:
    """Kurulumda (1.3) girilen gerçek şerit genişliği (m); ölçek dosyası yoksa None."""
    try:
        with np.load(setup_files(name)["olcek"], allow_pickle=False) as data:
            return float(data["lane_width_m"])
    except (OSError, KeyError, ValueError):
        return None


def offset_metrics(path: Path) -> list[tuple]:
    table = pd.read_csv(path, encoding="utf-8-sig")
    values = pd.to_numeric(table["sapma_m"], errors="coerce").dropna()
    widths = pd.to_numeric(table["serit_genisligi_m"], errors="coerce").dropna()
    if values.empty:
        return [("Yanal sapma", "—", "Hiçbir karede şerit bulunamadı.")]
    first = table.dropna(subset=["sapma_m"]).iloc[0]
    return [("Yanal sapma" + (" (ilk kare)" if len(values) > 1 else ""), f"{float(first['sapma_m']):+.3f} m",
             f"Yön: {first['yon']} (+ araç merkezin sağında, − solunda)"),
            ("Ortalama sapma", f"{values.mean():+.3f} m", f"{len(values)} kare"),
            ("Şerit genişliği", f"{widths.mean():.2f} m" if not widths.empty else "—",
             f"Kurulumdaki gerçek genişlik {setup_lane_width():.2f} m." if setup_lane_width() else None),
            ("Şeritten çıkma uyarısı", f"{int((table['cikis_uyarisi'] == 'evet').sum())} kare")]


def coefficient_metrics(path: Path) -> list[tuple]:
    d = _npz(path)
    return [("Viraj sınıfı", str(d["category"][0]), "düz / hafif / sert (belirsiz: şeritler zıt bükülüyor)"),
            ("Yön", str(d["direction"][0])),
            ("A (metre cinsinden)", f"{float(d['mean_a_m'][0]):+.2e} 1/m", "≈ 0 düz; |A| büyüdükçe viraj sertleşir.")]


def curvature_metrics(path: Path) -> list[tuple]:
    d = _npz(path)
    radius = float(d["radius_m"][0])
    return [("Eğrilik yarıçapı R", f"{radius:.0f} m" if np.isfinite(radius) else "düz yol", "Büyük R hafif viraj."),
            ("Yön", str(d["direction"][0])),
            ("Sol / sağ R", f"{float(d['left_radius_m'][0]):.0f} / {float(d['right_radius_m'][0]):.0f} m")]


def detection_metrics(path: Path) -> list[tuple]:
    d = _npz(path)
    names = pd.Series(d["class_name"]).value_counts()
    top = ", ".join(f"{n} {c}" for n, c in names.head(3).items()) or "—"
    return [("Kare", f"{len(d['frame'])}"), ("Tespit", f"{len(d['det_frame'])}"), ("En sık sınıflar", top)]


def relation_metrics(path: Path) -> list[tuple]:
    d = _npz(path)
    risks = pd.Series(d["frame_risk"]).value_counts()
    return [("Tespit", f"{len(d['status'])}"), ("Şeritteki tespit", f"{int((d['status'] == 'seritte').sum())}"),
            ("Risk (kare)", f"kırmızı {risks.get('kirmizi', 0)} · sarı {risks.get('sari', 0)} · "
                            f"yeşil {risks.get('yesil', 0)}", "Sınırlar 5.5'in ayarlarındaki kırmızı ve sarı mesafe.")]


def polynomial_metrics(path: Path) -> list[tuple]:
    d = _npz(path)
    left, right = np.atleast_2d(d["left_fit"]), np.atleast_2d(d["right_fit"])
    items = [("Kare", f"{len(left)}"),
             ("Sol şerit uydurulan", f"{int(np.isfinite(left[:, 0]).sum())} kare"),
             ("Sağ şerit uydurulan", f"{int(np.isfinite(right[:, 0]).sum())} kare")]
    if len(left) == 1 and np.isfinite(left[0, 0]):
        items.append(("Sol [A, B, C]", ", ".join(f"{v:.3g}" for v in left[0])))
    return items
