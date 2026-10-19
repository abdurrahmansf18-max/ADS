"""
Katalog — kurulumlar ve veriler: kurulum adları, klasörleri, videolar, YOLO modelleri ve kurulumun 5 adımının planı.

Kurulum adı kamerayı (ve takılışını) temsil eder; satranç fotoğrafları, test kareleri, videolar ve kurulum dosyaları
bu adın klasörlerindedir. SetupPlan bir kurulumun 5 adımının girdi ve çıktı klasörlerini, bitip bitmediğini ve güncel
olup olmadığını bilir; setup_commands adımların komutlarını kurar (komut satırındakiyle aynı).
"""

import glob
import re
from dataclasses import dataclass
from pathlib import Path

from .yollar import *  # noqa: F401,F403  (klasörler ve sabitler)


def setup_dir(name: str) -> Path:
    """Kurulumun klasörü: data/uretilen_veri/kurulum_dosyalari/<ad>/."""
    return SETUP_DIR / name


def setup_files(name: str) -> dict[str, Path]:
    """Bir kurulumun (ör. "kolay_video") üç dosyası, kendi klasöründe: kalibrasyon, perspektif, ölçek."""
    return {part: setup_dir(name) / file for part, file in SETUP_FILE_NAMES.items()}


def setup_names() -> list[str]:
    """Klasöründe en az bir kurulum dosyası bulunan kurulumlar."""
    if not SETUP_DIR.is_dir():
        return []
    return sorted(folder.name for folder in SETUP_DIR.iterdir() if folder.is_dir() and folder != BACKUP_DIR
                  and any((folder / file).is_file() for file in SETUP_FILE_NAMES.values()))


def _folders(root: Path) -> set[str]:
    return {folder.name for folder in root.iterdir() if folder.is_dir()} if root.is_dir() else set()


def setup_candidates() -> list[str]:
    """Kurulum adları: dosyası olan kurulumlar ve aynı adla satranç fotoğrafı ya da test karesi klasörü olanlar
    (yeni kurulum, klasörleri açılınca listede görünür; sıfırlanan bir kurulum verisi durduğu için kalır)."""
    names = set(setup_names()) | _folders(PHOTOS_DIR) | _folders(TEST_FRAMES_DIR) | _folders(VIDEOS_DIR)
    return sorted(name for name in names if valid_setup_name(name))


def setup_folders(name: str) -> dict[str, Path]:
    """Kurulumun dört klasörü: toplanan veriler (fotoğraf, kare, video) ve üretilen kurulum dosyaları."""
    return {"fotograflar": PHOTOS_DIR / name, "kareler": TEST_FRAMES_DIR / name, "videolar": VIDEOS_DIR / name,
            "kurulum": setup_dir(name)}


def create_setup(name: str) -> None:
    """Yeni kurulumun dört klasörünü boş olarak açar; kurulum listede görünür."""
    if not valid_setup_name(name):
        raise ValueError(f"Geçersiz kurulum adı: {name}")
    for folder in setup_folders(name).values():
        folder.mkdir(parents=True, exist_ok=True)


def setup_status(name: str) -> dict[str, bool]:
    """Kurulum dosyalarının her biri var mı?"""
    return {part: path.is_file() for part, path in setup_files(name).items()}


def missing_setup(name: str) -> list[str]:
    """Eksik kurulum dosyalarını üreten gereksinim numaraları, kurulum sırasıyla (ör. ["1.2", "1.3"])."""
    status = setup_status(name)
    return [SETUP_PARTS[part][1] for part in SETUP_PARTS if not status[part]]


def setup_for_video(video: str | Path) -> str | None:
    """Videonun kurulumu: videolar/<ad>/ klasöründeyse o ad; değilse (ör. yüklenen video) adına uyan kurulum
    (kurulum adı videonun ön eki)."""
    video = Path(video)
    if video.parent.parent == VIDEOS_DIR and valid_setup_name(video.parent.name):
        return video.parent.name
    stem = video.stem
    matches = [name for name in setup_candidates() if stem == name or stem.startswith(name + "_")]
    return max(matches, key=len) if matches else None


def setup_videos(name: str) -> list[Path]:
    """Kurulumun videoları: videolar/<ad>/ içindekiler ve klasör dışında olup adı bu kuruluma uyanlar."""
    return [video for video in videos() if setup_for_video(video) == name]


MODEL_SIZES = {"n": "nano", "s": "küçük", "m": "orta", "l": "büyük", "x": "çok büyük"}


def yolo_models() -> list[Path]:
    """data/hazir_veri/yolo_modeli içindeki YOLO modelleri (.pt): varsayılan (yolo11n.pt) başta, sonra boyut sırasıyla
    (n, s, m, l, x)."""
    found = list(MODEL_DIR.glob("*.pt")) if MODEL_DIR.is_dir() else []
    order = list(MODEL_SIZES)
    return sorted(found, key=lambda p: (p != MODEL_PATH, p.stem[:-1],
                                        order.index(p.stem[-1]) if p.stem[-1:] in order else len(order), p.name))


def model_label(path: Path) -> str:
    """Seçim kutusundaki ad: "yolo11s.pt — küçük (s), 18.4 MB"."""
    path = Path(path)
    size = MODEL_SIZES.get(path.stem[-1:].lower())
    mb = f"{path.stat().st_size / 1e6:.1f} MB" if path.is_file() else "yok"
    return f"{path.name} — {size} ({path.stem[-1]}), {mb}" if size else f"{path.name} — {mb}"


def videos() -> list[Path]:
    """data/ortam_verisi/videolar içindeki videolar: kurulum klasörlerindekiler (videolar/<ad>/) ve klasör
    dışındakiler."""
    if not VIDEOS_DIR.is_dir():
        return []
    found = [*VIDEOS_DIR.iterdir(),
             *(p for folder in VIDEOS_DIR.iterdir() if folder.is_dir() for p in folder.iterdir())]
    return sorted((p for p in found if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
                  key=lambda p: (str(p.parent.relative_to(VIDEOS_DIR)), p.name))


def video_label(path: Path) -> str:
    """Seçim kutusundaki ad: "kurulum/video.mp4" (klasör dışındaysa yalnız dosya adı)."""
    path = Path(path)
    return str(path.relative_to(VIDEOS_DIR)) if path.is_relative_to(VIDEOS_DIR) else path.name


# Kurulum adı olamayanlar: yedek klasörü ve ara çıktı klasörü başka bir şeyin klasörüyle aynı olacak adlar
# (outputs/1.2/<ad>/ ile kus_bakisi*, outputs/1.1/<ad>_duzeltilmis/ ile video şerit hazırlığının video_duzeltilmis/'i).
RESERVED_SETUP_NAMES = {"yedek", "video", "kus_bakisi", "kus_bakisi_duz_yol", "kus_bakisi_maskeler"}


def valid_setup_name(name: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_]{2,40}", name or "")) and name not in RESERVED_SETUP_NAMES


SETUP_NAME_HELP = ("Küçük harf, rakam ve alt çizgi (ör. yeni_kamera); videoları videolar/<ad>/ klasörüne konur. "
                   "Olamayanlar: " + ", ".join(sorted(RESERVED_SETUP_NAMES)) + " (arayüzün kendi klasörleri).")




@dataclass(frozen=True)
class SetupPlan:
    """Bir kurulum adının (ör. "zor_video") 5 adımı için klasörler ve dosyalar."""
    name: str
    photos_dir: Path         # 1.1 satranç fotoğrafları (gerçek veri; bu kurulumun kamerasından)
    frames_dir: Path         # düz yol kareleri (gerçek veri)
    corrected_dir: Path      # 1.1 düzeltilmiş kareler
    verify_dir: Path         # 1.2 kontrol görüntüsü
    birdseye_dir: Path       # 1.2 kuş bakışı düz yol kareleri
    scale_dir: Path          # 1.3 karşılaştırma görüntüsü
    corners_dir: Path        # 1.1 köşe kontrol görüntüleri

    @property
    def calibration(self) -> Path:
        return setup_files(self.name)["kalibrasyon"]

    @property
    def perspective(self) -> Path:
        return setup_files(self.name)["perspektif"]

    @property
    def scale(self) -> Path:
        return setup_files(self.name)["olcek"]

    def straight_frames(self) -> list[Path]:
        return sorted(p for p in self.frames_dir.glob("duz_yol_*") if p.suffix.lower() in IMAGE_EXTENSIONS)

    def corrected_frames(self) -> list[Path]:
        return sorted(p for p in self.corrected_dir.glob("duz_yol_*_undistorted.*")
                      if p.suffix.lower() in IMAGE_EXTENSIONS)

    def birdseye_frames(self) -> list[Path]:
        return sorted(p for p in self.birdseye_dir.glob("duz_yol_*_warped.*") if p.suffix.lower() in IMAGE_EXTENSIONS)

    def photos(self, folder: Path | None = None) -> list[Path]:
        folder = Path(folder or self.photos_dir)
        return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if folder.is_dir() else []

    def done(self) -> list[bool]:
        """Her adımın çıktısı diskte var mı (1.1 kalibrasyon … 1.3 ölçek)?"""
        return [self.calibration.is_file(), bool(self.corrected_frames()), self.perspective.is_file(),
                bool(self.birdseye_frames()), self.scale.is_file()]

    def outputs(self, step: int) -> list[Path]:
        """Adımın diskteki bütün çıktıları (sıfırlama bunları siler): kurulum dosyası ve adımın outputs/ altına
        yazdığı her şey. 1: düzeltme, düz yol kareleriyle birlikte klasördeki deneme karelerini de düzeltir.
        2: perspektif hesabı kontrol görüntüsünün yanında kuş bakışı halini de yazar.

        Kendi klasörleri olan kurulumda (kolay_video dışındakiler) o klasörlerdeki her dosya kurulumundur.
        kolay_video, Tek işlem'in kurulum işlemleriyle aynı klasörleri kullanır (argümansız zincir); orada yalnız
        bu kurulumun karelerinin adını taşıyan dosyalar kurulumundur (Tek işlem'de başka karelerle üretilenler değil)."""
        def files(folder: Path, *patterns: str) -> list[Path]:
            return [p for pattern in patterns for p in folder.glob(pattern)] if folder.is_dir() else []
        frames = ([p.stem for p in self.frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
                  if self.frames_dir.is_dir() else [])
        straight = [p.stem for p in self.straight_frames()]
        if self.name != DEFAULT_SETUP:
            found = ([self.calibration, *files(self.corners_dir, "*")], files(self.corrected_dir, "*"),
                     [self.perspective, *files(self.verify_dir, "*")], files(self.birdseye_dir, "*"),
                     [self.scale, *files(self.scale_dir, "*")])[step]
        else:
            stems = [glob.escape(stem) for stem in (frames if step == 1 else straight)]
            found = ([self.calibration, *files(self.corners_dir, "*")],
                     [p for s in stems for p in files(self.corrected_dir, f"{s}_undistorted.*")],
                     [self.perspective, *(p for s in stems for p in files(
                         self.verify_dir, f"{s}_undistorted_verification.*", f"{s}_undistorted_warped.*"))],
                     [p for s in stems for p in files(self.birdseye_dir, f"{s}_undistorted_warped.*")],
                     [self.scale, *(p for s in stems for p in files(
                         self.scale_dir, f"{s}_undistorted_warped_karsilastirma.png"))])[step]
        return sorted({p for p in found if p.is_file()})

    def outdated(self) -> list[list[str]]:
        """Her adım için, çıktısından sonra değişen girdiler (boş liste: çıktı güncel).

        Bir adımın çıktısı kendinden önceki bir adımın çıktısından (ya da düz yol karelerinden) eskiyse eski
        girdilerle üretilmiştir; make gibi dosyaların değişme zamanına bakılır."""
        main = ([self.calibration], self.corrected_frames(), [self.perspective], self.birdseye_frames(), [self.scale])
        times = []
        for files in main:
            stamps = [p.stat().st_mtime for p in files if p.is_file()]
            times.append((min(stamps), max(stamps)) if stamps else None)
        frames = max((p.stat().st_mtime for p in self.straight_frames()), default=None)
        result = []
        for step, own in enumerate(times):
            newer = []
            if own is not None:
                if step >= 1 and frames is not None and frames > own[0]:
                    newer.append("düz yol kareleri")
                newer += [f"adım {k + 1} ({SETUP_STEPS[k]})" for k in range(step) if times[k] and times[k][1] > own[0]]
            result.append(newer)
        return result


def setup_outdated(name: str) -> list[str]:
    """Güncel olmayan kurulum dosyaları (ör. ["Piksel–metre ölçeği"]): ana hat çalışır ama sonuç eski girdilere
    göre olabilir."""
    outdated = setup_plan(name).outdated()
    return [SETUP_PARTS[part][0] for part, step in (("perspektif", 2), ("olcek", 4)) if outdated[step]]


def setup_plan(name: str) -> SetupPlan:
    """Kurulumun klasörleri. Girdiler ve kurulum dosyaları adın klasöründe (kalibrasyon_fotograflari/<ad>/,
    test_kareleri/<ad>/, kurulum_dosyalari/<ad>/). Ara çıktılar: kolay_video modüllerin varsayılan klasörlerini
    kullanır (argümansız zincirle aynı), diğer adlar outputs/ altında kendi klasörlerini."""
    o11 = OUTPUTS_DIR / "1.1 Kamera kalibrasyonu ve distorsiyon düzeltme"
    o12 = OUTPUTS_DIR / "1.2 Perspektif dönüşümü"
    o13 = OUTPUTS_DIR / "1.3 Piksel–metre ölçeklemesi"
    if name == DEFAULT_SETUP:
        corrected, verify, birdseye, scale_dir = (o11 / "duzeltilmis_kareler", o12, o12 / "kus_bakisi_duz_yol", o13)
    else:
        corrected, verify, birdseye, scale_dir = (o11 / f"{name}_duzeltilmis", o12 / name, o12 / name / "kus_bakisi",
                                                  o13 / name)
    return SetupPlan(name, PHOTOS_DIR / name, TEST_FRAMES_DIR / name, corrected, verify,
                     birdseye, scale_dir, o11 / "koseler" / name)


def setup_commands(plan: SetupPlan, step: int, values: dict) -> list[tuple[Path, list]]:
    """Kurulum adımının komut(lar)ı — komut satırındaki gibi (adım 0..4)."""
    if step == 0:
        args = ["--images", values["photos"], "--pattern", *values["pattern"], "--output", plan.calibration,
                "--corners-dir", plan.corners_dir]
        return [(SCRIPTS["1.1-kal"], args)]
    if step == 1:
        return [(SCRIPTS["1.1-duz"], ["--input", plan.frames_dir, "--calib", plan.calibration,
                                       "--output", plan.corrected_dir])]
    if step == 2:
        points = [int(v) for point in values["src"] for v in point]
        return [(SCRIPTS["1.2-persp"], ["--image", values["image"], "--src", *points, "--output", plan.perspective,
                                         "--verify-dir", plan.verify_dir])]
    if step == 3:
        return [(SCRIPTS["1.2-warp"], ["--transform", plan.perspective, "--input", frame, "--output", plan.birdseye_dir])
                for frame in plan.corrected_frames()]
    return [(SCRIPTS["1.3"], ["--input", plan.birdseye_dir, "--lane-width-m", values["lane_width_m"],
                              "--dash-length-m", values["dash_length_m"], "--out", plan.scale, "--compare",
                              "--out-dir", plan.scale_dir])]
