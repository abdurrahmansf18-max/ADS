"""
Katalog — sıfırlama ve silme: her düğmenin kalıcı olarak sileceği yollar ve güvenli silme.

outputs/ altındaki her dosya ya bir kurulumun, ya ana hattın ya da Tek işlem'indir; bölümler çakışmaz. Silme yalnız
izin verilen yerlerde yapılır (outputs/, kurulum_dosyalari/ içi, YOLO cihaz ölçümü; Sil'de o kurulumun toplanan
verileri); listede tek bir izinsiz yol varsa hiçbir şey silinmez. Boşalan klasörler de kaldırılır.
"""

import glob
import shutil
from pathlib import Path

from .kurulumlar import (setup_candidates, setup_dir, setup_folders, setup_for_video, setup_plan,
                         setup_videos, valid_setup_name, videos)
from .moduller import load_module
from .yollar import (BACKUP_DIR, INTEGRATION_OUTPUT_DIR, OUTPUTS_DIR, PHOTOS_DIR, SCRIPTS, SETUP_DIR, TEST_FRAMES_DIR,
                     UI_OUTPUT_DIR, VIDEOS_DIR)


# =========================================================================== sıfırlama
RESET_SECTIONS = {"kurulum": "Kurulum", "ana_hat": "Ana hat", "tek_islem": "Tek işlem", "hepsi": "Her şey"}


def device_cache() -> Path:
    """5.1'in YOLO cihaz ölçümü (GPU mu CPU mu; silinirse ilk çalıştırmada yeniden ölçülür)."""
    return load_module(SCRIPTS["5.1"]).DEVICE_CACHE


def reset_targets(section: str, name: str | None = None) -> dict[str, list[Path]]:
    """Bir bölüm sıfırlanınca kalıcı olarak silinecek yollar, başlıklara göre (yalnız diskte olanlar).

    kurulum: seçili kurulumun klasörü (kalibrasyon, perspektif, ölçek; boş da olsa klasörüyle, 1. adım yeniden
    açar), 5 adımın outputs/ altındaki bütün ara çıktıları ve yedekleri · ana_hat: işlenmiş videolar ve tablolar ·
    tek_islem: kurulum ve ana hat dışındaki bütün çıktılar · hepsi: kurulum_dosyalari/ altındaki bütün kurulum
    klasörleri ve yedekler, bütün çıktılar, YOLO cihaz ölçümü. Gerçek veriler (videolar, fotoğraflar, test kareleri,
    model) hiçbirinde yoktur; boşalan çıktı klasörleri de silinir (delete_targets). Kurulum, toplanan verileri
    durduğu için listede kalır."""
    groups: dict[str, list[Path]] = {}

    def add(label: str, paths) -> None:
        found = sorted({Path(p) for p in paths if Path(p).exists()})
        if found:
            groups[label] = found

    if section == "kurulum":
        plan = setup_plan(name)
        add("Kurulum dosyaları", [setup_dir(name)])
        add("Kurulum ara çıktıları",
            [p for step in range(5) for p in plan.outputs(step) if p.is_relative_to(OUTPUTS_DIR)])
        add("Kurulum yedekleri", BACKUP_DIR.glob(f"*/{glob.escape(name)}") if BACKUP_DIR.is_dir() else [])
    elif section == "ana_hat":
        add("Ana hat çıktıları", [UI_OUTPUT_DIR, INTEGRATION_OUTPUT_DIR])
    elif section == "tek_islem":
        keep = {p for setup in setup_candidates() for step in range(5) for p in setup_plan(setup).outputs(step)}
        skip = (UI_OUTPUT_DIR, INTEGRATION_OUTPUT_DIR)
        add("Tek işlem çıktıları", [p for p in OUTPUTS_DIR.rglob("*") if p.is_file() and p not in keep
                                    and not any(p.is_relative_to(folder) for folder in skip)]
            if OUTPUTS_DIR.is_dir() else [])
    elif section == "hepsi":
        add("Kurulum dosyaları", [folder for folder in SETUP_DIR.iterdir() if folder.is_dir() and folder != BACKUP_DIR]
            if SETUP_DIR.is_dir() else [])
        add("Kurulum yedekleri", [BACKUP_DIR])
        add("Bütün çıktılar", [OUTPUTS_DIR])
        add("YOLO cihaz ölçümü", [device_cache()])
    else:
        raise ValueError(f"Bilinmeyen bölüm: {section}")
    return groups


def _deletable(path: Path, setup: str | None = None) -> bool:
    """Silinebilecek yol mu: outputs/ (kendisi ya da içi), kurulum klasörünün içi, YOLO cihaz ölçümü. setup
    verilirse (yalnız kurulumu silerken) o kurulumun toplanan veri klasörleri ve içleri de."""
    path = path.resolve()
    outputs, setups = OUTPUTS_DIR.resolve(), SETUP_DIR.resolve()
    if path == outputs or path.is_relative_to(outputs) or (path != setups and path.is_relative_to(setups)):
        return True
    if setup is not None and valid_setup_name(setup):
        own = [(root / setup).resolve() for root in (PHOTOS_DIR, TEST_FRAMES_DIR, VIDEOS_DIR)]
        if any(path.is_relative_to(folder) for folder in own):
            return True
        if path.parent == VIDEOS_DIR.resolve() and setup_for_video(path) == setup:
            return True                                     # klasör dışındaki, adı bu kuruluma uyan video
    return path == device_cache().resolve()


def delete_setup_targets(name: str) -> dict[str, list[Path]]:
    """Kurulumu silmek: bu ada ait her şey. Sıfırlamanın sildikleri (kurulum klasörü, ara çıktılar, yedekler), ana
    hat sonuçları ve toplanan veriler (satranç fotoğrafları, test kareleri, videolar). Diğer kurulumlara dokunulmaz."""
    targets = reset_targets("kurulum", name)
    # Arayüzün sonuçları kurulumun klasöründe; komut satırının sonuçları (outputs/entegrasyon) yalnız video adıyla
    # yazıldığı için, aynı adlı videosu başka bir kurulumda da olanlar silinmez.
    others = {video.stem for video in videos() if setup_for_video(video) not in (name, None)}
    stems = [video.stem for video in setup_videos(name) if video.stem not in others]
    results = [UI_OUTPUT_DIR / name,
               *(INTEGRATION_OUTPUT_DIR / f"{stem}_islenmis{ext}" for stem in stems for ext in (".mp4", ".csv"))]
    collected = [*(setup_folders(name)[part] for part in ("fotograflar", "kareler", "videolar")),
                 *(video for video in setup_videos(name) if video.parent == VIDEOS_DIR)]
    for label, paths in (("Ana hat sonuçları", results),
                         ("Toplanan veriler (fotoğraflar, kareler, videolar)", collected)):
        found = sorted({Path(path) for path in paths if Path(path).exists()})
        if found:
            targets[label] = found
    return targets


def delete_targets(paths, setup: str | None = None) -> int:
    """Yolları kalıcı olarak siler (klasörleri içindekilerle) ve boşalan çıktı ve yedek klasörlerini kaldırır;
    dönüş: silinen dosya sayısı. setup: yalnız kurulumu silerken (o kurulumun toplanan verilerine izin verir).
    İzin verilen yerlerin dışında bir yol varsa hiçbir şey silinmez (ValueError)."""
    paths = [Path(p) for p in paths]
    outside = [str(p) for p in paths if not _deletable(p, setup)]
    if outside:
        raise ValueError("Bu yollar silinemez: " + ", ".join(outside))
    count = 0
    for path in paths:
        if path.is_dir():
            count += sum(1 for p in path.rglob("*") if p.is_file())
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
            count += 1
    for root in (OUTPUTS_DIR, BACKUP_DIR):            # boşalan klasörler (en içten); outputs/ boşaldıysa o da
        if root.is_dir():
            for folder in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                if not any(folder.iterdir()):
                    folder.rmdir()
            if not any(root.iterdir()):
                root.rmdir()
    return count
