"""
Katalog — işlem tanımı ve kuralları: bir işlemin girdileri (Input), ayarları (Setting), çıktı görünümleri (View),
kilitleri (Lock) ve komutu. İşlemlerin listesi islem_listesi.py'dedir.

Kilit her zaman diskteki dosyaya bakılarak belirlenir; böylece işlem arayüzden de komut satırından da çalıştırılsa
doğru sonuç verir.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .moduller import module_defaults
from .yollar import PROJECT_ROOT, TEK_ISLEM_PAGE


SELECTED_OPERATION = "tek_islem_secili"   # Tek işlem sayfasında açılacak işlem (session_state anahtarı)


@dataclass
class Lock:
    """Kilit: önce çalıştırılacak işlem ya da eksik veri (arayüzde bilesenler/kilit.py gösterir)."""
    requirement: str            # önce çalıştırılacak gereksinim no (ör. "2.3") ya da "" (ör. eksik veri)
    name: str                   # işlemin adı (ör. "Sliding Windows")
    reason: str = ""            # neden kilitli (ör. eksik dosya)
    page: str | None = None     # götüren düğmenin sayfası (ör. "sayfalar/tek_islem.py")
    target: str | int | None = None   # o sayfada seçilecek işlem ya da adım
    target_key: str = SELECTED_OPERATION   # seçimin yazılacağı session_state anahtarı

    @property
    def label(self) -> str:
        return f"{self.requirement} {self.name}".strip()


@dataclass(frozen=True)
class Input:
    flag: str                       # komut satırı seçeneği; "" ise komuta eklenmez (komutu işlem kurar)
    label: str
    dest: str | None = None         # modülün parse_args alanı: varsayılan yol oradan okunur
    default: Path | None = None     # dest yoksa sabit varsayılan
    producer: str | None = None     # bu dosyayı üreten işlem; None: gerçek veri ya da hazır dosya
    real_data: bool = False         # kullanıcının gerçek verisi (yükleme seçeneği verilir)
    pattern: str | None = None      # klasörse içinde bulunması gereken dosyalar (boş klasör = yok)
    options: Callable | None = None # seçenek listesi veren fonksiyon (ör. modeller): yol yerine seçim kutusu


@dataclass(frozen=True)
class Setting:
    flag: str
    label: str
    dest: str
    kind: str                       # int, float, choice, flag, noflag, pair_int, pair_float, points, floats, optional_int
    choices: tuple = ()
    names: dict | None = None       # seçeneklerin ekranda görünen adları
    low: float | None = None
    high: float | None = None
    step: float | None = None
    help: str = ""
    parts: tuple = ()               # iki sayılı ayarda sayıların adları (ör. ("sütun", "satır"))


@dataclass(frozen=True)
class View:
    kind: str                       # metrics, npz, images, csv, file
    path: Path                      # dosya ya da klasör
    pattern: str = "*"              # klasörde gösterilecek dosyalar
    label: str = ""
    pairs: Path | None = None       # yan yana gösterilecek girdi (klasör ya da dosya)
    metrics: Callable | None = None
    chart: tuple | None = None      # csv için (x sütunu, [y sütunları])


@dataclass(frozen=True)
class Operation:
    key: str
    number: str
    name: str
    group: str
    purpose: str
    script: Path
    inputs: tuple = ()
    settings: tuple = ()
    fixed: tuple = ()               # her zaman eklenen argümanlar
    views: Callable = field(default=lambda inp, val: [])
    protected: Callable | None = None   # kurulum dosyaları klasörüne yazdıkları (onay + yedek)
    commands: Callable | None = None    # çok komutlu işlem: (girdiler, ayarlar) -> [(betik, argümanlar)]

    @property
    def label(self) -> str:
        return f"{self.number} {self.name}"


def input_default(op: Operation, item: Input) -> Path:
    if item.default is not None:
        return Path(item.default)
    value = getattr(module_defaults(op.script), item.dest)
    return Path(value[0] if isinstance(value, (list, tuple)) else value)


def setting_default(op: Operation, item: Setting):
    return getattr(module_defaults(op.script), item.dest)


def default_inputs(op: Operation) -> list[Path]:
    return [input_default(op, item) for item in op.inputs]


def default_values(op: Operation) -> dict:
    return {item.dest: setting_default(op, item) for item in op.settings}


# --------------------------------------------------------------------------- kilit ve durum
def path_ready(path: Path, pattern: str | None = None) -> bool:
    """Dosya var mı; klasörse içinde (desene uyan) dosya var mı?"""
    path = Path(path)
    if path.is_file():
        return True
    if path.is_dir():
        return any(p.is_file() for p in path.glob(pattern or "*"))
    return False


def locks(op: Operation, inputs: list[Path]):
    """Eksik girdiler için kilitler: önce çalıştırılacak işlem (numara + ad) ya da eksik gerçek veri."""
    from .islem_listesi import OPERATIONS
    result = []
    for item, path in zip(op.inputs, inputs):
        if path_ready(path, item.pattern):
            continue
        shown = _relative(path)
        if item.producer:
            producer = OPERATIONS[item.producer]
            result.append(Lock(producer.number, producer.name, f"{item.label} yok: `{shown}`",
                               TEK_ISLEM_PAGE, producer.key))
        else:
            kind = "Gerçek veri" if item.real_data else "Hazır dosya"
            result.append(Lock("", item.label, f"{kind} yok: `{shown}`"))
    return result


def view_is_dir(view: View) -> bool:
    """Görünüm bir klasör mü (görüntü listesi ya da desenli dosya listesi)? Klasör adlarında nokta olabilir
    (ör. "1.3 Piksel–metre ölçeklemesi"); bu yüzden uzantıya değil görünümün türüne bakılır."""
    return view.kind == "images" or (view.kind == "file" and view.pattern != "*")


def view_ready(view: View) -> bool:
    return path_ready(view.path, view.pattern) if view_is_dir(view) else Path(view.path).is_file()


def primary_ready(op: Operation, inputs: list[Path], values: dict) -> bool:
    """İşlemin asıl çıktısı (ilk değer kartı olmayan görünüm) diskte var mı?"""
    views = [v for v in op.views(inputs, values) if v.kind != "metrics"]
    return bool(views) and view_ready(views[0])


def status(op: Operation) -> str:
    """Listede gösterilen durum (varsayılan girdilerle): done (çıktısı var), locked ya da ready.
    Çıktısı olan işlem ✓ görünür; yeniden çalıştırmak için girdisi eksikse kilit kartın içinde gösterilir."""
    inputs = default_inputs(op)
    if primary_ready(op, inputs, default_values(op)):
        return "done"
    return "locked" if locks(op, inputs) else "ready"


def build_commands(op: Operation, inputs: list[Path], values: dict) -> list[tuple[Path, list]]:
    """Çalıştırılacak komut(lar): komut satırındaki gibi seçenek + değer."""
    if op.commands is not None:
        return op.commands(inputs, values)
    args: list = []
    for item, path in zip(op.inputs, inputs):
        if item.flag:
            args += [item.flag, path]
    for item in op.settings:
        args += setting_args(item, values.get(item.dest))
    return [(op.script, [*args, *op.fixed])]


def setting_args(item: Setting, value) -> list:
    if item.kind == "flag":
        return [item.flag] if value else []
    if item.kind == "noflag":                       # store_false seçeneği (ör. --no-pair-filter)
        return [] if value else [item.flag]
    if value is None or value == "":
        return []
    if item.kind in ("pair_int", "pair_float"):
        return [item.flag, *value]
    if item.kind == "points":
        return [item.flag, *[int(v) for point in value for v in point]]
    if item.kind == "floats":
        return [item.flag, *value]
    return [item.flag, value]


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
