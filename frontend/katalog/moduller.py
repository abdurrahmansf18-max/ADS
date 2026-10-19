"""
Katalog — gereksinim modüllerine erişim: modülü dosyasından yükler, komut satırı varsayılanlarını ve ArgumentParser'ını
okur, arayüzün kuracağı komutu modülün kendi kurallarıyla (parse_args, validate_args) çalıştırmadan dener.

Kurallar ve varsayılanlar tek yerde, modülde kalır; arayüz onları buradan okur. Modül dosyası değişirse (değişme
zamanına bakılarak) yeniden yüklenir.
"""

import argparse
import contextlib
import importlib.util
import inspect
import io
import sys
import threading
from functools import lru_cache
from pathlib import Path

from .yollar import BACKEND_DIR, SCRIPTS


@lru_cache(maxsize=None)
def _module(script: str, mtime: float):
    path = Path(script)
    name = "ads_ui_" + "".join(ch if ch.isalnum() else "_" for ch in str(path.relative_to(BACKEND_DIR)))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


def load_module(script: Path):
    """Gereksinim modülünün kendisi (ör. kurulum sayfasında 1.2'nin önizleme fonksiyonları için)."""
    return _module(str(script), Path(script).stat().st_mtime)


@lru_cache(maxsize=None)
def _defaults(script: str, mtime: float):
    path = Path(script)
    module = _module(script, mtime)
    argv, sys.argv = sys.argv, [str(path)]
    try:
        return module.parse_args([]) if inspect.signature(module.parse_args).parameters else module.parse_args()
    finally:
        sys.argv = argv


def module_defaults(script: Path):
    """Modülün komut satırı varsayılanları (kendi AYARLAR bölümünden); modül değişirse yeniden okunur."""
    return _defaults(str(script), Path(script).stat().st_mtime)


def check_commands(commands: list[tuple[Path, list]]) -> list[str]:
    """Komutların ayarlarını modülün kendi kurallarıyla dener (çalıştırmadan): parse_args ve varsa
    validate_args. Dönüş: modülün reddedeceği ayarların mesajları (boşsa çalıştırılabilir). Kurallar tek yerde,
    modülde kalır; arayüz komut satırının vereceği hatayı önceden gösterip Çalıştır'ı kilitler."""
    problems = []
    for script, args in commands:
        module = load_module(script)
        argv = [str(a) for a in args]
        saved, sys.argv = sys.argv, [str(script), *argv]
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                parsed = module.parse_args(argv) if inspect.signature(module.parse_args).parameters \
                    else module.parse_args()
            if hasattr(module, "validate_args"):
                module.validate_args(parsed)
        except SystemExit:                          # argparse: yanlış tür ya da eksik değer
            lines = [line for line in stderr.getvalue().splitlines() if line.strip()]
            problems.append(lines[-1].split("error:", 1)[-1].strip() if lines else "Ayar okunamadı.")
        except Exception as exc:                    # modülün kendi kontrolü (InputError, ValueError ...)
            problems.append(str(exc).splitlines()[0] if str(exc) else type(exc).__name__)
        finally:
            sys.argv = saved
    return list(dict.fromkeys(problems))


class _Captured(Exception):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser


_CAPTURE_LOCK = threading.Lock()


def _call_parse_args(module, argv: list[str]):
    """Modülün parse_args'ını komut satırındaki gibi çağırır (argüman almayanlar sys.argv'den okur)."""
    saved, sys.argv = sys.argv, [str(getattr(module, "__file__", "modul")), *argv]
    try:
        if inspect.signature(module.parse_args).parameters:
            return module.parse_args(argv)
        return module.parse_args()
    finally:
        sys.argv = saved


def parser_of(module) -> argparse.ArgumentParser | None:
    """Modülün kendi ArgumentParser'ı: parse_args çağrılır, ayrıştırma anında parser yakalanır. Yalnız bu iş
    parçacığındaki çağrı yakalanır; aynı anda başka bir yerde çalışan parse_args etkilenmez."""
    original = argparse.ArgumentParser.parse_args
    owner = threading.get_ident()

    def capture(self, args=None, namespace=None):
        if threading.get_ident() == owner:
            raise _Captured(self)
        return original(self, args, namespace)

    with _CAPTURE_LOCK:
        argparse.ArgumentParser.parse_args = capture
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _call_parse_args(module, [])
        except _Captured as caught:
            return caught.parser
        except Exception:                       # parser kurulamadı: sabit ayarlar gösterilmez
            return None
        finally:
            argparse.ArgumentParser.parse_args = original
    return None


@lru_cache(maxsize=None)
def _parser(script: str, mtime: float):
    return parser_of(_module(script, mtime))


def module_parser(script: Path) -> argparse.ArgumentParser | None:
    return _parser(str(script), Path(script).stat().st_mtime)


def parse_command(module, argv) -> argparse.Namespace | None:
    """Komutun ayarlarını modülün kendi parse_args'ıyla çözer (çalıştırmadan); çözülemezse None."""
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return _call_parse_args(module, [str(a) for a in argv])
    except (SystemExit, Exception):
        return None


def script_key(script: Path) -> str:
    return next((key for key, path in SCRIPTS.items() if Path(path) == Path(script)), Path(script).stem)
