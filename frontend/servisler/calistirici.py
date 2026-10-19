"""
Arayüz — komut çalıştırıcı: bir gereksinim modülünü komut satırındaki gibi arka planda çalıştırır.

Komut ayrı bir süreç olarak başlar; çıktısı satır satır, canlı toplanır. Sonuç komut satırıyla birebir
aynıdır: aynı dosyalara yazılır, modülün HATA / UYARI mesajları olduğu gibi gelir. Çalışırken durdurulabilir.
Bir iş birden çok komuttan oluşabilir (ör. video şerit hazırlığı): komutlar sırayla çalışır, biri hata verirse
sonrakiler çalışmaz. Her komut çıktıda komut satırındaki gibi "$ ..." satırıyla başlar.
"""

import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

MAX_LINES = 2000          # bellekte tutulan en çok çıktı satırı (uzun videolarda)

# Modüllerin çıktısı beklemeden gelsin, kod klasörlerine __pycache__ yazılmasın, ultralytics ayarları
# yazılabilir bir klasöre gitsin.
CHILD_ENV = {"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "YOLO_CONFIG_DIR": "/tmp/Ultralytics"}


class Job:
    """Arka planda sırayla çalışan bir ya da birkaç komut."""

    def __init__(self, steps: list[tuple[list[str], Path]], title: str = ""):
        self.steps = [([str(a) for a in args], Path(cwd)) for args, cwd in steps]
        self.title = title
        self.lines: list[str] = []
        self.returncode: int | None = None
        self.started: float | None = None
        self.finished: float | None = None
        self.stopped = False
        self.current = 0                       # çalışan (ya da son çalışan) komutun sırası
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ gösterim
    @staticmethod
    def shown(args: list[str], cwd: Path) -> str:
        """Bir komutun terminalde görünen hâli (python ile başlar)."""
        visible = ["python", *args[1:]] if args and args[0] == sys.executable else args
        return f"$ cd {shlex.quote(str(cwd))}\n$ {shlex.join(visible)}"

    @property
    def command_text(self) -> str:
        """Bütün komutlar (terminal kutusunda her biri çalışmaya başlarken de yazılır)."""
        return "\n".join(self.shown(args, cwd) for args, cwd in self.steps)

    @property
    def running(self) -> bool:
        return self.started is not None and self.finished is None

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self.lines)

    # ------------------------------------------------------------------ çalıştırma
    def start(self) -> "Job":
        self.started = time.time()
        threading.Thread(target=self._run_steps, daemon=True).start()
        return self

    def _append(self, line: str) -> None:
        with self._lock:
            self.lines.append(line.rstrip("\n"))
            if len(self.lines) > MAX_LINES:
                del self.lines[: len(self.lines) - MAX_LINES]

    def _run_steps(self) -> None:
        env = {**os.environ, **CHILD_ENV}
        try:
            for index, (args, cwd) in enumerate(self.steps):
                if self.stopped:
                    break
                self.current = index
                if index:
                    self._append("")
                for line in self.shown(args, cwd).splitlines():
                    self._append(line)
                try:
                    self._process = subprocess.Popen(
                        args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, errors="replace",
                    )
                except OSError as exc:
                    self._append(f"HATA: Komut başlatılamadı: {exc}")
                    self.returncode = 1
                    break
                for line in self._process.stdout:
                    self._append(line)
                self.returncode = self._process.wait()
                if self.returncode != 0:
                    break              # hata: sonraki komutlar çalışmaz
        finally:
            self.finished = time.time()

    def stop(self) -> None:
        """Komutu durdurur (önce kibarca, 3 sn içinde kapanmazsa zorla)."""
        if self._process is None or not self.running:
            return
        self.stopped = True
        self._process.terminate()
        try:
            self._process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._process.kill()

    # ------------------------------------------------------------------ sonuç
    def outcome(self) -> str:
        """"running", "stopped", "error", "warning" ya da "success"."""
        if self.running:
            return "running"
        if self.stopped:
            return "stopped"
        if self.returncode != 0:
            return "error"
        return "warning" if message_blocks(self.snapshot(), "UYARI:") else "success"


def message_blocks(lines: list[str], prefix: str) -> list[list[str]]:
    """"HATA:" / "UYARI:" ile başlayan satırlar ve altlarındaki girintili açıklama satırları (blok blok)."""
    blocks, current = [], None
    for line in lines:
        if line.startswith(prefix):
            current = [line[len(prefix):].strip()]
            blocks.append(current)
        elif current is not None and line.startswith(("  ", "\t")) and line.strip():
            current.append(line.strip())
        else:
            current = None
    return blocks


def python_job(script: Path, args: list[str], title: str = "") -> Job:
    """Bir modül dosyasını kendi klasöründe çalıştıran iş (komut satırındaki gibi)."""
    return sequence_job([(script, args)], title)


def sequence_job(commands: list[tuple[Path, list]], title: str = "") -> Job:
    """Modül dosyalarını sırayla, her birini kendi klasöründe çalıştıran iş."""
    return Job([([sys.executable, Path(script).name, *[str(a) for a in args]], Path(script).parent)
                for script, args in commands], title)


def backup_files(paths: list[Path], backup_root: Path) -> Path | None:
    """Var olan dosyaların yedeğini backup_root/<tarih-saat>/ altına alır; yedek klasörünü döndürür. Dosya
    backup_root'un üst klasöründeyse oradaki yeri korunur (ör. yedek/<tarih>/kolay_video/kalibrasyon.npz)."""
    existing = [Path(p) for p in paths if Path(p).is_file()]
    if not existing:
        return None
    target = Path(backup_root) / time.strftime("%Y%m%d-%H%M%S")
    target.mkdir(parents=True, exist_ok=True)
    for path in existing:
        try:
            relative = path.resolve().relative_to(Path(backup_root).resolve().parent)
        except ValueError:
            relative = Path(path.name)
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / relative)
    return target
