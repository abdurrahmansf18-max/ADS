#!/usr/bin/env bash
# ADS konteynerini (ads-cv-lab) baslatir: NVIDIA GPU varsa GPU ile, yoksa CPU ile.
# Konteyner acilinca Jupyter Lab ve arayuz kendiliginden baslar (servisler.sh); elle bir sey baslatilmaz.
#
# Kullanim (proje kokunden):
#   bash docker/baslat.sh            # konteyneri baslat
#   bash docker/baslat.sh --build    # once imaji kur (ilk kez ya da docker/ icindeki dosyalar degisince)
#   ADS_GPU=0 bash docker/baslat.sh  # GPU olsa da CPU ile baslat
#   ADS_GPU=1 bash docker/baslat.sh  # GPU'yu zorla (bulunamazsa Docker hata verir)
#
# 1) Konteyner: host'ta calisan bir NVIDIA karti (nvidia-smi) ve Docker'da NVIDIA eklentisi
#    (NVIDIA Container Toolkit) varsa ekran karti konteynere verilir; yoksa konteyner CPU ile acilir.
# 2) Cihaz secimi: GPU varsa YOLO GPU'da ve CPU'da kisa bir olcumle denenir, hizli olan secilir
#    (zayif GPU + guclu CPU olan bir bilgisayarda CPU secilebilir). Olcum bir kez yapilir ve
#    konteyner yeniden kurulana kadar saklanir. Iki durumda da ayni imaj calisir.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILES=(-f "$DIR/docker-compose.yml")

has_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 &&
        docker info 2>/dev/null | grep -qiE "runtimes:.*nvidia|nvidia\.com/gpu"
}

case "${ADS_GPU:-auto}" in
    1) use_gpu=1 ;;
    0) use_gpu=0 ;;
    *) if has_gpu; then use_gpu=1; else use_gpu=0; fi ;;
esac

if [[ $use_gpu == 1 ]]; then
    FILES+=(-f "$DIR/docker-compose.gpu.yml")
    echo "GPU: NVIDIA ekran karti kullanilacak."
else
    echo "GPU: bulunamadi ya da kapatildi; konteyner CPU ile calisacak (YOLO daha yavas)."
fi

if [[ "${1:-}" == "--build" ]]; then
    docker compose "${FILES[@]}" build
fi
docker compose "${FILES[@]}" up -d

echo
echo "Cihaz secimi (YOLO icin GPU / CPU olcumu):"
docker exec -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
    -w "/workspace/backend/Carpisma_riski_ve_engel_farkindaligi/5.1 YOLO ile nesne tespiti" \
    ads-cv-lab python yolo_tespit.py --choose-device \
    || echo "  Olcum yapilamadi; ilk YOLO calistirmasinda yapilacak."

# Arayuz ve Jupyter konteynerle birlikte baslar; hazir olana kadar (en cok 60 sn) beklenir.
ready() {
    docker exec ads-cv-lab python -c "import sys, urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:$1', timeout=2)
except Exception:
    sys.exit(1)" >/dev/null 2>&1
}
for _ in $(seq 1 60); do
    ready 8501/_stcore/health && ready 8888/api && break
    sleep 1
done

echo
if ready 8501/_stcore/health; then
    echo "Arayuz  : http://127.0.0.1:${UI_PORT:-8501}"
else
    echo "Arayuz  : henuz acilmadi; nedeni icin: docker logs ads-cv-lab"
fi
token=$(docker exec ads-cv-lab jupyter server list 2>/dev/null | grep -o 'token=[0-9a-f]*' | head -1 || true)
echo "Jupyter : http://127.0.0.1:${JUPYTER_PORT:-8890}/lab${token:+?$token}"
echo "Konsol  : docker exec -it ads-cv-lab bash"
