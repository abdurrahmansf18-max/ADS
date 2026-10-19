#!/usr/bin/env bash
# Konteynerin baslangic komutu: konteyner acilinca Jupyter Lab ve arayuz (Streamlit) birlikte baslar.
#
#   Jupyter : http://127.0.0.1:8890  (token: docker exec ads-cv-lab jupyter server list)
#   Arayuz  : http://127.0.0.1:8501
#
# Backend ayrica baslatilmaz: gereksinim dosyalari arayuzden "Calistir" ile ya da komut satirindan
# cagrilinca calisan Python programlaridir.
# Arayuz kapanirsa (cokme ya da kod degisince yeniden baslatmak icin durdurulma) 3 sn sonra
# kendiliginden yeniden acilir. Konteyner durdurulunca ikisi de kapanir.
set -u

arayuz() {
    cd /workspace/frontend || { echo "[arayuz] /workspace/frontend yok; arayuz baslatilmadi"; return; }
    while true; do
        streamlit run app.py
        echo "[arayuz] kapandi; 3 sn sonra yeniden baslatiliyor"
        sleep 3
    done
}

arayuz &
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.root_dir=/workspace
