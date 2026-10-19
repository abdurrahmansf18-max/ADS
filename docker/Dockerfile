FROM python:3.12.10-slim-bookworm

# Host kullanicisiyla ayni UID/GID: /workspace'te olusan dosyalar root'a ait olmaz
ARG USER_NAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

# NVIDIA: konteyner ekran kartini hesaplama icin kullanir (surucu host'tan gelir).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN groupadd --gid ${USER_GID} ${USER_NAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USER_NAME} \
    && python -m venv ${VIRTUAL_ENV} \
    && chown -R ${USER_UID}:${USER_GID} ${VIRTUAL_ENV}

USER ${USER_NAME}

COPY --chown=${USER_UID}:${USER_GID} requirements.txt /tmp/requirements.txt
# ultralytics (5.1) opencv-python'i da kurar. opencv-python ile headless ayni cv2
# klasorunu paylasir ve birlikte kurulmamalidir: opencv-python kaldirilir, headless
# (requirements.txt'teki surum) bagimliliksiz yeniden kurulur.
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-deps --force-reinstall "$(grep -E '^opencv-python-headless==' /tmp/requirements.txt)" \
    && rm /tmp/requirements.txt

WORKDIR /workspace

# Arayuz (streamlit, frontend/): konteyner disindan erisilsin diye 0.0.0.0'da,
# tarayici acmadan ve kullanim verisi gondermeden calisir. Konteynerle birlikte kendiliginden baslar.
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 8888: Jupyter Lab, 8501: arayuz (streamlit)
EXPOSE 8888 8501

# Baslangic: Jupyter Lab ve arayuz birlikte (servisler.sh). En sonda: degisince paketler yeniden kurulmaz.
COPY --chmod=0755 servisler.sh /usr/local/bin/ads-servisler
CMD ["ads-servisler"]
