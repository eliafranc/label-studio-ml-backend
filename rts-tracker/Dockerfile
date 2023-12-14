FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ "Europe/Berlin"
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git \
    apt-utils \
    build-essential \
    software-properties-common \
    wget \
    unzip \
    curl \
    git \
    cmake \
    ca-certificates\
    vim \
    autoconf \
    rsync \
    automake \
    mesa-utils \
    sudo \
    parallel \
    libopencv-dev \
    libboost-all-dev \
    libusb-1.0-0-dev \
    libcairo2-dev \
    libhdf5-dev \
    hdf5-tools \
    libglew-dev \
    libglfw3-dev \
    libcanberra-gtk-module \
    libboost-program-options-dev \
    libeigen3-dev \
    ffmpeg \
    python3-pip \
    python3-tk \
    python3-dev \
    ninja-build \
    libturbojpeg \
    x11-apps

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4

WORKDIR /app
COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt

COPY . ./

RUN git clone --recurse-submodules https://github.com/eliafranc/pytracking.git && \
    cd pytracking && \
    git submodule update --init --recursive && \
    chmod +x install.sh && \
    ./install.sh

CMD exec gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
