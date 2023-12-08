FROM ubuntu:22.04

RUN apt-get -y update && \
    apt-get install -y python3-pip

RUN apt-get install -y git

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-13 g++-13


# Add other build deps
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake ninja-build git libboost-iostreams1.74-dev

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev

RUN pip install numpy torch torchvision torchaudio && \
    pip install git+https://github.com/5had3z/pybind11-stubgen.git black konductor

RUN cd / && \
    git clone https://github.com/5had3z/sc2-serializer.git && \
    cd sc2-serializer && \
    git submodule update --init --recursive && \
    CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
    cmake -B build -G Ninja -DSC2_PY_READER=OFF -DSC2_TESTS=OFF -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel --config Release

RUN apt-get purge -y python3-setuptools && \
    apt-get -y install curl && \
    curl -O https://bootstrap.pypa.io/get-pip.py  && \
    python3 get-pip.py && \
    rm get-pip.py && \
    python3 -m pip install --upgrade pip

RUN echo ""

RUN cd /sc2-serializer && \ 
    python3 -m pip install .

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y sudo

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1001 ubuntu && \
    echo 'ubuntu:ubuntu' | chpasswd

USER ubuntu



WORKDIR /app
COPY . .