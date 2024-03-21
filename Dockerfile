# Download and compile zlib-ng
FROM ubuntu:22.04 AS zlib-ng-builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget tar build-essential ninja-build cmake

WORKDIR /opt/zlib-ng
RUN wget https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.1.5.tar.gz && \
    tar -xf 2.1.5.tar.gz && \
    rm 2.1.5.tar.gz &&\
    cmake -B build -S zlib-ng-2.1.5 -G Ninja -DZLIB_COMPAT=ON -DZLIB_ENABLE_TESTS=OFF -DWITH_NATIVE_INSTRUCTIONS=ON && \
    cmake --build build --parallel

FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y libboost-iostreams1.74-dev

COPY --from=zlib-ng-builder /opt/zlib-ng/build/libz.so.1.3.0.zlib-ng /opt/zlib-ng/libz.so.1.3.0.zlib-ng
ENV LD_PRELOAD=/opt/zlib-ng/libz.so.1.3.0.zlib-ng

RUN pip install scikit-learn xgboost matplotlib sc2reader wandb

RUN pip install git+https://github.com/5had3z/sc2-serializer@b09b18f

WORKDIR /app

RUN mkdir database_tools && cd database_tools && \
    wget https://raw.githubusercontent.com/5had3z/sc2-serializer/main/scripts/replay_sql.py

RUN pip install git+https://github.com/5had3z/konductor@f2bbb63

ARG COMMIT
RUN [ ! -z "${COMMIT}" ]
ENV COMMIT_SHA=${COMMIT}

COPY . .
