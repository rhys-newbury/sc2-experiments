FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y libboost-iostreams1.74-dev

RUN mkdir /code && cd /code && \
    pip install --upgrade pip && \
    git clone https://github.com/5had3z/sc2-serializer.git --recursive && \
    cd sc2-serializer && \
    git submodule update --init --recursive && \
    git submodule update --remote && \
    pip install . && \
    pip install git+https://github.com/rhys-newbury/konductor wandb

RUN pip install scikit-learn xgboost matplotlib

WORKDIR /app
COPY . .
