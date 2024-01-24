FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y libboost-iostreams1.74-dev

RUN pip install scikit-learn xgboost matplotlib sc2reader wandb

RUN pip install git+https://github.com/5had3z/konductor git+https://github.com/5had3z/sc2-serializer

WORKDIR /app
COPY . .
