# Minimal Dockerfile for AQD dev/testing
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    libreadline-dev zlib1g-dev libssl-dev \
    libxml2-dev libxslt-dev libicu-dev \
    libnlohmann-json3-dev \
    python3 python3-pip python3-venv \
    wget curl ca-certificates pkg-config \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/aqd
COPY . .

# Build shared LightGBM lib and trainers by default
RUN make lgbm-lib && make gnn lightgbm -j$(nproc)

# Expose helpful defaults
ENV AQD_PG_HOST=localhost \
    AQD_PG_PORT=5432 \
    AQD_PG_USER=postgres

CMD ["/bin/bash"]

