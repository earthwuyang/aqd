# Use official PostgreSQL 16 base image
FROM postgres:16

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DUCKDB_VERSION=1.1.3

# Install dependencies as root
USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libssl-dev \
    zlib1g-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    postgresql-server-dev-16 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install DuckDB library
RUN cd /tmp && \
    wget https://github.com/duckdb/duckdb/releases/download/v${DUCKDB_VERSION}/libduckdb-linux-amd64.zip && \
    unzip libduckdb-linux-amd64.zip && \
    cp libduckdb.so /usr/local/lib/ && \
    cp duckdb.h /usr/local/include/ && \
    ldconfig

# Clone and build pg_duckdb extension
RUN cd /tmp && \
    git clone https://github.com/duckdb/pg_duckdb.git && \
    cd pg_duckdb && \
    make && \
    make install

# Create workspace for AQD implementation
RUN mkdir -p /home/aqd_workspace && \
    chown -R postgres:postgres /home/aqd_workspace

# Copy AQD implementation files
COPY --chown=postgres:postgres . /home/aqd_workspace/

# Install Python dependencies
RUN cd /home/aqd_workspace && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    pip install -r requirements.txt

# Create initialization script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Initialize PostgreSQL if needed\n\
if [ ! -s "$PGDATA/PG_VERSION" ]; then\n\
    initdb\n\
fi\n\
\n\
# Start PostgreSQL\n\
pg_ctl start -D "$PGDATA" -o "-c listen_addresses=*"\n\
\n\
# Create test database and install extension\n\
createdb aqd_test || true\n\
psql -d aqd_test -c "CREATE EXTENSION IF NOT EXISTS pg_duckdb;" || true\n\
\n\
echo "PostgreSQL with pg_duckdb extension is ready!"\n\
\n\
cd /home/aqd_workspace\n\
. venv/bin/activate\n\
\n\
exec "$@"' > /home/aqd_workspace/init_aqd.sh && \
    chmod +x /home/aqd_workspace/init_aqd.sh && \
    chown postgres:postgres /home/aqd_workspace/init_aqd.sh

USER postgres
WORKDIR /home/aqd_workspace

EXPOSE 5432

CMD ["./init_aqd.sh", "bash"]