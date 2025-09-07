#!/bin/bash
set -e

echo "=== Installing PostgreSQL with pg_duckdb support in current directory ==="

# Set up directories in current working directory
CURRENT_DIR="$(pwd)"
PG_BUILD_DIR="${CURRENT_DIR}/postgresql_build"
PG_INSTALL_DIR="${CURRENT_DIR}/postgresql_local"
PG_DATA_DIR="${CURRENT_DIR}/postgresql_data"

echo "Current directory: ${CURRENT_DIR}"
echo "Build directory: ${PG_BUILD_DIR}"
echo "Install directory: ${PG_INSTALL_DIR}"
echo "Data directory: ${PG_DATA_DIR}"

# Create directories
mkdir -p "${PG_BUILD_DIR}"
mkdir -p "${PG_INSTALL_DIR}"
mkdir -p "${PG_DATA_DIR}"

# Download PostgreSQL 16.4 if not already present
if [ ! -f "${PG_BUILD_DIR}/postgresql-16.4.tar.gz" ]; then
    echo "Downloading PostgreSQL 16.4..."
    cd "${PG_BUILD_DIR}"
    wget https://ftp.postgresql.org/pub/source/v16.4/postgresql-16.4.tar.gz
fi

# Extract if not already extracted
if [ ! -d "${PG_BUILD_DIR}/postgresql-16.4" ]; then
    echo "Extracting PostgreSQL..."
    cd "${PG_BUILD_DIR}"
    tar -xzf postgresql-16.4.tar.gz
fi

# Configure and build PostgreSQL
echo "Configuring PostgreSQL build..."
cd "${PG_BUILD_DIR}/postgresql-16.4"

./configure \
    --prefix="${PG_INSTALL_DIR}" \
    --with-openssl \
    --without-icu \
    --enable-debug \
    --enable-cassert

echo "Compiling PostgreSQL..."
make -j$(nproc)

echo "Installing PostgreSQL..."
make install

echo "Initializing database cluster..."
export PATH="${PG_INSTALL_DIR}/bin:$PATH"
cd "${CURRENT_DIR}"

# Initialize database
"${PG_INSTALL_DIR}/bin/initdb" -D "${PG_DATA_DIR}" --auth=trust

# Configure PostgreSQL
echo "Configuring PostgreSQL..."
cat >> "${PG_DATA_DIR}/postgresql.conf" << EOF

# Custom configuration for AQD development
port = 5433
max_connections = 100
shared_buffers = 128MB
dynamic_shared_memory_type = posix
log_destination = 'stderr'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'all'
log_min_duration_statement = 0
EOF

# Update pg_hba.conf for local connections
echo "host all all 127.0.0.1/32 trust" >> "${PG_DATA_DIR}/pg_hba.conf"
echo "local all all trust" >> "${PG_DATA_DIR}/pg_hba.conf"

echo "=== PostgreSQL Installation Complete ==="
echo "Install directory: ${PG_INSTALL_DIR}"
echo "Data directory: ${PG_DATA_DIR}"
echo "Port: 5433"
echo ""
echo "To start PostgreSQL:"
echo "  export PATH=${PG_INSTALL_DIR}/bin:\$PATH"
echo "  pg_ctl -D ${PG_DATA_DIR} -l ${PG_DATA_DIR}/logfile start"
echo ""
echo "To connect:"
echo "  psql -h localhost -p 5433 -U $(whoami) postgres"