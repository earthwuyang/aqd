#!/bin/bash
# Install PostgreSQL from source WITHOUT ICU dependency

set -e

echo "=== Installing PostgreSQL from Source (No ICU) ==="

# Clean up previous attempt
rm -rf ~/postgresql_build

# Create directories
mkdir -p ~/postgresql_build
cd ~/postgresql_build

# Download PostgreSQL 16 source
echo "Downloading PostgreSQL 16..."
wget https://ftp.postgresql.org/pub/source/v16.4/postgresql-16.4.tar.gz
tar -xzf postgresql-16.4.tar.gz
cd postgresql-16.4

# Configure build WITHOUT ICU
echo "Configuring PostgreSQL build (without ICU)..."
./configure \
    --prefix=$HOME/postgresql \
    --enable-debug \
    --enable-cassert \
    --enable-depend \
    --with-openssl \
    --with-readline \
    --without-icu

# Compile (using multiple cores)
echo "Compiling PostgreSQL..."
make -j$(nproc)

# Install
echo "Installing PostgreSQL..."
make install

# Add PostgreSQL binaries to PATH
echo "export PATH=\$HOME/postgresql/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$HOME/postgresql/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
export PATH=$HOME/postgresql/bin:$PATH
export LD_LIBRARY_PATH=$HOME/postgresql/lib:$LD_LIBRARY_PATH

# Initialize database
echo "Initializing PostgreSQL database..."
mkdir -p ~/postgresql_data
$HOME/postgresql/bin/initdb -D ~/postgresql_data --encoding=UTF8 --locale=C

# Configure PostgreSQL
echo "Configuring PostgreSQL..."
echo "listen_addresses = '*'" >> ~/postgresql_data/postgresql.conf
echo "port = 5432" >> ~/postgresql_data/postgresql.conf
echo "max_connections = 100" >> ~/postgresql_data/postgresql.conf
echo "shared_buffers = 256MB" >> ~/postgresql_data/postgresql.conf

# Configure authentication
echo "host all all 0.0.0.0/0 trust" >> ~/postgresql_data/pg_hba.conf
echo "local all all trust" >> ~/postgresql_data/pg_hba.conf

# Start PostgreSQL server
echo "Starting PostgreSQL server..."
$HOME/postgresql/bin/pg_ctl -D ~/postgresql_data -l ~/postgresql_data/logfile start

# Wait for server to start
sleep 5

# Create AQD database and user
echo "Creating AQD database and user..."
$HOME/postgresql/bin/createdb aqd_test
$HOME/postgresql/bin/psql -d aqd_test -c "CREATE USER aqd_user WITH PASSWORD 'aqd_password' SUPERUSER;"

echo "✓ PostgreSQL installation complete!"
echo "  - Server running on port 5432"
echo "  - Database: aqd_test"
echo "  - User: aqd_user"
echo "  - Password: aqd_password"
echo "  - Data directory: ~/postgresql_data"

# Test connection
echo "Testing connection..."
$HOME/postgresql/bin/psql -d aqd_test -U aqd_user -c "SELECT version();" || echo "Connection test failed"

# Show server status
$HOME/postgresql/bin/pg_ctl -D ~/postgresql_data status

echo "=== PostgreSQL Setup Complete ==="
echo "To stop: $HOME/postgresql/bin/pg_ctl -D ~/postgresql_data stop"
echo "To restart: $HOME/postgresql/bin/pg_ctl -D ~/postgresql_data restart"