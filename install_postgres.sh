#!/bin/bash
# Install PostgreSQL from source for AQD data collection

set -e

echo "=== Installing PostgreSQL from Source ==="

# Create directories
mkdir -p ~/postgresql_build
cd ~/postgresql_build

# Download PostgreSQL 16 source
echo "Downloading PostgreSQL 16..."
wget https://ftp.postgresql.org/pub/source/v16.4/postgresql-16.4.tar.gz
tar -xzf postgresql-16.4.tar.gz
cd postgresql-16.4

# Configure build
echo "Configuring PostgreSQL build..."
./configure \
    --prefix=$HOME/postgresql \
    --enable-debug \
    --enable-cassert \
    --enable-depend \
    --with-openssl \
    --with-readline

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
$HOME/postgresql/bin/initdb -D ~/postgresql_data

# Start PostgreSQL server
echo "Starting PostgreSQL server..."
$HOME/postgresql/bin/pg_ctl -D ~/postgresql_data -l ~/postgresql_data/logfile start

# Wait for server to start
sleep 5

# Create AQD database and user
echo "Creating AQD database and user..."
$HOME/postgresql/bin/createdb aqd_test
$HOME/postgresql/bin/psql -d aqd_test -c "CREATE USER aqd_user WITH PASSWORD 'aqd_password';"
$HOME/postgresql/bin/psql -d aqd_test -c "GRANT ALL PRIVILEGES ON DATABASE aqd_test TO aqd_user;"
$HOME/postgresql/bin/psql -d aqd_test -c "ALTER USER aqd_user CREATEDB;"

echo "✓ PostgreSQL installation complete!"
echo "  - Server running on port 5432"
echo "  - Database: aqd_test"
echo "  - User: aqd_user"
echo "  - Password: aqd_password"
echo "  - Data directory: ~/postgresql_data"

# Test connection
echo "Testing connection..."
$HOME/postgresql/bin/psql -d aqd_test -U aqd_user -c "SELECT version();" || echo "Connection test failed"

echo "=== PostgreSQL Setup Complete ==="