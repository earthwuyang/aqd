# Makefile for AQD (Adaptive Query Dispatcher) System
# Builds PostgreSQL with AQD extensions and supporting tools

.PHONY: all clean postgres lightgbm gnn data-collection test help setup env-info venv-check venv-deps

# Configuration
POSTGRES_SRC = postgres_src
BUILD_DIR = build
INSTALL_DIR = $(PWD)/install
DATA_DIR = $(PWD)/data
# Canonical Postgres data directory (align with README and scripts)
PGDATA_DIR = $(PWD)/pgdata

# PostgreSQL build configuration
PG_CONFIG_OPTIONS = --prefix=$(INSTALL_DIR) \
                   --enable-debug \
                   --enable-cassert \
                   --enable-depend \
                   --with-openssl

# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -O2 -g -Wall -Wextra
CXXFLAGS = -O2 -g -Wall -Wextra -std=c++17

# Library paths - using local dependencies
LIGHTGBM_INCLUDE = -I$(PWD)/include
LIGHTGBM_LIBS = $(PWD)/lib/lib_lightgbm.a -lgomp -lpthread
JSON_LIBS = 
OPENSSL_LIBS = -lssl -lcrypto

# Python environment
PYTHON = python3
VENV_DIR = venv

all: setup postgres lightgbm gnn

help:
	@echo "AQD (Adaptive Query Dispatcher) Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build everything (PostgreSQL + AQD + LightGBM trainer)"
	@echo "  setup         - Set up build environment and dependencies"
	@echo "  postgres      - Build PostgreSQL with AQD extensions"
	@echo "  lightgbm      - Build LightGBM trainer"
	@echo "  data-collection - Set up data collection pipeline"
	@echo "  test          - Run basic tests"
	@echo "  gnn           - Build GNN trainer"
	@echo "  gnn-labels    - Generate GNN labels CSV from execution data"
	@echo "  install-deps  - Install system dependencies"
	@echo "  clean         - Clean build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  make all                    # Full build"
	@echo "  make postgres               # PostgreSQL only"
	@echo "  make lightgbm               # LightGBM trainer only"
	@echo "  make test                   # Run tests"

setup:
	@echo "Setting up build environment..."
	$(MAKE) install-deps
	$(MAKE) $(BUILD_DIR)
	$(MAKE) $(VENV_DIR)
	$(MAKE) venv-deps
	@echo "\n✅ Setup complete. Activate the virtualenv before running Python commands:"
	@echo "   source $(VENV_DIR)/bin/activate"
	@echo "or use: $(VENV_DIR)/bin/python your_script.py"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(INSTALL_DIR)
	mkdir -p $(DATA_DIR)

$(VENV_DIR):
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip setuptools wheel

venv-deps: $(VENV_DIR) requirements.txt
	@echo "Installing Python packages from requirements.txt into venv..."
	$(VENV_DIR)/bin/pip install --upgrade -r requirements.txt
	@echo "Verifying virtualenv packages..."
	-$(VENV_DIR)/bin/python -c "import sys; print('Python:', sys.executable)"
	-$(VENV_DIR)/bin/python -c "import importlib; mods=['numpy','pandas','duckdb','lightgbm'];\
for m in mods:\
    \n    \
    print(m+':', getattr(importlib.import_module(m), '__version__', 'unknown'))"

venv-check:
	@echo "Checking venv package availability..."
	-$(VENV_DIR)/bin/python -c "import sys; print('Python:', sys.executable)"
	-$(VENV_DIR)/bin/python -c "import importlib; mods=['numpy','pandas','duckdb','lightgbm','sklearn'];\
import traceback;\
print('Modules:');\
\
\
\
[print(' ', m, getattr(importlib.import_module(m), '__version__', 'unknown')) for m in mods]"

env-info:
	@echo "System Python: $$(command -v python3)"
	@echo "Venv Python:   $(VENV_DIR)/bin/python"
	@echo "Venv Pip List (top 10):"
	-$(VENV_DIR)/bin/pip list | head -n 12

requirements.txt:
	@echo "Creating requirements.txt for Python dependencies..."
	@echo 'psycopg2-binary>=2.9.0' > requirements.txt
	@echo 'duckdb>=0.9.0' >> requirements.txt
	@echo 'pandas>=1.5.0' >> requirements.txt
	@echo 'numpy>=1.21.0' >> requirements.txt
	@echo 'tqdm>=4.64.0' >> requirements.txt
	@echo 'scikit-learn>=1.1.0' >> requirements.txt
	@echo 'lightgbm>=4.0.0' >> requirements.txt
	@echo 'shap>=0.41.0' >> requirements.txt
	@echo 'matplotlib>=3.5.0' >> requirements.txt
	@echo 'seaborn>=0.11.0' >> requirements.txt
	@echo 'jupyter>=1.0.0' >> requirements.txt
	@echo "requirements.txt created"

install-deps:
	@echo "Installing system dependencies..."
	sudo apt-get update
	sudo apt-get install -y \
		build-essential \
		cmake \
		pkg-config \
		libssl-dev \
		libreadline-dev \
		libxml2-dev \
		libxslt-dev \
		libicu-dev \
		zlib1g-dev \
		nlohmann-json3-dev \
		python3-dev \
		python3-venv \
		python3-pip \
		git \
		tmux
	@echo "System dependencies installed"

# PostgreSQL with AQD extensions
postgres: $(BUILD_DIR)/postgres_configured $(BUILD_DIR)/aqd_integrated
	@echo "Building PostgreSQL with AQD extensions..."
	# Ensure all generated headers are ready (errcodes.h, catalog *_d.h, etc.)
	cd $(POSTGRES_SRC) && $(MAKE) -C src/backend generated-headers
	# Now run the parallel build
	cd $(POSTGRES_SRC) && $(MAKE) -j$(shell nproc)
	cd $(POSTGRES_SRC) && $(MAKE) install
	@echo "PostgreSQL with AQD built and installed to $(INSTALL_DIR)"

$(BUILD_DIR)/postgres_configured: $(POSTGRES_SRC)/configure
	@echo "Configuring PostgreSQL..."
	# Ensure build/install dirs exist even under parallel make
	@mkdir -p $(BUILD_DIR) $(INSTALL_DIR) $(DATA_DIR)
	cd $(POSTGRES_SRC) && ./configure $(PG_CONFIG_OPTIONS)
	touch $@

$(BUILD_DIR)/aqd_integrated: aqd_feature_logger.h aqd_feature_logger.c aqd_query_router.h aqd_query_router.c aqd_integration.patch
	@echo "Integrating AQD into PostgreSQL source..."
	
	# Copy AQD files to PostgreSQL source
	cp aqd_feature_logger.h $(POSTGRES_SRC)/src/include/
	cp aqd_feature_logger.c $(POSTGRES_SRC)/src/backend/utils/misc/
	cp aqd_query_router.h $(POSTGRES_SRC)/src/include/
	cp aqd_query_router.c $(POSTGRES_SRC)/src/backend/utils/misc/
	
	# Apply integration patch only if not already integrated
	@if ! grep -q "aqd_define_guc_variables" $(POSTGRES_SRC)/src/backend/utils/misc/guc.c; then \
	  echo "Applying AQD integration patch..."; \
	  cd $(POSTGRES_SRC) && patch -p1 < ../aqd_integration.patch; \
	else \
	  echo "AQD integration patch already applied; skipping."; \
	fi
	
	touch $@

# LightGBM trainer
lightgbm: $(BUILD_DIR)/lightgbm_trainer

$(BUILD_DIR)/lightgbm_trainer: lightgbm_trainer.cpp
	@echo "Building LightGBM trainer..."
	$(CXX) $(CXXFLAGS) $(LIGHTGBM_INCLUDE) -o $@ $< $(LIGHTGBM_LIBS) $(JSON_LIBS)
	@echo "LightGBM trainer built: $@"

# GNN trainers
gnn: $(BUILD_DIR)/gnn_trainer $(BUILD_DIR)/gnn_trainer_real

$(BUILD_DIR)/gnn_trainer: gnn_trainer.cpp rginn.c rginn.h | $(BUILD_DIR)
	@echo "Building simple GNN trainer..."
	# nlohmann-json is header-only (installed via nlohmann-json3-dev); link with rginn core
	$(CXX) $(CXXFLAGS) -o $@ gnn_trainer.cpp rginn.c -I/usr/include
	@echo "Simple GNN trainer built: $@"

$(BUILD_DIR)/gnn_trainer_real: gnn_trainer_real.cpp rginn.c rginn.h | $(BUILD_DIR)
	@echo "Building real GNN trainer with backpropagation..."
	$(CXX) $(CXXFLAGS) -o $@ gnn_trainer_real.cpp rginn.c -I/usr/include
	@echo "Real GNN trainer built: $@"

gnn-labels:
	@echo "Generating GNN labels from execution data..."
	$(VENV_DIR)/bin/python make_gnn_labels.py

# Data collection pipeline
data-collection: $(VENV_DIR) data_collector.py generate_benchmark_queries.py import_benchmark_datasets.py
	@echo "Setting up data collection pipeline..."
	chmod +x data_collector.py
	chmod +x generate_benchmark_queries.py
	chmod +x import_benchmark_datasets.py
	@echo "Data collection pipeline ready"

# Testing
test: postgres lightgbm
	@echo "Running AQD system tests..."
	@echo "1. Testing PostgreSQL with AQD..."
	$(INSTALL_DIR)/bin/psql --version
	
	@echo "2. Testing LightGBM trainer..."
	echo '[]' > /tmp/test_data.json
	$(BUILD_DIR)/lightgbm_trainer /tmp/test_data.json /tmp/test_model.txt || true
	
	@echo "3. Testing data collection..."
	$(VENV_DIR)/bin/python -c "import psycopg2, duckdb; print('Database drivers OK')"
	
	@echo "Basic tests completed"

# Data import and query generation
import-data: data-collection $(INSTALL_DIR)/bin/postgres
	@echo "Importing benchmark datasets..."
	$(VENV_DIR)/bin/python import_benchmark_datasets.py

generate-queries: data-collection
	@echo "Generating benchmark queries..."
	$(VENV_DIR)/bin/python generate_benchmark_queries.py --output-dir $(DATA_DIR)/queries

collect-data: data-collection $(INSTALL_DIR)/bin/postgres
	@echo "Collecting training data..."
	# Start PostgreSQL in background
	$(INSTALL_DIR)/bin/pg_ctl -w -t 30 -D $(PGDATA_DIR) -l $(PGDATA_DIR)/server.log start || true
	sleep 2
	
	# Run data collection
	$(VENV_DIR)/bin/python data_collector.py
	
	# Stop PostgreSQL (fast)
	$(INSTALL_DIR)/bin/pg_ctl -m fast -D $(PGDATA_DIR) stop || true

train-model: lightgbm
	@echo "Training LightGBM model..."
	@if [ -f $(DATA_DIR)/aqd_training_*.json ]; then \
		$(BUILD_DIR)/lightgbm_trainer $(DATA_DIR)/aqd_training_*.json $(DATA_DIR)/aqd_model.txt; \
	else \
		echo "No training data found. Run 'make collect-data' first."; \
	fi

# Database initialization
init-db: postgres
	@echo "Initializing PostgreSQL database..."
	mkdir -p $(PGDATA_DIR)
	@if [ -f $(PGDATA_DIR)/PG_VERSION ]; then \
		echo "PGDATA already initialized at $(PGDATA_DIR)"; \
	else \
		$(INSTALL_DIR)/bin/initdb -D $(PGDATA_DIR); \
	fi
	
	# Configure PostgreSQL for AQD (compiled-in; no shared_preload_libraries needed)
	@sed -i '/^shared_preload_libraries.*aqd/d' $(PGDATA_DIR)/postgresql.conf || true
	@grep -q "^aqd.enable_feature_logging" $(PGDATA_DIR)/postgresql.conf || \
		echo "aqd.enable_feature_logging = false" >> $(PGDATA_DIR)/postgresql.conf
	@grep -q "^aqd.feature_log_path" $(PGDATA_DIR)/postgresql.conf || \
		echo "aqd.feature_log_path = '$(DATA_DIR)/aqd_features.csv'" >> $(PGDATA_DIR)/postgresql.conf
	@grep -q "^aqd.routing_method" $(PGDATA_DIR)/postgresql.conf || \
		echo "aqd.routing_method = 1" >> $(PGDATA_DIR)/postgresql.conf
	@grep -q "^aqd.cost_threshold" $(PGDATA_DIR)/postgresql.conf || \
		echo "aqd.cost_threshold = 10000" >> $(PGDATA_DIR)/postgresql.conf
	
	@echo "Database initialized in $(PGDATA_DIR)"

start-db: init-db
	@echo "Starting PostgreSQL server (idempotent)..."
	@if $(INSTALL_DIR)/bin/pg_ctl -D $(PGDATA_DIR) status >/dev/null 2>&1; then \
		echo "PostgreSQL already running"; \
	else \
		$(INSTALL_DIR)/bin/pg_ctl -w -t 30 -D $(PGDATA_DIR) -l $(PGDATA_DIR)/server.log start; \
		echo "PostgreSQL started"; \
	fi
	sleep 1
	@# Create benchmark database if not exists
	@$(INSTALL_DIR)/bin/createdb -h localhost -p 5432 benchmark_datasets 2>/dev/null || true
	@echo "PostgreSQL server ready"

stop-db:
	@echo "Stopping PostgreSQL server..."
	$(INSTALL_DIR)/bin/pg_ctl -m fast -D $(PGDATA_DIR) stop || true

# Safe restart (fast, with timeout and fallback)
restart-db:
	@echo "Restarting PostgreSQL server (fast)..."
	$(INSTALL_DIR)/bin/pg_ctl -w -t 30 -m fast -D $(PGDATA_DIR) restart || \
	( echo "Fast restart timed out, forcing immediate restart..." ; \
	  $(INSTALL_DIR)/bin/pg_ctl -m immediate -D $(PGDATA_DIR) restart )
	@echo "PostgreSQL server restarted"

# Harden AQD settings for stability (disable logging + decision application)
safe-db:
	@echo "Hardening AQD settings for stability..."
	@sed -i "/^aqd.enable_feature_logging/s/=.*/= false/" $(PGDATA_DIR)/postgresql.conf || true
	@grep -q "^aqd.apply_decision" $(PGDATA_DIR)/postgresql.conf || \
		echo "aqd.apply_decision = off" >> $(PGDATA_DIR)/postgresql.conf
	@# Clear ALTER SYSTEM overrides that force ML routing
	@sed -i "/^aqd.dispatch_method/d;/^aqd.lightgbm_model_path/d" $(PGDATA_DIR)/postgresql.auto.conf || true
	@echo "AQD settings hardened. Run 'make restart-db' to apply."

# Full pipeline
full-pipeline: all init-db start-db import-data collect-data train-model
	@echo ""
	@echo "=== AQD Full Pipeline Completed ==="
	@echo "PostgreSQL with AQD: $(INSTALL_DIR)/bin/postgres"
	@echo "LightGBM trainer: $(BUILD_DIR)/lightgbm_trainer"
	@echo "Training data: $(DATA_DIR)/aqd_training_*.json"
	@echo "Trained model: $(DATA_DIR)/aqd_model.txt"
	@echo "Database data: $(PGDATA_DIR)"
	@echo ""
	@echo "To test the system:"
	@echo "1. Start database: make start-db"
	@echo "2. Connect: $(INSTALL_DIR)/bin/psql -h localhost -p 5432 benchmark_datasets"
	@echo "3. Test routing: SELECT * FROM pg_stat_activity;"
	@echo "4. Stop database: make stop-db"

# Benchmarking
benchmark: full-pipeline
	@echo "Running AQD benchmarks..."
	$(VENV_DIR)/bin/python benchmark_runner.py

# Documentation
docs:
	@echo "AQD Documentation available in README.md"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(INSTALL_DIR)  
	rm -rf $(VENV_DIR)
	cd $(POSTGRES_SRC) && $(MAKE) clean || true
	rm -f aqd_*.txt aqd_*.json aqd_*.csv
	@echo "Clean completed"

distclean: clean
	@echo "Deep cleaning..."
	rm -rf $(DATA_DIR)
	cd $(POSTGRES_SRC) && $(MAKE) distclean || true
	rm -f $(POSTGRES_SRC)/src/include/aqd_*.h
	rm -f $(POSTGRES_SRC)/src/backend/utils/misc/aqd_*.c
	@echo "Deep clean completed"

.SECONDARY: # Don't delete intermediate files
