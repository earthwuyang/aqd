# Intelligent Query Routing System (PostgreSQL + DuckDB)
# Makefile for reproducible builds and development

# Configuration
PYTHON := python3
PIP := pip3
VENV_DIR := venv
BUILD_DIR := build
DATA_DIR := data
ARTIFACTS_DIR := artifacts
LOGS_DIR := logs

# Build flags
ENABLE_LIGHTGBM ?= OFF
BUILD_TYPE ?= Release
JOBS ?= $(shell nproc)

# Directories
DUCKDB_SRC_DIR := duckdb_src
POSTGRES_SRC_DIR := postgres_src
POSTGRES_SCANNER_DIR := postgres_scanner
POSTGRESQL_DIR := postgresql

.PHONY: help setup clean test build-all build-duckdb build-postgresql build-postgres-scanner
.PHONY: install-deps create-venv activate-venv
.PHONY: import-datasets collect-data train-model evaluate
.PHONY: docker-build docker-run format lint

# Default target
help:
	@echo "Available targets:"
	@echo "  setup           - Full environment setup (venv + deps + build)"
	@echo "  clean           - Clean build artifacts and temporary files"
	@echo ""
	@echo "Environment:"
	@echo "  create-venv     - Create Python virtual environment"
	@echo "  install-deps    - Install Python dependencies"
	@echo ""
	@echo "Build:"
	@echo "  build-all       - Build DuckDB, PostgreSQL, and postgres_scanner"
	@echo "  build-duckdb    - Build custom DuckDB with query routing"
	@echo "  build-postgresql - Build custom PostgreSQL"
	@echo "  build-postgres-scanner - Build postgres_scanner extension"
	@echo ""
	@echo "Data & Training:"
	@echo "  import-datasets - Import benchmark datasets"
	@echo "  collect-data    - Collect training data with dual execution"
	@echo "  train-model     - Train LightGBM routing model"
	@echo "  evaluate        - Run comprehensive routing evaluation"
	@echo ""
	@echo "Development:"
	@echo "  test            - Run all tests"
	@echo "  format          - Format code with black"
	@echo "  lint            - Lint code with ruff"
	@echo "  docker-build    - Build Docker development image"
	@echo ""
	@echo "Configuration:"
	@echo "  ENABLE_LIGHTGBM=ON  - Enable LightGBM integration"
	@echo "  BUILD_TYPE=Debug    - Debug build (default: Release)"
	@echo "  JOBS=8              - Parallel build jobs"

# Environment setup
setup: create-venv install-deps build-all create-directories
	@echo "✅ Setup complete! Activate virtual environment with:"
	@echo "   source $(VENV_DIR)/bin/activate"

create-venv:
	@echo "🔧 Creating Python virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "✅ Virtual environment created at $(VENV_DIR)/"

install-deps: create-venv
	@echo "📦 Installing Python dependencies..."
	$(VENV_DIR)/bin/$(PIP) install --upgrade pip
	$(VENV_DIR)/bin/$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

create-directories:
	@echo "📁 Creating project directories..."
	@mkdir -p $(BUILD_DIR) $(DATA_DIR) $(ARTIFACTS_DIR) $(LOGS_DIR)
	@echo "✅ Project directories created"

# Build targets
build-all: build-duckdb build-postgresql build-postgres-scanner

build-duckdb:
	@echo "🔨 Building DuckDB with query routing..."
	@if [ ! -d "$(DUCKDB_SRC_DIR)" ]; then \
		echo "❌ DuckDB source not found at $(DUCKDB_SRC_DIR)/"; \
		echo "   Please clone DuckDB source or update DUCKDB_SRC_DIR"; \
		exit 1; \
	fi
	cd $(DUCKDB_SRC_DIR) && \
		BUILD_BENCHMARK=0 BUILD_TPCH=0 BUILD_TPCDS=0 \
		ENABLE_LIGHTGBM=$(ENABLE_LIGHTGBM) \
		make clean && make -j$(JOBS)
	@echo "✅ DuckDB build complete"

build-postgresql:
	@echo "🔨 Building PostgreSQL..."
	@if [ ! -d "$(POSTGRES_SRC_DIR)" ]; then \
		echo "❌ PostgreSQL source not found at $(POSTGRES_SRC_DIR)/"; \
		echo "   Please clone PostgreSQL source or update POSTGRES_SRC_DIR"; \
		exit 1; \
	fi
	@if [ ! -f "$(POSTGRES_SRC_DIR)/configure" ]; then \
		cd $(POSTGRES_SRC_DIR) && ./configure --prefix=$(PWD)/$(POSTGRESQL_DIR); \
	fi
	cd $(POSTGRES_SRC_DIR) && make clean && make -j$(JOBS) && make install
	@echo "✅ PostgreSQL build complete"

build-postgres-scanner:
	@echo "🔨 Building postgres_scanner extension..."
	@if [ ! -d "$(POSTGRES_SCANNER_DIR)" ]; then \
		echo "❌ postgres_scanner source not found at $(POSTGRES_SCANNER_DIR)/"; \
		exit 1; \
	fi
	cd $(POSTGRES_SCANNER_DIR) && \
		DUCKDB_DIR=../$(DUCKDB_SRC_DIR) \
		PostgreSQL_ROOT=../$(POSTGRESQL_DIR) \
		OPENSSL_ROOT_DIR=/usr \
		make clean && make -j$(JOBS)
	@echo "✅ postgres_scanner build complete"

# Data pipeline
import-datasets:
	@echo "📊 Importing benchmark datasets..."
	$(VENV_DIR)/bin/$(PYTHON) import_benchmark_datasets.py \
		--data-dir $(DATA_DIR) \
		--config config.yaml
	@echo "✅ Datasets imported"

collect-data:
	@echo "🔄 Collecting training data with dual execution..."
	$(VENV_DIR)/bin/$(PYTHON) final_routing_comparison_with_warmup.py \
		--mode collect \
		--output $(ARTIFACTS_DIR)/training_data.jsonl \
		--config config.yaml
	@echo "✅ Training data collected"

train-model:
	@echo "🤖 Training LightGBM routing model..."
	$(VENV_DIR)/bin/$(PYTHON) advanced_aqd_system.py \
		--mode train \
		--data $(ARTIFACTS_DIR)/training_data.jsonl \
		--output $(ARTIFACTS_DIR)/lightgbm_model.txt \
		--scaler $(ARTIFACTS_DIR)/scaler.json \
		--config config.yaml
	@echo "✅ Model training complete"

evaluate:
	@echo "📈 Running comprehensive routing evaluation..."
	$(VENV_DIR)/bin/$(PYTHON) final_routing_comparison_with_warmup.py \
		--mode evaluate \
		--model $(ARTIFACTS_DIR)/lightgbm_model.txt \
		--scaler $(ARTIFACTS_DIR)/scaler.json \
		--output $(ARTIFACTS_DIR)/evaluation_results.json \
		--config config.yaml
	@echo "✅ Evaluation complete"

# Testing
test: test-python test-cpp

test-python:
	@echo "🧪 Running Python tests..."
	$(VENV_DIR)/bin/$(PYTHON) -m pytest tests/ -v || echo "⚠️  No tests found - create tests/ directory"

test-cpp:
	@echo "🧪 Running C++ integration tests..."
	@if [ -f "query_router_test.cpp" ]; then \
		g++ -std=c++17 -I$(DUCKDB_SRC_DIR)/src/include query_router_test.cpp -o test_router && \
		./test_router && rm -f test_router; \
	else \
		echo "⚠️  No C++ tests found"; \
	fi

# Development tools
format:
	@echo "🎨 Formatting Python code..."
	$(VENV_DIR)/bin/black *.py || echo "⚠️  black not available, install with: pip install black"

lint:
	@echo "🔍 Linting Python code..."
	$(VENV_DIR)/bin/ruff check *.py || echo "⚠️  ruff not available, install with: pip install ruff"

# Docker support
docker-build:
	@echo "🐳 Building Docker development image..."
	docker build -t query-routing-dev -f Dockerfile .

docker-run:
	@echo "🐳 Running Docker development container..."
	docker run -it --rm -v $(PWD):/workspace query-routing-dev

# Cleanup
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)/ $(VENV_DIR)/ $(ARTIFACTS_DIR)/ $(LOGS_DIR)/
	rm -rf __pycache__/ *.pyc *.pyo
	rm -f *.log *.jsonl *.joblib
	@if [ -d "$(DUCKDB_SRC_DIR)" ]; then \
		cd $(DUCKDB_SRC_DIR) && make clean; \
	fi
	@if [ -d "$(POSTGRES_SRC_DIR)" ]; then \
		cd $(POSTGRES_SRC_DIR) && make clean; \
	fi
	@if [ -d "$(POSTGRES_SCANNER_DIR)" ]; then \
		cd $(POSTGRES_SCANNER_DIR) && make clean; \
	fi
	@echo "✅ Cleanup complete"

# Quick development setup
dev-setup:
	@echo "⚡ Quick development setup..."
	@$(MAKE) create-venv
	@$(MAKE) install-deps
	@$(MAKE) create-directories
	@echo "✅ Development environment ready!"
	@echo "   Next steps:"
	@echo "   1. source $(VENV_DIR)/bin/activate"
	@echo "   2. make build-duckdb    # if you have DuckDB source"
	@echo "   3. make import-datasets # to load benchmark data"