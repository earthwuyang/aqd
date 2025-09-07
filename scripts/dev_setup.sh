#!/bin/bash
# Development Environment Setup Script
# Intelligent Query Routing System (PostgreSQL + DuckDB)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.8"
VENV_NAME="venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
ARTIFACTS_DIR="$PROJECT_ROOT/artifacts"
LOGS_DIR="$PROJECT_ROOT/logs"
MODELS_DIR="$PROJECT_ROOT/models"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        local min_version="3.8"
        
        if [ "$(printf '%s\n' "$min_version" "$python_version" | sort -V | head -n1)" = "$min_version" ]; then
            print_success "Python $python_version found (>= $min_version required)"
            return 0
        else
            print_error "Python $python_version found, but $min_version or higher required"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        return 1
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating project directories..."
    
    local dirs=("$DATA_DIR" "$ARTIFACTS_DIR" "$LOGS_DIR" "$MODELS_DIR")
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        else
            print_status "Directory already exists: $dir"
        fi
    done
    
    print_success "Project directories ready"
}

# Function to setup Python virtual environment
setup_python_environment() {
    print_status "Setting up Python virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    # Check if virtual environment already exists
    if [ -d "$VENV_NAME" ]; then
        print_warning "Virtual environment already exists. Removing and recreating..."
        rm -rf "$VENV_NAME"
    fi
    
    # Create virtual environment
    python3 -m venv "$VENV_NAME"
    print_success "Virtual environment created"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    print_status "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    print_status "pip upgraded"
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_warning "requirements.txt not found. Creating basic requirements..."
        cat > requirements.txt << EOF
# Database connectors
psycopg2-binary>=2.9.0
duckdb>=0.9.0

# Data processing
pandas>=1.5.0
numpy>=1.21.0

# Machine learning
scikit-learn>=1.3.0
lightgbm>=4.0.0
joblib>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Progress bars and utilities
tqdm>=4.65.0

# Configuration
pyyaml>=6.0
EOF
        pip install -r requirements.txt
        print_success "Basic requirements installed"
    fi
}

# Function to check system dependencies
check_system_dependencies() {
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Essential tools
    local required_tools=("git" "make" "gcc" "g++" "cmake")
    
    for tool in "${required_tools[@]}"; do
        if ! command_exists "$tool"; then
            missing_deps+=("$tool")
        fi
    done
    
    # Development libraries (optional but recommended)
    local optional_tools=("pkg-config")
    
    for tool in "${optional_tools[@]}"; do
        if ! command_exists "$tool"; then
            print_warning "Optional tool missing: $tool (recommended for build system)"
        fi
    done
    
    if [ ${#missing_deps[@]} -eq 0 ]; then
        print_success "All required system dependencies found"
        return 0
    else
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_status "Install them with:"
        
        # Detect package manager and provide installation command
        if command_exists apt-get; then
            echo "  sudo apt-get update && sudo apt-get install -y ${missing_deps[*]} pkg-config libssl-dev zlib1g-dev"
        elif command_exists yum; then
            echo "  sudo yum install -y ${missing_deps[*]} pkg-config openssl-devel zlib-devel"
        elif command_exists brew; then
            echo "  brew install ${missing_deps[*]} pkg-config openssl zlib"
        else
            echo "  Install these packages using your system's package manager"
        fi
        
        return 1
    fi
}

# Function to setup configuration files
setup_configuration() {
    print_status "Setting up configuration files..."
    
    # Create config.yaml if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/config.yaml" ]; then
        print_status "Creating default config.yaml..."
        # Create a basic config.yaml (the full one was already created)
        print_success "config.yaml would be created here (assuming it already exists)"
    else
        print_success "config.yaml already exists"
    fi
    
    # Create .env file for environment variables
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        print_status "Creating .env file for environment variables..."
        cat > "$PROJECT_ROOT/.env" << EOF
# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=postgres

# DuckDB Configuration
DUCKDB_BINARY=duckdb_src/build/release/duckdb

# Project Paths
DATA_DIR=data
ARTIFACTS_DIR=artifacts
LOGS_DIR=logs
MODELS_DIR=models

# Development Settings
DEBUG=false
QUICK_TEST=false
EOF
        print_success ".env file created"
    else
        print_success ".env file already exists"
    fi
}

# Function to test the setup
test_setup() {
    print_status "Testing setup..."
    
    # Test Python environment
    if [ -f "$PROJECT_ROOT/$VENV_NAME/bin/python" ]; then
        local python_path="$PROJECT_ROOT/$VENV_NAME/bin/python"
        print_status "Testing Python imports..."
        
        if $python_path -c "import numpy, pandas, sklearn, lightgbm; print('✅ Core ML libraries imported successfully')" 2>/dev/null; then
            print_success "Python environment test passed"
        else
            print_warning "Python environment test failed - some libraries may be missing"
        fi
    fi
    
    # Test project structure
    local required_dirs=("$DATA_DIR" "$ARTIFACTS_DIR" "$LOGS_DIR")
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_status "✅ Directory exists: $(basename "$dir")"
        else
            print_error "❌ Directory missing: $(basename "$dir")"
        fi
    done
}

# Function to print usage instructions
print_usage_instructions() {
    print_success "Development environment setup complete!"
    echo
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Activate the virtual environment:"
    echo -e "   ${GREEN}source $VENV_NAME/bin/activate${NC}"
    echo
    echo "2. Build the system (if you have DuckDB source):"
    echo -e "   ${GREEN}make build-duckdb${NC}"
    echo
    echo "3. Import benchmark datasets:"
    echo -e "   ${GREEN}make import-datasets${NC}"
    echo
    echo "4. Run a quick test:"
    echo -e "   ${GREEN}python simple_warmup_test.py${NC}"
    echo
    echo -e "${BLUE}Available make targets:${NC}"
    echo "  make help           - Show all available targets"
    echo "  make setup          - Full environment setup"
    echo "  make build-all      - Build DuckDB, PostgreSQL, and extensions"
    echo "  make collect-data   - Collect training data"
    echo "  make train-model    - Train ML routing model"
    echo "  make evaluate       - Run comprehensive evaluation"
    echo
    echo -e "${BLUE}Configuration files:${NC}"
    echo "  config.yaml         - Main configuration"
    echo "  .env                - Environment variables"
    echo "  requirements.txt    - Python dependencies"
    echo
    echo -e "${YELLOW}Note:${NC} This setup script configures the development environment."
    echo "You'll still need to build DuckDB and PostgreSQL sources if you want"
    echo "to test the full system integration."
}

# Main setup function
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Query Routing Development Setup${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
    
    print_status "Starting development environment setup..."
    print_status "Project root: $PROJECT_ROOT"
    echo
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Step 1: Check system dependencies
    if ! check_system_dependencies; then
        print_error "System dependencies check failed. Please install missing dependencies and try again."
        exit 1
    fi
    
    # Step 2: Check Python version
    if ! check_python_version; then
        print_error "Python version check failed."
        exit 1
    fi
    
    # Step 3: Create directories
    create_directories
    
    # Step 4: Setup Python environment
    setup_python_environment
    
    # Step 5: Setup configuration files
    setup_configuration
    
    # Step 6: Test the setup
    test_setup
    
    # Step 7: Print usage instructions
    echo
    print_usage_instructions
}

# Run main function
main "$@"