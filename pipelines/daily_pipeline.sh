#!/bin/bash
#
# Portfolio Management Pipeline Orchestrator
# This script manages the execution of Jupyter notebooks in sequence
# and handles errors gracefully.
#

# Exit immediately if a command exits with a non-zero status
set -e

# Enable debugging with -x flag (uncomment if needed)
# set -x

# Check if running with bash
if [ -z "$BASH_VERSION" ]; then
    echo "❌ This script requires bash. Please run with: bash daily_pipeline.sh"
    exit 1
fi

# Get the directory where this script is located (bash-compatible method)
if [ -n "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    # Fallback for other shells
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

##############################################################################
# HELPER FUNCTIONS
##############################################################################

log() {
    local message=$1
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message"
}

check_dependency() {
    local cmd=$1
    if ! command -v "$cmd" &> /dev/null; then
        log "❌ Required dependency not found: $cmd"
        exit 1
    fi
}

##############################################################################
# DUCKDB SETUP
##############################################################################

setup_duckdb() {
    log "Setting up DuckDB..."
    
    # Check if DuckDB is already installed
    if command -v duckdb &> /dev/null; then
        log "✅ DuckDB is already installed."
        duckdb --version
        return 0
    fi
    
    log "Installing DuckDB..."
    curl https://install.duckdb.org | sh
    
    # Add DuckDB to PATH and verify installation
    DUCKDB_PATH="$HOME/.duckdb/cli/latest"
    if [ -f "$DUCKDB_PATH/duckdb" ]; then
        log "Adding DuckDB to PATH..."
        export PATH="$DUCKDB_PATH:$PATH"
        log "✅ DuckDB installed successfully."
        $DUCKDB_PATH/duckdb --version
        return 0
    else
        log "❌ DuckDB installation failed."
        return 1
    fi
}

##############################################################################
# VARIABLES
##############################################################################

QUARTO_VERSION="1.6.39"
QUARTO_ARCH="linux-amd64"

# Update PATH for this session
export PATH="$PATH:$HOME/.local/bin"

##############################################################################
# QUARTO SETUP
##############################################################################

# Function to check Quarto version
check_quarto_version() {
    local CURRENT_QUARTO=$(which quarto 2>/dev/null)
    if [ -n "$CURRENT_QUARTO" ]; then
        local INSTALLED_VERSION=$("$CURRENT_QUARTO" --version 2>/dev/null)
        if [ "$INSTALLED_VERSION" = "$QUARTO_VERSION" ]; then
            log "✅ Quarto $QUARTO_VERSION is already installed."
            return 0
        fi
    fi
    return 1
}

# Function to check if we're in a virtual environment
in_virtualenv() {
    if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_DEFAULT_ENV" ]; then
        return 0  # True
    else
        return 1  # False
    fi
}

setup_quarto() {
    log "Setting up Quarto..."
    
    # Install required packages for Quarto
    if in_virtualenv; then
        # Try to find the correct pip in the virtual environment
        local VENV_PIP=""
        
        # Check for conda environment first
        if [ -n "$CONDA_DEFAULT_ENV" ]; then
            log "Detected conda environment: $CONDA_DEFAULT_ENV"
            # Use the current python's pip
            VENV_PIP="$(which pip 2>/dev/null)"
            if [ -z "$VENV_PIP" ]; then
                VENV_PIP="$(which pip3 2>/dev/null)"
            fi
        elif [ -n "$VIRTUAL_ENV" ]; then
            log "Detected virtual environment: $VIRTUAL_ENV"
            # For regular virtual environments, try python -m pip first
            if python -m pip --version &>/dev/null; then
                log "Using python -m pip for package management"
                VENV_PIP="python -m pip"
            else
                # Check common virtual environment pip locations
                if [ -f "$VIRTUAL_ENV/bin/pip" ]; then
                    VENV_PIP="$VIRTUAL_ENV/bin/pip"
                elif [ -f "$VIRTUAL_ENV/bin/pip3" ]; then
                    VENV_PIP="$VIRTUAL_ENV/bin/pip3"
                fi
            fi
        fi
        
        if [ -n "$VENV_PIP" ]; then
            log "Using pip: $VENV_PIP"
            
            # Check if packages are already installed
            if python -c "import jupyter, yaml" &>/dev/null; then
                log "✅ Required Python packages are already installed."
            else
                log "Installing required packages..."
                if [[ "$VENV_PIP" == "python -m pip" ]]; then
                    python -m pip install jupyter pyyaml
                else
                    "$VENV_PIP" install jupyter pyyaml
                fi
                log "✅ Packages installed successfully."
            fi
        else
            log "❌ Could not find pip in virtual environment."
            log "Trying to use python -m pip instead..."
            
            # Check if packages are already installed
            if python -c "import jupyter, yaml" &>/dev/null; then
                log "✅ Required Python packages are already installed."
            else
                log "Installing required packages using python -m pip..."
                python -m pip install jupyter pyyaml
                log "✅ Packages installed successfully."
            fi
        fi
    else
        log "⚠️ Not running in a virtual environment."
        
        # Check if required packages are already installed
        python3 -c "import jupyter, yaml" &>/dev/null
        if [ $? -eq 0 ]; then
            log "✅ Required Python packages are already installed."
        else
            log "Required Python packages need to be installed."
            read -p "Do you want to create a new virtual environment? (y/n): " create_venv
            
            if [[ "$create_venv" == "y" || "$create_venv" == "Y" ]]; then
                log "Creating virtual environment..."
                python3 -m venv .venv
                source .venv/bin/activate
                python3 -m pip install jupyter pyyaml
                log "✅ Virtual environment created and packages installed."
            else
                log "Installing packages to user directory..."
                python3 -m pip install --user jupyter pyyaml
                if [ $? -ne 0 ]; then
                    log "❌ Failed to install packages."
                    log "You may need to install packages manually:"
                    log "  python3 -m pip install --user jupyter pyyaml"
                    read -p "Continue anyway? (y/n): " continue_anyway
                    if [[ "$continue_anyway" != "y" && "$continue_anyway" != "Y" ]]; then
                        exit 1
                    fi
                fi
            fi
        fi
    fi
    
    # If Quarto is not installed or version doesn't match, proceed with installation
    if ! check_quarto_version; then
        log "Installing Quarto $QUARTO_VERSION..."
        mkdir -p "$HOME/opt" "$HOME/.local/bin"
        curl -LO "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-${QUARTO_ARCH}.tar.gz"
        tar -xzf "quarto-${QUARTO_VERSION}-${QUARTO_ARCH}.tar.gz" -C "$HOME/opt"
        ln -sf "$HOME/opt/quarto-${QUARTO_VERSION}/bin/quarto" "$HOME/.local/bin/quarto"
        
        if ! grep -q "$HOME/.local/bin" "$HOME/.bashrc"; then
            echo 'export PATH="$PATH:$HOME/.local/bin"' >> "$HOME/.bashrc"
        fi
        
        # Verify installation
        quarto check || { log "❌ Quarto installation failed"; exit 1; }
        
        rm "quarto-${QUARTO_VERSION}-${QUARTO_ARCH}.tar.gz"
    fi
    
    # Print Quarto version for debugging
    quarto --version
}

##############################################################################
# NOTEBOOK FUNCTIONS
##############################################################################

# Function to validate the JSON structure of a notebook
validate_notebook() {
    local notebook=$1
    log "Validating JSON structure of $notebook..."
    if jq empty "$notebook" >/dev/null 2>&1; then
        log "✅ JSON structure of $notebook is valid."
        return 0
    else
        log "❌ JSON structure of $notebook is invalid."
        return 1
    fi
}

# Function to repair a corrupted notebook
repair_notebook() {
    local notebook=$1
    log "Attempting to repair $notebook..."
    
    # Validate the JSON structure
    if validate_notebook "$notebook"; then
        log "✅ Notebook $notebook appears valid. No repair needed."
        return 0
    fi

    # Attempt to convert the notebook to a Python script
    log "Converting $notebook to a Python script..."
    if jupyter nbconvert --to script "$notebook" --output "${notebook%.ipynb}.py"; then
        log "✅ Successfully converted $notebook to a Python script."
        log "Please manually review and recreate the notebook if needed."
        return 0
    else
        log "❌ Failed to convert $notebook to a Python script."
    fi

    # Create a placeholder notebook if all else fails
    log "Creating a placeholder notebook for $notebook..."
    cat <<EOF > "$notebook"
{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
EOF
    log "⚠️ Placeholder notebook created for $notebook. Please manually restore its content."
    return 1
}

# Function to run a notebook and check for errors
run_notebook() {
    local notebook_path=$1
    local full_path="$PROJECT_ROOT/$notebook_path"
    
    log "Running $notebook_path..."
    
    # First check if the notebook exists
    if [ ! -f "$full_path" ]; then
        log "❌ Notebook $full_path does not exist."
        log "Looking for notebook in current directory: $(pwd)"
        log "Project root: $PROJECT_ROOT"
        log "Available files in data directory:"
        if [ -d "$PROJECT_ROOT/data" ]; then
            ls -la "$PROJECT_ROOT/data/" | grep -E '\.(ipynb|py)$' || log "No .ipynb or .py files found"
        else
            log "Data directory does not exist at $PROJECT_ROOT/data"
        fi
        return 1
    fi
    
    # Check if the notebook has valid JSON structure
    if ! validate_notebook "$full_path"; then
        log "❌ Notebook $full_path has invalid JSON structure. Attempting repair..."
        repair_notebook "$full_path" || return 1
    fi
    
    # Change to project root directory for execution
    cd "$PROJECT_ROOT"
    
    # Add the project root to Python path for imports
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Try to execute the notebook
    if jupyter nbconvert --to notebook --execute "$notebook_path" --inplace --ExecutePreprocessor.timeout=600; then
        log "✅ Successfully completed $notebook_path"
    else
        log "❌ Error executing $notebook_path. This might be due to issues with the code inside the notebook."
        log "Please review the notebook manually."
        log "Common issues:"
        log "  - Missing Python modules (check imports)"
        log "  - Incorrect file paths"
        log "  - API keys or environment variables not set"
        return 1
    fi
}

##############################################################################
# MAIN EXECUTION
##############################################################################

main() {
    log "===== Portfolio Management Pipeline ====="
    log "Script directory: $SCRIPT_DIR"
    log "Project root: $PROJECT_ROOT"
    
    # Check if we're in the right directory structure
    if [ ! -d "$PROJECT_ROOT/data" ]; then
        log "❌ Expected data directory not found at $PROJECT_ROOT/data"
        log "Please ensure you're running this script from the correct location."
        exit 1
    fi
    
    # Setup Quarto
    setup_quarto
    
    # Setup DuckDB
    setup_duckdb || exit 1
    
    log "Starting execution of notebooks in sequence..."
    
    # Execute notebooks in sequence
    log "Step 1/4: Scraping tickers..."
    run_notebook "data/scrape_tickers.ipynb" || exit 1
    
    log "Step 2/4: Scraping quotes..."
    run_notebook "data/scrape_quotes.ipynb" || {
        log "⚠️ Step 2 failed. This might be due to missing Python modules."
        log "Please check that all required modules are installed and importable."
        log "Try running: python -c 'from py.fetch_price_history import fetch_price_history_openbb'"
        exit 1
    }

    log "Step 3/4: Fetching Quotes Datasets into a Single Dataset..."
    run_notebook "data/fetch_datasets.ipynb" || exit 1
    
    log "Step 4/4: Load .csv Files to DuckDB (data.db)..."
    local duckdb_script="$PROJECT_ROOT/data/duckdb_fetch_database.sh"
    if [ -f "$duckdb_script" ]; then
        chmod +x "$duckdb_script"
        cd "$PROJECT_ROOT"
        ./data/duckdb_fetch_database.sh || exit 1
    else
        log "❌ DuckDB script not found: $duckdb_script"
        exit 1
    fi
    
    log "===== Pipeline completed successfully! ====="
}

# Execute the main function
main