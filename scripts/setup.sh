#!/usr/bin/env bash
# ==============================================================================
# Industrial PPE Monitoring System - Automated Quickstart Setup Script
# ==============================================================================
# Sets up Python virtual environment, dependencies, environment configuration,
# required runtime directories, model weights, and frontend packages.
# ==============================================================================

set -euo pipefail

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

log_info() {
    echo -e "${BLUE}[INFO]${RESET} $*"
}

log_step() {
    echo -e "\n${BOLD}${BLUE}==>${RESET} ${BOLD}$*${RESET}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${RESET} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $*" >&2
}

# Resolve project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Default options
PYTHON_BIN=""
SKIP_FRONTEND=false
SKIP_MODEL=false
RUN_TESTS=false

print_usage() {
    cat <<EOF
Usage: ./scripts/setup.sh [OPTIONS]

Automated setup for the Industrial PPE Monitoring System.

Options:
  --python <path>       Specify path to Python executable (default: auto-detect python3 / python)
  --skip-frontend       Skip installing frontend npm dependencies
  --skip-model          Skip checking and downloading model checkpoint
  --run-tests           Run pytest test suite after installation completes
  -h, --help            Show this help message and exit

Examples:
  ./scripts/setup.sh
  ./scripts/setup.sh --skip-frontend
  ./scripts/setup.sh --run-tests
EOF
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --skip-frontend)
            SKIP_FRONTEND=true
            shift
            ;;
        --skip-model)
            SKIP_MODEL=true
            shift
            ;;
        --run-tests)
            RUN_TESTS=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

echo -e "${BOLD}======================================================"
echo -e "      Industrial PPE Monitoring System Setup          "
echo -e "======================================================${RESET}"
log_info "Repository root: ${REPO_ROOT}"

# ------------------------------------------------------------------------------
# 1. Locate Python 3.10+
# ------------------------------------------------------------------------------
log_step "Checking Python environment..."

if [[ -z "${PYTHON_BIN}" ]]; then
    if command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
    elif command -v python &>/dev/null; then
        PYTHON_BIN="python"
    else
        log_error "No Python interpreter found. Please install Python 3.10 or newer."
        exit 1
    fi
fi

if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    log_error "Specified Python binary not found: ${PYTHON_BIN}"
    exit 1
fi

PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_MAJOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.minor)')"

if [[ "${PY_MAJOR}" -lt 3 ]] || { [[ "${PY_MAJOR}" -eq 3 ]] && [[ "${PY_MINOR}" -lt 10 ]]; }; then
    log_error "Python 3.10+ is required. Found Python ${PY_VERSION} at $(command -v "${PYTHON_BIN}")"
    exit 1
fi

log_success "Found Python ${PY_VERSION} ($("${PYTHON_BIN}" -c 'import sys; print(sys.executable)'))"

# ------------------------------------------------------------------------------
# 2. Virtual Environment Setup (.venv)
# ------------------------------------------------------------------------------
log_step "Configuring virtual environment..."

VENV_DIR="${REPO_ROOT}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    log_info "Creating virtual environment at ${VENV_DIR}..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    log_success "Virtual environment created."
else
    log_info "Virtual environment already exists at ${VENV_DIR}."
fi

# Detect activation script (Unix vs Windows / MSYS)
ACTIVATE_SCRIPT=""
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"
elif [[ -f "${VENV_DIR}/Scripts/activate" ]]; then
    ACTIVATE_SCRIPT="${VENV_DIR}/Scripts/activate"
fi

if [[ -z "${ACTIVATE_SCRIPT}" ]]; then
    log_error "Could not find virtual environment activation script in ${VENV_DIR}"
    exit 1
fi

# Activate venv in current subshell
# shellcheck source=/dev/null
source "${ACTIVATE_SCRIPT}"
log_success "Virtual environment activated."

VENV_PYTHON="$(python -c 'import sys; print(sys.executable)')"
log_info "Using virtualenv Python: ${VENV_PYTHON}"

# ------------------------------------------------------------------------------
# 3. Install Python Dependencies
# ------------------------------------------------------------------------------
log_step "Installing Python dependencies..."

log_info "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel --quiet

if [[ -f "requirements.txt" ]]; then
    log_info "Installing packages from requirements.txt..."
    python -m pip install -r requirements.txt
    log_success "Python dependencies installed successfully."
else
    log_warn "requirements.txt not found in ${REPO_ROOT}."
fi

# ------------------------------------------------------------------------------
# 4. Environment Configuration (.env)
# ------------------------------------------------------------------------------
log_step "Checking environment configuration..."

if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        log_success "Created .env from .env.example (customize this file with your RTSP URLs if needed)."
    else
        log_warn ".env.example not found. Please create a .env file if RTSP streaming is required."
    fi
else
    log_info ".env file already exists."
fi

# ------------------------------------------------------------------------------
# 5. Required Runtime Directories
# ------------------------------------------------------------------------------
log_step "Creating runtime directories..."

DIRS_TO_CREATE=(
    "models"
    "data/live"
    "evidence"
    "outputs"
    "test_images"
)

for dir_path in "${DIRS_TO_CREATE[@]}"; do
    mkdir -p "${REPO_ROOT}/${dir_path}"
done
log_success "Runtime directories verified: ${DIRS_TO_CREATE[*]}"

# ------------------------------------------------------------------------------
# 6. Model Checkpoint Setup (best.pt)
# ------------------------------------------------------------------------------
if [[ "${SKIP_MODEL}" = false ]]; then
    log_step "Checking PPE detection model checkpoint..."

    MODEL_PATH="${REPO_ROOT}/models/best.pt"
    if [[ -f "${MODEL_PATH}" ]]; then
        log_success "Model checkpoint exists: models/best.pt ($(du -h "${MODEL_PATH}" 2>/dev/null | cut -f1 || ls -lh "${MODEL_PATH}" | awk '{print $5}'))"
    else
        log_info "models/best.pt not found. Attempting to download checkpoint from Hugging Face..."
        python - << 'EOF'
import sys
import shutil
from pathlib import Path

repo_root = Path.cwd()
model_target = repo_root / "models" / "best.pt"
repo_id = "Hansung-Cho/yolov8-ppe-detection"
filename = "best.pt"

try:
    from huggingface_hub import hf_hub_download
    print(f"Downloading {filename} from {repo_id}...")
    cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached_path, model_target)
    print(f"Successfully saved checkpoint to: {model_target}")
except Exception as e:
    print(f"WARNING: Automatic download failed: {e}", file=sys.stderr)
    print("Please copy the tested checkpoint manually to models/best.pt.", file=sys.stderr)
    sys.exit(1)
EOF
        if [[ -f "${MODEL_PATH}" ]]; then
            log_success "Model checkpoint downloaded successfully to models/best.pt"
        else
            log_warn "Model download could not be completed automatically. You can copy 'best.pt' into 'models/' manually."
        fi
    fi
else
    log_info "Skipping model setup as requested (--skip-model)."
fi

# ------------------------------------------------------------------------------
# 7. Frontend Dependencies (React / Vite)
# ------------------------------------------------------------------------------
if [[ "${SKIP_FRONTEND}" = false ]]; then
    log_step "Setting up frontend operator dashboard..."

    if [[ -d "frontend" ]]; then
        if command -v npm &>/dev/null; then
            log_info "Installing frontend dependencies with npm..."
            (
                cd frontend
                npm install --no-audit --no-fund
            )
            log_success "Frontend dependencies installed."
        else
            log_warn "Node.js/npm not detected in PATH. Frontend setup skipped."
            log_warn "To run the React dashboard, install Node.js (v18+) and run: cd frontend && npm install"
        fi
    else
        log_info "No frontend directory found, skipping."
    fi
else
    log_info "Skipping frontend setup as requested (--skip-frontend)."
fi

# ------------------------------------------------------------------------------
# 8. Optional Tests
# ------------------------------------------------------------------------------
if [[ "${RUN_TESTS}" = true ]]; then
    log_step "Running test suite (pytest)..."
    python -m pytest tests -q || {
        log_warn "Some tests failed or reported warnings. Check pytest output above."
    }
fi

# ------------------------------------------------------------------------------
# Summary & Next Steps
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}======================================================"
echo -e "            Setup Completed Successfully!             "
echo -e "======================================================${RESET}"

echo -e "\n${BOLD}Quick Start Guide:${RESET}"
echo -e "  1. Activate the environment:"
echo -e "     ${BLUE}source ${ACTIVATE_SCRIPT}${RESET}"
echo -e "     ${RESET}(On Windows CMD/PowerShell: ${BLUE}.venv\\\\Scripts\\\\activate${RESET})"
echo -e ""
echo -e "  2. Start the Backend API (FastAPI):"
echo -e "     ${BLUE}python -m src.api${RESET}               # http://127.0.0.1:8000"
echo -e "     ${BLUE}MOCK_DATA=true python -m src.api${RESET}   # with simulated stream"
echo -e ""
echo -e "  3. Start the Frontend Dashboard (React/Vite):"
echo -e "     ${BLUE}cd frontend && npm run dev${RESET}       # http://127.0.0.1:5173"
echo -e ""
echo -e "  4. Run inference CLI:"
echo -e "     ${BLUE}python -m src.main --mode image --source test_images/test.jpg --save-output outputs/test_image.jpg${RESET}"
echo -e ""
echo -e "  5. Run test suite:"
echo -e "     ${BLUE}python -m pytest tests -q${RESET}"
echo -e ""
