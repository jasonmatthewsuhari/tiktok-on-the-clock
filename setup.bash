set -euo pipefail

# --- config, pls do not change :) ---
ENV_NAME="tiktok-env"
PY_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- helpers ---
log() { printf "\n\033[1m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }

# Ensure conda exists, warn the userr if it is not found.
# if this fails here, pls follow: https://docs.anaconda.com/miniconda/install/
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda not found in PATH."
  echo "Install Miniconda/Anaconda and run: conda init bash (then restart your shell)"
  exit 1
fi

# Create the conda env if it currently does not exist for the user
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  log "Creating conda env '$ENV_NAME' (Python $PY_VERSION)..."
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
else
  log "Conda env '$ENV_NAME' already exists."
fi

# Activate only if not already active
if [[ "${CONDA_DEFAULT_ENV-}" == "$ENV_NAME" ]]; then
  log "Already inside '$ENV_NAME'; skipping activation."
else
  log "Activating '$ENV_NAME'..."
  conda activate "$ENV_NAME"
fi

# Upgrade pip toolchain & install pip-tools for the pipcompile tool :)
log "Upgrading pip/setuptools/wheel and installing pip-tools..."
python -m pip install --upgrade pip setuptools wheel pip-tools

# Compile the reqs like usual
if [[ -f "$PROJECT_ROOT/requirements.in" ]]; then
  log "Compiling requirements.in -> requirements.txt ..."
  (cd "$PROJECT_ROOT" && python -m piptools compile requirements.in --resolver=backtracking)
fi

# Install from requirements.txt AFTER compiling the reqs
if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
  log "Installing from requirements.txt with pip-sync..."
  (cd "$PROJECT_ROOT" && python -m piptools sync requirements.txt)
else
  log "No requirements.txt found; skipping install."
fi

log "DONE. Active env: '$CONDA_DEFAULT_ENV'"
