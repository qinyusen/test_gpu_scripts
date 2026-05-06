#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${GPU_TOOLS_DIR:-${H200_TOOLS_DIR:-/opt/h200-test-tools}}"
JOBS="${MAKE_JOBS:-$(nproc)}"
VERBOSE="${VERBOSE:-0}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

check_root() {
    if [[ $EUID -ne 0 ]]; then
        warn "Not running as root. Some installations may fail."
        warn "Re-run with: sudo $0"
    fi
}

detect_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        fail "nvidia-smi not found. Install NVIDIA drivers first."
        exit 1
    fi
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    log "Detected GPU: $gpu_name"
}

install_system_deps() {
    log "Installing system dependencies..."
    if command -v apt-get &>/dev/null; then
        apt-get update -qq
        apt-get install -y -qq build-essential git cmake wget curl \
            openmpi-bin libopenmpi-dev openssh-client \
            infiniband-diags ibverbs-utils perftest \
            python3 python3-pip python3-venv \
            2>/dev/null || warn "Some apt packages failed (may already be installed)"
    elif command -v yum &>/dev/null; then
        yum groupinstall -y "Development Tools" 2>/dev/null || true
        yum install -y git cmake wget curl \
            openmpi openmpi-devel openssh-clients \
            infiniband-diags libibverbs-utils perftest \
            python3 python3-pip \
            2>/dev/null || warn "Some yum packages failed"
    elif command -v dnf &>/dev/null; then
        dnf groupinstall -y "Development Tools" 2>/dev/null || true
        dnf install -y git cmake wget curl \
            openmpi openmpi-devel openssh-clients \
            infiniband-diags libibverbs-utils perftest \
            python3 python3-pip \
            2>/dev/null || warn "Some dnf packages failed"
    else
        warn "Unsupported package manager. Install deps manually."
    fi
    ok "System dependencies"
}

install_python_deps() {
    log "Installing Python dependencies..."
    pip3 install --quiet rich pyyaml 2>/dev/null || pip install --quiet rich pyyaml
    ok "Python dependencies (rich, pyyaml)"
}

install_nvbandwidth() {
    log "Installing nvbandwidth..."
    local src="$INSTALL_DIR/nvbandwidth"
    if [[ -x "$src/nvbandwidth" ]]; then
        ok "nvbandwidth already installed at $src/nvbandwidth"
        return
    fi

    mkdir -p "$INSTALL_DIR"
    git clone --depth 1 https://github.com/NVIDIA/nvbandwidth.git "$src" 2>/dev/null
    cd "$src"
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>/dev/null
    make -j"$JOBS" 2>/dev/null
    if [[ -x "$src/build/nvbandwidth" ]]; then
        cp "$src/build/nvbandwidth" "$src/nvbandwidth"
        ok "nvbandwidth installed at $src/nvbandwidth"
    else
        warn "nvbandwidth build failed. Try building manually in $src"
    fi
}

install_nccl_tests() {
    log "Installing nccl-tests..."
    local src="$INSTALL_DIR/nccl-tests"
    if [[ -x "$src/build/all_reduce_perf" ]]; then
        ok "nccl-tests already installed at $src/build/"
        return
    fi

    mkdir -p "$INSTALL_DIR"
    git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git "$src" 2>/dev/null
    cd "$src"

    local cuda_home="${CUDA_HOME:-/usr/local/cuda}"
    if [[ ! -d "$cuda_home" ]]; then
        warn "CUDA_HOME not found at $cuda_home. Set CUDA_HOME env var."
        return
    fi

    make MPI=1 MPI_HOME=/usr -j"$JOBS" 2>/dev/null || \
        make CUDA_HOME="$cuda_home" -j"$JOBS" 2>/dev/null || \
        warn "nccl-tests build failed. Try: cd $src && make MPI=1"

    if [[ -x "$src/build/all_reduce_perf" ]]; then
        ok "nccl-tests installed at $src/build/"
    else
        warn "nccl-tests build incomplete"
    fi
}

install_gpu_burn() {
    log "Installing gpu-burn..."
    local src="$INSTALL_DIR/gpu-burn"
    if [[ -x "$src/gpu_burn" ]]; then
        ok "gpu-burn already installed at $src/gpu_burn"
        return
    fi

    mkdir -p "$INSTALL_DIR"
    git clone --depth 1 https://github.com/wilicc/gpu-burn.git "$src" 2>/dev/null
    cd "$src"
    make -j"$JOBS" 2>/dev/null || warn "gpu-burn build failed"
    if [[ -x "$src/gpu_burn" ]]; then
        ok "gpu-burn installed at $src/gpu_burn"
    else
        warn "gpu-burn build incomplete"
    fi
}

check_dcgm() {
    log "Checking DCGM..."
    if command -v nv-hostengine &>/dev/null || command -v dcgmi &>/dev/null; then
        ok "DCGM already installed"
        return
    fi
    if dpkg -l datacenter-gpu-manager &>/dev/null 2>&1; then
        ok "DCGM package installed"
        return
    fi
    warn "DCGM not found. Install from: https://docs.nvidia.com/datacenter/dcgm/latest/installation-guide.html"
    warn "  Ubuntu: sudo apt install datacenter-gpu-manager"
    warn "  Or:     curl -fsSL https://deb.nvidia.com/datacenter-gpu-manager/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-dcgm.gpg"
}

check_rdma_tools() {
    log "Checking RDMA tools..."
    local found=0
    for tool in ib_write_bw ib_read_bw ib_write_lat ib_read_lat ibstat ibv_devinfo; do
        if command -v "$tool" &>/dev/null; then
            found=$((found + 1))
        else
            warn "  $tool not found (install: perftest infiniband-diags)"
        fi
    done
    if [[ $found -gt 0 ]]; then
        ok "$found/$RDMA_TOOL_COUNT RDMA tools found" 2>/dev/null || ok "Some RDMA tools found"
    fi
}

print_summary() {
    echo ""
    echo "=========================================="
    echo " GPU Test Suite - Installation Summary"
    echo "=========================================="
    echo ""
    echo " Install directory: $INSTALL_DIR"
    echo ""
    echo " Tools status:"

    for tool_path in \
        "$INSTALL_DIR/nvbandwidth/nvbandwidth:nvbandwidth" \
        "$INSTALL_DIR/nccl-tests/build/all_reduce_perf:nccl-tests" \
        "$INSTALL_DIR/gpu-burn/gpu_burn:gpu-burn"; do
        path="${tool_path%%:*}"
        name="${tool_path##*:}"
        if [[ -x "$path" ]]; then
            echo -e "  [${GREEN}✓${NC}] $name"
        else
            echo -e "  [${YELLOW}?${NC}] $name (not built)"
        fi
    done

    echo ""
    echo " System tools:"
    for cmd in nvidia-smi mpirun nvbandwidth ib_write_bw dcgmi; do
        if command -v "$cmd" &>/dev/null; then
            echo -e "  [${GREEN}✓${NC}] $cmd"
        else
            echo -e "  [${YELLOW}-${NC}] $cmd (not found)"
        fi
    done

    echo ""
    echo " Usage:"
    echo "   python3 h200_tester.py              # Interactive menu"
    echo "   python3 h200_tester.py --test all    # Full suite"
    echo ""
}

main() {
    echo ""
    echo "=========================================="
    echo " GPU Test Suite - Dependency Installer"
    echo "=========================================="
    echo ""

    check_root
    detect_gpu

    mkdir -p "$INSTALL_DIR"

    install_system_deps
    install_python_deps
    install_nvbandwidth
    install_nccl_tests
    install_gpu_burn
    check_dcgm
    check_rdma_tools

    print_summary
}

main "$@"
