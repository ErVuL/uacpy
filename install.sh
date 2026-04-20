#!/usr/bin/env bash
#
# install.sh - Build and install all UACPY native model binaries
#
# Compiles the following acoustic propagation models:
#   - OALIB (Acoustics-Toolbox): Kraken, KrakenField, Scooter, SPARC, Bellhop (Fortran)
#   - BellhopCUDA: C++/CUDA Bellhop (optional, for GPU acceleration)
#   - OASES: Ocean Acoustics and Seismics (best-effort)
#   - mpiramS: Parabolic Equation model (broadband PE)
#
# Layout produced:
#   uacpy/bin/oalib/       — Kraken, Scooter, SPARC, Bellhop (Fortran)
#   uacpy/bin/bellhopcuda/ — BellhopCXX / BellhopCUDA (C++/CUDA)
#   uacpy/bin/oases/       — OASES suite
#   uacpy/bin/mpirams/     — mpiramS PE model
#
# By default runs interactively. Use -y/--yes for non-interactive mode.
#
set -euo pipefail

# -------------------------
# Colors for pretty output
# -------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -------------------------
# Directories & defaults
# -------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OALIB_DIR="${SCRIPT_DIR}/uacpy/third_party/Acoustics-Toolbox"   # Fortran suite
BHC_DIR="${SCRIPT_DIR}/uacpy/third_party/bellhopcuda"           # C++/CUDA bellhop
OASES_DIR="${SCRIPT_DIR}/uacpy/third_party/oases"               # OASES
MPIRAMS_DIR="${SCRIPT_DIR}/uacpy/third_party/mpiramS"           # mpiramS PE
BIN_ROOT="${SCRIPT_DIR}/uacpy/bin"

BIN_DIR_OALIB="${BIN_ROOT}/oalib"
BIN_DIR_BELLHOP="${BIN_ROOT}/bellhopcuda"
BIN_DIR_OASES="${BIN_ROOT}/oases"
BIN_DIR_MPIRAMS="${BIN_ROOT}/mpirams"

# Default behavior: interactive
AUTO_YES=0         # 0 = interactive (prompt the user); 1 = assume "yes"
FORCE=0
BELLHOP_VERSION="" # "fortran", "cxx", or "cuda" (empty => prompt/auto)

# OpenMP default (user requested yes)
ENABLE_OPENMP=1

# -------------------------
# Helpers
# -------------------------
command_exists() {
    command -v "$1" &> /dev/null
}

run_sudo() {
    if [[ "$(id -u)" -ne 0 ]]; then
        sudo "$@"
    else
        "$@"
    fi
}

get_nproc() {
    nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2
}

# Prompt function honors AUTO_YES
prompt_yes_no() {
    # $1 = question text
    if [[ $AUTO_YES -eq 1 ]]; then
        return 0
    fi
    while true; do
        read -r -p "$1 [y/N]: " ans
        case "$ans" in
            [Yy]* ) return 0 ;;
            [Nn]*|"" ) return 1 ;;
            * ) echo "Please answer y or n." ;;
        esac
    done
}

install_package() {
    # $1 package name (best-effort; different package managers may differ)
    pkg="$1"
    case "$PACKAGE_MANAGER" in
        apt)
            run_sudo apt-get update -y
            run_sudo apt-get install -y "$pkg"
            ;;
        dnf)
            run_sudo dnf install -y "$pkg"
            ;;
        yum)
            run_sudo yum install -y "$pkg"
            ;;
        pacman)
            run_sudo pacman -S --noconfirm "$pkg"
            ;;
        brew)
            brew install "$pkg"
            ;;
        *)
            echo -e "${YELLOW}Unknown package manager; please install $pkg manually.${NC}"
            return 1
            ;;
    esac
}

# -------------------------
# Parse CLI args
# -------------------------
print_help() {
    cat <<EOF
Usage: $0 [options]

Builds all native acoustic model binaries for UACPY.
By default, runs interactively and prompts for options.

Options:
  -y, --yes            Non-interactive mode - assume yes and auto-detect
  --force              Force rebuild even if binaries exist
  --bellhop [fortran|cxx|cuda]
                       Pre-select Bellhop variant:
                         fortran — Original Fortran Bellhop (part of OALIB)
                         cxx     — C++ Bellhop (CPU, requires CMake)
                         cuda    — CUDA Bellhop (GPU, requires nvcc + CMake)
  -h, --help           Show this help

Examples:
  ./install.sh                     # Interactive: prompts for choices
  ./install.sh -y                  # Auto-detect everything, no prompts
  ./install.sh --bellhop cuda      # Use CUDA Bellhop, prompt for rest
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)
            AUTO_YES=1
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --bellhop)
            shift
            if [[ $# -gt 0 ]]; then
                BELLHOP_VERSION="$1"
                shift
            else
                echo -e "${RED}--bellhop requires an argument: fortran|cxx|cuda${NC}"
                exit 1
            fi
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Warning: Unknown argument: $1${NC}"
            shift
            ;;
    esac
done

# -------------------------
# OS detection
# -------------------------
OS=""
PACKAGE_MANAGER=""
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    if command_exists apt-get; then
        PACKAGE_MANAGER="apt"
    elif command_exists dnf; then
        PACKAGE_MANAGER="dnf"
    elif command_exists yum; then
        PACKAGE_MANAGER="yum"
    elif command_exists pacman; then
        PACKAGE_MANAGER="pacman"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    PACKAGE_MANAGER="brew"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  UACPY Model Installer${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Detected OS: ${GREEN}$OS${NC}"
if [[ $AUTO_YES -eq 1 ]]; then
    echo -e "Mode: ${YELLOW}Non-interactive (auto-detect)${NC}"
else
    echo -e "Mode: ${GREEN}Interactive${NC}"
fi
echo ""

# -------------------------
# Choose Bellhop variant
# -------------------------
choose_bellhop() {
    if [[ -n "$BELLHOP_VERSION" ]]; then
        case "$BELLHOP_VERSION" in
            fortran|cxx|cuda) return 0 ;;
            *)
                echo -e "${YELLOW}Invalid --bellhop argument: ${BELLHOP_VERSION}. Ignoring.${NC}"
                BELLHOP_VERSION=""
                ;;
        esac
    fi

    # Non-interactive: auto-detect best available
    if [[ $AUTO_YES -eq 1 ]]; then
        if command_exists nvcc; then
            BELLHOP_VERSION="cuda"
        elif command_exists cmake && (command_exists g++ || command_exists clang++); then
            BELLHOP_VERSION="cxx"
        else
            BELLHOP_VERSION="fortran"
        fi
        return 0
    fi

    # interactive selection
    echo -e "${BLUE}Choose Bellhop variant to install:${NC}"
    echo ""
    echo "  1) Fortran    — Original Bellhop (reliable, no extra dependencies)"
    echo "  2) C++ (CPU)  — Faster than Fortran (requires CMake + C++ compiler)"
    if command_exists nvcc; then
        echo -e "  3) CUDA (GPU) — Fastest, runs on NVIDIA GPU ${GREEN}(nvcc detected)${NC}"
    else
        echo -e "  3) CUDA (GPU) — Fastest, runs on NVIDIA GPU ${YELLOW}(nvcc not found)${NC}"
    fi
    echo ""
    read -r -p "Enter choice [1]: " ch
    ch="${ch:-1}"
    case "$ch" in
        1) BELLHOP_VERSION="fortran" ;;
        2) BELLHOP_VERSION="cxx" ;;
        3) BELLHOP_VERSION="cuda" ;;
        *) echo -e "${YELLOW}Invalid choice; defaulting to Fortran${NC}"; BELLHOP_VERSION="fortran" ;;
    esac
}

choose_bellhop

echo -e "Selected Bellhop variant: ${GREEN}${BELLHOP_VERSION}${NC}"
echo ""

# -------------------------
# Check and install prerequisites
# -------------------------

# gfortran
if ! command_exists gfortran; then
    echo -e "${YELLOW}gfortran not found.${NC}"
    if [[ "$OS" == "macOS" ]]; then
        if ! command_exists brew; then
            if prompt_yes_no "Homebrew not found. Install Homebrew?"; then
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            else
                echo -e "${RED}Please install gfortran (via Homebrew) and re-run.${NC}"
                exit 1
            fi
        fi
        echo -e "${BLUE}Installing gcc (includes gfortran) via Homebrew...${NC}"
        brew install gcc
        # Homebrew may produce gfortran-<ver> but not plain 'gfortran'; try to symlink where possible
        if ! command_exists gfortran; then
            gf_bin="$(command -v gfortran-14 || command -v gfortran-13 || command -v gfortran-12 || true)"
            if [[ -n "$gf_bin" ]]; then
                if [[ -w "$(dirname "$gf_bin")" ]]; then
                    ln -sf "$gf_bin" "$(dirname "$gf_bin")/gfortran" || true
                else
                    if prompt_yes_no "Create /usr/local/bin/gfortran symlink to $gf_bin (requires sudo)?"; then
                        run_sudo ln -sf "$gf_bin" /usr/local/bin/gfortran || true
                    fi
                fi
            fi
        fi
    else
        # linux package manager
        case "$PACKAGE_MANAGER" in
            apt)
                run_sudo apt-get update -y
                run_sudo apt-get install -y gfortran make
                ;;
            dnf)
                run_sudo dnf install -y gcc-gfortran make
                ;;
            yum)
                run_sudo yum install -y gcc-gfortran make
                ;;
            pacman)
                run_sudo pacman -S --noconfirm gcc-fortran make
                ;;
            *)
                echo -e "${RED}Could not detect package manager. Please install gfortran manually.${NC}"
                exit 1
                ;;
        esac
    fi

    if ! command_exists gfortran; then
        echo -e "${RED}gfortran still not found after attempted install. Aborting.${NC}"
        exit 1
    fi
fi
echo -e "✓ gfortran: ${GREEN}$(gfortran --version | head -n1)${NC}"

# make
if ! command_exists make; then
    echo -e "${YELLOW}make not found.${NC}"
    if [[ "$OS" == "macOS" ]]; then
        echo "Installing Xcode Command Line Tools (contains make)..."
        xcode-select --install || true
        echo "Please finish Xcode install if prompted and re-run."
        exit 0
    else
        install_package make
    fi
fi
echo -e "✓ make: ${GREEN}$(make --version | head -n1)${NC}"

# CMake (required for bellhopcxx/bellhopcuda)
check_cmake() {
    if command_exists cmake; then
        echo -e "✓ cmake: ${GREEN}$(cmake --version | head -n1)${NC}"
        return 0
    fi
    echo -e "${YELLOW}cmake not found.${NC}"
    if [[ "$OS" == "macOS" ]]; then
        if prompt_yes_no "Install cmake via Homebrew?"; then
            brew install cmake
        else
            return 1
        fi
    else
        if prompt_yes_no "Install cmake using package manager?"; then
            install_package cmake
        else
            return 1
        fi
    fi
    command_exists cmake
}

# C++ compiler
check_cxx_compiler() {
    if command_exists g++; then
        echo -e "✓ g++: ${GREEN}$(g++ --version | head -n1)${NC}"
        return 0
    elif command_exists clang++; then
        echo -e "✓ clang++: ${GREEN}$(clang++ --version | head -n1)${NC}"
        return 0
    fi

    echo -e "${YELLOW}C++ compiler not found.${NC}"
    if [[ "$OS" == "macOS" ]]; then
        brew install gcc
    else
        if prompt_yes_no "Install C++ compiler via package manager?"; then
            case "$PACKAGE_MANAGER" in
                apt) run_sudo apt-get update -y; run_sudo apt-get install -y g++;;
                dnf|yum) run_sudo $PACKAGE_MANAGER install -y gcc-c++;;
                pacman) run_sudo pacman -S --noconfirm gcc;;
                *) echo -e "${RED}Please install g++ or clang++ manually${NC}"; return 1;;
            esac
        else
            return 1
        fi
    fi
    command_exists g++ || command_exists clang++
}

# CUDA
check_cuda() {
    if command_exists nvcc; then
        cuda_ver=$(nvcc --version | grep -o -E "release [0-9]+\.[0-9]+" | head -n1 || true)
        echo -e "✓ nvcc found: ${GREEN}${cuda_ver}${NC}"
        return 0
    fi
    echo -e "${YELLOW}CUDA (nvcc) not found.${NC}"
    return 1
}

# GLM (submodule) detection
check_glm() {
    # check common system install first
    if command_exists pkg-config && pkg-config --exists glm 2>/dev/null; then
        echo -e "✓ glm found via pkg-config"
        return 0
    fi
    # look for the glm submodule in bellhopcuda repo
    if [ -f "$BHC_DIR/glm/glm/glm.hpp" ]; then
        echo -e "✓ glm submodule present"
        return 0
    fi
    # try to initialize submodules if available
    if [ -d "$BHC_DIR" ]; then
        if prompt_yes_no "GLM not found. Initialize git submodules for bellhopcuda?"; then
            (cd "$SCRIPT_DIR" && git submodule update --init --recursive) || {
                echo -e "${RED}git submodule update failed${NC}"
                return 1
            }
            if [ -f "$BHC_DIR/glm/glm/glm.hpp" ]; then
                echo -e "${GREEN}✓ glm submodule initialized${NC}"
                return 0
            fi
        fi
    fi
    return 1
}

# Pre-checks for build paths
if [ ! -d "$OALIB_DIR" ]; then
    echo -e "${RED}Acoustics-Toolbox (OALIB) not found at:${NC}"
    echo "  $OALIB_DIR"
    echo "Please place the Acoustics-Toolbox in uacpy/third_party/Acoustics-Toolbox"
    exit 1
fi
echo -e "✓ Found OALIB (Acoustics-Toolbox): ${GREEN}$OALIB_DIR${NC}"

if [[ "$BELLHOP_VERSION" == "cxx" || "$BELLHOP_VERSION" == "cuda" ]]; then
    if [ ! -d "$BHC_DIR" ]; then
        echo -e "${RED}bellhopcuda directory not found at:${NC}"
        echo "  $BHC_DIR"
        exit 1
    fi
    echo -e "✓ Found bellhopcuda repo: ${GREEN}$BHC_DIR${NC}"
fi

# If building CXX/CUDA, ensure cmake and C++ compiler and glm are available
if [[ "$BELLHOP_VERSION" == "cxx" || "$BELLHOP_VERSION" == "cuda" ]]; then
    if ! check_cmake; then
        echo -e "${RED}CMake required. Aborting.${NC}"
        exit 1
    fi
    if ! check_cxx_compiler; then
        echo -e "${RED}C++ compiler required. Aborting.${NC}"
        exit 1
    fi
    if [[ "$BELLHOP_VERSION" == "cuda" ]]; then
        if ! check_cuda; then
            echo -e "${YELLOW}CUDA not present; falling back to C++/CPU bellhop (cxx).${NC}"
            BELLHOP_VERSION="cxx"
        fi
    fi
    if ! check_glm; then
        echo -e "${RED}GLM dependency missing for bellhopcuda. Aborting.${NC}"
        exit 1
    fi
fi

echo ""

# -------------------------
# Make bin dirs
# -------------------------
mkdir -p "$BIN_DIR_OALIB"
mkdir -p "$BIN_DIR_OASES"
mkdir -p "$BIN_DIR_BELLHOP"
mkdir -p "$BIN_DIR_MPIRAMS"

# -------------------------
# Build bellhopcxx / bellhopcuda (if selected)
# -------------------------
if [[ "$BELLHOP_VERSION" == "cxx" || "$BELLHOP_VERSION" == "cuda" ]]; then
    echo -e "${BLUE}=== Building Bellhop (${BELLHOP_VERSION}) ===${NC}"
    cd "$BHC_DIR"

    BUILD_DIR="$BHC_DIR/build"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=Release -DBHC_ENABLE_TESTS=OFF"
    if [[ "$BELLHOP_VERSION" == "cuda" ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBHC_ENABLE_CUDA=ON"
        echo -e "  - CUDA support: ON"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBHC_ENABLE_CUDA=OFF"
        echo -e "  - CUDA support: OFF (CPU build)"
    fi

    echo -e "${BLUE}Configuring CMake...${NC}"
    cmake -S "$BHC_DIR" -B "$BUILD_DIR" $CMAKE_OPTIONS

    echo -e "${BLUE}Building (this may take a while)...${NC}"
    NPROC=$(get_nproc)
    cmake --build "$BUILD_DIR" --config Release -j"$NPROC"

    echo -e "${GREEN}✓ Bellhop build finished${NC}"
    echo ""
fi

# -------------------------
# Configure OALIB (Fortran) build
# -------------------------
echo -e "${BLUE}=== Configuring OALIB (Acoustics-Toolbox) ===${NC}"

if [ ! -f "$OALIB_DIR/Makefile.orig" ]; then
    cp "$OALIB_DIR/Makefile" "$OALIB_DIR/Makefile.orig" || true
fi

# Compose Makefile.local with OpenMP enabled if requested
# Use single export lines for portability to /bin/sh invoked make
OALIB_MAKELOCAL="$OALIB_DIR/Makefile.local"
cat > "$OALIB_MAKELOCAL" <<'EOF'
# Auto-generated Makefile.local for OALIB (Acoustics-Toolbox)
export FC=gfortran
# OpenMP enabled and single-export FFLAGS
export FFLAGS="-march=native -O2 -ffast-math -funroll-loops -fomit-frame-pointer -mtune=native -I../misc -I../tslib -fopenmp"
export RM=rm
export CC=gcc
export CFLAGS=-g -fopenmp
export LAPACK_LIBS=-llapack
EOF

echo -e "✓ Wrote $OALIB_MAKELOCAL"
echo ""

# -------------------------
# Build OALIB Fortran models
# -------------------------
echo -e "${BLUE}=== Building OALIB (Fortran models) ===${NC}"
cd "$OALIB_DIR"

echo -e "${YELLOW}Cleaning previous builds...${NC}"
make clean 2>/dev/null || true

NPROC=$(get_nproc)

if [[ "$BELLHOP_VERSION" == "fortran" ]]; then
    echo -e "  - Building OALIB including Fortran Bellhop..."
    make all -j"$NPROC"
else
    echo -e "  - Building OALIB core components (Kraken, KrakenField, Scooter, SPARC)..."
    make -C Kraken -j"$NPROC"
    make -C KrakenField -j"$NPROC"
    make -C Scooter -j"$NPROC"
fi

echo -e "${GREEN}✓ OALIB build finished${NC}"
echo ""

# -------------------------
# Build OASES (best-effort)
# -------------------------
OASES_URL="http://acoustics.mit.edu/faculty/henrik/LAMSS/pub/Oases/oases.tar.gz"

echo -e "${BLUE}=== Building OASES (best-effort) ===${NC}"
if [ ! -d "$OASES_DIR" ]; then
    echo -e "${YELLOW}OASES source not found at: $OASES_DIR${NC}"
    echo -e "${BLUE}OASES is distributed separately by MIT and is not bundled with UACPY.${NC}"
    if prompt_yes_no "Would you like to download OASES from $OASES_URL?"; then
        echo -e "${BLUE}Downloading OASES...${NC}"
        OASES_TMP="$(mktemp -d)"
        if curl -fSL "$OASES_URL" -o "$OASES_TMP/oases.tar.gz"; then
            tar -xzf "$OASES_TMP/oases.tar.gz" -C "$OASES_TMP"
            # The tarball extracts to Oases_export/; rename to match expected path
            if [ -d "$OASES_TMP/Oases_export" ]; then
                mv "$OASES_TMP/Oases_export" "$OASES_DIR"
                echo -e "${GREEN}✓ OASES source downloaded and placed at $OASES_DIR${NC}"
            else
                echo -e "${RED}✗ Unexpected archive structure. Skipping OASES.${NC}"
            fi
        else
            echo -e "${RED}✗ Failed to download OASES. Skipping.${NC}"
        fi
        rm -rf "$OASES_TMP"
    else
        echo -e "${YELLOW}Skipping OASES installation.${NC}"
    fi
fi

if [ -d "$OASES_DIR" ]; then
    cd "$OASES_DIR"

    # Do not overwrite shell OSTYPE; use OASES-specific variables
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OASES_HOSTTYPE="i386"
        OASES_OSTYPE="linux-linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OASES_HOSTTYPE="i386"
        OASES_OSTYPE="darwin-darwin"
    else
        OASES_HOSTTYPE="$(uname -m)"
        OASES_OSTYPE="${OSTYPE//[^[:alnum:]\.-]/}"
    fi

    export OASES_ROOT="$OASES_DIR"
    export OASES_BIN="$OASES_DIR/bin/${OASES_HOSTTYPE}-${OASES_OSTYPE}"
    export OASES_LIB="$OASES_DIR/lib/${OASES_HOSTTYPE}-${OASES_OSTYPE}"

    mkdir -p "$OASES_BIN" "$OASES_LIB" "$OASES_DIR/src/${OASES_HOSTTYPE}-${OASES_OSTYPE}"

    echo -e "${BLUE}Starting OASES make (log -> /tmp/oases_build.log)${NC}"
    export FC_STM="gfortran"
    export FFLGS="-O2 -std=legacy -fallow-argument-mismatch"

    set +e
    make OASES_ROOT="$OASES_DIR" oases 2>&1 | tee /tmp/oases_build.log
    OASES_STATUS=${PIPESTATUS[0]:-1}
    set -e

    if [[ $OASES_STATUS -eq 0 ]]; then
        echo -e "${GREEN}✓ OASES build completed${NC}"
    else
        echo -e "${YELLOW}⚠ OASES build had issues. See /tmp/oases_build.log${NC}"
    fi

    # Try to copy installed binaries (if any)
    OASES_INSTALLED=0
    POSSIBLE_OASES_BINS=(oasn2_bin oast2 oasr2 oasp2 oass2 oassp2)
    mkdir -p "$BIN_DIR_OASES"
    for b in "${POSSIBLE_OASES_BINS[@]}"; do
        if [ -f "$OASES_BIN/$b" ]; then
            cp "$OASES_BIN/$b" "$BIN_DIR_OASES/$b"
            chmod +x "$BIN_DIR_OASES/$b"
            echo -e "  ✓ Installed OASES binary: ${GREEN}$b${NC}"
            ((OASES_INSTALLED++))
        fi
    done

    # Create compatibility symlinks if appropriate
    (cd "$BIN_DIR_OASES" || true; \
        [[ -f oast2 && ! -f oast ]] && ln -sf oast2 oast || true; \
        [[ -f oasp2 && ! -f oasp ]] && ln -sf oasp2 oasp || true; \
        [[ -f oasr2 && ! -f oasr ]] && ln -sf oasr2 oasr || true; \
        [[ -f oasn2_bin && ! -f oasn ]] && ln -sf oasn2_bin oasn || true; \
    ) || true

    if [[ $OASES_INSTALLED -eq 0 ]]; then
        echo -e "${YELLOW}No OASES executables installed. Check /tmp/oases_build.log${NC}"
    else
        echo -e "${GREEN}✓ Installed $OASES_INSTALLED OASES components${NC}"
    fi
fi
echo ""

# -------------------------
# Build mpiramS (Fortran PE model)
# -------------------------
echo -e "${BLUE}=== Building mpiramS (Parabolic Equation) ===${NC}"
if [ -d "$MPIRAMS_DIR" ]; then
    cd "$MPIRAMS_DIR"

    # Ensure obj/ and mod/ directories exist
    mkdir -p obj mod

    echo -e "${YELLOW}Cleaning previous mpiramS builds...${NC}"
    make clean 2>/dev/null || true

    echo -e "${BLUE}Compiling mpiramS (single-processor version, double precision)...${NC}"
    set +e
    make 2>&1 | tee /tmp/mpirams_build.log
    MPIRAMS_STATUS=${PIPESTATUS[0]:-1}
    set -e

    if [[ $MPIRAMS_STATUS -eq 0 ]] && [ -f "$MPIRAMS_DIR/s_mpiram" ]; then
        cp "$MPIRAMS_DIR/s_mpiram" "$BIN_DIR_MPIRAMS/s_mpiram"
        chmod +x "$BIN_DIR_MPIRAMS/s_mpiram"
        echo -e "${GREEN}✓ Installed mpiramS binary: s_mpiram${NC}"
    else
        echo -e "${YELLOW}⚠ mpiramS build failed. See /tmp/mpirams_build.log${NC}"
    fi
else
    echo -e "${YELLOW}mpiramS source not found at: $MPIRAMS_DIR. Skipping.${NC}"
fi
echo ""

# -------------------------
# Install executables to bin directories
# -------------------------
echo -e "${BLUE}=== Installing executables to ${BIN_ROOT} ===${NC}"
INSTALLED_COUNT=0

# Install bellhopcxx/bellhopcuda artifacts (if built)
if [[ "$BELLHOP_VERSION" == "cxx" || "$BELLHOP_VERSION" == "cuda" ]]; then
    BUILD_DIR="$BHC_DIR/build"
    SEARCH_NAMES=(bellhopcxx bellhopcxx* bellhopcuda bellhopcuda*)

    for name in "${SEARCH_NAMES[@]}"; do
        # find executables matching name patterns
        while IFS= read -r -d $'\0' file; do
            base=$(basename "$file")
            cp "$file" "$BIN_DIR_BELLHOP/$base"
            chmod +x "$BIN_DIR_BELLHOP/$base"
            echo -e "  ✓ Installed bellhop binary: ${GREEN}$base${NC}"
            ((INSTALLED_COUNT++))
        done < <(find "$BUILD_DIR" -type f -executable -name "$name" -print0 2>/dev/null || true)
    done

    if [[ $INSTALLED_COUNT -eq 0 ]]; then
        echo -e "${YELLOW}No bellhop (cxx/cuda) executables found in build tree.${NC}"
    fi
fi

# Install OALIB (Fortran) executables
OALIB_EXECUTABLES=(
    "Bellhop/bellhop.exe"
    "Bellhop/bellhop3d.exe"
    "KrakenField/field.exe"
    "Kraken/kraken.exe"
    "Kraken/krakenc.exe"
    "Kraken/bounce.exe"
    "Scooter/scooter.exe"
    "Scooter/sparc.exe"
)

for path in "${OALIB_EXECUTABLES[@]}"; do
    if [ -f "$OALIB_DIR/$path" ]; then
        bn=$(basename "$path")
        cp "$OALIB_DIR/$path" "$BIN_DIR_OALIB/$bn"
        chmod +x "$BIN_DIR_OALIB/$bn"
        echo -e "  ✓ Installed OALIB binary: ${GREEN}$bn${NC}"
        ((INSTALLED_COUNT++))
    else
        echo -e "  ${YELLOW}Not found (OALIB): $path${NC}"
    fi
done
echo ""

# Install OASES copies were already attempted above (BIN_DIR_OASES)

# Final check: at least one executable installed
if [[ $INSTALLED_COUNT -eq 0 ]]; then
    echo -e "${RED}Error: No executables were installed. Aborting.${NC}"
    exit 1
fi

# -------------------------
# Sanity tests (non-invasive)
# -------------------------
echo -e "${BLUE}=== Running quick executable sanity checks ===${NC}"

test_runnable() {
    exe="$1"
    if [ -x "$exe" ]; then
        # try --version or --help; ignore exit code
        ("$exe" --version >/dev/null 2>&1) || ("$exe" --help >/dev/null 2>&1) || true
        echo -e "  ✓ Runnable: ${GREEN}$(basename "$exe")${NC}"
        return 0
    else
        echo -e "  ${YELLOW}Not executable or missing: $(basename "$exe")${NC}"
        return 1
    fi
}

# Test a sample of installed binaries (if present)
if [ -f "$BIN_DIR_OALIB/kraken.exe" ]; then
    test_runnable "$BIN_DIR_OALIB/kraken.exe" || true
fi
if [ -f "$BIN_DIR_OALIB/bellhop.exe" ]; then
    test_runnable "$BIN_DIR_OALIB/bellhop.exe" || true
fi
# test bellhopcuda if present
if ls "$BIN_DIR_BELLHOP"/bellhop* 1> /dev/null 2>&1; then
    for f in "$BIN_DIR_BELLHOP"/bellhop*; do
        [ -x "$f" ] && test_runnable "$f" || true
        break
    done
fi
# test OASES first binary if present
if ls "$BIN_DIR_OASES" 1> /dev/null 2>&1; then
    for f in "$BIN_DIR_OASES"/*; do
        [ -x "$f" ] && test_runnable "$f" || true
        break
    done
fi
# test mpiramS
if [ -f "$BIN_DIR_MPIRAMS/s_mpiram" ]; then
    test_runnable "$BIN_DIR_MPIRAMS/s_mpiram" || true
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Installation completed${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Installed models:"
echo "  Kraken/Scooter/SPARC:  $BIN_DIR_OALIB"
if [[ "$BELLHOP_VERSION" == "fortran" ]]; then
    echo "  Bellhop (Fortran):     $BIN_DIR_OALIB"
else
    echo "  Bellhop (${BELLHOP_VERSION}):       $BIN_DIR_BELLHOP"
fi
if [ -f "$BIN_DIR_MPIRAMS/s_mpiram" ]; then
    echo "  RAM (mpiramS PE):      $BIN_DIR_MPIRAMS"
fi
if ls "$BIN_DIR_OASES"/* 1>/dev/null 2>&1; then
    echo "  OASES:                 $BIN_DIR_OASES"
fi
echo ""
echo "Quick test:"
echo "  cd uacpy && python -c \"import uacpy; print(uacpy.__version__)\""
echo "  python uacpy/examples/example_01_basic_shallow_water.py"
echo ""
exit 0

