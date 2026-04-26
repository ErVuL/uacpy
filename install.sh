#!/usr/bin/env bash
#
# install.sh - Build and install all UACPY native model binaries
#
# Compiles the following acoustic propagation models:
#   - OALIB (Acoustics-Toolbox): Kraken, KrakenField, Scooter, SPARC, Bellhop (Fortran)
#   - BellhopCUDA: C++/CUDA Bellhop (optional, for GPU acceleration)
#   - OASES: Ocean Acoustics and Seismics (optional, downloaded from MIT)
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
INSTALL_OASES=""   # "yes" or "no" (empty => prompt/auto)

# OpenMP default (user requested yes)
ENABLE_OPENMP=1

# -------------------------
# Helpers
# -------------------------
command_exists() {
    command -v "$1" &> /dev/null
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

# install.sh is a pure builder — it does not install system packages itself.
# When a dependency is missing we print the per-OS install command and exit so
# the user (or CI image) can provision once and rerun. The full per-OS list
# also lives in README.md → "Install dependencies".
fail_missing() {
    # $1 = tool / library name, $2 = pre-formatted install hint
    echo -e "${RED}Missing dependency: $1${NC}" >&2
    echo -e "${YELLOW}$2${NC}" >&2
    echo -e "See README.md → 'Install dependencies' for the full per-OS list." >&2
    exit 1
}

# Render an install hint for the detected package manager.
# Args: $1 apt-pkg  $2 dnf/yum-pkg  $3 pacman-pkg  $4 brew-pkg
hint_for() {
    case "$PACKAGE_MANAGER" in
        apt)    echo "Install with: sudo apt-get install -y $1" ;;
        dnf)    echo "Install with: sudo dnf install -y $2" ;;
        yum)    echo "Install with: sudo yum install -y $2" ;;
        pacman) echo "Install with: sudo pacman -S --noconfirm $3" ;;
        brew)   echo "Install with: brew install $4" ;;
        *)      echo "Install one of: $1 (apt) / $2 (dnf) / $3 (pacman) / $4 (brew)" ;;
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
                       Pre-select which Bellhop variant to build *in addition
                       to* Fortran Bellhop. Fortran Bellhop is always built as
                       the reference implementation.
                         fortran — Fortran only (no C++/CUDA build)
                         cxx     — Fortran + C++ Bellhop (CPU, requires CMake)
                         cuda    — Fortran + CUDA Bellhop (GPU, needs nvcc + CMake)
  --oases [yes|no]     Download and build OASES (MIT, distributed separately):
                         yes — Download from MIT and build OAST/OASN/OASR/OASP
                         no  — Skip OASES entirely
  -h, --help           Show this help

Examples:
  ./install.sh                     # Interactive: prompts for choices
  ./install.sh -y                  # Auto-detect everything, no prompts
  ./install.sh --bellhop cuda      # Use CUDA Bellhop, prompt for rest
  ./install.sh --oases no          # Skip OASES, prompt for rest
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
        --oases)
            shift
            if [[ $# -gt 0 ]]; then
                INSTALL_OASES="$1"
                shift
            else
                echo -e "${RED}--oases requires an argument: yes|no${NC}"
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
    # Fortran Bellhop is always built as the reference implementation; this
    # prompt only controls whether to *also* build a C++/CUDA variant.
    echo -e "${BLUE}Fortran Bellhop will be built.${NC}"
    echo "Additionally build a C++/CUDA variant?"
    echo ""
    echo "  1) No           — Fortran only"
    echo "  2) Yes, C++ CPU — Also build bellhopcxx (requires CMake + C++ compiler)"
    if command_exists nvcc; then
        echo -e "  3) Yes, CUDA    — Also build bellhopcuda (GPU) ${GREEN}(nvcc detected)${NC}"
    else
        echo -e "  3) Yes, CUDA    — Also build bellhopcuda (GPU) ${YELLOW}(nvcc not found)${NC}"
    fi
    echo ""
    read -r -p "Enter choice [1]: " ch
    ch="${ch:-1}"
    case "$ch" in
        1) BELLHOP_VERSION="fortran" ;;
        2) BELLHOP_VERSION="cxx" ;;
        3) BELLHOP_VERSION="cuda" ;;
        *) echo -e "${YELLOW}Invalid choice; defaulting to Fortran only${NC}"; BELLHOP_VERSION="fortran" ;;
    esac
}

choose_bellhop

echo -e "Selected Bellhop variant: ${GREEN}${BELLHOP_VERSION}${NC}"
echo ""

# -------------------------
# Choose whether to install OASES
# -------------------------
# OASES is not redistributable with UACPY and must be downloaded from MIT.
# We ask up-front so the user can opt out before any network traffic or build
# time is spent.
choose_oases() {
    if [[ -n "$INSTALL_OASES" ]]; then
        case "$INSTALL_OASES" in
            yes|no) return 0 ;;
            *)
                echo -e "${YELLOW}Invalid --oases argument: ${INSTALL_OASES}. Ignoring.${NC}"
                INSTALL_OASES=""
                ;;
        esac
    fi

    # If the source is already present locally, default to yes (don't waste
    # the existing checkout). Otherwise default to yes in -y mode as well,
    # preserving the previous non-interactive behavior.
    if [[ $AUTO_YES -eq 1 ]]; then
        INSTALL_OASES="yes"
        return 0
    fi

    echo -e "${BLUE}Install OASES (OAST, OASN, OASR, OASP)?${NC}"
    echo "  OASES is distributed separately by MIT and is not bundled with UACPY."
    echo "  Saying yes will download the source from acoustics.mit.edu and build it."
    echo ""
    if prompt_yes_no "Download and install OASES?"; then
        INSTALL_OASES="yes"
    else
        INSTALL_OASES="no"
    fi
}

choose_oases

if [[ "$INSTALL_OASES" == "yes" ]]; then
    echo -e "OASES: ${GREEN}will be installed${NC}"
else
    echo -e "OASES: ${YELLOW}skipped${NC}"
fi
echo ""

# -------------------------
# Check and install prerequisites
# -------------------------

# gfortran (always required: OALIB, mpiramS, OASES are all Fortran)
if ! command_exists gfortran; then
    fail_missing "gfortran" \
        "$(hint_for gfortran gcc-gfortran gcc-fortran gcc)"
fi
echo -e "✓ gfortran: ${GREEN}$(gfortran --version | head -n1)${NC}"

# make (always required)
if ! command_exists make; then
    if [[ "$OS" == "macOS" ]]; then
        fail_missing "make" \
            "Install Xcode Command Line Tools: xcode-select --install"
    else
        fail_missing "make" "$(hint_for make make make make)"
    fi
fi
echo -e "✓ make: ${GREEN}$(make --version | head -n1)${NC}"

# CMake (required for bellhopcxx/bellhopcuda only — not for the Fortran build)
check_cmake() {
    if command_exists cmake; then
        echo -e "✓ cmake: ${GREEN}$(cmake --version | head -n1)${NC}"
        return 0
    fi
    fail_missing "cmake" "$(hint_for cmake cmake cmake cmake)"
}

# C++ compiler (required for bellhopcxx/bellhopcuda only)
check_cxx_compiler() {
    if command_exists g++; then
        echo -e "✓ g++: ${GREEN}$(g++ --version | head -n1)${NC}"
        return 0
    elif command_exists clang++; then
        echo -e "✓ clang++: ${GREEN}$(clang++ --version | head -n1)${NC}"
        return 0
    fi
    fail_missing "C++ compiler (g++ or clang++)" \
        "$(hint_for g++ gcc-c++ gcc gcc)"
}

# git (required when fetching GLM for bellhopcxx/bellhopcuda)
check_git() {
    if command_exists git; then
        return 0
    fi
    fail_missing "git" "$(hint_for git git git git)"
}

# curl (required only if downloading OASES)
check_curl() {
    if command_exists curl; then
        return 0
    fi
    fail_missing "curl" "$(hint_for curl curl curl curl)"
}

# LAPACK (required: Kraken/Scooter link with -llapack)
# - Linux: probe ldconfig for liblapack.so*
# - macOS: probe Homebrew's lapack formula. We do NOT fall back to Accelerate
#   because the OALIB Makefiles call -llapack literally; pointing at brew's
#   prefix is the simplest fix that keeps the upstream Makefiles untouched.
# Side effect: sets LAPACK_LIBS in the global scope to the right link flags.
check_lapack() {
    if [[ "$OS" == "macOS" ]]; then
        if command_exists brew && brew list lapack &>/dev/null; then
            local prefix
            prefix="$(brew --prefix lapack 2>/dev/null || true)"
            if [[ -n "$prefix" && -f "$prefix/lib/liblapack.dylib" ]]; then
                LAPACK_LIBS="-L${prefix}/lib -llapack"
                echo -e "✓ lapack: ${GREEN}${prefix}${NC}"
                return 0
            fi
        fi
        fail_missing "lapack" "Install with: brew install lapack"
    fi

    # Linux: trust ldconfig's cache. Some minimal images don't ship ldconfig
    # in PATH, so fall back to a direct file probe in common library dirs.
    if command_exists ldconfig && ldconfig -p 2>/dev/null | grep -q "liblapack\.so"; then
        LAPACK_LIBS="-llapack"
        echo -e "✓ lapack: ${GREEN}found via ldconfig${NC}"
        return 0
    fi
    for d in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/local/lib; do
        if compgen -G "$d/liblapack.so*" >/dev/null; then
            LAPACK_LIBS="-llapack"
            echo -e "✓ lapack: ${GREEN}found in $d${NC}"
            return 0
        fi
    done
    fail_missing "lapack (development library)" \
        "$(hint_for liblapack-dev lapack-devel lapack lapack)"
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

# bellhopcuda + GLM submodule sanity check. bellhopcuda is now a git submodule
# pinned to an upstream tag (see .gitmodules); GLM is a nested submodule of
# bellhopcuda. Both are populated by `git submodule update --init --recursive`,
# which we attempt automatically if the directory looks empty.
check_bellhopcuda_submodule() {
    if [ -f "$BHC_DIR/CMakeLists.txt" ] && [ -f "$BHC_DIR/glm/glm/glm.hpp" ]; then
        echo -e "✓ bellhopcuda submodule (with GLM) initialized"
        return 0
    fi

    # Try to initialize automatically — works when the user did a fresh clone
    # without --recurse-submodules.
    if [ -f "$SCRIPT_DIR/.gitmodules" ] && command_exists git; then
        echo -e "${BLUE}Initializing bellhopcuda + GLM submodules...${NC}"
        (cd "$SCRIPT_DIR" && git submodule update --init --recursive) 2>&1 | tail -5
        if [ -f "$BHC_DIR/CMakeLists.txt" ] && [ -f "$BHC_DIR/glm/glm/glm.hpp" ]; then
            echo -e "${GREEN}✓ submodules initialized${NC}"
            return 0
        fi
    fi

    fail_missing "bellhopcuda submodule" \
        "Run: git submodule update --init --recursive (from the uacpy repo root)"
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
    check_cmake
    check_cxx_compiler
    if [[ "$BELLHOP_VERSION" == "cuda" ]]; then
        if ! check_cuda; then
            echo -e "${YELLOW}CUDA not present; falling back to C++/CPU bellhop (cxx).${NC}"
            BELLHOP_VERSION="cxx"
        fi
    fi
    check_bellhopcuda_submodule
fi

echo ""

# -------------------------
# Make bin dirs
# -------------------------
# Clean up anything at the target path that isn't a real directory — e.g. a
# dangling symlink left over from a previous install (mkdir -p would fail with
# "File exists" on those, even though the target doesn't resolve).
ensure_dir() {
    local d="$1"
    if [ -L "$d" ] && [ ! -d "$d" ]; then
        echo -e "${YELLOW}Removing dangling symlink: $d${NC}"
        rm -f "$d"
    elif [ -e "$d" ] && [ ! -d "$d" ]; then
        echo -e "${YELLOW}Removing non-directory at: $d${NC}"
        rm -f "$d"
    fi
    mkdir -p "$d"
}

ensure_dir "$BIN_DIR_OALIB"
ensure_dir "$BIN_DIR_OASES"
ensure_dir "$BIN_DIR_BELLHOP"
ensure_dir "$BIN_DIR_MPIRAMS"

# -------------------------
# Build bellhopcxx / bellhopcuda (if selected)
# -------------------------
if [[ "$BELLHOP_VERSION" == "cxx" || "$BELLHOP_VERSION" == "cuda" ]]; then
    echo -e "${BLUE}=== Building Bellhop (${BELLHOP_VERSION}) ===${NC}"
    cd "$BHC_DIR"

    BUILD_DIR="$BHC_DIR/build"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    # BHC_BUILD_EXAMPLES=OFF skips the bellhopcuda/examples/*.cpp programs —
    # uacpy doesn't use them, and at least examples/background.cpp is missing
    # an explicit #include <cstring> that GCC >= 13 no longer transitively
    # provides, which breaks the overall build.
    # bellhopcuda v1.5+ sets CMAKE_CUDA_ARCHITECTURES=native, so nvcc targets
    # the local GPU automatically — no GPU-name table or CUDA_ARCH_OVERRIDE
    # workaround needed.
    CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=Release -DBHC_ENABLE_TESTS=OFF -DBHC_BUILD_EXAMPLES=OFF"
    if [[ "$BELLHOP_VERSION" == "cuda" ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBHC_ENABLE_CUDA=ON"
        echo -e "  - CUDA support: ON (compute arch: native — auto-targeted)"
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
# Build OALIB (Acoustics-Toolbox) Fortran models
# -------------------------
# We always build via `make all` from the root of the toolbox. Running
# `make -C <subdir>` directly bypasses the root Makefile's `export FC=gfortran`
# (so $(FC) falls back to GNU make's built-in default `f77`) and also skips
# the `misc/` and `tslib/` static libraries that Kraken/Scooter depend on.
#
# We override FC/FFLAGS/etc. on the command line so they take precedence over
# the hardcoded values in the root Makefile (command-line overrides beat
# Makefile assignments unless -e is used). The -k flag keeps going past
# individual target failures so a broken Bellhop Fortran build doesn't stop
# Kraken/Scooter from completing; the copy step downstream verifies each
# expected binary is present.
echo -e "${BLUE}=== Building OALIB (Acoustics-Toolbox) ===${NC}"

# OALIB always links LAPACK; check upfront so we fail before a long build.
# check_lapack also fills in $LAPACK_LIBS with the right -L/-l flags (Homebrew
# lapack on macOS isn't on the default linker path, so a bare -llapack fails).
LAPACK_LIBS="-llapack"
check_lapack

cd "$OALIB_DIR"

OALIB_FFLAGS="-march=native -O2 -ffast-math -funroll-loops -fomit-frame-pointer -mtune=native -I../misc -I../tslib"
if [[ $ENABLE_OPENMP -eq 1 ]]; then
    OALIB_FFLAGS="$OALIB_FFLAGS -fopenmp"
    OALIB_CFLAGS="-g -fopenmp"
else
    OALIB_CFLAGS="-g"
fi

echo -e "${YELLOW}Cleaning previous builds...${NC}"
make clean 2>/dev/null || true

# OALIB is built serially (no -j). The subdirectory Makefiles (misc/, Kraken/…)
# declare Fortran module dependencies incompletely — e.g. the rule for
# AttenMod.o lists only AttenMod.o as its target, not the attenmod.mod that
# sspMod.o depends on — so a parallel build races and fails with
# "No rule to make target 'attenmod.mod'". Serial build is fast enough.
echo -e "  - Compiler: gfortran"
echo -e "  - FFLAGS:   $OALIB_FFLAGS"
echo -e "  - Parallelism: serial (upstream Makefiles race under -j)"

set +e
make -k all \
    FC=gfortran \
    FFLAGS="$OALIB_FFLAGS" \
    CC=gcc \
    CFLAGS="$OALIB_CFLAGS" \
    LAPACK_LIBS="$LAPACK_LIBS" \
    MAKEFLAGS= \
    2>&1 | tee /tmp/oalib_build.log
OALIB_STATUS=${PIPESTATUS[0]:-1}
set -e

if [[ $OALIB_STATUS -eq 0 ]]; then
    echo -e "${GREEN}✓ OALIB build finished${NC}"
else
    echo -e "${YELLOW}⚠ OALIB build had issues (exit $OALIB_STATUS). See /tmp/oalib_build.log${NC}"
    echo -e "${YELLOW}  Continuing — any binaries that built will still be installed.${NC}"
fi
echo ""

# -------------------------
# Build OASES
# -------------------------
OASES_URL="http://acoustics.mit.edu/faculty/henrik/LAMSS/pub/Oases/oases.tar.gz"

if [[ "$INSTALL_OASES" != "yes" ]]; then
    echo -e "${YELLOW}=== Skipping OASES (not selected) ===${NC}"
    echo ""
else
echo -e "${BLUE}=== Building OASES ===${NC}"
if [ ! -d "$OASES_DIR" ]; then
    # curl is only needed when fetching OASES — check here so a cxx-only build
    # without OASES selected doesn't require it.
    check_curl
    echo -e "${BLUE}Downloading OASES from $OASES_URL ...${NC}"
    OASES_TMP="$(mktemp -d)"
    if curl -fSL "$OASES_URL" -o "$OASES_TMP/oases.tar.gz"; then
        tar -xzf "$OASES_TMP/oases.tar.gz" -C "$OASES_TMP"
        # The tarball extracts to Oases_export/; rename to match expected path
        if [ -d "$OASES_TMP/Oases_export" ]; then
            mv "$OASES_TMP/Oases_export" "$OASES_DIR"
            echo -e "${GREEN}✓ OASES source downloaded and placed at $OASES_DIR${NC}"
        else
            echo -e "${RED}✗ Unexpected archive structure. Skipping OASES build.${NC}"
        fi
    else
        echo -e "${RED}✗ Failed to download OASES. Skipping build.${NC}"
    fi
    rm -rf "$OASES_TMP"
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

    # The OASES root Makefile looks up per-platform settings using the key
    # "${HOSTTYPE}-${OSTYPE}" (e.g. "FC.i386-linux-linux" at line 196). Bash's
    # $HOSTTYPE / $OSTYPE are shell builtins that aren't exported, so without
    # passing them explicitly the key becomes "-" and we fall through to the
    # "FC.- = f77" stub (writes into bin/-/, lib/-/, src/-/).
    #
    # Setting HOSTTYPE=i386 / OSTYPE=linux-linux picks the Linux entry, but
    # that entry is wedged in 1996: FC=g77 (gone since ~2005) and
    # FFLAGS="-O3 -m486 -malign-double -fstrength-reduce -fexpensive-optimizations"
    # (-m486 is rejected by modern gcc/gfortran). We override FC_STMNT, CC_STMNT,
    # FFLAGS, and CFLAGS on the make command line so they beat the in-Makefile
    # platform-keyed assignments.
    #
    # The compile flags must be embedded in FC_STMNT rather than passed via
    # FFLAGS: src/makefile has a handful of rules that invoke bare "$(FC)"
    # without "$(FFLGS)" (e.g. oasgun21.o at line 69), so flags via FFLAGS
    # never reach those files. Putting them in FC_STMNT means every
    # invocation of the compiler picks them up.
    #
    #   -fallow-argument-mismatch  — demote type-mismatch errors (F77 code
    #                                relies on these; gfortran 10+ rejects by default)
    #   -std=legacy                — re-enable F77 extensions (Hollerith, etc.)
    #   -fno-automatic             — SAVE locals by default (F77 semantics)
    OASES_FC="gfortran -fallow-argument-mismatch -std=legacy -fno-automatic"
    OASES_FFLAGS="-O2"
    OASES_CFLAGS="-O2"

    echo -e "${BLUE}Starting OASES make (log -> /tmp/oases_build.log)${NC}"
    echo -e "  - HOSTTYPE=${OASES_HOSTTYPE}  OSTYPE=${OASES_OSTYPE}"
    echo -e "  - FC_STMNT='${OASES_FC}'  FFLAGS=${OASES_FFLAGS}"

    set +e
    make \
        HOSTTYPE="$OASES_HOSTTYPE" \
        OSTYPE="$OASES_OSTYPE" \
        OASES_ROOT="$OASES_DIR" \
        FC_STMNT="$OASES_FC" \
        CC_STMNT=gcc \
        FFLAGS="$OASES_FFLAGS" \
        CFLAGS="$OASES_CFLAGS" \
        LFLAGS="" \
        oases 2>&1 | tee /tmp/oases_build.log
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
            OASES_INSTALLED=$((OASES_INSTALLED + 1))
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
fi  # end of INSTALL_OASES gate

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

# Install bellhopcxx/bellhopcuda artifacts (if built).
# SetupCommon.cmake sets CMAKE_RUNTIME_OUTPUT_DIRECTORY = ${CMAKE_SOURCE_DIR}/bin,
# so the linked binaries actually land in $BHC_DIR/bin, NOT $BUILD_DIR. Search
# both to stay robust against future CMake changes.
if [[ "$BELLHOP_VERSION" == "cxx" || "$BELLHOP_VERSION" == "cuda" ]]; then
    BUILD_DIR="$BHC_DIR/build"
    BHC_OUT_DIR="$BHC_DIR/bin"
    SEARCH_NAMES=(bellhopcxx bellhopcxx2d bellhopcxx3d bellhopcxxnx2d \
                  bellhopcuda bellhopcuda2d bellhopcuda3d bellhopcudanx2d)

    declare -A SEEN_BHC=()
    for search_root in "$BHC_OUT_DIR" "$BUILD_DIR"; do
        [ -d "$search_root" ] || continue
        for name in "${SEARCH_NAMES[@]}"; do
            while IFS= read -r -d $'\0' file; do
                base=$(basename "$file")
                [[ -n "${SEEN_BHC[$base]:-}" ]] && continue
                SEEN_BHC[$base]=1
                cp "$file" "$BIN_DIR_BELLHOP/$base"
                chmod +x "$BIN_DIR_BELLHOP/$base"
                echo -e "  ✓ Installed bellhop binary: ${GREEN}$base${NC}"
                INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
            done < <(find "$search_root" -type f -executable -name "$name" -print0 2>/dev/null || true)
        done
    done

    if [[ ${#SEEN_BHC[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No bellhop (cxx/cuda) executables found in $BHC_OUT_DIR or $BUILD_DIR.${NC}"
    fi
fi

# Install OALIB (Fortran) executables. Bellhop Fortran is only installed when
# the user picked the fortran variant — the cxx/cuda binaries live in
# $BIN_DIR_BELLHOP and uacpy picks them up from there.
OALIB_EXECUTABLES=(
    "KrakenField/field.exe"
    "Kraken/kraken.exe"
    "Kraken/krakenc.exe"
    "Kraken/bounce.exe"
    "Scooter/scooter.exe"
    "Scooter/sparc.exe"
)
# Fortran Bellhop is always built and installed as the reference implementation,
# regardless of whether a C++/CUDA variant was also selected.
OALIB_EXECUTABLES+=("Bellhop/bellhop.exe" "Bellhop/bellhop3d.exe")

for path in "${OALIB_EXECUTABLES[@]}"; do
    if [ -f "$OALIB_DIR/$path" ]; then
        bn=$(basename "$path")
        cp "$OALIB_DIR/$path" "$BIN_DIR_OALIB/$bn"
        chmod +x "$BIN_DIR_OALIB/$bn"
        echo -e "  ✓ Installed OALIB binary: ${GREEN}$bn${NC}"
        INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
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
echo "  Bellhop (Fortran):     $BIN_DIR_OALIB"
if [[ "$BELLHOP_VERSION" == "cxx" || "$BELLHOP_VERSION" == "cuda" ]]; then
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

