#!/usr/bin/env bash
#
# install.sh - Build and install all UACPY native model binaries
#
# Compiles the following acoustic propagation models:
#   - OALIB (Acoustics-Toolbox): Kraken, KrakenField, Scooter, SPARC, Bellhop (Fortran)
#   - BellhopCUDA: C++/CUDA Bellhop (optional, for GPU acceleration)
#   - OASES: Ocean Acoustics and Seismics (optional, downloaded from MIT)
#   - mpiramS: Parabolic Equation model (broadband PE)
#   - ramsurf: Collins-style PE family (rams0.5 elastic, ramsurf1.5 rough surface)
#
# Layout produced:
#   uacpy/bin/oalib/       — Kraken, Scooter, SPARC, Bellhop (Fortran)
#   uacpy/bin/bellhopcuda/ — BellhopCXX / BellhopCUDA (C++/CUDA)
#   uacpy/bin/oases/       — OASES suite
#   uacpy/bin/mpirams/     — mpiramS PE model
#   uacpy/bin/ramsurf/     — Collins rams0.5 (elastic), ramsurf1.5 (rough surface)
#
# By default runs interactively. Use -y/--yes for non-interactive mode.
#
set -euo pipefail

# Per-run build logs live under a mktemp dir (fixed /tmp names collide
# between users on a shared machine and can be pre-created as symlinks);
# allocated after argument parsing so --help leaves nothing behind.
BUILD_LOG_DIR=""

# What the durable build record actually managed to leave on disk. Empty means
# it could not be written; the summary reads these rather than asserting.
BUILD_RECORD_WRITTEN=""
BUILD_LOGS_COPIED=""

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
RAMSURF_DIR="${SCRIPT_DIR}/uacpy/third_party/ramsurf"           # Collins RAM family
RAMGEO_DIR="${SCRIPT_DIR}/uacpy/third_party/ramgeo"             # RAMGEO (RD layered fluid)
BIN_ROOT="${SCRIPT_DIR}/uacpy/bin"

BIN_DIR_OALIB="${BIN_ROOT}/oalib"
BIN_DIR_BELLHOP="${BIN_ROOT}/bellhopcuda"
BIN_DIR_OASES="${BIN_ROOT}/oases"
BIN_DIR_MPIRAMS="${BIN_ROOT}/mpirams"
BIN_DIR_RAMSURF="${BIN_ROOT}/ramsurf"
BIN_DIR_RAMGEO="${BIN_ROOT}/ramgeo"

# Offline data cache (downloaded public datasets for uacpy.data's local backend;
# gitignored, never bundled). Override at runtime with $UACPY_DATA_CACHE.
DATA_CACHE_DIR="${UACPY_DATA_CACHE:-${SCRIPT_DIR}/data_cache}"
# WOA23 download settings: 1° annual+monthly temperature & salinity (decav).
WOA_RESOLUTION="1.00"
WOA_CODE="01"
WOA_DECADE="decav"
# GEBCO 2025 ice-surface elevation grid, direct NetCDF from CEDA (no auth,
# ~7.5 GB). The digest is checked before the download is promoted into the
# cache; clear it to disable that check.
GEBCO_NC_URL="https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2025/ice_surface_elevation/netcdf/GEBCO_2025.nc?download=1"
GEBCO_MD5="cd18ddc4162134465af310f714d50f01"

# GlobSed v3 total sediment thickness NetCDF (NOAA NCEI archive). Fetched with
# curl: NCEI/Akamai throttles Python urllib to a trickle, but serves curl fast.
GLOBSED_NC_URL="https://www.ncei.noaa.gov/data/oceans/archive/arc0231/0305030/1.1/data/0-data/GlobSed/GlobSed_package3/GlobSed-v3.nc"

# Bellhopcuda upstream release we build against. Tags are mutable on the
# upstream side, so an immutable commit SHA is preferred. When
# BELLHOPCUDA_COMMIT_SHA is non-empty install.sh checks out that exact
# commit; otherwise it falls back to BELLHOPCUDA_TAG with a loud warning.
BELLHOPCUDA_TAG="v1.5"
BELLHOPCUDA_COMMIT_SHA="b396d40ba49c2f349258b9687cfae8ff8323828f"

# Default behavior: interactive
AUTO_YES=0         # 0 = interactive (prompt the user); 1 = assume "yes"
FORCE=0
BELLHOP_VERSION="" # "fortran", "cxx", or "cuda" (empty => prompt/auto)
INSTALL_OASES=""   # "yes" or "no" (empty => prompt/auto)
INSTALL_DATA=""    # comma list of $DATA_IDS | "all" | "no" (empty => skip)
# Every dataset id a download_* function below implements — the one source of
# truth for both --data validation and the "all" expansion.
DATA_IDS="gebco woa23 sediment emodnet coastline globsed crust1 diesing seaice glodap wind graw"
BUILD_MODELS=1     # 0 with --no-models: skip all native builds (data-only install)

# -------------------------
# Per-component build status
# -------------------------
# Every build phase sets one of these to "ok" / "skipped" / "failed" and
# (optionally) appends a one-line note. The final summary prints every
# component in the same format so the user sees a single, consistent report.
STATUS_OALIB="skipped"        # Bellhop (Fortran) + Kraken/KrakenC/Bounce/Scooter/SPARC/KrakenField
STATUS_BELLHOPCUDA="skipped"  # bellhopcxx / bellhopcuda (optional)
STATUS_OASES="skipped"        # OAST / OASN / OASR / OASP (optional)
STATUS_MPIRAMS="skipped"      # mpiramS PE
STATUS_RAMSURF="skipped"      # rams0.5 + ramsurf1.5
STATUS_RAMGEO="skipped"       # RAMGEO (RD layered fluid)
STATUS_DATA="skipped"         # offline data cache (GEBCO / WOA23 / sediment)
NOTE_OALIB=""
NOTE_BELLHOPCUDA=""
NOTE_OASES=""
NOTE_MPIRAMS=""
NOTE_RAMSURF=""
NOTE_RAMGEO=""
NOTE_DATA=""

# Pretty-print one component status row. Used by the final summary.
print_status_row() {
    local label="$1" status="$2" note="$3"
    local color symbol
    case "$status" in
        ok)      color="$GREEN";  symbol="✓ installed" ;;
        partial) color="$YELLOW"; symbol="◐ partial  " ;;
        failed)  color="$RED";    symbol="✗ failed   " ;;
        skipped) color="$YELLOW"; symbol="– skipped  " ;;
        *)       color="$NC";     symbol="? unknown  " ;;
    esac
    printf "  ${color}%s${NC}  %-22s %s\n" "$symbol" "$label" "$note"
}

# Fortran architecture flags shared by every Fortran build in this script
# (OALIB, mpiramS, ramsurf). OASES is excluded — it ships -O2 with no
# -march and is already portable. Default targets the build host: -march=native
# on x86, -mcpu=native on aarch64. gfortran on aarch64 rejects CPU names like
# apple-m2 under -march= (which only accepts the fixed armv8.x-a/armv9.x-a
# list) — the equivalent knob there is -mcpu=. Set UACPY_FORTRAN_ARCH_FLAGS
# (e.g. "-march=x86-64-v3") to produce binaries portable across a CPU family
# instead of pinned to the build host.
case "$(uname -m)" in
    arm64|aarch64) FORTRAN_ARCH_FLAGS="${UACPY_FORTRAN_ARCH_FLAGS:--mcpu=native}" ;;
    *)             FORTRAN_ARCH_FLAGS="${UACPY_FORTRAN_ARCH_FLAGS:--march=native -mtune=native}" ;;
esac

# -------------------------
# Helpers
# -------------------------
command_exists() {
    command -v "$1" &> /dev/null
}

get_nproc() {
    nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2
}

prompt_yes_no() {
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

# One resilient-download policy shared by EVERY curl fetch (OASES + each data
# grid). For multi-GB single streams this matters:
#   --http1.1          HTTP/2 mid-stream CANCELs (curl exit 92) on long transfers
#   -C -               resume the partial `-o` file from its current size
#   --retry-all-errors plain --retry ignores transport errors like exit 92 / 18
#   --retry 3 / delay  quick in-process retries for transient blips
#   --connect-timeout  bound the connect phase so a dead host fails fast
# Curl's *internal* --retry does not reliably resume mid-stream (it restarts from
# the launch offset), so a flaky server that closes the connection early (curl
# exit 18, "bytes missing") makes no forward progress. We therefore wrap it in an
# outer loop that RE-INVOKES curl: each fresh call resumes from the partial via
# `-C -`, and because these hosts (CEDA/NCEI/WOA) answer a Range request with
# `206 Partial Content`, the file grows monotonically across a flaky link instead
# of restarting a 7.5 GB download from zero. Bounded so a no-range / dead host
# still terminates.
# Usage: robust_curl <url> <output-path> [extra curl args…]  (extras precede URL).
robust_curl() {
    local url="$1" out="$2"; shift 2
    local attempt=0 max_attempts=20 rc http_code
    while :; do
        http_code=$(curl -fL --http1.1 --retry 3 --retry-delay 5 --retry-all-errors \
                --connect-timeout 30 -C - -o "$out" -w '%{http_code}' "$@" "$url")
        rc=$?
        if (( rc == 0 )); then
            return 0
        fi
        # Exit 22 means the server answered with an HTTP error (>= 400, since
        # -f). Those are permanent apart from the rate-limit codes, and
        # --retry-all-errors already re-sent each one 4 times inside curl, so
        # continuing the outer loop just burns the budget: a renamed URL would
        # cost ~80 requests before the caller heard about it, and a 26-file
        # download loop hours. Report it now.
        if (( rc == 22 )) && [[ "$http_code" != "429" && "$http_code" != "408" ]]; then
            echo -e "  ${RED}HTTP ${http_code} for ${url} — permanent, not retrying${NC}" >&2
            return 1
        fi
        attempt=$((attempt + 1))
        if (( attempt >= max_attempts )); then
            return 1
        fi
        local have=0; [ -f "$out" ] && have=$(wc -c < "$out" 2>/dev/null || echo 0)
        echo -e "  ${YELLOW}download interrupted (outer retry ${attempt}/${max_attempts}," \
                "have ${have} bytes) — resuming in 5s...${NC}" >&2
        sleep 5
    done
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
                         yes — Download from MIT and build OAST/OASN/OASR/OASP/OASS/OASSP
                         no  — Skip OASES entirely
  --data [LIST]        Download public datasets for uacpy.data's offline backend
                       into ./data_cache (gitignored). LIST is a comma list of:
                         gebco     — GEBCO 2025 bathymetry grid (~7.5 GB, CEDA)
                         woa23     — WOA23 T/S climatology grids (NCEI)
                         sediment  — NCEI grain-size DB (public domain, ~3 MB)
                         emodnet   — EMODnet seabed substrate, Folk 5cl,
                                     European seas (CC-BY, ~200 MB; needs shapely)
                         coastline — Natural Earth land polygons (public domain)
                         globsed   — GlobSed total sediment thickness, global
                                     (NOAA NCEI, public domain, ~11 MB)
                         crust1    — CRUST1.0 layered Vp/Vs/density, global,
                                     low-frequency seabed (~1 MB, no formal licence)
                         diesing   — Diesing 2020 global deep-sea seafloor
                                     lithology map (CC-BY, ~40 MB; needs pyproj)
                         seaice    — NSIDC sea-ice concentration monthly
                                     climatology (public domain; needs tifffile)
                         glodap    — GLODAPv2.2016b mapped seawater pH, global
                                     (CC-BY, ~211 MB tarball → pH grid)
                         wind      — NBS 10 m wind-speed monthly climatology
                                     (NOAA, public domain; needs netCDF4)
                         graw      — Graw 2021 predicted seabed bulk density,
                                     global 5' (Zenodo, CC-BY, ~37 MB; needs netCDF4)
                         all       — all of the above   |   no — skip (default)
  --no-models          Skip ALL native propagation-model builds (no compilers
       (--data-only)   needed). Pure-Python install — pairs with --data for an
                       offline data-only setup. Build prerequisites are skipped.
  -h, --help           Show this help

Examples:
  ./install.sh                     # Interactive: prompts for choices
  ./install.sh -y                  # Auto-detect everything, no prompts
  ./install.sh --bellhop cuda      # Use CUDA Bellhop, prompt for rest
  ./install.sh --oases no          # Skip OASES, prompt for rest
  ./install.sh --data sediment     # Only the small public-domain sediment DB
  ./install.sh --data crust1,globsed    # Low-frequency layered seabed stack
  ./install.sh --data all          # Full stack (gebco,woa23,sediment,emodnet,coastline,globsed,crust1,diesing,seaice,glodap,wind,graw)
  ./install.sh --no-models --data all   # Data-only: no compilers, offline stack
EOF
}

# Guard the value of a flag that takes one.
# Args: $1 flag, $2 the next token (may be absent), $3 the accepted spellings.
# An option-shaped token is never a value: `--data --yes` would otherwise set
# INSTALL_DATA='--yes' and drop the -y the user did pass, committing a
# multi-hour run to choices they never made.
require_value() {
    if [[ -z "${2:-}" || "$2" == -* ]]; then
        echo -e "${RED}$1 requires an argument: $3${NC}" >&2
        echo "Run '$0 --help' for the accepted values." >&2
        exit 1
    fi
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
            require_value --bellhop "${2:-}" 'fortran|cxx|cuda'
            BELLHOP_VERSION="$2"
            shift 2
            ;;
        --oases)
            require_value --oases "${2:-}" 'yes|no'
            INSTALL_OASES="$2"
            shift 2
            ;;
        --data)
            require_value --data "${2:-}" 'gebco,woa23,sediment,emodnet,coastline,globsed,crust1,diesing,seaice,glodap,wind,graw|all|no'
            INSTALL_DATA="$2"
            shift 2
            ;;
        --no-models|--data-only)
            BUILD_MODELS=0
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            # Stop rather than warn and continue: the run that follows is hours
            # long, and a misspelt flag means it is not the run that was asked
            # for.
            echo -e "${RED}Unknown argument: $1${NC}" >&2
            echo "Run '$0 --help' for the accepted flags." >&2
            exit 1
            ;;
    esac
done

# Per-run build logs (kept after the run for post-mortem reading).
BUILD_LOG_DIR="$(mktemp -d -t uacpy_build_logs.XXXXXX)"

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
    # Apple Silicon's Homebrew lives under /opt/homebrew/bin and isn't on
    # the default user PATH. Without this, `command_exists brew` and
    # subsequently `gfortran` fail even when the user has installed them.
    if [ -d /opt/homebrew/bin ]; then
        export PATH="/opt/homebrew/bin:$PATH"
    fi
    if command_exists brew; then
        BREW_GCC_PREFIX="$(brew --prefix gcc 2>/dev/null || true)"
        if [ -n "$BREW_GCC_PREFIX" ] && [ -d "$BREW_GCC_PREFIX/bin" ]; then
            export PATH="$BREW_GCC_PREFIX/bin:$PATH"
        fi
    fi
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

# OpenMP usage per component:
#   - OALIB (AT):     no OpenMP directives
#   - bellhopcxx/cuda: no OpenMP in the cmake config
#   - ramsurf:        no OpenMP directives
#   - mpiramS:        !$OMP loop in peramx.f90 — requires -fopenmp
# Only mpiramS links libgomp, which gfortran provides natively on every
# supported platform (brew's libomp is the LLVM/clang runtime and is
# unrelated).

# Offer a data-only install interactively (mirrors choose_oases / choose_data).
# An explicit --no-models flag, or -y, bypasses the prompt.
choose_models() {
    if [[ "$BUILD_MODELS" != "1" ]]; then
        return 0          # --no-models already set on the command line
    fi
    if [[ $AUTO_YES -eq 1 ]]; then
        return 0          # non-interactive default: build the models
    fi
    echo -e "${BLUE}Build native propagation models (Bellhop, Kraken, RAM, OASES, ...)?${NC}"
    echo "  Compiling them needs gfortran/make (and optionally cmake/nvcc)."
    echo "  Answer No for a pure-Python, data-only install (no compilers needed)."
    echo ""
    if ! prompt_yes_no "Build propagation models?"; then
        BUILD_MODELS=0
    fi
    return 0
}

choose_models

# --no-models: a data-only install. Preset the model selections so their prompts
# are skipped, mark every model component "skipped", and gate the build region
# below. The offline data cache (choose_data + download) still runs.
if [[ "$BUILD_MODELS" != "1" ]]; then
    echo -e "${YELLOW}--no-models: skipping native propagation-model builds (data-only).${NC}"
    echo ""
    BELLHOP_VERSION="fortran"   # value unused; just bypasses the prompt
    INSTALL_OASES="no"
    NOTE_OALIB="skipped (--no-models)"
    NOTE_BELLHOPCUDA="skipped (--no-models)"
    NOTE_OASES="skipped (--no-models)"
    NOTE_MPIRAMS="skipped (--no-models)"
    NOTE_RAMSURF="skipped (--no-models)"
    NOTE_RAMGEO="skipped (--no-models)"
fi

# -------------------------
# Choose Bellhop variant
# -------------------------
choose_bellhop() {
    if [[ -n "$BELLHOP_VERSION" ]]; then
        case "$BELLHOP_VERSION" in
            fortran|cxx|cuda) return 0 ;;
            *)
                # Stop rather than fall through to the default, as above: a
                # misspelt value otherwise silently selects a different -- and
                # more expensive -- backend than the one asked for.
                echo -e "${RED}Invalid --bellhop argument: ${BELLHOP_VERSION}${NC}" >&2
                echo "Expected 'fortran', 'cxx' or 'cuda' (lowercase)." >&2
                exit 1
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

if [[ "$BUILD_MODELS" == "1" ]]; then
    choose_bellhop
    echo -e "Selected Bellhop variant: ${GREEN}${BELLHOP_VERSION}${NC}"
    echo ""
fi

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
                # Stop, for the same reason an unknown FLAG stops the run: the
                # build that follows is hours long. Warning and clearing the
                # value handed the run to the AUTO_YES default below, so
                # `--oases NO` (a plausible typo for `no`) downloaded ~100 MB
                # from acoustics.mit.edu and built the whole OASES suite --
                # the exact opposite of the request, at exit 0.
                echo -e "${RED}Invalid --oases argument: ${INSTALL_OASES}${NC}" >&2
                echo "Expected 'yes' or 'no' (lowercase)." >&2
                exit 1
                ;;
        esac
    fi

    # Non-interactive mode defaults to yes; the build section below skips
    # the download when the source tree is already present locally.
    if [[ $AUTO_YES -eq 1 ]]; then
        INSTALL_OASES="yes"
        return 0
    fi

    echo -e "${BLUE}Install OASES (OAST, OASN, OASR, OASP, OASS, OASSP)?${NC}"
    echo "  OASES is distributed separately by MIT and is not bundled with UACPY."
    echo "  Saying yes will download the source from acoustics.mit.edu and build it."
    echo ""
    if prompt_yes_no "Download and install OASES?"; then
        INSTALL_OASES="yes"
    else
        INSTALL_OASES="no"
    fi
}

if [[ "$BUILD_MODELS" == "1" ]]; then
    choose_oases
    if [[ "$INSTALL_OASES" == "yes" ]]; then
        echo -e "OASES: ${GREEN}will be installed${NC}"
    else
        echo -e "OASES: ${YELLOW}skipped${NC}"
    fi
    echo ""
fi

# Choose which (if any) offline datasets to download for uacpy.data's local
# backend. Mirrors choose_oases: respects an explicit --data flag, otherwise
# prompts interactively. Because the grids are large (GEBCO ~4 GB), -y does NOT
# auto-download — the user must opt in via --data or the prompt.
validate_data_selection() {
    # Drop unknown ids from a --data list, the way choose_bellhop / choose_oases
    # reject their flags. Unvalidated, a typo matches no ``data_requested``
    # below, short-circuits every download, and still reports the cache as
    # installed — a green summary row for a directory that was never filled.
    local kept="" dropped="" tok
    local IFS=','
    for tok in $INSTALL_DATA; do
        [[ -z "$tok" ]] && continue
        if [[ " $DATA_IDS " == *" $tok "* ]]; then
            kept="${kept}${tok},"
        else
            dropped="${dropped}${tok} "
        fi
    done
    dropped="${dropped% }"
    if [[ -n "$dropped" ]]; then
        echo -e "${YELLOW}Unknown --data dataset(s): ${dropped}${NC}"
        echo -e "${YELLOW}Valid ids: ${DATA_IDS// /, } (or 'all' / 'no').${NC}"
        NOTE_DATA="${NOTE_DATA} unknown:${dropped// /,}"
    fi
    INSTALL_DATA="${kept%,}"
    if [[ -z "$INSTALL_DATA" ]]; then
        # Nothing recognised: the request cannot be honoured at all, so the
        # summary says failed rather than skipped or ok.
        INSTALL_DATA="no"
        [[ -n "$dropped" ]] && STATUS_DATA="failed"
        return 1
    fi
    return 0
}

choose_data() {
    if [[ -n "$INSTALL_DATA" ]]; then
        case "$INSTALL_DATA" in
            all|no) ;;
            *) validate_data_selection || true ;;
        esac
        return 0
    fi
    if [[ $AUTO_YES -eq 1 ]]; then
        INSTALL_DATA="no"
        return 0
    fi
    echo -e "${BLUE}Offline data cache (uacpy.data local backend → ./data_cache)?${NC}"
    echo "  Optional public datasets so uacpy.data works offline, with no rate"
    echo "  limits. Downloaded only if you opt in; never bundled. Mostly public"
    echo "  domain / CC-BY; CRUST1.0 has no formal licence (verify for commercial)."
    echo ""
    if ! prompt_yes_no "Configure offline datasets now?"; then
        INSTALL_DATA="no"
        return 0
    fi
    local sel=""
    prompt_yes_no "  • Coastline: Natural Earth land polygons (~tens of MB, public domain)?" \
        && sel="${sel}coastline,"
    prompt_yes_no "  • Sediment samples: DECK41 + grain-size (~tens of MB, public domain)?" \
        && sel="${sel}sediment,"
    prompt_yes_no "  • EMODnet seabed substrate: Folk 5cl, European seas (~200 MB, CC-BY 4.0)?" \
        && sel="${sel}emodnet,"
    prompt_yes_no "  • GlobSed total sediment thickness, global (~11 MB, public domain)?" \
        && sel="${sel}globsed,"
    prompt_yes_no "  • CRUST1.0 layered Vp/Vs/density, global, low-freq (~1 MB, no formal licence)?" \
        && sel="${sel}crust1,"
    prompt_yes_no "  • Diesing 2020 global deep-sea seafloor lithology (~40 MB, CC-BY 4.0)?" \
        && sel="${sel}diesing,"
    prompt_yes_no "  • Graw 2021 predicted seabed bulk density, global 5' (~37 MB, CC-BY 4.0)?" \
        && sel="${sel}graw,"
    prompt_yes_no "  • NSIDC sea-ice monthly climatology (built from ~120 grids, public domain)?" \
        && sel="${sel}seaice,"
    prompt_yes_no "  • NBS 10 m wind-speed monthly climatology (built from ERDDAP grids, public domain)?" \
        && sel="${sel}wind,"
    prompt_yes_no "  • WOA23 sound-speed climatology grids (~hundreds of MB, public domain)?" \
        && sel="${sel}woa23,"
    prompt_yes_no "  • GLODAPv2.2016b mapped seawater pH, global (~211 MB tarball → pH grid, CC-BY 4.0)?" \
        && sel="${sel}glodap,"
    prompt_yes_no "  • GEBCO 2025 bathymetry grid (~4 GB, public domain)?" \
        && sel="${sel}gebco,"
    INSTALL_DATA="${sel%,}"
    if [[ -z "$INSTALL_DATA" ]]; then
        INSTALL_DATA="no"
    fi
    return 0
}

choose_data

if [[ -n "$INSTALL_DATA" && "$INSTALL_DATA" != "no" ]]; then
    echo -e "Offline data: ${GREEN}${INSTALL_DATA}${NC}"
else
    echo -e "Offline data: ${YELLOW}skipped${NC}"
fi
echo ""

# =====================================================================
# Native model builds (prerequisites + compilation). Gated by --no-models:
# a data-only install skips this entire block (no compilers required) and
# proceeds straight to the offline data cache + summary. The helper functions
# defined inside are only used inside; the data block below does not need them.
# =====================================================================
if [[ "$BUILD_MODELS" == "1" ]]; then

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


# tar (required only when extracting the OASES tarball)
check_tar() {
    if command_exists tar; then
        return 0
    fi
    fail_missing "tar" "$(hint_for tar tar tar tar)"
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
        fixup_bhc_dotgit
        pin_bellhopcuda_tag
        return 0
    fi

    # Try to initialize automatically — works when the user did a fresh clone
    # without --recurse-submodules.
    if [ -f "$SCRIPT_DIR/.gitmodules" ] && command_exists git; then
        echo -e "${BLUE}Initializing bellhopcuda + GLM submodules...${NC}"
        (cd "$SCRIPT_DIR" && git submodule update --init --recursive) 2>&1 | tail -5
        if [ -f "$BHC_DIR/CMakeLists.txt" ] && [ -f "$BHC_DIR/glm/glm/glm.hpp" ]; then
            echo -e "${GREEN}✓ submodules initialized${NC}"
            fixup_bhc_dotgit
            pin_bellhopcuda_tag
            return 0
        fi
    fi

    fail_missing "bellhopcuda submodule" \
        "Run: git submodule update --init --recursive (from the uacpy repo root)"
}

# Workaround for upstream bellhopcuda issue: config/CMakeLists.txt installs a
# clang-format pre-commit hook by copying into ${PROJECT_SOURCE_DIR}/.git/hooks/.
# When bellhopcuda is a submodule, .git is a regular file (a "gitdir: ..."
# pointer), so file(COPY) fails with "Not a directory". Replace the .git file
# with a symlink to the resolved gitdir so .git/hooks/ resolves correctly.
fixup_bhc_dotgit() {
    if [ ! -f "$BHC_DIR/.git" ]; then
        return 0   # already a directory or symlink
    fi
    local gitdir_rel gitdir_abs
    gitdir_rel="$(sed -n 's|^gitdir: ||p' "$BHC_DIR/.git")"
    if [ -z "$gitdir_rel" ]; then
        return 0
    fi
    gitdir_abs="$(cd "$BHC_DIR" && cd "$gitdir_rel" 2>/dev/null && pwd)" || return 0
    rm -f "$BHC_DIR/.git"
    ln -s "$gitdir_abs" "$BHC_DIR/.git"
}

# Force the bellhopcuda submodule HEAD to $BELLHOPCUDA_COMMIT_SHA (preferred,
# immutable) or $BELLHOPCUDA_TAG (mutable upstream alias) as a fallback.
pin_bellhopcuda_tag() {
    if ! command_exists git; then
        echo -e "${YELLOW}git not available; skipping bellhopcuda tag verification${NC}"
        return 0
    fi

    # Refuse to clobber uncommitted work in the submodule. A developer running
    # install.sh after editing third_party/bellhopcuda/ would otherwise lose
    # those changes silently.
    if [ -n "$(git -C "$BHC_DIR" status --porcelain 2>/dev/null)" ]; then
        echo -e "${YELLOW}bellhopcuda submodule has uncommitted changes; skipping pin to avoid clobbering work.${NC}"
        return 0
    fi

    # Prefer the commit SHA when set — tags are mutable upstream and a SHA
    # locks the supply chain.
    local pin_ref
    if [ -n "$BELLHOPCUDA_COMMIT_SHA" ]; then
        pin_ref="$BELLHOPCUDA_COMMIT_SHA"
    else
        echo -e "${YELLOW}BELLHOPCUDA_COMMIT_SHA is unset; pinning to mutable tag ${BELLHOPCUDA_TAG}. Set the SHA in install.sh for supply-chain integrity.${NC}"
        pin_ref="$BELLHOPCUDA_TAG"
    fi

    local current_sha current_tag
    current_sha="$(git -C "$BHC_DIR" rev-parse HEAD 2>/dev/null || true)"
    current_tag="$(git -C "$BHC_DIR" describe --tags --exact-match 2>/dev/null || true)"
    if [ -n "$BELLHOPCUDA_COMMIT_SHA" ] && [[ "$current_sha" == "$BELLHOPCUDA_COMMIT_SHA"* ]]; then
        echo -e "✓ bellhopcuda at commit ${GREEN}${BELLHOPCUDA_COMMIT_SHA}${NC}"
        return 0
    fi
    if [ -z "$BELLHOPCUDA_COMMIT_SHA" ] && [[ "$current_tag" == "$BELLHOPCUDA_TAG" ]]; then
        echo -e "✓ bellhopcuda at tag ${GREEN}${BELLHOPCUDA_TAG}${NC}"
        return 0
    fi

    echo -e "${BLUE}Pinning bellhopcuda to ${pin_ref}...${NC}"

    # Shallow clones (CI) may not have tags — fetch on demand.
    if ! git -C "$BHC_DIR" rev-parse --verify --quiet "${pin_ref}^{commit}" >/dev/null 2>&1; then
        echo -e "  Fetching from origin..."
        git -C "$BHC_DIR" fetch --tags origin 2>&1 | tail -3 || true
    fi
    if ! git -C "$BHC_DIR" rev-parse --verify --quiet "${pin_ref}^{commit}" >/dev/null 2>&1; then
        fail_missing "bellhopcuda ref '${pin_ref}'" \
            "Ref not found in ${BHC_DIR}. Edit BELLHOPCUDA_TAG / BELLHOPCUDA_COMMIT_SHA in install.sh or fetch manually."
    fi

    if ! git -C "$BHC_DIR" checkout --quiet "${pin_ref}"; then
        fail_missing "bellhopcuda checkout" \
            "Could not check out '${pin_ref}' in ${BHC_DIR}"
    fi

    # GLM (nested submodule) commit may differ between revisions — re-sync it.
    git -C "$BHC_DIR" submodule update --init --recursive 2>&1 | tail -3 || true

    echo -e "${GREEN}✓ bellhopcuda pinned to ${pin_ref}${NC}"
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
    # bellhopcuda comes via the git submodule + nested GLM; both need git.
    check_git
    if [[ "$BELLHOP_VERSION" == "cuda" ]]; then
        # If we landed on cuda the user explicitly asked for it (auto-detect
        # only picks cuda when nvcc is present). Refuse to silently downgrade —
        # the user's intent (GPU build) wouldn't be honoured and the resulting
        # CPU binary would surprise downstream tooling.
        if ! check_cuda; then
            echo -e "${RED}--bellhop cuda was requested but nvcc is not on PATH.${NC}" >&2
            echo -e "${YELLOW}Install the CUDA toolkit, or pass --bellhop cxx for the CPU-only C++ build, or --bellhop fortran for the reference Fortran build.${NC}" >&2
            exit 1
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
ensure_dir "$BIN_DIR_RAMSURF"
ensure_dir "$BIN_DIR_RAMGEO"

# -------------------------
# Build bellhopcxx / bellhopcuda (if selected)
# -------------------------
if [[ "$BELLHOP_VERSION" == "cxx" || "$BELLHOP_VERSION" == "cuda" ]]; then
    echo -e "${BLUE}=== Building Bellhop (${BELLHOP_VERSION}) ===${NC}"
    cd "$BHC_DIR"

    BUILD_DIR="$BHC_DIR/build"
    # Default to incremental rebuilds (CMake handles dirty-source dependency
    # tracking). --force wipes the build dir so cmake reconfigures from scratch.
    if [ "$FORCE" -eq 1 ]; then
        rm -rf "$BUILD_DIR"
    fi
    mkdir -p "$BUILD_DIR"

    # BHC_BUILD_EXAMPLES=OFF skips the bellhopcuda/examples/*.cpp programs.
    # uacpy doesn't use them, and at least examples/background.cpp depends
    # on a transitive #include <cstring> that GCC >= 13 no longer provides,
    # so leaving examples enabled breaks the overall build.
    # bellhopcuda v1.5+ sets CMAKE_CUDA_ARCHITECTURES=native via SetupCommon,
    # so nvcc auto-targets the local GPU.
    CMAKE_OPTIONS="-DCMAKE_BUILD_TYPE=Release -DBHC_BUILD_EXAMPLES=OFF"
    if [[ "$BELLHOP_VERSION" == "cuda" ]]; then
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBHC_ENABLE_CUDA=ON"
        echo -e "  - CUDA support: ON (compute arch: native — auto-targeted)"
    else
        CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBHC_ENABLE_CUDA=OFF"
        echo -e "  - CUDA support: OFF (CPU build)"
    fi

    # Run cmake under set +e so a CMake failure (missing CUDA arch, broken GLM
    # checkout, etc.) doesn't abort the whole installer — OALIB/mpiramS/ramsurf
    # are independent of bellhopcxx/cuda and should still get a chance to build.
    echo -e "${BLUE}Configuring CMake...${NC}"
    set +e
    cmake -S "$BHC_DIR" -B "$BUILD_DIR" $CMAKE_OPTIONS
    BHC_CONFIGURE_RC=$?

    if [ $BHC_CONFIGURE_RC -eq 0 ]; then
        echo -e "${BLUE}Building (this may take a while)...${NC}"
        NPROC=$(get_nproc)
        cmake --build "$BUILD_DIR" --config Release -j"$NPROC"
        BHC_BUILD_RC=$?
    else
        BHC_BUILD_RC=1
    fi
    set -e

    if [ $BHC_CONFIGURE_RC -eq 0 ] && [ $BHC_BUILD_RC -eq 0 ]; then
        echo -e "${GREEN}✓ Bellhop (${BELLHOP_VERSION}) build finished${NC}"
        STATUS_BELLHOPCUDA="ok"
        NOTE_BELLHOPCUDA="${BELLHOP_VERSION} → $BIN_DIR_BELLHOP"
    else
        echo -e "${YELLOW}⚠ Bellhop (${BELLHOP_VERSION}) build failed (cmake exit codes: configure=$BHC_CONFIGURE_RC, build=$BHC_BUILD_RC); continuing with other components.${NC}"
        STATUS_BELLHOPCUDA="failed"
        NOTE_BELLHOPCUDA="cmake exit codes (configure=$BHC_CONFIGURE_RC, build=$BHC_BUILD_RC)"
    fi
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

cd "$OALIB_DIR"

# AT has no OpenMP directives and no C in our build path
# (misc/tslib/Bellhop/Kraken/KrakenField/Scooter are pure Fortran; the .c
# files under Matlab/ aren't built). FFLAGS-only injection; the AT root
# Makefile's `export CC=gcc CFLAGS=-g` covers the C side.
# -O1 is required: KRAKENC's root finder fails at higher optimisation on
# at/tests/Noise/Sduct (documented in the AT root Makefile).
OALIB_FFLAGS="${FORTRAN_ARCH_FLAGS} -O1 -ffast-math -funroll-loops -fomit-frame-pointer -I../misc -I../tslib"

# Default rebuild is incremental — make tracks dependencies. --force wipes
# everything for a from-scratch build (slower but bypasses any stale .mod /
# .o issues).
if [ "$FORCE" -eq 1 ]; then
    echo -e "${YELLOW}Cleaning previous builds (--force)...${NC}"
    make clean 2>/dev/null || true
fi

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
    MAKEFLAGS= \
    2>&1 | tee "${BUILD_LOG_DIR}"/oalib_build.log
OALIB_STATUS=${PIPESTATUS[0]:-1}
set -e

if [[ $OALIB_STATUS -eq 0 ]]; then
    echo -e "${GREEN}✓ OALIB build finished${NC}"
    STATUS_OALIB="ok"
    NOTE_OALIB="$BIN_DIR_OALIB"
else
    echo -e "${YELLOW}⚠ OALIB build had issues (exit $OALIB_STATUS). See ${BUILD_LOG_DIR}/oalib_build.log${NC}"
    echo -e "${YELLOW}  Continuing — any binaries that built will still be installed.${NC}"
    STATUS_OALIB="partial"
    NOTE_OALIB="see ${BUILD_LOG_DIR}/oalib_build.log"
fi
echo ""

# -------------------------
# Build OASES
# -------------------------
OASES_URL="https://acoustics.mit.edu/faculty/henrik/LAMSS/pub/Oases/oases.tar.gz"

# Pin the OASES tarball to a known-good sha256 to defend against supply-chain
# tampering on the acoustics.mit.edu download. install.sh refuses to extract a
# tarball whose digest does not match; on CI ($CI set) an empty pin is fatal,
# on dev workstations it triggers a one-time warning and proceeds.
# Recompute with: sha256sum "$OASES_TMP/oases.tar.gz" after a verified install.
OASES_EXPECTED_SHA256="5822e0f039eb97fc4c02405848d115d4f05ea6be00feb7c1bd5689560ad3731b"

if [[ "$INSTALL_OASES" != "yes" ]]; then
    echo -e "${YELLOW}=== Skipping OASES (not selected) ===${NC}"
    NOTE_OASES="not selected (rerun with --oases yes)"
    echo ""
else
echo -e "${BLUE}=== Building OASES ===${NC}"
if [ ! -d "$OASES_DIR" ]; then
    # curl + tar are only needed when fetching OASES — check here so a
    # cxx-only build without OASES selected doesn't require them.
    check_curl
    check_tar
    echo -e "${BLUE}Downloading OASES from $OASES_URL ...${NC}"
    OASES_TMP="$(mktemp -d)"
    # An abort between here and the rm at the end of this function must not
    # leak the scratch dir (plus its ~100 MB tarball).
    trap 'rm -rf "$OASES_TMP"' EXIT
    trap 'rm -rf "$OASES_TMP"; exit 130' INT TERM
    OASES_TARBALL="$OASES_TMP/oases.tar.gz"

    # --proto-redir =https blocks a redirect from downgrading to HTTP (kept as an
    # extra arg through the shared resilient-download policy).
    set +e
    robust_curl "$OASES_URL" "$OASES_TARBALL" \
        --proto '=https' --proto-redir '=https' --tlsv1.2
    OASES_CURL_RC=$?
    set -e

    if [ $OASES_CURL_RC -ne 0 ]; then
        echo -e "${RED}✗ Failed to download OASES (curl exit $OASES_CURL_RC). Skipping build.${NC}"
        STATUS_OASES="failed"
        NOTE_OASES="download failed (curl exit $OASES_CURL_RC)"
    else
        # Optional supply-chain pin: if a known-good digest is set, refuse to
        # extract a tarball that doesn't match. We never enforce on an empty
        # pin (first-install / dev workflow) but we do warn.
        # macOS ships `shasum -a 256` instead of `sha256sum`; pick whichever
        # is available so the pin enforces on either platform.
        OASES_SHA_OK=1
        SHA256_CMD=""
        if command_exists sha256sum; then
            SHA256_CMD="sha256sum"
        elif command_exists shasum; then
            SHA256_CMD="shasum -a 256"
        fi
        if [ -n "$OASES_EXPECTED_SHA256" ] && [ -n "$SHA256_CMD" ]; then
            OASES_ACTUAL_SHA=$($SHA256_CMD "$OASES_TARBALL" | awk '{print $1}')
            if [ "$OASES_ACTUAL_SHA" != "$OASES_EXPECTED_SHA256" ]; then
                echo -e "${RED}✗ OASES tarball checksum mismatch — refusing to extract.${NC}"
                echo -e "${RED}    expected: ${OASES_EXPECTED_SHA256}${NC}"
                echo -e "${RED}    got:      ${OASES_ACTUAL_SHA}${NC}"
                STATUS_OASES="failed"
                NOTE_OASES="sha256 mismatch (expected ${OASES_EXPECTED_SHA256}, got ${OASES_ACTUAL_SHA})"
                OASES_SHA_OK=0
            fi
        elif [ -n "$OASES_EXPECTED_SHA256" ] && [ -z "$SHA256_CMD" ]; then
            echo -e "${YELLOW}Neither sha256sum nor shasum found; cannot verify OASES tarball checksum. Install coreutils or perl-shasum to enable verification.${NC}"
        elif [ -z "$OASES_EXPECTED_SHA256" ]; then
            if [ -n "${CI:-}" ]; then
                echo -e "${RED}✗ OASES_EXPECTED_SHA256 is unset and \$CI is set — refusing to proceed.${NC}"
                echo -e "${RED}  Pin OASES_EXPECTED_SHA256 in install.sh before running on CI.${NC}"
                STATUS_OASES="failed"
                NOTE_OASES="OASES_EXPECTED_SHA256 unset on CI"
                OASES_SHA_OK=0
            else
                echo -e "${YELLOW}OASES_EXPECTED_SHA256 is unset; skipping checksum verification. Pin it for supply-chain integrity.${NC}"
            fi
        fi

        if [ "$OASES_SHA_OK" -eq 1 ]; then
            set +e
            tar -xzf "$OASES_TARBALL" -C "$OASES_TMP"
            OASES_TAR_RC=$?
            set -e
            if [ $OASES_TAR_RC -ne 0 ]; then
                echo -e "${RED}✗ tar extraction failed (exit $OASES_TAR_RC). Skipping OASES build.${NC}"
                STATUS_OASES="failed"
                NOTE_OASES="tar extract failed (exit $OASES_TAR_RC)"
            else
                # Locate the extracted root by finding the Makefile rather
                # than hard-coding "Oases_export/" — upstream has changed
                # tarball layouts before and silent fallthrough to a "skipped"
                # status row was masking real failures. BSD find on macOS
                # has no -printf, so derive the parent dir with dirname.
                OASES_MAKEFILE=$(find "$OASES_TMP" -maxdepth 2 -name Makefile 2>/dev/null | head -n 1)
                OASES_EXTRACTED_ROOT=""
                if [ -n "$OASES_MAKEFILE" ]; then
                    OASES_EXTRACTED_ROOT=$(dirname "$OASES_MAKEFILE")
                fi
                if [ -n "$OASES_EXTRACTED_ROOT" ] && [ -d "$OASES_EXTRACTED_ROOT" ]; then
                    mv "$OASES_EXTRACTED_ROOT" "$OASES_DIR"
                    echo -e "${GREEN}✓ OASES source downloaded and placed at $OASES_DIR${NC}"
                else
                    echo -e "${RED}✗ Could not locate Makefile in extracted archive (depth ≤ 2). Skipping OASES build.${NC}"
                    STATUS_OASES="failed"
                    NOTE_OASES="unexpected archive layout (no Makefile found)"
                fi
            fi
        fi
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

    echo -e "${BLUE}Starting OASES make (log -> ${BUILD_LOG_DIR}/oases_build.log)${NC}"
    echo -e "  - HOSTTYPE=${OASES_HOSTTYPE}  OSTYPE=${OASES_OSTYPE}"
    echo -e "  - FC_STMNT='${OASES_FC}'  FFLAGS=${OASES_FFLAGS}"

    # RANLB=ranlib: the OASES root Makefile only defines RANLIB.<host>-<os>
    # for the platforms it shipped with (linux, sun4-solaris, alpha-osf1, …).
    # There is no i386-darwin-darwin entry, so the lookup returns empty and
    # src/makefile's `$(RANLIB) $@` rule becomes ` /…/oaslib.a` — make then
    # tries to execute the archive as a command ("Permission denied"). Both
    # GNU and BSD `ranlib` produce a Mach-O/ELF archive index, so passing
    # it unconditionally is portable.
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
        RANLB=ranlib \
        oases 2>&1 | tee "${BUILD_LOG_DIR}"/oases_build.log
    OASES_STATUS=${PIPESTATUS[0]:-1}
    set -e

    if [[ $OASES_STATUS -eq 0 ]]; then
        echo -e "${GREEN}✓ OASES build completed${NC}"
    else
        echo -e "${YELLOW}⚠ OASES build had issues. See ${BUILD_LOG_DIR}/oases_build.log${NC}"
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
        echo -e "${YELLOW}No OASES executables installed. Check ${BUILD_LOG_DIR}/oases_build.log${NC}"
        STATUS_OASES="failed"
        NOTE_OASES="see ${BUILD_LOG_DIR}/oases_build.log"
    else
        echo -e "${GREEN}✓ Installed $OASES_INSTALLED OASES components${NC}"
        STATUS_OASES="ok"
        NOTE_OASES="$OASES_INSTALLED binaries → $BIN_DIR_OASES"
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

    if [ "$FORCE" -eq 1 ]; then
        echo -e "${YELLOW}Cleaning previous mpiramS builds (--force)...${NC}"
        make clean 2>/dev/null || true
    fi

    echo -e "${BLUE}Compiling mpiramS (single-processor version, double precision)...${NC}"
    # -O2 without -ffast-math: PE with complex Padé recursion accumulates
    # phase over thousands of range steps, where reassociation, reciprocal
    # approximations, and -ffinite-math-only would all be unsafe. -fopenmp
    # is required for the !$OMP loop in peramx.f90; -flto links objects with
    # link-time optimisation.
    MPIRAMS_FFLAGS="-O2 ${FORTRAN_ARCH_FLAGS} -fopenmp -I mod -Wall -fuse-linker-plugin"
    MPIRAMS_LDFLAGS="-O2 ${FORTRAN_ARCH_FLAGS} -fopenmp -flto"
    # FC=gfortran is required: GNU make's built-in default is FC=f77, which
    # is set before the Makefile's `FC ?= gfortran` runs (so ?= no-ops). On
    # Ubuntu the f77 symlink resolves to gfortran; on macOS no such alias
    # exists and the build fails with "f77: No such file or directory".
    set +e
    make FC=gfortran FFLAGS="$MPIRAMS_FFLAGS" LDFLAGS="$MPIRAMS_LDFLAGS" 2>&1 | tee "${BUILD_LOG_DIR}"/mpirams_build.log
    MPIRAMS_STATUS=${PIPESTATUS[0]:-1}
    set -e

    if [[ $MPIRAMS_STATUS -eq 0 ]] && [ -f "$MPIRAMS_DIR/s_mpiram" ]; then
        cp "$MPIRAMS_DIR/s_mpiram" "$BIN_DIR_MPIRAMS/s_mpiram"
        chmod +x "$BIN_DIR_MPIRAMS/s_mpiram"
        echo -e "${GREEN}✓ Installed mpiramS binary: s_mpiram${NC}"
        STATUS_MPIRAMS="ok"
        NOTE_MPIRAMS="$BIN_DIR_MPIRAMS"
    else
        echo -e "${YELLOW}⚠ mpiramS build failed. See ${BUILD_LOG_DIR}/mpirams_build.log${NC}"
        STATUS_MPIRAMS="failed"
        NOTE_MPIRAMS="see ${BUILD_LOG_DIR}/mpirams_build.log"
    fi
else
    echo -e "${YELLOW}mpiramS source not found at: $MPIRAMS_DIR. Skipping.${NC}"
    STATUS_MPIRAMS="skipped"
    NOTE_MPIRAMS="source missing: $MPIRAMS_DIR"
fi
echo ""

# -------------------------
# Build ramsurf (Collins RAM family: rams0.5 elastic, ramsurf1.5 rough surface)
# -------------------------
echo -e "${BLUE}=== Building ramsurf (Collins RAM family) ===${NC}"
if [ -d "$RAMSURF_DIR" ]; then
    cd "$RAMSURF_DIR"

    if [ "$FORCE" -eq 1 ]; then
        echo -e "${YELLOW}Cleaning previous ramsurf builds (--force)...${NC}"
        make clean 2>/dev/null || true
    fi

    echo -e "${BLUE}Compiling rams0.5 (elastic) / ramsurf1.5 (rough surface)...${NC}"
    # -O2 without -ffast-math: long-range PE with complex Padé recursion is
    # too sensitive to reassociation, reciprocal approximations, and
    # -ffinite-math-only for unsafe-math to be acceptable. `-std=legacy -w`
    # accepts the F77-era idioms in Calvo's sources.
    # -fdefault-real-8 -fdefault-double-8: the Collins march outside epade is
    # single precision in stock (ram.pdf section 3), which makes refining dz past
    # ~lambda/120 lose more to round-off than the finer grid gains — measured
    # 2.19 dB against KrakenC at dz=0.0125 m, against 0.11 dB in double. The
    # epade interface declarations are promoted in the sources to match
    # (third_party/MODIFICATIONS.md). These flags are REQUIRED, not optional:
    # io/ramsurf_reader.py reads tl.grid / pcomplex.bin as 8-byte records, so a
    # single-precision build produces files uacpy cannot parse.
    RAMSURF_FFLAGS="-O2 ${FORTRAN_ARCH_FLAGS} -std=legacy -w -fdefault-real-8 -fdefault-double-8"
    # FC=gfortran: same reason as mpiramS — make's built-in FC=f77 wins
    # over the Makefile's `FC ?= gfortran`, and no f77 alias exists on macOS.
    set +e
    make FC=gfortran FFLAGS="$RAMSURF_FFLAGS" 2>&1 | tee "${BUILD_LOG_DIR}"/ramsurf_build.log
    RAMSURF_STATUS=${PIPESTATUS[0]:-1}
    set -e

    if [[ $RAMSURF_STATUS -eq 0 ]]; then
        RAMSURF_INSTALLED=0
        for bin in rams0.5 ramsurf1.5; do
            if [ -f "$RAMSURF_DIR/$bin" ]; then
                cp "$RAMSURF_DIR/$bin" "$BIN_DIR_RAMSURF/$bin"
                chmod +x "$BIN_DIR_RAMSURF/$bin"
                echo -e "${GREEN}✓ Installed ramsurf binary: $bin${NC}"
                RAMSURF_INSTALLED=$((RAMSURF_INSTALLED + 1))
            else
                echo -e "${YELLOW}⚠ Expected binary not built: $bin${NC}"
            fi
        done
        if [[ $RAMSURF_INSTALLED -eq 2 ]]; then
            STATUS_RAMSURF="ok"
            NOTE_RAMSURF="$BIN_DIR_RAMSURF"
        elif [[ $RAMSURF_INSTALLED -gt 0 ]]; then
            STATUS_RAMSURF="partial"
            NOTE_RAMSURF="$RAMSURF_INSTALLED/2 binaries → $BIN_DIR_RAMSURF"
        else
            STATUS_RAMSURF="failed"
            NOTE_RAMSURF="see ${BUILD_LOG_DIR}/ramsurf_build.log"
        fi
    else
        echo -e "${YELLOW}⚠ ramsurf build failed. See ${BUILD_LOG_DIR}/ramsurf_build.log${NC}"
        STATUS_RAMSURF="failed"
        NOTE_RAMSURF="see ${BUILD_LOG_DIR}/ramsurf_build.log"
    fi
else
    echo -e "${YELLOW}ramsurf source not found at: $RAMSURF_DIR. Skipping.${NC}"
    STATUS_RAMSURF="skipped"
    NOTE_RAMSURF="source missing: $RAMSURF_DIR"
fi
echo ""

# -------------------------
# Build RAMGEO (range-dependent layered fluid PE)
# -------------------------
echo -e "${BLUE}=== Building RAMGEO (range-dependent layered fluid) ===${NC}"
if [ -d "$RAMGEO_DIR" ]; then
    cd "$RAMGEO_DIR"

    if [ "$FORCE" -eq 1 ]; then
        echo -e "${YELLOW}Cleaning previous RAMGEO builds (--force)...${NC}"
        make clean 2>/dev/null || true
    fi

    echo -e "${BLUE}Compiling ramgeo (Collins layered fluid PE)...${NC}"
    # Same numerics rationale as ramsurf: -O2 (no -ffast-math) for the
    # complex Padé recursion, -std=legacy -w for the F77-era source.
    # -fdefault-real-8 -fdefault-double-8: the Collins march outside epade is
    # single precision in stock (ram.pdf section 3), which makes refining dz past
    # ~lambda/120 lose more to round-off than the finer grid gains — measured
    # 2.19 dB against KrakenC at dz=0.0125 m, against 0.11 dB in double. The
    # epade interface declarations are promoted in the sources to match
    # (third_party/MODIFICATIONS.md). These flags are REQUIRED, not optional:
    # io/ramsurf_reader.py reads tl.grid / pcomplex.bin as 8-byte records, so a
    # single-precision build produces files uacpy cannot parse.
    RAMGEO_FFLAGS="-O2 ${FORTRAN_ARCH_FLAGS} -std=legacy -w -fdefault-real-8 -fdefault-double-8"
    # FC=gfortran: make's built-in FC=f77 would otherwise win over the
    # Makefile's `FC ?= gfortran`, and no f77 alias exists on macOS
    # (gfortran ships via the Homebrew `gcc` formula). No OpenMP — RAMGEO is
    # single-threaded, like ramsurf.
    set +e
    make FC=gfortran FFLAGS="$RAMGEO_FFLAGS" 2>&1 | tee "${BUILD_LOG_DIR}"/ramgeo_build.log
    RAMGEO_STATUS=${PIPESTATUS[0]:-1}
    set -e

    if [[ $RAMGEO_STATUS -eq 0 ]] && [ -f "$RAMGEO_DIR/ramgeo" ]; then
        cp "$RAMGEO_DIR/ramgeo" "$BIN_DIR_RAMGEO/ramgeo"
        chmod +x "$BIN_DIR_RAMGEO/ramgeo"
        echo -e "${GREEN}✓ Installed RAMGEO binary: ramgeo${NC}"
        STATUS_RAMGEO="ok"
        NOTE_RAMGEO="$BIN_DIR_RAMGEO"
    else
        echo -e "${YELLOW}⚠ RAMGEO build failed. See ${BUILD_LOG_DIR}/ramgeo_build.log${NC}"
        STATUS_RAMGEO="failed"
        NOTE_RAMGEO="see ${BUILD_LOG_DIR}/ramgeo_build.log"
    fi
else
    echo -e "${YELLOW}RAMGEO source not found at: $RAMGEO_DIR. Skipping.${NC}"
    STATUS_RAMGEO="skipped"
    NOTE_RAMGEO="source missing: $RAMGEO_DIR"
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

    # bash 3.2 (macOS default) has no associative arrays, so track seen
    # basenames as a space-padded string and test membership with a case glob.
    # find -perm -u+x replaces GNU find's -executable (absent from BSD find).
    SEEN_BHC=" "
    BHC_INSTALLED=0
    for search_root in "$BHC_OUT_DIR" "$BUILD_DIR"; do
        [ -d "$search_root" ] || continue
        for name in "${SEARCH_NAMES[@]}"; do
            while IFS= read -r -d $'\0' file; do
                base=$(basename "$file")
                case "$SEEN_BHC" in *" $base "*) continue ;; esac
                SEEN_BHC="$SEEN_BHC$base "
                cp "$file" "$BIN_DIR_BELLHOP/$base"
                chmod +x "$BIN_DIR_BELLHOP/$base"
                echo -e "  ✓ Installed bellhop binary: ${GREEN}$base${NC}"
                INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
                BHC_INSTALLED=$((BHC_INSTALLED + 1))
            done < <(find "$search_root" -type f -perm -u+x -name "$name" -print0 2>/dev/null || true)
        done
    done

    if [ "$BHC_INSTALLED" -eq 0 ]; then
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

# Every uacpy model wrapper depends on one of these binaries; missing any of
# them silently turns the corresponding wrapper into a runtime error. Treat
# them all as required and downgrade STATUS_OALIB to "failed" if any is
# absent — better a loud failure now than a confusing ExecutableNotFoundError
# at user runtime.
OALIB_REQUIRED_BASENAMES=(field.exe kraken.exe krakenc.exe bounce.exe scooter.exe sparc.exe bellhop.exe bellhop3d.exe)
OALIB_MISSING=()

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

for bn in "${OALIB_REQUIRED_BASENAMES[@]}"; do
    if [ ! -x "$BIN_DIR_OALIB/$bn" ]; then
        OALIB_MISSING+=("$bn")
    fi
done

if [ ${#OALIB_MISSING[@]} -gt 0 ]; then
    OALIB_MISSING_LIST="$(IFS=,; echo "${OALIB_MISSING[*]}")"
    echo -e "${RED}✗ OALIB build is missing required binaries: ${OALIB_MISSING_LIST}${NC}"
    STATUS_OALIB="failed"
    NOTE_OALIB="missing: ${OALIB_MISSING_LIST} (see ${BUILD_LOG_DIR}/oalib_build.log)"
fi
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
    local exe="$1"
    if [ -x "$exe" ]; then
        # Acoustics-Toolbox binaries read CLI args as env-file roots; do
        # not invoke them here.
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
# test ramsurf (just one — they share the build path)
if [ -f "$BIN_DIR_RAMSURF/ramsurf1.5" ]; then
    test_runnable "$BIN_DIR_RAMSURF/ramsurf1.5" || true
fi

echo ""

fi   # end native model builds (--no-models skips the block above)

# -------------------------
# Offline data cache (uacpy.data local backend)
# -------------------------
# Public, fetched-not-bundled datasets for the offline backend. Downloaded into
# DATA_CACHE_DIR (gitignored) only when requested via --data; like OASES, never
# committed. GEBCO + WOA23 are stable public-domain URLs; the sediment sources
# are NCEI public-domain point databases.
data_requested() {
    case ",${INSTALL_DATA_NORM}," in
        *",$1,"*) return 0 ;;
        *) return 1 ;;
    esac
}

download_gebco() {
    local dir="${DATA_CACHE_DIR}/gebco"
    mkdir -p "$dir"
    if [[ "$FORCE" != "1" ]] && ls "$dir"/*.nc >/dev/null 2>&1; then
        echo -e "${GREEN}✓ GEBCO grid already cached → ${dir}${NC}"; return 0
    fi
    echo -e "${BLUE}Downloading GEBCO 2025 grid (~7.5 GB) — this is large...${NC}"
    # Download to a temp name and rename only on success (atomic; a partial file
    # never looks like a valid cached grid).
    # Keep the partial ``.part`` on failure so a re-run resumes (robust_curl's
    # ``-C -``) instead of restarting 7.5 GB from zero.
    local tmp="${dir}/GEBCO_2025.nc.part"
    if ! robust_curl "$GEBCO_NC_URL" "$tmp"; then
        echo -e "${RED}✗ GEBCO download failed — re-run --data gebco to resume from the partial file${NC}"
        NOTE_DATA="${NOTE_DATA} gebco:failed"; return 1
    fi
    # Verify the digest before the file is promoted to the cache, the way the
    # OASES tarball is pinned. ``-C -`` resumes onto whatever ``.part`` a
    # previous run left, so a remote file that changed between runs would
    # otherwise be spliced together from two versions and installed as a valid
    # grid. A mismatch drops the .part: resuming it again could only reproduce
    # the same bad file.
    if [[ -n "$GEBCO_MD5" ]]; then
        local md5_cmd="" actual_md5=""
        if command_exists md5sum; then
            md5_cmd="md5sum"
        elif command_exists md5; then       # macOS
            md5_cmd="md5 -q"
        fi
        if [[ -n "$md5_cmd" ]]; then
            actual_md5=$($md5_cmd "$tmp" | awk '{print $1}')
            if [[ "$actual_md5" != "$GEBCO_MD5" ]]; then
                echo -e "${RED}✗ GEBCO grid md5 mismatch — refusing to install it.${NC}"
                echo -e "${RED}    expected: ${GEBCO_MD5}${NC}"
                echo -e "${RED}    got:      ${actual_md5}${NC}"
                rm -f "$tmp"
                NOTE_DATA="${NOTE_DATA} gebco:md5-mismatch"; return 1
            fi
        else
            echo -e "${YELLOW}Neither md5sum nor md5 found; cannot verify the GEBCO grid.${NC}"
        fi
    fi
    mv -f "$tmp" "${dir}/GEBCO_2025.nc"
    echo -e "${GREEN}✓ GEBCO grid ready → ${dir}${NC}"; return 0
}

download_woa23() {
    local dir="${DATA_CACHE_DIR}/woa23"; mkdir -p "$dir"
    local base="https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA"
    local ok=1 var folder p fname url header_shown=0 got=0
    for var in t s; do
        if [[ "$var" == "t" ]]; then folder="temperature"; else folder="salinity"; fi
        for p in 00 01 02 03 04 05 06 07 08 09 10 11 12; do
            fname="woa23_${WOA_DECADE}_${var}${p}_${WOA_CODE}.nc"
            if [[ "$FORCE" != "1" && -s "${dir}/${fname}" ]]; then continue; fi
            url="${base}/${folder}/netcdf/${WOA_DECADE}/${WOA_RESOLUTION}/${fname}"
            if [[ "$header_shown" == "0" ]]; then
                echo -e "${BLUE}Downloading WOA23 ${WOA_RESOLUTION}° T/S grids (annual + 12 months)...${NC}"
                header_shown=1
            fi
            # Fetch to .part and move into place only on success: an
            # interrupted transfer must never leave a truncated file the
            # -s re-run guard above would accept as complete.
            if robust_curl "$url" "${dir}/${fname}.part"; then
                mv -f "${dir}/${fname}.part" "${dir}/${fname}"
                got=1
            else
                echo -e "${RED}  ✗ ${fname}${NC}"; rm -f "${dir}/${fname}.part"; ok=0
                if [[ "${got:-0}" == "0" ]]; then
                    echo -e "${YELLOW}◐ WOA23: the first grid failed with none fetched — the server looks unreachable; stopping before the remaining files. Re-run --data woa23 when it answers.${NC}"
                    NOTE_DATA="${NOTE_DATA} woa23:unreachable"; return 1
                fi
            fi
        done
    done
    if [[ "$ok" == "1" ]]; then
        if [[ "$header_shown" == "0" ]]; then
            echo -e "${GREEN}✓ WOA23 grids already cached → ${dir}${NC}"
        else
            echo -e "${GREEN}✓ WOA23 grids ready → ${dir}${NC}"
        fi
        return 0
    fi
    echo -e "${YELLOW}◐ WOA23 partially downloaded${NC}"; NOTE_DATA="${NOTE_DATA} woa23:partial"; return 1
}

download_sediment() {
    local dir="${DATA_CACHE_DIR}/sediment"; mkdir -p "$dir"
    if [[ "$FORCE" != "1" && ( -s "${dir}/grainsize.csv" || -s "${dir}/deck41.csv" ) ]]; then
        echo -e "${GREEN}✓ Sediment samples already cached → ${dir}${NC}"; return 0
    fi
    # Automatic: the NCEI grain-size DB (G00127, public domain) is downloaded and
    # normalized to grainsize.csv by uacpy.data.download_sediment_db — if uacpy is
    # importable in this Python. Otherwise fall back to a manual-placement note.
    if python3 -c "import uacpy.data" >/dev/null 2>&1; then
        echo -e "${BLUE}Downloading + normalizing NCEI grain-size DB (~3 MB)...${NC}"
        if UACPY_INSTALL_CACHE_DIR="$dir" python3 -c "import os; import uacpy.data as d; d.download_sediment_db(cache_dir=os.environ['UACPY_INSTALL_CACHE_DIR'], verbose=True)"; then
            echo -e "${GREEN}✓ Sediment grain-size DB ready → ${dir}${NC}"; return 0
        fi
        echo -e "${YELLOW}◐ Automatic sediment download failed.${NC}"
    else
        echo -e "${YELLOW}◐ uacpy not importable here — sediment needs manual placement.${NC}"
    fi
    echo "    Fetch it from Python once uacpy is installed:"
    echo "      python -c \"import uacpy.data as d; d.download_sediment_db()\""
    echo "    or drop a CSV into ${dir}/grainsize.csv (cols: latitude, longitude, mean_phi)."
    NOTE_DATA="${NOTE_DATA} sediment:manual"
    return 1
}

download_emodnet() {
    local dir="${DATA_CACHE_DIR}/emodnet"; mkdir -p "$dir"
    # .npz, not the .pkl this used to be: the index was unpickled on every read,
    # which runs whatever code the file contains. A cache still holding the .pkl
    # must fall through and rebuild — emodnet_local refuses to read it.
    if [[ "$FORCE" != "1" && -s "${dir}/seabed_substrate.npz" ]]; then
        echo -e "${GREEN}✓ EMODnet seabed substrate already cached → ${dir}${NC}"; return 0
    fi
    # EMODnet Geology seabed substrate (Folk 5cl, 1:1M, CC-BY) is paged from the
    # public WFS and stored as a local polygon index by emodnet_local.
    if python3 -c "import uacpy.data.emodnet_local" >/dev/null 2>&1; then
        echo -e "${BLUE}Downloading EMODnet seabed substrate (European seas)...${NC}"
        if UACPY_INSTALL_CACHE_DIR="$dir" python3 -c "import os; from uacpy.data import emodnet_local; emodnet_local.download_emodnet_db(cache_dir=os.environ['UACPY_INSTALL_CACHE_DIR'], verbose=True)"; then
            echo -e "${GREEN}✓ EMODnet seabed substrate ready → ${dir}${NC}"; return 0
        fi
        echo -e "${YELLOW}◐ Automatic EMODnet download failed.${NC}"
    else
        echo -e "${YELLOW}◐ uacpy (shapely) not importable here — EMODnet skipped.${NC}"
    fi
    echo "    Fetch it from Python once uacpy is installed:"
    echo "      python -c \"from uacpy.data import emodnet_local; emodnet_local.download_emodnet_db()\""
    NOTE_DATA="${NOTE_DATA} emodnet:manual"
    return 1
}

download_coastline() {
    local dir="${DATA_CACHE_DIR}/coastline"; mkdir -p "$dir"
    # All three resolutions, or the component rebuilds: an interrupted run
    # can leave one file of the set, and a one-file check would accept that
    # partial set as cached on every later install.
    if [[ "$FORCE" != "1" && -s "${dir}/ne_110m_land.geojson" \
          && -s "${dir}/ne_50m_land.geojson" \
          && -s "${dir}/ne_10m_land.geojson" ]]; then
        echo -e "${GREEN}✓ Coastline polygons already cached → ${dir}${NC}"; return 0
    fi
    if python3 -c "import uacpy.visualization.basemap" >/dev/null 2>&1; then
        echo -e "${BLUE}Downloading Natural Earth coastline (public domain)...${NC}"
        if UACPY_INSTALL_CACHE_DIR="$dir" python3 -c "import os; from uacpy.visualization.basemap import download_coastline; download_coastline(cache_dir=os.environ['UACPY_INSTALL_CACHE_DIR'], verbose=True)"; then
            echo -e "${GREEN}✓ Coastline polygons ready → ${dir}${NC}"; return 0
        fi
        echo -e "${YELLOW}◐ Automatic coastline download failed.${NC}"
    else
        echo -e "${YELLOW}◐ uacpy not importable here — coastline skipped.${NC}"
    fi
    echo "    Fetch it from Python once uacpy is installed:"
    echo "      python -c \"from uacpy.visualization.basemap import download_coastline; download_coastline()\""
    NOTE_DATA="${NOTE_DATA} coastline:manual"
    return 1
}

download_globsed() {
    local dir="${DATA_CACHE_DIR}/globsed"; mkdir -p "$dir"
    if [[ "$FORCE" != "1" && -s "${dir}/GlobSed-v3.nc" ]]; then
        echo -e "${GREEN}✓ GlobSed grid already cached → ${dir}${NC}"; return 0
    fi
    # curl, not python: NCEI throttles urllib to a trickle but serves curl fast.
    echo -e "${BLUE}Downloading GlobSed v3 sediment thickness (~11 MB)...${NC}"
    local tmp="${dir}/GlobSed-v3.nc.part"
    if ! robust_curl "$GLOBSED_NC_URL" "$tmp"; then
        echo -e "${RED}✗ GlobSed download failed${NC}"; rm -f "$tmp"
        NOTE_DATA="${NOTE_DATA} globsed:failed"; return 1
    fi
    mv -f "$tmp" "${dir}/GlobSed-v3.nc"
    echo -e "${GREEN}✓ GlobSed grid ready → ${dir}${NC}"; return 0
}

download_crust1() {
    local dir="${DATA_CACHE_DIR}/crust1"; mkdir -p "$dir"
    # All four grids, or the component rebuilds: extraction writes them
    # one by one, so an interrupted run can leave a partial set that a
    # one-file check would accept as cached forever.
    if [[ "$FORCE" != "1" && -s "${dir}/crust1.bnds" \
          && -s "${dir}/crust1.vp" && -s "${dir}/crust1.vs" \
          && -s "${dir}/crust1.rho" ]]; then
        echo -e "${GREEN}✓ CRUST1.0 grids already cached → ${dir}${NC}"; return 0
    fi
    if python3 -c "import uacpy.data.crust1_local" >/dev/null 2>&1; then
        echo -e "${BLUE}Downloading CRUST1.0 layered crustal model (~1 MB, no formal licence)...${NC}"
        if UACPY_INSTALL_CACHE_DIR="$dir" python3 -c "import os; from uacpy.data import crust1_local; crust1_local.download_crust1_db(cache_dir=os.environ['UACPY_INSTALL_CACHE_DIR'], verbose=True)"; then
            echo -e "${GREEN}✓ CRUST1.0 grids ready → ${dir}${NC}"; return 0
        fi
        echo -e "${YELLOW}◐ Automatic CRUST1.0 download failed.${NC}"
    else
        echo -e "${YELLOW}◐ uacpy not importable here — CRUST1.0 skipped.${NC}"
    fi
    echo "    Fetch it from Python once uacpy is installed:"
    echo "      python -c \"from uacpy.data import crust1_local; crust1_local.download_crust1_db()\""
    NOTE_DATA="${NOTE_DATA} crust1:manual"
    return 1
}

download_diesing() {
    local dir="${DATA_CACHE_DIR}/diesing"; mkdir -p "$dir"
    if [[ "$FORCE" != "1" && -s "${dir}/lithology_classes.tif" ]]; then
        echo -e "${GREEN}✓ Diesing lithology map already cached → ${dir}${NC}"; return 0
    fi
    # Diesing 2020 global deep-sea seafloor lithology (CC-BY); zip downloaded and
    # the lithology raster extracted by diesing_local (PANGAEA serves urllib fast).
    if python3 -c "import uacpy.data.diesing_local" >/dev/null 2>&1; then
        echo -e "${BLUE}Downloading Diesing 2020 deep-sea lithology (CC-BY, ~40 MB)...${NC}"
        if UACPY_INSTALL_CACHE_DIR="$dir" python3 -c "import os; from uacpy.data import diesing_local; diesing_local.download_diesing_db(cache_dir=os.environ['UACPY_INSTALL_CACHE_DIR'], verbose=True)"; then
            echo -e "${GREEN}✓ Diesing lithology map ready → ${dir}${NC}"; return 0
        fi
        echo -e "${YELLOW}◐ Automatic Diesing download failed.${NC}"
    else
        echo -e "${YELLOW}◐ uacpy not importable here — Diesing skipped.${NC}"
    fi
    echo "    Fetch it from Python once uacpy is installed:"
    echo "      python -c \"from uacpy.data import diesing_local; diesing_local.download_diesing_db()\""
    echo "    (querying it also needs uacpy's pyproj)"
    NOTE_DATA="${NOTE_DATA} diesing:manual"
    return 1
}

download_seaice() {
    local dir="${DATA_CACHE_DIR}/seaice"; mkdir -p "$dir"
    # .npz, not the .pkl this used to be — see download_emodnet.
    if [[ "$FORCE" != "1" && -s "${dir}/seaice_climatology.npz" ]]; then
        echo -e "${GREEN}✓ Sea-ice climatology already cached → ${dir}${NC}"; return 0
    fi
    # Builds a monthly climatology from NSIDC Sea Ice Index grids (needs tifffile).
    if python3 -c "import uacpy.data.seaice_local, tifffile" >/dev/null 2>&1; then
        echo -e "${BLUE}Building NSIDC sea-ice monthly climatology (downloads ~120 grids; a few minutes)...${NC}"
        if UACPY_INSTALL_CACHE_DIR="$dir" python3 -c "import os; from uacpy.data import seaice_local; seaice_local.download_seaice_db(cache_dir=os.environ['UACPY_INSTALL_CACHE_DIR'], verbose=True)"; then
            echo -e "${GREEN}✓ Sea-ice climatology ready → ${dir}${NC}"; return 0
        fi
        echo -e "${YELLOW}◐ Automatic sea-ice build failed.${NC}"
    else
        echo -e "${YELLOW}◐ uacpy (tifffile) not importable here — sea ice skipped.${NC}"
    fi
    echo "    Build it from Python once uacpy is installed:"
    echo "      python -c \"from uacpy.data import seaice_local; seaice_local.download_seaice_db()\""
    NOTE_DATA="${NOTE_DATA} seaice:manual"
    return 1
}

download_glodap() {
    local dir="${DATA_CACHE_DIR}/glodap"; mkdir -p "$dir"
    if [[ "$FORCE" != "1" && -s "${dir}/GLODAPv2.2016b.pHtsinsitutp.nc" ]]; then
        echo -e "${GREEN}✓ GLODAP pH grid already cached → ${dir}${NC}"; return 0
    fi
    # GLODAPv2.2016b mapped seawater pH (CC-BY); the ~211 MB product tarball is
    # downloaded and only the in-situ pH grid extracted by glodap_local.
    if python3 -c "import uacpy.data.glodap_local" >/dev/null 2>&1; then
        echo -e "${BLUE}Downloading GLODAPv2.2016b mapped pH (CC-BY, ~211 MB tarball)...${NC}"
        if UACPY_INSTALL_CACHE_DIR="$dir" python3 -c "import os; from uacpy.data import glodap_local; glodap_local.download_glodap_db(cache_dir=os.environ['UACPY_INSTALL_CACHE_DIR'], verbose=True)"; then
            echo -e "${GREEN}✓ GLODAP pH grid ready → ${dir}${NC}"; return 0
        fi
        echo -e "${YELLOW}◐ Automatic GLODAP download failed.${NC}"
    else
        echo -e "${YELLOW}◐ uacpy not importable here — GLODAP skipped.${NC}"
    fi
    echo "    Fetch it from Python once uacpy is installed:"
    echo "      python -c \"from uacpy.data import glodap_local; glodap_local.download_glodap_db()\""
    NOTE_DATA="${NOTE_DATA} glodap:manual"
    return 1
}

download_wind() {
    local dir="${DATA_CACHE_DIR}/wind"; mkdir -p "$dir"
    if [[ "$FORCE" != "1" && -s "${dir}/wind_climatology.npz" ]]; then
        echo -e "${GREEN}✓ NBS wind climatology already cached → ${dir}${NC}"; return 0
    fi
    # Builds a monthly NBS 10 m wind-speed climatology from the CoastWatch ERDDAP
    # monthly grids (needs netCDF4). Public domain.
    if python3 -c "import uacpy.data.wind_local, netCDF4" >/dev/null 2>&1; then
        echo -e "${BLUE}Building NBS wind monthly climatology (downloads ERDDAP grids; a few minutes)...${NC}"
        if UACPY_INSTALL_CACHE_DIR="$dir" python3 -c "import os; from uacpy.data import wind_local; wind_local.download_wind_db(cache_dir=os.environ['UACPY_INSTALL_CACHE_DIR'], verbose=True)"; then
            echo -e "${GREEN}✓ NBS wind climatology ready → ${dir}${NC}"; return 0
        fi
        echo -e "${YELLOW}◐ Automatic wind build failed.${NC}"
    else
        echo -e "${YELLOW}◐ uacpy (netCDF4) not importable here — wind skipped.${NC}"
    fi
    echo "    Build it from Python once uacpy is installed:"
    echo "      python -c \"from uacpy.data import wind_local; wind_local.download_wind_db()\""
    NOTE_DATA="${NOTE_DATA} wind:manual"
    return 1
}

download_graw() {
    local dir="${DATA_CACHE_DIR}/graw"; mkdir -p "$dir"
    local out="${dir}/Dataset_S2.nc"
    if [[ "$FORCE" != "1" && -s "$out" ]]; then
        echo -e "${GREEN}✓ Graw density grid already cached → ${dir}${NC}"; return 0
    fi
    # Graw 2021 predicted global seabed bulk density (Zenodo, CC-BY 4.0, ~37 MB).
    echo -e "${BLUE}Downloading Graw 2021 seabed density grid (~37 MB)...${NC}"
    # Staged .part + robust_curl (the shared download policy): a truncated
    # final file would pass the -s guard above on every later run.
    if robust_curl "https://zenodo.org/records/3762390/files/Dataset_S2.nc" "${out}.part"; then
        mv -f "${out}.part" "$out"
        echo -e "${GREEN}✓ Graw density grid ready → ${dir}${NC}"; return 0
    fi
    rm -f "${out}.part"
    echo "    Fetch it from Python once uacpy is installed:"
    echo "      python -c \"from uacpy.data import graw_local; graw_local.download_graw_db()\""
    NOTE_DATA="${NOTE_DATA} graw:manual"
    return 1
}

case "$INSTALL_DATA" in
    all) INSTALL_DATA_NORM="${DATA_IDS// /,}" ;;
    no|"") INSTALL_DATA_NORM="" ;;
    # Already reduced to known ids by validate_data_selection.
    *) INSTALL_DATA_NORM="$INSTALL_DATA" ;;
esac

if [[ -n "$INSTALL_DATA_NORM" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
        echo -e "${RED}--data needs curl, which was not found. Skipping data download.${NC}"
        STATUS_DATA="failed"; NOTE_DATA="curl not found"
    else
        mkdir -p "$DATA_CACHE_DIR"
        DATA_OK=1
        data_requested gebco     && { download_gebco     || DATA_OK=0; }
        data_requested woa23     && { download_woa23     || DATA_OK=0; }
        data_requested sediment  && { download_sediment  || DATA_OK=0; }
        data_requested emodnet   && { download_emodnet   || DATA_OK=0; }
        data_requested coastline && { download_coastline || DATA_OK=0; }
        data_requested globsed   && { download_globsed   || DATA_OK=0; }
        data_requested crust1    && { download_crust1    || DATA_OK=0; }
        data_requested diesing   && { download_diesing   || DATA_OK=0; }
        data_requested seaice    && { download_seaice    || DATA_OK=0; }
        data_requested glodap    && { download_glodap    || DATA_OK=0; }
        data_requested wind      && { download_wind      || DATA_OK=0; }
        data_requested graw      && { download_graw      || DATA_OK=0; }
        if [[ "$DATA_OK" == "1" ]]; then STATUS_DATA="ok"; else STATUS_DATA="partial"; fi
        echo -e "${BLUE}Data cache: ${DATA_CACHE_DIR}${NC}"
    fi
fi

# -------------------------
# Final harmonized summary
# -------------------------
# If the GPU/CXX build wasn't selected, mark it explicitly so the row prints
# with a useful "skipped" reason instead of an empty default.
if [[ "$BUILD_MODELS" == "1" && "$BELLHOP_VERSION" == "fortran" ]]; then
    NOTE_BELLHOPCUDA="not selected (rerun with --bellhop cxx|cuda)"
fi

# Decide overall outcome: any "failed" row → failed, else any "partial" row →
# partial, otherwise ok. A component that failed outright is not a warning.
OVERALL="ok"
for s in "$STATUS_OALIB" "$STATUS_BELLHOPCUDA" "$STATUS_OASES" \
         "$STATUS_MPIRAMS" "$STATUS_RAMSURF" "$STATUS_RAMGEO" \
         "$STATUS_DATA"; do
    if [[ "$s" == "failed" ]]; then
        OVERALL="failed"
        break
    elif [[ "$s" == "partial" ]]; then
        OVERALL="partial"
    fi
done

# -------------------------------------------------------------------
# Durable build record
# -------------------------------------------------------------------
# uacpy/bin/ is gitignored and machine-specific, and the per-run logs live
# under the mktemp dir allocated at the top of this file — deliberately, since
# a fixed /tmp name collides between users on a shared machine and can be
# pre-created as a symlink — which $TMPDIR clears on reboot. So "which
# compiler, which flags, which commit produced these binaries" has no on-disk
# answer the next morning, and that is the question a numerical result that
# will not reproduce on another machine asks first. Write the answer next to
# the binaries and copy the logs alongside it.
tool_version() {
    local tool="$1"
    if command_exists "$tool"; then
        "$tool" --version 2>/dev/null | head -1 || true
    else
        echo "not installed"
    fi
}

# Best-effort throughout, and called with `|| true`: the record is strictly
# less important than the build it records. Under `set -e` an unwritable
# BIN_ROOT — a root-owned uacpy/bin/ left by an earlier `sudo ./install.sh`, a
# read-only checkout, a full disk — would otherwise abort a run in which every
# component built, skipping the summary and exiting non-zero on a complete
# install.
write_build_record() {
    local info="${BIN_ROOT}/BUILD_INFO.txt"
    local rc=0
    ensure_dir "$BIN_ROOT" 2>/dev/null || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo -e "${YELLOW}Cannot create ${BIN_ROOT}; build record skipped${NC}"
        return 0
    fi
    # `2>/dev/null` precedes `> "$info"` so a refused output redirect is silent
    # here and reported by the line below instead of as a bare "Permission
    # denied"; and the status is captured with `|| rc=$?` rather than tested by
    # `if !`, which does not see a failed redirect on a compound command.
    {
        echo "uacpy build record — written by install.sh"
        echo "written        : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "host           : $(uname -srm)"
        echo "checkout       : ${SCRIPT_DIR}"
        echo "outcome        : ${OVERALL}"
        echo ""
        echo "[toolchain]"
        echo "gfortran       : $(tool_version gfortran)"
        echo "gcc            : $(tool_version gcc)"
        echo "g++            : $(tool_version g++)"
        echo "cmake          : $(tool_version cmake)"
        echo "make           : $(tool_version make)"
        echo "nvcc           : $(tool_version nvcc)"
        echo ""
        echo "[flags]"
        echo "arch           : ${FORTRAN_ARCH_FLAGS:-<unset>}"
        echo "OALIB FFLAGS   : ${OALIB_FFLAGS:-<not built>}"
        echo "mpiramS FFLAGS : ${MPIRAMS_FFLAGS:-<not built>}"
        echo "ramsurf FFLAGS : ${RAMSURF_FFLAGS:-<not built>}"
        echo "ramgeo FFLAGS  : ${RAMGEO_FFLAGS:-<not built>}"
        echo ""
        echo "[components]"
        echo "OALIB          : ${STATUS_OALIB}"
        echo "bellhopcuda    : ${STATUS_BELLHOPCUDA} (${BELLHOP_VERSION:-<unset>})"
        echo "mpiramS        : ${STATUS_MPIRAMS}"
        echo "ramsurf        : ${STATUS_RAMSURF}"
        echo "ramgeo         : ${STATUS_RAMGEO}"
        echo "OASES          : ${STATUS_OASES}"
        echo "data cache     : ${STATUS_DATA}"
        echo ""
        echo "[provenance]"
        echo "OASES tarball sha256 (expected): ${OASES_EXPECTED_SHA256:-<not fetched>}"
        echo "submodules:"
        if command_exists git && [ -d "${SCRIPT_DIR}/.git" ]; then
            echo "  HEAD $(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
            git -C "$SCRIPT_DIR" submodule status --recursive 2>/dev/null \
                | sed 's/^/  /' || echo "  (submodule status unavailable)"
        else
            echo "  (not a git checkout, or git is not installed)"
        fi
        echo ""
        echo "Vendored-source modifications and their rationale:"
        echo "  uacpy/third_party/MODIFICATIONS.md"
    } 2>/dev/null > "$info" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo -e "${YELLOW}Cannot write ${info}; build record skipped${NC}"
        return 0
    fi
    BUILD_RECORD_WRITTEN="$info"
    echo -e "${BLUE}Build record: ${info}${NC}"

    # The logs themselves, copied out of the mktemp dir that a reboot clears.
    # Staged and swapped: removing the previous copy before a `cp` that is
    # allowed to fail would leave the user with neither.
    if [ -d "$BUILD_LOG_DIR" ]; then
        local staged="${BIN_ROOT}/build_logs.incoming"
        rm -rf "$staged" 2>/dev/null || true
        if cp -a "$BUILD_LOG_DIR" "$staged" 2>/dev/null; then
            rm -rf "${BIN_ROOT}/build_logs" 2>/dev/null || true
            if mv "$staged" "${BIN_ROOT}/build_logs" 2>/dev/null; then
                BUILD_LOGS_COPIED="${BIN_ROOT}/build_logs"
            else
                BUILD_LOGS_COPIED="$staged"
            fi
        else
            rm -rf "$staged" 2>/dev/null || true
        fi
    fi
}

write_build_record || true

if [[ "$OVERALL" == "ok" ]]; then
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  UACPY installation completed${NC}"
    echo -e "${GREEN}============================================${NC}"
elif [[ "$OVERALL" == "partial" ]]; then
    echo -e "${YELLOW}============================================${NC}"
    echo -e "${YELLOW}  UACPY installation finished with warnings${NC}"
    echo -e "${YELLOW}============================================${NC}"
else
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}  UACPY installation FAILED${NC}"
    echo -e "${RED}============================================${NC}"
fi
echo ""
echo "Component summary:"
print_status_row "OALIB (Fortran)"   "$STATUS_OALIB"      "$NOTE_OALIB"
print_status_row "Bellhop (cxx/cuda)" "$STATUS_BELLHOPCUDA" "$NOTE_BELLHOPCUDA"
print_status_row "mpiramS (PE)"      "$STATUS_MPIRAMS"    "$NOTE_MPIRAMS"
print_status_row "Collins RAM family" "$STATUS_RAMSURF"   "$NOTE_RAMSURF"
print_status_row "RAMGEO (RD layered)" "$STATUS_RAMGEO"   "$NOTE_RAMGEO"
print_status_row "OASES suite"       "$STATUS_OASES"      "$NOTE_OASES"
print_status_row "Offline data cache" "$STATUS_DATA"      "$NOTE_DATA"
echo ""
echo -e "${BLUE}Notes:${NC}"
echo "  - OALIB row covers Bellhop (Fortran), Kraken, KrakenC, Bounce, Scooter, SPARC, KrakenField."
echo "  - Per-build logs (this run): ${BUILD_LOG_DIR}/{oalib,oases,mpirams,ramsurf,ramgeo}_build.log"
if [[ -n "$BUILD_LOGS_COPIED" ]]; then
    echo "  - Copied durably to: ${BUILD_LOGS_COPIED}/"
else
    echo "  - Not copied durably: \$TMPDIR above is cleared on reboot."
fi
if [[ -n "$BUILD_RECORD_WRITTEN" ]]; then
    echo "  - Build record: ${BUILD_RECORD_WRITTEN}"
else
    echo "  - Build record: not written (${BIN_ROOT} is not writable)."
fi
echo ""
echo "Quick test:"
echo "  cd uacpy && python -c \"import uacpy; print(uacpy.__version__)\""
echo "  python uacpy/examples/example_01_basic_shallow_water.py"
echo ""

if [[ "$OVERALL" != "ok" ]]; then
    echo -e "${RED}One or more components ${OVERALL}. Exiting non-zero.${NC}"
    exit 1
fi
exit 0

