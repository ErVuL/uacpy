@echo off
REM install.bat - Build and install all UACPY native model binaries (Windows)
REM
REM Compiles the following acoustic propagation models:
REM   - OALIB (Acoustics-Toolbox): Kraken, KrakenField, Scooter, SPARC, Bellhop, Bounce (Fortran)
REM   - BellhopCUDA: C++/CUDA Bellhop (optional, for GPU acceleration)
REM   - OASES: Ocean Acoustics and Seismics (optional, downloaded from MIT)
REM   - mpiramS: Parabolic Equation model (broadband PE)
REM
REM Layout produced:
REM   uacpy\bin\oalib\       - Kraken, Scooter, SPARC, Bellhop, Bounce (Fortran)
REM   uacpy\bin\bellhopcuda\ - BellhopCXX / BellhopCUDA (C++/CUDA)
REM   uacpy\bin\oases\       - OASES suite
REM   uacpy\bin\mpirams\     - mpiramS PE model
REM
REM By default runs interactively. Use -y for non-interactive mode.
REM
REM Supports: Windows (with MinGW-w64/MSYS2 or MSVC)
REM Usage: install.bat [-y] [--force] [--bellhop fortran|cxx|cuda] [-h]
REM

setlocal enabledelayedexpansion

REM -------------------------
REM Directories & defaults
REM -------------------------
set "SCRIPT_DIR=%~dp0"
set "OALIB_DIR=%SCRIPT_DIR%uacpy\third_party\Acoustics-Toolbox"
set "BHC_DIR=%SCRIPT_DIR%uacpy\third_party\bellhopcuda"
set "OASES_DIR=%SCRIPT_DIR%uacpy\third_party\oases"
set "MPIRAMS_DIR=%SCRIPT_DIR%uacpy\third_party\mpiramS"
set "BIN_ROOT=%SCRIPT_DIR%uacpy\bin"

set "BIN_DIR_OALIB=%BIN_ROOT%\oalib"
set "BIN_DIR_BELLHOP=%BIN_ROOT%\bellhopcuda"
set "BIN_DIR_OASES=%BIN_ROOT%\oases"
set "BIN_DIR_MPIRAMS=%BIN_ROOT%\mpirams"

REM Default behavior: interactive
set "AUTO_YES=0"
set "FORCE=0"
set "BELLHOP_VERSION="
set "INSTALL_OASES="
set "ENABLE_OPENMP=1"

REM -------------------------
REM Parse CLI args
REM -------------------------
:parse_args
if "%~1"=="" goto :done_args
if "%~1"=="-y" (
    set "AUTO_YES=1"
    shift
    goto :parse_args
)
if "%~1"=="--yes" (
    set "AUTO_YES=1"
    shift
    goto :parse_args
)
if "%~1"=="--force" (
    set "FORCE=1"
    shift
    goto :parse_args
)
if "%~1"=="--bellhop" (
    shift
    if "%~1"=="" (
        echo [31m--bellhop requires an argument: fortran^|cxx^|cuda[0m
        exit /b 1
    )
    set "BELLHOP_VERSION=%~1"
    shift
    goto :parse_args
)
if "%~1"=="--oases" (
    shift
    if "%~1"=="" (
        echo [31m--oases requires an argument: yes^|no[0m
        exit /b 1
    )
    set "INSTALL_OASES=%~1"
    shift
    goto :parse_args
)
if "%~1"=="-h" goto :print_help
if "%~1"=="--help" goto :print_help
echo [33mWarning: Unknown argument: %~1[0m
shift
goto :parse_args

:print_help
echo Usage: install.bat [options]
echo.
echo Builds all native acoustic model binaries for UACPY.
echo By default, runs interactively and prompts for options.
echo.
echo Options:
echo   -y, --yes            Non-interactive mode - assume yes and auto-detect
echo   --force              Force rebuild even if binaries exist
echo   --bellhop [fortran^|cxx^|cuda]
echo                        Pre-select which Bellhop variant to build *in
echo                        addition to* Fortran Bellhop. Fortran Bellhop is
echo                        always built as the reference implementation.
echo                          fortran - Fortran only (no C++/CUDA build)
echo                          cxx     - Fortran + C++ Bellhop (CPU)
echo                          cuda    - Fortran + CUDA Bellhop (GPU)
echo   --oases [yes^|no]     Download and build OASES (MIT, distributed separately):
echo                          yes - Download from MIT and build OAST/OASN/OASR/OASP
echo                          no  - Skip OASES entirely
echo   -h, --help           Show this help
echo.
echo Examples:
echo   install.bat                     # Interactive: prompts for choices
echo   install.bat -y                  # Auto-detect everything, no prompts
echo   install.bat --bellhop cuda      # Use CUDA Bellhop, prompt for rest
echo   install.bat --oases no          # Skip OASES, prompt for rest
exit /b 0

:done_args

echo ============================================
echo   UACPY Model Installer
echo ============================================
echo.
echo Detected OS: Windows

if "%AUTO_YES%"=="1" (
    echo Mode: [33mNon-interactive (auto-detect)[0m
) else (
    echo Mode: [32mInteractive[0m
)
echo.

REM -------------------------
REM Choose Bellhop variant
REM -------------------------

REM Validate --bellhop argument if provided
if not "%BELLHOP_VERSION%"=="" (
    if "%BELLHOP_VERSION%"=="fortran" goto :bellhop_chosen
    if "%BELLHOP_VERSION%"=="cxx" goto :bellhop_chosen
    if "%BELLHOP_VERSION%"=="cuda" goto :bellhop_chosen
    echo [33mInvalid --bellhop argument: %BELLHOP_VERSION%. Ignoring.[0m
    set "BELLHOP_VERSION="
)

REM Non-interactive: auto-detect best available
if "%AUTO_YES%"=="1" (
    where nvcc >nul 2>&1
    if !errorlevel! equ 0 (
        set "BELLHOP_VERSION=cuda"
    ) else (
        where cmake >nul 2>&1
        if !errorlevel! equ 0 (
            where g++ >nul 2>&1
            if !errorlevel! equ 0 (
                set "BELLHOP_VERSION=cxx"
            ) else (
                where cl >nul 2>&1
                if !errorlevel! equ 0 (
                    set "BELLHOP_VERSION=cxx"
                ) else (
                    set "BELLHOP_VERSION=fortran"
                )
            )
        ) else (
            set "BELLHOP_VERSION=fortran"
        )
    )
    goto :bellhop_chosen
)

REM Interactive: prompt user
if "%BELLHOP_VERSION%"=="" (
    call :prompt_bellhop_version
)

:bellhop_chosen
echo Selected Bellhop variant: [32m%BELLHOP_VERSION%[0m
echo.

REM -------------------------
REM Choose whether to install OASES
REM -------------------------
REM OASES is not redistributable with UACPY and must be downloaded from MIT.
REM We ask up-front so the user can opt out before any network traffic or
REM build time is spent.

REM Validate --oases argument if provided
if not "%INSTALL_OASES%"=="" (
    if /i "%INSTALL_OASES%"=="yes" (
        set "INSTALL_OASES=yes"
        goto :oases_chosen
    )
    if /i "%INSTALL_OASES%"=="no" (
        set "INSTALL_OASES=no"
        goto :oases_chosen
    )
    echo [33mInvalid --oases argument: %INSTALL_OASES%. Ignoring.[0m
    set "INSTALL_OASES="
)

REM Non-interactive: default to yes (preserve previous behavior)
if "%AUTO_YES%"=="1" (
    set "INSTALL_OASES=yes"
    goto :oases_chosen
)

REM Interactive: prompt user
echo [34mInstall OASES (OAST, OASN, OASR, OASP)?[0m
echo   OASES is distributed separately by MIT and is not bundled with UACPY.
echo   Saying yes will download the source from acoustics.mit.edu and build it.
echo.
set /p "OASES_ANSWER=Download and install OASES? [y/N]: "
if /i "!OASES_ANSWER!"=="y" (
    set "INSTALL_OASES=yes"
) else if /i "!OASES_ANSWER!"=="yes" (
    set "INSTALL_OASES=yes"
) else (
    set "INSTALL_OASES=no"
)

:oases_chosen
if "%INSTALL_OASES%"=="yes" (
    echo OASES: [32mwill be installed[0m
) else (
    echo OASES: [33mskipped[0m
)
echo.

REM -------------------------
REM Check and install prerequisites
REM -------------------------
echo Checking prerequisites...
echo.

REM gfortran
where gfortran >nul 2>&1
if %errorlevel% neq 0 (
    goto :install_gfortran
) else (
    echo [32m+ gfortran found[0m
    for /f "tokens=*" %%v in ('gfortran --version 2^>^&1 ^| findstr /r "^GNU"') do echo   %%v
)

REM make
set "MAKE_CMD="
where make >nul 2>&1
if %errorlevel% equ 0 (
    echo [32m+ make found[0m
    set "MAKE_CMD=make"
) else (
    where mingw32-make >nul 2>&1
    if !errorlevel! equ 0 (
        echo [32m+ mingw32-make found[0m
        set "MAKE_CMD=mingw32-make"
    ) else (
        goto :install_make
    )
)

REM LAPACK — probe via gfortran's library search. If gfortran can resolve
REM liblapack.dll.a it returns the absolute path; if not, it echoes the input
REM string back unchanged. Done before the OALIB build so we fail fast with a
REM clear MSYS2/pacman hint instead of an opaque linker error 5 minutes later.
set "LAPACK_PROBE="
for /f "delims=" %%p in ('gfortran -print-file-name^=liblapack.dll.a 2^>nul') do set "LAPACK_PROBE=%%p"
if /i "%LAPACK_PROBE%"=="liblapack.dll.a" (
    set "LAPACK_PROBE="
    for /f "delims=" %%p in ('gfortran -print-file-name^=liblapack.a 2^>nul') do set "LAPACK_PROBE=%%p"
    if /i "!LAPACK_PROBE!"=="liblapack.a" set "LAPACK_PROBE="
)
if not defined LAPACK_PROBE goto :install_lapack
echo [32m+ lapack found[0m
echo   %LAPACK_PROBE%

REM CMake and C++ compiler (for bellhopcxx/bellhopcuda)
if "%BELLHOP_VERSION%"=="cxx" goto :check_cxx_tools
if "%BELLHOP_VERSION%"=="cuda" goto :check_cxx_tools
goto :check_directories

:check_cxx_tools
echo.
echo Checking C++/CUDA tools...

REM Check for CMake
where cmake >nul 2>&1
if %errorlevel% neq 0 (
    echo [31mCMake is required for building bellhopcxx/bellhopcuda[0m
    echo.
    echo Please install CMake from: https://cmake.org/download/
    if "%AUTO_YES%"=="0" pause
    exit /b 1
) else (
    echo [32m+ cmake found[0m
    for /f "tokens=*" %%v in ('cmake --version 2^>^&1 ^| findstr /r "cmake version"') do echo   %%v
)

REM Check for C++ compiler (MSVC or MinGW)
set "CXX_COMPILER="
where cl >nul 2>&1
if %errorlevel% equ 0 (
    echo [32m+ MSVC found[0m
    set "CXX_COMPILER=msvc"
) else (
    where g++ >nul 2>&1
    if !errorlevel! equ 0 (
        echo [32m+ g++ found[0m
        set "CXX_COMPILER=mingw"
    ) else (
        echo [31mC++ compiler is required (MSVC or MinGW g++)[0m
        echo Please install Visual Studio or MinGW-w64
        if "%AUTO_YES%"=="0" pause
        exit /b 1
    )
)

REM Check for CUDA if user selected CUDA version
if "%BELLHOP_VERSION%"=="cuda" (
    where nvcc >nul 2>&1
    if !errorlevel! neq 0 (
        echo [33mCUDA toolkit not found. Falling back to C++ (bellhopcxx).[0m
        set "BELLHOP_VERSION=cxx"
    ) else (
        echo [32m+ CUDA found[0m
        for /f "tokens=*" %%v in ('nvcc --version 2^>^&1 ^| findstr /r "release"') do echo   %%v
    )
)

REM bellhopcuda is a git submodule; GLM lives inside it as a nested submodule.
REM Both are populated by `git submodule update --init --recursive` from the
REM uacpy repo root. If either is missing, attempt to initialize automatically.
if exist "%BHC_DIR%\CMakeLists.txt" if exist "%BHC_DIR%\glm\glm\glm.hpp" (
    echo [32m+ bellhopcuda submodule (with GLM) initialized[0m
    goto :submodule_ok
)

where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [31mgit is required to initialize the bellhopcuda submodule[0m
    if "%AUTO_YES%"=="0" pause
    exit /b 1
)

if exist "%SCRIPT_DIR%.gitmodules" (
    echo [34mInitializing bellhopcuda + GLM submodules...[0m
    pushd "%SCRIPT_DIR%"
    git submodule update --init --recursive
    popd
)

if exist "%BHC_DIR%\CMakeLists.txt" if exist "%BHC_DIR%\glm\glm\glm.hpp" (
    echo [32m+ submodules initialized[0m
    goto :submodule_ok
)

echo [31mMissing dependency: bellhopcuda submodule[0m
echo   Run from the uacpy repo root:
echo     git submodule update --init --recursive
echo   See README.md - "Install dependencies" for the full list.
if "%AUTO_YES%"=="0" pause
exit /b 1

:submodule_ok

REM Workaround for upstream bellhopcuda issue: config/CMakeLists.txt installs a
REM clang-format pre-commit hook by copying into ${PROJECT_SOURCE_DIR}/.git/hooks/.
REM When bellhopcuda is a submodule, .git is a regular file (a "gitdir: ..."
REM pointer), so file(COPY) fails with "Not a directory". Replace the .git file
REM with a directory junction to the resolved gitdir so .git/hooks/ resolves.
if exist "%BHC_DIR%\.git\" goto :bhc_dotgit_ok
if not exist "%BHC_DIR%\.git" goto :bhc_dotgit_ok
set "BHC_GITDIR_REL="
for /f "tokens=1,*" %%a in ('type "%BHC_DIR%\.git"') do (
    if /i "%%a"=="gitdir:" set "BHC_GITDIR_REL=%%b"
)
if not defined BHC_GITDIR_REL goto :bhc_dotgit_ok
pushd "%BHC_DIR%"
pushd "%BHC_GITDIR_REL%" 2>nul
if errorlevel 1 (
    popd
    goto :bhc_dotgit_ok
)
set "BHC_GITDIR_ABS=%CD%"
popd
popd
del "%BHC_DIR%\.git" >nul 2>&1
mklink /J "%BHC_DIR%\.git" "%BHC_GITDIR_ABS%" >nul
:bhc_dotgit_ok

goto :check_directories

REM -------------------------
REM Missing prerequisites: instructions
REM -------------------------
:install_gfortran
echo [33mgfortran is required but not installed.[0m
echo.
echo To compile the Acoustics Toolbox on Windows, you need MinGW-w64 with gfortran.
echo.
echo Installation Options:
echo.
echo 1. MSYS2 (Recommended):
echo    - Download from: https://www.msys2.org/
echo    - Install MSYS2
echo    - Open MSYS2 MINGW64 terminal
echo    - Run: pacman -S mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-make
echo    - Add C:\msys64\mingw64\bin to your PATH
echo.
echo 2. MinGW-w64:
echo    - Download from: https://github.com/niXman/mingw-builds-binaries/releases
echo    - Extract and add bin directory to PATH
echo.
echo 3. TDM-GCC:
echo    - Download from: https://jmeubank.github.io/tdm-gcc/
echo    - Install with Fortran support
echo.
echo After installation, add the compiler to your PATH and run this script again.
echo.
echo See README.md - "Install dependencies" for the full list.
if "%AUTO_YES%"=="0" pause
exit /b 1

:install_make
echo [33mmake is required but not installed.[0m
echo.
echo MinGW's make is typically called 'mingw32-make' or 'make'.
echo Please ensure make is installed with your MinGW distribution.
echo.
echo If using MSYS2: pacman -S mingw-w64-x86_64-make
echo If using MinGW-w64: make.exe should be included
echo.
echo See README.md - "Install dependencies" for the full list.
if "%AUTO_YES%"=="0" pause
exit /b 1

:install_lapack
echo [33mLAPACK is required but not installed.[0m
echo.
echo Kraken/Scooter link with -llapack at build time. On Windows the
echo standard source is MSYS2's MinGW-w64 LAPACK package:
echo.
echo     pacman -S mingw-w64-x86_64-lapack
echo.
echo (run from the MSYS2 MINGW64 shell, then re-run install.bat from
echo  the same shell so gfortran can find the library).
echo.
echo See README.md - "Install dependencies" for the full list.
if "%AUTO_YES%"=="0" pause
exit /b 1

REM -------------------------
REM Check source directories
REM -------------------------
:check_directories
echo.

if not exist "%OALIB_DIR%" (
    echo [31mAcoustics-Toolbox (OALIB) not found at:[0m
    echo   %OALIB_DIR%
    echo Please place the Acoustics-Toolbox in uacpy\third_party\Acoustics-Toolbox
    if "%AUTO_YES%"=="0" pause
    exit /b 1
)
echo [32m+ Found OALIB (Acoustics-Toolbox): %OALIB_DIR%[0m

if "%BELLHOP_VERSION%"=="cxx" goto :check_bhc_dir
if "%BELLHOP_VERSION%"=="cuda" goto :check_bhc_dir
goto :create_bin_dirs

:check_bhc_dir
if not exist "%BHC_DIR%" (
    echo [31mbellhopcuda directory not found at:[0m
    echo   %BHC_DIR%
    if "%AUTO_YES%"=="0" pause
    exit /b 1
)
echo [32m+ Found bellhopcuda: %BHC_DIR%[0m

:create_bin_dirs
echo.

REM Create all bin directories. If the path already exists but isn't a real
REM directory (e.g. leftover file or broken mklink symlink), remove it first
REM so mkdir doesn't fail with "already exists".
call :ensure_dir "%BIN_DIR_OALIB%"
call :ensure_dir "%BIN_DIR_BELLHOP%"
call :ensure_dir "%BIN_DIR_OASES%"
call :ensure_dir "%BIN_DIR_MPIRAMS%"

REM -------------------------
REM Build bellhopcxx / bellhopcuda (if selected)
REM -------------------------
if "%BELLHOP_VERSION%"=="cxx" goto :build_bellhopcuda
if "%BELLHOP_VERSION%"=="cuda" goto :build_bellhopcuda
goto :build_fortran

:build_bellhopcuda
echo === Building Bellhop (%BELLHOP_VERSION%) ===
echo.

cd /d "%BHC_DIR%"

set "BUILD_DIR=%BHC_DIR%\build"
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
mkdir "%BUILD_DIR%"

REM BHC_BUILD_EXAMPLES=OFF skips bellhopcuda/examples/*.cpp — uacpy doesn't
REM use them, and at least examples\background.cpp is missing an explicit
REM #include <cstring> that newer compilers no longer transitively provide.
set "CMAKE_OPTIONS=-DCMAKE_BUILD_TYPE=Release -DBHC_ENABLE_TESTS=OFF -DBHC_BUILD_EXAMPLES=OFF"

REM bellhopcuda v1.5+ sets CMAKE_CUDA_ARCHITECTURES=native, so nvcc targets
REM the local GPU automatically — no GPU-name table or CUDA_ARCH_OVERRIDE
REM workaround needed.
if "%BELLHOP_VERSION%"=="cuda" (
    set "CMAKE_OPTIONS=!CMAKE_OPTIONS! -DBHC_ENABLE_CUDA=ON"
    echo   - CUDA support: ON ^(compute arch: native — auto-targeted^)
) else (
    set "CMAKE_OPTIONS=!CMAKE_OPTIONS! -DBHC_ENABLE_CUDA=OFF"
    echo   - CUDA support: OFF (CPU build)
)

echo Configuring CMake...
cmake -S "%BHC_DIR%" -B "%BUILD_DIR%" %CMAKE_OPTIONS%

if %errorlevel% neq 0 (
    echo [31mCMake configuration failed![0m
    if "%AUTO_YES%"=="0" pause
    exit /b 1
)

echo Building (this may take a while)...
set "CORES=%NUMBER_OF_PROCESSORS%"
if not defined CORES set "CORES=2"

cmake --build "%BUILD_DIR%" --config Release -j %CORES%

if %errorlevel% neq 0 (
    echo [31mBellhop build failed![0m
    if "%AUTO_YES%"=="0" pause
    exit /b 1
)

echo [32m+ Bellhop build finished[0m
echo.

REM -------------------------
REM Build OALIB (Acoustics-Toolbox) Fortran models
REM -------------------------
REM Always build via `make all` from the root of the toolbox. Running
REM `make -C <subdir>` directly bypasses the root Makefile's `export FC=gfortran`
REM so $(FC) falls back to GNU make's default (f77) and subdir libs like
REM misc/libmisc.a are never built.
REM
REM OALIB is built serially — the subdir Makefiles declare Fortran .mod
REM dependencies incompletely (e.g. AttenMod.o's rule doesn't list attenmod.mod
REM as a target), so -j races and fails with "No rule to make target
REM 'attenmod.mod'". Serial build is fast enough.
:build_fortran
echo === Building OALIB (Acoustics-Toolbox) ===
echo.

cd /d "%OALIB_DIR%"

set "OALIB_FFLAGS=-march=native -O2 -ffast-math -funroll-loops -fomit-frame-pointer -mtune=native -I../misc -I../tslib"
if "%ENABLE_OPENMP%"=="1" (
    set "OALIB_FFLAGS=%OALIB_FFLAGS% -fopenmp"
    set "OALIB_CFLAGS=-g -fopenmp"
) else (
    set "OALIB_CFLAGS=-g"
)

echo Cleaning previous builds...
%MAKE_CMD% clean >nul 2>&1

echo   - Compiler: gfortran
echo   - FFLAGS:   %OALIB_FFLAGS%
echo   - Parallelism: serial (upstream Makefiles race under -j)

%MAKE_CMD% -k all FC=gfortran FFLAGS="%OALIB_FFLAGS%" CC=gcc CFLAGS="%OALIB_CFLAGS%" LAPACK_LIBS=-llapack
set "OALIB_STATUS=%errorlevel%"

if %OALIB_STATUS% equ 0 (
    echo [32m+ OALIB build finished[0m
) else (
    echo [33m! OALIB build had issues (exit %OALIB_STATUS%). Continuing —[0m
    echo [33m  any binaries that built will still be installed.[0m
    echo Common issues:
    echo - Fortran syntax errors: Try updating gfortran
)
echo.

REM -------------------------
REM Build OASES
REM -------------------------
set "OASES_URL=http://acoustics.mit.edu/faculty/henrik/LAMSS/pub/Oases/oases.tar.gz"

if not "%INSTALL_OASES%"=="yes" (
    echo [33m=== Skipping OASES (not selected) ===[0m
    echo.
    goto :build_mpirams
)

echo === Building OASES ===

if not exist "%OASES_DIR%" (
    goto :download_oases
)
goto :build_oases

:download_oases
echo [34mDownloading OASES...[0m

REM curl / tar are only required when fetching OASES. Fail with a clear
REM message instead of silently skipping — silent-skip masks the real reason
REM the OAST/OASN/etc. binaries never appear.
where curl >nul 2>&1
if %errorlevel% neq 0 (
    echo [31mMissing dependency: curl[0m
    echo   curl is required to download OASES from %OASES_URL%
    echo   On MSYS2: pacman -S curl
    echo   Or download manually and extract to: %OASES_DIR%
    echo   See README.md - "Install dependencies" for the full list.
    if "%AUTO_YES%"=="0" pause
    exit /b 1
)

where tar >nul 2>&1
if %errorlevel% neq 0 (
    echo [31mMissing dependency: tar[0m
    echo   tar is required to extract the OASES archive.
    echo   On MSYS2: pacman -S tar
    echo   Or extract manually to: %OASES_DIR%
    echo   See README.md - "Install dependencies" for the full list.
    if "%AUTO_YES%"=="0" pause
    exit /b 1
)

set "OASES_TMP=%TEMP%\oases_download_%RANDOM%"
mkdir "%OASES_TMP%" 2>nul

curl -fSL "%OASES_URL%" -o "%OASES_TMP%\oases.tar.gz"
if %errorlevel% neq 0 (
    echo [31mFailed to download OASES. Skipping.[0m
    rmdir /s /q "%OASES_TMP%" 2>nul
    goto :build_mpirams
)

tar -xzf "%OASES_TMP%\oases.tar.gz" -C "%OASES_TMP%"
if %errorlevel% neq 0 (
    echo [31mFailed to extract OASES archive. Skipping.[0m
    rmdir /s /q "%OASES_TMP%" 2>nul
    goto :build_mpirams
)

REM The tarball extracts to Oases_export\; rename to match expected path
if exist "%OASES_TMP%\Oases_export" (
    move "%OASES_TMP%\Oases_export" "%OASES_DIR%" >nul
    echo [32m+ OASES source downloaded and placed at %OASES_DIR%[0m
) else (
    echo [31mUnexpected archive structure. Skipping OASES.[0m
    rmdir /s /q "%OASES_TMP%" 2>nul
    goto :build_mpirams
)

rmdir /s /q "%OASES_TMP%" 2>nul

if not exist "%OASES_DIR%" (
    echo [33mOASES download failed. Skipping OASES.[0m
    goto :build_mpirams
)

:build_oases

cd /d "%OASES_DIR%"

REM Determine platform tag for OASES directory structure
set "OASES_HOSTTYPE=i386"
set "OASES_OSTYPE=windows-windows"

set "OASES_ROOT=%OASES_DIR%"
set "OASES_BIN=%OASES_DIR%\bin\%OASES_HOSTTYPE%-%OASES_OSTYPE%"
set "OASES_LIB=%OASES_DIR%\lib\%OASES_HOSTTYPE%-%OASES_OSTYPE%"

if not exist "%OASES_BIN%" mkdir "%OASES_BIN%"
if not exist "%OASES_LIB%" mkdir "%OASES_LIB%"
if not exist "%OASES_DIR%\src\%OASES_HOSTTYPE%-%OASES_OSTYPE%" mkdir "%OASES_DIR%\src\%OASES_HOSTTYPE%-%OASES_OSTYPE%"

REM The OASES Makefile looks up per-platform compiler/flags via a key
REM "${HOSTTYPE}-${OSTYPE}". Even when that key exists, the Linux entry
REM hardcodes "FC=g77" and "-m486" flags from 1996, both unusable on modern
REM toolchains. We override FC_STMNT/CC_STMNT/FFLAGS/CFLAGS on the CLI so
REM they beat the in-Makefile platform-keyed assignments.
REM
REM Flags must be embedded in FC_STMNT (not FFLAGS): a handful of rules in
REM src/makefile invoke "$(FC)" without "$(FFLGS)", so FFLAGS never reaches
REM those files. Flags inside FC_STMNT ride along with every compile.
REM   -fallow-argument-mismatch  — demote type-mismatch errors (F77 code relies on these)
REM   -std=legacy                — re-enable F77 extensions (Hollerith, etc.)
REM   -fno-automatic             — SAVE locals by default (F77 semantics)
set "OASES_FC=gfortran -fallow-argument-mismatch -std=legacy -fno-automatic"
set "OASES_FFLAGS=-O2"
set "OASES_CFLAGS=-O2"

echo Starting OASES build...
echo   - HOSTTYPE=%OASES_HOSTTYPE%  OSTYPE=%OASES_OSTYPE%
echo   - FC_STMNT='%OASES_FC%'  FFLAGS=%OASES_FFLAGS%

%MAKE_CMD% HOSTTYPE="%OASES_HOSTTYPE%" OSTYPE="%OASES_OSTYPE%" OASES_ROOT="%OASES_DIR%" FC_STMNT="%OASES_FC%" CC_STMNT=gcc FFLAGS="%OASES_FFLAGS%" CFLAGS="%OASES_CFLAGS%" LFLAGS="" oases 2>&1
set "OASES_STATUS=%errorlevel%"

if %OASES_STATUS% equ 0 (
    echo [32m+ OASES build completed[0m
) else (
    echo [33m! OASES build had issues (non-critical, continuing)[0m
)

REM Copy OASES binaries
set "OASES_INSTALLED=0"
for %%b in (oasn2_bin oast2 oasr2 oasp2 oass2 oassp2) do (
    if exist "%OASES_BIN%\%%b.exe" (
        copy "%OASES_BIN%\%%b.exe" "%BIN_DIR_OASES%\%%b.exe" >nul
        echo   [32m+ Installed OASES binary: %%b.exe[0m
        set /a OASES_INSTALLED+=1
    ) else if exist "%OASES_BIN%\%%b" (
        copy "%OASES_BIN%\%%b" "%BIN_DIR_OASES%\%%b" >nul
        echo   [32m+ Installed OASES binary: %%b[0m
        set /a OASES_INSTALLED+=1
    )
)

REM Create compatibility copies (Windows doesn't have symlinks easily)
if exist "%BIN_DIR_OASES%\oast2.exe" if not exist "%BIN_DIR_OASES%\oast.exe" copy "%BIN_DIR_OASES%\oast2.exe" "%BIN_DIR_OASES%\oast.exe" >nul
if exist "%BIN_DIR_OASES%\oast2" if not exist "%BIN_DIR_OASES%\oast" copy "%BIN_DIR_OASES%\oast2" "%BIN_DIR_OASES%\oast" >nul
if exist "%BIN_DIR_OASES%\oasp2.exe" if not exist "%BIN_DIR_OASES%\oasp.exe" copy "%BIN_DIR_OASES%\oasp2.exe" "%BIN_DIR_OASES%\oasp.exe" >nul
if exist "%BIN_DIR_OASES%\oasp2" if not exist "%BIN_DIR_OASES%\oasp" copy "%BIN_DIR_OASES%\oasp2" "%BIN_DIR_OASES%\oasp" >nul
if exist "%BIN_DIR_OASES%\oasr2.exe" if not exist "%BIN_DIR_OASES%\oasr.exe" copy "%BIN_DIR_OASES%\oasr2.exe" "%BIN_DIR_OASES%\oasr.exe" >nul
if exist "%BIN_DIR_OASES%\oasr2" if not exist "%BIN_DIR_OASES%\oasr" copy "%BIN_DIR_OASES%\oasr2" "%BIN_DIR_OASES%\oasr" >nul
if exist "%BIN_DIR_OASES%\oasn2_bin.exe" if not exist "%BIN_DIR_OASES%\oasn.exe" copy "%BIN_DIR_OASES%\oasn2_bin.exe" "%BIN_DIR_OASES%\oasn.exe" >nul
if exist "%BIN_DIR_OASES%\oasn2_bin" if not exist "%BIN_DIR_OASES%\oasn" copy "%BIN_DIR_OASES%\oasn2_bin" "%BIN_DIR_OASES%\oasn" >nul

if %OASES_INSTALLED% equ 0 (
    echo [33mNo OASES executables installed. Build may have failed.[0m
) else (
    echo [32m+ Installed %OASES_INSTALLED% OASES components[0m
)
echo.

REM -------------------------
REM Build mpiramS (Fortran PE model)
REM -------------------------
:build_mpirams
echo === Building mpiramS (Parabolic Equation) ===

if not exist "%MPIRAMS_DIR%" (
    echo [33mmpiramS source not found at: %MPIRAMS_DIR%. Skipping.[0m
    goto :install_executables
)

cd /d "%MPIRAMS_DIR%"

REM Ensure obj/ and mod/ directories exist
if not exist "obj" mkdir obj
if not exist "mod" mkdir mod

echo Cleaning previous mpiramS builds...
%MAKE_CMD% clean >nul 2>&1

echo Compiling mpiramS (single-processor version, double precision)...
%MAKE_CMD% 2>&1
set "MPIRAMS_STATUS=%errorlevel%"

if %MPIRAMS_STATUS% equ 0 (
    if exist "%MPIRAMS_DIR%\s_mpiram" (
        copy "%MPIRAMS_DIR%\s_mpiram" "%BIN_DIR_MPIRAMS%\s_mpiram" >nul
        echo [32m+ Installed mpiramS binary: s_mpiram[0m
    ) else if exist "%MPIRAMS_DIR%\s_mpiram.exe" (
        copy "%MPIRAMS_DIR%\s_mpiram.exe" "%BIN_DIR_MPIRAMS%\s_mpiram.exe" >nul
        echo [32m+ Installed mpiramS binary: s_mpiram.exe[0m
    ) else (
        echo [33m! mpiramS build produced no executable[0m
    )
) else (
    echo [33m! mpiramS build failed (non-critical, continuing)[0m
)
echo.

REM -------------------------
REM Install executables to bin directories
REM -------------------------
:install_executables
echo === Installing executables to %BIN_ROOT% ===
echo.

set "INSTALLED_COUNT=0"

REM Install bellhopcxx/bellhopcuda artifacts (if built)
if "%BELLHOP_VERSION%"=="cxx" goto :install_bhc_bins
if "%BELLHOP_VERSION%"=="cuda" goto :install_bhc_bins
goto :install_oalib_bins

:install_bhc_bins
REM SetupCommon.cmake sets CMAKE_RUNTIME_OUTPUT_DIRECTORY = %BHC_DIR%\bin,
REM so the primary binaries land there rather than inside the build tree.
REM Search both locations for robustness.
set "BUILD_DIR=%BHC_DIR%\build"
set "BHC_OUT_DIR=%BHC_DIR%\bin"

for %%e in (bellhopcxx.exe bellhopcxx2d.exe bellhopcxx3d.exe bellhopcxxnx2d.exe bellhopcuda.exe bellhopcuda2d.exe bellhopcuda3d.exe bellhopcudanx2d.exe) do (
    if not exist "%BIN_DIR_BELLHOP%\%%e" (
        for /f "delims=" %%f in ('dir /s /b "%BHC_OUT_DIR%\%%e" 2^>nul') do (
            copy "%%f" "%BIN_DIR_BELLHOP%\%%e" >nul 2>&1
            if exist "%BIN_DIR_BELLHOP%\%%e" (
                echo   [32m+ Installed bellhop binary: %%e[0m
                set /a INSTALLED_COUNT+=1
            )
        )
    )
    if not exist "%BIN_DIR_BELLHOP%\%%e" (
        for /f "delims=" %%f in ('dir /s /b "%BUILD_DIR%\%%e" 2^>nul') do (
            copy "%%f" "%BIN_DIR_BELLHOP%\%%e" >nul 2>&1
            if exist "%BIN_DIR_BELLHOP%\%%e" (
                echo   [32m+ Installed bellhop binary: %%e[0m
                set /a INSTALLED_COUNT+=1
            )
        )
    )
)

if %INSTALLED_COUNT% equ 0 (
    echo   [33mNo bellhop (%BELLHOP_VERSION%) executables found in %BHC_OUT_DIR% or %BUILD_DIR%.[0m
)
echo.

:install_oalib_bins
echo Installing OALIB (Fortran) executables:

REM Always install these core binaries
for %%e in (KrakenField\field.exe Kraken\kraken.exe Kraken\krakenc.exe Kraken\bounce.exe Scooter\scooter.exe Scooter\sparc.exe) do (
    if exist "%OALIB_DIR%\%%e" (
        for %%f in ("%%e") do set "EXE_NAME=%%~nxf"
        copy "%OALIB_DIR%\%%e" "%BIN_DIR_OALIB%\!EXE_NAME!" >nul
        echo   [32m+ Installed: !EXE_NAME![0m
        set /a INSTALLED_COUNT+=1
    ) else (
        echo   [33m  Not found: %%e[0m
    )
)

REM Fortran Bellhop is always built and installed as the reference
REM implementation, regardless of whether a C++/CUDA variant was also selected.
for %%e in (Bellhop\bellhop.exe Bellhop\bellhop3d.exe) do (
    if exist "%OALIB_DIR%\%%e" (
        for %%f in ("%%e") do set "EXE_NAME=%%~nxf"
        copy "%OALIB_DIR%\%%e" "%BIN_DIR_OALIB%\!EXE_NAME!" >nul
        echo   [32m+ Installed: !EXE_NAME![0m
        set /a INSTALLED_COUNT+=1
    ) else (
        echo   [33m  Not found: %%e[0m
    )
)

echo.

REM Final check: at least one executable installed
if %INSTALLED_COUNT% equ 0 (
    echo [31mError: No executables were installed![0m
    if "%AUTO_YES%"=="0" pause
    exit /b 1
)

REM -------------------------
REM Sanity tests
REM -------------------------
echo === Running quick executable sanity checks ===

if exist "%BIN_DIR_OALIB%\kraken.exe" (
    "%BIN_DIR_OALIB%\kraken.exe" < nul >nul 2>&1
    echo   [32m+ Runnable: kraken.exe[0m
)

if exist "%BIN_DIR_OALIB%\bellhop.exe" (
    "%BIN_DIR_OALIB%\bellhop.exe" < nul >nul 2>&1
    echo   [32m+ Runnable: bellhop.exe[0m
)
if exist "%BIN_DIR_BELLHOP%\bellhopcxx.exe" (
    "%BIN_DIR_BELLHOP%\bellhopcxx.exe" --help >nul 2>&1
    echo   [32m+ Runnable: bellhopcxx.exe[0m
)
if exist "%BIN_DIR_BELLHOP%\bellhopcuda.exe" (
    "%BIN_DIR_BELLHOP%\bellhopcuda.exe" --help >nul 2>&1
    echo   [32m+ Runnable: bellhopcuda.exe[0m
)

if exist "%BIN_DIR_MPIRAMS%\s_mpiram.exe" (
    "%BIN_DIR_MPIRAMS%\s_mpiram.exe" < nul >nul 2>&1
    echo   [32m+ Runnable: s_mpiram.exe[0m
) else if exist "%BIN_DIR_MPIRAMS%\s_mpiram" (
    "%BIN_DIR_MPIRAMS%\s_mpiram" < nul >nul 2>&1
    echo   [32m+ Runnable: s_mpiram[0m
)

REM Test first OASES binary if present
for %%f in ("%BIN_DIR_OASES%\*") do (
    "%%f" < nul >nul 2>&1
    echo   [32m+ Runnable: %%~nxf[0m
    goto :done_oases_test
)
:done_oases_test

echo.
echo [32m============================================[0m
echo [32m  Installation completed[0m
echo [32m============================================[0m
echo.
echo Installed models:
echo   Kraken/Scooter/SPARC/Bounce:  %BIN_DIR_OALIB%

echo   Bellhop (Fortran):            %BIN_DIR_OALIB%
if "%BELLHOP_VERSION%"=="cxx"  echo   Bellhop (cxx):                %BIN_DIR_BELLHOP%
if "%BELLHOP_VERSION%"=="cuda" echo   Bellhop (cuda):               %BIN_DIR_BELLHOP%

if exist "%BIN_DIR_MPIRAMS%\s_mpiram.exe" echo   RAM (mpiramS PE):             %BIN_DIR_MPIRAMS%
if exist "%BIN_DIR_MPIRAMS%\s_mpiram" echo   RAM (mpiramS PE):             %BIN_DIR_MPIRAMS%

set "HAS_OASES=0"
for %%f in ("%BIN_DIR_OASES%\*") do set "HAS_OASES=1"
if %HAS_OASES% equ 1 echo   OASES:                        %BIN_DIR_OASES%

echo.
echo Quick test:
echo   cd uacpy ^&^& python -c "import uacpy; print(uacpy.__version__)"
echo   python uacpy\examples\example_01_basic_shallow_water.py
echo.

if "%AUTO_YES%"=="0" pause
goto :eof

REM ============================================
REM Function: ensure_dir  —  make a directory, cleaning up any non-directory
REM entry at that path first (e.g. leftover file or broken symlink).
REM ============================================
:ensure_dir
set "_ED_PATH=%~1"
if exist "%_ED_PATH%\" (
    REM Already a real directory — nothing to do.
    goto :eof
)
if exist "%_ED_PATH%" (
    echo [33mRemoving non-directory entry at: %_ED_PATH%[0m
    del /f /q "%_ED_PATH%" 2>nul
    rmdir "%_ED_PATH%" 2>nul
)
mkdir "%_ED_PATH%"
goto :eof

REM ============================================
REM Function: prompt_bellhop_version
REM ============================================
:prompt_bellhop_version
REM Fortran Bellhop is always built as the reference implementation; this
REM prompt only controls whether to *also* build a C++/CUDA variant.
echo [34mFortran Bellhop will be built.[0m
echo Additionally build a C++/CUDA variant?
echo.
echo   1^) No           - Fortran only
echo   2^) Yes, C++ CPU - Also build bellhopcxx (requires CMake + C++ compiler)

where nvcc >nul 2>&1
if %errorlevel% equ 0 (
    echo   3^) Yes, CUDA    - Also build bellhopcuda (GPU) [32m(nvcc detected)[0m
) else (
    echo   3^) Yes, CUDA    - Also build bellhopcuda (GPU) [33m(nvcc not found)[0m
)

echo.

:prompt_loop
set /p "CHOICE=Enter choice [1]: "
if "%CHOICE%"=="" set "CHOICE=1"

if "%CHOICE%"=="1" (
    set "BELLHOP_VERSION=fortran"
    goto :eof
) else if "%CHOICE%"=="2" (
    set "BELLHOP_VERSION=cxx"
    goto :eof
) else if "%CHOICE%"=="3" (
    set "BELLHOP_VERSION=cuda"
    goto :eof
) else (
    echo [33mInvalid choice; please enter 1, 2, or 3.[0m
    goto :prompt_loop
)
