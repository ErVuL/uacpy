# Modifications to Vendored Native Codebases

This document summarizes all changes applied to the original Fortran/C source
code shipped with uacpy, with exact diffs.

Patching a vendored file shifts the line numbers uacpy's comments cite into
it. See `docs/DEV.md` §9.1 for what a patch owes those citations.

---

## Collins RAM family — double-precision build

`ramgeo1.5.f`, `ramsurf1.5.f` and `rams0.5.f` are built with
`-fdefault-real-8 -fdefault-double-8` (their `Makefile`s), and the four explicit
kind declarations at the `epade` interface are promoted to match:

| file | was | now |
|---|---|---|
| `ramgeo1.5.f:451,453` | `complex*8 pd1,pd2` / `real*4 k0,c0,dr` | `complex*16` / `real*8` |
| `ramsurf1.5.f:461,463` | same | same |
| `rams0.5.f:900,901` | `complex*8 ci8,g0,pd1,pd2,nu8` / `real*4 k0,c0,dr,theta` | `complex*16` / `real*8` |

**Why.** Collins' own guide (`mpiramS/doc/ram.pdf` §3) states the stock policy —
*"The subroutines that compute the coefficients of the rational approximation are
written in double precision. Everything else is written in single precision"* —
and its limit: *"Double precision is required for both ram.f and ramp.f when the
number of depth grid points is large."* `matrc` scales its discretisation terms
as `1/dz**2` while the physical terms stay O(0.1), so on a fine grid each matrix
entry becomes a small difference of large numbers. Measured on a 250 Hz Pekeris
guide against KrakenC, refining `dz` made the answer **worse**:

| `dz` (m) | single build | double build |
|---|---|---|
| 0.1 | 0.192 dB | 0.163 dB |
| 0.025 | 0.486 dB | 0.116 dB |
| 0.0125 | **2.186 dB** | **0.106 dB** |

i.e. the convergence test both manuals prescribe returned a *diverging* sequence.
uacpy's own array widening (below) had raised the reachable grid size into that
regime. mpiramS was already double via its `kinds.f90` patch; this brings the
Collins trio in line.

**The promotions are required, not cosmetic.** `-fdefault-real-8` does not touch
an explicit `complex*8`/`real*4`, so without them the caller passes `complex*16`
to an `epade` still expecting `complex*8` and the run produces NaN from the first
range step — it does not degrade quietly.

**Consequence for readers.** `tl.grid` and `pcomplex.bin` are now 8-byte records;
`io/ramsurf_reader.py` reads `'f8'`/`'c16'` accordingly. Every RAM number moves
slightly.


## Acoustics Toolbox (Bellhop, Kraken, Scooter, Bounce, SPARC)

Vendored from https://github.com/oalib-acoustics/Acoustics-Toolbox at commit
`8b4682b` ("sync with 2024_12_25 sources" plus repo housekeeping). Two source
patches are applied:

### `KrakenField/field.f90` -- out-of-bounds sentinel fix

`EvaluateADMod` and `EvaluateCMMod` both declare their `rProf` argument as
`rProf(NProf + 1)` and use the extra element as a sentinel value
(`EvaluateAD` writes it; `EvaluateCM` reads it in loop guards — Fortran
`.AND.` does not short-circuit).  However, `ReadVector` (in
`misc/SourceReceiverPositions.f90`) only allocates `MAX(3, NProf)` elements,
so for `NProf >= 3` the access to `rProf(NProf + 1)` goes past the end of
the array.  Every range-dependent KrakenField run (coupled or adiabatic,
`n_segments >= 3`) hits this.  Submitted upstream from the ErVuL fork as
branch `fix-field-rprof-out-of-bounds`.

**Fix:** after `ReadVector` returns, reallocate `rProf` with `NProf + 1`
elements and set `rProf(NProf + 1) = HUGE(rProf(1))`.

```diff
@@ -106,6 +106,18 @@
   CALL ReadVector( NProf, RProf, 'Profile ranges, RProf', 'km' )
   RProf = RProf / 1000.0   ! convert m back to km (undoing what ReadVector did)
 
+  ! EvaluateAD and EvaluateCM declare their rProf dummy as rProf( NProf + 1 )
+  ! and use the extra element as a range sentinel (EvaluateAD writes it).
+  ! ReadVector only allocates MAX( 3, NProf ) elements, so for NProf >= 3 the
+  ! actual argument is one element too small; extend it and set the sentinel.
+  BLOCK
+    REAL (KIND=8), ALLOCATABLE :: rProfTmp( : )
+    ALLOCATE( rProfTmp( NProf + 1 ) )
+    rProfTmp( 1 : NProf ) = rProf( 1 : NProf )
+    rProfTmp( NProf + 1 ) = HUGE( rProf( 1 ) )
+    CALL MOVE_ALLOC( rProfTmp, rProf )
+  END BLOCK
+
   IF ( NProf == 1      ) THEN
      WRITE( PRTFile, * ) 'Range-independent calculation'
   ELSE
```

### `misc/interpolation.f90` -- non-terminating segment search

`interp1` walks a segment index `iseg` over the tabulated abscissa `x(1:N)`.
Valid segments are `1 : N-1`, but the right-hand search clamps at `N-2`:

```fortran
DO WHILE ( xi( I ) > x( iseg + 1 ) )
   IF ( iseg < N - 2 ) THEN
      iseg = iseg + 1
   END IF
END DO
```

Once `iseg` is clamped the `IF` stops incrementing while the `DO WHILE`
keeps testing the same condition, so **any query point in the final segment
`[x(N-1), x(N)]` loops forever**.  The left-hand search has the same defect
at `iseg == 1` for any query below `x(1)`.  Neither hangs on typical input:
a finely sampled table makes the final segment a narrow sliver, and the
callers query well inside the range.  It is reachable from
`KrakenField/field.f90:198`, which shades modes by a `.sbp` source beam
pattern at angles `RadDeg*ATAN(SQRT(kz2)/k)` — bounded by the critical
angle, so a *coarse* pattern (e.g. the three points `-90, 0, 90`) puts
`x(N-1)` at `0` deg and every mode angle then spins.  Reproduced standalone:
with `x = [-90,-45,0,45,90]`, `xi = 30` returns, `xi = 70` never does.
`Bellhop` is unaffected — `bellhop.f90:267-274` does its own `maxloc`
bracket search and never calls `interp1`.

**Fix:** move the clamp into the loop condition, so each search stops when
`iseg` can no longer move, making the last segment reachable.  Out-of-table
queries are handled by the separate `R` clamp described below.  Other callers:
`Scooter/fields.f90`,
`KrakenField/EvaluatepdqMod.f90`, `KrakenField/Evaluate3DMod.f90`.

**Why `N-1` is the intended bound, not `N-2`.**  `Bellhop/bellhop.f90:265-274`
performs the same operation — bracket a beam-pattern angle, then interpolate
linearly — but hand-rolled rather than via `interp1`, and clamps its segment
index explicitly:

```fortran
IBP = MAX( IBP, 1 )               ! don't go before beginning of table
IBP = MIN( IBP, NSBPPts - 1 )     ! don't go past end of table
```

Same codebase, same author, same task, bound `N-1`, with the comment stating
the intent.  Bellhop's own out-of-range behaviour is to extrapolate: with `IBP`
pinned at `NSBPPts-1` and an angle past the table its `s` exceeds 1, and below
the table `s` goes negative.

Note that extrapolating a beam pattern can yield a negative shading factor
(a query past the table on a decaying pattern gives e.g. `-0.167`).  That is
pre-existing AT behaviour.  The interpolation parameter is therefore clamped to
`[0, 1]` here, so an out-of-table query holds the end value instead of
extrapolating:

```diff
-       R       = ( xi( I ) - x( iseg ) ) / ( x( iseg + 1 ) - x( iseg ) )
+       R       = ( xi( I ) - x( iseg ) ) / ( x( iseg + 1 ) - x( iseg ) )
+       R       = MIN( MAX( R, 0.0D0 ), 1.0D0 )
        yi( I ) = ( 1.0 - R ) * y( iseg ) + R * y( iseg + 1 )
```

The clamp is inert for in-range queries — verified bit-identical before and
after (`xi = 30` on `x = [-90,-45,0,45,90]` gives `0.66666666666666674` either
way); it only changes queries outside `[ x(1), x(N) ]`.

The equivalent inline interpolation in `Bellhop/bellhop.f90:273` and
`Bellhop/bellhop3D.f90` is **deliberately left unclamped**.  Bellhop has three
implementations — the Fortran binary plus `bellhopcxx` / `bellhopcuda`, and
`Bellhop(backend=None)` selects CUDA first — and the port extrapolates too:
`bellhopcuda/src/common_run.hpp:61-80` returns `0` below the table and `n-1`
above, `trace.hpp:139` clamps only that *index* to `n-2`, and `trace.hpp:144`
uses `s` unclamped.  Clamping one of three back ends would make the same input
shade differently depending on `backend=`, which is worse than the uniform
pre-existing behaviour.  `interp1` has no such counterpart.

### Root `Makefile` -- intentionally unpatched

The upstream root `Makefile` bakes a CPU-specific flag into its default
`FFLAGS` (`-mcpu=apple-m2`), which does not compile on x86.  No source patch
is needed: `install.sh` passes `FFLAGS=` as a *make command-line variable*
(`make all FC=gfortran FFLAGS="..."`), which takes precedence over any
Makefile-level `export FFLAGS=` assignment, including in the sub-makes.
Building this tree outside uacpy requires passing `FFLAGS=` explicitly.

---

## ramsurf (Collins-style RAM family)

Vendored from https://github.com/quiet-oceans/ramsurf — BSD-3
(Calvo / Guelton). Files live flat at `third_party/ramsurf/`. The
Fortran solvers actually built by `install.sh` are:

- `rams0.5.f` — *elastic* PE (RAMS), flat surface, layered piecewise
  bottom with shear (cs/attns/Lamé)
- `ramsurf1.5.f` — fluid PE with variable surface zsrf(r) for rough
  surfaces / beach geometry [Collins, JASA 97 (1995)]

`ram1.5.f` (Collins's original fluid PE) is also vendored as a
reference but is not built — uacpy's RAM dispatcher uses mpiramS for
that regime. Its range dimension carries the same `mr=505` widening as the
built backends (`ram1.5.f:53`), applied without a `UACPY:` marker; its depth
dimension `mz=8000` is stock. Being unbuilt, the widening has no runtime
effect — it is recorded here so a vendor refresh does not silently drop it and
so the line is not mistaken for stock. `ramclr.f` (PostScript plotter) and the autotools
artefacts (`configure.ac`, `Makefile.am`) are kept untouched alongside
a plain top-level `Makefile` actually used for the build. LICENSE is
kept verbatim.

### `outpt` patch — complex-envelope dump

Calvo's original `outpt` writes only real TL to `tl.grid`, which discards
the phase needed to assemble a broadband transfer function. Both
`rams0.5.f` and `ramsurf1.5.f` are patched to also dump the complex PE
envelope to a parallel `pcomplex.bin`, mirroring `tl.grid`'s record
geometry. The two binaries store *different* envelopes: `ramsurf1.5.f`
writes `u·f3 / sqrt(r)`, while `rams0.5.f` writes `u / sqrt(r)` (its
`outpt` takes no `f3` argument). They also differ in the travelling-wave
carrier `exp(+i k0 r)` — baked into `u` in `rams0.5.f` (via the `g0`
march step), factored out in `ramsurf1.5.f` — so the RAM Python wrapper
applies a per-backend correction (conjugate ψ for both; an extra
`exp(-i k0 r)` for ramsurf only) before tagging the result
`phase_reference='travelling_wave'`, so every broadband-capable model
presents the same shape of H(f) to the IFFT pipeline. See the per-binary
discussion below for the full derivation.

#### `ramsurf1.5.f` diff

```diff
@@ -31,6 +31,11 @@
       open(unit=1,status='old',file='ram.in')
       open(unit=2,status='unknown',file='tl.line')
       open(unit=3,status='unknown',file='tl.grid',form='unformatted')
+c     UACPY: complex envelope u*f3/sqrt(r) per output range, sequential
+c     unformatted records mirroring tl.grid geometry. Used by uacpy's
+c     RAM dispatcher to assemble broadband H(f) by looping frequencies.
+      open(unit=11,status='unknown',file='pcomplex.bin',
+     >  form='unformatted')
@@ -50,6 +55,7 @@
       close(1)
       close(2)
       close(3)
+      close(11)
@@ -140,6 +146,7 @@
     7 continue
       write(3)lz
+      write(11)lz
@@ -414,7 +421,7 @@
       subroutine outpt(mz,mdr,ndr,ndz,iz,nzplt,lz,ir,dir,eps,r,f3,u,tlg)
-      complex ur,u(mz)
+      complex ur,u(mz),urg(mz)
       real f3(mz),tlg(mz)
@@ -431,9 +438,13 @@
       ur=u(i)*f3(i)
       j=j+1
       tlg(j)=-20.0*alog10(cabs(ur)+eps)+10.0*alog10(r+eps)
+c     UACPY: same envelope as tlg uses, with cylindrical-spreading
+c     factor included. Carrier exp(+i k0 r) is still factored out
+c     here; the Python wrapper bakes the engineering travelling-wave
+c     carrier exp(-i k0 r) in before tagging.
+      urg(j)=ur/sqrt(r+eps)
     1 continue
       write(3)(tlg(j),j=1,lz)
+      write(11)(urg(j),j=1,lz)
       end if
```

#### `rams0.5.f` diff

Same shape as `ramsurf1.5.f` — open unit 11 in the main, mirror `lz`
header into it, declare a local complex `urg(mz)` in `outpt`, store
`ur/sqrt(r+eps)`, write the array per range step, close unit 11.

Both Collins drivers consume `pcomplex.bin` via `read_pcomplex_grid`:
`uacpy.models.ram.RAM._run_collins` for narrowband `COHERENT_TL`, and
`RAM._run_collins_broadband` for `BROADBAND` / `TIME_SERIES`, which loops
the binary over the Q/T-derived frequency vector and assembles a
broadband `Field` (complex `data` over `coords={depth, range,
frequency}`). Every Collins-backend
run returns complex pressure, so a binary built from unpatched sources
does not degrade to real TL — it fails outright, with the wrapper
naming `pcomplex.bin` and telling the user to rebuild. The complex
envelope is bit-exact w.r.t. the existing `tl.grid` magnitude
(`-20·log10(|pcomplex|)` reproduces `tl.grid` to 0.0000 dB on a Pekeris
reference run).

`rams0.5.f` and `ramsurf1.5.f` store *different* envelopes despite the
identical-looking `outpt` patch: rams0.5's `solve(... g0)` multiplies
the field by `g0 = exp(i k₀ Δr)` at every range step (`rams0.5.f:849-850`),
so its `u` accumulates the full `exp(+i k₀ r)` carrier — same
convention as mpiramS' `psif`. ramsurf1.5's `solve` has no `g0`
argument (`ramsurf1.5.f:310`); the carrier is absorbed into the
matrix coefficients by the operator function
`g(x) = (1−νx)²·exp(α·log(1+x) + i σ (√(1+x)−1))` (`ramsurf1.5.f:566-567`),
so its `u` carries no `exp(+i k₀ r)`. ramsurf1.5 stores
`u · f3 / sqrt(r+eps)` (the density-jump-rescaled, range-rescaled
envelope; matches what `tlg` is computed from). Since `f3 ∈ ℝ`, the
wrapper's `conj(H) · exp(−i k₀ r)` post-multiply still recovers the
engineering travelling-wave convention.
`_run_collins_broadband` therefore branches on `kind`: rams gets
`np.conj(H)` only, ramsurf gets `np.conj(H) · exp(−i k₀(ω) r)`. After this convention bookkeeping all
three RAM backends land the IFFT peak at `r/c₀` (matching JKPS
*Computational Ocean Acoustics* §8.2 eq. 8.1–8.4 within real
waveguide modal dispersion ~20 ms).

### `ramsurf1.5.f` — enlarged array dimensions

Stock dimensions are too small for uacpy's fine Lytaev range/depth grids — the
same problem the ramgeo section describes, and `ramsurf1.5.f` is the file
ramgeo's enlargement was matched *to*. The declaration now reads:

```fortran
      parameter (mr=505,mz=20002,mp=10)   ! ramsurf1.5.f:24
```

giving a usable depth grid of `nz ≤ 20000` (the code indexes to `nz+2`, as the
other fluid backends do). The stock depth dimension is `mz=8000`, still visible
on `ram1.5.f:53` — Collins's original fluid PE, vendored alongside as an
unbuilt reference. Only `mz` on that line is stock; its `mr=505` is the same
widening the built backends carry (see above).

Upstream's own bounds checks (`Need to increase parameter …`,
`ramsurf1.5.f:124-132`) are left as they are: their conditions are written
against `nz+2` / `np` / `i`, so they keep working at the larger dimensions and
still stop the run rather than overrun. `uacpy.models.ram._COLLINS_ARRAY_LIMITS`
carries the matching per-backend limit and `tests/test_ram_backends.py` asserts
the two agree by parsing these guard expressions out of the source.

### `rams0.5.f` — enlarged array dimensions and bounds checks

Stock RAMS dimensions `parameter (mr=100,mz=10000,mp=10)` are too small for
uacpy's Lytaev grids, and unlike its siblings rams0.5 shipped with **no
bounds checks at all**, so an overrun corrupted memory instead of failing.
Both are fixed:

```diff
-      parameter (mr=100,mz=10000,mp=10)
+      parameter (mr=505,mz=40004,mp=10)
```

```diff
+      if(2*nz+4.gt.mz)then
+      write(*,*)'   Need to increase parameter mz to ',2*nz+4
+      stop
+      end if
+      if(np.gt.mp)then
+      write(*,*)'   Need to increase parameter mp to ',np
+      stop
+      end if
+      if(i.gt.mr)then
+      write(*,*)'   Need to increase parameter mr to ',i
+      stop
+      end if
       do 3 i=1,2*nz+4
```

**Why `mz` is 40004 here but 20002 for the fluid codes.** rams0.5 is elastic
and interleaves the field vector, so it indexes the depth arrays to `2*nz+4`
(`rams0.5.f:141`, and `2*nz` at `:673-675`, `:778-825`) where ramgeo/ramsurf
index `nz+2`. `mz=40004` therefore gives all three the same usable depth grid,
`nz ≤ 20000`. Cost is ~48 MB of static arrays (14 `(mz,mp)` complex arrays),
against ~12 MB before.

The bounds-check condition is `2*nz+4`, not the `nz+2` its siblings use —
copying theirs would have understated rams' capacity by 2×.
`uacpy.models.ram._COLLINS_ARRAY_LIMITS` carries the same per-backend rate,
and `tests/test_ram_backends.py` asserts the two agree by parsing these
guard expressions out of the sources.

Verified inert: on an input that fits the stock dimensions, builds from the
pre-patch and post-patch sources produce **byte-identical** `tl.grid` and
`pcomplex.bin`. (Note separately that a rebuild on a newer gfortran than the
one that produced a given binary shifts TL by ~0.005 dB — `mz` is the leading
dimension of the `(mz,mp)` arrays, so it also changes column stride and hence
vectorisation. Both effects are far below any physical tolerance.)

### Build system

uacpy supplies a minimal `Makefile` (`gfortran -O2 -std=legacy -w`) that
builds only the two binaries uacpy dispatches to (`rams0.5`,
`ramsurf1.5`). This replaces upstream's autotools setup (`configure.ac` /
`Makefile.am`) which targets a wider set of executables uacpy doesn't use.

### Dispatcher

The RAM dispatcher (`uacpy.models.ram`) selects whichever binary matches
the environment: elastic bottom → rams0.5; altimetry → ramsurf1.5;
default fluid + flat surface → mpiramS for in-process Fortran broadband.
Collins backends loop in Python (one subprocess per frequency).

---

## ramgeo (RAMGEO — range-dependent layered fluid PE)

Vendored at `third_party/ramgeo/` as a single source, `ramgeo1.5.f`
(Collins' RAMGEO, version 1.5g). Sourced from the **Acoustics Toolbox** `RAM/`
bundle (Porter's AT, mirroring `oalib.hlsresearch.com/Modes/AcousticsToolbox/`);
the vendored file is byte-for-byte that copy plus the two patches below.

- **Licence:** **public domain** — a U.S. Government work (Collins, NRL). No
  explicit licence accompanies the code (NRL/OALIB distribute it freely with no
  copyright or licence notice). Obtained from the Acoustics Toolbox `RAM/`
  bundle, which merely redistributes Collins' original; bundling does not
  relicense it.
- **What it is:** the split-step Padé PE [Collins, JASA 93, 1736 (1993)]
  with *"multiple sediment layers that parallel the bathymetry"* — i.e. a
  range-dependent **layered fluid** seabed. Reads `ramgeo.in`, writes
  `tl.line` (text) and `tl.grid` (unformatted), the same output family as
  `rams0.5` / `ramsurf1.5`. Built by a plain top-level `Makefile`
  (mirrors `ramsurf/`: `gfortran -O2 -std=legacy -w`, single source →
  `ramgeo` binary); `install.sh` installs it to `uacpy/bin/ramgeo/`.
  `ramgeo.in` is the upstream sample, kept for a smoke test.

Two patches give it full parity with the other Collins backends (the same
two `ramsurf1.5.f` carries):

### Enlarged array dimensions

Stock RAMGEO dimensions `parameter (mr=100,mz=8000,mp=10)` are too small
for uacpy's fine Lytaev range/depth grids (a few-hundred-Hz run can need
>8000 depth points), causing a silent array overflow and an empty
`tl.grid`. Enlarged to match the patched `ramsurf1.5.f`:

```diff
-      parameter (mr=100,mz=8000,mp=10)
+      parameter (mr=505,mz=20002,mp=10)
```

### `outpt` patch — complex-envelope dump

Stock `outpt` writes only real TL to `tl.grid`, discarding the phase a
broadband transfer function needs. Patched to also dump the complex PE
envelope `u·f3 / sqrt(r)` to a parallel `pcomplex.bin`, mirroring
`tl.grid`'s record geometry — **the identical envelope and convention as
`ramsurf1.5.f`** (carrier `exp(+i k0 r)` factored out; the Python wrapper
applies the same `'ramsurf'` correction in `psi_to_travelling_wave`). This
is what lets `RAM(backend='ramgeo')` return complex pressure for
`COHERENT_TL` and serve `BROADBAND` / `TIME_SERIES` — every mode reads
`pcomplex.bin`, so an unpatched binary fails all three.

```diff
       open(unit=3,status='unknown',file='tl.grid',form='unformatted')
+      open(unit=11,status='unknown',file='pcomplex.bin',
+     >  form='unformatted')
...
       write(3)lz
+      write(11)lz
...
       subroutine outpt(mz,mdr,ndr,ndz,iz,nzplt,lz,ir,dir,eps,r,f3,u,tlg)
-      complex ur,u(mz)
+      complex ur,u(mz),urg(mz)
...
       tlg(j)=-20.0*alog10(cabs(ur)+eps)+10.0*alog10(r+eps)
+      urg(j)=ur/sqrt(r+eps)
     1 continue
       write(3)(tlg(j),j=1,lz)
+      write(11)(urg(j),j=1,lz)
```

---

## mpiramS (RAM parabolic-equation model)

### `Makefile` -- portability

Removes the hardcoded compiler path and Intel micro-architecture target so the
build works on any machine with gfortran installed.

```diff
@@ -8,7 +8,7 @@
 
 ###########################################
 # Gnu g77/gfortran options (64 bit)
-FC = /usr/bin/gfortran-13
+FC ?= gfortran
 
 #FFLAGS = -march=native -mtune=native -fopenmp -m64 -mfpmath=sse -I $(MODDIR) -Wall -finline-functions -ffast-math -fno-strength-reduce -falign-functions=2  -O3 -fomit-frame-pointer 
 #FFLAGS = -g -pg -march=native -fopenmp -m64 -mfpmath=sse -I $(MODDIR) -Wall 
@@ -19,8 +19,8 @@
 #LDFLAGS = -fopenmp -march=native -mtune=native
 #LDFLAGS = -g -pg -fopenmp -march=native -mtune=native 
 
-FFLAGS = -Ofast -march=icelake-client -fopenmp -I $(MODDIR) -Wall -fuse-linker-plugin
-LDFLAGS = -Ofast -fopenmp -march=icelake-client -flto
+# -march=native baked-in flag commented out by uacpy: produces CPU-specific
+# binaries that break wheel/sdist consumers and macOS/ARM cross-compiles.
+# install.sh sets FORTRAN_ARCH_FLAGS via FFLAGS=/LDFLAGS= on the command
+# line; users invoking `make` directly should pass FFLAGS=/LDFLAGS=
+# explicitly if they want CPU tuning.
+# FFLAGS  = -Ofast -march=native -fopenmp -I $(MODDIR) -Wall -fuse-linker-plugin
+# LDFLAGS = -Ofast -fopenmp -march=native -flto
+FFLAGS  ?= -Ofast -fopenmp -I $(MODDIR) -Wall -fuse-linker-plugin
+LDFLAGS ?= -Ofast -fopenmp -flto
```

`?=` is used (rather than `=`) so install.sh's command-line `FFLAGS=` /
`LDFLAGS=` injection always wins; users invoking `make` directly can set
their own via environment.

### `src/kinds.f90` -- single to double precision

Every `real(kind=wp)` variable in mpiramS inherits from this parameter.  The PE
algorithm accumulates phase over thousands of range steps; at low frequencies
and long ranges single-precision arithmetic loses significance in the complex
exponential `exp(i k dr)`.  The original code already used double precision
(`wp2`) for selected critical operations, acknowledging the limitation.  The
Python reader (`io/mpirams_reader.py`) expects float64 output.

```diff
@@ -1,6 +1,6 @@
 module kinds
 
-integer,parameter :: wp = kind(1.0e0)
+integer,parameter :: wp = kind(1.0d0)
  integer,parameter :: wp2 = kind(1.0d0)
 
  end module kinds
```

### `src/matrc.f90` -- safe complex-zero initialization

Fortran `allocate` does not initialise memory.  Multiplying by zero does not
produce zero when the operand is NaN or Inf (`0 * NaN = NaN`).

```diff
@@ -28,8 +28,8 @@
   allocate(f1(nz+2),f2(nz+2))
 ! zero all the r and s arrays, otherwise they are filled with garbage
 ! f arrays are overwritten and are o.k.
-  r1=0.0_wp*r1; r2=0.0_wp*r2; r3=0.0_wp*r3
-  s1=0.0_wp*s1; s2=0.0_wp*s2; s3=0.0_wp*s3
+  r1=cmplx(0.0_wp,0.0_wp,wp); r2=cmplx(0.0_wp,0.0_wp,wp); r3=cmplx(0.0_wp,0.0_wp,wp)
+  s1=cmplx(0.0_wp,0.0_wp,wp); s2=cmplx(0.0_wp,0.0_wp,wp); s3=cmplx(0.0_wp,0.0_wp,wp)
 
 ! Defined in ram.f, since they dont have to be recalculated each step
 !  a1=k0*k0*sixth
@@ -75,7 +75,7 @@
   deallocate(f1,f2,ksq)
  
 ! The matrix decomposition.
-  allocate(rfact(no)) ; rfact=0.0_wp*rfact 
+  allocate(rfact(no)) ; rfact=cmplx(0.0_wp,0.0_wp,wp)
   do id=i1,iz
     rfact=cmplx(1.0_wp2/(r2(id,:)-r1(id,:)*r3(id-1,:)),kind=wp)
     r1(id,:)=r1(id,:)*rfact
```

### `src/solvetri.f90` -- safe complex-zero initialization

Same `0 * NaN` fix as `matrc.f90`.

```diff
@@ -16,7 +16,7 @@
   nz=size(r1,1)
   no=size(r1,2)    ! no is just np, the number of pade coefficients, i.e., 4. 
 
-  allocate(v(nz)); v=0.0_wp*v
+  allocate(v(nz)); v=cmplx(0.0_wp,0.0_wp,wp)
 
   nz=nz-2
   nz1=nz+1
```

### `src/envdata.f90` -- data module for new sediment variables

Added module-level variables to support the extended sediment model:

- `nzs` -- number of sediment depth points (was implicitly 4).
- `isedrd` -- flag: 0 = range-independent, 1 = range-dependent sediment.
- `nrp_sed` -- number of range-dependent sediment profiles.
- `rp_sed(:)` -- range points for sediment profiles (metres).
- `cs`, `rho`, `attn` changed from 1-D `(4)` to 2-D `(nzs, nrp_sed)`.

```diff
@@ -8,12 +8,21 @@
 implicit none
 
 real(kind=wp) :: sedlayer
+integer :: nzs                                             ! number of sediment depth points
 real(kind=wp),dimension(:),allocatable :: zg               ! depth grid with deltaz spacing
 real(kind=wp),dimension(:),allocatable :: rb,zb            ! bathymetry
 real(kind=wp),dimension(:),allocatable :: rp,zw            ! range and depths of sound speeds
 real(kind=wp),dimension(:,:),allocatable ::  cw            ! sound speeds, etc.
-real(kind=wp),dimension(:), allocatable :: cs,rho,attn    
-                           ! bottom properties are simple and range independent - four values.
+
+! Bottom sediment properties.
+! When isedrd==0 (range-independent): cs(nzs,1), rho(nzs,1), attn(nzs,1) — single profile.
+! When isedrd==1 (range-dependent):   cs(nzs,nrp_sed), rho(nzs,nrp_sed), attn(nzs,nrp_sed)
+!   with rp_sed(nrp_sed) giving the range points (in metres).
+integer :: isedrd                                          ! 0=range-indep, 1=range-dep sediment
+integer :: nrp_sed                                         ! number of sediment range profiles
+real(kind=wp),dimension(:),   allocatable :: rp_sed        ! sediment range points (m)
+real(kind=wp),dimension(:,:), allocatable :: cs,rho,attn   ! bottom properties (nzs, nrp_sed)
 
  end module envdata
```

### `src/ram.f90` -- bug fixes and range-dependent sediment

#### Sub-bottom refresh on every bathymetry change (upslope staleness)

Upstream deferred rebuilding the sub-bottom arrays until the water depth had
moved more than 20 m, with an explicitly uncertain comment:

```fortran
if (abs(izll-iz)*deltaz > 20.0_wp) then
! The depth has changed by more than 20 m; update the bottom profiles
! This is mainly for attenuation and density.
! Don't need to call this for EVERY depth change! (I don't think...)
```

But `profl` fills `cw`/`cb`/`rhob`/`attn` at **absolute** depth indices and
`matrc.f90:50-55` reads them at the current `iz`
(`forall (id=(iz+1):(nz+2)) f1(id)=rhob(id)/alpb(id) … ksq(id)=ksqb(id)`), so
between rebuilds up to 20 m of seabed immediately below a **rising** seafloor
still carries **water** sound speed while being treated as bottom. Downslope is
unaffected: there the stale band lands *above* `iz`, where `matrc` uses the
water arrays anyway — which is why the defect is one-sided.

**Fix:** rebuild whenever `iz` changes, matching what the SSP branch twenty
lines below already does (`if (ir/=irl) … iflag=iflag+2`, no threshold).

```diff
 if (iz/=izl)  then
      upd=1
-     if (abs(izll-iz)*deltaz > 20.0_wp) then
-        iflag=iflag+1
-        izll=iz
-     end if
+     iflag=iflag+1
+     izll=iz
  end if
```

Measured on a 200 → 100 m wedge over 6 km, 100 Hz, `dr=10 dz=0.5 zmax=500`,
against ramgeo on the identical grid:

| | median | p90 | max |
|---|---|---|---|
| upslope, before | 4.38 dB | 13.58 | 29.59 |
| upslope, after | **0.17 dB** | 0.39 | 1.11 |
| downslope (control), after | 0.19 dB | 0.49 | 1.10 |

Upslope and downslope now agree to 0.02 dB median — the asymmetry that
identified the bug is gone. Cost is one extra `profl` call per range step where
the seafloor index moves.

#### NaN-safe initialisation (same pattern as matrc/solvetri)

```diff
@@ -96,8 +104,8 @@
   ! Self starter
   if (size(zsrc)==1) then
     allocate(uu(nz+2))
-    ! zero uu
-    uu=0.0_wp*uu
+    ! zero uu (use assignment, not multiply — 0*NaN=NaN on uninitialized memory)
+    uu=cmplx(0.0_wp,0.0_wp,wp)
     ! Conditions for the delta function.
     zsc=1.0_wp+zsrc(1)/deltaz
     izs=floor(zsc)
@@ -106,8 +114,8 @@
     uu(izs+1)=      delzs* sqrt(2.0_wp*pi/k0)/(deltaz*alpw(izs))
 
     ! Divide the delta function by (1-X)**2 to get a smooth rhs.
-    allocate(pdu(np),pdl(np)); pdu=0.0_wp*pdu; pdl=0.0_wp*pdl
-    pdu(1)=cmplx(0.0_wp,wp); pdl(1)=cmplx(-1.0_wp,wp)
+    allocate(pdu(np),pdl(np)); pdu=cmplx(0.0_wp,0.0_wp,wp); pdl=cmplx(0.0_wp,0.0_wp,wp)
+    pdu(1)=cmplx(0.0_wp,0.0_wp,wp); pdl(1)=cmplx(-1.0_wp,0.0_wp,wp)
     call matrc
     call solvetri
     call solvetri
@@ -148,9 +156,9 @@
   if (size(zsrc)==size(zg)) uu=zsrc/f3
 
   if (.not.allocated(psi)) allocate(psi(nz+2,nr))
-    psi=0.0_wp*psi
+    psi=cmplx(0.0_wp,0.0_wp,wp)
   if (.not.allocated(rout)) allocate(rout(nr))
-    rout=0.0_wp*rout
+    rout=0.0_wp
```

#### Variable declarations for range-dependent sediment

```diff
@@ -35,8 +35,8 @@
 real(kind=wp),dimension(:),intent(in) :: zsrc,rg 
 
 integer :: iflag
-integer :: ii,n,nz,nr,nb,ir,irl,izl,izll,irr,upd,izs
-integer :: ir0(1)
+integer :: ii,n,nz,nr,nb,ir,irl,izl,izll,irr,upd,izs,ir_sed,irl_sed
+integer :: ir0(1), ir0_sed(1)
 real(kind=wp) :: omega, dr, rend, rnow, rint, rsc
 real(kind=wp) :: delzs,zbc,zsc
 real(kind=wp) :: rint1(1),zbc1(1), maxrb1
```

#### `rnow` initialisation fix

The original set `rnow = rg(1)` (first output range), which skipped PE
self-starter propagation from range zero.  The modified version keeps
`rnow = 0` so the field is correctly marched from the source.

Side effect, recorded because nothing else states it: `rnow` is read four
lines later by `rsc = abs(rend - rnow) - rs` (`ram.f90:68`), so on a
multi-range output grid `rsc` becomes `rg(nr) - rs` where upstream it was
`rg(nr) - rg(1) - rs`. That makes the `ram.f90:251` stability-constraint
branch fire slightly earlier. It fires within the first output segment
either way — the branch compares the distance left to the *current* output
range rather than the absolute range marched, which is what makes
`rs_stability` largely inert on a multi-range grid (upstream behaviour;
`RAM._warn_rs_stability_inert_on_a_multi_range_grid` warns about it).

```diff
@@ -59,7 +59,6 @@
   nr=size(rg)
   rend=rg(nr)
   rnow=0.0_wp
-  if (nr>1) rnow=rg(1)
 
   dr=deltar    ! dr is deltar of peramx; 
             ! it may need to adjust to get to the range rend precisely
```

#### Range-dependent sediment index tracking in march loop

```diff
@@ -76,6 +75,15 @@
   ir=ir0(1)
   irl=ir
 
+  ! Initialize sediment range index
+  if (isedrd==1 .and. nrp_sed>1) then
+     ir0_sed=minloc(abs(rp_sed-(rnow+dr/2.0_wp)))
+     ir_sed=ir0_sed(1)
+  else
+     ir_sed=1
+  end if
+  irl_sed=ir_sed
+
   nb=size(rb)
   allocate(rb1(nb+1),zb1(nb+1))
```

```diff
@@ -194,10 +202,21 @@
       ! Varying profiles - using profile closest to present range.
       ir0=minloc(abs(rp-rint))
       irl=ir; ir=ir0(1)
-      if (ir/=irl) then 
+      if (ir/=irl) then
       ! sound speed has changed; update profiles and call matrc
            iflag=iflag+2
-           upd=1    
+           upd=1
+      end if
+
+      ! Varying sediment profiles (range-dependent bottom)
+      if (isedrd==1 .and. nrp_sed>1) then
+         ir0_sed=minloc(abs(rp_sed-rint))
+         irl_sed=ir_sed; ir_sed=ir0_sed(1)
+         if (ir_sed/=irl_sed) then
+            ! sediment profile has changed; need to update bottom
+            if (mod(iflag,2)==0) iflag=iflag+1  ! ensure iflag includes 1 (bottom update)
+            upd=1
+         end if
       end if
```

#### `profl` subroutine -- configurable nzs-point sediment with range selection

The original used a hardcoded 4-point sediment model (surface, seafloor,
sedlayer depth, domain bottom) with range-independent properties.  The modified
version accepts `nzs` depth points and selects the nearest sediment profile by
range when `isedrd == 1`.

```diff
@@ -246,12 +265,12 @@
 integer, intent(in) :: iflag  ! iflag=3 update all; iflag=1 update bathymetry; iflag=2 update sound speed
 real(kind=wp), intent(in) :: r, omega
 
-integer :: ir,ii
-integer :: ir0(1)
-real(kind=wp) :: depth
-real(kind=wp), dimension(:) :: rwork(1),zwork(4)
+integer :: ir,ii,ir_sed,iz
+integer :: ir0(1), ir0_sed(1)
+real(kind=wp) :: depth, dz_sed
+real(kind=wp), dimension(:) :: rwork(1)
 real(kind=wp), dimension(:,:) :: work(1,1)
-real(kind=wp), dimension(:), allocatable :: csg,attng
+real(kind=wp), dimension(:), allocatable :: csg,attng,zwork,cs_local,rho_local,attn_local
 
   n=size(zg)
   if (.not.allocated(rhob)) allocate(cwg(n),rhob(n),ksqw(n),alpw(n),alpb(n),ksqb(n))
@@ -265,25 +284,44 @@
 
 if (iflag==1.or.iflag==3) then    ! update sediment sound speed, density, attenuation
     allocate(csg(n),attng(n))
+    allocate(zwork(nzs),cs_local(nzs),rho_local(nzs),attn_local(nzs))
+
+!   Select sediment profile for this range
+    if (isedrd==1 .and. nrp_sed>1) then
+       ir0_sed=minloc(abs(rp_sed-r)); ir_sed=ir0_sed(1)
+    else
+       ir_sed=1
+    end if
+    cs_local   = cs(:,ir_sed)
+    rho_local  = rho(:,ir_sed)
+    attn_local = attn(:,ir_sed)
 
 !   First find the depth at this range
     rwork(1)=r
     work(:,1)=interp1(rb,zb,rwork,zb(1))
     depth=work(1,1)
-    ! The four values of depth that go with cs, rho, and attn
-    zwork(1)=0.0_wp; zwork(2)=depth
-    zwork(3)=depth+sedlayer; zwork(4)=max(zg(n),zwork(3)+1.0E-6)
-
-! Set up sediment sound speed to increase linearly below the sea floor, with
-! a sedlayer-m thick sediment layer. 
-    csg=gorp(zwork,cs,zg)
+
+!   Construct nzs depth points: surface, seafloor, nzs-3 interior sediment
+!   points evenly spaced, then domain bottom.
+    zwork(1)=0.0_wp
+    zwork(2)=depth
+    if (nzs > 3) then
+       dz_sed = sedlayer / real(nzs-3, wp)
+       do iz=3,nzs-1
+          zwork(iz) = depth + real(iz-2, wp) * dz_sed
+       end do
+    end if
+    zwork(nzs) = max(zg(n), zwork(nzs-1)+1.0E-6_wp)
+
+! Set up sediment sound speed profile (linearly interpolated over nzs points).
+    csg=gorp(zwork,cs_local,zg)
     csg=cwg+csg
 
 ! Set up the sediment density and attenuation profiles.
-! Attenuation and density follow the bottom.
-    rwork(1)=rho(1) 
-    rhob=gorp(zwork,rwork,zg)   ! send it only one value, so gorp does the easy thing; rhob a constant.
-    attng=gorp(zwork,attn,zg)
+    rhob=gorp(zwork,rho_local,zg)
+    attng=gorp(zwork,attn_local,zg)
+
+    deallocate(zwork,cs_local,rho_local,attn_local)
  end if
```

#### `gorp` function -- NaN-safe broadcast

```diff
@@ -323,7 +361,7 @@
   
   if (size(y)==1) then
     !forall(ii=1:n) gorp(ii)=y(1)
-    gorp=y(1)+0.0_wp*gorp
+    gorp=y(1)
     return
   end if
```

#### March step never restored after a partial step

`dr` is set once, outside the march loop (`dr=deltar` at `:63`, with the sign
flip at `:66`); the only other assignment is `dr=rend-rnow` inside the
output-range loop. That branch shrinks the step to the remainder needed to land
exactly on an output range — and nothing restores it. Every later output range
therefore marched at that leftover step.

Because uacpy writes **every receiver range** into `ranges.dat`, the cost of a
march grew with the *number of output ranges* rather than with range. Measured
on a 100 m Pekeris waveguide at 250 Hz over the same 10 km, `OMP_NUM_THREADS=1`:

| output ranges | before | after |
|---|---|---|
| 1 | 0.576 s | 0.576 s |
| 10 | 1.870 s (3.2x) | 0.669 s (1.2x) |
| 50 | 8.700 s (15.1x) | 1.067 s (1.9x) |

Left alone, this is a cost defect and not an accuracy one — the shrunken step
is *smaller*, so TL moved by only ~2e-4 dB. But it also meant the marched `dr`
was not the `dr` uacpy resolves and reports in the `Result` metadata.

> **Correction.** Two claims first recorded here were wrong, and both are
> measured below in *Remainder tested against the live step*.
>
> 1. "A cost defect, not an accuracy one" describes the *upstream* code. It
>    does not describe this patch: restoring the step is what made an
>    **overshoot** possible, because the test above it compared the remainder
>    against the live `dr` rather than against `deltar`. Upstream, `dr` only
>    ever shrinks, so the march can never pass its target at any grid. This
>    patch is what put the accuracy defect there; the entry below is what
>    closes it, and the two belong together.
> 2. "After the fix TL at a given range is bit-identical however many other
>    output ranges are requested" is false and stays false, though the
>    magnitude is now negligible. mpiramS lands on each requested range, so
>    the output grid decides how each leg is decomposed into Padé steps, and a
>    rational approximant of `exp` does not compose exactly. At the 61 ranges
>    common to a 25 m and a 50 m output grid over the same water — same
>    source, same `dr=40` — the two runs disagreed by **3.585 dB median /
>    20.755 dB max** with only this patch applied, and by **5.96e-6 dB median
>    / 3.83e-5 dB max** with both. Pinned by
>    `test_ram_backends.TestTlDoesNotDependOnTheOutputGrid`.

**Fix:** restore the full step when it has been shrunk, as the `else` of the
same test, reusing that branch's `upd=1` so `matrc` rebuilds the matrices for
the restored `dre`.

```diff
       if (abs(rend-rnow)<abs(dr)) then
         dr=rend-rnow
         dre=abs(dr)
         ip=1
         call epade
         upd=1
+      else if (abs(abs(dr)-abs(deltar))>tiny(deltar)) then
+        dr=sign(deltar,rend-rnow)
+        dre=abs(dr)
+        ip=1
+        call epade
+        upd=1
       end if
```

#### Remainder tested against the live step, so the march overshot its target

The shrink test above compares the remainder against `dr` — the loop's live,
possibly already-shrunk step — rather than against `deltar`. On its own that is
harmless, because upstream `dr` only ever shrinks and a remainder can never
exceed it. Paired with the step restore added above it is not: after a shrink,
an output range whose remainder is longer than that leftover step but shorter
than `deltar` fails the test, falls into the restore branch, and is marched a
full `deltar` **past** its own target. The next iteration shrinks to the
*negative* remainder and walks "backward" onto the range it missed.

Nothing conjugates the field for a backward step. `dre=abs(dr)` at `:189`,
`:201` and `:255` strips the sign, and `epade` builds its coefficients from
`dre`, so the backward step applies a **forward** propagator — the field is
marched further away from the source, not back toward it, and is then written
out under the label of a range it has already passed. That `epade` is sign-aware
is not an inference: the self-starter's own deliberate backward step at `:134`
writes `dre=-abs(dr)`. `grep -rn conjg src/` is empty.

The misplacement accumulates over the march, so "roughly twice as far" — the
step doubles — understates it badly. On uacpy's own 200 m / 25 Hz
pressure-release fixture (121 output ranges at 25 m, auto `deltar` = 305.8 m),
replaying this branch exactly: half the output ranges take a backward step, and
the field written at the last range, labelled **3300 m**, has been propagated
**37 020 m — 11.2x its own label**.

Measured with a step counter compiled into `ram.f90` (`OMP_NUM_THREADS=1`):

| fixture | backward steps | total steps | median &#124;ΔTL&#124; vs the closed-form modal sum |
|---|---|---|---|
| auto `deltar`=305.8, 121 ranges @ 25 m | 60 | 181 | 3.548 dB |
| …after the fix | **0** | **121** | **2.127 dB** |
| `dr`=40 pinned, 121 ranges @ 25 m | 120 | 248 | 3.868 dB |
| …after the fix | **0** | **128** | **1.691 dB** |

The correct march is also the cheaper one: an overshoot costs a forward *and* a
backward step per output range.

**Fix:** test the remainder against `deltar`. A remainder shorter than a full
step then shrinks onto its target, and a longer one is marched at the restored
`deltar`, so `rnow` can never pass `rend` — at any `deltar`, on any output grid.
No cap on `deltar` and no knowledge of the receiver grid is needed for that,
which is what let uacpy drop its Python-side `dr` cap (`_mpirams_dr_output_cap`,
whose own floor-less `min` over the output gaps sized a 5 000 000-step march
from a legal 2 mm receiver pair).

```diff
-      if (abs(rend-rnow)<abs(dr)) then
+      if (abs(rend-rnow)<abs(deltar)) then
         dr=rend-rnow
         dre=abs(dr)
         ip=1
         call epade
         upd=1
       else if (abs(abs(dr)-abs(deltar))>tiny(deltar)) then
```

The restore branch stays: without it, `dr` never returns to `deltar` and the
upstream cost defect above comes back.

### `src/peramx.f90` -- I/O rewrite (largest change)

#### Sequential unformatted output for `psif.dat`

The original output used direct-access I/O with a single fixed record size
sized to the LARGEST record (typically the `rout` array of `nr` reals). Every
depth record (only `1 + 2*nf` reals of useful data) was zero-padded out to
`max(8, nf, nr, 1+2*nf) * wp` bytes. For typical use (e.g. `nr=500`,
`nzo=2000-5000`) the file ballooned to **multi-GB of mostly zeros**, hitting
tmpfs limits and OOM-killing the binary.

Switched to ``access='sequential', form='unformatted'``. Each record is now
self-sized; depth records take `(1 + 2*nf) * wp ≈ 56 bytes` instead of
`max(...) * wp ≈ 4000 bytes`. The output file shrinks ~500× (e.g. 6 GB → 12 MB
for typical example_26 settings). ``recl.dat`` is no longer needed and is
not written. The Python reader uses ``scipy.io.FortranFile`` to parse
gfortran's standard sequential-unformatted record markers.

```diff
-block
-  integer :: rl1
-  inquire(iolength=rl1) fc        ! iolength of one real(wp)
-  length = max(8, nf, nr, 1+2*nf) * rl1
-end block
-
-open(nunit, form='formatted',file='recl.dat')
-write(nunit,*) length
-close(nunit)
-
-open(nunit, access='direct',recl=length,file='psif.dat')
-write(nunit,rec=1) Nsam,real(nf,wp),real(nzo,wp),real(nr,wp),c0,cmin,fs,Q
-write(nunit,rec=2) frq
-write(nunit,rec=3) rout
-do ir=1,nr
-  do ii=1,nzo
-    write(nunit,rec=3+(ir-1)*nzo+ii) zg1(ii), &
-        ((real(psif(ii,jj,ir))),(aimag(psif(ii,jj,ir))),jj=1,nf)
-  end do
-end do
+open(nunit, access='sequential', form='unformatted', file='psif.dat')
+write(nunit) Nsam,real(nf,wp),real(nzo,wp),real(nr,wp),c0,cmin,fs,Q
+write(nunit) frq
+write(nunit) rout
+do ir=1,nr
+  do ii=1,nzo
+    write(nunit) zg1(ii), &
+        ((real(psif(ii,jj,ir))),(aimag(psif(ii,jj,ir))),jj=1,nf)
+  end do
+end do
```

#### User-pinned PE reference speed `c0`

Reads a positive `c0_user` (m/s) from a new line in `in.pe` between `dzm` and
the SSP filename. The value is the PE expansion speed
(``exp(ik0*x)`` carrier factored out of the Helmholtz solution), and lets
the caller pick the Lytaev (2023) Eq. 15 optimum that centres the spectrum
``[ξ_min, ξ_max]`` around 0 and minimises the Padé approximation error.
The binary stops with an error if the value is non-positive — the caller
is required to supply a sensible reference speed.

```diff
+real(kind=wp) :: c0_user      ! PE reference speed; required positive
 ...
 read (nunit,*) dzm                    ! output depth decimation (integer)
+read (nunit,*) c0_user                ! PE reference speed (m/s); must be positive
 read (nunit,'(a)') name1              ! sound speed filename
 ...
-! mean sound speed
-n=size(cw)
-c0=sum(cw)/n
+if (c0_user <= 0.0_wp) then
+   print *, 'ERROR: c0_user must be positive in in.pe (got ', c0_user, ')'
+   stop 1
+end if
+c0=c0_user
 ic0=1.0_wp/c0
 cmin=minval(cw)
```

#### Free-format input parsing and longer filename buffers

The original fixed-format reads (`read(2,'(f4.0)')`) required values to fit in
exact column widths.  Free-format `read(nunit,*)` is standard practice for
program-controlled input files.  Filename buffers increased from 20 to 256
characters to accommodate full paths.

```diff
@@ -22,19 +22,19 @@
 
 real(kind=wp),dimension(:),allocatable :: zg1
 complex(kind=wp),dimension(:),allocatable :: psi1
-complex(kind=wp),dimension(:,:),allocatable :: psif
+complex(kind=wp),dimension(:,:,:),allocatable :: psif  ! (nzo, nf, nr)
 
 ! input parameters - c.f., file "in.pe"
 integer :: dzm, iflat, ihorz, ibot
 real(kind=wp) :: fc,Q,T,dum
 real(kind=wp),dimension(:),allocatable :: zsrc,rmax
-character(len=20) :: name1,name2     ! sound speed and bathymetry filenames
+character(len=256) :: name1,name2,name3,name4  ! ssp, bathymetry, ranges, sediment filenames
 real(kind=wp),dimension(:),allocatable :: eps
 real(kind=wp),dimension(:,:),allocatable :: cq
 
 integer :: nss
 
-integer :: nb,nzp,nrp,nrp0,n,nf1,nf
+integer :: nb,nzp,nrp,nrp0,nf1,nf,nr
 real(kind=wp) :: bw, fs, Nsam, df, tmp
 real(kind=wp),dimension(:),allocatable :: frq
 
@@ -44,10 +44,10 @@
 real(kind=wp) :: rate
 integer :: t1,t2,cr,cm
 
-integer :: ii,jj,iff,length
+integer :: ii,jj,iff,ir
 
 integer, parameter :: nunit=2
-complex(kind=wp), parameter :: j=cmplx(0.0_wp,1.0_wp)
+complex(kind=wp), parameter :: j=cmplx(0.0_wp,1.0_wp,wp)
 complex(kind=wp) :: scl
```

```diff
-allocate(zsrc(1),rmax(1))
+allocate(zsrc(1))
 
 open(nunit,file='in.pe',status='old')
-read (2,'(f4.0)')  fc                ! skip the first line - read a dummy
-read (2,'(f4.0,1x,f2.0)') fc,Q       ! center frequency and Q
-read (2,'(f4.1)') T              ! time window width
-read (2,'(f6.1)') zsrc(1)        ! source depth
-read (2,'(f12.3)') rmax(1)       ! receiver range
-read (2,'(f5.2)') deltaz         ! depth accuracy parameter
-read (2,'(f6.2)') deltar         ! range accuracy parameter
-read (2,'(i1,1x,i1)') np,nss     ! np -# pade coefficients
-                                 ! ns -# stability terms
-read (2,'(f7.1)') rs             ! stability range
-read (2,'(i2)') dzm              ! dzm - depth decimation
-read (2,'(a20)') name1           ! sound speed filename; "munk" will just use canonical
-name1=trim(name1) ! remove trailing blanks
-read (2,'(i1)') iflat            ! 0=no flat earth transform, 1=yes
-read (2,'(i1)') ihorz            ! 0=no horizontal linear interpolation, 1=yes
-read (2,'(i1)') ibot             ! 0=no bottom, 1=bottom and read a file
-read (2,'(a20)') name2           ! bathymetry filename; ignored if ibot=0
-name2=trim(name2) ! remove trailing blanks
+read (nunit,*) dum                    ! skip the first line (comment/title)
+read (nunit,*) fc, Q                  ! center frequency (Hz) and Q value
+read (nunit,*) T                      ! time window width (s)
+read (nunit,*) zsrc(1)                ! source depth (m)
+read (nunit,*) deltaz                 ! depth accuracy parameter (m)
+read (nunit,*) deltar                 ! range accuracy parameter (m)
+read (nunit,*) np, nss                ! np-# pade coefficients, ns-# stability terms
+read (nunit,*) rs                     ! stability range (m)
+read (nunit,*) dzm                    ! output depth decimation (integer)
+read (nunit,'(a)') name1              ! sound speed filename
+name1=trim(adjustl(name1))
+read (nunit,*) iflat                  ! 0=no flat earth transform, 1=yes
+read (nunit,*) ihorz                  ! 0=no horizontal linear interpolation, 1=yes
+read (nunit,*) ibot                   ! 0=no bottom, 1=bottom and read a file
+read (nunit,'(a)') name2              ! bathymetry filename
+name2=trim(adjustl(name2))
+read (nunit,'(a)') name3              ! output ranges filename
+name3=trim(adjustl(name3))
```

#### Configurable sediment properties (read from `in.pe` instead of hardcoded)

Note: the original `rmax` line (`read (2,'(f12.3)') rmax(1)`) is removed from
the input parsing block above -- it is replaced by the external ranges file
(see "Multiple output ranges" below).

`sedlayer`, `nzs`, `cs`, `rho`, `attn` and the range-dependent sediment flag
`isedrd` are now read from `in.pe`.  Supports an optional external sediment
profile file when `isedrd == 1`.

```diff
+! Read bottom properties (sedlayer, nzs, cs, rho, attn)
+read (nunit,*) sedlayer
+read (nunit,*) nzs
+read (nunit,*) isedrd
+
+if (isedrd==1) then
+   ! Range-dependent sediment: read filename and load profiles
+   read (nunit,'(a)') name4
+   name4=trim(adjustl(name4))
+
+   ! Temporary defaults (overridden by file)
+   nrp_sed=1
+   allocate(cs(nzs,1),rho(nzs,1),attn(nzs,1))
+   cs(:,1)  = 0.0_wp
+   rho(:,1) = 1.2_wp
+   attn(:,1)= 0.5_wp
+
+   close(nunit)
+
+   ! Read sediment profile file (same format as SSP: "-1 range_km" headers)
+   print *,'Reading sediment file: ', trim(name4)
+   open(nunit,file=name4,status='old')
+
+   ! First pass: count profiles
+   deallocate(cs,rho,attn)
+   nrp_sed=0
+   do
+      read(nunit,*,end=4) dum
+      if (dum<0) nrp_sed=nrp_sed+1
+   end do
+4  print *,'Found ',nrp_sed,' sediment profiles.'
+   rewind(nunit)
+
+   allocate(rp_sed(nrp_sed), cs(nzs,nrp_sed), rho(nzs,nrp_sed), attn(nzs,nrp_sed))
+
+   ! Second pass: read profiles (nzs values per line)
+   do ii=1,nrp_sed
+      read(nunit,*) dum, rp_sed(ii)
+      rp_sed(ii) = rp_sed(ii)*1000.0_wp   ! convert km to m
+      read(nunit,*) (cs(jj,ii), jj=1,nzs)
+      read(nunit,*) (rho(jj,ii), jj=1,nzs)
+      read(nunit,*) (attn(jj,ii), jj=1,nzs)
+   end do
+   close(nunit)
+
+else
+   ! Range-independent sediment: read nzs-element arrays from in.pe
+   nrp_sed=1
+   allocate(cs(nzs,1),rho(nzs,1),attn(nzs,1))
+   read (nunit,*) (cs(jj,1), jj=1,nzs)
+   read (nunit,*) (rho(jj,1), jj=1,nzs)
+   read (nunit,*) (attn(jj,1), jj=1,nzs)
+   close(nunit)
+end if
```

Replaces the original hardcoded bottom properties block:

```diff
-! Sediment layer thickness - meters
-sedlayer=300.0_wp
-
-! Sediment sound speed - this will be speed relative to the water sound speed.
-! Four values are given:  at the surface, at the bottom, sedlayer-m below the bottom,
-! and at the center of the earth...
-allocate(cs(4))
-cs(1)=0.0_wp   
-cs(2)=0.0_wp
-cs(3)=200.0_wp 
-cs(4)=200.0_wp
-
-allocate(rho(4))
-rho(1)=1.2_wp 
-rho(2)=1.2_wp
-rho(3)=1.2_wp 
-rho(4)=1.2_wp
-
-allocate(attn(4))
-attn(1)=0.5_wp  
-attn(2)=0.5_wp
-attn(3)=5.0_wp  
-attn(4)=5.0_wp
+! Bottom properties (sedlayer, nzs, cs, rho, attn) were read from in.pe above.
+! cs is sediment speed relative to the water sound speed (nzs values:
+!   at surface, at seafloor, evenly spaced through sediment, at domain bottom).
+! rho is sediment density (g/cm^3), nzs values.
+! attn is sediment attenuation (dB/wavelength), nzs values.
```

#### Multiple output ranges from external file

The original computed the field at a single range.  The modified version reads
an arbitrary list of output ranges from an external file and stores the field
at each one.  `psif` is now 3-D `(nzo, nf, nr)`.

```diff
+! Read output ranges from file
+print *,'Reading output ranges file: ', trim(name3)
+open(nunit,file=name3,status='old')
+nr=0
+do
+   read(nunit,*,end=6) dum
+   nr=nr+1
+end do
+6 print *,'Found ',nr,' output ranges.'
+rewind(nunit)
+allocate(rmax(nr))
+do ii=1,nr
+   read(nunit,*) rmax(ii)
+end do
+close(nunit)
```

#### Updated diagnostic print statements

Format widths adjusted for new parameter ranges, multi-range output info added,
sediment properties echoed.  Debug print for `c0`/`cmin` added.

```diff
@@ -67 (continued: print block)
 print '(a)','INPUT PARAMETERS:'
 print '(a,f10.2)','Center frequency (Hz): ', fc
-print '(a,f2.0)','Q: ', Q
+print '(a,f4.1)','Q: ', Q
 print '(a,f5.2)','Bandwidth (f0/Q - Hz): ', fc/Q
-print '(a,f4.1)','Time window width (s): ', T
-print '(a,f6.1)','Source depth (m): ', zsrc(1)
-print '(a,f12.3)','Range (m): ', rmax(1)
+print '(a,f6.1)','Time window width (s): ', T
+print '(a,f8.1)','Source depth (m): ', zsrc(1)
+print '(a,i6)','Number of output ranges: ', nr
+print '(a,f12.1)','First range (m): ', rmax(1)
+print '(a,f12.1)','Last range (m): ', rmax(nr)
 ...
-print '(a,i2)','Output depth decimation: ', dzm
-print '(a,a)','Sound speed filename: ', name1
+print '(a,i4)','Output depth decimation: ', dzm
+print '(a,a)','Sound speed filename: ', trim(name1)
 ...
-print '(a,a)','Ocean bottom filename: ', name2
+print '(a,a)','Ocean bottom filename: ', trim(name2)
+print '(a,a)','Ranges filename: ', trim(name3)
+print '(a,f8.1)','Sediment layer (m): ', sedlayer
+print '(a,i4)','Sediment depth points (nzs): ', nzs
+if (isedrd==1) then
+   print '(a,i4,a)','Sediment: range-dependent (',nrp_sed,' profiles)'
+else
+   print '(a,*(f8.2))','Sediment speed (cs): ', cs(:,1)
+   print '(a,*(f8.3))','Sediment density (rho): ', rho(:,1)
+   print '(a,*(f8.3))','Sediment attenuation: ', attn(:,1)
+end if
```

```diff
@@ -196,6 +277,7 @@
 c0=sum(cw)/n
 ic0=1.0_wp/c0
 cmin=minval(cw)     ! minimum sound speed for calculating tdelay
+print '(a,f10.2,a,f10.2)', 'c0=',c0,' cmin=',cmin
```

#### Horizontal interpolation range fix

The original used `rmax(1)` (single range) for the SSP horizontal
interpolation grid.  Now uses `maxval(rmax)` to span all output ranges.

```diff
@@ -159,8 +240,8 @@
    
    if (ihorz==1) then
    ! Horizontal
-     nrp=nint(rmax(1)/10000.0_wp)
-     call linspace(rp, rp0(1),rmax(1),nrp)
+     nrp=nint(maxval(rmax)/10000.0_wp)
+     call linspace(rp, rp0(1),maxval(rmax),nrp)
      allocate(cw(nzp,nrp))
      do jj=1,nzp
         cw(jj,:)=interp1(rp0,cq(jj,:),rp,cq(jj,1))
```

#### OpenMP race condition fix and multi-range parallel loop

`zg` (the depth grid) is pre-allocated before the `!$OMP PARALLEL` region.
Without this, multiple threads entering `ram()` could simultaneously find `zg`
unallocated and race on allocation.  The inner loop now iterates over output
ranges.

```diff
-allocate(psif(nzo,nf))
+allocate(psif(nzo,nf,nr))
+
+! Pre-allocate zg before the parallel region to avoid an OpenMP race
+! condition: without this, multiple threads entering ram() could
+! simultaneously see zg as unallocated and both try to allocate it.
+call linspace(zg, 0.0_wp, zmax, icount)
 
  call system_clock(count_rate=cr)
  call system_clock(count_max=cm)
  rate=real(cr)
 
-print *,nf,' total frequencies'
+print *,nf,' total frequencies, ',nr,' output ranges'
```

```diff
-!$OMP PARALLEL PRIVATE (psi1,omega,scl,t1,t2,cr,rate) 
+!$OMP PARALLEL PRIVATE (psi1,omega,scl,t1,t2,cr,rate,ir)
 allocate(psi1(nzo))
 !$OMP DO SCHEDULE(STATIC,1)
   do iff=1,nf
@@ -345,16 +397,13 @@
 
     call ram(zsrc,rmax)
 
-! The miracle of fortran95!
-    psi1=psi(1:icount:dzm,1)
-
-    omega=2.0_wp*pi*frqq 
-    ! 3-D
-    scl=exp(j*(omega/c0*rout(1) + pi/4.0_wp))/4.0_wp/pi
-    ! 2-D
-    ! k0=omega/c0
-    !scl=j*exp(j*omega/c0*rout)/sqrt(8.0_wp*pi*k0)
-    psif(:,iff)=scl*psi1
+    omega=2.0_wp*pi*frqq
+    do ir=1,nr
+      psi1=psi(1:icount:dzm,ir)
+      ! 3-D scaling
+      scl=exp(j*(omega/c0*rout(ir) + pi/4.0_wp))/4.0_wp/pi
+      psif(:,iff,ir)=scl*psi1
+    end do
```

#### Flat-earth inverse-transform guard

The original applied the inverse flat-earth depth correction unconditionally at
output time, even when `iflat=0` (no forward transform was applied).  The
modified version wraps it in `if (iflat==1)`.

```diff
-! Remove the flat-earth transform (or most of it, anyways)
-allocate(eps(nzo))
-eps=zg1*invRe
-zg1=zg1/(1.0_wp+(1.0_wp/2.0_wp)*eps+(1.0_wp/3.0_wp)*eps*eps)
-deallocate(eps)
+!  Remove the flat-earth transform (or most of it, anyways)
+  if (iflat==1) then
+    allocate(eps(nzo))
+    eps=zg1*invRe
+    zg1=zg1/(1.0_wp+(1.0_wp/2.0_wp)*eps+(1.0_wp/3.0_wp)*eps*eps)
+    deallocate(eps)
+  end if
```
