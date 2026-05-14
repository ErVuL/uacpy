# UACPY Scientific Audit (Round 2)

A second audit pass focused on **scientific correctness** — the
underlying physics formulas, the units thread through every solver, and
the conventions where UACPY meets the upstream Fortran or the textbook
references. The earlier `AUDIT.md` (first audit) caught most of the
code-level / I/O / packaging problems and they have since been fixed
(BUILD-1/2/3, CORE-1, VIS-1/2/3/4, IO-1/2/3/4/5, TEST-1/2, the C and D
clean-up batches, the metadata-registry + naming standardisation). This
round was run as a five-agent team against the propagation physics:

1. **Bellhop / arrivals → H(f) → IFFT** — Porter user guide §9, Jensen
   chapter 3, the AT MATLAB `delayandsum.m` and Fortran `influence.f90`.
2. **Kraken / KrakenC / KrakenField** — Porter KRAKEN 2001 manual,
   Jensen chapter 5, the AT Fortran `kraken.f90` / `field.f90` /
   `EvaluateMod.f90` / `Matlab/ReadWrite/read_modes_bin.m`.
3. **Scooter / SPARC** — Jensen chapter 4, the AT Fortran `scooter.f90`
   / `sparc.f90` / `TransformG.f90`, the canonical MATLAB `fieldsco.m`.
4. **RAM / Padé split-step** — RAM manual, Collins 1989/1991/1993,
   Lytaev 2022 (`docs/RAM*.pdf`, `collins*.pdf`, `lytaev2022.pdf`),
   `MODIFICATIONS.md` for the vendored-source diffs.
5. **OASES family** — the OASES LaTeX manual under
   `third_party/oases/doc/*.tex`, Jensen chapter 4 (Schmidt is a
   co-author), the Fortran writer in `oases/src/oasiun*.f`.
6. **DSP / noise / absorption / acoustics** — Wenz 1962, Thorp 1967,
   Francois & Garrison 1982, Mackenzie 1981, Minnaert (bubble),
   arlpy upstream, AT `AttenMod.f90`.

Findings below are organised by severity. Each cites the canonical
reference (paper, manual, or upstream source line) used to verify the
divergence.

---

## TL;DR

Five **critical** findings — most would silently produce
plausible-looking but wrong numbers under operating conditions UACPY
itself does not test today:

1. **Bellhop volume attenuation (`models/bellhop.py:148`)**: the
   `delayandsum` path uses `exp(-Im(τ))` instead of the AT-canonical
   `exp(ω·Im(τ))`. Wrong sign AND missing the `ω` factor. The same
   error pattern recurs in `_arrivals_to_tf:1264` with a different
   wrong scaling. The bug is silent because UACPY's tests run in
   low-loss regimes where the divergence is sub-percent; in
   high-frequency / long-range / Francois-Garrison regimes it can be
   20+ orders of magnitude.
2. **Kraken modes-reader halfspace offset
   (`io/modes_reader.py:452`)**: `read_modes_bin` reads Top/Bot from
   record `iRecProfile + 2`; both the AT `Matlab/ReadWrite/read_modes_bin.m:130`
   and `Kraken/kraken.f90:603` use `iRecProfile + 1`. The
   off-by-one drops the user inside the first phi-mode record, so the
   `'A'` halfspace check fails and the wrapper falls through to the
   vacuum/rigid branch.
3. **OAST writer block-gating bug (`io/oases_writer.py:543-563`)**:
   Blocks X, XI, XII are emitted **unconditionally** but `oast.tex`
   §IX/X/XI/XII says they are gated on options C, D, f, Z respectively.
   Default-option OAST runs (no C/D/f/Z) carry three extra lines that
   shift subsequent block parsing. Today's tests run with options that
   make the off-by-three benign at EOF; as soon as a user enables one of
   the gated options the parser shifts.
4. **OASP SSP gradient flattening (`io/oases_writer.py:937-939`)**:
   every water-column layer is written with `CS=0` so OASP treats a
   Munk-like gradient SSP as a stack of isovelocity slabs. The OAST
   writer (lines 482-491) correctly encodes gradients with the
   `CS=-|c_next|` Airy-layer convention per `oast.tex:330-334`.
5. **`convert_attenuation_units` Q and L off by factor 2
   (`core/absorption.py:151, 154`)**: the Q→Np/m formula is `2π·f /
   (Q·c)` but AT `AttenMod.f90:78` is `ω / (2·c·Q) = π·f/(c·Q)` —
   exactly 2× too large. Same factor on the L side. Round-trip through
   Q or L cancels the error so isolated tests pass; any cross-check
   against an AT-produced value disagrees.

Plus one **Critical-in-spec, latent-in-practice** finding:

- **OASES `_inject_volume_attenuation` collides with OASES Block II
  option letters**. `Bounce` injects `'T'` for Thorp, which means
  "transfer-function output" in OASN. `Biological` injects `'B'`, which
  in OASR activates the P-Slow Biot reflection-coefficient path —
  Fortran abort when the bottom isn't Biot. The intent was good
  (record the formula choice); the namespace overload is the bug.

Several **high-impact** items: KrakenField range segmentation never
uses change-point alignment in production (`Kraken.n_segments` is
always set, so the change-point logic is dead code), the broadband
default constant `BANDWIDTH_FACTOR=1.0` produces a band centred on
`1.5·fc` for Scooter/Kraken/Bellhop (not `fc` as the docstring claims),
RAM's `_run_tl` returns dtype-complex pressure with phase identically
zero, and a number of OASES writer edge-cases that silently drop
multi-component / non-isotropic / multi-Fourier-order output.

---

## Critical

### SC-1. Bellhop volume-attenuation factor: wrong sign + missing `ω`

**`uacpy/models/bellhop.py:148`** and **`uacpy/models/bellhop.py:1264`**.

The arrivals-to-time-series path computes
```python
atten = np.exp(-delay_imag[ia])
```

The canonical formula, verified against three upstream sources:

- **AT MATLAB `delayandsum.m:134`**:
  `VolumeLoss = exp( omega * imag( Arr.delay( iarr ) ) )`
- **AT `InfluenceGeoHat.m:116`** & **`InfluenceGeoGaussian.m:126`**:
  `contri = ( abs(amp) .* exp( omega * imag(delay) ) ./ W ).^2 .* W`
- **AT `delayandsum.m:130` comment**: *"delay is a complex value with
  the imaginary part holding loss due to volume attenuation. Thus,
  e(-i·ω·delay) has a phase delay and an amplitude decay."*

Mechanism: the per-arrival pressure contribution is
`A · exp(-i·ω·τ)` with complex `τ = Re(τ) + i·Im(τ)`. Expanding,
`exp(-i·ω·(Re + i·Im)) = exp(-i·ω·Re) · exp(ω·Im)`. The amplitude
factor is `exp(ω·Im(τ))`. For attenuating rays AT's CRCI
(`misc/AttenMod.f90:21-81`) produces `Im(c) > 0` and Step.f90:73
integrates `1/CMPLX(c, cimag)` over the ray → `Im(τ) < 0` →
`exp(ω·Im(τ)) < 1`. Correct.

UACPY computes `exp(-Im(τ))`. Two divergences:
- **Sign**: missing factor of −1 cancels with the canonical formula's
  positive ω, but UACPY went the wrong direction.
- **Scale**: drops the factor of `ω = 2π·fc`. For fc=10 kHz and a long
  attenuated ray with `Im(τ) ≈ −1 ms`, the canonical loss is
  `exp(−62.8) ≈ 5·10⁻²⁸` (path effectively killed); UACPY computes
  `exp(0.001) ≈ 1.001` (loss ignored, with faint amplification).

The comment at `bellhop.py:145-147` reads:
*"Volume attenuation evaluated at fc (narrowband approximation).
delay_imag is τ_i with α(fc)·r ≡ τ_i"*. But `delay_imag` comes
straight from `read_arr_file` which copies the value from
`ArrMod.f90:122`'s `AIMAG(delay)` — no scaling. The author's mental
model was that `delay_imag` was already absorbed; it isn't.

`_arrivals_to_tf` has a related-but-different bug:
```python
atten = np.exp(-delay_imag[ia] * omega / (2.0 * np.pi * fc))
```

Canonical is `exp(ω·Im(τ))`, i.e. `2π·f` in the exponent. The
`omega/(2π·fc) = f/fc` factor used by UACPY normalises out the
frequency carrier — wrong scaling category.

**Fix**: `bellhop.py:148` → `atten = np.exp(2*np.pi*fc * delay_imag[ia])`;
`bellhop.py:1264` → `atten = np.exp(omega * delay_imag[ia])`.

Cross-validation path: drop the `Bellhop broadband → IFFT` time series
against the AT MATLAB `delayandsum.m` output for the bundled
`Acoustics-Toolbox/tests/Munk` case; current UACPY diverges in the
late-time tail; after the fix the two should agree within
sample-quantisation.

### SC-2. Kraken modes-reader Top/Bot record offset off by one

**`uacpy/io/modes_reader.py:452`**:
```python
fid.seek((iRecProfile + 2) * lrecl, 0)
```

Both upstream readers use `+1`:
- **AT Fortran `KrakenField/ReadModes.f90:69`**:
  `READ( ModeFile, REC = IRecProfile + 1 ) ... BCTop ...`
- **AT MATLAB `read_modes_bin.m:130`**:
  `rec = iRecProfile + 1; fseek( fid, rec * lrecl, -1 ); ... Modes.Top.BC = fread( fid, 1, '*char' );`

The Fortran writer at `Kraken/kraken.f90:603-605` puts Top/Bot in
exactly that record:
```fortran
WRITE( MODFile, REC = iRecProfile + 1 ) &
     HSTop%BC, ..., HSBot%BC, ...
```

UACPY's `+2` lands inside the first phi-mode record (`iRecProfile + 1 +
mode` with `mode=1`). The byte interpreted as `Top["BC"]` is the high
byte of a `float32` mode value — virtually never `'A'`. Cascade at
`modes_reader.py:609-624`: `if Modes["Top"]["BC"] == "A":` is False,
the halfspace path falls through to the else branch (rho=1, gamma=0,
phi=0), and any downstream `k²`/γ calculation silently zeroes out the
halfspace radiation contribution.

**Scope**: only callers of the public `read_modes()` API are affected.
The `kraken.py` model wrappers extract only `k`, `phi`, `z` from
`read_modes_bin` (kraken.py:196-198) — they never touch Top/Bot — so
TL results from `KrakenField.run(...)` are unaffected. But the public
`from uacpy.io import read_modes` path returns garbage halfspace data.

**Fix**: `(iRecProfile + 2) * lrecl` → `(iRecProfile + 1) * lrecl`.
Add a Pekeris-bottom regression that asserts
`read_modes(...)['Top']['BC']` == `'A'`.

### SC-3. OAST writer emits Blocks X/XI/XII unconditionally

**`uacpy/io/oases_writer.py:543-563`**.

Per `oast.tex` table at §IX-XII (citing the LaTeX in
`third_party/oases/doc/oast.tex`):

| Block | Content | Gating |
|---|---|---|
| IX | TL axes | Only for options A, D, T |
| X | Depth axes | Only for options C, D |
| XI | Contour levels | Only for option C, f |
| XII | SVP axes | Only for option Z |

UACPY gates Block IX correctly (lines 533-541) but emits Blocks X, XI,
XII unconditionally (lines 545, 549, 551-563). Default OAST options is
`'N J T'` (no C, no D, no f, no Z), so the file always carries three
spurious lines after Block IX that the Fortran reader consumes as
trailing data. EOF tolerance is doing the work — tests pass because
the spurious tail lands at file end. Any user that enables one of the
gated options shifts the parser by three lines and reads garbage.

**Fix**: same gating pattern as Block IX, against the matching
option-character sets.

### SC-4. OASP writer flattens every SSP gradient layer to isovelocity

**`uacpy/io/oases_writer.py:937-939`**:
```python
for i in range(len(ssp_data)):
    d, c = ssp_data[i]
    f.write(f"{d:.2f} {c:.2f} 0 0.0 0 1.0 0 0 0\n")
```

The third column is `CS` (shear or — for water layers — the next-layer
compressional speed). Per `oast.tex:330-334`:
*"If CS < 0, it is the compressional velocity at bottom of layer, which
is treated as fluid with 1/c² linear"* (Airy-layer encoding for a
gradient).

The OAST writer (`oases_writer.py:482-491`) correctly encodes a
depth-dependent SSP with `CS = -|c_next|`. The OASP writer skips this
and emits `CS=0` for every layer — telling OASES "isovelocity to the
next interface". A user supplying a Munk SSP, downward-refracting
summer profile, or any non-constant `c(z)` gets a silently-corrupted
OASP run.

**Fix**: replicate the `_emit_water_column` logic from OAST (or share
the helper between the two writers).

### SC-5. `convert_attenuation_units` Q and L are off by factor 2

**`uacpy/core/absorption.py:151, 154, 170, 173`**.

UACPY's Q → Np/m at line 151:
```python
alpha_nepers_m = 2 * np.pi * frequency / (alpha * sound_speed)
                # = ω / (c · Q)
```

AT's canonical formula at `AttenMod.f90:78`:
```fortran
IF ( c * alpha /= 0.0 ) alphaT = omega / ( 2.0 * c * alpha )
                                  ! = ω / (2 · c · Q)
                                  ! = π · f / (c · Q)
```

UACPY's value is 2× too large. The reverse mapping at line 170
(`Q = ω / (α · c)`) returns 2× the correct Q.

UACPY's L → Np/m at line 154:
```python
alpha_nepers_m = alpha * np.pi * frequency / sound_speed
                # = L · π · f / c = L · k / 2
```

AT at `AttenMod.f90:80`:
```fortran
IF ( c /= 0.0 ) alphaT = alpha * omega / c
                          ! = L · ω / c = L · k
```

UACPY's value is half AT's. Round-trip Q→dB/m→Q or L→dB/m→L cancels
the error so tests using the helper self-consistently pass; any
cross-check against an AT-produced value disagrees.

**Fix**: replace lines 151 and 154 with the AT formulas; lines 170 and
173 likewise.

---

## High

### SH-1. OASES option-letter overload via `_inject_volume_attenuation`

**`uacpy/io/oases_writer.py:40-65`**.

Acoustics-Toolbox `TopOpt` position-4 codes — `T` (Thorp), `F`
(Francois-Garrison), `B` (biological) — are injected into the OASES
options string. But each letter has a different meaning in each OASES
sub-model:

- **OASN `T`** = transfer-function output to `.trf`, **requires NDNS=1**
  (`oasn.tex:159-164`)
- **OASR `T`** = generate `.rco` table (`oasr.tex:146-149`)
- **OASR `B`** = **P-Slow wave reflection coefficient** for **Biot
  layers only** (`oasr.tex:132-134`)
- **OASES `F`** = not defined in any Block II table → emits an
  "UNKNOWN OPTION" diagnostic but otherwise harmless

So an OASR user with a `Biological` env silently switches OASR to the
Biot path; on a non-Biot bottom OASES aborts with an unhelpful Fortran
error. **Fix**: don't try to record formula choice in the options
string. Stash it on `result.metadata['absorption_formula']` instead
(the metadata registry just landed; this is the kind of thing it
exists for).

### SH-2. Broadband default band biased above `fc`

**`uacpy/core/constants.py:24-30`** vs **`models/scooter.py:295-310`,
`models/kraken.py:1425-1432`, `models/bellhop.py:1168-1170`**.

`core/constants.py:24-30` docstring claims:
> `[fc·(1-BW/2), fc·(1+BW/2)]` (clipped to [0, 2·fc])

All three consumers actually compute:
```python
np.linspace(max(1, fc*(1-BW)), fc*(1+BW), N)
```

With `BW=1.0` (default) the band is `[1, 2·fc]`. Arithmetic centre is
`(1+2·fc)/2 ≈ fc + 0.5` — i.e. **`1.5·fc`**, not `fc`. The source
frequency sits at ~1/3 of the band, biasing every downstream H(f) →
p(t) IFFT toward upper sidebands. Bellhop's `_run_broadband` docstring
also propagates the wrong formula.

**Fix**: pick one. Two reasonable options:
- **Docstring → match code**: `[fc·(1−BW), fc·(1+BW)]`, clipped to
  `[1, ∞)`.
- **Code → match docstring** (and centre on fc):
  `[fc·(1−BW/2), fc·(1+BW/2)]`.

The latter is what the user almost certainly wants, especially for
Scooter (the previous Scooter-only default was `[0.5·fc, 2·fc, N=64]`
— ±1 octave log-symmetric, which the canonical wavenumber-integration
use case expects).

### SH-3. KrakenField range segmentation ignores RD change-points in production

**`uacpy/models/coupled_modes.py:57-83`** + **`uacpy/models/kraken.py:1166`**.

`segment_environment_by_range(env, n_segments=10)` — `self.n_segments`
defaults to 10 and is **always** passed, so the function's first
branch (line 57: `if n_segments is not None: segment_ranges_m =
np.linspace(...)`) always wins. The change-point-aware logic (lines
64-83 — unions `bathy.ranges`, `ssp.ranges` (if RD), `bottom.ranges`
(if RD), inserts intermediates wherever the gap exceeds
`max_segment_length=2000 m`) is dead code in 100% of production calls.

Per Jensen §5.7 (page ~360), adiabatic and coupled-mode theory assume
slowly-varying environment within each segment; segment edges should
coincide with environment discontinuities (bathymetric corners, SSP
switching points, sediment-layer transitions). Uniform linspace
alignment means a 1-km wedge gets its discontinuity sliced mid-segment.

**Fix**: pass `n_segments=None` by default; let the change-point logic
run. Promote `n_segments` to a kwarg override for users who want a
forced uniform decomposition.

### SH-4. RAM `_run_tl` returns dtype-complex pressure with phase identically zero

**`uacpy/models/ram.py:2054-2065`**.

```python
psi_mag = np.abs(pressure_rcv) * 4.0 * np.pi
p_mag = psi_mag / np.sqrt(r_safe)[np.newaxis, :]
pressure_field = p_mag.astype(np.complex128)
```

The returned `Field` is dtyped complex but `Im(pressure_field) ≡ 0`.
The broadband path at lines 2170-2196 correctly carries phase via
`exp(-i·k0·r)/√r` multiplied by the conjugate of the mpiramS `psif`;
the COHERENT_TL path strips it.

Consequence: downstream code doing `np.angle(field.data)` or coherent
addition with another model's field sees zero phase. `field.tl` works
fine (it only needs `|p|`), and the docstring at line 2058
acknowledges "consumers that only need TL can read field.tl, which
depends on |p| only". The structural issue: a `Field` of dtype complex
claims to carry phase; downstream consumers don't check.

**Fix**: either (a) compute and keep the phase as `_run_broadband`
already does, (b) emit a real-valued `Field` and document, or (c)
attach `phase_reference=None` plus a `magnitude_only=True` metadata
flag so downstream can detect.

### SH-5. RAM `_pade_optimizer.theta_max` unit footgun

**`uacpy/models/_pade_optimizer.py:259, 284-285`**.

Default value: `theta_max = np.deg2rad(30.0) ≈ 0.524`. Docstring at
line 285 says "Default 30°". Internal use is radians (line 321:
`np.sin(theta_max)`). All in-tree callers (ram.py:1684, ram.py:496)
correctly pass `np.deg2rad(self.theta_max)`, so live code is fine. But
a direct user reading the docstring and calling
`optimize_grid(theta_max=30)` computes the Padé spectrum on `sin²(30
rad) ≈ 0.98` — a meaningless cap.

**Fix**: accept degrees publicly to match the docstring (convert
internally), or rename to `theta_max_rad`.

### SH-6. Scooter `read_grn_file` hard-codes little-endian

**`uacpy/io/grn_reader.py:64-118`**.

Uses literal `'<i'`, `'<f4'`, `'<f8'` everywhere. The recent batch-C
endian-aware rewrite in `_fortran_helpers` and `oalib_reader` did not
propagate here. Big-endian `.grn` files (cross-compiled mpiramS on a
non-x86 host) silently return garbage Green's functions.

**Fix**: probe the first 4 bytes via `_fortran_helpers.detect_endian`
and thread the byte order through every `np.fromfile` / `struct`
call.

### SH-7. SPARC pulse-type alphabet over-accepts `T` and `C`

**`uacpy/models/sparc.py:39`** (`_PULSE_TYPE_POS1`).

UACPY accepts `set('PRASHNGFBMTC')` (12 chars). Fortran `sparc.f90`'s
`GetPar` SELECT CASE at lines 126-148 covers exactly `P R A S H N M G
F B` (10 chars) and calls `ERROUT('Unknown source type')` for anything
else. `T` and `C` are valid in `tslib/cans.f90:82,87` (the pulse
*evaluator*), but `sparc.f90`'s `GetPar` never reaches `cans.f90` for
them. The wrapper docstring even claims "strictly validated in
sparc.f90 SELECT CASE" — it isn't.

**Fix**: drop `T` and `C` from the set.

### SH-8. Bellhop default `bandwidth_factor=1.0` violates user-guide sub-banding guidance

**`uacpy/core/constants.py:30`**.

User Guide §9 page 45: *"the ray/beam process uses attenuation values
at the center frequency of the calculation. For sources with a great
deal of bandwidth one should divide the frequency band into sub-bands
so that the appropriate attenuation is used in each sub-band."*

UACPY's default `bandwidth_factor=1.0` builds a 200%-relative-bandwidth
H(f) grid from a single fc arrivals run. Ray geometry (caustic count,
beam-width, eigenray bracketing) is mildly frequency-dependent at the
edges. The result is a smooth H(f) that *looks* broadband but is built
from a single fc arrivals snapshot.

**Fix**: lower the default to ~0.25 or 0.5, document the sub-banding
requirement for wider bands, and refuse `bw > 0.5` without an explicit
override.

### SH-9. OASN multi-frequency path silently collapses non-equispaced grids

**`uacpy/models/oases.py:511-582`** + **`uacpy/io/oases_writer.py:160-162`**.

OASR and OASP route a user `frequencies=` vector through
`_oases_resample_frequencies` to warn-and-resample. OASN's `run()`
does not. `_resolve_freq_sweep` in the writer silently collapses to
`(min, max, len)` — so a user `Source(frequencies=[100, 200, 500])`
becomes `(100, 500, 3) = [100, 300, 500]` with no warning.

**Fix**: call `_oases_resample_frequencies` from OASN's run path too.

### SH-10. OASP TRF reader drops MSUFT / ISROW / NOUT axes

**`uacpy/io/oases_reader.py:664-672`**.

The OASES Fortran writer (`oases/src/oasiun23.f:305-311`) loops
`IS=1..ISROW, M=1..MSUFT, JRH=1..NPLOTS, JRV=1..IR` writing one record
of `NOUT` complex values per quad. The reader allocates
`transfer_function = np.zeros((nf, nplots, nrd), dtype=np.complex64)`
(no IS / MSUFT / NOUT axes) and stores only `complex(rec[0], rec[1])`,
i.e. only the first complex of the `NOUT` field components. For
NOUT > 1 (pressure + velocity), MSUFT > 1 (tilted arrays), or ISROW > 1
(decomposed seismograms via option `U`) the extra axes are silently
discarded.

**Fix**: detect these cases in the header read, allocate the right
shape, store all axes. Or refuse the run with a clear error when the
options demand multi-axis output.

### SH-11. SPARC snapshot path missing the IFFT × 2 factor for one-sided spectrum

**`uacpy/io/grn_reader.py:410`** vs **`uacpy/io/oalib_reader.py:1331`**.

`sparc_snapshot_to_field` computes `G_freq = np.fft.fft(G, axis=0) *
dt` and selects one positive-frequency bin without the
conjugate-symmetric ×2 factor. `rts_to_pressure` does:
```python
p_at_freq = 2.0 * p_freq[freq_idx, :] / np.sum(window)
```

The ×2 recovers full amplitude from a real time-domain trace. The
snapshot path is therefore 6 dB **lower** in absolute level than the
R-mode path. This compounds with the R-vs-D Fortran-side `√π`
asymmetry and explains why `test_sparc_snapshot_vs_horizontal` has to
tolerate `median_abs < 8 dB`.

**Fix**: apply the same ×2 factor (no window in the snapshot path, so
no `/ window_sum` needed).

### SH-12. `Kraken._supports_elastic_media = True` is a contract lie

**`uacpy/models/kraken.py:567`**.

Real-arith `kraken.exe` will not produce modes for elastic media. The
flag's contract per `models/base.py:189-205` is "this env shape works
with this model". Setting True skips the projection-layer collapse,
lets `kraken.exe` execute, and surfaces a Fortran-level mode-not-found
error. The wrapper's `_modes_error_message` provides a helpful "switch
to KrakenC" message, but this is an error suggestion, not auto-routing.
The earlier `AUDIT.md` finding `[CORE-models.py:573]` flagged this; the
current behaviour still violates the flag's contract.

**Fix**: either flip to `False` (force collapse to fluid, lose physics)
or auto-route at `run()` time to KrakenC. The latter preserves user
intent and is a single-method refactor (`KrakenField._select_kraken_exe`
already does it).

### SH-13. `default C_LOW_FACTOR=0.95` skips slow interfacial modes

**`uacpy/core/constants.py:15`**.

Per the KRAKEN manual: *"CLOW will… mainly… exclude interfacial modes
(e.g. a Scholte wave)."* The safe default for `c_low` is 0. Setting
`c_low = 0.95·c_min` drops slow trapped modes whose phase speed lies in
`[0.95·c_min, c_min)`. For a soft sediment (`c_sed < c_water`) this
discards Stoneley modes that **do** propagate.

**Fix**: default `c_low_factor=0.0` (manual's recommendation); make
0.95 opt-in for users who specifically want to skip Scholte modes.

### SH-14. OASN block-V replica y-axis writer always emits `"0 0 1"`

**`uacpy/io/oases_writer.py:773-775`** vs **`models/oases.py:545-548`**.

The writer accepts `replica_xmin/xmax/nx` and `replica_zmin/zmax/nz`
but **never reads** `replica_ymin/ymax/ny`. The model wrapper
correctly converts y-axis kwargs from m to km, then they're dropped at
the writer. Three-dimensional replica grids (`y > 0`) are unreachable.

**Fix**: accept the y-range kwargs in the writer; mirror the x/z
pattern.

### SH-15. OASN discrete-source x/y written without unit conversion

**`uacpy/io/oases_writer.py:751-754`**.

`oasn.tex:343`: discrete-source x/y are kilometres on disk. The writer
takes them verbatim from `ds['x'] / ds['y']`. The model wrapper
(`oases.py:545-548`) converts the **replica** grid x/y from m to km,
leaving `discrete_sources` untouched. So
`OASN.run(..., discrete_sources=[{'x': 2500, ...}])` writes `2500 km`
on disk instead of the intended 2.5 km.

**Fix**: apply `m_to_km` symmetrically on the discrete-source x/y in
the wrapper.

### SH-16. Thorp coefficients drift from JKPS Eq. 1.34

**`uacpy/core/absorption.py:64`**.

UACPY: `α = 0.11·f²/(1+f²) + 44·f²/(4100+f²) + 2.75e-4·f² + 0.003`
JKPS Eq. 1.34 (used by AT `AttenMod.f90:94`):
`α = 3.3e-3 + 0.11·f²/(1+f²) + 44·f²/(4100+f²) + 3e-4·f²`

Two coefficient drifts:
- Viscous: 2.75e-4 (UACPY) vs 3e-4 (JKPS/AT). 9% off at high f; at 100
  kHz the viscous term is 2750 vs 3000 dB/km — 250 dB/km difference,
  dwarfing the relaxation terms.
- Constant: 0.003 (UACPY) vs 3.3e-3 (JKPS/AT). 0.3 vs 0.33 dB/km.

UACPY's values match arlpy and many MATLAB implementations; JKPS uses
an updated coefficient set. Pick one (JKPS/AT for consistency with the
rest of the AT-driven UACPY stack) and document the choice.

### SH-17. `beamform()` NL convention undocumented

**`uacpy/acoustic_signal/processing.py:144-150, 169`**.

After the recent rename (audit VIS-2), the formula
`snr = 20·log10(|e @ p|) + SL - NL` is correct, but:

1. The steering-vector matrix from `planewave_rep` is unit-normalised,
   so `|e @ p|` sums coherently for an N-element array → the array
   gain `10·log10 N` is implicitly folded in. NL must therefore be
   *per-element* noise level, not array-total. Document this.
2. NL is documented as "dB re 1 µPa²/Hz" but the formula treats it as a
   wideband level (no integration bandwidth). For broadband signals
   the user must pre-integrate. Document.

---

## Medium

### SM-1. SSP-precision loss in OAST/OASP layer write

**`uacpy/io/oalib_writer.py:178-183`** (`write_ssp`): writes sound
speeds with `%6.1f`. Munk profiles at `1502.345 m/s` silently round to
`1502.3` across every column. Bump to `%.4f` or `%.6f`.

### SM-2. RAM `_compute_grid_lytaev` λ cap mixes c_min/(5·f) and 5×optimiser

**`uacpy/models/ram.py:1725-1735`**. Two independent multiplicative
tightenings applied to the Lytaev `dr` for the elastic backend. Tighter
wins. Empirical, defensible, undocumented — comment cites
`test_cross_model_agreement.py::_pekeris_elastic` validation. Both
contribute, neither is obviously redundant. Worth a comment explaining
why both are needed.

### SM-3. OASN block-V replica `CMAXS` defaults to `c_p · 1.05`

**`uacpy/io/oases_writer.py:759, 779`**. OASN manual recommends
`CMAXS=1E8` for surface noise (steep angles); the writer uses 1E8 for
surface/deep but `c_p · 1.05` for discrete-source/replica. Magic
constant with no env-aware fallback. Bottom interactions near the
critical angle silently truncated.

### SM-4. RAM auto-loosen ladder reduces θ_max to 10° silently

**`uacpy/models/ram.py:1672-1710`**. When Lytaev declares ε=1e-3
infeasible, ε is multiplied by 3 up to 8 times, then θ_max steps down
from 30° to 10°. Wide-angle PE physics breaks down at 10°. The warning
fires but understates the situation — at 10° the PE is paraxial-only.
Either raise the floor to 15-20° or reject θ_max < 15°.

### SM-5. SPARC R-vs-D bias is mechanically `√π·exp(...)` not a tolerance choice

**`uacpy/models/sparc.py:567-593`** + **`third_party/Acoustics-Toolbox/Scooter/sparc.f90:292,622-623`**.
`sparc.f90` R-mode accumulates `RTSrr += √2·dk·√k·U·exp(...)·√(1/r)`;
D-mode accumulates `RTSrz` without `1/√r` then applies `Scale =
1/√(π·Pos%Rr(1))` at write-out. The per-range D-mode loop in the
wrapper fixes this at `√π` (~5.0 dB). The R-vs-D test tolerance of
3 dB residual + 10 dB bias is **physically defensible** (the Fortran
asymmetry is documented), but neither the test docstring nor the
wrapper note cites `sparc.f90:292`. Add the citation.

### SM-6. `Modes.with_attenuation` discards complex `1/√k`

**`uacpy/core/results.py:1720`**: `inv_sqrt_k = 1.0 / np.sqrt(np.abs(k))`
discards the phase contribution from `Im(k)`. For weakly-attenuated
modes the phase error is O(Im(k)/Re(k)) per mode — negligible for TL
but matters for phase-sensitive MFP. Fix: `np.sqrt(k)` (numpy handles
complex sqrt with the principal branch).

### SM-7. RAM `_run_collins_one_freq` Q=1e6 collapses bw to ~0

**`uacpy/models/ram.py:1397-1404, 1910-1911`**. Default Q=1e6 makes
`bw = fc/Q ≈ 0`, and mpiramS's broadband sweep collapses to 2-3 bins.
Works in practice; the choice is arbitrary. Either document or pick a
Q-based gating rule.

### SM-8. `compute_windnoise` water_depth fallback silent

**`uacpy/noise/noise.py:102-107`**. Falls back to deep `cst=42` for any
unrecognised `water_depth` with no warning. Combined with the
Beaufort-vs-knots wind-speed convention asymmetry, easy to silently
mis-configure.

### SM-9. OAST `_inject_volume_attenuation` adds `F` to OAST options string

**`uacpy/io/oases_writer.py:42-65`**. See SH-1. For OAST specifically,
`F` is not defined in OAST's Block II, so OASES emits "UNKNOWN OPTION"
but continues — benign. The bug class is the same as SH-1; flagged
here for completeness.

### SM-10. OAST/OASN/OASR write SSP every layer; OAST caps at 15

**`uacpy/io/oases_writer.py:432-442 vs 697-704, 937-939`**. OAST limits
the water-column SSP to 15 layers (comment: "OASES numerical issue
threshold"). OASN and OASP do not. For a 50-point user SSP, OASN/OASP
will write all 50 layers, potentially exceeding OASES's internal
limit. Either subsample symmetrically or document the cap.

### SM-11. OASES `_oases_resample_frequencies` silent on truncation

**`uacpy/models/oases.py:801-857`**. Resamples without explaining
whether the user's exact frequencies are preserved. For OASR/OASP this
matters because the user typically wants specific frequencies, not a
nearest-bin approximation.

### SM-12. SPARC `n_t_out=501` is not power-of-2

**`uacpy/models/sparc.py:195`**. `np.fft.fft` on length 501 = 3·167 is
~10× slower than the next power of two. Auto-round-up to 512 or 1024
or 2048 depending on user-supplied `T`.

### SM-13. `WenzNoise` shipping `c2='no'=4` is dead code

**`uacpy/noise/noise.py:142, 250-254`**. Line 142 sets
`_SHIPPING_C2['no'] = 4`, line 250 special-cases `'no'` to `-np.inf`,
making the 4 unreachable. Cosmetic.

### SM-14. `SEL.compute` window-energy normalisation drifts

**`uacpy/acoustic_signal/analysis.py:466-473`**. `spectrogram(...,
scaling='spectrum', noverlap=0, mode='psd')` with Hann gives
amplitude² normalised by `(sum(window))²`. Reconstructing `∫|p|²dt`
across blocks at noverlap=0 with a Hann window has a ~8/3 power-vs-amp
factor that the formula doesn't compensate. Switch to a rectangular
window for energy integration, or document the Hann-induced bias.

### SM-15. `fourier_synthesis` time vector ignores `Tstart`

**`uacpy/acoustic_signal/processing.py:446`**. `time = np.linspace(0,
Tmax - dt, Nfreq)` always starts at 0 regardless of `Tstart`. Users
passing `Tstart = r/c` expect the time axis to be absolute travel-time
anchored. Document or return `time + Tstart`.

### SM-16. `FRF` H1 docstring uses non-standard `Pyx/Pxx` notation

**`uacpy/acoustic_signal/analysis.py:700-705, 881-885`**. Docstring
says `H1 = Pyx/Pxx`; code computes `conj(Pxy)/Pxx` where `Pxy =
csd(y, x)`. Math is correct (scipy's `csd(x,y) = E[X*Y]`), the symbol
assignment is non-standard. Rename `Pxy → Pyx` or update docstring.

---

## Low

### SL-1. Bellhop `delayandsum` `c0` parameter is unused

**`uacpy/models/bellhop.py:81`**. Accepted but never used inside the
function body. Vestigial from a prior signature.

### SL-2. `bellhop.py` duplicate `Notes` sections in docstring

**`uacpy/models/bellhop.py:250, 266`**. Two consecutive `Notes`
headings in the `Bellhop` class docstring (artifact of an earlier
docstring move). Fold into one.

### SL-3. `read_arr_file` units undocumented

**`uacpy/io/oalib_reader.py:402` docstring**. The returned `phases`,
`src_angles`, `rcv_angles` are all in degrees per `ArrMod.f90:120`; the
docstring doesn't say. One-line addition.

### SL-4. `_read_oasp_trf_binary` declares unreachable else branch

**`uacpy/io/oases_reader.py:659-662`**. `nf = max(1, mx - lx + 1)` so
the `else` for `nf < 1` is dead.

### SL-5. Ricker / cans 'R' offset disagreement

**`uacpy/acoustic_signal/generation.py:385-386 vs 274-275`**.
`ricker_wavelet` uses peak at `t = 8/(2π·F)`, `cans('R')` uses peak at
`t = 5/(2π·F)`. Same envelope, different offsets. Pick one and
document.

### SL-6. `pekeris_root` is documented under "arlpy adaptation" but isn't

**`uacpy/core/acoustics.py:507-552` + `third_party/arlpy/NOTICE`**. The
NOTICE update covered the absorption-class divergence; an explicit note
that `pekeris_root` is *not* arlpy-adapted would close the audit loop.

### SL-7. `Kraken._read_modes_file` passes `freq=0.0` to `read_modes_bin`

**`uacpy/models/kraken.py:418`**. For broadband Kraken runs (TopOpt(6)
= 'B') the wrapper picks the lowest stored frequency, not the source.
Currently latent (single-freq MODES enforced upstream).

### SL-8. `Modes` with attenuation cannot round-trip to `.mod` for field.exe

**`uacpy/core/results.py:1648`**. The analytical-sum convention
(`exp(+i·k·r)`) gives positive Im(k) for decay; field.exe uses
`exp(-i·k·r)` and zeroes positive Im(k). Two coherent conventions but
not cross-compatible. Latent.

### SL-9. AT 'm' (dB/m power-law) and 'F' (dB/(m·kHz)) units not in
`convert_attenuation_units`

**`uacpy/core/absorption.py:141-176`**. AT recognises both; UACPY's
helper raises `ConfigurationError` on either. Either add or warn.

### SL-10. `FKTransform.compute` lacks PSD normalisation

**`uacpy/acoustic_signal/analysis.py:1476-1479`**. `FK = fftshift(fft2(data))`
is unnormalised; the dB label claims a PSD-like quantity. Cosmetic.

---

## Recent-change regression check

Items the audit team specifically rechecked against the just-completed
B / C / D batches and the metadata-registry rewrite:

- `io/units.py` km↔m helpers — applied in most write sites, but
  `bellhop_writer.py:419` (`r_box/1000.0`), `oalib_writer.py:629`
  (`r/1000.0`), and `models/oases.py:548` (replica xy `/ 1000.0`)
  still use inline arithmetic. **Numerically identical today**;
  consistency only.
- `_DEFAULT_BROADBAND_*` constants in `core/constants.py` — the SH-2
  finding above is a docstring/code mismatch introduced by the
  unification. Should be reconciled (preference: change the code to
  match the docstring, centring the band on fc).
- Endian auto-detect — works correctly in `oalib_reader.read_shd_bin`,
  `oases_reader.read_oasn_covariance`, `read_oasn_replicas`, and
  `ramsurf_reader._read_lz_records`. Two readers still hard-code
  little-endian: `grn_reader.read_grn_file` (SH-6 above) and
  `oases_reader.read_oasp_trf` (related, lower priority).
- `Result.list_metadata` + the `_DOCUMENTED_METADATA` registry — the
  expanded registry covers everything actually written by every
  wrapper. The earlier-flagged `('Bellhop', 'time')` entry (incorrectly
  added — `time` is a coord not a metadata key) was removed. The
  `_UNIVERSAL_METADATA` lookup for `prt_file` is now wired into
  `list_metadata()`.
- OASES factory typo handling — works correctly. `inspect` is no
  longer imported; typos cascade through `super().__init__(**kwargs)`
  to `PropagationModel.__init__` (typed-only signature) and raise
  `TypeError`.
- Marker-derivation in `test_examples_integration.py` — confirmed
  correct. Examples 09/10/12/25 (pure-Python) are now unmarked; ex_02
  correctly does not carry `requires_oases`.
- Munk regression — physically meaningful and runs in <100 ms;
  thresholds are 50% of measured values.
- SPARC R-vs-D residual check — physically defensible (the bias is the
  documented Fortran `√π` asymmetry); flagged in SM-5 only because the
  test docstring should cite `sparc.f90:292`.

---

## Recommended priority order

The five **Critical** findings (SC-1 through SC-5) are all real bugs
that produce silently-wrong output under realistic operating
conditions. Recommended order to address them:

1. **SC-5** — `convert_attenuation_units` Q/L factor-2 fix. Mechanical,
   no API change, two-line edit.
2. **SC-1** — Bellhop volume attenuation. Two-line edit per function;
   add a Munk-band cross-check vs AT MATLAB `delayandsum.m`.
3. **SC-2** — `modes_reader.py` Top/Bot offset. Single-line edit;
   regression with a Pekeris `.mod` asserting `Top['BC'] == 'A'`.
4. **SC-4** — OASP SSP gradient flattening. Replicate OAST's
   `_emit_water_column` logic; add a Munk-OASP test.
5. **SC-3** — OAST block-X/XI/XII gating. Match the Block IX gating
   pattern; add tests with options `D`, `Z`, `f` that exercise each.

Then the **High** findings, prioritised by user-visibility:

- **SH-1** (option-letter overload) — fix before any user runs OASR/OASN
  with `Biological` absorption.
- **SH-2** (broadband-band centring) — decide docstring-or-code; align
  Bellhop / Scooter / KrakenField.
- **SH-3** (KrakenField segmentation) — flip default; the dead
  change-point logic is the right answer.
- **SH-12** (Kraken elastic flag) — auto-route to KrakenC.
- **SH-4** (RAM `_run_tl` phase) — either keep phase or document
  magnitude-only.

The remaining High items are individually scoped fixes; Medium and Low
are housekeeping.

---

## Audit method

Five general-purpose agents ran in parallel against disjoint slices.
Each was given the full architecture context (CLAUDE.md, AUDIT.md), the
upstream reference docs / PDFs / books, and a slice-specific brief
asking for severity-tagged findings with file:line citations and
authoritative-reference page numbers. The findings were then
deduplicated where slices overlapped and verified independently for
the critical items (in particular the Bellhop SC-1 finding was
double-checked against the AT MATLAB `delayandsum.m:134`,
`InfluenceGeoHat.m:116`, `InfluenceGeoGaussian.m:126`, and the
Fortran `ArrMod.f90:122` / `AttenMod.f90:21-81` / `Step.f90:73`
chain).

Per-slice raw agent transcripts are not shipped as separate files —
their content is folded into the sections above.
