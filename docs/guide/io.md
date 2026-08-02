# File I/O — the layer between metres and the native formats

> `uacpy.io` · 82 public names · every reader and writer the models run on

Underneath the Python API, uacpy drives seven native solvers by writing text
and binary files, launching a subprocess, and parsing what comes back. Each of
those solvers has its own file formats, its own record layout, and its own
opinion about units. `uacpy.io` is the only part of the package that knows any
of that.

Two things follow, and they are the spine of this page:

1. **`io` is the only place unit conversion happens.** uacpy is SI-metres
   everywhere else. Kilometres exist inside these file formats and on plot
   axes — nowhere in the API.
2. **`io` is public.** The same functions the models call are importable, so
   you can read a `.shd` another tool produced, or the outputs a pinned
   `work_dir` left behind, without reverse-engineering a record layout.

```python
import uacpy
uacpy.io.read_shd_file('run/model.shd')
```

---

## 1. The one rule: metres in Python, kilometres on disk

Every public reader returns metres. Every public writer accepts metres. The
conversion happens inside, in [`uacpy/io/units.py`](../../uacpy/io/units.py),
which exists so that "did I convert?" is a grep and not an audit.

Write a bathymetry and read it back:

```python
import numpy as np
import uacpy

bathymetry = np.column_stack([[0.0, 5000.0, 10000.0],    # range, metres
                              [100.0, 150.0, 200.0]])    # depth, metres
uacpy.io.write_bty_file('t.bty', bathymetry)
```

The file on disk is in kilometres, because that is what Bellhop's `.bty` reader
expects:

```
'LS'
3
0.000000 100.000000
5.000000 150.000000
10.000000 200.000000
```

And reading it gives metres back:

```python
data, interp_type = uacpy.io.read_bathymetry('t.bty')
data[0, 1:-1]     # array([    0.,  5000., 10000.])  ← metres
data[1, 1:-1]     # array([  100.,   150.,   200.])
```

(`read_bathymetry` pads the profile with a `±1e50` sentinel point at each end —
constant extrapolation, the convention Bellhop's boundary code wants. That is
what the `1:-1` trims.)

### What converts, and what does not

| Quantity | Python API | On disk | Helper |
|---|---|---|---|
| Range, depth, thickness | metres | **km** for AT `.env` / `.bty` / `.ati` / `.ssp` / `.flp`, OASES `.dat` range specs, and the range axis of mpiramS' SSP and sediment files | `m_to_km` / `km_to_m` |
| Range, depth | metres | **metres** for `ram.in` / `rams.in` / `ramgeo.in`, mpiramS' bathymetry and output-range files, and inside `.shd` (AT converts before writing the header) | — |
| Frequency | Hz | Hz | `hz_to_khz` / `khz_to_hz` where a format wants kHz |
| Reflection **phase** | radians | degrees | `deg_to_rad` / `rad_to_deg` |
| Grazing **angle** | degrees | degrees | — |

The two that catch people: the RAM family is metres on disk almost
everywhere (unlike everything Acoustics-Toolbox — the exceptions are the range
axes of mpiramS' SSP and sediment decks), and reflection-coefficient tables are
degrees for the *angle* axis on both sides but radians-in-Python /
degrees-on-disk for the *phase* column. `read_reflection_coefficient` returns
`phi` in radians and `write_reflection_coefficient` takes radians, matching the
[`ReflectionCoefficient`](results.md) result.

`units.py` itself is not exported on `uacpy.io.__all__` — it is plumbing the
readers and writers share, not something a caller needs.

---

## 2. Three binary idioms

Almost every format in the package is one of three shapes. Knowing which one
you are looking at is most of the work of reading it.

### (a) Fortran sequential unformatted records

The layout a Fortran `WRITE` to an unformatted sequential unit produces:

```
[int32 N][N bytes of payload][int32 N]
```

The length marker appears twice, which is what makes the format
self-validating: if head and tail disagree you have either a truncated file or
the wrong byte order. `read_fortran_record` reads one, checks both markers,
and raises `FileFormatError` on a mismatch; `read_fortran_record_marker` reads a
bare marker when the payload is unpacked by hand.

**Byte order is detected, not assumed.** `detect_endian` reads the first four
bytes and picks the interpretation that yields a plausible record length — a
sane positive integer under `2**28`. Usually only one byte order qualifies;
when both do it takes the smaller marker, and when neither does the file is
corrupt and it says so. Decoding a big-endian file emits a one-shot
`UserWarning`: it works, but uacpy's CI is little-endian, so that path is not
exercised.

Used by: OASES `.trf` / `.rpo`, the RAM family's `tl.grid` and `pcomplex.bin`,
and mpiramS' `psif.dat` (which goes through `scipy.io.FortranFile` — it is
written by a binary uacpy just launched on this host, so its byte order is the
host's by construction).

### (b) Direct-access records, read with seeks and `np.fromfile`

The Acoustics-Toolbox `.shd`, `.mod` and `.grn` files are Fortran
*direct-access* files: a fixed record length `recl` (in 4-byte words) sits in
the first four bytes, and every subsequent record starts at `k * 4 * recl`.
There are no per-record length markers, so reading is a sequence of `seek` +
`np.fromfile(count=…)` calls at computed offsets.

This is why `read_shd_bin` can pull a single frequency slice out of a broadband
file without walking the whole thing: the record index is arithmetic.

### (c) `struct`-packed mixed headers

Where a record holds a heterogeneous tuple — `(int, int)`, `(float, float,
int)` — `np.fromfile` is the wrong tool and `struct.unpack` with an explicit
endian prefix is used instead. The OASES `.xsm` and `.rpo` headers are the main
consumers: `n_rcv, n_freq = struct.unpack(endian + 'ii', f.read(8))` and
friends.

### Header counts are bounded before they allocate

Every binary reader that sizes a NumPy array off an integer header field checks
it against the file size first. A corrupt or hostile file claiming
`n_rcv = 0x7fffffff` would otherwise drive a multi-terabyte `np.zeros` before a
single data record is validated. The check is arithmetic — no data item can
occupy fewer than its own itemsize on disk, so no count can exceed
`file_size // item_bytes` — and a violation raises `FileFormatError`.

### And the ASCII decks

The Acoustics-Toolbox `.env`, `.flp`, `.bty` and `.ati` decks are text, but
they are *Fortran* text: list-directed `READ` statements that stop at the first
`!`, quoted character literals that may also arrive unquoted, and vector records
that keep consuming lines until `N` values have been read — or until a `/`
terminates them early, which triggers a *generated* vector (two values before
the slash means equally spaced, one means replicated).
`strip_fortran_comment`, `strip_fortran_quotes` and `read_vector` implement
those rules so no reader has to re-derive them. The OASES `.dat` and RAM
`ram.in` decks are plain fixed-layout text that uacpy only writes.

Every helper named in this section — `read_fortran_record`,
`read_fortran_record_marker`, `detect_endian`, `read_vector` and the string
strippers — lives in
[`uacpy/io/_fortran_helpers.py`](../../uacpy/io/_fortran_helpers.py) and is
deliberately **private**: mechanism, not interface.

---

## 3. Acoustics Toolbox formats

Bellhop, Kraken, Scooter, SPARC and Bounce share one input format and a family
of outputs. See [Bellhop](../models/bellhop.md), [Kraken](../models/kraken.md),
[Scooter](../models/scooter.md), [SPARC](../models/sparc.md) and
[Bounce](../models/bounce.md) for what each does with them.

| File | Direction | Idiom | Carries |
|---|---|---|---|
| `.env` | in | ASCII | The whole problem: title, frequency, media, SSP, boundaries, source/receiver geometry |
| `.flp` | in | ASCII | Field parameters for `field.exe` — how to sum Kraken's modes into a field |
| `.shd` | out | direct-access binary | Complex pressure, `(Ntheta, Nsz, Nrz, Nrr)` |
| `.arr` | out | ASCII | Bellhop arrivals: amplitude, delay and bounce counts per path |
| `.ray` | out | ASCII or binary | Bellhop ray paths |
| `.mod` / `.moa` | out | direct-access binary / ASCII | Kraken mode shapes and wavenumbers |
| `.grn` | out | direct-access binary | Scooter / SPARC wavenumber-domain Green's function |
| `.rts`, `.ts` | out | ASCII | SPARC time series; a simpler generic time series |
| `.prt` | out | ASCII | The binary's diagnostic log — where AT puts fatal errors instead of stderr |

### `.env` — written in pieces

There is no single `write_env_file`. Each model has its own entry point —
`write_bellhop_env_file`, `write_kraken_env_file`, `write_scooter_env_file`,
`write_sparc_env_file`, `write_bounce_input_file` — because the run-type and
option characters diverge, but they compose the same section writers:
`write_header`, `write_ssp_section`, `write_layer_sections`,
`write_bottom_section`, `write_source_depths`, `write_receiver_depths`,
`write_receiver_ranges`, `write_phase_speed_and_rmax`.

Three `resolve_*` helpers decide the option characters before anything is
written, and are public so a wrapper can log the resolved values:

- `resolve_ssp_interp(env, model_interp)` — the user-facing `interp_ssp` after
  auto-resolution (`'quad'` for a range-dependent SSP, `'linear'` otherwise).
- `resolve_ssp_topopt(env, model_interp)` — the AT `TopOpt(1)` character it maps
  to. The only environment-side override is `ssp.shape='isovelocity'`, which
  forces `'C'`; see [environment](environment.md).
- `resolve_phase_speed_bounds(env, c_low, c_high)` — the modal phase-speed
  window. A vacuum or rigid bottom resolves to the AT "no upper limit" idiom
  rather than capping on a placeholder sound speed, which would silently
  truncate the mode spectrum.

`writable_layers(bottom)` is the guard that drops sub-resolution sediment
layers: a layer thinner than the `.1f` depth format can express would become a
zero-thickness AT medium.

`write_multi_profile_env` handles Kraken's range-dependent mode, where
`kraken.exe` reads profile blocks sequentially from one `.env` and writes all
of their modes into one `.mod`. Every block is padded to the same `n_mesh` and
`NMedia` because the `.mod` record length is fixed from the first profile and
must not grow.

### `.shd` — pressure out

`read_shd_file` is the everyday entry point: it returns a
[`Field`](results.md) of complex narrowband pressure, or a `ResultStack` when
the file carries several source depths. Multi-frequency files raise — use
`read_shd_bin` and build the broadband `Field` yourself, one frequency slice at
a time.

```python
tl = uacpy.Bellhop(work_dir='./run').run(env, source, receiver)
shd = uacpy.io.read_shd_file(tl.metadata['shd_file'])
shd.data.shape      # (100, 250), complex64
```

`read_shd_bin` returns the raw dictionary: `title`, `PlotType`, `freqVec`,
`Pos` (source and receiver coordinates, in metres) and a single-frequency
`pressure` cube. Cells the engine never wrote — Bellhop's `r = 0` column, ray
shadow zones, an empty modal sum — are exact zeros on disk and come back as
`NaN`, uacpy's no-data convention.

### `.grn` — the Green's function, and the transform onto ranges

Scooter and SPARC solve in the wavenumber domain, so the `.grn` file is not a
field yet. Four functions take it the rest of the way:

| Function | Produces |
|---|---|
| `grn_to_field` | one frequency → complex narrowband `Field` |
| `grn_to_transfer_function` | all frequencies → broadband `Field`, `H(z, r, f)` |
| `sparc_snapshot_to_field` | steady-state pressure at one frequency from a SPARC snapshot |
| `sparc_snapshot_to_time_field` | range-domain time evolution of a SPARC snapshot |

The Hankel transform mirrors `fieldsco.m`, Porter's reference implementation.
SPARC is detected by the `'SPARC'` prefix in the file title, because the two
writers use the header differently: for SPARC the `freqVec` slot actually holds
output *times*, and the wavenumber grid is frequency-independent.

### `.prt` — where the errors are

AT binaries write `*** FATAL ERROR ***` to `<base>.prt`, not to stderr.
`read_prt(path, tail_bytes=…)` returns the log text (or `None` if absent), and
the model layer appends its tail to `ModelExecutionError` so the actual cause
surfaces instead of a "check the .prt file" pointer.

---

## 4. Boundary, reflection and beam-pattern files

These are the small auxiliary decks that hang off an `.env` by base-name
convention. Bellhop opens `<env>.bty`, `<env>.ati`, `<env>.brc`, `<env>.trc`,
`<env>.sbp` — the names are not in the `.env`, they are implied by it.

| File | What it is | Read | Write |
|---|---|---|---|
| `.bty` | Bathymetry vs range | `read_bathymetry` | `write_bty_file`, `write_bty_long_format`, `write_bty_3d` |
| `.ati` | Sea-surface altimetry vs range | `read_altimetry` | `write_ati_file` |
| `.ssp` | Range-dependent sound-speed matrix | `read_ssp_2d`, `read_ssp_3d` | `write_ssp` |
| `.brc` / `.irc` | Bottom / internal reflection coefficient `R(θ)` | `read_reflection_coefficient` | `write_reflection_coefficient` |
| `.trc` | Top reflection coefficient `R(θ)` | `read_reflection_coefficient` | `write_reflection_coefficient` |
| `.sbp` | Source beam pattern, angle vs dB re peak | `read_source_beam_pattern` | `write_source_beam_pattern` |

**Short vs long `.bty`.** `write_bty_file` writes range and depth only.
`write_bty_long_format` adds five geoacoustic columns per range node — `c_p`,
`c_s`, `ρ`, `α_p`, `α_s` — which is how a range-dependent bottom reaches
Bellhop without collapsing it. `read_bathymetry` returns whichever it finds,
as rows `2:7` of the array when the file's type field says `'L'` in position 2.

**Staging.** Because the auxiliary files are found by base name, a table
produced somewhere else has to be copied next to the `.env` that names it.
`stage_reflection_file(reflection_file, env_path, boundary='bottom')` does that
and returns the destination; a table already sitting at the destination — a
[Bounce](../models/bounce.md) run whose `.brc` is in the same pinned `work_dir`
— is left alone. `stage_source_beam_pattern(pattern, dest)` takes either a path
to copy or an `(N, 2)` array of `[angle_deg, level_dB]` to materialise.

**`dedupe_reflection_file`.** BOUNCE tabulates `R(θ)` by sweeping phase
velocity, which produces many samples that round to the same grazing angle —
hundreds of duplicate 0° rows are typical. Bellhop's Fortran tolerates a
non-decreasing angle axis; `bellhopcuda` enforces strict monotonicity and
aborts. This rewrites the file keeping only strictly-increasing angles. It is
lossy by exactly one sample per genuine collision, which is a slight
under-resolution near grazing and the reason it is a separate, named step
rather than something the writer does silently.

---

## 5. OASES formats

[OASES](../models/oases.md) takes one ASCII input per sub-model and writes a
different binary per output.

| File | Direction | Idiom | Produced by / for |
|---|---|---|---|
| `.dat` | in | ASCII | All four sub-models — `write_oast_input`, `write_oasp_input`, `write_oasr_input`, `write_oasn_input` |
| `.plp` + `.plt` | out | ASCII | OAST transmission loss: `.plp` is the grid metadata, `.plt` the values |
| `.trf` | out | Fortran sequential, or ASCII | OASP broadband transfer function |
| `.xsm` | out | direct-access + `struct` headers | OASN sensor cross-spectral (covariance) matrices |
| `.rpo` | out | Fortran sequential + `struct` headers | OASN signal replicas |
| `.rco` / `.trc` | out | ASCII | OASR reflection coefficients, sampled in slowness (`.rco`) or angle (`.trc`) |

`read_oast_tl(path, receiver_depths)` returns TL on OAST's **native** range
grid, not yours: OAST picks its own range sampling via an FFT, and the `.plp`
metadata file is the only record of it. Resampling onto your receiver grid is
an explicit `Field.resample_to` call, so the interpolation is visible rather
than buried in the reader.

Note the `.trc` collision: OASR can write a `.trc` reflection table, and
Bellhop reads a `.trc` top-boundary table. They are different files with
different readers — `read_oasr_reflection_coefficients` versus
`read_reflection_coefficient`.

---

## 6. RAM family formats

[RAM](../models/ram.md) is a dispatcher over four parabolic-equation binaries,
and they do not share an input format.

### mpiramS

| File | Direction | Written by |
|---|---|---|
| `in.pe` | in | `write_inpe` |
| SSP file | in | `write_ssp_file` |
| bathymetry file | in | `write_bth_file` |
| output-ranges file | in | `write_ranges_file` |
| sediment file | in | `write_sediment_file` (range-dependent seabed) |
| `psif.dat` | out | read by `read_psif` |

`psif.dat` is Fortran sequential unformatted, double precision throughout. Its
header record is eight reals, then a frequency vector, then a range vector,
then — for each of the `nr` ranges — `nzo` depth records of `1 + 2·nf` reals
each: depth, followed by interleaved real/imaginary parts of `ψ` at every
frequency. `read_psif` returns a dict with `psif` of shape `(nzo, nf, nr)` and
`rout` in metres. It is the one reader that takes the **directory** holding
`psif.dat` rather than a path to the file — hand it the `work_dir`.

The header scalars are renamed on the way out — Fortran's `Nsam` and `cmin`
become `n_samples` and `c_min` — so a consumer can forward them straight into
`Result.metadata` without a second mapping table.

### The Collins binaries: rams0.5, ramsurf1.5, ramgeo

| File | Direction | Written by / read by |
|---|---|---|
| `rams.in` / `ram.in` / `ramgeo.in` | in | `write_ramin(..., kind='rams'\|'ramsurf'\|'ramgeo')` |
| `tl.line` | out | `read_tl_line` — ASCII `range TL` at one receiver depth |
| `tl.grid` | out | `read_tl_grid` — Fortran sequential, real TL on the `(z, r)` grid |
| `pcomplex.bin` | out | `read_pcomplex_grid` — **uacpy-patched**, the complex envelope |

`tl.grid` and `pcomplex.bin` have identical record geometry: record 1 is a
single `int32 lz` (stored depth points), records 2..N each hold `lz` samples —
`real*4` for `tl.grid`, `complex*8` for `pcomplex.bin` — one record per output
range step. Both readers take `dr`, `ndr`, `dz`, `ndz` from the input deck to
reconstruct the axes, plus a `depth_index_offset` that differs between binaries:
`ramsurf1.5` writes from grid index `ndz` (offset 0) while `rams0.5` writes from
`1 + ndz` (offset 1) and skips the `z = 0` node.

### `pcomplex.bin` is a local patch

Stock Collins `outpt` writes only real TL, which discards the phase a broadband
transfer function needs. uacpy patches `rams0.5.f`, `ramsurf1.5.f` and
`ramgeo1.5.f` to also dump the complex PE envelope to a parallel
`pcomplex.bin`, mirroring `tl.grid`'s record geometry. The full diffs and the
reasoning are in
[`uacpy/third_party/MODIFICATIONS.md`](../../uacpy/third_party/MODIFICATIONS.md).

Two details from that record matter when you read the file directly:

- **The binaries store different envelopes.** `ramsurf1.5` and `ramgeo1.5`
  write `u·f3 / sqrt(r)`; `rams0.5` writes `u / sqrt(r)` — its `outpt` takes no
  `f3` argument.
- **They differ in the carrier.** `rams0.5`'s march multiplies by
  `g0 = exp(i k₀ Δr)` at every step, so `exp(+i k₀ r)` is baked into its `u` —
  the same convention as mpiramS' `psif`. `ramsurf1.5` absorbs the carrier into
  its operator coefficients instead, so its `u` carries none. The RAM wrapper
  applies a per-backend correction before tagging the result
  `phase_reference='travelling_wave'`, which is what lets every broadband model
  hand the same shape of `H(f)` to the IFFT pipeline — see
  [results](results.md).

MODIFICATIONS.md also records that the complex envelope is bit-exact against
the existing `tl.grid` magnitude: `-20·log10(|pcomplex|)` reproduces `tl.grid`
to 0.0000 dB on a Pekeris reference run. And that a binary built from unpatched
sources does not silently degrade to real TL — every Collins-backend run reads
`pcomplex.bin`, so it fails outright, naming the file and telling you to
rebuild.

---

## 7. `FileManager` — scratch directories, `work_dir` and `cleanup`

Every model run executes its binary in a scratch directory. `FileManager` owns
that directory's lifetime, and the models reach it through two constructor
knobs: `work_dir` and `cleanup`.

```python
uacpy.io.FileManager(use_tmpfs=False, base_dir=None, prefix='uacpy_', cleanup=True)
```

| Method | What it does |
|---|---|
| `create_work_dir()` | Make a fresh uniquely-named directory under `base_dir`. uacpy owns it. |
| `adopt_work_dir(path)` | Use a caller-named directory, taking ownership **only if it did not already exist**. |
| `get_path(name)` | Full path inside the work dir, creating it on demand. |
| `cleanup_work_dir()` | Remove uacpy's scratch — see below. |

It is also a context manager: `with FileManager(use_tmpfs=True) as fm:` creates
on entry and cleans on exit if `cleanup` is set.

### Ownership decides what cleanup removes

A directory uacpy created is removed whole. A directory you named and already
had is not: `adopt_work_dir` snapshots its existing entries, and cleanup removes
only what this run added. That is why `cleanup=True` with `work_dir='.'` cannot
take your tree.

The snapshot approach — rather than tracking `get_path` calls — is deliberate:
the native binaries write files uacpy never names (`tl.grid`, `.prt`, `.shd`),
so only a before/after comparison catches all of them.

Cleanup failures are swallowed. A stale NFS lock or a file handle a subprocess
has not released must never mask the original exception a caller is trying to
surface.

### `use_tmpfs`

`use_tmpfs=True` puts the scratch directory on `/dev/shm`, which is worth it for
the models that write large intermediate grids. It is ignored — with a warning
— when `work_dir` is pinned, because a named directory cannot be relocated to
RAM. Point `work_dir` at a tmpfs mount yourself if you want both.

### The `cleanup` default, and how you tell

| `work_dir` | `cleanup` default | Result |
|---|---|---|
| `None` | `True` | Fresh temp dir, wiped after `run()` |
| pinned | `False` | Your directory, files kept |

Pinning a `work_dir` is how you chain models — [Bounce](../models/bounce.md)
writes a `.brc` that Bellhop then reads — and how you get at the raw solver
output.

**The absence of a `*_file` key in `result.metadata` is the documented signal
that the directory was cleaned.** Paths are attached if and only if the scratch
survives, so a key that exists always points at a file that exists:

```python
kept = uacpy.Bellhop(work_dir='./run').run(env, source, receiver)
sorted(kept.metadata)          # ['prt_file', 'shd_file']

wiped = uacpy.Bellhop().run(env, source, receiver)
sorted(wiped.metadata)         # []
'shd_file' in wiped.metadata   # False
```

Which keys appear depends on the model — only the Acoustics-Toolbox binaries
write a `.prt`, so `prt_file` is on their results and not on RAM's or OASES'.
Rather than assume, call `result.list_metadata()`, which describes every key
currently attached. See [results](results.md).

One constraint worth knowing before you fan runs out with `run_parallel`: a
pinned `work_dir` must be unique per job. Concurrent workers sharing one
directory collide on the models' fixed scratch filenames, and
[`run_parallel`](utilities.md) raises `ConfigurationError` rather than let them.

---

## 8. Errors

Two typed exceptions carry the subpackage's own failures, and the split is by
cause, not by call site:

- **`FileFormatError`** — the file is wrong. Malformed, truncated, the wrong
  format, a record-marker mismatch, a header count the file size cannot
  support.
- **`ConfigurationError`** — the *argument* is wrong. A base directory that
  does not exist or is not writable, an unknown `kind=`, a missing required
  parameter.

Both inherit from `UACPYError`, so one `except` catches the family, and both
carry a `remediation` argument that renders as a *How to fix* line under the
message. See [environment](environment.md) for the wider exception hierarchy.

A path that does not exist is neither: the readers open the file and let the
standard `FileNotFoundError` through, so code guarding a user-supplied path
wants that in the `except` clause too.

**A malformed file arrives as a `FileFormatError`, not a bare `ValueError`.**
File-format readers do `int()` and `float()` on tokens, call `next()` on line
iterators, and index binary records — so a bad file naturally leaks a
`ValueError`, `IndexError`, `KeyError`, `StopIteration`, `EOFError`,
`ZeroDivisionError`, `OverflowError` or `struct.error`. The
`typed_format_error` decorator converts that set, and only that set, into
`FileFormatError`:

```
FileFormatError: read_reflection_coefficient: could not parse junk.brc — the
file is malformed, truncated, or not the expected format (ValueError: invalid
literal for int() with base 10: 'not a number').

How to fix:
Verify the file was produced by the matching model/writer and downloaded
completely; a partial file or a wrong format triggers this.
```

The conversion set is deliberately narrow. `AttributeError`, `TypeError` and
`NameError` are **not** converted: those signal a defect in uacpy, and dressing
them as `FileFormatError` would send you off to debug a file that is fine.

The decorator is applied per reader, and two public ones do not carry it:
`read_rts_file` and `read_tl_line` still surface a bare `ValueError` on
malformed input.

---

## 9. Reference — the whole public surface

Every name in `uacpy.io.__all__`, grouped by format family.

### Plumbing

| Name | |
|---|---|
| `FileManager` | Scratch-directory lifetime; the machinery behind `work_dir` / `cleanup` |
| `equally_spaced(x, tol=1e-9)` | Is this axis uniformly sampled? Decides compact vs explicit axis encoding |

### Acoustics Toolbox — readers

| Name | |
|---|---|
| `read_shd_file` | `.shd` → `Field` (or `ResultStack` for several source depths) |
| `read_shd_bin` | `.shd` binary → raw dict, one frequency slice |
| `read_shd_asc` | ASCII `.shd` → raw dict |
| `read_arr_file` | `.arr` → `Arrivals` |
| `read_ray_file` | `.ray` → `Rays` |
| `read_ssp_2d` | 2-D `.ssp` → depths, ranges (m), `c` matrix |
| `read_ssp_3d` | 3-D `.ssp` for BELLHOP3D |
| `read_flp` | `.flp` field-parameters deck → dict, including the 4-character option word |
| `read_flp3d` | `.flp` for FIELD3D |
| `read_rts_file` | SPARC `.rts` time series → dict |
| `rts_to_pressure` | `.rts` dict + frequency → complex pressure |
| `read_ts` | Generic ASCII `.ts` time series |
| `read_prt` | `.prt` diagnostic log (whole file, or a trailing `tail_bytes`) |
| `read_modes` | `.mod`/`.moa` by extension, plus derived half-space terms |
| `read_modes_bin` | Binary `.mod`; `frequency=` selects the closest entry of a multi-frequency file — the default `0.0` silently gives you the lowest |
| `read_modes_asc` | ASCII `.moa` |
| `get_component` | Pull one stress-displacement component out of a modes dict |
| `read_grn_file` | `.grn` Green's function → dict |
| `grn_to_field` | `.grn` at one frequency → complex `Field` |
| `grn_to_transfer_function` | `.grn` across frequencies → broadband `Field` |
| `sparc_snapshot_to_field` | SPARC snapshot → steady-state `Field` at one frequency |
| `sparc_snapshot_to_time_field` | SPARC snapshot → range-domain time `Field` |

### Acoustics Toolbox — writers

| Name | |
|---|---|
| `write_bellhop_env_file` | Bellhop `.env` (run type, beams, boxes) |
| `write_kraken_env_file` | Kraken `.env` |
| `write_scooter_env_file` | Scooter `.env` |
| `write_sparc_env_file` | SPARC `.env` (pulse, time window, output mode) |
| `write_bounce_input_file` | Bounce `.env` |
| `write_multi_profile_env` | Multi-profile `.env` for range-dependent Kraken |
| `write_header` | Title / frequency / `TopOpt` block |
| `write_absorption_block` | Post-`TopOpt` volume-absorption block |
| `write_fg_params` | Francois–Garrison T/S/pH/depth line |
| `write_bio_layers` | Biological attenuation layers |
| `write_broadband_freqs` | Broadband frequency vector |
| `write_ssp_section` | Water-column SSP block |
| `write_layer_sections` | One SSP block per sediment layer (`NMEDIA > 1`) |
| `write_bottom_section` | Bottom boundary block |
| `writable_layers` | Layers thick enough to be a distinct AT medium |
| `write_source_depths` | Source-depth section |
| `write_receiver_depths` | Receiver-depth section |
| `write_receiver_ranges` | Receiver-range section (m → km) |
| `write_phase_speed_and_rmax` | `cLow`/`cHigh` line and `RMax` in km |
| `write_fieldflp` | `.flp` for FIELD/FIELDS |
| `write_field3dflp` | `.flp` for FIELD3D |
| `write_ssp` | Range-dependent `.ssp` matrix (m → km) |
| `resolve_ssp_interp` | Resolved `interp_ssp` for an env / model pair |
| `resolve_ssp_topopt` | The AT `TopOpt(1)` character it maps to |
| `resolve_phase_speed_bounds` | Effective `(c_low, c_high)` |

### Boundary, reflection and beam-pattern files

| Name | |
|---|---|
| `read_bathymetry` | `.bty` → array (m), interpolation type; long format carries geoacoustics |
| `read_altimetry` | `.ati` → array (m), interpolation type |
| `read_boundary_3d` | 3-D boundary block for BELLHOP3D |
| `read_reflection_coefficient` | `.brc`/`.irc`/`.trc` → `theta` (deg), `R`, `phi` (rad) |
| `read_source_beam_pattern` | `.sbp` → angle / level array |
| `write_bty_file` | `.bty`, short format (range, depth) |
| `write_bty_long_format` | `.bty` with per-range `c_p`, `c_s`, `ρ`, `α_p`, `α_s` |
| `write_bty_3d` | 3-D `.bty` for BELLHOP3D |
| `write_ati_file` | `.ati` altimetry |
| `write_reflection_coefficient` | `.brc`/`.trc` from angles (deg) + complex or `[amp, phase_rad]` |
| `write_source_beam_pattern` | `.sbp` from angles (deg) + dB re peak |
| `stage_reflection_file` | Copy a table to the `<env>.brc`/`.trc` name the binary opens |
| `stage_source_beam_pattern` | Materialise a `.sbp` from a path or an `(N, 2)` array |
| `dedupe_reflection_file` | Rewrite `.brc`/`.irc` with a strictly-increasing angle axis |

### OASES

| Name | |
|---|---|
| `write_oast_input` | OAST `.dat` (transmission loss) |
| `write_oasp_input` | OASP `.dat` (broadband pulse) |
| `write_oasr_input` | OASR `.dat` (reflection coefficients) |
| `write_oasn_input` | OASN `.dat` (noise covariance / replicas) |
| `read_oast_tl` | `.plp` + `.plt` → TL on OAST's native range grid |
| `read_oasp_trf` | `.trf` transfer function |
| `read_oasr_reflection_coefficients` | `.rco`/`.trc` reflection table |
| `read_oasn_covariance` | `.xsm` cross-spectral matrices |
| `read_oasn_replicas` | `.rpo` signal replicas |

### RAM family

| Name | |
|---|---|
| `write_inpe` | mpiramS `in.pe` |
| `write_ssp_file` | mpiramS SSP, 1-D or range-dependent (range axis m → km) |
| `write_bth_file` | mpiramS bathymetry, `range(m) depth(m)` pairs |
| `write_ranges_file` | mpiramS output ranges (m) |
| `write_sediment_file` | mpiramS range-dependent sediment profiles (range axis m → km) |
| `read_psif` | mpiramS `psif.dat` → `psif` of shape `(nzo, nf, nr)`; takes the containing **directory**, not the file |
| `write_ramin` | Collins `ram.in` / `rams.in` / `ramgeo.in` (metres) |
| `read_tl_line` | `tl.line` → ranges (m), TL (dB) |
| `read_tl_grid` | `tl.grid` → ranges, depths, TL grid |
| `read_pcomplex_grid` | uacpy-patched `pcomplex.bin` → complex envelope |

---

## 10. Where to go next

- **[Results](results.md)** — what the readers hand back: `Field`, `Rays`,
  `Modes`, `Arrivals`, and the `metadata` keys the work dir populates.
- **[Environment](environment.md)** — the carriers the writers serialise, and
  the collapse policy that decides what survives into the file.
- **[Utilities](utilities.md)** — `run_parallel`, TL metrics, material presets,
  logging.
- **[Model index](../models/README.md)** — which model writes which format.
- **[`uacpy/third_party/MODIFICATIONS.md`](../../uacpy/third_party/MODIFICATIONS.md)**
  — every local patch to the vendored Fortran, with diffs.
- **[`DOCUMENTATION.md`](../../DOCUMENTATION.md)** — the terse API reference.

---

**See also:** [documentation index](../README.md) · [results](results.md) ·
[environment](environment.md) · [utilities](utilities.md) ·
[model index](../models/README.md)
