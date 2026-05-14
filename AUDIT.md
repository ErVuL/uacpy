# UACPY Audit Report

Audit conducted in parallel across five disjoint slices: (1) core + models,
(2) I/O layer, (3) tests + examples, (4) visualization + signal + noise,
(5) build + packaging + vendored sources. Each slice was given the
architecture context from CLAUDE.md, the user-facing claims in
DOCUMENTATION.md, and the maintainer's own "needs auditing" roadmap from
README.md.

The codebase was LLM-bootstrapped. The maintainer has been explicit that this
is a beta release pending audit. Findings below are organised by **what could
silently corrupt user results** first, then by API contract violations,
maintenance debt, and patterns worth fixing systemically.

---

## TL;DR

The Python wrapper layer, the model contract, and most of the file-format
writers are coherent. The systemic risks cluster in four places:

1. **The Kraken-modes reader is broken end-to-end.** Three separate defects
   in `io/modes_reader.py` — wrong file extension expectation, ASCII path
   references keys that don't exist, broadband halfspace uses the wrong
   frequency index. Public `read_modes()` calls fail or silently return wrong
   numbers.
2. **The build script declares success on partial failure.** `install.sh`
   always `exit 0`; CI passes green even when a native binary fails to build.
   Combined with `requires_binary` test markers, this masks regressions
   permanently.
3. **Supply-chain pins are TODO slots, not values.** OASES SHA256 empty,
   bellhopcuda commit SHA empty, `curl` invocation lacks protocol pinning.
   The README acknowledges this; it must be resolved before tagged release.
4. **Tests are mostly smoke.** `isinstance + isfinite + 0 < tl < 200`
   dominates. The roadmap's stated requirement of Pekeris/Munk canonical
   regressions is only partially met for Pekeris (good) and not at all for
   Munk (the `munk_env` fixture is actually a parabolic SSP). Cross-model
   agreement tolerances are loose enough that a 10 dB miscalibration
   passes.

The 14 critical findings below should be fixed before tagging v1.0. The
~80 high/medium findings are tracked per slice further down.

---

## Critical findings (silently wrong physics / data / build)

The slice tag in `[brackets]` shows which audit surfaced the issue.

### IO-1. `read_modes_bin` requires `.moA`; doubles the extension. [io]

`uacpy/io/modes_reader.py:358-359` — `if not filename.endswith(".moA"): filename = filename + ".moA"`. Kraken/KrakenC emit `.mod`. The
`models/kraken.py:403-404` wrapper papers over this with a `shutil.copy`,
but the public `from uacpy.io import read_modes` path calls `read_modes_bin(filename)`
directly with the user's `.mod`, producing `foo.mod.moA` and `FileNotFoundError`.

### IO-2. `read_modes()` references keys ASCII reader never sets. [io]

`uacpy/io/modes_reader.py:567-589` — post-processing accesses
`Modes["freqVec"]`, `Modes["Top"]["BC"]`, `Modes["Bot"]["BC"]`. `read_modes_asc`
returns none of these. Every ASCII `.moa` file read crashes with `KeyError`.

### IO-3. Broadband halfspace `Top` uses `freqVec[0]` not `freqVec[freq_index]`. [io]

`uacpy/io/modes_reader.py:571` — line 581 (`Bot`) correctly uses the requested
frequency index; line 571 (`Top`) does not. Broadband modes consumed by
matched-field or covariance code carry a wrong-frequency `k²` and `gamma` on
the top halfspace.

### IO-4. Reflection-coefficient phase round-trip silently scales by 180/π. [io]

`uacpy/io/refl_io.py:99` reads file column as degrees, returns `phi` in radians.
`uacpy/io/refl_io.py:242` writes the value verbatim as if it were degrees.
A user who reads then re-writes a `.brc` corrupts the next Bellhop run that
loads the coefficients via top-type `'F'`. Pick one convention and apply
symmetrically.

### IO-5. `read_boundary_3d` returns coordinates in km, not metres. [io]

`uacpy/io/bathy_io.py:122` — the 2-D readers `read_bathymetry`/`read_altimetry`
multiply by 1000 to convert km→m (matching the package convention); the 3-D
reader returns raw file values. The docstring even hedges "typically in km".
Anyone using a 3-D `.bty` is off by 1000× on x/y.

### CORE-1. `result.metadata['*_file']` paths are recorded *before* cleanup runs. [core]

`uacpy/models/base.py:1250-1276` and every concrete model
(`bellhop.py:900`/`913-914`, `bounce.py:425`/`437-438`, similar in
kraken/oases/ram/scooter/sparc) attach paths to `result.metadata` and *then*
the `finally` block calls `cleanup_work_dir()`. DOCUMENTATION.md §2 and §6
explicitly promise the inverse: "absence is the signal that the dir was
wiped." With the current code, `result.metadata.get('shd_file')` returns a
non-None path that opens to `FileNotFoundError`. Fix at the base: clear keys
after cleanup, or check `path.exists()` before assignment.

### BUILD-1. `install.sh` always exits 0 regardless of component failure. [build]

`uacpy/install.sh:1209-1243` — `OVERALL` is computed from `STATUS_*` and only
changes the banner text. CI's `./install.sh -y --bellhop cxx --oases yes`
(`.github/workflows/ci.yml:75`) passes green when OALIB or mpiramS or OASES
fails to build, because pytest then silently skips on `requires_binary` /
`requires_oases`. **Recommendation:** `exit 1` on any `failed`/`partial`
status unless explicit `--allow-partial`.

### BUILD-2. OASES download is unpinned and unprotected. [build]

`uacpy/install.sh:773` — `OASES_EXPECTED_SHA256=""` empty. Line 791 is bare
`curl -fSL` with no `--proto '=https'`, no `--tlsv1.2`, no
`--proto-redir '=https'`. The README already flags the empty SHA. Combine
with the trust-on-first-use CI cache key (`hashFiles('install.sh')`) and the
result is "MIT redirected once during install → poisoned Fortran installed
forever in CI cache, every PR validated against poisoned binaries".

### BUILD-3. Undocumented modification to Acoustics-Toolbox root Makefile. [build]

A `Makefile.orig` sits next to `Makefile` in
`uacpy/third_party/Acoustics-Toolbox/`, evidencing an in-tree edit
(header comment + `-march=native -mtune=native` flag stripping). No entry
in `third_party/MODIFICATIONS.md`. CLAUDE.md explicitly says every
modification needs a documented diff. The change happens to be benign
because `install.sh` re-injects `FORTRAN_ARCH_FLAGS`, but the process
violation undermines the audit trail for downstream users / packagers.

### VIS-1. `compare()` double-inverts depth axis for TL-vs-depth slices. [vis]

`uacpy/visualization/plots.py:542-545` — sequential `if value == 'tl': ax.invert_yaxis()`
and `if common_axis == 'depth': ax.invert_yaxis()` both fire on the natural
overlay case (`compare([f.at(range=5000), …], value='tl')`), unflipping the
axis silently. Depth=0 ends up at the bottom of the plot. Fix:
`should_invert = (value == 'tl') ^ (common_axis == 'depth')`.

### VIS-2. `beamform()` returns SNR mislabeled as "power". [vis]

`uacpy/acoustic_signal/processing.py:174-175` — the function returns
`20·log10(|e @ pressure|) + SL - NL`. The docstring calls this "Beamformed
power in dB"; it's actually receive-level minus noise-level. A user who
adds back a noise term double-counts. Either drop the `- NL` or rename the
return to `SNR_dB`.

### VIS-3. TL metrics ignore axis values. [vis]

`uacpy/core/metrics.py:63-72` — `tl_rmse` / `tl_max_error` / `tl_bias` check
that `field_a.shape == field_b.shape` but never that the underlying
coordinate vectors match. Two fields sampled at `depths=[0,50,100]` vs
`[0,25,75]` return a non-warning RMSE computed across physically different
sample points. `compare_models` has the same defect with `vmin/vmax`.

### VIS-4. `plot_field(projection='polar')` fabricates azimuthal data. [vis]

`uacpy/visualization/plots.py:425-436` — replaces the per-depth field with
`Z.mean(axis=0)` and tiles that 1-D range curve around the full azimuth.
Every direction shows the same depth-averaged TL. DOCUMENTATION.md §7
advertises this as "polar TL view", which a reasonable user reads as a
true bird's-eye plot. Rename to `plot_field_azimuthally_symmetric` or
remove.

### TEST-1. Modal-model agreement test tolerance is 15 dB. [tests]

`uacpy/tests/test_modal_model_agreement.py:155` — the canonical agreement
check between KrakenField / Scooter / OAST on a Pekeris-like env asserts
`max_diff < 15.0 dB` at a single (50 m, 5 km) point. Any of the three could
be silently 10 dB off and this passes. KrakenField vs Scooter on
hard-bottom Pekeris should agree to <1 dB. **Tighten to ≤2 dB.**

### TEST-2. SPARC R-vs-D consistency does not compare numerically. [tests]

`uacpy/tests/test_sparc_output_modes.py:163-173` — the docstring
acknowledges that any tolerant `assert_allclose` would pass for almost
anything, so the body only checks `0 < tl < 200`. SPARC's two output modes
could disagree by 30 dB and this passes. The dedicated regression
mentioned in the comment ("belongs in a benchmark with stored reference
data") does not exist anywhere in `tests/`.

---

## Per-slice findings

### Core + Models

**High**

- `OASES(...)` factory silently filters unknown kwargs via
  `inspect.signature(cls.__init__)` (`models/oases.py:1338`). DOCUMENTATION.md
  §11 explicitly tells users mistypes will raise `TypeError`. A typo like
  `OASES(angels=...)` is dropped silently.
- `Source`, `Receiver`, `Environment`, `BoundaryProperties`, `SoundSpeedProfile`,
  `SedimentLayer`, `Biological*`, `ConstantAbsorption` all raise bare
  `ValueError` / `TypeError`. DOC §5 promises `ConfigurationError` for input
  validation. `core/exceptions.ConfigurationError` exists for this; only
  `models/base.py` and some writers use it. Catch-by-typed-exception silently
  misses every constructor error.
- `Bellhop._run_broadband` (`models/bellhop.py:1034-1102`) has the docstring
  *after* an early `raise ConfigurationError`. `help()` returns `None`; users
  can't see the documented broadband API.
- `Kraken._supports_elastic_media = True` (`models/kraken.py:573`) — real-arith
  `kraken.exe` actually fails on elastic envs. The flag's contract is "does
  this env shape work with this model?", so the projection layer skips the
  collapse and lets a runtime Fortran error happen instead of gracefully
  collapsing. Either flip to `False` (with auto-route to KrakenC like Bellhop
  has for Bounce), or split the capability concept.
- `_compute_broadband_field` (`models/kraken.py:1428-1429`) docstring says shape
  `(n_depths, n_freqs, n_ranges)`. Actual shape is `(n_depths, n_ranges, n_freqs)`
  matching the other broadband-capable wrappers — fix the docstring.

**Medium**

- Three sites hard-code broadband default `np.linspace(fc*0.5, fc*2.0, 64)` or
  `128`: `models/bellhop.py:1175-1177` (has named constants, good),
  `models/scooter.py:302` (magic 64), `models/kraken.py:1433` (magic 64). DOC
  doesn't mention the default. Promote to `core/constants.py`.
- `generate_sea_surface` (`core/environment.py:1326-1389`) uses
  `omega_p = g / wind_speed_ms` for the Pierson–Moskowitz peak frequency.
  Standard P–M is `0.855·g / U_19.5`; uacpy documents `wind_speed_ms` as 10 m
  height. Needs height correction or doc update; `alpha_pm = 8.1e-3`,
  `beta_pm = 0.74` need citations.
- `_pade_optimizer.py:259` — default `theta_max = np.deg2rad(30.0)` but the
  docstring says "Default 30°". Internal use is radians. Inconsistent with
  the call site at `ram.py:236` which passes degrees.
- `_clip_receiver_depths` silently mutates the receiver to drop samples
  near the seafloor; warning is gated by `self.verbose`, so default users
  see receiver_count change without explanation
  (`models/base.py:1331-1371`).
- `Bounce._run_*` uses `hasattr(source.frequencies, '__len__')` defensively
  (`models/bounce.py:409, 467`) — `Source.__init__` already guarantees
  ndarray via `np.atleast_1d`. Dead branch.
- `OASP._run` adjusts `n_time_samples` using `df_user = freqs[0]` as a
  fallback when only one frequency is passed (`models/oases.py:1129-1142`).
  That's not `df`, it's the frequency itself. Silent grid shift.

**Low**

- `_CANONICAL_AXIS_ORDER` defined at `core/results.py:203`, never read.
- `_stack.py:14-17` ternary is a no-op (`hard if hard == RLIM_INFINITY else hard`).
- `UACPYError.__init__` type annotation `remediation: str = None` should be
  `Optional[str]`.

### I/O layer

**Critical** (already in top list): IO-1, IO-2, IO-3, IO-4, IO-5.

**High**

- `write_field3dflp` (`oalib_writer.py:1112-1152`) does **not** convert m→km
  for receiver/source x/y, while its 2-D sibling `write_fieldflp` (line 922)
  does. The 3-D writer's docstring says "Sx in km, Rr in km" so callers are
  forced to pre-convert — silently asymmetric from the rest of the writer
  surface.
- `read_flp3d` (`oalib_reader.py:1163`) converts `r_rcv` (km→m) but not
  `r_offsets`. The 2-D reader at line 1056 also doesn't convert `r_offsets`.
  Consumers downstream assume metres throughout.
- `oases_reader.read_oast_tl` silently edge-pads when the binary file is
  shorter than expected (`oases_reader.py:138-145`). A truncated OAST run
  produces a TL field that looks plausible at far range but is fabricated.
  Should raise unless truncation is at most one record.
- `read_oasn_covariance` hard-codes 8-byte records and little-endian
  (`oases_reader.py:275-321`). Big-endian or 2-word OASN builds silently
  return garbage covariances.
- `_fortran_helpers.read_fortran_record_marker` (`_fortran_helpers.py:15-24`)
  hard-codes native `'i'` byte order while every other helper accepts an
  `endian=` parameter. `ramsurf_reader._read_lz_records` (`ramsurf_reader.py:62, 82, 88, 90`)
  is similarly hard-coded little-endian. RAM cross-compiled to big-endian
  decodes silently wrong.
- `bellhop_writer.py:208` flips altimetry sign in the writer; `read_altimetry`
  returns the file column verbatim. Round-trip silently inverts.
- `write_ssp` (`oalib_writer.py:178-183`) writes sound speeds at `%6.1f`
  (1-decimal precision). Munk profiles silently round to 1502.3 m/s
  everywhere.
- `write_multi_profile_env` docstring claims segments are `(range_km, Environment)`
  but the only caller (`models/kraken.py:1193`) passes metres. The value is
  unused so the bug is silent, but anyone constructing segments manually by
  the doc is off by 1000×.

**Medium**

- `read_shd_bin` (`oalib_reader.py:175-274`) hard-codes single-x/single-y
  source in the indexing arithmetic. Multi-x `.shd` files silently return
  only the first source.
- `merge_shd_files` (`shd_utils.py:11-194`) saves as MATLAB `.mat` format
  but names the output `.shd`. Anyone calling `read_shd_bin` on the merged
  output gets a binary-format error. Also, the receiver-position consistency
  check is commented out (line 134-138) — two `.shd` files with different
  receiver grids merge anyway.
- File handle leak: `read_shd_asc` opens via `fid = open(filepath)` without
  a context manager (`oalib_reader.py:328`).
- `equally_spaced` tolerance is absolute `1e-9` independent of axis
  magnitude — too tight for km-scale ranges, too loose for cm-scale
  features.
- Title/option strings in `write_fieldflp` / `write_field3dflp` are written
  unquoted/unsanitized (`oalib_writer.py:940, 943, 1105, 1106`). `Environment`
  itself sanitizes via `_sanitize_title`, but these writers bypass that.
- Dead/unused public exports in `io/__init__.py`: `merge_shd_files`,
  `read_modes`, `read_modes_asc`, `read_shd_asc`, `read_ssp_3d`,
  `read_boundary_3d`, `read_flp3d`, `write_field3dflp`, `write_bty_3d`,
  `read_ts`, `read_source_beam_pattern`. None used by `models/`. Several
  broken (the modes ones — see IO-1/2/3).
- `read_vector` (`_fortran_helpers.py:155-174`) silently linspaces when an
  explicit list has exactly `Nx-1` values (one short of Nx). Should raise.
- `refl_io.py:92-96` fallback "old format" branch reads malformed lines but
  copies the previous index forward — all-constants past the first triple,
  no warning.

### Tests + examples

**Critical** (top list): TEST-1, TEST-2, plus:

- `test_physical_sanity.py:332` — the only Munk test asserts
  `tl_mid < tl_surface + 20`. Asserts nothing physical. The roadmap calls
  out Munk regressions explicitly.
- `test_examples_integration.py:91` — asserts only `returncode == 0`. With
  25 examples, this catches "import broke" but no numerical correctness,
  no PNG-finite check.

**High**

- Six test files re-define `simple_env`, `elastic_env`, `pekeris_env`,
  `receiver_grid` locally instead of using `conftest.py` (test_bellhop.py,
  test_kraken.py, test_physical_sanity.py, test_oases_comprehensive.py,
  test_volume_attenuation.py, test_elastic_boundaries.py — the last
  redefines `elastic_env` four times in four classes with different values:
  sound_speed 1600 vs 1700, density 1.5–1.9).
- The `munk_env` fixture in `conftest.py:50` is actually a parabolic SSP
  (`c_axis*(1 + 0.00737*((depths-axis_depth)/axis_depth)**2)`), missing
  the `η - 1 + exp(-η)` Munk term. Rename to `parabolic_ssp_env` and add a
  real `munk_env`.
- `_OASES_STEMS` in `test_examples_integration.py:30-36` lists
  `example_02_sound_speed_profiles.py`, which doesn't import any OASES
  model. Under `-m "not requires_oases"` example_02 is skipped despite not
  needing OASES.
- `test_cross_model_broadband.py:73` calls private API
  `Bellhop(verbose=False)._run_broadband(...)`. Tests shouldn't reach into
  private methods; either expose via public `run(run_mode=RunMode.BROADBAND)`
  or move the test next to the implementation.
- Examples 10 (signal processing), 25 (canonical presets), and 09 (ambient
  noise) are marked `requires_binary` in `test_examples_integration.py:54`
  but don't run any binary. They're skipped on default CI.

**Medium**

- `test_models.py` is overwhelmingly `isinstance + shape + isfinite + 0<tl`
  smoke. `test_oases_comprehensive.py:67-95` likewise. No cross-model
  agreement check for OAST/OASN/OASR despite the modal-family family
  agreement being a low-hanging benchmark.
- `test_visualization.py` is 100% smoke. Acceptable for visualization, but
  worth documenting explicitly.
- `test_physical_sanity.py:189-190` allows `mean_diff < 5 dB`, `max_diff < 10 dB`
  between Bellhop and Kraken on Pekeris. The tighter regression at
  `test_cross_model_agreement.py::_pekeris_fluid` caps the same comparison at
  6 dB RMSE — duplicate but weaker; delete.
- `test_elastic_boundaries.py:111` asserts `mean_diff > 0.5 dB` between
  elastic and fluid TL. A model wired backwards (ignoring shear) would
  fail by `=0`, but a 0.6 dB shift would pass while masking a 30 dB shear
  coupling bug.

**Coverage gaps** (no test at all):

- `BellhopCUDA` × `{RAYS, EIGENRAYS, ARRIVALS}`
- `Bellhop` × 3-D environment
- `KrakenC` × layered elastic bottom
- `Scooter` × `BROADBAND` × elastic bottom
- `OAST|OASN|OASR|OASP` × numerical agreement vs Scooter
- `RAM(mpiramS)` × `TIME_SERIES` × envelope-shape validation
- Bellhop × altimetry on its own (currently only exercised via
  `_altimetry_consistency` which is `slow`-tagged and ramsurf-guarded)
- `.arr` write→read round-trip, `.mod` round-trip, elastic RD-bottom `.bty`
  round-trip

### Visualization + signal + noise

**Critical** (top list): VIS-1, VIS-2, VIS-3, VIS-4.

**High**

- `cw` and `sweep` advertised in `DOCUMENTATION.md:1132` ("Waveforms (CW,
  chirps, …) | tone_burst, lfm_chirp, hfm_chirp, **cw**, **sweep**, …") and
  in §8 ("chirps return (signal, time); cw/sweep return signal only") — but
  neither exists in `acoustic_signal/generation.py` or
  `acoustic_signal/__init__.py`. Doc lies or code is missing.
- `third_party/arlpy/NOTICE:14-18` claims `core/acoustics.py` contains
  arlpy-adapted `absorption` (Francois–Garrison) and `absorption_filter`.
  Neither function exists in `core/acoustics.py`. F–G logic lives in
  `core/absorption.py` under a different class hierarchy with no arlpy
  header.
- `lfm_chirp` variable name `f_inst` (`generation.py:481`) is actually the
  time-averaged frequency, not the instantaneous frequency. Phase polynomial
  works out correctly, but the name misleads.
- `_auto_tl_limits` (`plots.py:131-139`) uses `median + 0.75·std` rounded to
  10 dB, with no documented inspection or override hook. For
  near-source-dominated fields this clips the far-field to a single colour.
- `compare_models()` shares `vmin/vmax` across panels but never re-samples
  to a common grid. Different-grid fields render correctly individually,
  shared colourbar is meaningless.
- DOCUMENTATION.md §7 (line 1068) claims "The uacpy rcParams (grid, fonts,
  colours) are applied automatically on import." Module docstring at
  `visualization/__init__.py:19-20` says the opposite. Code matches the
  module docstring (no auto-apply).

**Medium**

- `apply_professional_style()` (`visualization/style.py:46-128`) mutates
  `mpl.rcParams` globally with no `rc_context` wrapper or scoped variant.
  Documented as opt-in, but persists across the process if called inside a
  script.
- `analysis.py:144, 257` hard-codes `cmap="jet"` for `PPSD.plot()` and
  `Spectrogram.plot()`. No user override path. `jet` has been deprecated
  perceptually for a decade.
- `make_bandlimited_noise` (`processing.py:271-339`) is filtered Gaussian
  noise normalized to unit RMS post-filter — realized in-band PSD is higher
  than requested by `BW_nominal / BW_effective`. ~1-2 dB error for narrow
  bands. Document or replace with FFT-domain band-limited noise.
- `fourier_synthesis` IFFT (`processing.py:437-441`) uses 2× scaling
  assuming Hermitian symmetry around DC; warning fires only when `Tstart == 0`,
  not when `freq_vec[0] > 0`. Pad spectrum below `freq_vec[0]` with zeros or
  document the DC-anchored requirement.
- `cans`, `nwave`, `mseq`, `make_mseq_probe`, `make_noise_waveform` in
  `generation.py` are unimported and not exposed in `__init__.py`. Partial
  duplicates of public helpers. Delete or promote.
- `sample_rate` vs `fs` naming split half-and-half. `tone_burst`, `lfm_chirp`,
  `hfm_chirp`, `add_noise`, `make_bandlimited_noise` use `sample_rate=`.
  `ssrp`, `bpsk_modulate`, every `analysis.py` class uses `fs=`. CLAUDE.md
  says `fs` is universal. Pick one, deprecate the other.

**Low**

- `noise.py:251-253` shipping `<=0 → -inf` is a heuristic mask; cleaner to
  declare an explicit valid frequency range and warn outside.
- `noise.py:142` `_SHIPPING_C2['no'] = 4` is dead (line 250 forces `-inf` for
  `'no'`).
- `analysis.py:938, 974` add `np.finfo(float).eps` to `np.fft.rfft(x)` —
  cosmetic regularization that does nothing useful.

### Build, packaging, vendored sources

**Critical** (top list): BUILD-1, BUILD-2, BUILD-3.

**High**

- `pyproject.toml:80` `addopts = "-n logical --dist=worksteal"` is
  unconditional, but `pytest-xdist` is in the `test` extra (line 50) not
  base. `pip install -e .` followed by `pytest uacpy/tests/` errors with
  "unrecognized arguments: -n". Either base-install xdist or move xdist
  flags out of `addopts`.
- `install.sh:1159` smoke-tests binaries via `"$exe" --version`, but
  Acoustics-Toolbox binaries treat that as a Bellhop env-file root and
  emit `--version.prt` next to the binary. Pollutes the source tree on
  every install. The presence of `uacpy/third_party/ramsurf/--version.prt`
  confirms this.
- README claims `tar` and `git` are "always" required
  (`README.md:80-88`), but `check_tar` is invoked only inside the OASES
  download branch (`install.sh:783-784`) and `check_git` only under
  `--bellhop cxx|cuda` (line 594). Either docs lie or checks need
  hoisting.
- `install.sh --force --oases yes` does not run `make clean` on the OASES
  tree (the OASES block lacks the `make clean` that OALIB / bellhopcuda /
  mpiramS / ramsurf all do under `$FORCE`). README line 154 advertises
  `--force` as "full clean rebuild of every selected component".
- `bellhopcuda` submodule pinned to mutable tag `v1.5`. `install.sh:56`
  `BELLHOPCUDA_COMMIT_SHA=""` empty. Same trust-on-first-use problem as
  OASES SHA, called out in the CI workflow comment but not fixed.

**Medium**

- `install.sh:1145` `INSTALLED_COUNT` only counts OALIB + bellhopcxx/cuda;
  mpiramS, ramsurf, OASES tracked separately and don't contribute to the
  "did anything install?" gate.
- mpiramS Makefile drifted from MODIFICATIONS.md. Documented at
  `MODIFICATIONS.md:184-188` as
  `FFLAGS = -Ofast -march=native -fopenmp -I $(MODDIR) -Wall -fuse-linker-plugin`,
  actual `third_party/mpiramS/Makefile:34-35` is
  `FFLAGS ?= -Ofast -fopenmp -I $(MODDIR) -Wall -fuse-linker-plugin` (no
  `-march=`, `?=` for overrideability).
- macOS path in `install.sh:282-291` silently disables OpenMP if libomp is
  absent, setting `ENABLE_OPENMP=0`. Fair fallback but not announced loudly.

**Low / housekeeping**

- `.gitignore` doesn't catch `uacpy/third_party/ramsurf/rams0.5`,
  `ramsurf1.5`, or `uacpy/third_party/mpiramS/s_mpiram` (no extension,
  trailing `.5` is digit). `*.exe` rule catches OALIB Linux binaries by
  coincidence.
- Leftovers in vendored tree:
  `uacpy/third_party/Acoustics-Toolbox/Makefile.orig`, `Makefile.local`
  (unused stale auto-generated file),
  `uacpy/third_party/Acoustics-Toolbox/Bellhop/influence3D copy.f90`,
  `influence3D copy 2.f90`, `influence3D copy 3.f90`,
  `Matlab/Bellhop/ssp copy.m` (macOS Finder duplicates),
  `uacpy/third_party/mpiramS/recl.dat` (pre-rewrite build artifact,
  MODIFICATIONS.md:514 says it's no longer needed),
  `uacpy/third_party/ramsurf/--version.prt` (from BUILD-1159 above).
- CI cache key (`ci.yml:68`) is `hashFiles('install.sh')`. Once
  `OASES_EXPECTED_SHA256` is set this becomes correct; until then, cache
  invalidates on unrelated comment edits and never invalidates on
  unrelated upstream URL changes.

**MODIFICATIONS.md ↔ source reconciliation:**

| Documented modification | Status |
|---|---|
| `KrakenField/field.f90` rProf sentinel | ✓ Present and matches |
| `mpiramS/src/kinds.f90` precision | ✓ |
| `mpiramS/src/matrc.f90` NaN-safe init | ✓ |
| `mpiramS/src/solvetri.f90` NaN-safe init | ✓ |
| `mpiramS/src/envdata.f90` sediment vars | ✓ |
| `mpiramS/src/ram.f90` `rnow=0.0_wp` | ✓ |
| `mpiramS/src/peramx.f90` `c0_user` + psif | ✓ |
| `ramsurf/rams0.5.f` outpt + pcomplex.bin | ✓ |
| `ramsurf/ramsurf1.5.f` outpt + pcomplex.bin | ✓ |
| `ramsurf/Makefile` minimal gfortran wrapper | ✓ |
| `mpiramS/Makefile` FFLAGS | ✗ Drifted — see medium item above |
| Acoustics-Toolbox root `Makefile` | ✗ Undocumented — see BUILD-3 |

---

## Cross-cutting patterns

### Unit conversion is owned per-writer/reader, not centrally

`km↔m` happens in `write_bty_file` (m→km), `write_ati_file` (m→km),
`write_receiver_ranges` (m→km), `write_field3dflp` (no conversion despite
its 2-D sibling doing it), `read_bathymetry` (km→m), `read_altimetry`
(km→m), `read_boundary_3d` (no conversion), `read_flp` (km→m for `r_rcv`
and `r_prof`, no conversion for `r_offsets`). One missed `* 1000.0` is
silent and propagates to every downstream consumer. **Fix:** single helper
`def km_to_m(x): return np.asarray(x) * 1000.0` and apply uniformly.

### Endianness is silently assumed little-endian

`_fortran_helpers.read_fortran_record_marker`,
`ramsurf_reader._read_lz_records`, `oalib_reader.read_shd_bin`,
`oases_reader.read_oasn_covariance` all hard-code little-endian. The
codebase has no big-endian story. Either detect on first record-marker
read or accept an `endian=` parameter through every public reader.

### Typed exceptions used inconsistently

`core/exceptions.ConfigurationError` is the documented "bad input"
exception in DOC §5. It's used by `models/base.py` and a few writers, not
by `Source`, `Receiver`, `Environment`, or any of the boundary/SSP
dataclasses. Bare `ValueError` in the core dataclasses defeats every user
who catches `uacpy.ConfigurationError`.

### Field/grid alignment is unenforced

Three sites compare multiple `Field`s cell-by-cell without checking
coordinate vectors agree: `core/metrics.tl_*`,
`visualization.compare`, `visualization.compare_models`. All three should
share a `core.results._assert_aligned(*fields, axes=('depth','range'))`
helper.

### Smoke-as-validation dominates the test suite

The dominant assertion shape is
`isinstance(result, X) and np.all(np.isfinite(result.data)) and (0 < tl).all() and (tl < 200).all()`.
This catches "model crashed, returned NaN, returned obviously-wrong
magnitudes" — nothing else. Pekeris pin-points are good
(`test_cross_model_agreement.py`, `test_ram_backends.py::TestRamPekerisReference`)
but use KrakenField as the reference, which is circular outside the
analytically-tractable Pekeris case. **Action:** ship
`tests/reference_data/*.npz` with stored references, add an
`assert_field_matches_reference(field, ref, rtol=...)` helper, and
populate at least Munk + a layered-bottom case.

### Supply-chain pin slots are TODOs

Both `OASES_EXPECTED_SHA256` and `BELLHOPCUDA_COMMIT_SHA` are placeholder
empty strings with comments saying "paste after first verified install".
Every week these stay empty is a week of trust-on-first-use exposure for
downstream installers and CI. Fix once.

### Documentation drift is detectable mechanically

Three independent doc-vs-code mismatches in the visualization slice alone
(auto-applied rcParams; `cw`/`sweep` exports; NOTICE adaptation list).
The Kraken `_compute_broadband_field` axis order in the docstring is wrong.
DOC §11 claims OASES factory raises `TypeError`; code silently drops.
A `make docs-check` target that diffs DOCUMENTATION.md against
`__all__`-equivalent symbol tables would catch most of these mechanically.

### Capability flags have three different "handled elsewhere" semantics

`Kraken._supports_elastic_media = True` because PRT-error redirection
exists. `RAM._supports_*` all `True` because the dispatcher routes at
`run()` time. `Bellhop._supports_*` flags mean what they say (with
`auto_bounce` as an explicit separate flag). Three different contracts,
indistinguishable from the flag values. Either add a second flag
(`_handles_via_dispatcher` / `_handles_via_routing`), or move
PRT-aware error handling into the projection layer so the flag stays the
literal contract `models/base.py:189-205` describes.

---

## Recommended priority order

**Before v1.0 (critical-path):**

1. Fix `io/modes_reader.py` (IO-1, IO-2, IO-3). The public modes-reading
   path is broken end-to-end today.
2. Fix `io/refl_io.py` phase round-trip (IO-4). Silent data corruption.
3. Fix `io/bathy_io.py:122` 3-D bathy km↔m (IO-5).
4. Make `install.sh` exit non-zero on partial failure (BUILD-1). This
   is one if-statement; it unlocks every CI signal currently masked.
5. Pin `OASES_EXPECTED_SHA256` and `BELLHOPCUDA_COMMIT_SHA` (BUILD-2).
   Add `--proto '=https' --tlsv1.2` to the OASES curl.
6. Document the in-tree Acoustics-Toolbox Makefile diff in
   `MODIFICATIONS.md` (BUILD-3) and reconcile the mpiramS Makefile drift.
7. Fix `result.metadata['*_file']` paths after `cleanup_work_dir()`
   (CORE-1). Either clear them or check existence.

**Before tagging release:**

8. Replace bare `ValueError` in `core/source.py`, `core/receiver.py`,
   `core/environment.py`, `core/absorption.py` constructors with
   `ConfigurationError`.
9. Tighten `test_modal_model_agreement.py:155` from 15 dB to ≤2 dB
   (TEST-1). Tighten `test_physical_sanity.py:332` Munk assertion
   meaningfully. Either restore SPARC R-vs-D numerical comparison or
   delete the test as misleading (TEST-2).
10. Rename `munk_env` fixture to `parabolic_ssp_env` and add a real Munk
    fixture and a Munk canonical regression with stored reference.
11. Fix `visualization.compare()` double-inversion (VIS-1) and
    `compare_models()` / `metrics.tl_*` grid-alignment checks (VIS-3).
12. Either implement or remove `visualization.plot_field(projection='polar')`
    (VIS-4).
13. Fix `beamform()` documentation/return semantics (VIS-2).
14. Add Pekeris/Munk numerical regression suite with stored references
    (roadmap item, currently the dominant test-suite weakness).

**Before next minor release (housekeeping):**

15. Sweep dead code from `io/__init__.py`: either back the unused public
    exports with tests or remove (`merge_shd_files`, `read_modes_asc`,
    `read_shd_asc`, `read_ssp_3d`, `read_boundary_3d`, `read_flp3d`,
    `write_field3dflp`, `write_bty_3d`, `read_ts`,
    `read_source_beam_pattern`).
16. Sweep dead code from `acoustic_signal/generation.py`: `cans`, `nwave`,
    `mseq`, `make_mseq_probe`, `make_noise_waveform`.
17. Unify `sample_rate=` vs `fs=` across signal/noise/analysis.
18. Centralise `km_to_m` / `m_to_km` helpers in `io/utils.py` and
    refactor every writer/reader to use them.
19. Centralise endianness handling: one `endian` parameter through every
    public reader, with auto-detect from the first record marker.
20. Sweep redundant fixtures in `tests/`; rebuild `_OASES_STEMS` and
    `_NO_BINARY_STEMS` by static grep rather than hand-maintained lists.
21. Strip remaining vendored leftovers (`Makefile.orig`, `Makefile.local`,
    `*copy*.f90`, `recl.dat`, `--version.prt`) and tighten `.gitignore`.

---

## Audit method

Five general-purpose agents ran in parallel against disjoint file sets,
each instructed to:

- read CLAUDE.md and DOCUMENTATION.md for context;
- read every file in its slice;
- categorise findings as Critical / High / Medium / Low with file:line
  citations;
- not edit anything;
- return a single-file report capped at ~2500 words.

Findings were deduplicated where slices overlapped (mostly at the
core↔io and visualization↔core/metrics boundaries) and re-prioritised
into the consolidated lists above. Per-slice agent reports are not
shipped as separate files — their full content has been folded into the
"Per-slice findings" section.
