# Reproducibility — what comes back the same, and what you must pin

> `PYTHONHASHSEED` · `OPENBLAS_NUM_THREADS` · `rng=` · `backend=` ·
> `realization=`

Two people need this page. One is writing a methods section and has to say
what a number depends on. The other has been handed a colleague's script and
wants the colleague's figure back, not a near-miss.

The answer is not one word. uacpy is a Python analysis layer over vendored
Fortran, C and CUDA solvers, and those three layers fail differently: the
Python layer is deterministic outright, the solvers are deterministic once you
have chosen which binary runs, and the linear algebra underneath both moves in
its last bits with the thread count. What follows is what has been **measured**
on this package, section by section, with the limits of each measurement
stated alongside it.

Everything below is a repeat-run claim on **one machine, one build, one
NumPy**. [§6](#6-what-has-not-been-measured) says what that excludes.

---

## 1. The short answer

| You want | Pin | You get |
|---|---|---|
| The same number twice from an analysis call | nothing | byte-identical |
| The same noise, fading or BER twice | `rng=default_rng(seed)` | byte-identical |
| The same field from a model twice | `work_dir=`, and `backend='fortran'` for Bellhop | byte-identical |
| The same *bits* out of a beamformer or MFP surface | `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` | byte-identical |
| The same *decision* out of a detector | nothing | unchanged either way |

The last row is the one worth internalising. Thread count and backend choice
move the last few bits of a float; across everything measured they have never
moved a peak, a bearing or a detection. Bit-identity and correctness are
different problems, and only the first of them needs a pinned environment.

---

## 2. The analysis layer is deterministic

Call any of uacpy's analysis entry points twice with the same arguments and
the second answer is byte-identical to the first. Measured over 58 public
entry points, and the identity survives every ambient variable that was
varied: separate processes, different `PYTHONHASHSEED` values, different
locales, different timezones, and 1,014 concurrent invocations dispatched into
an 8-thread pool.

Three properties hold that up.

**No library function draws from a global random state.** Not Python's
`random` module, not NumPy's legacy process-global `RandomState`. Randomness
enters uacpy through an explicit `rng` or `seed` argument and nowhere else, so
there is no hidden stream a caller cannot see or seed. This is gated in the
suite by a source scan, not by a sampling test —
`uacpy/tests/test_determinism.py` walks every shipped module, example and
figure script and fails on `np.random.<draw>`, `import random`, or a
non-generator import from `numpy.random`.

**Exactly ten public functions draw at all**, and every one of them takes a
seeding argument. **Nine of the ten draw afresh unless you seed them** — those
are the ones a reproducible run has to pin:

| Module | Functions | Seeded by | Default |
|---|---|---|---|
| `uacpy.acoustic_signal` | `add_noise`, `make_bandlimited_noise`, `make_noise_waveform`, `synthesize_noise_from_psd` | `rng=` | unseeded |
| `uacpy.comms` | `awgn`, `fading_taps`, `simulate_link`, `ber_sweep` | `rng=` | unseeded |
| `uacpy` | `generate_sea_surface` | `seed=` | unseeded |
| `uacpy.comms` | `schmidl_cox_preamble` | `seed=` | **fixed seed** |

`schmidl_cox_preamble` is the tenth and the exception: its `seed=` defaults to a
fixed integer, not `None`, so two unseeded calls return the same preamble and it
needs no action from you. It is listed because the point of this table is that
it is exhaustive — a drawing function absent from it is a stream you cannot know
to look for.

`generate_sea_surface` builds a Pierson-Moskowitz surface realisation and takes
an integer `seed=` rather than a generator. The data layer reaches it through
`fetch_sea_surface(seed=…)` and `fetch_environment(sea_surface_seed=…)`, both
defaulting to `None`; a plain `fetch_environment` call draws nothing, because
its `altimetry_sources` defaults to `None` too.

Each of the nine returns identical output for a given seed or seeded generator,
and a fresh draw when the seeding argument is omitted — the right default for a
Monte-Carlo sweep, and the wrong one for a figure you intend to publish:

```python
import numpy as np
import uacpy.comms as comms

rng = np.random.default_rng(12345)
link = comms.simulate_link('qpsk', ebn0_db=12.0, n_bits=20000, rng=rng)
```

**Every set-to-output path is sorted.** Python's set iteration order depends
on the hash seed, so a set converted straight to a list is a run-to-run
difference with no cause a reader can see. No `list()`, `tuple()` or `join()`
in the package takes an unsorted set: 72 such conversions across 24 modules go
through `sorted()`, and none goes without it. That is what makes the
`PYTHONHASHSEED` result above a property of the code rather than a lucky
measurement, and it is gated the same way — by an AST scan over the package,
in `test_determinism.py`.

---

## 3. The models, one row each

Every model below was run twice into a pinned `work_dir` and the numeric
output compared byte for byte.

| Model | Repeat run | Carve-out |
|---|---|---|
| `Bellhop()` (default backend) | ✗ **not** bit-reproducible | auto-picks a multithreaded backend — [§4](#4-the-two-things-that-move) |
| `Bellhop(backend='fortran')` | ✅ `.shd` byte-identical | `.prt` is not — CPU-time stamp |
| `Kraken` (`kraken`, `krakenc`) | ✅ | `.prt` is not — CPU-time stamp |
| `Scooter` | ✅ | `.prt` is not — CPU-time stamp |
| `SPARC` | ✅ | `.prt` is not — CPU-time stamp |
| `Bounce` | ✅ every output file, `.brc` / `.irc` / `.prt` | — |
| `RAM` (`mpiramS`, `ramgeo`, `rams0.5`, `ramsurf1.5`) | ✅ all four backends | — |
| `OAST`, `OASP`, `OASR`, `OASN`, `OASS` | ✅ | — |
| `OASSP` | ✅ at a fixed `realization=` | different `realization` values are different draws, by design |

**Exactly four binaries stamp a clock into their listing**, and that stamp is
the only thing anywhere above that moves on a repeat run: Kraken, Scooter,
SPARC and Bellhop's fortran backend each write a `CPU Time = …` line into their
own `.prt` diagnostics, so that one file is not byte-identical. **No numeric
output depends on it.** Everything else is byte-identical throughout — Bounce
is an Acoustics Toolbox model but writes no stamp, so its `.prt` matches along
with its `.brc` and `.irc`, and RAM and the six OASES models write no such file
at all. If you are hashing a run, hash the file you depend on rather than the
whole work directory.

**`realization=` is the seed.** OASSP's rough-surface draw was traced end to
end from the constructor argument to the Fortran seed it sets, with no clock,
PID, file count or directory-iteration dependence anywhere on that path. Two
runs at `realization=0` agree bit for bit; `realization=1` is a different
surface because you asked for one.

**Decks are stable too.** The input files uacpy writes for the solvers are
byte-identical across 15 configurations — separate processes, `PYTHONHASHSEED`
values, locales, timezones and work-directory paths. So a deck diff between
two runs means a real input difference, never formatting drift; it is a usable
first check when two results disagree.

**Twice in one process is fine.** Calling the same model twice inside one
interpreter, into one pinned `work_dir`, returns byte-identical results.
Measured on five models, chosen for where module-level state accumulates.

---

## 4. The two things that move

Neither is a bug in uacpy, and neither has ever changed a decision. Both are
documented in full where they bite, and are summarised here only far enough to
tell you whether to go and read them.

**The default Bellhop backend.** `Bellhop()` with no `backend=` auto-picks
cuda → cxx → fortran, silently, and the multithreaded backends accumulate beam
contributions in completion order. Two default runs of the same environment
wrote a byte-identical `model.env` but a differing `model.shd`, by up to
1.53e-05 dB — about one ULP of the complex64 field. Pass
`Bellhop(backend='fortran')` for a field you can hash. The full measurement,
including which files differ in which direction, is in the Bellhop page's
[§7 Gotchas](../models/bellhop.md#7-gotchas).

**The BLAS thread count.** Beamforming, MVDR and MUSIC all end in a large
matrix product — as does anything else built on one — and OpenBLAS partitions
such a product across threads, so its summation order follows
`OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS`. Spectra move by 1e-16 to 1e-13
relative. Nothing discrete
moved: peak bearings were identical at both problem sizes tested, and the
JANUS detector returned an identical detection index in 750 of 750 real
detections at one thread versus eight. The array-processing page's
[§10 Gotchas](arrays.md#10-gotchas) has the measurement, the ULP-perturbation
check behind the detector claim, and the reason a laptop and a cluster node
disagree by default.

If you need bits, pin both variables before NumPy is imported:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

---

## 5. A checklist for a reproducible run

1. **Seed every draw.** `rng=np.random.default_rng(seed)` on any of the nine
   unseeded-by-default functions in
   [§2](#2-the-analysis-layer-is-deterministic) — `seed=` on
   `generate_sea_surface`, or `sea_surface_seed=` when you reach it through
   `fetch_environment` — and record the seed next to the result.
   `schmidl_cox_preamble`, the tenth, already carries a fixed seed.
2. **Pin the Bellhop backend** — `Bellhop(backend='fortran')` — if a Bellhop
   field is in the chain. Check `result.backend` to confirm what ran.
3. **Pin `realization=`** on OASSP.
4. **Pin the BLAS threads** in the environment, before the interpreter starts.
5. **Pin `work_dir=`** and keep the deck (`cleanup=False`) if you may need to
   show what was actually solved.
6. **Hash the payload, not the directory** — the `.prt` listing carries a
   timestamp on the Acoustics Toolbox models.
7. **Record the machine and the NumPy version.** Nothing above was measured
   across either.

---

## 6. What has *not* been measured

A guarantee whose limits are unstated will be read as broader than it is.
These are open, not known-bad — nobody has looked:

- **`run_parallel` across worker counts.** Repeat runs at a *fixed* worker
  count are covered by the per-model rows above; whether a batch reduces
  identically at 2 workers and at 8 has not been tested.
- **Environments richer than the small test scenarios.** The model rows were
  measured on compact cases. Range-dependent bathymetry, layered elastic
  seabeds and broadband sweeps exercise much more solver state, and no repeat
  measurement covers them.
- **Cross-machine and cross-NumPy reproducibility.** Every result on this page
  is same-machine, same-build, same-NumPy. A different CPU, a different BLAS
  build or a different NumPy release can move the last bits of anything here,
  including the rows marked ✅.

Treat those three as unpinned. If your workflow depends on one, measure it
before you claim it.

---

## 7. Where this connects

- **[Array processing](arrays.md)** — the BLAS threading measurement in full,
  and why a moved bit did not move a bearing.
- **[Bellhop](../models/bellhop.md)** — backend selection, and the
  reproducibility difference between the fortran and multithreaded backends.
- **[Communications](comms.md)** — `simulate_link`, `ber_sweep` and the
  Monte-Carlo sweeps whose `rng` you need to seed.
- **[File I/O](io.md)** — `work_dir` and `cleanup`, the deck files the
  stability claim above is about, and the readers that parse them back.
- **[Utilities](utilities.md)** — `run_parallel`, whose cross-worker behaviour
  is on the not-measured list.

---

**See also:** [documentation index](../README.md) ·
[array processing](arrays.md) · [Bellhop](../models/bellhop.md) ·
[communications](comms.md) · [file I/O](io.md) · [utilities](utilities.md)
