# Continuous Integration — UACPY

This document describes the CI/CD pipeline that gates pull requests on the
[`ErVuL/uacpy`](https://github.com/ErVuL/uacpy) repository. It covers what
runs, why, how to install it on GitHub, and how to interpret the results.

If you only need the headline: every PR triggers a Linux runner that lints
the Python code with flake8, restores native binaries from cache (or builds
them via `install.sh` on cache miss), and runs the full test suite minus
tests marked `slow`. A PR cannot merge until both checks are green.

The native build is the slow part (~15–20 min); the test suite is quick.
Two GitHub Actions caches keep that cost off Python-only PRs:

- a **static** cache for OASES (key `Linux-X64-uacpy-oases-v1`) — OASES is
  unmaintained upstream so it never invalidates by content; bump the `-vN`
  suffix to force a rebuild.
- a **dynamic** cache for OALIB + bellhopcuda + mpiramS + ramsurf, keyed on
  the hash of `install.sh` + `third_party/Acoustics-Toolbox/**` +
  `third_party/mpiramS/**` + `third_party/ramsurf/**` +
  `third_party/MODIFICATIONS.md`.

A typical Python-only PR hits both caches and finishes in ~5 min total.

---

## 1. Files involved

| Path                                 | Purpose                                                        |
| ------------------------------------ | -------------------------------------------------------------- |
| `.github/workflows/ci.yml`           | The CI workflow definition. GitHub auto-discovers it.          |
| `install.sh`                         | Pinned to `BELLHOPCUDA_TAG="v1.5"` (line 49). New `pin_bellhopcuda_tag()` helper enforces the pin at build time. |
| `CI.md` (this file)                  | Documentation only.                                            |

Nothing else needs to change in the codebase.

---

## 2. What runs on a pull request

The workflow declares **two jobs**, run sequentially:

```
lint  ────►  build-and-test
(5 min)      (5 min cache hit / 15–25 min cache miss)
```

### Job 1 — `Lint (flake8)`

A 5-minute Python-only sanity check. Runs `flake8` with a deliberately narrow
scope:

```
flake8 uacpy/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

| Selector | What it catches                                                             |
| -------- | --------------------------------------------------------------------------- |
| `E9`     | Syntax errors (indentation, parsing failures)                               |
| `F63`    | Invalid comparisons (`==` on non-comparable types)                          |
| `F7`     | Logic errors (e.g. `assert` on a tuple — always truthy, never asserted)     |
| `F82`    | Undefined names (typos, missing imports)                                    |

Style/PEP8 (`E1`, `E2`, `W…`) is **not** enforced. The intent is to fail
loudly on real bugs without forcing a style cleanup. Tighten later by
committing a `.flake8` config and removing the `--select` filter.

### Job 2 — `Build native binaries + run tests`

The headline job. `needs: lint` means it only runs if lint is green —
saves runner minutes when there's a syntax error.

Steps in order:

1. **Checkout (recursive submodules).** `actions/checkout@v4` with
   `submodules: recursive` materializes `bellhopcuda` and its nested `glm`
   submodule. When `install.sh` actually runs (cache miss), the
   `gitdir:`-pointer file triggers the `fixup_bhc_dotgit` workaround.

2. **Set up Python 3.12** (`actions/setup-python@v5`) with pip caching.

3. **Install system build deps**:
   ```
   gfortran make cmake g++ curl tar
   ```
   The Linux/Debian list from the README, minus tools the runner already has.
   Always installed — they're cheap and several are needed even on cache hit
   (e.g. `tar` for the cache restore itself).

4. **Configure OASES strategy.** Inspects `OASES_MIRROR_TOKEN` and emits
   `token_available=yes|no`. Token-less runs (fork PRs) can still consume
   a populated OASES binary cache — the token only governs whether OASES
   can be **rebuilt** from sources.

5. **Restore OASES binary cache.** Static key
   `${{ runner.os }}-${{ runner.arch }}-uacpy-oases-v1`. See § 3.

6. **Restore other native-binary cache** (oalib + bellhopcuda + mpirams + ramsurf).
   Dynamic key based on a content hash. See § 4.

7. **Plan build (cache-aware)** — the `plan` step combines the two cache
   outcomes with the token state and emits three flags consumed by later
   steps:
   - `install_oases` — `yes`/`no`, passed verbatim to `install.sh --oases`
   - `need_mirror_fetch` — whether to clone the private OASES mirror
   - `run_install` — whether to invoke `install.sh` at all

   The decision matrix:

   | rest cache | OASES cache | token | `run_install` | `install_oases` | mirror fetched? |
   | ---------- | ----------- | ----- | ------------- | --------------- | --------------- |
   | hit        | hit         | any   | **no**        | no              | no              |
   | hit        | miss        | yes   | yes           | yes             | yes             |
   | hit        | miss        | no    | **no**        | no              | no              |
   | miss       | hit         | any   | yes           | no              | no              |
   | miss       | miss        | yes   | yes           | yes             | yes             |
   | miss       | miss        | no    | yes           | no              | no              |

8. **Fetch OASES from private mirror** (skipped unless `need_mirror_fetch=yes`):
   ```bash
   git clone --depth 1 \
     "https://x-access-token:${TOKEN}@github.com/ErVuL/oases.git" \
     /tmp/oases-repo
   tar -xzf /tmp/oases-repo/oases.tgz -C /tmp/oases-extract
   # move extracted tree to uacpy/third_party/oases/
   ```
   Once the directory exists, `install.sh` skips its own MIT download.

9. **`pip install -e ".[dev]"`** — editable install plus the `[dev]` extras
   (`pytest`, `pytest-cov`, `black`, `flake8`). Always runs (the Python
   package isn't cached; pip's HTTP cache is, via `setup-python`).

10. **`./install.sh -y --bellhop cxx --oases <install_oases>`** — runs only
    when `run_install=yes`. On a hit-everywhere PR this step is skipped
    entirely. When it does run it builds:
    - **OALIB** (Acoustics-Toolbox: Bellhop Fortran, Kraken, KrakenC,
      KrakenField, Scooter, SPARC, Bounce)
    - **bellhopcuda CXX** (CPU C++ Bellhop) — pinned to `v1.5`. Also
      exercises the GLM nested submodule and the `fixup_bhc_dotgit`
      workaround.
    - **OASES** (OAST/OASN/OASR/OASP) — only if the mirror was fetched
    - **mpiramS** (RAM PE)

    CUDA Bellhop is intentionally **not** built. GitHub-hosted runners have
    no GPU, so a CUDA build would compile but couldn't be tested.

11. **List installed binaries** — always runs. Diagnostic that confirms
    what landed in `uacpy/bin/{oalib,bellhopcuda,oases,mpirams,ramsurf}`
    regardless of whether it came from cache or fresh build. The
    `ramsurf` directory holds two binaries built from the same source
    tree: `rams0.5` (Collins elastic PE) and `ramsurf1.5` (Collins
    rough-surface PE).

12. **Determine pytest marker.** Inspects `uacpy/bin/oases/` after the
    restore + (optional) build. If non-empty → `marker="not slow"`. If
    empty → `marker="not slow and not requires_oases"`. This decouples
    test selection from token availability.

13. **Run tests**:
    ```
    pytest uacpy/tests/ -m "<marker>" -v --maxfail=20
    ```
    `TMPDIR=$RUNNER_TEMP` keeps `FileManager`'s tmpfs paths off `/dev/shm`
    (which is small on hosted runners).

14. **Save other native-binary cache** — `actions/cache/save@v4`. Gated
    on `success() && cache-rest restored as miss`. Placed *after* pytest
    so a red test run cannot poison the cache.

15. **Save OASES binary cache** — `actions/cache/save@v4`. Additionally
    gated on `steps.plan.outputs.install_oases == 'yes'` so a token-less
    run that left `uacpy/bin/oases/` empty does not save under the
    static key (which would otherwise persist forever).

16. **Upload build logs on failure** — `/tmp/oalib_build.log`,
    `/tmp/oases_build.log`, `/tmp/mpirams_build.log`,
    `/tmp/ramsurf_build.log` are uploaded as the `build-logs`
    artifact when any earlier step failed. Download from the Actions
    UI to debug a build break.

---

## 3. Native-binary caching

Compiling all four binaries from scratch costs ~15–20 min. On a Python-only
PR (no change to `install.sh` or any third-party source) that work is pure
waste. The workflow caches the build output, keyed so that the cache
invalidates exactly when (and only when) the inputs to the build change.

### Two caches, two policies

| Cache id | Path(s) | Key | Invalidates when… |
| -------- | ------- | --- | ----------------- |
| `cache-oases` | `uacpy/bin/oases` | `${{ runner.os }}-${{ runner.arch }}-uacpy-oases-v1` | **Static** — only when the `-vN` suffix is bumped, or the cache is manually deleted, or 7 days of LRU eviction passes |
| `cache-rest`  | `uacpy/bin/{oalib,bellhopcuda,mpirams,ramsurf}` | `${{ runner.os }}-${{ runner.arch }}-uacpy-bin-${{ hashFiles('install.sh', 'uacpy/third_party/Acoustics-Toolbox/**', 'uacpy/third_party/mpiramS/**', 'uacpy/third_party/ramsurf/**', 'uacpy/third_party/MODIFICATIONS.md') }}` | Any of the hashed files change |

`runner.arch` is in both keys so a future shift of `ubuntu-latest` to
ARM64 invalidates the cache automatically rather than restoring an x86_64
binary on an aarch64 runner.

OASES is treated specially because the upstream project is **unmaintained**:
the source tarball at MIT will not change, and the local build inputs
(compiler flags inside `install.sh`'s OASES block) almost never change. A
content-based key would still invalidate the OASES cache every time
`install.sh` is touched anywhere; a static key avoids that.

The bellhopcuda submodule SHA is **not** in the rest-cache key. The
`BELLHOPCUDA_TAG="v1.5"` constant in `install.sh` (§ 4) pins the version,
so the install.sh hash already captures it transitively.

### Restore / save are split — and save runs *after* the tests

Both caches use the `actions/cache/restore` and `actions/cache/save`
sub-actions rather than the unified `actions/cache@v4`. The save steps
sit at the end of the job, **after `pytest`**, gated on `success()` and
on the matching restore having missed. Two reasons:

1. **A failing test never poisons the cache.** With unified
   `actions/cache@v4`, the post-step uploads the cache regardless of
   whether subsequent steps (including pytest) succeeded. If install.sh
   half-succeeded — e.g. partial OALIB build, `INSTALLED_COUNT > 0`, exit
   0 — the broken binaries would be cached and every subsequent run
   would skip the build and run tests against them. Splitting + the
   `if: success()` guard means a red pytest skips the save.

2. **No empty-OASES cache poisoning.** Token-less runs invoke
   `install.sh --oases no`, which leaves `uacpy/bin/oases/` empty (the
   directory is created by `ensure_dir` regardless). Without gating, the
   post-step would upload that empty directory under the static key
   `…-uacpy-oases-v1`. **All future runs** would see `oases_hit=true`
   and skip OASES forever — even after a token is provisioned, until
   someone manually purges. The save step is gated on
   `steps.plan.outputs.install_oases == 'yes'`, so it only saves when
   install.sh just built OASES.

### Why CI overrides `-march`

Two of `install.sh`'s Fortran components hardcode `-march=native` for
local performance:

- **OALIB** (in `install.sh` itself, OALIB FFLAGS construction)
- **mpiramS** (in `uacpy/third_party/mpiramS/Makefile`, lines 22-23, both
  FFLAGS and LDFLAGS)

That's the right default for a local install — you get every
instruction-set extension your CPU supports. But `-march=native` produces
a binary tied to the *build host's* microarchitecture. GitHub-hosted
runners do not pin a single CPU model, so a cache populated on a Skylake
host (AVX-512 in use) could `SIGILL` when restored on a Zen-class host.

To make the cached binaries safe to reuse across the runner pool, the
build step in `ci.yml` sets:

```yaml
env:
  UACPY_FORTRAN_ARCH_FLAGS: "-march=x86-64-v3"
```

`x86-64-v3` is the Haswell-era baseline (AVX2 / BMI / FMA / popcnt /
LZCNT / MOVBE — every x86_64 CPU from ~2013 onward, Intel and AMD).
`install.sh` reads this env var with `:-`-default fallback, so local
installs are unaffected (default stays `-march=native -mtune=native`).

The env var governs **every Fortran build in `install.sh`**:

| Component   | Architecture flag source                   | Honors `UACPY_FORTRAN_ARCH_FLAGS`? |
| ----------- | ------------------------------------------ | --------------------------------- |
| OALIB       | `install.sh` constructs OALIB FFLAGS       | yes                               |
| mpiramS     | upstream Makefile, overridden via `make FFLAGS=…` on the command line | yes |
| OASES       | upstream Makefile, `-O2` only, no `-march` | n/a — already portable            |
| bellhopcuda | CMake default Release flags, no `-march`   | n/a — already portable            |

If you ever see a CI test fail with `Illegal instruction (core dumped)`,
shrink the level: `-march=x86-64-v2` (Nehalem/SSE4.2, ~2008) is the next
step down. Bumping the level (`-v4`, AVX-512) is safe only if you're
*certain* every runner the cache lands on supports it.

### `restore-keys` is intentionally not set

`actions/cache` supports `restore-keys` for fuzzy-prefix fallback. We
don't use it. A partial / stale cache would mean running tests against
binaries that don't match the current sources — a correctness risk worse
than the speed loss it would mitigate. On miss we rebuild from scratch.

### What hits and what doesn't

- **Edit a Python file in `uacpy/uacpy/`** → both caches hit → `install.sh`
  skipped entirely. Test job runs in ~5 min total.
- **Edit `install.sh` (e.g. add a CFLAG)** → rest cache misses, OASES
  hits → `install.sh --oases no` runs (rebuilds oalib + bellhopcuda +
  mpirams + ramsurf; reuses cached OASES). ~10 min.
- **Edit a Fortran source under `third_party/Acoustics-Toolbox/`,
  `third_party/mpiramS/`, or `third_party/ramsurf/`** → rest cache
  misses, OASES hits → same as above.
- **First run after the cache was evicted** → both miss → full build.
  This is also the steady-state cost when no caching exists (i.e. nothing
  worse than today).

### Forcing a rebuild

Three ways, in order of bluntness:

1. **Bump the OASES key suffix** — edit `.github/workflows/ci.yml`,
   change `…-uacpy-oases-v1` → `…-v2`. Use this when an OASES rebuild
   is genuinely needed (toolchain bump, compiler-flag tweak inside the
   OASES block of `install.sh`).
2. **Delete a specific cache** — from the GitHub UI:
   <https://github.com/ErVuL/uacpy/actions/caches>, or from the CLI:
   ```bash
   gh cache list --repo ErVuL/uacpy
   gh cache delete <cache-key-or-id> --repo ErVuL/uacpy
   ```
3. **Touch the inputs** — any commit that changes `install.sh` or anything
   under `third_party/Acoustics-Toolbox/`, `third_party/mpiramS/`, or
   `third_party/MODIFICATIONS.md` will naturally invalidate the rest
   cache on the next run.

### GitHub cache lifecycle reminders

- 7-day LRU eviction (any cache untouched for 7 days is dropped).
- 10 GB total budget per repo (oldest evicted first when over).
- Caches are **branch-scoped** with a fallback chain: a PR branch reads
  its own cache first, then the base branch, then the default branch. So
  a brand-new feature branch usually inherits `main`'s caches on first run.
- Caches scoped to a deleted branch are deleted with the branch.

The `main`-branch cache is the one PRs almost always pull from. Keep it
warm by ensuring at least one `main` build per week succeeds. If the repo
is quiet for >7 days, add a weekly `schedule:` cron that runs the workflow
on `main` to re-populate.

### Cost on a cache miss

Identical to the no-cache baseline. `actions/cache@v4` falls through
silently, `install.sh` runs, and the post-job step writes the cache for
next time. There is no error path specific to a cache miss — it's the
default code path.

---

## 4. The bellhopcuda v1.5 pin

`install.sh` declares the pinned tag as a top-level constant (line 49):

```bash
BELLHOPCUDA_TAG="v1.5"
```

After the bellhopcuda submodule is initialized, the new `pin_bellhopcuda_tag()`
helper enforces it:

1. If `git -C bellhopcuda describe --tags --exact-match` already prints
   `v1.5`, do nothing.
2. Otherwise, ensure the tag exists locally (`git fetch --tags origin` if
   missing — necessary on shallow clones / CI checkouts).
3. `git checkout v1.5` in the submodule.
4. Re-run `git submodule update --init --recursive` inside bellhopcuda so
   the GLM nested submodule lands at whatever commit the v1.5 tree records.

This is defensive even when the parent repo's recorded gitlink is already
v1.5 — it costs ~50 ms in the no-op case and removes a silent failure mode
(parent repo's gitlink drifts to `master` HEAD because someone did a
`git add bellhopcuda/` without thinking).

**To upgrade later**, change one line in `install.sh`. The pinned tag is
re-validated in CI on every PR.

---

## 5. One-time GitHub setup

Do these three things, in order, exactly once:

### 5.A — Push the workflow

```bash
git add .github/workflows/ci.yml install.sh CI.md
git commit -m "Add CI workflow + pin bellhopcuda to v1.5"
git push
```

The first run starts immediately on push. Watch it under the **Actions** tab.

### 5.B — Generate a fine-grained PAT for the OASES mirror

1. Visit <https://github.com/settings/tokens?type=beta> → **Generate new token (fine-grained)**.
2. **Token name**: `uacpy-ci-oases-readonly`.
3. **Expiration**: pick a value (1 year is reasonable; you'll need to rotate).
4. **Resource owner**: `ErVuL`.
5. **Repository access**: **Only select repositories** → tick `ErVuL/oases` only.
6. **Repository permissions** → **Contents: Read-only**. Leave everything
   else as "No access".
7. Click **Generate token**, copy the value (`github_pat_…`) — it's only
   shown once.

### 5.C — Store the token as a secret on `ErVuL/uacpy`

1. <https://github.com/ErVuL/uacpy/settings/secrets/actions> → **New repository secret**.
2. **Name**: `OASES_MIRROR_TOKEN` (case-sensitive — must match the workflow exactly).
3. **Secret**: paste the PAT.
4. **Add secret**.

The workflow references it as `${{ secrets.OASES_MIRROR_TOKEN }}`. The value
is automatically redacted in logs.

### 5.D — Make both checks required for merge

Trigger the workflow at least once first (push or open a PR) — GitHub only
lets you mark a check "required" after it has appeared in a previous run.

Then:

1. <https://github.com/ErVuL/uacpy/settings/branches> → **Add branch protection rule**.
2. Branch name pattern: `main`.
3. Tick:
   - **Require a pull request before merging**
   - **Require status checks to pass before merging** → in the search box,
     find and tick both:
     - `Lint (flake8)`
     - `Build native binaries + run tests`
   - **Require branches to be up to date before merging** (recommended).
4. Save.

From now on, no PR to `main` can merge until both checks are green.

---

## 6. PR lifecycle

When a PR is opened or pushed to:

1. Yellow dot — **CI / Lint (flake8) — In progress** appears within a few seconds.
2. After ~5 min, lint resolves green or red. If red, build-and-test does
   not run.
3. If lint is green, **CI / Build native binaries + run tests — In progress**
   appears.
4. After ~5 min (cache hit) or ~15–25 min (cache miss), the merge button
   enables (both green) or stays disabled (one or both red).
5. Click **Details** on a failed check to jump straight to its logs.
6. From the Actions tab, you can re-trigger a failed run without pushing
   new commits ("Re-run failed jobs"), useful when you suspect a flaky
   network or runner-side issue.

---

## 7. Reproducing the gate locally

If you want to run the same checks before pushing:

```bash
# Lint (matches CI exactly)
pip install flake8
flake8 uacpy/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Full build + tests (matches CI's "internal PR" path)
pip install -e ".[dev]"
./install.sh -y --bellhop cxx --oases yes
pytest uacpy/tests/ -m "not slow" -v
```

For the OASES step locally you don't need the PAT — `install.sh` will
download from MIT directly when `uacpy/third_party/oases/` doesn't exist
yet.

---

## 8. Troubleshooting

| Symptom                                                          | Likely cause                                                        | Fix                                                                                                          |
| ---------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `git clone … oases.git` returns 403 / 404                        | PAT missing, expired, or scoped to the wrong repo                   | Re-issue the PAT (§ 5.B), re-save the secret (§ 5.C). Token must have **Contents: Read** on `ErVuL/oases`      |
| `tar: oases.tgz: Cannot open`                                    | Mirror file isn't named `oases.tgz`                                 | Either rename the file in the mirror, or change the path in step 5 of `ci.yml`                               |
| `install.sh` reports OASES build failure with no errors visible  | OASES tarball layout doesn't match what install.sh expects          | Inspect the tarball's top-level layout; pin the matching `mv` line in step 5 of `ci.yml`                     |
| `bellhopcuda tag 'v1.5'` not found                               | Submodule clone is shallow and tags weren't fetched                 | Already handled by `pin_bellhopcuda_tag()` (does `git fetch --tags`). If it still fails, the tag was renamed upstream — update `BELLHOPCUDA_TAG` in `install.sh` |
| Many `requires_binary` tests fail                                | One or more binaries didn't build                                   | Check step 8's binary listing in the logs. Download the `build-logs` artifact for the per-component log     |
| `OASES_MIRROR_TOKEN not set` warning on internal PRs             | Secret never added, or repo settings don't expose secrets to PRs    | Re-do § 5.C. Verify under repo Settings → Secrets and variables → Actions                                     |
| Tests time out at 60 min                                         | A binary hung on bad input                                          | Bump `timeout-minutes`, and consider adding `pytest-timeout` to fail individual tests at, say, 5 min         |
| Runner runs out of disk                                          | OALIB sources + builds total ~250 MB; should fit, but artifacts pile up | Add `df -h` as a debug step before the build to confirm                                                     |
| Tests fail mysteriously after a toolchain change, with no source change to explain it | Stale cached binaries built with the old toolchain        | Bump `…-uacpy-oases-v1` → `…-v2` for OASES; for the rest cache, touch any hashed input or delete via `gh cache delete` (§ 3) |
| `Build native binaries` step is **skipped** unexpectedly         | Both caches hit — by design                                         | Inspect the **Plan build (cache-aware)** step: it logs which caches hit and what flags were chosen. If you need a forced rebuild, see § 3 "Forcing a rebuild" |
| OASES tests run on a fork PR with no token                       | Fork PR inherited a populated OASES binary cache from `main` — also by design | If undesired, gate the marker step on `steps.oases.outputs.token_available`. Otherwise this is the intended behaviour |
| Cache-save post-step fails with "Cache size of … exceeds 10GB"   | Repo cache budget exhausted                                         | Delete old caches: `gh cache list --repo ErVuL/uacpy` then `gh cache delete <key>` |

---

## 9. What the gate gives you, in one paragraph

Every PR is gated by a fresh Ubuntu VM that lints the Python source with
flake8 (real-bug rules only), restores two binary caches (a static one for
OASES, a content-keyed one for OALIB + bellhopcuda + mpiramS), and on cache
miss recursively pulls submodules, fetches the private OASES mirror via
PAT, forces the bellhopcuda submodule to the pinned `v1.5` upstream tag,
and runs `./install.sh -y --bellhop cxx --oases <yes|no>` end-to-end —
validating the script, the GLM nested-submodule path, the
`fixup_bhc_dotgit` workaround, and the affected binary builds. The full
test suite (minus `slow`) then runs against whatever binaries are present.
A regression in the wrappers, the I/O readers, the build script, or any
of the third-party patches all fail the merge; a Python-only PR that
touches none of those finishes in ~5 min thanks to the caches.
