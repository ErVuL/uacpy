"""Pytest configuration and fixtures for UACPY tests.

Three **autouse** fixtures below apply to every test in the suite whether or
not the test names them, so a test file cannot see them from its own source:

* ``_seed_numpy`` reseeds ``numpy.random`` before each test. Under
  ``-n logical --dist=worksteal`` (the ``addopts`` in ``pyproject.toml``)
  tests do not run in file order, so any fixture built from ``numpy.random``
  would otherwise draw different data depending on which worker picked it up.
* ``_release_matplotlib_figures`` closes all figures afterwards, so a test
  that plots without closing does not leak them into the next test's
  ``plt.gcf()`` or exhaust the figure limit over a long session.
* ``_redirect_tempdir`` points ``tempfile.tempdir`` at the per-test
  ``tmp_path``. Models built with ``work_dir=None`` allocate their scratch
  directory through ``tempfile``, so this is what makes pytest reap them.

Those three, and the module-private helpers, have no textual callers: pytest
registers a fixture from its decorator, so a static "unused symbol" scan will
flag them. They are all live — do not delete them.

Fixtures used by name across the suite start at ``simple_env``.
"""

# Lock matplotlib to a non-interactive backend before any test imports it.
# Must run before any other matplotlib import in the test session.
import matplotlib

matplotlib.use("Agg")

import contextlib  # noqa: E402
import socket  # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import uacpy  # noqa: E402
from pathlib import Path

_HAS_NETWORK = None


def _has_network():
    """One-shot connectivity probe (cached): can we open an outbound socket?"""
    global _HAS_NETWORK
    if _HAS_NETWORK is None:
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=2).close()
            _HAS_NETWORK = True
        except OSError:
            _HAS_NETWORK = False
    return _HAS_NETWORK


def pytest_collection_modifyitems(items):
    """Marker bookkeeping:

    * OASES ships as a native binary, so every ``requires_oases`` test also
      needs a binary → auto-attach ``requires_binary`` (so the pure-Python
      subset ``-m 'not requires_binary'`` excludes OASES tests too).
    * ``requires_network`` tests hit live external services (the ``uacpy.data``
      fetchers); auto-skip them when there is no internet so an offline
      ``pytest`` run stays green. The ``addopts`` in ``pyproject.toml`` already
      deselect them, so this fires only for a run that asked for them back with
      ``-m requires_network``.
    """
    offline_skip = pytest.mark.skip(reason="no network (requires_network)")
    offline = not _has_network()
    for item in items:
        if (item.get_closest_marker('requires_oases')
                and not item.get_closest_marker('requires_binary')):
            item.add_marker(pytest.mark.requires_binary)
        if offline and item.get_closest_marker('requires_network'):
            item.add_marker(offline_skip)


@pytest.fixture(autouse=True)
def _seed_numpy():
    """Seed numpy.random before every test to keep fixture data reproducible."""
    np.random.seed(0xACED)


@pytest.fixture(autouse=True)
def _release_matplotlib_figures():
    """Close every matplotlib figure after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.fixture(autouse=True)
def _redirect_tempdir(tmp_path, monkeypatch):
    """Route ``tempfile.gettempdir()`` to per-test ``tmp_path`` so any
    ``FileManager`` scratch dir is reaped by pytest (xdist-safe, no
    /dev/shm leakage)."""
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))


@pytest.fixture
def marker_fraction_inside():
    """``fn(ax, marker='*', side='left') -> float``: the share of a marker's
    width that lies inside the x limits of ``ax``, by the same marker-size
    rule the plotters pad the axis with. ``marker`` picks the line by its
    marker glyph (``'*'`` the source star, ``'o'`` the receiver dots);
    ``side`` picks the point that can be clipped on that spine — the
    smallest x for ``'left'``, the largest for ``'right'``."""
    from uacpy.visualization.plots._common import _marker_half_width_in_data

    def fraction(ax, marker='*', side='left'):
        line = next(ln for ln in ax.get_lines() if ln.get_marker() == marker)
        xs = np.ravel(line.get_xdata())
        x = float(xs.max() if side == 'right' else xs.min())
        half = _marker_half_width_in_data(ax, line.get_markersize())
        lo, hi = sorted(ax.get_xlim())
        return (min(x + half, hi) - max(x - half, lo)) / (2.0 * half)
    return fraction


#: Keyword names ``make_pekeris`` routes to the bottom half-space rather than
#: to the Environment.
_PEKERIS_BOTTOM_KEYS = frozenset((
    'acoustic_type', 'sound_speed', 'density', 'attenuation', 'roughness',
    'shear_speed', 'shear_attenuation'))


def make_pekeris(**overrides):
    """Build the canonical 100-m Pekeris waveguide: 1500 m/s isovelocity water
    over a 1700 m/s fluid half-space (density 1.8, attenuation 0.5).

    A plain function rather than a fixture, so test files can call it from
    module-level helpers and constants (``from uacpy.tests.conftest import
    make_pekeris``). Keywords named in ``_PEKERIS_BOTTOM_KEYS`` override the
    bottom half-space; every other keyword goes to :class:`uacpy.Environment`
    (``name=``, ``bathymetry=``, ``ssp=``, ``surface=``, or a wholesale
    ``bottom=``).
    """
    bottom_kw = dict(acoustic_type='half-space', sound_speed=1700.0,
                     density=1.8, attenuation=0.5)
    bottom_kw.update({k: overrides.pop(k) for k in list(overrides)
                      if k in _PEKERIS_BOTTOM_KEYS})
    env_kw = dict(bathymetry=100.0, ssp=1500.0,
                  bottom=uacpy.BoundaryProperties(**bottom_kw))
    env_kw.update(overrides)
    return uacpy.Environment(**env_kw)


def make_halfspace(sound_speed, **kwargs):
    """A fluid half-space at ``sound_speed`` (m/s), density 1.8 and
    attenuation 0.3 unless given; every other keyword goes to
    :class:`uacpy.BoundaryProperties`."""
    return uacpy.BoundaryProperties(
        acoustic_type='half-space', sound_speed=sound_speed,
        density=kwargs.pop('density', 1.8),
        attenuation=kwargs.pop('attenuation', 0.3), **kwargs)


def layered_seabed_column(thickness, speed):
    """Two sediment layers (``thickness`` then twice it; ``speed`` then
    ``speed + 50`` m/s; the upper one elastic) over an elastic 1900 m/s
    half-space."""
    return uacpy.SeabedColumn(
        layers=[uacpy.SedimentLayer(thickness=thickness, sound_speed=speed,
                                    density=1.5, attenuation=0.5,
                                    shear_speed=400.0, shear_attenuation=1.0),
                uacpy.SedimentLayer(thickness=2.0 * thickness,
                                    sound_speed=speed + 50,
                                    density=1.7, attenuation=0.6)],
        halfspace=uacpy.BoundaryProperties(
            acoustic_type='half-space', sound_speed=1900.0, density=2.0,
            attenuation=0.1, shear_speed=600.0, shear_attenuation=0.5),
    )


def wide_range_dependent_env(n_columns=24, n_bathy=97, r_end=60000.0):
    """A bottom with many columns under a wavy seafloor with many nodes.

    The two axes deliberately do not line up, so most bathymetry nodes fall
    between bottom columns and the nearest-column rule actually has to choose.
    """
    ranges = np.linspace(0.0, r_end, n_columns)
    bottom = uacpy.Bottom(
        columns=[layered_seabed_column(10.0 + 5.0 * (i % 7),
                                       1650.0 + 10.0 * (i % 11))
                 for i in range(n_columns)],
        ranges=ranges)
    r_bathy = np.linspace(0.0, r_end, n_bathy)
    z_bathy = 120.0 + 60.0 * np.sin(r_bathy / r_end * 6.0 * np.pi)
    return uacpy.Environment(
        name='wide-rd',
        bathymetry=list(zip(r_bathy.tolist(), z_bathy.tolist())),
        ssp=1500.0, bottom=bottom)


def range_independent_layered_env():
    """One ``layered_seabed_column`` under a sloping 60-400 m seafloor."""
    return uacpy.Environment(
        name='ri', bathymetry=[(0.0, 60.0), (5000.0, 400.0)],
        ssp=1500.0, bottom=layered_seabed_column(4.0, 1600.0))


def water_density_env(**kw):
    """A 100-m, 1500 m/s guide named ``'rho'`` over a 1700 m/s half-space of
    density 1.5; every keyword goes to :class:`uacpy.Environment`."""
    kw.setdefault('name', 'rho')
    kw.setdefault('bathymetry', 100.0)
    kw.setdefault('ssp', 1500.0)
    kw.setdefault('bottom', uacpy.BoundaryProperties(
        sound_speed=1700.0, density=1.5, attenuation=0.5))
    return uacpy.Environment(**kw)


@contextlib.contextmanager
def recorded_warnings():
    """``warnings.catch_warnings(record=True)`` with ``simplefilter('always')``:
    yields the list every warning raised inside the block is appended to."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        yield rec


def warning_messages(fn, needle):
    """Run ``fn`` and return the warning messages containing ``needle``."""
    with recorded_warnings() as rec:
        fn()
    return [str(w.message) for w in rec if needle in str(w.message)]


# ── the seams a test stops or watches a run at ──────────────────────────
#
# ``launch_spy``, ``stop_at`` and ``stage_spy`` are the one place the suite
# names the method that starts a binary and the stage hooks of
# ``PropagationModel``; a test names a stage, so renaming a hook is one edit
# here.


class LaunchReached(Exception):
    """Raised by the ``launch_spy`` stub: the run reached its binary."""


class StageReached(Exception):
    """Raised by the ``stop_at`` stub: the run reached the stage named in
    ``args[0]``."""


#: The method every engine starts a binary through.
_LAUNCH_METHOD = '_run_subprocess'

#: The ``PropagationModel`` stage hook each stage name stands for.
_STAGE_HOOKS = {
    'project': '_project_environment',   # stage 1
    'settings': '_resolve_settings',     # stage 3
    'write': '_write_input',             # stage 4: the deck
    'launch': '_launch',                 # stage 4: the binary
    'read': '_read_output',              # stage 4: the binary's output
    'result': '_to_result',              # stage 5
}


@pytest.fixture
def launch_spy(monkeypatch):
    """``launch_spy(target, then=None) -> list``: replace the method that
    starts a binary on ``target`` (an engine class or instance) with a stub
    that appends ``(cmd, cwd, kwargs)`` to the returned list and raises
    :class:`LaunchReached`, or returns ``then(cmd, cwd, **kwargs)`` when
    ``then`` is given."""
    def install(target, then=None):
        calls = []

        def stub(cmd, cwd, **kwargs):
            calls.append((cmd, cwd, kwargs))
            if then is None:
                raise LaunchReached()
            return then(cmd, cwd, **kwargs)

        if isinstance(target, type):
            monkeypatch.setattr(
                target, _LAUNCH_METHOD,
                lambda self, cmd, cwd, **kwargs: stub(cmd, cwd, **kwargs))
        else:
            monkeypatch.setattr(target, _LAUNCH_METHOD, stub)
        return calls
    return install


@pytest.fixture
def stop_at(monkeypatch):
    """``stop_at(target, stage)``: replace the hook of ``stage`` (a key of
    ``_STAGE_HOOKS``) on ``target`` (an engine class or instance) with a
    stub that raises ``StageReached(stage)``, so a run stops as it enters
    that stage."""
    def install(target, stage):
        def stub(*args, **kwargs):
            raise StageReached(stage)
        monkeypatch.setattr(target, _STAGE_HOOKS[stage], stub)
    return install


@pytest.fixture
def stage_spy(monkeypatch):
    """``stage_spy(target, stage) -> list``: wrap the hook of ``stage`` on
    ``target`` (a class) so each call appends its keyword arguments to the
    returned list, then runs the hook."""
    def install(target, stage):
        name = _STAGE_HOOKS[stage]
        hook = getattr(target, name)
        calls = []

        def spy(self, *args, **kwargs):
            calls.append(kwargs)
            return hook(self, *args, **kwargs)

        monkeypatch.setattr(target, name, spy)
        return calls
    return install


# ── the registered engines ──────────────────────────────────────────────


def engine_params(value='class'):
    """One ``pytest.param`` per engine of ``uacpy.models._registry.ENGINES``,
    in registry order, with the engine's class name as its id and the
    ``requires_<install>`` markers of the installs its entry declares
    (``EngineEntry.requires``). The value is the engine class, or its class
    name when ``value='name'``.

    A plain function rather than a fixture, so a module-level
    ``parametrize`` can call it (``from uacpy.tests.conftest import
    engine_params``). A test parametrised over it meets an engine added to
    the registry on its first run."""
    from uacpy.models._registry import ENGINES

    params = []
    for entry in ENGINES.values():
        marks = [getattr(pytest.mark, f'requires_{install}')
                 for install in entry.requires]
        arg = entry.load() if value == 'class' else entry.class_name
        params.append(pytest.param(arg, marks=marks, id=entry.class_name))
    return params


def engine_names() -> set:
    """The class names of the registered engines."""
    from uacpy.models._registry import ENGINES
    return {entry.class_name for entry in ENGINES.values()}


def engine_entry(name):
    """The registry entry (``EngineEntry``) of the engine whose class is
    named ``name``."""
    from uacpy.models._registry import ENGINES

    return next(e for e in ENGINES.values() if e.class_name == name)


def build_engine(name, **kwargs):
    """An instance of the registered engine whose class is named ``name``,
    built with its registry ``example_kwargs`` and ``kwargs`` (which win)."""
    entry = engine_entry(name)
    return entry.load()(**{**dict(entry.example_kwargs), **kwargs})


@pytest.fixture
def simple_env():
    """Simple isovelocity environment."""
    return uacpy.Environment(
        name="Test Environment",
        bathymetry=100.0,
        ssp=1500.0,
    )


@pytest.fixture
def parabolic_ssp_env():
    """100-m shallow-water env with a parabolic SSP centred at 50 m.

    Not the canonical Munk profile (which carries an exponential
    ``η - 1 + exp(-η)`` term and channels at ~1300 m in deep water).
    Used for SSP-shape smoke checks where the only requirement is "a
    non-flat profile with a minimum somewhere".
    """
    from uacpy.core.environment import SoundSpeedProfile
    depths = np.linspace(0, 100, 21)
    axis_depth = 50
    c_axis = 1485
    sound_speeds = c_axis * (1 + 0.00737 * ((depths - axis_depth) / axis_depth) ** 2)

    return uacpy.Environment(
        name="Parabolic SSP",
        bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs(
            np.column_stack([depths, sound_speeds])
        ),
    )


@pytest.fixture
def munk_env():
    """Deep-water Munk profile (canonical, axis at 1300 m).

    Built via :meth:`SoundSpeedProfile.from_munk`, which implements
    ``c(z) = c_min * (1 + ε * (η - 1 + exp(-η)))`` with
    ``η = 2(z - z_axis)/z_axis``, ``c_min = 1500 m/s``, ``ε = 7.37e-3``.
    Bathymetry is 5 km with a fluid half-space bottom.
    """
    from uacpy.core.environment import SoundSpeedProfile
    return uacpy.Environment(
        name="Munk Profile",
        bathymetry=5000.0,
        ssp=SoundSpeedProfile.from_munk(5000.0),
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            density=1.8,
            attenuation=0.3,
        ),
    )


@pytest.fixture
def range_dependent_env():
    """Range-dependent environment with bathymetry."""
    ranges = np.linspace(0, 10000, 11)
    depths = np.linspace(80, 120, 11)
    bathymetry = np.column_stack([ranges, depths])

    return uacpy.Environment(
        name="Range Dependent",
        ssp=1500.0,
        bathymetry=bathymetry,
    )


@pytest.fixture
def source():
    """Standard acoustic source."""
    return uacpy.Source(depths=50.0, frequencies=100.0)


@pytest.fixture
def receiver_grid():
    """Standard receiver grid."""
    return uacpy.Receiver(
        depths=np.linspace(10, 90, 9),
        ranges=np.linspace(100, 5000, 11)
    )


@pytest.fixture
def receiver_small():
    """Small receiver grid for fast tests."""
    return uacpy.Receiver(
        depths=np.array([25.0, 50.0, 75.0]),
        ranges=np.array([1000.0, 3000.0, 5000.0])
    )


@pytest.fixture
def receiver(receiver_grid):
    """Default receiver grid (alias for ``receiver_grid``)."""
    return receiver_grid


@pytest.fixture
def halfspace_bottom():
    """Standard fluid half-space sediment used by Pekeris-style tests."""
    from uacpy.core.environment import BoundaryProperties
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0,
        density=1.8,
        attenuation=0.5,
    )


@pytest.fixture
def elastic_bottom():
    """Half-space with shear (used by elastic-bottom Pekeris cases)."""
    from uacpy.core.environment import BoundaryProperties
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0,
        shear_speed=400.0,
        density=1.8,
        attenuation=0.5,
        shear_attenuation=0.8,
    )


@pytest.fixture
def pekeris_env(halfspace_bottom):
    """Classic 100-m Pekeris waveguide with a fluid half-space bottom."""
    return make_pekeris(name="Pekeris (fluid bottom)", bottom=halfspace_bottom)


@pytest.fixture
def elastic_env(elastic_bottom):
    """Pekeris waveguide with an elastic half-space bottom (shear=400 m/s)."""
    return uacpy.Environment(
        name="Pekeris (elastic bottom)",
        bathymetry=100.0,
        ssp=1500.0,
        bottom=elastic_bottom,
    )


# Reference Acoustics-Toolbox SSP files vendored under third_party.
_AT_REF_DIR = (Path(__file__).resolve().parent.parent /
               "third_party" / "Acoustics-Toolbox" / "tests")


# ── a two-layer water column and a measured absorption table ────────────
#
# Warm and salty over cold and fresher, with a 20 m thermocline between the
# layers: the Francois-Garrison profile the absorption tests of the law, the
# AT and RAM decks and the engines share.

TWO_LAYER_DEPTHS = np.array([0.0, 40.0, 60.0, 100.0])
TWO_LAYER_TEMPERATURE = np.array([20.0, 20.0, 8.0, 8.0])
TWO_LAYER_SALINITY = np.array([35.0, 35.0, 34.0, 34.0])


def two_layer_absorption():
    """The two-layer column as a Francois-Garrison profile (pH 8)."""
    from uacpy.core.absorption import FrancoisGarrison
    return FrancoisGarrison(
        temperature=TWO_LAYER_TEMPERATURE, salinity=TWO_LAYER_SALINITY,
        pH=8.0, depths=TWO_LAYER_DEPTHS)


def two_layer_dB_per_m(frequency, depth):
    """The formula at ``depth`` with the two-layer water there, by hand."""
    from uacpy.core.acoustics.attenuation import absorption_francois_garrison
    t = np.interp(depth, TWO_LAYER_DEPTHS, TWO_LAYER_TEMPERATURE)
    s = np.interp(depth, TWO_LAYER_DEPTHS, TWO_LAYER_SALINITY)
    return float(absorption_francois_garrison(frequency, t, s, 8.0,
                                             depth)) / 1000.0


def measured_absorption_table(depths=(0.0, 100.0)):
    """A measured α(f, z) with no law behind it (``model=None``): 2 depths
    × 2 frequencies, dB/km."""
    from uacpy.core.absorption import AbsorptionCoefficient
    return AbsorptionCoefficient(
        frequencies=np.array([1000.0, 10000.0]),
        data=np.array([[0.06, 0.90], [0.04, 0.50]]), units='dB/km',
        depths=np.asarray(depths, dtype=float))


def at_deck_water_rows(deck_text):
    """``(depth, c, alphaI)`` of every row of an AT deck written as a water
    SSP row (zero shear and shear attenuation)."""
    import re
    rows = re.findall(r'^  (\S+) (\S+) 0\.000000 \S+ (\S+) 0\.000000 /$',
                      deck_text, flags=re.M)
    return np.array([[float(v) for v in row] for row in rows])
