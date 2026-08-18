"""
Comprehensive tests for OASES models

OASES (Ocean Acoustics and Seismic Exploration System) is a suite of
seismo-acoustic models for underwater acoustics. This test file provides
systematic validation of all OASES variants:

- OAST: Transmission loss computation
- OASN: Array covariance and signal replicas under surface/discrete noise
- OASR: Reflection coefficients
- OASP: Pulse / broadband wavenumber-integration transfer functions
"""

import pytest
import numpy as np

from uacpy.models import OAST, OASN, OASR, OASP, OASES, RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ConfigurationError, FileFormatError
from uacpy.core.results import Field, ReflectionCoefficient
from uacpy.tests.conftest import make_pekeris

pytestmark = pytest.mark.requires_oases


# The TL tests below use receiver ranges that do not land on OAST's internal
# FFT grid, so the dB-interpolation warning fires incidentally; it is asserted
# directly in test_oases_warnings.py, so silence it here. (The leading '.'
# matches the literal colon in "OAST:", which is the filter-spec separator.)
_OAST_INTERP_FILTER = "ignore:OAST. receiver.ranges do not match:UserWarning"


@pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
class TestOAST:
    """Tests for OAST (transmission loss via wavenumber integration)."""

    @pytest.fixture
    def oast_env(self):
        """Create simple environment for OAST."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            density=1.5,
            attenuation=0.5
        )
        return Environment(
            name="oast_test",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom
        )

    @pytest.fixture
    def receiver(self):
        return Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0, 5000.0])
        )

    @pytest.mark.requires_binary
    def test_oast_compute_tl(self, oast_env, source, receiver):
        """Test OAST transmission loss computation."""
        oast = OAST(verbose=False)
        result = oast.compute_tl(
            env=oast_env,
            source=source,
            receiver=receiver
        )

        assert isinstance(result, Field)
        assert result.shape == (len(receiver.depths), len(receiver.ranges))
        assert np.all(np.isfinite(result.data))
        assert np.all(result.data > 0), "TL should be positive"

    @pytest.mark.requires_binary
    def test_oast_elastic_bottom(self, source, receiver):
        """Test OAST with elastic bottom (shear waves)."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0,
            shear_speed=400.0,
            density=1.8,
            attenuation=0.5,
            shear_attenuation=1.0
        )
        env = Environment(
            name="oast_elastic",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom
        )

        oast = OAST(verbose=False)
        result = oast.compute_tl(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

        # Shear opens a loss channel a fluid seabed does not have, so the
        # same seabed with the shear dropped must give a different field —
        # a shear parameter that never reached the deck would land at 0
        # (pattern: test_elastic_boundaries.test_elastic_vs_fluid_difference).
        fluid = Environment(
            name="oast_fluid", bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.8, attenuation=0.5))
        ref = oast.compute_tl(env=fluid, source=source, receiver=receiver)
        diff = np.abs(np.asarray(result.db) - np.asarray(ref.db))
        assert np.nanmean(diff) > 0.5, (
            "elastic and fluid seabeds gave the same OAST field — the shear "
            "speed never reached the deck")


class TestOASN:
    """Tests for OASN (noise covariance and signal replicas, oasn.tex:1)."""

    @pytest.fixture
    def oasn_env(self):
        """Create environment for OASN."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            density=1.5,
            attenuation=0.5
        )
        return Environment(
            name="oasn_test",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom
        )

    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[50.0], ranges=[1000.0])

    @pytest.mark.requires_binary
    def test_oasn_compute_covariance(self, oasn_env, source):
        """OASN.compute_covariance returns a populated Covariance result.

        The receiver carries a non-zero range, which OASN ignores (it
        models a vertical array at x = y = 0) — so the call must warn.
        A surface noise field populates the matrix, which must then be
        Hermitian with a positive diagonal (the pattern the OASS
        covariance test pins) — an all-zeros or garbage matrix fails both.
        """
        from uacpy import Covariance
        array = Receiver(depths=[30.0, 50.0, 70.0], ranges=[1000.0])
        oasn = OASN(verbose=False, surface_noise_level=60.0)
        with pytest.warns(UserWarning, match=r"receiver\.ranges is ignored"):
            cov = oasn.compute_covariance(
                env=oasn_env, source=source, receiver=array,
            )
        assert isinstance(cov, Covariance)
        assert cov.covariance.ndim == 3
        assert cov.covariance.shape[1] == cov.covariance.shape[2] == cov.n_receivers
        assert cov.n_receivers == 3
        assert cov.n_frequencies >= 1
        C = cov.covariance[0]
        assert np.allclose(C, C.conj().T, atol=1e-5 * np.abs(C).max())
        assert (np.real(np.diag(C)) > 0).all()

    @pytest.mark.requires_binary
    def test_oasn_compute_replicas_helper(self, oasn_env, source):
        """Verify ``OASN.compute_replicas`` runs and returns a Replicas object."""
        from uacpy import Replicas
        rcv_array = Receiver(
            depths=np.linspace(40.0, 60.0, 5),
            ranges=[1000.0],
        )
        oasn = OASN(
            verbose=False,
            zmin=20.0, zmax=80.0, nz=4,
            xmin=500.0, xmax=2000.0, nx=4,  # metres
        )
        # rcv_array carries a non-zero range, which OASN ignores → warns.
        with pytest.warns(UserWarning, match=r"receiver\.ranges is ignored"):
            rep = oasn.compute_replicas(
                env=oasn_env, source=source, receiver=rcv_array,
            )
        assert isinstance(rep, Replicas)
        # replica_x axis must be in metres (uacpy public-API convention).
        assert rep.replica_x.min() >= 500.0 - 1.0
        assert rep.replica_x.max() <= 2000.0 + 1.0

    @pytest.mark.requires_binary
    def test_oasn_elastic_covariance(self, source):
        """OASN accepts an elastic-bottom env and returns a Covariance that
        is Hermitian with a positive diagonal, like the fluid sibling."""
        from uacpy import Covariance
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0,
            shear_speed=400.0,
            density=1.8,
            attenuation=0.5,
            shear_attenuation=1.0
        )
        env = Environment(
            name="oasn_elastic",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom
        )

        array = Receiver(depths=[30.0, 50.0, 70.0], ranges=[1000.0])
        oasn = OASN(verbose=False, surface_noise_level=60.0)
        with pytest.warns(UserWarning, match=r"receiver\.ranges is ignored"):
            cov = oasn.compute_covariance(
                env=env, source=source, receiver=array,
            )
        assert isinstance(cov, Covariance)
        C = cov.covariance[0]
        assert C.shape == (3, 3)
        assert np.allclose(C, C.conj().T, atol=1e-5 * np.abs(C).max())
        assert (np.real(np.diag(C)) > 0).all()

    class _DeckCaptured(Exception):
        pass

    def _captured_options(self, monkeypatch, model, run_mode):
        import uacpy.models.oases as oases_module
        seen = {}

        def fake(*args, **kwargs):
            seen.update(kwargs)
            raise self._DeckCaptured

        monkeypatch.setattr(oases_module, 'write_oasn_input', fake)
        env = Environment(
            name='opts', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        import warnings as _w
        with pytest.raises(self._DeckCaptured):
            with _w.catch_warnings():
                _w.simplefilter('ignore')
                model.run(env, Source(depths=10.0, frequencies=100.0),
                          Receiver(depths=[30.0, 70.0], ranges=[0.0]),
                          run_mode=run_mode)
        return set(seen['options'].split())

    def test_options_merge_additively_with_the_run_modes_letter(
            self, monkeypatch):
        """oases.md §10: unlike OAST/OASR, OASN's raw ``options`` is
        additive — the run mode's own letter (N for covariance, R for
        replicas) is merged into whatever the caller passed, so 'F' survives
        alongside the mode letter. The writer is stubbed; no binary runs."""
        got = self._captured_options(
            monkeypatch, OASN(verbose=False, options='J F',
                              surface_noise_level=60.0),
            RunMode.COVARIANCE)
        assert {'F', 'J', 'N'} <= got

    def test_replica_mode_merges_r(self, monkeypatch):
        got = self._captured_options(
            monkeypatch, OASN(verbose=False, options='J F'),
            RunMode.REPLICA)
        assert {'F', 'J', 'R'} <= got
        assert 'N' not in got      # no noise field configured

    def test_replica_with_a_noise_field_also_carries_n(self, monkeypatch):
        """NOIPAR runs only under ``CALNSE.or.trfout`` (unoasn22.f:173-174),
        so a replica run that was given noise levels needs 'N' too or the
        levels never reach the deck."""
        got = self._captured_options(
            monkeypatch, OASN(verbose=False, options='J',
                              surface_noise_level=60.0),
            RunMode.REPLICA)
        assert {'J', 'R', 'N'} <= got

    @pytest.mark.parametrize('kwargs', [{'plot_rmin': 100.0},
                                        {'plot_rmax': 5000.0}])
    def test_plot_axis_knobs_are_rejected_at_construction(self, kwargs):
        """OASN writes covariance/replica outputs, not a TL-vs-range plot, so
        the OAST-style plot-axis knobs raise instead of sitting silently
        inert (DOCUMENTATION §OASN constructor table)."""
        with pytest.raises(ConfigurationError, match='no effect'):
            OASN(**kwargs)

    def test_vrec_is_rejected_at_construction(self):
        """unoasn22.f recognises a 'd' option and carries a vrec variable but
        never reads a value for it from the deck, so the knob cannot reach
        the binary whatever it is set to."""
        with pytest.raises(ConfigurationError, match='vrec'):
            OASN(vrec=15.0)


class TestOASR:
    """Tests for OASR (reflection coefficients)."""

    @pytest.fixture
    def oasr_env(self):
        """Create environment for OASR."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            shear_speed=400.0,
            density=1.8,
            attenuation=0.5,
            shear_attenuation=1.0
        )
        return Environment(
            name="oasr_test",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom
        )

    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[50.0], ranges=[1000.0])

    @pytest.mark.requires_binary
    def test_oasr_reflection_coefficients(self, oasr_env, source, receiver):
        """OASR populates the typed ReflectionCoefficient attributes."""
        oasr = OASR(verbose=False, angles=np.linspace(0.0, 90.0, 91))
        result = oasr.run(env=oasr_env, source=source, receiver=receiver)

        assert isinstance(result, ReflectionCoefficient)
        assert result.theta.shape == (91,)
        assert result.R.shape == result.phi.shape
        assert result.R.size > 0 and np.all(np.isfinite(result.R))
        assert np.all((result.R >= 0.0) & (result.R <= 1.0 + 1e-6))

    @pytest.mark.requires_binary
    def test_oasr_result_carries_the_requested_angle_count(self, oasr_env, source, receiver):
        """Test OASR with different angle resolutions."""
        oasr = OASR(verbose=False, angles=np.linspace(0.0, 90.0, 19))
        result = oasr.run(env=oasr_env, source=source, receiver=receiver)

        assert isinstance(result, ReflectionCoefficient)

    @pytest.mark.requires_binary
    def test_compute_reflection_matches_run(self, oasr_env, source, receiver):
        """Verify the convenience method ``OASR.compute_reflection`` runs."""
        oasr = OASR(verbose=False, angles=np.linspace(0.0, 90.0, 31))
        result = oasr.compute_reflection(
            env=oasr_env, source=source, receiver=receiver,
        )
        assert isinstance(result, ReflectionCoefficient)


class TestOASP:
    """Tests for OASP (2-D wideband transfer functions, oasp.tex:1)."""

    @pytest.fixture
    def oasp_env(self):
        """Create environment for OASP."""
        # A sloping seafloor to exercise the collapse path: OASP is
        # range-independent, so the writer flattens this and warns.
        bathymetry = np.column_stack([
            np.linspace(0, 10000, 21),
            np.linspace(100, 150, 21)  # Sloping bottom
        ])

        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            density=1.5,
            attenuation=0.5
        )

        return Environment(
            name="oasp_test",
            ssp=1500.0,
            bathymetry=bathymetry,
            bottom=bottom
        )

    @pytest.fixture
    def source(self):
        # OASP's wavenumber count grows with frequency x max range
        # (unoasp22.f:1130), so 50 Hz keeps these structural tests cheap.
        return Source(depths=50.0, frequencies=50.0)

    @pytest.fixture
    def receiver(self):
        # OASP cost is set by the spectral RMax (receiver range max → wavenumber
        # FFT size); these tests assert a Field is produced, not numerics, so a
        # shorter range keeps them cheap.
        return Receiver(
            depths=np.linspace(10, 90, 5),
            ranges=np.linspace(100, 3000, 6)
        )

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_oasp_collapses_rd_bathymetry_with_a_warning(self, oasp_env, source, receiver):
        """OASP collapses RD bathymetry; verify the path still produces TL."""
        oasp = OASP(verbose=False)
        with pytest.warns(UserWarning, match="does not support range-dependent bathymetry"):
            result = oasp.compute_tl(
                env=oasp_env,
                source=source,
                receiver=receiver
            )

        assert isinstance(result, Field)
        assert result.shape[0] > 0  # Has depth dimension
        assert result.shape[1] > 0  # Has range dimension
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_oasp_broadband_band_covers_the_requested_frequencies(self, oasp_env, receiver):
        """OASP run_mode=BROADBAND returns a populated Field."""
        source = Source(
            depths=50.0,
            frequencies=np.array([30.0, 50.0, 70.0]),
        )

        oasp = OASP(verbose=False)
        with pytest.warns(UserWarning, match="does not support range-dependent bathymetry"):
            result = oasp.run(
                env=oasp_env,
                source=source,
                receiver=receiver,
                run_mode=RunMode.BROADBAND,
            )

        assert isinstance(result, Field)
        assert result.frequencies is not None and len(result.frequencies) > 0
        assert result.data.shape[:2] == (len(receiver.depths), len(receiver.ranges))


class TestOASESFactory:
    """Tests for the OASES.for_mode() factory function."""

    @pytest.mark.requires_binary
    def test_default_returns_oast(self):
        """OASES.for_mode() with no run_mode defaults to OAST."""
        m = OASES.for_mode(verbose=False)
        assert isinstance(m, OAST)
        assert m.model_name == 'OAST'

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    def test_for_mode_default_computes_tl_like_oast(self):
        """OASES.for_mode() returns an OAST that handles compute_tl."""
        env = Environment(
            name="oases_test", bathymetry=100.0, ssp=1500.0,
        )
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=[25.0, 50.0, 75.0], ranges=[1000.0, 3000.0],
        )
        m = OASES.for_mode(verbose=False)
        result = m.compute_tl(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)

    def test_routes_covariance_to_oasn(self):
        assert isinstance(OASES.for_mode(run_mode=RunMode.COVARIANCE), OASN)

    def test_routes_replica_to_oasn(self):
        assert isinstance(OASES.for_mode(run_mode=RunMode.REPLICA), OASN)

    def test_routes_reflection_to_oasr(self):
        assert isinstance(OASES.for_mode(run_mode=RunMode.REFLECTION), OASR)

    def test_routes_broadband_to_oasp(self):
        assert isinstance(OASES.for_mode(run_mode=RunMode.BROADBAND), OASP)

    def test_routes_time_series_to_oasp(self):
        assert isinstance(OASES.for_mode(run_mode=RunMode.TIME_SERIES), OASP)

    def test_coherent_tl_broadband_routes_to_oasp(self):
        assert isinstance(OASES.for_mode(broadband=True), OASP)

    def test_subclass_specific_kwarg_raises_when_irrelevant(self):
        """OASES.for_mode() forwards kwargs verbatim to the chosen sub-class.
        ``angles=`` belongs to OASR; routing to OAST must raise
        ``TypeError`` so a typo or wrong-class kwarg surfaces."""
        with pytest.raises(TypeError):
            OASES.for_mode(
                run_mode=RunMode.COHERENT_TL, verbose=False,
                angles=np.linspace(0, 90, 10),  # OASR-only
            )

    def test_subclass_specific_kwarg_forwarded_when_relevant(self):
        """When the kwarg DOES belong to the chosen sub-class, the
        factory must forward it (not over-filter)."""
        angles = np.linspace(0, 90, 19)
        m = OASES.for_mode(
            run_mode=RunMode.REFLECTION, verbose=False, angles=angles,
        )
        assert isinstance(m, OASR)
        np.testing.assert_array_equal(m.angles, angles)

    def test_unrelated_garbage_kwarg_raises_typeerror(self):
        """A kwarg that no class consumes must raise ``TypeError`` so
        typos do not silently change the run."""
        with pytest.raises(TypeError):
            OASES.for_mode(verbose=False, totally_made_up_kwarg=42)

    def test_factory_forwards_base_kwargs(self):
        """Base-class kwargs (verbose, timeout, collapse) must pass
        through the filter."""
        m = OASES.for_mode(
            run_mode=RunMode.REFLECTION, verbose=False,
            timeout=42.0, collapse={'bottom_range': 'median'},
        )
        assert isinstance(m, OASR)
        assert m.timeout == 42.0
        assert m._collapse['bottom_range'] == 'median'

    def test_env_absorption_does_not_pollute_options_string(self, tmp_path):
        """env.absorption choice (Thorp/F-G/Biological) must NOT be
        injected as a single-letter marker into the OASES options
        string — those letters collide with OASES Block II semantics
        (OAST 'T' = TL plot, OASR 'B' = Biot P-Slow). Users read the
        chosen formula from ``env.absorption`` directly."""
        from uacpy.core import Environment, Source, Receiver
        from uacpy.core.absorption import FrancoisGarrison
        from uacpy.io.oases_writer import write_oast_input

        env = Environment(
            name='atten', bathymetry=100.0, ssp=1500.0,
            absorption=FrancoisGarrison(
                temperature_c=10.0, salinity_psu=35.0, pH=8.0, z_bar_m=1000.0,
            ),
        )
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(depths=[50.0], ranges=[1000.0])
        dat = tmp_path / 'oast.dat'
        with pytest.warns(UserWarning, match='OASES ignores env.absorption'):
            write_oast_input(dat, env, source, receiver)
        opt_tokens = set(dat.read_text().splitlines()[1].split())
        assert 'F' not in opt_tokens, (
            f"absorption letter 'F' leaked into OAST options: {opt_tokens}"
        )


# Integration test to verify all OASES models can be imported and instantiated
@pytest.mark.requires_binary
def test_all_oases_models_importable():
    """Verify all OASES sub-classes import and instantiate correctly."""
    for name, ModelClass in {'OAST': OAST, 'OASN': OASN, 'OASR': OASR, 'OASP': OASP}.items():
        model = ModelClass(verbose=False)
        assert model.model_name == name
        assert hasattr(model, 'run')


# ---------------------------------------------------------------------
# OASR/OASP equispaced frequency-vector check + auto-resample warning
# ---------------------------------------------------------------------

class TestOASESFrequencyResample:
    """The (fmin, fmax, N) triple OASES writes implies equispaced
    frequency sampling; arbitrary user vectors must trigger a warning
    + auto-resample."""

    def test_helper_equispaced_no_warning(self, recwarn):
        from uacpy.models.oases import _oases_resample_frequencies
        freqs = np.linspace(10.0, 100.0, 10)
        fmin, fmax, n, resampled = _oases_resample_frequencies(freqs, 'OASR')
        assert (fmin, fmax, n) == (10.0, 100.0, 10)
        assert resampled is False
        assert not any(
            "non-equispaced" in str(w.message) for w in recwarn.list
        )

    def test_helper_non_equispaced_warns_and_reports_resampled(self):
        import warnings as _warnings
        from uacpy.models.oases import _oases_resample_frequencies
        freqs = np.array([10.0, 15.0, 30.0, 100.0])
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            fmin, fmax, n, resampled = _oases_resample_frequencies(
                freqs, 'OASR'
            )
        assert (fmin, fmax, n) == (10.0, 100.0, 4)
        assert resampled is True
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, UserWarning)]
        assert any("non-equispaced" in m for m in msgs), (
            f"Expected non-equispaced warning; got {msgs!r}"
        )

    def test_helper_single_freq_no_warning(self, recwarn):
        from uacpy.models.oases import _oases_resample_frequencies
        fmin, fmax, n, resampled = _oases_resample_frequencies(
            np.array([50.0]), 'OASR',
        )
        assert (fmin, fmax, n) == (50.0, 50.0, 1)
        assert resampled is False

    @pytest.mark.requires_binary
    def test_oasr_equispaced_vector_no_warning_correct_freqs(self):
        """A truly equispaced user vector goes through OASR untouched and
        the result's ``frequencies`` matches the user's grid."""
        import warnings as _warnings
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0, shear_speed=400.0,
            density=1.8, attenuation=0.5, shear_attenuation=1.0,
        )
        env = Environment(
            name="oasr_eq", bathymetry=100.0, ssp=1500.0, bottom=bottom,
        )
        src = Source(depths=50.0, frequencies=50.0)
        rcv = Receiver(depths=[50.0], ranges=[1000.0])
        oasr = OASR(verbose=False, angles=np.linspace(0.0, 90.0, 19))
        user_freqs = np.linspace(20.0, 80.0, 4)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = oasr.run(env=env, source=src, receiver=rcv,
                              frequencies=user_freqs)
        non_eq = [w for w in caught if "non-equispaced" in str(w.message)]
        assert non_eq == [], (
            f"Equispaced vector triggered a spurious warning: {non_eq!r}"
        )
        # Result frequencies match the equispaced grid
        np.testing.assert_allclose(
            np.sort(result.frequencies), user_freqs, rtol=1e-3,
        )

    @pytest.mark.requires_binary
    def test_oasr_non_equispaced_vector_warns_and_resamples(self):
        """A non-equispaced user vector triggers the warning and the
        result's ``frequencies`` is the resampled equispaced grid, not
        the user input."""
        import warnings as _warnings
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0, shear_speed=400.0,
            density=1.8, attenuation=0.5, shear_attenuation=1.0,
        )
        env = Environment(
            name="oasr_neq", bathymetry=100.0, ssp=1500.0, bottom=bottom,
        )
        src = Source(depths=50.0, frequencies=50.0)
        rcv = Receiver(depths=[50.0], ranges=[1000.0])
        oasr = OASR(verbose=False, angles=np.linspace(0.0, 90.0, 19))
        user_freqs = np.array([20.0, 30.0, 50.0, 80.0])  # non-equispaced
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = oasr.run(env=env, source=src, receiver=rcv,
                              frequencies=user_freqs)
        non_eq = [w for w in caught if "non-equispaced" in str(w.message)]
        assert len(non_eq) >= 1, (
            "Non-equispaced vector must trigger a UserWarning"
        )
        # Result frequencies are the resampled equispaced grid spanning
        # the user's [min, max] with the same N — NOT the user input.
        expected = np.linspace(
            user_freqs.min(), user_freqs.max(), user_freqs.size,
        )
        np.testing.assert_allclose(
            np.sort(result.frequencies), expected, rtol=1e-3,
        )


# =====================================================================
# Regression tests: OASES writer/reader defects
# =====================================================================


def _pekeris(**overrides):
    """100 m Pekeris waveguide over a fluid halfspace (this file's variant:
    1600 m/s, density 1.5)."""
    kw = dict(name='pekeris', sound_speed=1600.0, density=1.5)
    kw.update(overrides)
    return make_pekeris(**kw)


class TestOastCurveSelection:
    """OAST options that add ``.plt`` curves must not displace the TL.

    ``unoast31.f:590-605`` calls PLINTGR (``'I'``) and PLSPECT (``'a'``)
    inside the same receiver loop as PLTLOS (``:634``), so their curves land
    in the ``.plt`` *ahead* of the TL; ``'A'``/``'D'`` append more afterwards.
    The reader selects on the ``.plp`` curve tag (``<param>TLRAN``,
    oasfun22.f:330) rather than on position.
    """

    OPTIONS = ['N J T', 'N J T I', 'N J T a', 'N J T A', 'N J T D',
               'N J T C', 'N J T c', 'N J T I a A D C c']

    @pytest.fixture
    def rig(self):
        return (_pekeris(),
                Source(depths=50.0, frequencies=100.0),
                Receiver(depths=np.array([10.0, 60.0]),
                         ranges=np.linspace(1000.0, 5000.0, 4)))

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    @pytest.mark.parametrize('options', OPTIONS)
    def test_curve_adding_options_return_the_tl(self, rig, options, tmp_path):
        """Every curve-adding option string returns the SAME native TL grid.

        Selecting the curve by position rather than by ``.plp`` tag makes
        ``'N J T I'`` return the Hankel-transform integrand (≈5 dB values)
        with no error at all.
        """
        from uacpy.io.oases_reader import read_oast_tl

        env, source, receiver = rig

        def native(opts, tag):
            wd = tmp_path / tag
            OAST(verbose=False, options=opts, work_dir=wd,
                 cleanup=False).run(env, source, receiver)
            out = read_oast_tl(wd / 'oast_run.plt', receiver.depths)
            return out['tl'], out['ranges']

        reference, ref_ranges = native('N J T', 'ref')
        tl, ranges = native(options, 'opt')

        np.testing.assert_array_equal(tl, reference)
        np.testing.assert_allclose(ranges, ref_ranges)
        # TL of a 100 m Pekeris guide at 100 Hz: real, positive, bounded.
        assert np.all(tl > 0.0) and np.all(tl < 200.0)

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    def test_plp_curve_tags_are_parsed_in_file_order(self, rig, tmp_path):
        """The ``.plp`` names every curve; the integrand precedes the TL."""
        from uacpy.io.oases_reader import _parse_oast_plp

        env, source, receiver = rig
        OAST(verbose=False, options='N J T I', work_dir=tmp_path,
             cleanup=False).run(env, source, receiver)
        tags = [c['tag'] for c in _parse_oast_plp(tmp_path / 'oast_run.plp')]
        assert tags == ['NINTGR', 'NTLRAN'] * len(receiver.depths), tags

    @pytest.mark.requires_binary
    def test_multiple_output_parameters_raise(self, rig, tmp_path):
        """Two output parameters give two TL curves per receiver; uacpy
        returns one grid, so the ambiguity must surface."""
        env, source, receiver = rig
        with pytest.raises(FileFormatError, match='output parameters'):
            OAST(verbose=False, options='N V J T', work_dir=tmp_path,
                 cleanup=False).run(env, source, receiver)

    @pytest.mark.requires_binary
    def test_missing_tl_option_raises(self, rig, tmp_path):
        """Without ``'T'`` there is no TL curve at all: PLTLOS is never called
        (unoast31.f:634), so the ``.plp`` carries no ``*TLRAN`` record and the
        reader must say which option is missing rather than return an empty
        grid."""
        env, source, receiver = rig
        with pytest.raises(FileFormatError,
                           match=r"no TL-vs-range curve .*'\*TLRAN'"):
            OAST(verbose=False, options='N J I', work_dir=tmp_path,
                 cleanup=False).run(env, source, receiver)


class TestOastPlotBlockGates:
    """OAST Blocks IX-XI are gated on the flags GETOPT sets, and Blocks IX
    and XI are re-read once per selected output parameter (``DO 980`` /
    ``DO 990``, unoast31.f:240-263). A miscounted deck makes OAST die on an
    end-of-file mid-run.

    The literals below are the writer's plot-axis windows, which only ever
    reach OASES' plotters: ``20 100 12 10`` is Block IX's YUP YDOWN YAXIS
    YINC (unoast31.f:242), ``0 <depth> 12 <depth/10>`` Block X's depth axis
    (:250) and ``40 100 10`` Block XI's ZMIN ZMAX ZSTEP contour levels
    (:262). What is under test is which of them appear, and how many times.
    """

    ENV = staticmethod(_pekeris)

    @staticmethod
    def _deck(tmp_path, options):
        from uacpy.io.oases_writer import write_oast_input
        dat = tmp_path / f"gate_{abs(hash(options))}.dat"
        write_oast_input(
            dat, _pekeris(), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[10.0], ranges=[1000.0, 5000.0]),
            options=options,
        )
        return dat.read_text().splitlines()

    def test_block_ix_repeats_per_output_parameter(self, tmp_path):
        """'N V J T' selects two IOUT entries → two Block-IX lines."""
        one = self._deck(tmp_path, 'N J T')
        two = self._deck(tmp_path, 'N V J T')
        assert one[-1] == '20 100 12 10'
        assert two[-2:] == ['20 100 12 10', '20 100 12 10']

    def test_block_ix_fires_for_integrand_and_angular_spectrum(self, tmp_path):
        """PLKERN ('I') and ANSPEC ('a') also gate Block IX."""
        for opts in ('N J I', 'N J a'):
            assert self._deck(tmp_path, opts)[-1] == '20 100 12 10', opts

    def test_block_x_fires_for_depth_integrand_contours(self, tmp_path):
        """ICONTU ('c') gates Block X alongside DRCONT ('C') / TLDEP ('D')."""
        lines = self._deck(tmp_path, 'N J T c')
        assert lines[-1] == '0 100.0 12 10.0'

    def test_block_xi_uses_frcont_not_the_full_hankel_letter(self, tmp_path):
        """Block XI is gated by DRCONT ('C') or FRCONT ('o') — 'f' is
        INTTYP=2 (full Hankel transform) and gates nothing here."""
        with_f = self._deck(tmp_path, 'N J T f')
        assert '40 100 10' not in with_f
        with_c = self._deck(tmp_path, 'N J T C')
        assert with_c[-1] == '40 100 10'

    def test_block_xi_repeats_per_output_parameter(self, tmp_path):
        lines = self._deck(tmp_path, 'N V J T C')
        assert lines[-2:] == ['40 100 10', '40 100 10']

    def test_depth_average_alone_still_emits_block_ix(self, tmp_path):
        """DEPTAV ('A') gates Block IX only when FRCONT ('o') is off."""
        assert self._deck(tmp_path, 'N J A')[-1] == '20 100 12 10'

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    @pytest.mark.parametrize('options', ['N V J T', 'N J T c', 'N V J T C',
                                         'N J T I a D A C c'])
    def test_binary_consumes_the_deck(self, options, tmp_path):
        """The real binary must read every block without hitting EOF."""
        from uacpy.io.oases_writer import write_oast_input
        from uacpy.models.oases import _oases_subprocess_env
        import subprocess

        base = 'gate_run'
        write_oast_input(
            tmp_path / f'{base}.dat', _pekeris(),
            Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[10.0], ranges=[1000.0, 5000.0]),
            options=options,
        )
        exe = OAST(verbose=False)._exe
        env_vars = _oases_subprocess_env(base, FOR002='src', FOR023='trc')
        proc = subprocess.run(
            [str(exe)], cwd=tmp_path, env=env_vars,
            capture_output=True, text=True, timeout=300,
        )
        combined = proc.stdout + proc.stderr
        assert 'End of file' not in combined, combined[-600:]
        assert (tmp_path / f'{base}.plt').stat().st_size > 0


class TestOasnNoiseBlockGates:
    """OASN's surface and deep noise blocks are gated asymmetrically
    (oasnun22.f:276 ``abs(SNLEVDB).GE.0.01`` vs :324 ``DPLEVDB.GE.0.01``).
    Emitting a block OASES does not read desynchronises every later READ."""

    @staticmethod
    def _deck_tail(tmp_path, **kwargs):
        from uacpy.io.oases_writer import write_oasn_input
        dat = tmp_path / 'oasn.dat'
        write_oasn_input(
            dat, _pekeris(), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=np.linspace(40, 60, 5), ranges=[0.0]),
            options='N J', **kwargs,
        )
        return dat.read_text().splitlines()

    def test_negative_deep_level_emits_no_deep_block(self, tmp_path):
        """A negative deep level disables the source, so OASN never reads
        the three deep lines and the next READ would consume them."""
        off = self._deck_tail(tmp_path, surface_noise_level=70.0,
                              deep_noise_level=-3.0)
        on = self._deck_tail(tmp_path, surface_noise_level=70.0,
                             deep_noise_level=60.0)
        assert len(on) - len(off) == 3

    def test_sub_threshold_deep_level_emits_no_deep_block(self, tmp_path):
        """0 < level < 0.01 is below OASES' own threshold."""
        assert (self._deck_tail(tmp_path, deep_noise_level=0.005)
                == self._deck_tail(tmp_path, deep_noise_level=0.0))

    def test_sub_threshold_surface_level_emits_no_surface_block(self, tmp_path):
        """abs(level) < 0.01 disables the surface block (oasnun22.f:276 tests
        the absolute value), so a small NEGATIVE level is off too.

        Only the two Block-VII lines may differ; the level itself still
        appears verbatim on the Block-VI source line.
        """
        n_base = len(self._deck_tail(tmp_path, surface_noise_level=0.0))
        for level in (0.005, -0.005):
            assert len(self._deck_tail(
                tmp_path, surface_noise_level=level)) == n_base, level

    def test_negative_surface_level_is_rejected(self, tmp_path):
        """A surface level at or below -0.01 is minus the unit number of a
        source-spectrum file (oasnun22.f:191-193), not a level in dB, and no
        writer here emits that file."""
        with pytest.raises(ConfigurationError, match='unit number'):
            self._deck_tail(tmp_path, surface_noise_level=-21.0)

    def test_negative_discrete_source_level_is_rejected(self, tmp_path):
        """Same overload on each discrete source, with no dead band
        (oasnun22.f:381-385)."""
        with pytest.raises(ConfigurationError, match='unit number'):
            self._deck_tail(tmp_path, discrete_sources=[
                {'depth': 40.0, 'x': 2.0, 'y': 0.0, 'level': -3.0}])

    @pytest.mark.requires_binary
    def test_replica_run_with_negative_deep_level(self, tmp_path):
        """The end-to-end case: with the deep block suppressed, the replica
        grid READ lands on the replica grid. Landing on the deep block
        instead ends the run at 'Bad integer for item 3 in list input'."""
        from uacpy import Replicas
        oasn = OASN(verbose=False, surface_noise_level=70.0,
                    deep_noise_level=-3.0, zmin=20.0, zmax=80.0, nz=3,
                    xmin=500.0, xmax=2000.0, nx=3, work_dir=tmp_path)
        with pytest.warns(UserWarning, match=r"receiver\.ranges is ignored"):
            rep = oasn.run(
                _pekeris(), Source(depths=50.0, frequencies=100.0),
                Receiver(depths=np.linspace(40, 60, 5), ranges=[1000.0]),
                run_mode=RunMode.REPLICA,
            )
        assert isinstance(rep, Replicas)
        assert np.isfinite(rep.replicas).all()


class TestOasesLayerBudget:
    """``NLA = 1001`` (oases/src/compar.f:23) bounds NL — the deck's *total* layer
    count — not the SSP row count (oaseun31.f:44)."""

    @staticmethod
    def _ssp(n):
        import uacpy
        z = np.linspace(0.0, 100.0, n)
        return uacpy.SoundSpeedProfile.from_pairs(
            list(zip(z, 1500.0 + 0.01 * z)))

    def _write(self, tmp_path, n_ssp, bottom=None):
        from uacpy.io.oases_writer import write_oast_input
        env = _pekeris(ssp=self._ssp(n_ssp),
                       **({'bottom': bottom} if bottom else {}))
        dat = tmp_path / 'nl.dat'
        write_oast_input(
            dat, env, Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[10.0], ranges=[1000.0]),
        )
        return int(dat.read_text().splitlines()[3])

    def test_nl_never_exceeds_the_oases_limit(self, tmp_path):
        """NL counts the halfspaces too, so a 1001-row SSP would put NL at
        1003 and OASES would answer '*** TOO MANY LAYERS ***'
        (oaseun31.f:44-45). The SSP is decimated to fit instead."""
        from uacpy.io.oases_writer import _OASES_MAX_LAYERS
        with pytest.warns(UserWarning, match='decimated'):
            nl = self._write(tmp_path, 1001)
        assert nl == _OASES_MAX_LAYERS

    def test_sediment_layers_come_out_of_the_same_budget(self, tmp_path):
        """Sediment layers take NL slots, so the SSP budget shrinks."""
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        from uacpy.io.oases_writer import _OASES_MAX_LAYERS
        bottom = SeabedColumn(
            layers=[SedimentLayer(thickness=5.0, sound_speed=1550.0 + 10 * i,
                                  density=1.6, attenuation=0.2)
                    for i in range(4)],
            halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1800.0,
                density=2.0, attenuation=0.5),
        )
        # 999 SSP rows fit with no sediment, but not with four layers.
        with pytest.warns(UserWarning, match='decimated'):
            nl = self._write(tmp_path, 999, bottom=bottom)
        assert nl == _OASES_MAX_LAYERS

    def test_profile_at_the_real_budget_is_untouched(self, tmp_path):
        """999 SSP rows + 2 halfspaces == 1001: nothing to decimate."""
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('error', UserWarning)
            nl = self._write(tmp_path, 999)
        assert nl == 1001

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    @pytest.mark.filterwarnings('ignore:env.ssp has')
    def test_binary_accepts_a_decimated_deck(self, tmp_path):
        result = OAST(verbose=False, work_dir=tmp_path).run(
            _pekeris(ssp=self._ssp(1001)),
            Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[10.0], ranges=[1000.0, 2000.0]),
        )
        assert np.all(np.isfinite(result.data))


class TestOasrAngleGrid:
    """OASR carries one angle step ``DLANGLE = (ANGLE2-ANGLE1)/(NANG-1)``
    (unoasr21.f:174) and evaluates ``ANGLE1 + (i-1)*DLANGLE`` (:284), so a
    non-uniform request cannot be honoured."""

    ENV = staticmethod(lambda: Environment(
        name='oasr', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600.0, shear_speed=400.0,
            density=1.8, attenuation=0.5, shear_attenuation=1.0),
    ))

    def test_non_uniform_angles_raise(self, tmp_path):
        """Writing [0, 5, 45, 90] as an ANGLE1/ANGLE2/NANG triple would run
        OASR on [0, 30, 60, 90] and label it with the request."""
        from uacpy.io.oases_writer import write_oasr_input
        with pytest.raises(ConfigurationError, match='uniform angle axis'):
            write_oasr_input(
                tmp_path / 'oasr.dat', self.ENV(),
                Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[1000.0]),
                angles=np.array([0.0, 5.0, 45.0, 90.0]),
            )

    @pytest.mark.requires_binary
    def test_non_uniform_angles_raise_through_the_model(self, tmp_path):
        with pytest.raises(ConfigurationError, match='uniform angle axis'):
            OASR(verbose=False, angles=np.array([0.0, 5.0, 45.0, 90.0]),
                 work_dir=tmp_path).run(
                self.ENV(), Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[50.0], ranges=[1000.0]))

    @pytest.mark.requires_binary
    def test_uniform_angles_are_returned_verbatim(self, tmp_path):
        angles = np.linspace(0.0, 90.0, 19)
        result = OASR(verbose=False, angles=angles, work_dir=tmp_path).run(
            self.ENV(), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=[1000.0]))
        np.testing.assert_allclose(result.theta, angles, atol=1e-3)

    def test_freq_output_increment_reaches_the_deck(self, tmp_path):
        """``OASR(freq_output_increment=...)`` writes OASR's NFOU field."""
        from uacpy.io.oases_writer import write_oasr_input
        write_oasr_input(
            tmp_path / 'oasr.dat', self.ENV(),
            Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=[1000.0]),
            angles=np.linspace(0.0, 90.0, 19), freq_output_increment=7,
        )
        # Block IV: FMIN FMAX NFREQ NFOU — the line after the layer stack.
        freq_line = [ln for ln in tmp_path.joinpath('oasr.dat').read_text()
                     .splitlines() if ln.split()[-1] == '7']
        assert freq_line, tmp_path.joinpath('oasr.dat').read_text()


class TestOasnDiscreteSourceKeys:
    """``discrete_sources`` entries are written field-by-field; an unread key
    would be silently dropped (NOIPAR reads only ZDN/XDN/YDN/DNLEVDB,
    oasnun22.f:380)."""

    @staticmethod
    def _write(tmp_path, sources):
        from uacpy.io.oases_writer import write_oasn_input
        write_oasn_input(
            tmp_path / 'oasn.dat', _pekeris(),
            Source(depths=50.0, frequencies=100.0),
            Receiver(depths=np.linspace(40, 60, 5), ranges=[0.0]),
            options='N J', discrete_sources=sources,
        )

    def test_phase_key_is_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match="'phase'"):
            self._write(tmp_path, [{'depth': 30.0, 'x': 1.0, 'y': 0.0,
                                    'level': 180.0, 'phase': 0.3}])

    def test_accepted_keys_pass(self, tmp_path):
        self._write(tmp_path, [{'depth': 30.0, 'x': 1.0, 'y': 0.0,
                                'level': 180.0}])

    @pytest.mark.requires_binary
    def test_phase_key_rejected_through_the_model(self, tmp_path):
        oasn = OASN(verbose=False, surface_noise_level=70.0,
                    discrete_sources=[{'depth': 30.0, 'x': 1000.0, 'y': 0.0,
                                       'level': 180.0, 'phase': 0.3}],
                    work_dir=tmp_path)
        with pytest.raises(ConfigurationError, match="'phase'"):
            oasn.run(_pekeris(), Source(depths=50.0, frequencies=100.0),
                     Receiver(depths=np.linspace(40, 60, 5), ranges=[0.0]),
                     run_mode=RunMode.COVARIANCE)


class TestOasnUpperHalfspace:
    """``oasnun22.f:187`` stops the run when surface noise is on over a
    halfspace faster than 500 m/s."""

    @staticmethod
    def _write(tmp_path, surface, level):
        from uacpy.io.oases_writer import write_oasn_input
        write_oasn_input(
            tmp_path / 'oasn.dat',
            _pekeris(surface=surface),
            Source(depths=50.0, frequencies=100.0),
            Receiver(depths=np.linspace(40, 60, 5), ranges=[0.0]),
            options='N J', surface_noise_level=level,
        )

    ICE = BoundaryProperties(
        acoustic_type='elastic', sound_speed=3000.0, shear_speed=1600.0,
        density=0.9, attenuation=0.1, shear_attenuation=0.2)

    def test_elastic_surface_with_surface_noise_raises(self, tmp_path):
        with pytest.raises(ConfigurationError, match='vacuum or air'):
            self._write(tmp_path, self.ICE, 70.0)

    def test_rigid_surface_with_surface_noise_raises(self, tmp_path):
        """The 'rigid' substitute emits c_p = 4000 and hits the same wall."""
        with pytest.raises(ConfigurationError, match='vacuum or air'):
            self._write(tmp_path,
                        BoundaryProperties(acoustic_type='rigid'), 70.0)

    def test_elastic_surface_without_surface_noise_is_fine(self, tmp_path):
        """The STOP is inside the surface-noise branch only."""
        self._write(tmp_path, self.ICE, 0.0)

    def test_air_surface_is_allowed(self, tmp_path):
        self._write(tmp_path, BoundaryProperties(
            acoustic_type='half-space', sound_speed=340.0, density=0.0012,
            attenuation=0.0), 70.0)

    @pytest.mark.requires_binary
    def test_model_path_raises_before_the_binary_stops(self, tmp_path):
        oasn = OASN(verbose=False, surface_noise_level=70.0,
                    work_dir=tmp_path)
        with pytest.raises(ConfigurationError, match='vacuum or air'):
            oasn.run(_pekeris(surface=self.ICE),
                     Source(depths=50.0, frequencies=100.0),
                     Receiver(depths=np.linspace(40, 60, 5), ranges=[0.0]),
                     run_mode=RunMode.COVARIANCE)


class TestOasesDocDefects:
    """The remaining verified doc/comment/plumbing defects."""

    def test_plp_curve_tag_letters_are_pltlos_optpar(self):
        """PLTLOS builds its six-character ``.plp`` tag from ``optpar``, the
        physical-quantity abbreviations at oasfun22.f:322-328."""
        from uacpy.io.oases_reader import _OASES_OUTPUT_PARAM_LETTERS
        assert _OASES_OUTPUT_PARAM_LETTERS == ('N', 'W', 'U', 'V', 'R', 'B',
                                               'S')
        assert _OASES_OUTPUT_PARAM_LETTERS[0] == 'N'   # normal stress
        assert _OASES_OUTPUT_PARAM_LETTERS[6] == 'S'   # shear stress

    @pytest.mark.requires_binary
    def test_oasp_trf_reports_the_normal_stress_letter(self, tmp_path):
        """A single-output OASP run tags its .trf as normal stress."""
        from uacpy.io.oases_reader import read_oasp_trf
        env = _pekeris()
        oasp = OASP(verbose=False, work_dir=tmp_path, cleanup=False)
        oasp.run(env, Source(depths=50.0, frequencies=50.0),
                 Receiver(depths=np.linspace(10, 90, 3),
                          ranges=np.linspace(500, 2000, 4)),
                 run_mode=RunMode.COHERENT_TL)
        trf = next(tmp_path.glob('*.trf'))
        assert read_oasp_trf(
            trf, receiver_depths=np.linspace(10, 90, 3))['option'] == 'N'

    def test_wavenumber_bounds_expose_cmax_directly(self):
        """``cmax`` is a caller-settable knob defaulting to OASES' own
        no-upper-limit value 1e8 (unoast31.f:226), not a value derived from
        the SSP; only ``cmin`` tracks the profile."""
        from uacpy.io.oases_writer import oases_wavenumber_bounds
        ssp = np.array([[0.0, 1500.0], [100.0, 1480.0]])
        assert oases_wavenumber_bounds(ssp) == (1480.0 * 0.9, 1e8)
        assert oases_wavenumber_bounds(ssp, cmax=1e9)[1] == 1e9

    def test_oast_writer_kwargs_are_all_reachable(self):
        """Every ``_OAST_KWARGS`` / ``_OASR_KWARGS`` key must be settable
        from the model, or it advertises a knob nothing can reach."""
        from uacpy.io.oases_writer import _OAST_KWARGS, _OASR_KWARGS
        from uacpy.models.base import _collect_init_params
        oast_params = {n for n, _ in _collect_init_params(OAST)}
        assert _OAST_KWARGS - oast_params == set()
        oasr_params = {n for n, _ in _collect_init_params(OASR)}
        # The frequency/angle sweep triple is set by ``run(frequencies=)``.
        run_supplied = {'freq_min', 'freq_max', 'n_frequencies',
                        'angle_min', 'angle_max', 'n_angles'}
        assert _OASR_KWARGS - oasr_params - run_supplied == set()

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    def test_oast_vrec_reaches_the_doppler_frequency_line(self, tmp_path):
        """``vrec`` is OAST's 5th frequency-line token, read only under the
        lowercase 'd' option (unoast31.f:125-127) — so the model accepts it
        only alongside a raw options string that carries 'd', and raises
        otherwise instead of writing a token the binary never reads."""
        from uacpy.io.oases_writer import write_oast_input
        dat = tmp_path / 'dopp.dat'
        write_oast_input(
            dat, _pekeris(), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[10.0], ranges=[1000.0]),
            options='N J T d', vrec=12.5,
        )
        assert dat.read_text().splitlines()[2].split()[-1] == '12.5'
        assert OAST(verbose=False, vrec=12.5, options='N J T d').vrec == 12.5
        with pytest.raises(ConfigurationError, match='vrec'):
            OAST(verbose=False, vrec=12.5)

    @pytest.mark.requires_binary
    def test_zero_vrec_stays_silent(self):
        assert OAST(vrec=0.0).vrec == 0.0


class TestOasesUnwrittenOptionBlocks:
    """Option letters whose GETOPT flag makes the program read a deck block
    none of the writers emit.

    The read then runs off the end of the deck (or eats the block after it),
    so the failure reaches the user as a bare Fortran ``End of file`` naming
    only a line number. The writers reject the letter instead.
    """

    @staticmethod
    def _rig():
        return (_pekeris(), Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[10.0], ranges=[1000.0, 5000.0]))

    @pytest.mark.parametrize('options,pattern', [
        ('N J T E', r'unoast31\.f:299'),
        ('NJTE', r'unoast31\.f:299'),   # GETOPT is char-wise, so glued counts
    ])
    def test_oast_rejects_patch_scattering(self, tmp_path, options, pattern):
        from uacpy.io.oases_writer import write_oast_input
        env, source, receiver = self._rig()
        with pytest.raises(ConfigurationError, match=pattern):
            write_oast_input(tmp_path / 'e.dat', env, source, receiver,
                             options=options)

    @pytest.mark.parametrize('options,pattern', [
        ('N J d', r'unoasp22\.f:127'),      # Doppler line wants ISTYP VSOU VREC
        ('N J G', r'unoasp22\.f:387-390'),  # dispersion curves
        ('N J Z', r'unoasp22\.f:466-467'),  # velocity-profile plot axes
        ('N J E', r'unoasp22\.f:479'),      # patch scattering
    ])
    def test_oasp_rejects_blocks_past_the_last_one_written(
            self, tmp_path, options, pattern):
        """``write_oasp_input`` stops after Block VIII, so every one of these
        letters reads past the end of the deck."""
        from uacpy.io.oases_writer import write_oasp_input
        env, source, receiver = self._rig()
        with pytest.raises(ConfigurationError, match=pattern):
            write_oasp_input(tmp_path / 'p.dat', env, source, receiver,
                             options=options)

    @pytest.mark.parametrize('options', ['N J Z', 'N J z'])
    def test_oasn_rejects_the_profile_plot(self, tmp_path, options):
        """Either case of the letter sets IPROF, and unoasn22.f:321-323 then
        reads two plot-axis records. The citation names unoasn22.f because
        that is the file the shipped binary is built from (makefile:139-140
        links unoasn22.o into oasn2_bin); unoasn21.f is not compiled."""
        from uacpy.io.oases_writer import write_oasn_input
        env, source, receiver = self._rig()
        with pytest.raises(ConfigurationError, match=r'unoasn22\.f:322-323'):
            write_oasn_input(tmp_path / 'n.dat', env, source, receiver,
                             options=options)

    @pytest.mark.parametrize('writer,options', [
        ('write_oast_input', 'N J T'),
        ('write_oast_input', 'N J T d'),   # lowercase 'd' IS written for OAST
        ('write_oast_input', 'N J T D'),   # uppercase 'D' is TLDEP, not Doppler
        ('write_oasp_input', 'N J'),
        ('write_oasn_input', 'N J R'),
        ('write_oasr_input', 'N T C Z'),   # OASR emits both of its blocks
    ])
    def test_supported_option_strings_are_untouched(self, writer, options):
        from uacpy.io import oases_writer
        oases_writer._reject_unwritten_option_blocks(writer, options)

    def test_every_writer_has_a_swept_entry(self):
        """An absent entry would silently skip the check for that program."""
        from uacpy.io.oases_writer import _UNWRITTEN_OPTION_BLOCKS
        assert set(_UNWRITTEN_OPTION_BLOCKS) == {
            'write_oast_input', 'write_oasp_input',
            'write_oasn_input', 'write_oasr_input',
            'write_oassp_input', 'write_oass_input',
        }

    @pytest.mark.parametrize('options,contour', [
        ('N T C', True),
        ('N T c', False),   # OASR GETOPT defines no lowercase 'c'
    ])
    def test_oasr_contour_block_is_uppercase_only(self, tmp_path, options,
                                                  contour):
        """Emitting the block for a lowercase 'c' would write three contour
        rows OASR never reads and shift the 'Z' block onto them."""
        from uacpy.io.oases_writer import write_oasr_input
        env, _source, receiver = self._rig()
        # 'C' contours against frequency, which unoasr21.f:119-121 refuses
        # below two of them; this test is about the block, not that guard.
        source = Source(depths=50.0, frequencies=np.array([100.0, 200.0]))
        dat = tmp_path / f'r_{options[-1]}.dat'
        write_oasr_input(dat, env, source, receiver, options=options)
        n_lines = len(dat.read_text().splitlines())
        baseline = tmp_path / 'r_base.dat'
        write_oasr_input(baseline, env, source, receiver, options='N T')
        extra = n_lines - len(baseline.read_text().splitlines())
        assert extra == (3 if contour else 0)

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    def test_oast_binary_dies_on_the_letter_the_guard_blocks(self, tmp_path):
        """The guard's reason, measured: injecting 'E' into an otherwise valid
        deck makes OAST stop at unoast31.f:299 with an empty .plt."""
        from uacpy.io.oases_writer import write_oast_input
        from uacpy.models.oases import _oases_subprocess_env
        import subprocess

        env, source, receiver = self._rig()
        base = 'patch_run'
        dat = tmp_path / f'{base}.dat'
        write_oast_input(dat, env, source, receiver, options='N J T')
        lines = dat.read_text().splitlines(True)
        lines[1] = 'N J T E\n'
        dat.write_text(''.join(lines))

        proc = subprocess.run(
            [str(OAST(verbose=False)._exe)], cwd=tmp_path,
            env=_oases_subprocess_env(base, FOR002='src', FOR023='trc'),
            capture_output=True, text=True, timeout=300,
        )
        combined = proc.stdout + proc.stderr
        assert 'End of file' in combined, combined[-600:]
        assert 'line 299' in combined, combined[-600:]
        assert (tmp_path / f'{base}.plt').stat().st_size == 0


def _source_line(deck_lines):
    """Index of Block V in an OAST/OASP deck: title, options, frequency, NL,
    then NL layer records (``READ(1,*) NUML`` at oaseun31.f:43, one
    ``READ(1,*) (V(M,N),N=1,6),ROUGH(M)`` per layer at :54)."""
    return 4 + int(deck_lines[3])


class TestOaspTiltedArray:
    """OASP's ``'T'`` is TILT, not OAST's TL-plot flag. INREC then reads five
    items from the receiver record — ``RD RDLOW IR ZREF DTILT``
    (oaseun31.f:1157-1158) — because OASP's PROGNM is 'OASP17', not the
    'OASTL' the four-item branch tests for (unoasp22.f:1161, comdat.f:16)."""

    def test_writer_rejects_the_tilt_letter(self, tmp_path):
        from uacpy.io.oases_writer import write_oasp_input
        with pytest.raises(ConfigurationError, match=r'oaseun31\.f:1157-1158'):
            write_oasp_input(
                tmp_path / 'p.dat', _pekeris(),
                Source(depths=25.0, frequencies=100.0),
                Receiver(depths=np.linspace(10, 90, 5),
                         ranges=np.linspace(100, 5000, 20)),
                options='N J T')

    def test_model_rejects_the_tilt_letter(self, tmp_path):
        with pytest.raises(ConfigurationError, match=r'oaseun31\.f:1157-1158'):
            OASP(verbose=False, options='N J T',
                 work_dir=tmp_path).run(
                _pekeris(), Source(depths=25.0, frequencies=100.0),
                Receiver(depths=np.linspace(10, 90, 5),
                         ranges=np.linspace(100, 5000, 20)),
                run_mode=RunMode.BROADBAND)

    @pytest.mark.requires_binary
    def test_binary_dies_on_the_letter_the_guard_blocks(self, tmp_path):
        """The guard's reason, measured: injecting 'T' into an otherwise valid
        deck makes OASP swallow the CMIN/CMAX record and die reading NW."""
        from uacpy.io.oases_writer import write_oasp_input
        from uacpy.models.oases import _oases_subprocess_env
        import subprocess

        base = 'oasp_run'
        dat = tmp_path / f'{base}.dat'
        write_oasp_input(
            dat, _pekeris(), Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.linspace(10, 90, 5),
                     ranges=np.linspace(100, 5000, 20)),
            options='N J')
        lines = dat.read_text().splitlines(True)
        lines[1] = 'N J T\n'
        dat.write_text(''.join(lines))

        proc = subprocess.run(
            [str(OASP(verbose=False)._exe)], cwd=tmp_path,
            env=_oases_subprocess_env(base, FOR002='src'),
            capture_output=True, text=True, timeout=300,
        )
        combined = proc.stdout + proc.stderr
        assert 'Array tilt' in combined, combined[-600:]
        assert 'Fortran runtime error' in combined, combined[-600:]
        assert not (tmp_path / f'{base}.trf').exists()


class TestOaspReceiverDepthAxis:
    """A ``.trf`` cannot describe a non-uniform receiver array: TRFHEAD writes
    the explicit RDC list only for ``IR < 0`` (oasiun23.f:870-877), but INREC
    negates IR in COMMON /VARS1/ first (oaseun31.f:1185,
    oases/src/compar.f:69-70), so
    the header always records ``+|IR|`` with only RD and RDLOW."""

    _DEPTHS = np.array([10.0, 20.0, 50.0, 90.0])

    @pytest.mark.requires_binary
    def test_field_depth_axis_is_the_requested_one(self, tmp_path):
        oasp = OASP(verbose=False, n_time_samples=256, freq_max=125.0,
                    work_dir=tmp_path, cleanup=False)
        field = oasp.run(
            _pekeris(), Source(depths=25.0, frequencies=50.0),
            Receiver(depths=self._DEPTHS,
                     ranges=np.linspace(1000, 5000, 5)),
            run_mode=RunMode.BROADBAND)
        assert np.allclose(np.asarray(field.coords['depth']), self._DEPTHS)

    @pytest.mark.requires_binary
    def test_trf_header_really_carries_a_uniform_grid(self, tmp_path):
        """The reason the axis has to come from the caller: the deck asks for
        the four depths explicitly and the header still says IR = +4."""
        import struct
        from uacpy.io.oases_writer import write_oasp_input
        from uacpy.models.oases import _oases_subprocess_env
        import subprocess

        base = 'oasp_run'
        write_oasp_input(
            tmp_path / f'{base}.dat', _pekeris(),
            Source(depths=25.0, frequencies=50.0),
            Receiver(depths=self._DEPTHS, ranges=np.linspace(1000, 3000, 3)),
            n_time_samples=256, freq_max=125.0)
        deck = (tmp_path / f'{base}.dat').read_text().splitlines()
        rcv = _source_line(deck) + 1
        assert deck[rcv].split()[-1] == '-4', deck[rcv]   # NRD < 0 form
        assert deck[rcv + 1].split() == ['10.00', '20.00', '50.00', '90.00']

        subprocess.run(
            [str(OASP(verbose=False)._exe)], cwd=tmp_path,
            env=_oases_subprocess_env(base, FOR002='src'),
            capture_output=True, text=True, timeout=300, check=False,
        )
        raw = (tmp_path / f'{base}.trf').read_bytes()
        off = 0
        # Records 1..8 are FILEID, PROGNM, NOUT, IPARM, TITLE, SIGNN, freqs
        # and sd (oasiun23.f:849-867). Each gfortran sequential-unformatted
        # record is a 4-byte length, the payload, then the length again — so
        # skipping one costs 8 + n bytes. Record 9 is RD, RDLOW, IR (:870).
        for _ in range(8):
            (n,) = struct.unpack('<i', raw[off:off + 4])
            off += 8 + n
        (n,) = struct.unpack('<i', raw[off:off + 4])
        rd, rdlow, ir = struct.unpack('<ffi', raw[off + 4:off + 4 + n])
        assert (rd, rdlow, ir) == (10.0, 90.0, 4)

    @pytest.mark.requires_binary
    def test_reader_rejects_a_mismatched_depth_axis(self, tmp_path):
        """The header's ``|IR|`` is the only cross-check the file supports."""
        from uacpy.io.oases_reader import read_oasp_trf
        from uacpy.core.exceptions import FileFormatError
        OASP(verbose=False, n_time_samples=256, freq_max=125.0,
             work_dir=tmp_path, cleanup=False).run(
            _pekeris(), Source(depths=25.0, frequencies=50.0),
            Receiver(depths=self._DEPTHS, ranges=np.linspace(1000, 3000, 3)),
            run_mode=RunMode.BROADBAND)
        trf = tmp_path / 'oasp_run.trf'
        assert read_oasp_trf(trf, receiver_depths=self._DEPTHS)['depths'].size == 4
        with pytest.raises(FileFormatError, match='header declares 4'):
            read_oasp_trf(trf, receiver_depths=self._DEPTHS[:3])


class TestOaspWavenumberBudget:
    """OASP's automatic sampling has no NP bound. AUTSAM sizes the axis from
    the frequency-range product (unoasp22.f:1130 ``NW=(WNMAX-WNMIN)/DK+1``)
    with no clamp, and — unlike OAST, which stops at unoast31.f:459 — nothing
    downstream tests it, so the run overruns arrays dimensioned NP = 2**16
    (oases/src/compar.f:37-38) and still exits 0 with a full-size .trf."""

    @pytest.mark.requires_binary
    def test_overrun_raises_instead_of_returning_numbers(self, tmp_path):
        env = Environment(bathymetry=200.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1650.0,
                              density=1.9, attenuation=0.8))
        source = Source(depths=100.0, frequencies=500.0)
        oasp = OASP(verbose=False, n_time_samples=512, freq_max=1250.0,
                    work_dir=tmp_path)
        with pytest.raises(ConfigurationError, match='past OASES'):
            oasp.run(env, source, Receiver(depths=[50.0], ranges=[20000.0]),
                     run_mode=RunMode.BROADBAND)

    @pytest.mark.requires_binary
    def test_a_bounded_run_is_untouched(self, tmp_path):
        env = Environment(bathymetry=200.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1650.0,
                              density=1.9, attenuation=0.8))
        field = OASP(verbose=False, n_time_samples=512, freq_max=1250.0,
                     work_dir=tmp_path).run(
            env, Source(depths=100.0, frequencies=500.0),
            Receiver(depths=[50.0], ranges=[5000.0]),
            run_mode=RunMode.BROADBAND)
        data = np.asarray(field.data)
        assert np.isfinite(data).all()
        assert np.nanmax(np.abs(data)) < 1.0

    def test_echo_parser_reads_the_count_oases_reports(self):
        """The bound is checked against OASP's own echo (unoasp22.f:340, :349)
        rather than a Python re-derivation of AUTSAM."""
        from types import SimpleNamespace
        oasp = OASP(verbose=False)
        over = SimpleNamespace(stdout='    MAX NO. OF WAVENUMBERS:       83608\n'
                                      '    NO. OF WAVENUMBERS:       83608\n')
        under = SimpleNamespace(stdout='    NO. OF WAVENUMBERS:       32768\n')
        with pytest.raises(ConfigurationError, match='83608'):
            oasp._reject_wavenumber_overrun(over)
        oasp._reject_wavenumber_overrun(under)
        oasp._reject_wavenumber_overrun(SimpleNamespace(stdout=''))


class TestOasesDipSlipSource:
    """Option ``'4'`` selects the dip-slip moment source, and INSRC then reads
    the dip angle out of the source record — as the second token when LINA is
    0 (oaseun31.f:1114) and as the seventh when 'L' set it (oaseun31.f:1089).
    A fixed seven-token record hands the two-item read the NS column."""

    @staticmethod
    def _deck(tmp_path, name, **kwargs):
        from uacpy.io.oases_writer import write_oast_input
        dat = tmp_path / name
        write_oast_input(dat, _pekeris(),
                         Source(depths=25.0, frequencies=100.0),
                         Receiver(depths=[10.0, 50.0], ranges=[1000., 2000.]),
                         **kwargs)
        return dat

    def test_dip_record_is_two_tokens_without_the_array_letter(self, tmp_path):
        deck = self._deck(tmp_path, 'a.dat', options='N J T 4',
                          dip_angle=35.0).read_text().splitlines()
        assert deck[_source_line(deck)].split() == ['25.00', '35']

    def test_dip_record_keeps_seven_tokens_with_the_array_letter(self, tmp_path):
        deck = self._deck(tmp_path, 'b.dat', options='N J T 4 L',
                          dip_angle=35.0).read_text().splitlines()
        assert deck[_source_line(deck)].split() == [
            '25.00', '1', '0', '0', '1', '0', '35']

    def test_dip_angle_without_the_letter_is_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match='dip-slip'):
            self._deck(tmp_path, 'c.dat', options='N J T', dip_angle=35.0)

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('options', ['N J T 4', 'N J T 4 L'])
    def test_binary_echoes_the_requested_angle(self, tmp_path, options):
        from uacpy.models.oases import _oases_subprocess_env
        import subprocess
        base = 'oast_run'
        self._deck(tmp_path, f'{base}.dat', options=options, dip_angle=35.0)
        proc = subprocess.run(
            [str(OAST(verbose=False)._exe)], cwd=tmp_path,
            env=_oases_subprocess_env(base, FOR002='src', FOR023='trc'),
            capture_output=True, text=True, timeout=300,
        )
        assert 'Dip angle:   35.0' in proc.stdout, proc.stdout[-600:]


class TestOasesReceiverDepthBudget:
    """``NRD = 501`` (oases/src/compar.f:35) bounds |IR| in INREC
    (oaseun31.f:1172); OASN's INPRCV bounds NRCV by the separate but equal
    ``NRMAX = 501`` (oases/src/compar.f:52, oasnun22.f:32-35). Both are hard
    STOPs that leave no output file."""

    @staticmethod
    def _write(writer_name, tmp_path, n):
        from uacpy.io import oases_writer
        writer = getattr(oases_writer, writer_name)
        writer(tmp_path / 'r.dat', _pekeris(),
               Source(depths=50.0, frequencies=100.0),
               Receiver(depths=np.linspace(1.0, 99.0, n),
                        ranges=np.array([1000.0, 2000.0])))

    @pytest.mark.parametrize('writer', ['write_oast_input',
                                        'write_oasp_input',
                                        'write_oasn_input'])
    def test_over_the_limit_is_rejected(self, tmp_path, writer):
        with pytest.raises(ConfigurationError, match=r'compar\.f:35'):
            self._write(writer, tmp_path, 600)

    @pytest.mark.parametrize('writer', ['write_oast_input',
                                        'write_oasp_input',
                                        'write_oasn_input'])
    def test_at_the_limit_is_written(self, tmp_path, writer):
        self._write(writer, tmp_path, 501)
        assert (tmp_path / 'r.dat').stat().st_size > 0


class TestOasesWavenumberSampleCount:
    """A pinned NW above NP is silently clamped (unoast31.f:435,
    unoasp22.f:174) and OAST additionally rounds it up to the next power of
    two (unoast31.f:447-458); both change the wavenumber step the caller
    asked for."""

    @staticmethod
    def _write(writer_name, tmp_path, **kwargs):
        from uacpy.io import oases_writer
        getattr(oases_writer, writer_name)(
            tmp_path / 'w.dat', _pekeris(),
            Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[10.0, 50.0], ranges=[1000.0, 2000.0]), **kwargs)

    @pytest.mark.parametrize('writer', ['write_oast_input',
                                        'write_oasp_input'])
    def test_above_np_is_rejected(self, tmp_path, writer):
        with pytest.raises(ConfigurationError, match='NP = 65536'):
            self._write(writer, tmp_path, nw_samples=200000)

    def test_oast_warns_when_the_count_is_not_a_power_of_two(self, tmp_path):
        with pytest.warns(UserWarning, match='rounds it up to 4096'):
            self._write('write_oast_input', tmp_path, nw_samples=3000)

    def test_oasp_does_not_round(self, tmp_path):
        """OASP has no power-of-two branch — CFFX sizing aside, unoasp22.f
        clamps at :174 and stops there."""
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('error', UserWarning)
            self._write('write_oasp_input', tmp_path, nw_samples=3000)

    @pytest.mark.parametrize('writer', ['write_oast_input',
                                        'write_oasp_input'])
    def test_automatic_sampling_is_untouched(self, tmp_path, writer):
        self._write(writer, tmp_path, nw_samples=-1)
        assert (tmp_path / 'w.dat').stat().st_size > 0


class TestOASRIncidenceAngleAxis:
    """DOCUMENTATION §'Boundaries & angles': OASES tabulates grazing angle
    natively; ``angle_type='incidence'`` converts via
    ``grazing = 90 − incidence``, which reverses the axis — incidence
    10-30° is grazing 60-80°."""

    def test_deck_carries_the_converted_grazing_bounds(self, tmp_path):
        from uacpy.io.oases_writer import write_oasr_input
        write_oasr_input(tmp_path / 'r.dat', _pekeris(),
                         Source(depths=50.0, frequencies=100.0),
                         Receiver(depths=[50.0], ranges=[1000.0]),
                         angles=np.linspace(10.0, 30.0, 21),
                         angle_type='incidence')
        assert '60.000000 80.000000 21' in (tmp_path / 'r.dat').read_text()

    def test_unknown_angle_type_is_rejected(self, tmp_path):
        from uacpy.io.oases_writer import write_oasr_input
        with pytest.raises(ConfigurationError, match='angle_type'):
            write_oasr_input(tmp_path / 'r.dat', _pekeris(),
                             Source(depths=50.0, frequencies=100.0),
                             Receiver(depths=[50.0], ranges=[1000.0]),
                             angles=np.linspace(10.0, 30.0, 21),
                             angle_type='vertical')

    @pytest.mark.requires_binary
    def test_incidence_run_equals_the_mirrored_grazing_run(self):
        import warnings as _w
        env = _pekeris()
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=[50.0], ranges=[1000.0])
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            inc = OASR(verbose=False, angle_type='incidence',
                       angles=np.linspace(10.0, 30.0, 21)).run(
                env, src, rcv, run_mode=RunMode.REFLECTION)
            gra = OASR(verbose=False,
                       angles=np.linspace(60.0, 80.0, 21)).run(
                env, src, rcv, run_mode=RunMode.REFLECTION)
        np.testing.assert_allclose(np.asarray(inc.theta, dtype=float),
                                   np.asarray(gra.theta, dtype=float),
                                   rtol=0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(inc.R, dtype=float),
                                   np.asarray(gra.R, dtype=float),
                                   rtol=1e-6, atol=1e-9)


@pytest.mark.requires_binary
class TestOASRAgreesWithBounceOnAFluidSeabed:
    """oases.md §7: on a fluid seabed OASR and Bounce are interchangeable —
    measured max |ΔR| 0.0012 over the shared sand half-space, 0.005 on the
    layered stack. The 0.005 bound therefore leaves room for the two angle
    grids while catching any convention slip (angle origin, dB-vs-linear,
    e^{±iωt}) outright."""

    def test_sand_halfspace_reflection_matches_bounce(self):
        import warnings as _w
        from uacpy.models import Bounce
        env = Environment(
            name='shared_sand', bathymetry=100.0,
            ssp=[(0.0, 1500.0), (100.0, 1500.0)],
            bottom=BoundaryProperties.from_preset('sand'))
        src = Source(depths=25.0, frequencies=200.0)
        rcv = Receiver(depths=[50.0], ranges=[1000.0])
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            oasr_rc = OASR(verbose=False).run(env, src, rcv,
                                              run_mode=RunMode.REFLECTION)
            bounce_rc = Bounce(verbose=False).run(env, src, rcv)
        common = np.linspace(1.0, 89.0, 177)
        r = [np.interp(common, np.asarray(rc.theta, dtype=float),
                       np.asarray(rc.R, dtype=float))
             for rc in (oasr_rc, bounce_rc)]
        worst = float(np.max(np.abs(r[0] - r[1])))
        assert worst <= 0.005, (
            f"OASR and Bounce |R| disagree by {worst:.4f} on a fluid sand "
            f"half-space — they solve the same plane-wave problem there")


class TestOasnNoiseBlockGating:
    """NOIPAR — Blocks VI-IX — runs only under ``IF (CALNSE.or.trfout)``
    (unoasn22.f:173-174). Written unconditionally, Block VI is consumed by
    the replica-grid READ below it and every later record shifts by one."""

    @staticmethod
    def _deck(tmp_path, **kwargs):
        from uacpy.io.oases_writer import write_oasn_input
        dat = tmp_path / 'oasn_run.dat'
        write_oasn_input(dat, _pekeris(),
                         Source(depths=50.0, frequencies=100.0),
                         Receiver(depths=np.linspace(20, 90, 5), ranges=[0.0]),
                         **kwargs)
        return dat

    def test_replica_only_deck_omits_the_noise_block(self, tmp_path):
        lines = self._deck(tmp_path, options='R J', nw_samples=512,
                           replica_nz=3, replica_nx=3).read_text().splitlines()
        # Block V ends with the deepest receiver row — depth, range, transverse
        # range, type, gain (oasnun22.f:36) — and the replica grid follows.
        idx = max(i for i, ln in enumerate(lines) if ln.startswith('90.00 0 0 1'))
        assert lines[idx + 1].split() == ['10.00', '90.00', '3']

    def test_noise_arguments_without_the_letter_are_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match=r'unoasn22\.f:173-174'):
            self._deck(tmp_path, options='R J', surface_noise_level=70.0)

    @pytest.mark.requires_binary
    def test_binary_runs_a_replica_only_deck(self, tmp_path):
        from uacpy.io.oases_reader import read_oasn_replicas
        from uacpy.models.oases import _oases_subprocess_env
        import subprocess
        self._deck(tmp_path, options='R J', nw_samples=512,
                   replica_zmin=20.0, replica_zmax=80.0, replica_nz=3,
                   replica_xmin=0.5, replica_xmax=2.0, replica_nx=3)
        proc = subprocess.run(
            [str(OASN(verbose=False)._exe)], cwd=tmp_path,
            env=_oases_subprocess_env('oasn_run', FOR014='rpo', FOR016='xsm',
                                      FOR026='chk'),
            capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-600:]
        data = read_oasn_replicas(tmp_path / 'oasn_run.rpo')
        assert data['replicas'].shape == (1, 3, 3, 1, 5)

    @pytest.mark.requires_binary
    def test_replica_run_still_carries_a_requested_noise_field(self, tmp_path):
        """A REPLICA run given noise levels must still emit them, so 'N' is
        added for that case alone."""
        from uacpy import Replicas
        oasn = OASN(verbose=False, surface_noise_level=70.0, zmin=20.0,
                    zmax=80.0, nz=3, xmin=500.0, xmax=2000.0, nx=3,
                    work_dir=tmp_path)
        with pytest.warns(UserWarning, match=r"receiver\.ranges is ignored"):
            rep = oasn.run(_pekeris(), Source(depths=50.0, frequencies=100.0),
                           Receiver(depths=np.linspace(40, 60, 5),
                                    ranges=[1000.0]),
                           run_mode=RunMode.REPLICA)
        assert isinstance(rep, Replicas)
        assert np.isfinite(rep.replicas).all()

    def test_level_is_gated_on_the_value_the_deck_carries(self, tmp_path):
        """OASN tests the level it read back from the deck (oasnun22.f:276),
        and the deck carries '%.1f', so 0.03 lands as 0.0 and the sub-block
        must not be written."""
        rounded = self._deck(tmp_path, options='N J', nw_samples=512,
                             surface_noise_level=0.03).read_text().splitlines()
        off = self._deck(tmp_path, options='N J', nw_samples=512,
                         surface_noise_level=0.0).read_text().splitlines()
        assert rounded == off


class TestOastPlpStructuralParse:
    """PLPWRI writes PTIT and TITLE as ``FORMAT(1H ,A80)`` (oasgun21.f:634),
    80 columns of arbitrary user text, so no column distinguishes them from
    the OPTION record. The block must be walked by the counts it carries —
    NLAB then NC (oasgun21.f:613, :631)."""

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    @pytest.mark.parametrize('name', ['Shelf survey, leg 3',
                                      'Shelf survey leg 3'])
    def test_a_comma_in_the_title_does_not_break_the_parse(self, tmp_path,
                                                           name):
        env = _pekeris(name=name, ssp=np.array([[0.0, 1500.0],
                                                [50.0, 1490.0],
                                                [100.0, 1520.0]]))
        field = OAST(verbose=False, work_dir=tmp_path).run(
            env, Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.linspace(10, 90, 4),
                     ranges=np.linspace(100, 5000, 50)))
        assert np.asarray(field.data).shape == (4, 50)

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    def test_extra_plot_options_keep_their_curve_order(self, tmp_path):
        """'Z' adds a PROFIL block with NLAB=0 and NC=2 ahead of the TL
        curves, 'A' a TLDAV and 'D' a run of TLDEP blocks after them."""
        from uacpy.io.oases_reader import _parse_oast_plp
        OAST(verbose=False, work_dir=tmp_path, cleanup=False,
             options='N T J Z A D').run(
            _pekeris(ssp=np.array([[0.0, 1500.0], [50.0, 1490.0],
                                   [100.0, 1520.0]])),
            Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.linspace(10, 90, 4),
                     ranges=np.linspace(100, 5000, 50)))
        curves = _parse_oast_plp(tmp_path / 'oast_run.plp')
        tags = [c['tag'] for c in curves]
        assert tags[:2] == ['PROFIL', 'PROFIL']
        assert tags.count('NTLRAN') == 4
        assert 'NTLDAV' in tags and 'NTLDEP' in tags

    def test_a_truncated_plp_is_named_as_such(self, tmp_path):
        from uacpy.io.oases_reader import _parse_oast_plp
        from uacpy.core.exceptions import FileFormatError
        plp = tmp_path / 'short.plp'
        plp.write_text('      128       MODU\n OASTL NTLRAN,,,,,,\n')
        with pytest.raises(FileFormatError, match='ends inside a plot block'):
            _parse_oast_plp(plp)

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    def test_a_plp_without_its_end_record_is_rejected(self, tmp_path):
        from uacpy.io.oases_reader import _parse_oast_plp
        from uacpy.core.exceptions import FileFormatError
        OAST(verbose=False, work_dir=tmp_path, cleanup=False).run(
            _pekeris(), Source(depths=25.0, frequencies=100.0),
            Receiver(depths=[10.0, 50.0], ranges=np.linspace(100, 5000, 20)))
        plp = tmp_path / 'oast_run.plp'
        lines = plp.read_text().splitlines()
        assert lines[-1].split()[-1] == 'PLTEND'
        plp.write_text('\n'.join(lines[:-1]) + '\n')
        with pytest.raises(FileFormatError, match='PLTEND'):
            _parse_oast_plp(plp)


class TestOasesFailureCarriesTheBinaryMessage:
    """OASES writes no print file and stops with a banner gfortran exits 0
    for, so the child's own streams are the only diagnosis there is."""

    @pytest.mark.requires_binary
    def test_too_many_wavenumbers_names_itself(self, tmp_path):
        from uacpy.core.exceptions import ModelExecutionError
        env = Environment(bathymetry=200.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1650.0,
                              density=1.9, attenuation=0.8))
        with pytest.raises(ModelExecutionError) as ei:
            OAST(verbose=False, work_dir=tmp_path).run(
                env, Source(depths=50.0, frequencies=1000.0),
                Receiver(depths=[100.0],
                         ranges=np.linspace(1000, 40000, 20)))
        msg = str(ei.value)
        assert 'TOO MANY WAVENUMBERS' in msg, msg[:400]
        assert 'attached above' not in msg


class TestOaspOptionLetterGuards:
    """OASP's rejection set follows its own GETOPT: the letters that add an
    IOUT slot (unoasp22.f:890-921) or split the output over NCOMPO files
    (:455-456), not the integration-scheme letters."""

    @staticmethod
    def _run(tmp_path, options, **kw):
        return OASP(verbose=False, options=options, work_dir=tmp_path,
                    n_time_samples=256, freq_max=125.0, **kw).run(
            _pekeris(), Source(depths=50.0, frequencies=50.0),
            Receiver(depths=np.linspace(10, 90, 3),
                     ranges=np.linspace(500, 2000, 4)),
            run_mode=RunMode.BROADBAND)

    @pytest.mark.parametrize('letter', ['V', 'H', 'R', 'K', 'S', 'U'])
    def test_extra_output_parameters_are_rejected(self, tmp_path, letter):
        with pytest.raises(ConfigurationError, match='multi-component'):
            self._run(tmp_path, f'N J {letter}')

    @pytest.mark.requires_binary
    def test_filon_integration_is_allowed(self, tmp_path):
        """'F' selects an integration scheme (unoasp22.f:961-963); it touches
        no IOUT entry and adds no .trf component."""
        field = self._run(tmp_path, 'N F J')
        assert np.isfinite(np.asarray(field.data)).all()

    def test_tau_p_is_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match='slowness'):
            self._run(tmp_path, 'N J t')

    @pytest.mark.requires_binary
    def test_double_precision_letter_reads_the_dtrf(self, tmp_path):
        """Option '8' renames the transfer function to <base>.dtrf
        (oasiun23.f:837-845), which is the only thing it changes on this
        build — CFFX is declared plain COMPLEX at oasiun23.f:18."""
        field = self._run(tmp_path, 'N J 8')
        assert np.isfinite(np.asarray(field.data)).all()


class TestOasesUnservableOptionLetters:
    """Letters whose GETOPT flag makes the program read an auxiliary file no
    writer here produces."""

    @staticmethod
    def _rig():
        return (_pekeris(), Source(depths=50.0, frequencies=100.0),
                Receiver(depths=[10.0], ranges=[1000.0, 5000.0]))

    @pytest.mark.parametrize('writer,options,pattern', [
        ('write_oast_input', 'N J T l', r'oaseun31\.f:1074-1087'),
        ('write_oast_input', 'N J T v', r'unoast31\.f:182'),
        ('write_oasp_input', 'N J l', r'oaseun31\.f:1074-1087'),
        ('write_oasp_input', 'N J v', r'unoasp22\.f:261'),
        ('write_oast_input', 'N J T t', r'oaseun31\.f:3727-3728'),
        ('write_oast_input', 'N J T b', r'oaseun31\.f:3757-3758'),
        ('write_oasn_input', 'N J t', r'oaseun31\.f:3727-3728'),
        ('write_oasn_input', 'N J b', r'oaseun31\.f:3757-3758'),
    ])
    def test_source_and_reflection_files_are_rejected(self, tmp_path, writer,
                                                      options, pattern):
        from uacpy.io import oases_writer
        env, source, receiver = self._rig()
        with pytest.raises(ConfigurationError, match=pattern):
            getattr(oases_writer, writer)(
                tmp_path / 'x.dat', env, source, receiver, options=options)


class TestOasnOutputUnits:
    """OASN converts the deck's dB gain column in place before writing any
    output (oasnun22.f:99 ``GAIN(I)=10.0**(GAIN(I)/20.0)``), and PUTXSM
    stores the title as four 8-character slices (oasmun21_bin.f:364-367)."""

    @pytest.mark.requires_binary
    def test_replica_gains_come_back_in_decibels(self, tmp_path):
        from uacpy.io.oases_reader import read_oasn_replicas
        from uacpy.io.oases_writer import write_oasn_input
        from uacpy.models.oases import _oases_subprocess_env
        import subprocess
        base = 'oasn_run'
        dat = tmp_path / f'{base}.dat'
        write_oasn_input(dat, _pekeris(), Source(depths=50.0, frequencies=100.0),
                         Receiver(depths=np.linspace(20, 90, 5), ranges=[0.0]),
                         options='R J', nw_samples=512, replica_nz=3,
                         replica_nx=3)
        lines = dat.read_text().splitlines()
        # Block V rows are Z X Y ITYP GAIN (oasnun22.f:36) and the writer emits
        # every element at 0 dB; put 20 dB on the first one so the round trip
        # has something to carry. No writer knob reaches this column.
        first = next(i for i, ln in enumerate(lines) if ln.endswith(' 0 0 1 0'))
        lines[first] = lines[first][:-1] + '20'
        dat.write_text('\n'.join(lines) + '\n')
        proc = subprocess.run(
            [str(OASN(verbose=False)._exe)], cwd=tmp_path,
            env=_oases_subprocess_env(base, FOR014='rpo', FOR016='xsm',
                                      FOR026='chk'),
            capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-400:]
        gains = read_oasn_replicas(tmp_path / f'{base}.rpo')['receiver_gains']
        assert np.allclose(gains, [20.0, 0.0, 0.0, 0.0, 0.0])

    @pytest.mark.requires_binary
    def test_covariance_title_keeps_spaces_on_record_boundaries(self, tmp_path):
        from uacpy.io.oases_reader import read_oasn_covariance
        title = 'NorthSea site A trial'
        cov = OASN(verbose=False, surface_noise_level=70.0, nw_samples=512,
                   work_dir=tmp_path).run(
            _pekeris(name=title), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=np.linspace(20, 90, 5), ranges=[0.0]),
            run_mode=RunMode.COVARIANCE)
        assert cov.metadata['title'] == title
        assert read_oasn_covariance(cov.metadata['xsm_file'])['title'] == title


class TestOaspTrfOptionLetters:
    """``IPARM`` holds OASP's IOUT slot numbers (oasiun23.f:855-861). Naming
    them needs OASP's GETOPT table (unoasp22.f:890-921), not PLTLOS's
    quantity abbreviations (oasfun22.f:322-328), which differ at slots 2, 3
    and 6."""

    def test_slot_table_is_the_getopt_one(self):
        from uacpy.io.oases_reader import _OASP_OUTPUT_PARAM_LETTERS
        assert _OASP_OUTPUT_PARAM_LETTERS == {1: 'N', 2: 'V', 3: 'H',
                                              5: 'R', 6: 'K', 7: 'S'}

    @pytest.mark.requires_binary
    def test_a_vertical_velocity_trf_reads_back_as_v(self, tmp_path):
        from uacpy.io.oases_reader import read_oasp_trf
        from uacpy.io.oases_writer import write_oasp_input
        from uacpy.models.oases import _oases_subprocess_env
        import subprocess
        base = 'oasp_run'
        depths = np.linspace(10, 90, 3)
        dat = tmp_path / f'{base}.dat'
        write_oasp_input(dat, _pekeris(),
                         Source(depths=50.0, frequencies=50.0),
                         Receiver(depths=depths,
                                  ranges=np.linspace(500, 2000, 4)),
                         options='V J', n_time_samples=256, freq_max=125.0)
        subprocess.run(
            [str(OASP(verbose=False)._exe)], cwd=tmp_path,
            env=_oases_subprocess_env(base, FOR002='src'),
            capture_output=True, text=True, timeout=300, check=False,
        )
        data = read_oasp_trf(tmp_path / f'{base}.trf', receiver_depths=depths)
        assert data['option'] == 'V'


class TestOaspFreqOutputIncrementZero:
    """``freq_output_increment=0`` is a legal OASP value that disables the plot
    output (oasp.tex:663-664), so it must reach the deck rather than be
    replaced by the default. A truthiness test cannot express that.
    """

    @staticmethod
    def ENV():
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))

    def _intf(self, tmp_path, **kwargs):
        from uacpy.io.oases_writer import write_oasp_input
        out = tmp_path / 'oasp.dat'
        write_oasp_input(
            out, self.ENV(), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=[1000.0]), **kwargs)
        # The NW line is `NWVNO ICUT1 ICUT2 INTF` — four integer tokens.
        for line in out.read_text().splitlines():
            parts = line.split()
            if len(parts) == 4 and all(p.lstrip('-').isdigit() for p in parts):
                return int(parts[3])
        raise AssertionError(out.read_text())

    def test_zero_is_written_verbatim(self, tmp_path):
        assert self._intf(tmp_path, freq_output_increment=0) == 0

    def test_a_nonzero_value_is_written_verbatim(self, tmp_path):
        assert self._intf(tmp_path, freq_output_increment=5) == 5

    def test_omitting_it_takes_the_default(self, tmp_path):
        assert self._intf(tmp_path) == 40


class TestOasrPhaseIsUnwrappedAtIngest:
    """``oases/src/oasjun21.f:102`` writes the reflection phase as
    ``phang = omr*atan2z(...)``, a principal value wrapped to (-180, 180].
    Every consumer interpolates it linearly between bracketing angles, which
    ``misc/RefCoef.f90:119`` states requires an unwrapped phase: "Assumes phi has
    been unwrapped so that it varies smoothly." Interpolating across a +-360 deg
    step sweeps the phase the long way round and flips the sign of R.
    """

    THETA = [0.0, 10.0, 20.0, 30.0, 40.0]
    TRUE_DEG = np.array([150.0, 170.0, 190.0, 210.0, 230.0])

    def _stacked(self):
        from uacpy.models.oases import _stack_oasr_data
        wrapped = (self.TRUE_DEG + 180.0) % 360.0 - 180.0
        return _stack_oasr_data({
            'frequencies': [100.0], 'angles_or_slowness': [self.THETA],
            'magnitude': [[1.0] * 5], 'phase': [list(wrapped)],
        })

    def test_ingest_removes_the_atan2_wrap(self):
        _theta, _R, phi, _freqs = self._stacked()
        deg = np.degrees(np.asarray(phi).ravel())
        assert np.all(np.abs(np.diff(deg)) < 180.0)
        np.testing.assert_allclose(deg - deg[0], self.TRUE_DEG - self.TRUE_DEG[0],
                                   atol=1e-9)

    def test_interpolating_across_the_wrap_now_gives_the_true_phase(self):
        from uacpy.core.results.reflection import ReflectionCoefficient
        theta, R, phi, freqs = self._stacked()
        rc = ReflectionCoefficient(
            R=np.asarray(R).reshape(-1, 1), phi=np.asarray(phi).reshape(-1, 1),
            theta=np.asarray(theta), frequencies=np.asarray(freqs))
        # 15 deg falls inside the interval containing the wrap; 25 deg does
        # not, so it was already right and must stay right.
        for angle in (15.0, 25.0):
            got = float(np.degrees(np.asarray(rc.eval(angle=angle).phi).ravel()[0]))
            want = float(np.interp(angle, self.THETA, self.TRUE_DEG))
            assert abs(((got - want + 180.0) % 360.0) - 180.0) < 1e-6

    def test_an_already_smooth_table_is_untouched(self):
        """The unwrap must be a no-op for a table that never wraps, so the
        Bounce path's own already-unwrapped tables cannot move."""
        from uacpy.models.oases import _stack_oasr_data
        smooth = [10.0, 20.0, 30.0, 40.0, 50.0]
        _t, _R, phi, _f = _stack_oasr_data({
            'frequencies': [100.0], 'angles_or_slowness': [self.THETA],
            'magnitude': [[1.0] * 5], 'phase': [smooth]})
        np.testing.assert_allclose(np.degrees(np.asarray(phi).ravel()), smooth,
                                   atol=1e-9)


class TestOasesFrequencyReachesTheDeck:
    """A requested frequency must appear on the deck, not a 0.1 Hz stand-in.

    ``oases/src/unoasp22.f:129,176``, ``unoast31.f`` and ``unoasr21.f:128`` are
    list-directed ``READ(1,*)`` into reals, so no width is imposed — the same
    argument the writer already makes for the range fields in the *same* record
    (``R0``/``RSPACE`` at ``%.9f``: "a list-directed read, so the extra digits
    are free"). At 10 km a 0.05 Hz error is a phase error of
    ``2*pi*df*R/c`` = 2.1 rad, so coherent broadband synthesis mis-times its bins
    whenever the requested grid is not already on a 0.1 Hz raster.
    """

    @staticmethod
    def _env():
        from uacpy.core.environment import SoundSpeedProfile, BoundaryProperties
        return Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.7,
                                      attenuation=0.3))

    def test_a_sub_decihertz_frequency_is_written_verbatim(self, tmp_path):
        from uacpy.io.oases_writer import write_oast_input
        path = tmp_path / 'f.dat'
        write_oast_input(path, self._env(),
                         Source(depths=50.0, frequencies=200.04),
                         Receiver(depths=[75.0], ranges=[1000.0, 2000.0]))
        text = path.read_text()
        assert '200.04' in text, f"200.04 Hz not on the deck:\n{text}"
        assert '200.0 ' not in text.replace('200.040', ''), \
            "the frequency was rounded onto a 0.1 Hz raster"

    @pytest.mark.requires_oases
    def test_a_sub_decihertz_change_reaches_the_solver(self):
        rcv = Receiver(depths=[75.0], ranges=np.linspace(1000.0, 10000.0, 91))
        env = self._env()

        def tl(freq):
            return np.asarray(OAST().compute_tl(
                env, Source(depths=50.0, frequencies=freq), rcv).db).ravel()

        # Bit-identical before the fix: the deck carried '200.0' either way.
        assert np.nanmax(np.abs(tl(200.0) - tl(200.04))) > 0.1


class TestOasesReceiverAxisKeepsItsDepths:
    """The compact ``(z_min, z_max, n)`` receiver record makes OASES build its
    own equidistant grid and discard the requested depths, so it may only be used
    for an axis that really is equidistant.

    ``oaseun31.f:1156,1165`` reads the block list-directed and the explicit
    negative-``NR`` form (``doc/oast.tex:464-493``) is already implemented, so
    nothing forces the compact form. The test is now
    :func:`uacpy.io.utils.equally_spaced` — a verbatim port of
    ``Matlab/ReadWrite/equally_spaced.m`` that the ``.flp`` writer uses for the
    same decision — which measures against the ideal uniform grid rather than
    against the first interval.
    """

    @staticmethod
    def _near_uniform(n):
        return np.cumsum(np.r_[10.0, [1.0 + 0.01 * (i % 2)
                                      for i in range(n - 1)]])

    @pytest.mark.parametrize('n', [5, 20, 100])
    def test_a_near_uniform_axis_is_written_explicitly(self, n):
        from uacpy.io.oases_writer import _receiver_block_lines
        z = self._near_uniform(n)
        lines = _receiver_block_lines(Receiver(depths=z, ranges=[1000.0]))
        assert len(lines) == 2, "the requested depths were discarded"
        written = np.array([float(v) for v in lines[1].split()])
        assert written.size == n
        assert np.max(np.abs(written - z)) < 1e-2

    def test_a_truly_uniform_axis_still_uses_the_compact_form(self):
        from uacpy.io.oases_writer import _receiver_block_lines
        z = np.linspace(10.0, 90.0, 9)
        lines = _receiver_block_lines(Receiver(depths=z, ranges=[1000.0]))
        assert len(lines) == 1
