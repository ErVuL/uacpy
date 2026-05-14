"""
Comprehensive tests for OASES models

OASES (Ocean Acoustics and Seismic Exploration System) is a suite of
seismo-acoustic models for underwater acoustics. This test file provides
systematic validation of all OASES variants:

- OAST: Transmission loss computation
- OASN: Normal modes extraction
- OASR: Reflection coefficients
- OASP: Pulse / broadband wavenumber-integration transfer functions
"""

import pytest
import numpy as np

from uacpy.models import OAST, OASN, OASR, OASP, OASES, RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.results import Field, ReflectionCoefficient

pytestmark = pytest.mark.requires_oases


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
    def test_oast_instantiation(self):
        """Test creating OAST instance."""
        oast = OAST(verbose=False)
        assert oast.model_name == 'OAST'

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
        # TL values should be positive (loss)
        finite_data = result.data[np.isfinite(result.data)]
        if len(finite_data) > 0:
            assert np.all(finite_data > 0), "TL should be positive"

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


class TestOASN:
    """Tests for OASN (normal modes)."""

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
    def test_oasn_instantiation(self):
        """Test creating OASN instance."""
        oasn = OASN(verbose=False)
        assert oasn.model_name == 'OASN'

    @pytest.mark.requires_binary
    def test_oasn_compute_covariance(self, oasn_env, source, receiver):
        """OASN.compute_covariance returns a populated Covariance result."""
        from uacpy import Covariance
        oasn = OASN(verbose=False)
        cov = oasn.compute_covariance(env=oasn_env, source=source, receiver=receiver)
        assert isinstance(cov, Covariance)
        assert isinstance(cov, Covariance)
        assert cov.covariance.ndim == 3
        assert cov.covariance.shape[1] == cov.covariance.shape[2] == cov.n_receivers
        assert cov.n_frequencies >= 1

    @pytest.mark.requires_binary
    def test_oasn_compute_replicas_helper(self, oasn_env, source):
        """Verify ``OASN.compute_replicas`` runs and returns a Replicas object."""
        from uacpy import Replicas
        rcv_array = Receiver(
            depths=np.linspace(40.0, 60.0, 5),
            ranges=[1000.0],
        )
        oasn = OASN(verbose=False)
        rep = oasn.compute_replicas(
            env=oasn_env, source=source, receiver=rcv_array,
            replica_zmin=20.0, replica_zmax=80.0, replica_nz=4,
            replica_xmin=500.0, replica_xmax=2000.0, replica_nx=4,  # metres
        )
        assert isinstance(rep, Replicas)
        assert isinstance(rep, Replicas)
        # replica_x axis must be in metres (uacpy public-API convention).
        assert rep.replica_x.min() >= 500.0 - 1.0
        assert rep.replica_x.max() <= 2000.0 + 1.0

    @pytest.mark.requires_binary
    def test_oasn_elastic_covariance(self, source, receiver):
        """OASN accepts an elastic-bottom env and returns a Covariance."""
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

        oasn = OASN(verbose=False)
        cov = oasn.compute_covariance(env=env, source=source, receiver=receiver)
        assert isinstance(cov, Covariance)


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
    def test_oasr_instantiation(self):
        """Test creating OASR instance."""
        oasr = OASR(verbose=False)
        assert oasr.model_name == 'OASR'

    @pytest.mark.requires_binary
    def test_oasr_reflection_coefficients(self, oasr_env, source, receiver):
        """OASR populates the typed ReflectionCoefficient attributes."""
        from uacpy import ReflectionCoefficient
        oasr = OASR(verbose=False, angles=np.linspace(0.0, 90.0, 91))
        result = oasr.run(env=oasr_env, source=source, receiver=receiver)

        assert isinstance(result, ReflectionCoefficient)
        assert isinstance(result, ReflectionCoefficient)
        assert result.theta.shape == (91,)
        assert result.R.shape == result.phi.shape
        assert result.R.size > 0 and np.all(np.isfinite(result.R))
        assert np.all((result.R >= 0.0) & (result.R <= 1.0 + 1e-6))

    @pytest.mark.requires_binary
    def test_oasr_angle_resolution(self, oasr_env, source, receiver):
        """Test OASR with different angle resolutions."""
        oasr = OASR(verbose=False, angles=np.linspace(0.0, 90.0, 19))
        result = oasr.run(env=oasr_env, source=source, receiver=receiver)

        assert isinstance(result, ReflectionCoefficient)

    @pytest.mark.requires_binary
    def test_oasr_compute_reflection_helper(self, oasr_env, source, receiver):
        """Verify the convenience method ``OASR.compute_reflection`` runs."""
        oasr = OASR(verbose=False, angles=np.linspace(0.0, 90.0, 31))
        result = oasr.compute_reflection(
            env=oasr_env, source=source, receiver=receiver,
        )
        assert isinstance(result, ReflectionCoefficient)


class TestOASP:
    """Tests for OASP (parabolic equation)."""

    @pytest.fixture
    def oasp_env(self):
        """Create environment for OASP."""
        # OASP handles range-dependent scenarios well
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
        return Source(depths=50.0, frequencies=50.0)  # Lower frequency for PE

    @pytest.fixture
    def receiver(self):
        return Receiver(
            depths=np.linspace(10, 90, 9),
            ranges=np.linspace(100, 10000, 11)
        )

    @pytest.mark.requires_binary
    def test_oasp_instantiation(self):
        """Test creating OASP instance."""
        oasp = OASP(verbose=False)
        assert oasp.model_name == 'OASP'

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_oasp_range_dependent(self, oasp_env, source, receiver):
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
    def test_oasp_broadband(self, oasp_env, receiver):
        """OASP run_mode=BROADBAND returns a populated Field."""
        from uacpy.core.results import Field
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
        assert isinstance(result, Field)
        assert result.frequencies is not None and len(result.frequencies) > 0
        assert result.data.shape[:2] == (len(receiver.depths), len(receiver.ranges))


class TestOASESFactory:
    """Tests for the OASES() factory function."""

    @pytest.mark.requires_binary
    def test_default_returns_oast(self):
        """OASES() with no run_mode defaults to OAST."""
        m = OASES(verbose=False)
        assert isinstance(m, OAST)
        assert m.model_name == 'OAST'

    @pytest.mark.requires_binary
    def test_factory_compute_tl(self):
        """OASES() returns an OAST that handles compute_tl."""
        env = Environment(
            name="oases_test", bathymetry=100.0, ssp=1500.0,
        )
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=[25.0, 50.0, 75.0], ranges=[1000.0, 3000.0],
        )
        m = OASES(verbose=False)
        result = m.compute_tl(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)

    def test_routes_covariance_to_oasn(self):
        assert isinstance(OASES(run_mode=RunMode.COVARIANCE), OASN)

    def test_routes_replica_to_oasn(self):
        assert isinstance(OASES(run_mode=RunMode.REPLICA), OASN)

    def test_routes_reflection_to_oasr(self):
        assert isinstance(OASES(run_mode=RunMode.REFLECTION), OASR)

    def test_routes_broadband_to_oasp(self):
        assert isinstance(OASES(run_mode=RunMode.BROADBAND), OASP)

    def test_routes_time_series_to_oasp(self):
        assert isinstance(OASES(run_mode=RunMode.TIME_SERIES), OASP)

    def test_coherent_tl_broadband_routes_to_oasp(self):
        assert isinstance(OASES(broadband=True), OASP)

    def test_subclass_specific_kwarg_raises_when_irrelevant(self):
        """OASES() forwards kwargs verbatim to the chosen sub-class.
        ``angles=`` belongs to OASR; routing to OAST must raise
        ``TypeError`` so a typo or wrong-class kwarg surfaces."""
        with pytest.raises(TypeError):
            OASES(
                run_mode=RunMode.COHERENT_TL, verbose=False,
                angles=np.linspace(0, 90, 10),  # OASR-only
            )

    def test_subclass_specific_kwarg_forwarded_when_relevant(self):
        """When the kwarg DOES belong to the chosen sub-class, the
        factory must forward it (not over-filter)."""
        angles = np.linspace(0, 90, 19)
        m = OASES(
            run_mode=RunMode.REFLECTION, verbose=False, angles=angles,
        )
        assert isinstance(m, OASR)
        np.testing.assert_array_equal(m.angles, angles)

    def test_unrelated_garbage_kwarg_raises_typeerror(self):
        """A kwarg that no class consumes must raise ``TypeError`` so
        typos do not silently change the run."""
        with pytest.raises(TypeError):
            OASES(verbose=False, totally_made_up_kwarg=42)

    def test_factory_forwards_base_kwargs(self):
        """Base-class kwargs (verbose, timeout, collapse) must pass
        through the filter."""
        m = OASES(
            run_mode=RunMode.REFLECTION, verbose=False,
            timeout=42.0, collapse={'bottom': 'median'},
        )
        assert isinstance(m, OASR)
        assert m.timeout == 42.0
        assert m._collapse['bottom'] == 'median'

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
    + auto-resample (Finding #14)."""

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
