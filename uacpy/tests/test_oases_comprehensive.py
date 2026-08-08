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
from uacpy.core.exceptions import ConfigurationError, FileFormatError
from uacpy.core.results import Field, ReflectionCoefficient

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
        """OASN.compute_covariance returns a populated Covariance result.

        ``receiver`` carries a non-zero range, which OASN ignores (it
        models a vertical array at x = y = 0) — so the call must warn.
        """
        from uacpy import Covariance
        oasn = OASN(verbose=False)
        with pytest.warns(UserWarning, match=r"receiver\.ranges is ignored"):
            cov = oasn.compute_covariance(
                env=oasn_env, source=source, receiver=receiver,
            )
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
        with pytest.warns(UserWarning, match=r"receiver\.ranges is ignored"):
            cov = oasn.compute_covariance(
                env=env, source=source, receiver=receiver,
            )
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
        oasr = OASR(verbose=False, angles=np.linspace(0.0, 90.0, 91))
        result = oasr.run(env=oasr_env, source=source, receiver=receiver)

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
        # OASP cost is set by the spectral RMax (receiver range max → wavenumber
        # FFT size); these tests assert a Field is produced, not numerics, so a
        # shorter range keeps them cheap.
        return Receiver(
            depths=np.linspace(10, 90, 5),
            ranges=np.linspace(100, 3000, 6)
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
    """Tests for the OASES.for_mode() factory function."""

    @pytest.mark.requires_binary
    def test_default_returns_oast(self):
        """OASES.for_mode() with no run_mode defaults to OAST."""
        m = OASES.for_mode(verbose=False)
        assert isinstance(m, OAST)
        assert m.model_name == 'OAST'

    @pytest.mark.requires_binary
    @pytest.mark.filterwarnings(_OAST_INTERP_FILTER)
    def test_factory_compute_tl(self):
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


# =====================================================================
# Regression tests: OASES writer/reader defects
# =====================================================================


def _pekeris(**overrides):
    """100 m Pekeris waveguide over a fluid halfspace."""
    kw = dict(
        name='pekeris', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600.0,
            density=1.5, attenuation=0.5,
        ),
    )
    kw.update(overrides)
    return Environment(**kw)


class TestOastCurveSelection:
    """OAST options that add ``.plt`` curves must not displace the TL.

    ``unoast31.f:590-605`` calls PLINTGR (``'I'``) and PLSPECT (``'a'``)
    inside the same receiver loop as PLTLOS, so their curves land in the
    ``.plt`` *ahead* of the TL; ``'A'``/``'D'`` append more afterwards.
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
            tl, _d, ranges, _m = read_oast_tl(
                wd / 'oast_run.plt', receiver.depths)
            return tl, ranges

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
        """Without ``'T'`` there is no TL curve at all."""
        env, source, receiver = rig
        with pytest.raises((FileFormatError, Exception), match='T'):
            OAST(verbose=False, options='N J I', work_dir=tmp_path,
                 cleanup=False).run(env, source, receiver)


class TestOastPlotBlockGates:
    """OAST Blocks IX-XI are gated on the flags GETOPT sets, and Blocks IX
    and XI are re-read once per selected output parameter (``DO 980`` /
    ``DO 990``, unoast31.f:240-263). A miscounted deck makes OAST die on an
    end-of-file mid-run."""

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
        """A negative deep level disables the source; the three deep lines
        must not be written (they were eaten by the next READ)."""
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
        """abs(level) < 0.01 disables the surface block — including a small
        NEGATIVE level, which ``!= 0`` wrongly treated as enabled.

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
        """The end-to-end case: OASN aborted with 'Bad integer for item 3 in
        list input' because the replica grid READ consumed the deep block."""
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
        """A 1001-row SSP wrote NL=1003 and OASES answered
        '*** TOO MANY LAYERS ***' with no warning at all."""
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
    """OASR generates ``ANGLE1 + (i-1)*DLANGLE`` (unoasr21.f:173-177), so a
    non-uniform request cannot be honoured."""

    ENV = staticmethod(lambda: Environment(
        name='oasr', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600.0, shear_speed=400.0,
            density=1.8, attenuation=0.5, shear_attenuation=1.0),
    ))

    def test_non_uniform_angles_raise(self, tmp_path):
        """Requesting [0, 5, 45, 90] silently returned [0, 30, 60, 90]."""
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
        """``cmax`` was ``max(c_p * 1.1, 1e8)`` — the first term never won."""
        from uacpy.io.oases_writer import _oases_wavenumber_bounds
        ssp = np.array([[0.0, 1500.0], [100.0, 1480.0]])
        assert _oases_wavenumber_bounds(ssp) == (1480.0 * 0.9, 1e8)
        assert _oases_wavenumber_bounds(ssp, cmax=1e9)[1] == 1e9

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
        lowercase 'd' option (unoast31.f:125-127)."""
        from uacpy.io.oases_writer import write_oast_input
        dat = tmp_path / 'dopp.dat'
        write_oast_input(
            dat, _pekeris(), Source(depths=50.0, frequencies=100.0),
            Receiver(depths=[10.0], ranges=[1000.0]),
            options='N J T d', vrec=12.5,
        )
        assert dat.read_text().splitlines()[2].split()[-1] == '12.5'
        assert OAST(verbose=False, vrec=12.5).vrec == 12.5


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
    then NL layer records (oaseun31.f:43-110)."""
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
        for _ in range(8):                      # records 1..8
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
    (oaseun31.f:1172) and NRCV in OASN's INPRCV (oasnun22.f:32-35); both are
    hard STOPs that leave no output file."""

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
        # Block V ends with the last receiver row; the replica grid follows.
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
