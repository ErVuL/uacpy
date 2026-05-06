"""
Comprehensive tests for OASES models

OASES (Ocean Acoustics and Seismic Exploration System) is a suite of
seismo-acoustic models for underwater acoustics. This test file provides
systematic validation of all OASES variants:

- OAST: Transmission loss computation
- OASN: Normal modes extraction
- OASR: Reflection coefficients
- OASP: Parabolic equation (PE) for range-dependent propagation

These models were previously minimally tested despite being fully implemented.
"""

import pytest
import numpy as np

pytestmark = pytest.mark.requires_oases

from uacpy.models import OAST, OASN, OASR, OASP, OASES, RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ExecutableNotFoundError


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
            depth=100.0,
            sound_speed=1500.0,
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

        assert result.field_type == 'tl'
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
            depth=100.0,
            sound_speed=1500.0,
            bottom=bottom
        )

        oast = OAST(verbose=False)
        result = oast.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
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
            depth=100.0,
            sound_speed=1500.0,
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
        assert cov.field_type == 'covariance'
        assert cov.covariance.ndim == 3
        assert cov.covariance.shape[1] == cov.covariance.shape[2] == cov.n_receivers
        assert cov.n_frequencies >= 1

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
            depth=100.0,
            sound_speed=1500.0,
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
            depth=100.0,
            sound_speed=1500.0,
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
        oasr = OASR(verbose=False)
        result = oasr.run(
            env=oasr_env, source=source, receiver=receiver,
            angle_min=0.0, angle_max=90.0, n_angles=91,
        )

        assert isinstance(result, ReflectionCoefficient)
        assert result.field_type == 'reflection_coefficients'
        assert result.theta.shape == (91,)
        assert result.R.shape == result.phi.shape
        assert result.R.size > 0 and np.all(np.isfinite(result.R))
        assert np.all((result.R >= 0.0) & (result.R <= 1.0 + 1e-6))

    @pytest.mark.requires_binary
    def test_oasr_angle_resolution(self, oasr_env, source, receiver):
        """Test OASR with different angle resolutions."""
        oasr = OASR(verbose=False)

        # Test with coarse angle resolution
        result = oasr.run(
            env=oasr_env,
            source=source,
            receiver=receiver,
            angle_min=0.0,
            angle_max=90.0,
            n_angles=19  # 5-degree steps
        )

        assert result.field_type == 'reflection_coefficients'


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
            depth=100.0,
            sound_speed=1500.0,
            bathymetry=bathymetry,
            bottom=bottom
        )

    @pytest.fixture
    def source(self):
        return Source(depth=50.0, frequency=50.0)  # Lower frequency for PE

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
        """Test OASP range-dependent propagation."""
        oasp = OASP(verbose=False)
        result = oasp.compute_tl(
            env=oasp_env,
            source=source,
            receiver=receiver
        )

        assert result.field_type == 'tl'
        assert result.shape[0] > 0  # Has depth dimension
        assert result.shape[1] > 0  # Has range dimension
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_oasp_broadband(self, oasp_env, receiver):
        """OASP run_mode=BROADBAND returns a populated TransferFunction."""
        from uacpy.core.results import TransferFunction
        source = Source(
            depth=50.0,
            frequency=np.array([30.0, 50.0, 70.0]),
        )

        oasp = OASP(verbose=False)
        result = oasp.run(
            env=oasp_env,
            source=source,
            receiver=receiver,
            run_mode=RunMode.BROADBAND,
        )

        assert isinstance(result, TransferFunction)
        assert result.field_type == 'transfer_function'
        assert result.frequencies is not None and len(result.frequencies) > 0
        assert result.data.shape[:2] == (len(receiver.depths), len(receiver.ranges))


class TestOASESUnified:
    """Tests for unified OASES interface."""

    @pytest.mark.requires_binary
    def test_oases_instantiation(self):
        """Test unified OASES interface."""
        oases = OASES(verbose=False)
        assert oases.model_name == 'OASES'

    @pytest.mark.requires_binary
    def test_oases_auto_select_model(self):
        """Test that OASES automatically selects appropriate sub-model."""
        env = Environment(
            name="oases_test",
            depth=100.0,
            sound_speed=1500.0
        )
        source = Source(depth=50.0, frequency=100.0)
        receiver = Receiver(
            depths=[25.0, 50.0, 75.0],
            ranges=[1000.0, 3000.0]
        )

        oases = OASES(verbose=False)

        # Should default to OAST for TL computation
        result = oases.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'


# Integration test to verify all OASES models can be imported and instantiated
@pytest.mark.requires_binary
def test_all_oases_models_importable():
    """Verify all OASES models can be imported and instantiated."""
    models = {
        'OAST': OAST,
        'OASN': OASN,
        'OASR': OASR,
        'OASP': OASP,
        'OASES': OASES
    }

    for name, ModelClass in models.items():
        model = ModelClass(verbose=False)
        assert model.model_name == name, f"{name} has incorrect model_name"
        assert hasattr(model, 'run'), f"{name} missing run() method"
        assert hasattr(model, 'compute_tl') or name == 'OASN' or name == 'OASR', \
            f"{name} missing compute_tl() method"
