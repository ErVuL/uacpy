"""SPARC time-domain-focused tests."""

import warnings

import pytest
import numpy as np

from uacpy.core.results import TimeSeriesField
from uacpy.models import SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, Source, Receiver, BoundaryProperties
from uacpy.core.environment import (
    LayeredBottom, RangeDependentLayeredBottom, SedimentLayer,
)

pytestmark = pytest.mark.requires_binary


class TestSPARCBasic:
    """Basic tests for SPARC model (seismo-acoustic PE)."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_basic_tl(self):
        """Test basic SPARC TL computation."""
        env = Environment(
            name="sparc_test",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'),
        )
        source = Source(depths=50.0, frequencies=50.0)
        receiver = Receiver(
            depths=np.linspace(10, 90, 9),
            ranges=np.linspace(100, 5000, 11)
        )

        sparc = SPARC(verbose=False)
        result = sparc.compute_tl(env=env, source=source, receiver=receiver)

        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))


class TestSPARCTimeSeries:
    """SPARC's primary purpose: native time-domain pressure p(t)."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_time_series_returns_time_series_field(self):
        """SPARC TIME_SERIES returns a real-valued TimeSeriesField."""
        env = Environment(
            name="sparc_ts",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'),
        )
        source = Source(depths=50.0, frequencies=50.0)
        receiver = Receiver(
            depths=np.linspace(10, 90, 5),
            ranges=np.linspace(500, 5000, 6),
        )

        sparc = SPARC(verbose=False)
        result = sparc.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
        )

        assert isinstance(result, TimeSeriesField)
        assert result.data.shape[0] == len(receiver.depths)
        assert result.data.shape[1] == len(receiver.ranges)
        assert result.data.shape[2] > 0
        assert np.isrealobj(result.data)
        assert np.all(np.isfinite(result.data))
        assert hasattr(result, 'time')


# ---------------------------------------------------------------------
# Auto-rigidify walks the inner halfspace of LayeredBottom /
# RangeDependentLayeredBottom and flips its acoustic_type.
# ---------------------------------------------------------------------

class TestSPARCRigidifyLayered:
    """Pure-Python unit tests: do not invoke the SPARC binary."""

    def _hs_halfspace(self):
        return BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1800.0, density=1.8, attenuation=0.3,
        )

    def test_rigidify_layered_bottom_walks_to_halfspace(self):
        """``LayeredBottom`` has no top-level ``acoustic_type``; the
        rigid flag lives on its inner ``.halfspace`` and must be
        flipped there."""
        lb = LayeredBottom(
            layers=[
                SedimentLayer(thickness=10.0, sound_speed=1600.0,
                              density=1.5, attenuation=0.2),
            ],
            halfspace=self._hs_halfspace(),
        )
        env = Environment(
            name='sparc_lb_rigid',
            bathymetry=100.0, ssp=1500.0, bottom=lb,
        )
        sparc = SPARC(verbose=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = sparc._sparc_rigidify_halfspace(env)
        # Warning fired
        rigidify_msgs = [w for w in caught
                         if "auto-converting" in str(w.message)]
        assert len(rigidify_msgs) == 1
        # Inner halfspace acoustic_type flipped
        assert out.bottom.halfspace.acoustic_type == 'rigid'
        # The downstream writer dispatches on halfspace_at_range — must
        # now report 'rigid'.
        assert out.halfspace_at_range(0.0).acoustic_type == 'rigid'
        # Original env left intact (copy semantics)
        assert env.bottom.halfspace.acoustic_type == 'half-space'

    def test_rigidify_rd_layered_bottom_walks_each_profile(self):
        """``RangeDependentLayeredBottom`` has a halfspace inside each
        per-range profile; every one must be flipped."""
        profA = LayeredBottom(
            layers=[SedimentLayer(thickness=5.0, sound_speed=1550.0,
                                  density=1.3, attenuation=0.1)],
            halfspace=self._hs_halfspace(),
        )
        profB = LayeredBottom(
            layers=[SedimentLayer(thickness=5.0, sound_speed=1700.0,
                                  density=1.7, attenuation=0.2)],
            halfspace=self._hs_halfspace(),
        )
        rdl = RangeDependentLayeredBottom(
            ranges=np.array([0.0, 10000.0]),
            profiles=[profA, profB],
        )
        env = Environment(
            name='sparc_rdl_rigid',
            bathymetry=100.0, ssp=1500.0, bottom=rdl,
        )
        sparc = SPARC(verbose=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = sparc._sparc_rigidify_halfspace(env)
        rigidify_msgs = [w for w in caught
                         if "auto-converting" in str(w.message)]
        assert len(rigidify_msgs) == 1
        for prof in out.bottom.profiles:
            assert prof.halfspace.acoustic_type == 'rigid'
        # Originals untouched
        for prof in env.bottom.profiles:
            assert prof.halfspace.acoustic_type == 'half-space'

    def test_rigidify_vacuum_layered_is_passthrough(self):
        """A LayeredBottom whose halfspace is already ``vacuum`` must
        NOT trigger the warning and must remain ``vacuum``."""
        lb = LayeredBottom(
            layers=[
                SedimentLayer(thickness=10.0, sound_speed=1600.0,
                              density=1.5, attenuation=0.2),
            ],
            halfspace=BoundaryProperties(acoustic_type='vacuum'),
        )
        env = Environment(
            name='sparc_lb_vacuum',
            bathymetry=100.0, ssp=1500.0, bottom=lb,
        )
        sparc = SPARC(verbose=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = sparc._sparc_rigidify_halfspace(env)
        rigidify_msgs = [w for w in caught
                         if "auto-converting" in str(w.message)]
        assert len(rigidify_msgs) == 0
        assert out.bottom.halfspace.acoustic_type == 'vacuum'

    @pytest.mark.requires_binary
    def test_layered_bottom_runs_end_to_end(self, tmp_path):
        """SPARC + LayeredBottom completes a binary run. The emitted
        ``.env`` declares ``NMedia = 1 + n_sediment_layers`` so the
        Fortran reader consumes all medium blocks before parsing the
        bottom boundary marker."""
        lb = LayeredBottom(
            layers=[
                SedimentLayer(thickness=10.0, sound_speed=1600.0,
                              density=1.5, attenuation=0.2),
            ],
            halfspace=self._hs_halfspace(),
        )
        env = Environment(
            name='sparc_lb_e2e',
            bathymetry=100.0, ssp=1500.0, bottom=lb,
        )
        sparc = SPARC(verbose=False, work_dir=tmp_path, cleanup=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sparc.run(
                env,
                Source(depths=50.0, frequencies=200.0),
                Receiver(depths=np.array([50.0]),
                         ranges=np.array([1000.0])),
            )
        # Got a result of the right type and shape.
        assert res.data.shape[-1] == 1  # one receiver range
        # The emitted .env declares the correct NMedia (=2 for one layer).
        env_path = next(tmp_path.glob('**/*.env'))
        first_lines = env_path.read_text().splitlines()
        # Line 3 is NMedia (after title + frequency).
        assert int(first_lines[2].strip()) == 2, (
            f"NMedia should be 2 (water + 1 sediment layer); got "
            f"{first_lines[2]!r}"
        )
