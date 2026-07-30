"""SPARC time-domain-focused tests."""

import warnings

import pytest
import numpy as np

from uacpy.core.results import Field
from uacpy.models import SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, Source, Receiver, BoundaryProperties
from uacpy.core.environment import (
    SeabedColumn, Bottom, SedimentLayer,
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
            depths=np.linspace(10, 90, 5),
            ranges=np.linspace(100, 3000, 6)
        )

        sparc = SPARC(verbose=False)
        result = sparc.compute_tl(env=env, source=source, receiver=receiver)

        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))


class TestSPARCTimeSeries:
    """SPARC's primary purpose: native time-domain pressure p(t)."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_time_series_returns_time_series_field(self):
        """SPARC TIME_SERIES returns a real-valued Field."""
        env = Environment(
            name="sparc_ts",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='rigid'),
        )
        # Smoke test (type/shape/finite only). SPARC time-marching cost grows
        # steeply with frequency and range, so keep both modest here.
        source = Source(depths=50.0, frequencies=30.0)
        receiver = Receiver(
            depths=np.linspace(10, 90, 3),
            ranges=np.linspace(500, 2500, 4),
        )

        sparc = SPARC(verbose=False)
        result = sparc.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
        )

        assert isinstance(result, Field)
        assert result.data.shape[0] == len(receiver.depths)
        assert result.data.shape[1] == len(receiver.ranges)
        assert result.data.shape[2] > 0
        # range coord (SPARC's actual grid) length-matches the data columns
        assert result.coords['range'].shape[0] == result.data.shape[1]
        assert np.isrealobj(result.data)
        assert np.all(np.isfinite(result.data))
        assert result.times is not None and result.times.size > 0


# ---------------------------------------------------------------------
# Auto-rigidify walks the inner halfspace of SeabedColumn /
# Bottom and flips its acoustic_type.
# ---------------------------------------------------------------------

class TestSPARCRigidifyLayered:
    """Pure-Python unit tests: do not invoke the SPARC binary."""

    def _hs_halfspace(self):
        return BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1800.0, density=1.8, attenuation=0.3,
        )

    def test_rigidify_layered_bottom_walks_to_halfspace(self):
        """``SeabedColumn`` has no top-level ``acoustic_type``; the
        rigid flag lives on its inner ``.halfspace`` and must be
        flipped there."""
        lb = SeabedColumn(
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
        assert out.bottom.columns[0].halfspace.acoustic_type == 'rigid'
        # The downstream writer dispatches on bottom.halfspace_at — must
        # now report 'rigid'.
        assert out.bottom.halfspace_at(range=0.0).acoustic_type == 'rigid'
        # Original env left intact (copy semantics)
        assert env.bottom.columns[0].halfspace.acoustic_type == 'half-space'

    def test_rigidify_rd_layered_bottom_walks_each_profile(self):
        """``Bottom`` has a halfspace inside each
        per-range profile; every one must be flipped."""
        profA = SeabedColumn(
            layers=[SedimentLayer(thickness=5.0, sound_speed=1550.0,
                                  density=1.3, attenuation=0.1)],
            halfspace=self._hs_halfspace(),
        )
        profB = SeabedColumn(
            layers=[SedimentLayer(thickness=5.0, sound_speed=1700.0,
                                  density=1.7, attenuation=0.2)],
            halfspace=self._hs_halfspace(),
        )
        rdl = Bottom.from_columns([profA, profB], ranges=np.array([0.0, 10000.0]))
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
        for prof in out.bottom.columns:
            assert prof.halfspace.acoustic_type == 'rigid'
        # Originals untouched
        for prof in env.bottom.columns:
            assert prof.halfspace.acoustic_type == 'half-space'

    def test_rigidify_vacuum_layered_is_passthrough(self):
        """A SeabedColumn whose halfspace is already ``vacuum`` must
        NOT trigger the warning and must remain ``vacuum``."""
        lb = SeabedColumn(
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
        assert out.bottom.columns[0].halfspace.acoustic_type == 'vacuum'

    @pytest.mark.requires_binary
    def test_layered_bottom_runs_end_to_end(self, tmp_path):
        """SPARC + SeabedColumn completes a binary run. The emitted
        ``.env`` declares ``NMedia = 1 + n_sediment_layers`` so the
        Fortran reader consumes all medium blocks before parsing the
        bottom boundary marker."""
        lb = SeabedColumn(
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


def test_sparc_passes_source_geometry_to_the_transform(monkeypatch):
    """sparc.py:812 must forward Source.source_type, not the 'R' default."""
    import uacpy.models.sparc as sparc_mod

    seen = {}
    original = sparc_mod.sparc_snapshot_to_field

    def spy(*args, **kwargs):
        seen['source_type'] = kwargs.get('source_type')
        return original(*args, **kwargs)

    monkeypatch.setattr(sparc_mod, 'sparc_snapshot_to_field', spy)

    env = Environment(name='sp_geom', bathymetry=200.0, ssp=1500.0)
    rcv = Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))
    SPARC(output_mode='S').run(
        env, Source(depths=50, frequencies=200, source_type='line'), rcv)
    assert seen['source_type'] == 'X'


def test_sparc_rejects_geometry_outside_snapshot_mode():
    """output_mode 'R'/'D' never Hankel-transform, so they honour no geometry."""
    from uacpy.core.exceptions import ConfigurationError
    env = Environment(name='sp_rej', bathymetry=200.0, ssp=1500.0)
    rcv = Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))
    with pytest.raises(ConfigurationError, match="source_type"):
        SPARC(output_mode='R').run(
            env, Source(depths=50, frequencies=200, source_type='line'), rcv)


def test_time_series_warns_when_the_output_grid_aliases():
    """A TIME_SERIES grid whose Nyquist is below the source band must warn.

    SPARC keeps the caller's ``n_t_out`` for TIME_SERIES by contract, but the
    default (512 samples over a ~10 s window => fs ~51 Hz) puts a 100 Hz source
    well above Nyquist. The returned p(t) is plausible-looking and at the wrong
    frequency, so silence is the wrong behaviour.
    """
    import warnings
    env = Environment(name='p', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.5))
    src = Source(depths=50.0, frequencies=100.0)
    rcv = Receiver(depths=100.0, ranges=np.array([2000.0]))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        SPARC(verbose=False).run(env, src, rcv, run_mode=RunMode.TIME_SERIES)
    alias = [x for x in w if 'alias' in str(x.message)]
    assert alias, "aliased TIME_SERIES grid did not warn"
    assert 'n_t_out>=' in str(alias[0].message), "warning must name the fix"


def test_time_series_does_not_warn_when_adequately_sampled():
    """With enough samples for the band, no aliasing warning."""
    import warnings
    env = Environment(name='p', bathymetry=200.0, ssp=1500.0,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1800.0, density=1.8,
                                                attenuation=0.5))
    src = Source(depths=50.0, frequencies=20.0)
    rcv = Receiver(depths=100.0, ranges=np.array([2000.0]))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        SPARC(verbose=False, n_t_out=4096).run(
            env, src, rcv, run_mode=RunMode.TIME_SERIES)
    assert not [x for x in w if 'alias' in str(x.message)]
