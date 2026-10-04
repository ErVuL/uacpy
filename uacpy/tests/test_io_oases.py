"""The OASES decks and files (``uacpy.io.oases_writer``,
``uacpy.io.oases_reader``): the layer and roughness records, the OASN and
OASR blocks, and the frequency axes the readers build.
"""

import io
import numpy as np
import pytest
import re
import uacpy
from pathlib import Path
from uacpy.core import BoundaryProperties
from uacpy.core import Environment
from uacpy.core.bottom import SeabedColumn
from uacpy.core.boundary import SedimentLayer
from uacpy.core.exceptions import ConfigurationError
from uacpy.io.oases_writer import OasnNoise, OasnReplicaGrid


class TestOASRFrequencyOverride:
    """An explicit ``frequencies=`` override (passed by OASR.run as
    freq_min/freq_max/n_frequencies) must win over a multi-element
    ``source.frequencies`` — honouring it only for single-frequency sources
    would silently drop it for every sweep."""

    def _freq_line(self, path):
        # The OASR deck writes "<freq_min> <freq_max> <nfreq> <out_inc>" right after the
        # source line; locate it by the 4-token float/int signature.
        for ln in Path(path).read_text().splitlines():
            toks = ln.split()
            if len(toks) == 4:
                try:
                    return float(toks[0]), float(toks[1]), int(toks[2])
                except ValueError:
                    continue
        raise AssertionError("no frequency line found in OASR deck")

    def test_override_wins_over_multifreq_source(self, tmp_path):
        from uacpy.io.oases_writer import write_oasr_input
        from uacpy.core import Environment, Source, Receiver, BoundaryProperties
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(acoustic_type='half-space',
                                                    sound_speed=1600.0,
                                                    density=1.8,
                                                    attenuation=0.5))
        # Multi-element source: must not hijack the sweep.
        src = Source(depths=50.0, frequencies=[50.0, 100.0, 150.0])
        rcv = Receiver(depths=[50.0], ranges=[1000.0])
        out = tmp_path / "oasr.dat"
        write_oasr_input(str(out), env, src, rcv,
                         angles=np.linspace(0, 90, 91),
                         freq_min=200.0, freq_max=400.0, n_frequencies=5)
        freq_min, freq_max, nfreq = self._freq_line(out)
        assert (freq_min, freq_max, nfreq) == (200.0, 400.0, 5)

    def test_source_drives_sweep_without_override(self, tmp_path):
        from uacpy.io.oases_writer import write_oasr_input
        from uacpy.core import Environment, Source, Receiver, BoundaryProperties
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(acoustic_type='half-space',
                                                    sound_speed=1600.0,
                                                    density=1.8,
                                                    attenuation=0.5))
        src = Source(depths=50.0, frequencies=[50.0, 100.0, 150.0])
        rcv = Receiver(depths=[50.0], ranges=[1000.0])
        out = tmp_path / "oasr.dat"
        write_oasr_input(str(out), env, src, rcv,
                         angles=np.linspace(0, 90, 91))
        freq_min, freq_max, nfreq = self._freq_line(out)
        assert (freq_min, freq_max, nfreq) == (50.0, 150.0, 3)


class TestOaspTrfFrequencyAxis:
    """TRF bin indices are 1-based, so bin k is (k-1)*DLFRQ.

    ``oasiun22.f:1256-1261`` sets ``DLFRQP = 1/(DT*NX)`` and
    ``LX = nint(FMIN/DLFRQP + 1)``. Reading bin k as ``k/(dt*nx)`` shifts the
    whole axis up by one bin.
    """

    @staticmethod
    def _axis(lx, mx, nx, dt):
        """The reader's frequency-axis expression, isolated."""
        return np.array([((k - 1) / (dt * nx)) for k in range(lx, mx + 1)],
                        dtype=np.float64)

    def test_round_trips_the_oases_index_formula(self):
        nx, dt = 1024, 1.0 / 4096.0
        dlfrq = 1.0 / (dt * nx)
        freq_min, freq_max = 100.0, 400.0
        lx = int(round(freq_min / dlfrq + 1))       # oasiun22.f:1259
        mx = int(round(freq_max / dlfrq + 1))       # oasiun22.f:1261
        axis = self._axis(lx, mx, nx, dt)
        assert axis[0] == pytest.approx(freq_min, abs=0.5 * dlfrq)
        assert axis[-1] == pytest.approx(freq_max, abs=0.5 * dlfrq)

    def test_first_bin_is_dc_not_one_bin_up(self):
        nx, dt = 512, 1.0 / 2048.0
        assert self._axis(1, 4, nx, dt)[0] == pytest.approx(0.0)

    def test_spacing_is_the_dft_bin_width(self):
        nx, dt = 2048, 1.0 / 8192.0
        axis = self._axis(10, 20, nx, dt)
        np.testing.assert_allclose(np.diff(axis), 1.0 / (dt * nx), rtol=1e-12)


class TestOASNWavenumberSampling:
    """``n_wavenumbers`` reaches every OASN integration block. The replica and
    discrete-source wavenumber lines (``NWSIN``/``NWDIN``, unoasn22.f:227 and
    oasnun22.f:421) are what OASN samples the field on."""

    @staticmethod
    def _env_src_rcv():
        env = uacpy.Environment(name='oasn', bathymetry=100, ssp=1500)
        source = uacpy.Source(frequencies=100, depths=50)
        receiver = uacpy.Receiver(depths=np.array([30.0, 50.0, 70.0]),
                                  ranges=np.array([0.0]))
        return env, source, receiver

    @staticmethod
    def _write(tmp_path, **kwargs):
        import warnings
        from uacpy.io.oases_writer import write_oasn_input
        env, source, receiver = TestOASNWavenumberSampling._env_src_rcv()
        path = tmp_path / 'oasn.dat'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            write_oasn_input(path, env, source, receiver, **kwargs)
        return path.read_text().splitlines()

    def test_replica_block_honours_n_wavenumbers(self, tmp_path):
        lines = self._write(tmp_path, options='N R J', n_wavenumbers=512)
        assert lines[-1].split() == ['512', '1', '512']

    def test_replica_block_defaults_to_automatic(self, tmp_path):
        lines = self._write(tmp_path, options='N R J', n_wavenumbers=None)
        assert lines[-1].split()[0] == '-1'

    def test_discrete_source_block_honours_n_wavenumbers(self, tmp_path):
        lines = self._write(
            tmp_path, options='N J', n_wavenumbers=256,
            noise=OasnNoise(discrete_sources=[
                {'depth': 40.0, 'x': 1000.0, 'y': 0.0, 'level': 180.0}]))
        assert lines[-1].split() == ['256', '1', '256']


class TestOASESWriterKnobsReachTheDeck:
    """Every OASES writer takes ``**kwargs``, so a knob no block reads would be
    dropped without a trace and the run would quietly use the default."""

    @staticmethod
    def _args():
        env = uacpy.Environment(name='oases', bathymetry=100, ssp=1500)
        source = uacpy.Source(frequencies=100, depths=50)
        receiver = uacpy.Receiver(depths=np.array([30.0, 50.0, 70.0]),
                                  ranges=np.linspace(100.0, 5000.0, 10))
        return env, source, receiver

    @pytest.mark.parametrize('writer_name,options,bad', [
        ('write_oast_input', None, 'replica_nz'),
        ('write_oasn_input', 'N J', 'vrec'),
        ('write_oasn_input', 'N J', 'range_max'),
        ('write_oasp_input', None, 'nw_sample'),
        ('write_oasr_input', None, 'n_wavenumbers'),
    ])
    def test_a_parameter_the_writer_does_not_take_is_a_type_error(
            self, tmp_path, writer_name, options, bad):
        """The writers take keyword-only parameters, so Python itself
        refuses a keyword the deck does not read."""
        import warnings
        from uacpy.io import oases_writer
        writer = getattr(oases_writer, writer_name)
        env, source, receiver = self._args()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with pytest.raises(TypeError,
                               match=f"unexpected keyword argument '{bad}'"):
                writer(tmp_path / 'x.dat', env, source, receiver,
                       options=options, **{bad: 1.0})

    def test_documented_knobs_are_accepted(self, tmp_path):
        import warnings
        from uacpy.io.oases_writer import write_oasn_input
        env, source, receiver = self._args()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            write_oasn_input(tmp_path / 'ok.dat', env, source, receiver,
                             options='N R J', n_wavenumbers=256,
                             noise=OasnNoise(c_low=1400.0, c_high=1.0e8),
                             replica=OasnReplicaGrid(z=(None, None, 8)),
                             integration_offset=0.0)
        assert (tmp_path / 'ok.dat').exists()


class TestOASNNoiseWavenumberCounts:
    """The surface- and deep-noise blocks have no automatic-sampling branch:
    ``NOIPAR`` reads three explicit counts and sums them into ``NWVNON`` /
    ``NWVNOP`` (oasnun22.f:312, :358). A negative count makes that total
    negative and the block integrates nothing — the covariance silently
    collapses to the white-noise identity."""

    @staticmethod
    def _last_line(tmp_path, **kwargs):
        import warnings
        from uacpy.io.oases_writer import write_oasn_input
        env = uacpy.Environment(name='oasn', bathymetry=100, ssp=1500)
        source = uacpy.Source(frequencies=100, depths=50)
        receiver = uacpy.Receiver(depths=np.array([30.0, 50.0, 70.0]),
                                  ranges=np.array([0.0]))
        path = tmp_path / 'oasn.dat'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            write_oasn_input(path, env, source, receiver, options='N J',
                             **kwargs)
        return path.read_text().splitlines()[-1]

    @pytest.mark.parametrize('block', [
        {'noise': OasnNoise(surface_level=70.0)},
        {'noise': OasnNoise(deep_level=70.0)},
    ])
    @pytest.mark.parametrize('nw', [-1, 0, None])
    def test_automatic_falls_back_to_positive_counts(self, tmp_path, block, nw):
        counts = [int(v) for v in
                  self._last_line(tmp_path, n_wavenumbers=nw, **block).split()]
        assert len(counts) == 3
        assert sum(counts) > 0, "a non-positive total integrates nothing"

    @pytest.mark.parametrize('block', [
        {'noise': OasnNoise(surface_level=70.0)},
        {'noise': OasnNoise(deep_level=70.0)},
    ])
    def test_pinned_count_reaches_the_block(self, tmp_path, block):
        line = self._last_line(tmp_path, n_wavenumbers=800, **block)
        assert line.split()[:2] == ['800', '800']


class TestOasesInterfaceRoughness:
    """OASES reads column 7 of each layer record as that interface's RMS
    roughness (``src/oaseun31.f:54``; ``doc/oast.tex:42,48`` — the record opens
    with "D: Depth of interface" and RG is "RMS value of interface roughness").
    A record's RG therefore belongs to the interface at the TOP of its layer,
    the same convention Kraken uses for ``SSP%sigma``.
    """

    @staticmethod
    def _env(*, surface, seafloor, base):
        from uacpy.core import Environment
        from uacpy.core.boundary import BoundaryProperties, SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
        from uacpy.core.surface import Surface
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            surface=Surface([BoundaryProperties(acoustic_type='vacuum',
                                                roughness=surface)]),
            bottom=Bottom.from_column(SeabedColumn(
                layers=[SedimentLayer(thickness=30, sound_speed=1600,
                                      density=1.8, attenuation=0.5,
                                      roughness=seafloor)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800, density=2.0,
                    attenuation=0.6, roughness=base))))

    def test_each_record_carries_its_own_interface(self, tmp_path):
        from uacpy.core import Source, Receiver
        from uacpy.io.oases_writer import write_oast_input
        out = tmp_path / 'rg.dat'
        write_oast_input(out, self._env(surface=1.5, seafloor=2.0, base=0.7),
                         Source(depths=50, frequencies=100.0),
                         Receiver(depths=50, ranges=[5000.0]))
        rows = [ln.split() for ln in out.read_text().splitlines()
                if re.match(r'^[\d.]+(\s+[-\d.e+]+){7}$', ln)]
        by_depth = {r[0]: float(r[6]) for r in rows}
        # Record '0' is the upper half-space (layer 1); '0.00' the first water
        # layer (layer 2). ROUGH(1) is overwritten by ROUGH(2) at
        # oaseun31.f:377 and both manuals call RG(1) dummy (oast.tex:344,
        # oasp.tex:365), so the sea surface can only ride on the water record.
        assert by_depth['0'] == 0.0                       # RG(1), discarded
        assert by_depth['0.00'] == pytest.approx(1.5)     # sea surface
        assert by_depth['100.00'] == pytest.approx(2.0)   # seafloor
        assert by_depth['130.00'] == pytest.approx(0.7)   # base of the stack

    def test_unlayered_column_puts_halfspace_roughness_on_the_seafloor(self, tmp_path):
        from uacpy.core import Environment, Source, Receiver
        from uacpy.core.boundary import BoundaryProperties
        from uacpy.core.bottom import Bottom
        from uacpy.io.oases_writer import write_oast_input
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=Bottom.from_halfspace(BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1600,
                              density=1.5, attenuation=0.5, roughness=1.2)))
        out = tmp_path / 'rg1.dat'
        write_oast_input(out, env, Source(depths=50, frequencies=100.0),
                         Receiver(depths=50, ranges=[5000.0]))
        # Selected by the layer record's shape (8 columns), not by a numeric
        # prefix: the frequency record now carries the same digits, and keying on
        # a format is only stable while different records happen to print
        # differently.
        seabed = next(ln for ln in out.read_text().splitlines()
                      if len(ln.split()) == 8
                      and ln.split()[0].startswith('100.'))
        assert float(seabed.split()[6]) == pytest.approx(1.2)


class TestOasesSeaSurfaceRoughnessRecord:
    """``env.surface.roughness`` must land on ROUGH(2), not ROUGH(1).

    INENVI reads column 7 of every layer record into ``ROUGH(M)``
    (``oases/src/oaseun31.f:54``) and then unconditionally overwrites
    ``ROUGH(1)`` with ``ROUGH(2)`` the moment the read loop closes
    (``:376-377``). ``DO 1111 M=2,NUML`` compares ``LAYTYP(M-1)`` with
    ``LAYTYP(M)`` (``:381-383``), which fixes ROUGH(M) as the roughness of
    the interface at the TOP of layer M; ``oasnun22.f:250`` places OASN's
    surface-noise sheet at ``V(2,1)+V(1,2)/(30*FREQ1)+ROUGH(2)`` on the same
    reading. Both manuals state "RG(1) is dummy" (``doc/oast.tex:344``,
    ``doc/oasp.tex:365``). So the sea surface is the FIRST WATER LAYER's RG,
    and anything written to the upper-half-space record is discarded.
    """

    SURFACE_RG = 1.5

    @staticmethod
    def _env(ssp):
        from uacpy.core import Environment
        from uacpy.core.boundary import BoundaryProperties
        from uacpy.core.bottom import Bottom
        from uacpy.core.surface import Surface
        return Environment(
            bathymetry=100.0, ssp=ssp,
            surface=Surface([BoundaryProperties(
                acoustic_type='vacuum',
                roughness=TestOasesSeaSurfaceRoughnessRecord.SURFACE_RG)]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700, density=1.8,
                attenuation=0.5)))

    @staticmethod
    def _layer_records(text):
        """The deck's NL layer records, in deck order.

        The layer block opens with a bare integer NL (oaseun31.f:43) — the
        first such record in every OASES deck — followed by NL layer records
        (:54).
        """
        lines = text.splitlines()
        i = next(i for i, ln in enumerate(lines) if ln.strip().isdigit())
        n_layers = int(lines[i])
        return [ln.split() for ln in lines[i + 1:i + 1 + n_layers]]

    @pytest.mark.parametrize('writer_name,ssp', [
        ('write_oast_input', 1500.0),
        ('write_oast_input', [(0.0, 1500.0), (50.0, 1490.0), (100.0, 1510.0)]),
        ('write_oasn_input', [(0.0, 1500.0), (50.0, 1490.0), (100.0, 1510.0)]),
        ('write_oasp_input', [(0.0, 1500.0), (50.0, 1490.0), (100.0, 1510.0)]),
    ])
    def test_surface_rg_rides_on_the_first_water_record(
            self, tmp_path, writer_name, ssp):
        import warnings
        from uacpy.core import Source, Receiver
        import uacpy.io.oases_writer as W
        out = tmp_path / f'{writer_name}.dat'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            getattr(W, writer_name)(
                out, self._env(ssp), Source(depths=50, frequencies=100.0),
                Receiver(depths=[20.0, 50.0, 80.0],
                         ranges=np.linspace(100.0, 5000.0, 10)))
        records = self._layer_records(out.read_text())
        assert float(records[0][6]) == 0.0, "RG(1) is the dummy INENVI clobbers"
        assert float(records[1][6]) == pytest.approx(self.SURFACE_RG)
        # Interfaces inside the water column are SSP sample boundaries.
        assert all(float(r[6]) == 0.0 for r in records[2:-1])

    @pytest.mark.parametrize('writer_name,extra', [
        ('write_oast_input', 1), ('write_oasn_input', 1),
        ('write_oasp_input', 2),
    ])
    def test_every_layer_record_carries_its_full_column_count(self, tmp_path, writer_name, extra):
        """INENVI's reads are list-directed but positional, so column 7 has to
        stay column 7 and the inert padding past it has to stay put."""
        import warnings
        from uacpy.core import Source, Receiver
        import uacpy.io.oases_writer as W
        out = tmp_path / f'{writer_name}_arity.dat'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            getattr(W, writer_name)(
                out, self._env([(0.0, 1500.0), (50.0, 1490.0), (100.0, 1510.0)]),
                Source(depths=50, frequencies=100.0),
                Receiver(depths=[20.0, 50.0, 80.0],
                         ranges=np.linspace(100.0, 5000.0, 10)))
        records = self._layer_records(out.read_text())
        # The upper half-space record carries the manual's D CC CS AC AS RO
        # RG IG columns whatever the deck; the water and seabed records carry
        # this program's own trailing padding.
        assert len(records[0]) == 8
        assert all(len(r) == 7 + extra for r in records[1:])

    def test_gradient_first_layer_is_warned_about(self, tmp_path):
        """oaseun31.f:382-388 refuses roughness at an interface bounding a
        LAYTYP 2 (sound-speed-gradient) layer, but its STOP is commented out at
        :388 — so only a uacpy-side warning tells the caller."""
        from uacpy.core import Source, Receiver
        from uacpy.io.oases_writer import write_oast_input
        env = self._env([(0.0, 1500.0), (50.0, 1490.0), (100.0, 1510.0)])
        with pytest.warns(UserWarning, match='sound-speed gradient'):
            write_oast_input(tmp_path / 'grad.dat', env,
                             Source(depths=50, frequencies=100.0),
                             Receiver(depths=[20.0, 50.0, 80.0],
                                      ranges=np.linspace(100.0, 5000.0, 10)))

    def test_isovelocity_column_is_not_warned_about(self, tmp_path, recwarn):
        """An isovelocity first layer is LAYTYP 1, which the test at
        oaseun31.f:383 passes."""
        from uacpy.core import Source, Receiver
        from uacpy.io.oases_writer import write_oast_input
        write_oast_input(tmp_path / 'iso.dat', self._env(1500.0),
                         Source(depths=50, frequencies=100.0),
                         Receiver(depths=[20.0, 50.0, 80.0],
                                  ranges=np.linspace(100.0, 5000.0, 10)))
        assert not [w for w in recwarn
                    if 'sound-speed gradient' in str(w.message)]

    def test_oasr_layer_one_carries_no_surface_roughness(self, tmp_path):
        """OASR's layer 1 IS the water half-space (``SD=V(2,1)-1.0E-3``,
        unoasr21.f:89) — there is no sea surface in the model, and its RG is
        the same dummy. The reflecting interface is ROUGH(2), the seabed."""
        from uacpy.core import Environment, Source, Receiver
        from uacpy.core.boundary import BoundaryProperties
        from uacpy.core.bottom import Bottom
        from uacpy.core.surface import Surface
        from uacpy.io.oases_writer import write_oasr_input
        env = Environment(
            bathymetry=100.0, ssp=1500.0,
            surface=Surface([BoundaryProperties(acoustic_type='vacuum',
                                                roughness=self.SURFACE_RG)]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700, density=1.8,
                attenuation=0.5, roughness=0.9)))
        out = tmp_path / 'oasr.dat'
        write_oasr_input(out, env, Source(depths=50, frequencies=100.0),
                         Receiver(depths=[50.0], ranges=[1000.0]))
        records = self._layer_records(out.read_text())
        assert float(records[0][6]) == 0.0
        assert float(records[1][6]) == pytest.approx(0.9)   # seabed = ROUGH(2)


class TestOasesNegativeRoughnessRejected:
    """A negative RG changes the layer record's arity, so the writers refuse it.

    ``oaseun31.f:72`` branches on ``rough(m).lt.-1e-10``, backspaces and
    re-reads eight items with CLEN (``:75``); ``:77`` then tests
    ``clen(m).lt.-1e-10``, which is false for the CLEN==0 these decks supply,
    so ``:91`` takes the ``nvol=3`` branch and reads **nine** items from a
    record carrying eight. Every subsequent READ is shifted.

    ``BoundaryProperties`` and ``SedimentLayer`` reject it at construction and
    on assignment; these cases store it past the carrier's checks with
    ``object.__setattr__``, standing in for a carrier gap, and pin the
    writer-side guard behind them. The carrier-side refusal is pinned per
    node by ``test_input_validation.py::
    test_every_carrier_refuses_an_assignment_its_constructor_refuses``.
    """

    @staticmethod
    def _env():
        from uacpy.core import Environment
        from uacpy.core.boundary import BoundaryProperties, SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
        from uacpy.core.surface import Surface
        return Environment(
            bathymetry=100.0, ssp=1500.0,
            surface=Surface([BoundaryProperties(acoustic_type='vacuum')]),
            bottom=Bottom.from_column(SeabedColumn(
                layers=[SedimentLayer(thickness=30, sound_speed=1600,
                                      density=1.8, attenuation=0.5)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800, density=2.0,
                    attenuation=0.6))))

    @staticmethod
    def _write(env, tmp_path):
        from uacpy.core import Source, Receiver
        from uacpy.io.oases_writer import write_oast_input
        write_oast_input(tmp_path / 'neg.dat', env,
                         Source(depths=50, frequencies=100.0),
                         Receiver(depths=[50.0], ranges=[5000.0]))

    def test_negative_surface_roughness(self, tmp_path):
        env = self._env()
        object.__setattr__(env.surface.nodes[0], 'roughness', -1.0)
        with pytest.raises(ConfigurationError, match='oaseun31.f:72-93'):
            self._write(env, tmp_path)

    def test_negative_layer_roughness(self, tmp_path):
        env = self._env()
        object.__setattr__(env.bottom.columns[0].layers[0], 'roughness', -1.0)
        with pytest.raises(ConfigurationError, match='oaseun31.f:72-93'):
            self._write(env, tmp_path)

    def test_negative_halfspace_roughness(self, tmp_path):
        env = self._env()
        object.__setattr__(env.bottom.columns[0].halfspace, 'roughness', -1.0)
        with pytest.raises(ConfigurationError, match='oaseun31.f:72-93'):
            self._write(env, tmp_path)

    def test_zero_roughness_writes_normally(self, tmp_path):
        self._write(self._env(), tmp_path)
        assert (tmp_path / 'neg.dat').exists()


class TestOasnReplicaGridDeckDefaults:
    """oases.md §10 / DOCUMENTATION §OASN: the replica grid's ``None``
    defaults on disk — z spans 10 m to ``depth − 10`` over 20 points, x
    spans 0.1-10 km over 50, y is the degenerate 0/0 single point — and the
    deep noise sheet sits at half the water depth when its depth is unset.
    Deck-level asserts; no binary runs."""

    @staticmethod
    def _deck(tmp_path, options, **kw):
        from uacpy.io.oases_writer import write_oasn_input
        env = Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.8, attenuation=0.5))
        p = tmp_path / 'oasn_run.dat'
        write_oasn_input(p, env, uacpy.Source(depths=10.0, frequencies=100.0),
                         uacpy.Receiver(depths=[30.0, 70.0], ranges=[0.0]),
                         options=options, **kw)
        return p.read_text().splitlines()

    def test_replica_grid_defaults(self, tmp_path):
        lines = self._deck(tmp_path, 'R J')
        # ZSMIN ZSMAX NSRCZ / XSMIN XSMAX NSRCX / YSMIN YSMAX NSRCY —
        # consecutive rows (unoasn22.f:184-186); x/y are km on disk.
        z = lines.index('10.00 90.00 20')
        assert lines[z + 1] == '0.100000000 10.000000000 50'
        assert lines[z + 2] == '0.000000000 0.000000000 1'

    def test_deep_sheet_defaults_to_half_the_water_depth(self, tmp_path):
        lines = self._deck(tmp_path, 'N J',
                           noise=OasnNoise(deep_level=60.0))
        # DPSD (oasnun22.f:325) is a bare one-value record; 100 m water
        # puts the default sheet at 50 m.
        assert '50.00' in lines

    def test_explicit_replica_grid_overrides_the_defaults(self, tmp_path):
        lines = self._deck(tmp_path, 'R J', replica=OasnReplicaGrid(
            z=(20.0, 80.0, 5), x=(500.0, 2000.0, 3)))
        z = lines.index('20.00 80.00 5')
        assert lines[z + 1] == '0.500000000 2.000000000 3'


class TestFortranTitleDecoding:
    """OASS never assigns the title it hands ``PUTXSM``
    (``unoass21.f:34``, ``oasmun21_bin.f:364``), so the field arrives as NUL
    bytes where an assigned ``CHARACTER`` would carry blanks. Stripping one
    class and then the other leaves the first behind wherever it sits inside
    the second."""

    @pytest.mark.parametrize('raw,expected', [
        (b'TITLE\x00\x00   ', 'TITLE'),
        (b'   TITLE\x00\x00', 'TITLE'),
        (b'\x00\x00   TITLE   \x00', 'TITLE'),
        (b'\x00' * 32, ''),
        (b' ' * 32, ''),
        (b'OASS run 3      ', 'OASS run 3'),
    ])
    def test_nuls_and_blanks_come_off_together(self, raw, expected):
        from uacpy.io.oases_reader import _decode_fortran_title
        assert _decode_fortran_title(raw) == expected


class TestOasesLayerRecordsCarryTheATDecksDigits:
    """The OASES bottom record writes speeds, attenuations and density at
    the six decimals the AT decks use, so one Environment gives OAST/OASR
    and Bellhop/Kraken the same seabed: a 1.4449 g/cm³ density and a
    0.0004 dB/λ attenuation reach the deck unrounded."""

    def test_the_halfspace_record_keeps_every_digit(self):
        from uacpy.io.oases_writer import _emit_bottom_layers
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1600.25,
                              density=1.4449, attenuation=0.0004))
        out = io.StringIO()
        _emit_bottom_layers(out, env, 100.0)
        cp, cs, ap, as_, rho = (float(v) for v in out.getvalue().split()[1:6])
        assert (cp, ap, rho) == (1600.25, 0.0004, 1.4449)

    def test_every_sediment_layer_record_keeps_every_digit(self):
        from uacpy.io.oases_writer import _emit_bottom_layers
        hs = BoundaryProperties(acoustic_type='half-space', sound_speed=1800.0,
                                density=2.0, attenuation=0.5)
        layer = SedimentLayer(thickness=5.0, sound_speed=1600.25,
                              shear_speed=100.125, density=1.4449,
                              attenuation=0.0004, shear_attenuation=0.0007)
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=SeabedColumn(layers=[layer], halfspace=hs))
        out = io.StringIO()
        _emit_bottom_layers(out, env, 100.0)
        cp, cs, ap, as_, rho = (float(v) for v in
                                out.getvalue().splitlines()[0].split()[1:6])
        assert (cp, cs, ap, as_, rho) == (1600.25, 100.125, 0.0004, 0.0007,
                                          1.4449)

    def test_the_upper_halfspace_record_keeps_every_digit(self):
        from uacpy.io.oases_writer import _format_upper_halfspace
        ice = BoundaryProperties(acoustic_type='half-space',
                                 sound_speed=3500.25, shear_speed=1800.125,
                                 density=0.9171, attenuation=0.0004,
                                 shear_attenuation=0.0007)
        env = Environment(bathymetry=100.0, ssp=1500.0, surface=ice)
        cp, cs, ap, as_, rho = (float(v) for v in
                                _format_upper_halfspace(env).split()[1:6])
        assert (cp, cs, ap, as_, rho) == (3500.25, 1800.125, 0.0004, 0.0007,
                                          0.9171)
