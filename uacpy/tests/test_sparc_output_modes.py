"""Tests for SPARC's run mode and output mode.

SPARC is a time-domain code: it marches a pulse and its product is ``p(t)``.
Extracting CW transmission loss from that pulse is not quantitative — ``sparc.f90:313`` sets ``Atten = 0`` so the wavenumber
sum has no contour offset to move it off the real-axis modal poles (Scooter
uses ``Atten = Deltak``, ``scooter.f90:129``); ``Nk`` is sized across the whole
pulse band so only a fraction of the samples land in the analysis frequency's
window; and the default ``pulse_type='PN+B'`` band-pass is per-wavenumber while
``rts_to_pressure`` deconvolves a single scalar ``S(omega0)``. Measured on a
guide with exactly one propagating mode — where the exact TL is smooth and
monotone — SPARC shows 2.4 dB median error with 13 dB excursions against
Scooter's 0.07 dB, and converges under no grid setting.

``RunMode.COHERENT_TL`` is therefore withdrawn. ``output_mode`` is **not** a CW
concept: ``sparc.f90:579-609`` shows ``TopOpt(5:5)`` selecting three
*time-domain* outputs — ``'R'`` = RTS (horizontal array), ``'D'`` = RTS
(vertical array, ``RTSrz(ir,Itout)``), ``'S'`` = snapshot. ``'R'`` and ``'D'``
both return received time series. ``'S'`` writes a wavenumber-domain Green's function
(``Green(Itout, irz, ik)``); ``doc/sparc.htm`` says FIELDS must be run
afterwards to turn it into a pressure field, which uacpy does in-tree with one
inverse Hankel transform per output time. All three modes return received time
series and are cross-checked against each other here.
"""

import contextlib
import struct
from pathlib import Path

import pytest
import numpy as np

from uacpy import Environment, Source, Receiver, BoundaryProperties
from uacpy import Field
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError, UnsupportedFeatureError,
)
from uacpy.io.grn_reader import read_grn_file
from uacpy.models import SPARC
from uacpy.models.base import RunMode


@pytest.fixture
def sparc_simple_env():
    """Isovelocity environment configured for SPARC (vacuum bottom).

    Distinct name from the conftest ``simple_env`` so SPARC's vacuum-bottom
    requirement does not shadow the shared half-space fixture.
    """
    return Environment(
        name="Test Environment",
        bathymetry=100.0,
        ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='vacuum'),
    )


@pytest.fixture
def source_50hz():
    return Source(depths=50.0, frequencies=50.0)


@pytest.fixture
def receiver_grid():
    return Receiver(depths=np.array([30.0, 50.0, 70.0]),
                    ranges=np.linspace(100, 1000, 10))


class TestSPARCRunModeSurface:
    """What SPARC advertises must be what it delivers."""

    def test_time_series_is_the_only_supported_mode(self):
        assert SPARC.spec.modes == (RunMode.TIME_SERIES,)

    def test_time_series_is_the_default(self, sparc_simple_env, source_50hz,
                                        receiver_grid):
        m = SPARC(verbose=False)
        assert m._resolve_run_mode(None) == RunMode.TIME_SERIES

    @pytest.mark.requires_binary
    def test_coherent_tl_is_refused_with_a_usable_alternative(
            self, sparc_simple_env, source_50hz, receiver_grid):
        with pytest.raises(UnsupportedFeatureError) as ei:
            SPARC(verbose=False).run(sparc_simple_env, source_50hz,
                                     receiver_grid,
                                     run_mode=RunMode.COHERENT_TL)
        assert 'Scooter' in str(ei.value) or 'Kraken' in str(ei.value)


class TestSPARCOutputMode:
    """'R' and 'D' are both received time series; only 'S' is unavailable."""

    def test_unknown_output_mode_is_rejected(self):
        with pytest.raises(ConfigurationError):
            SPARC(verbose=False, output_mode='Z')

    @pytest.mark.requires_binary
    def test_vertical_array_returns_a_time_series(self, sparc_simple_env,
                                                  source_50hz):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            res = SPARC(verbose=False, output_mode='D').run(
                sparc_simple_env, source_50hz,
                Receiver(depths=np.array([30.0, 50.0, 70.0]),
                         ranges=np.array([1000.0])))
        assert isinstance(res, Field)
        assert 'time' in res.coords
        assert res.data.shape == (3, 1, len(res.coords['time']))
        assert np.all(np.isfinite(res.data)) and np.any(res.data != 0)

    @pytest.mark.requires_binary
    def test_vertical_and_horizontal_report_the_same_field(
            self, sparc_simple_env, source_50hz):
        """``Scooter/sparc.f90:292`` scales 'D' by 1/sqrt(pi*Rr) where 'R'
        carries 1/sqrt(r), so without a sqrt(pi) correction the vertical array
        reads 1/sqrt(pi) = 0.5642 of the horizontal one for the identical
        field. The 5% bound is set an order of magnitude below that 43.6%
        signature, so it catches a dropped or doubled sqrt(pi) while leaving
        room for the two paths' differing time interpolation.
        """
        import warnings as _w
        depths = np.array([30.0, 50.0, 70.0])
        rng = np.array([1000.0])
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            d = np.asarray(SPARC(verbose=False, output_mode='D').run(
                sparc_simple_env, source_50hz,
                Receiver(depths=depths, ranges=rng)).data)
            r = np.asarray(SPARC(verbose=False, output_mode='R').run(
                sparc_simple_env, source_50hz,
                Receiver(depths=depths, ranges=rng)).data)
        for i in range(len(depths)):
            peak_r = np.abs(r[i, 0]).max()
            assert np.abs(d[i, 0]).max() / peak_r == pytest.approx(1.0, abs=0.05)
            assert np.abs(d[i, 0] - r[i, 0]).max() / peak_r < 0.05

    @pytest.mark.requires_binary
    def test_horizontal_mode_returns_a_time_series(
            self, sparc_simple_env, source_50hz):
        import warnings as _w
        rcv = Receiver(depths=np.array([30.0, 50.0]),
                       ranges=np.array([1000.0]))
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            res = SPARC(verbose=False, output_mode='R').run(
                sparc_simple_env, source_50hz, rcv)
        assert isinstance(res, Field)
        assert 'time' in res.coords, f"expected a time axis; got {list(res.coords)}"
        assert np.all(np.isfinite(res.data))
        assert np.any(res.data != 0)


class TestSPARCEnvironmentHandling:
    """Behaviour that is independent of the withdrawn CW path."""

    @pytest.mark.requires_binary
    def test_halfspace_bottom_is_rigidified_with_a_warning(
            self, source_50hz, receiver_grid):
        """SPARC's writer supports only vacuum / rigid, so a half-space must be
        force-rigidified loudly rather than silently reinterpreted."""
        env = Environment(
            name='hs', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))
        with pytest.warns(UserWarning, match="rigid"):
            SPARC(verbose=False).run(
                env, source_50hz,
                Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])))

    def test_oversized_depth_axis_raises_before_any_launch(
            self, sparc_simple_env, source_50hz, monkeypatch):
        """``output_mode='R'`` runs one subprocess per receiver depth, so a
        depth axis past ``max_depths`` (default 20) raises
        ``UnsupportedFeatureError`` rather than queueing hours of solves —
        and it raises before any binary is launched."""
        m = SPARC(verbose=False, output_mode='R')
        monkeypatch.setattr(
            m, "_run_sparc",
            lambda *a, **k: pytest.fail("binary launched before the "
                                        "max_depths cap fired"))
        with pytest.raises(UnsupportedFeatureError, match='max_depths'):
            m.run(sparc_simple_env, source_50hz,
                  Receiver(depths=np.linspace(10.0, 90.0, 21),
                           ranges=np.array([1000.0])))

    def test_oversized_range_axis_raises_before_any_launch(
            self, sparc_simple_env, source_50hz, monkeypatch):
        """``'D'`` loops the binary per receiver range; the same cap guards
        that axis."""
        m = SPARC(verbose=False, output_mode='D')
        monkeypatch.setattr(
            m, "_run_sparc",
            lambda *a, **k: pytest.fail("binary launched before the "
                                        "max_depths cap fired"))
        with pytest.raises(UnsupportedFeatureError, match='max_depths'):
            m.run(sparc_simple_env, source_50hz,
                  Receiver(depths=np.array([50.0]),
                           ranges=np.linspace(100.0, 2100.0, 21)))

    @pytest.mark.requires_binary
    def test_one_binary_run_per_receiver_range_in_vertical_mode(
            self, sparc_simple_env, source_50hz, monkeypatch):
        """The vertical path loops the binary once per range and records the
        loop count in the result metadata."""
        import warnings as _w
        m = SPARC(verbose=False, output_mode='D')
        calls = {'n': 0}
        original = m._run_sparc

        def counting(*args, **kwargs):
            calls['n'] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(m, "_run_sparc", counting)
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            res = m.run(sparc_simple_env, source_50hz,
                        Receiver(depths=np.array([50.0]),
                                 ranges=np.array([600.0, 800.0, 1000.0])))
        assert calls['n'] == 3, f"expected one run per range, got {calls['n']}"
        assert res.metadata['n_range_runs'] == 3

    @pytest.mark.requires_binary
    def test_one_binary_run_per_receiver_depth(self, sparc_simple_env,
                                               source_50hz, monkeypatch):
        """The horizontal path loops the binary once per depth; the max_depths
        cap exists because of it."""
        import warnings as _w
        m = SPARC(verbose=False)
        calls = {'n': 0}
        original = m._run_sparc

        def counting(*args, **kwargs):
            calls['n'] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(m, "_run_sparc", counting)
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            m.run(sparc_simple_env, source_50hz,
                  Receiver(depths=np.array([30.0, 50.0, 70.0]),
                           ranges=np.array([1000.0])))
        assert calls['n'] == 3, f"expected one run per depth, got {calls['n']}"


@pytest.mark.requires_binary
class TestSPARCSnapshot:
    """``output_mode='S'`` — the whole field at each output time.

    SPARC writes the wavenumber-domain field per output time; the inverse
    Hankel transform per time recovers ``p(z, r, t)`` on the receiver grid in
    a *single* binary run, where 'R'/'D' loop per depth / per range.
    """

    @staticmethod
    def _env():
        return Environment(bathymetry=100.0, ssp=1500.0,
                           bottom=BoundaryProperties(acoustic_type='vacuum'))

    def test_snapshot_returns_the_full_grid_in_one_run(self):
        import warnings as _w
        depths = np.array([30.0, 50.0, 70.0])
        rngs = np.linspace(500.0, 2000.0, 4)
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            res = SPARC(verbose=False, output_mode='S').run(
                self._env(), Source(depths=50.0, frequencies=50.0),
                Receiver(depths=depths, ranges=rngs))
        assert set(res.coords) == {'depth', 'range', 'time'}
        assert res.data.shape == (3, 4, len(res.coords['time']))
        assert np.all(np.isfinite(res.data)) and np.any(res.data != 0)

    def test_snapshot_matches_the_horizontal_array(self):
        """Same field by a different route: the snapshot Hankel-transforms the
        wavenumber field while 'R' uses sparc.f90's own range synthesis. The
        conventions differ by exactly -2*sqrt(pi) (measured signed ratio
        -3.5449, std 0.0000), which the transform undoes. As in the 'D'-vs-'R'
        check, 5% sits far below the factor a mis-applied convention would
        leave behind."""
        import warnings as _w
        depths = np.array([30.0, 50.0])
        rngs = np.array([1000.0, 1500.0])
        src = Source(depths=50.0, frequencies=50.0)
        rcv = Receiver(depths=depths, ranges=rngs)
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            S = np.asarray(SPARC(verbose=False, output_mode='S').run(
                self._env(), src, rcv).data)
            R = np.asarray(SPARC(verbose=False, output_mode='R').run(
                self._env(), src, rcv).data)
        for i in range(len(depths)):
            for j in range(len(rngs)):
                peak = np.abs(R[i, j]).max()
                # Arrival timing must agree bin-for-bin.
                assert np.argmax(np.abs(S[i, j])) == np.argmax(np.abs(R[i, j]))
                assert np.abs(S[i, j] - R[i, j]).max() / peak < 0.05


# ── the snapshot's one DFT row ──────────────────────────────────────────────
#
# `sparc_snapshot_to_field` keeps one DFT row without transforming the rest.
# `np.fft.fft(cube, axis=0)[f_idx]` transforms every one of the `nt`
# frequencies and keeps one — 5.1x the cube in peak RSS, ~17 GB on a
# 512 x 200 x 4096 snapshot, for an (nrd, nk) slab. The row is a contraction
# against `exp(-2i pi f_idx n / nt)`, so what is pinned below is the equality,
# not the saving.
#
# The two branches of the reader are NOT entitled to the same precision, and
# that is the trap here. `np.fft.fft` on a complex64 cube is a
# single-precision transform, so the rectangular branch trades round-off for
# round-off; but `G * win[:, None, None]` with a float64 window promotes the
# cube first, so the windowed branch is a DOUBLE-precision transform today and
# a single-precision contraction there would be an accuracy regression rather
# than an equivalent.


def _rel(ref, got):
    ref = np.asarray(ref, np.complex128)
    got = np.asarray(got, np.complex128)
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(ref - got) / denom) if denom else 0.0


def _full_fft_row(G, f_idx, win=None):
    """The expression ``_dft_row`` replaced, kept here as the reference."""
    if win is None:
        return np.fft.fft(G, axis=0)[f_idx, :, :]
    return np.fft.fft(G * win[:, np.newaxis, np.newaxis], axis=0)[f_idx, :, :]


class TestSparcSnapshotKeepsOneDftRowWithoutTheOthers:
    """Row against full transform, on the bins that bracket the axis: DC, a
    quarter in, the Nyquist bin and the last one."""

    @staticmethod
    def _cube(nt=64, nrd=5, nk=32, seed=0):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((nt, nrd, nk)) +
                1j * rng.standard_normal((nt, nrd, nk))).astype(np.complex64)

    @staticmethod
    def _bins(nt):
        return (0, nt // 4, nt // 2, nt - 1)

    def test_the_rectangular_row_matches_the_full_single_precision_fft(self):
        from uacpy.io.grn_reader import _dft_row
        G = self._cube()
        for f_idx in self._bins(G.shape[0]):
            ref = _full_fft_row(G, f_idx)
            got = _dft_row(G, f_idx)
            assert got.dtype == ref.dtype == np.complex64
            assert got.shape == ref.shape
            assert _rel(ref, got) < 1e-6

    def test_the_windowed_row_holds_the_double_precision_the_upcast_gives_it(
            self):
        from uacpy.io.grn_reader import _dft_row
        G = self._cube()
        win = np.hanning(G.shape[0])
        for f_idx in self._bins(G.shape[0]):
            ref = _full_fft_row(G, f_idx, win)
            got = _dft_row(G, f_idx, win)
            assert got.dtype == ref.dtype == np.complex128
            assert _rel(ref, got) < 1e-12

    def test_a_single_precision_kernel_would_lose_five_decades_there(self):
        # Why the windowed path carries a complex128 kernel: contracting in
        # single precision is ~1e-7, five orders coarser than the transform it
        # stands in for. Correct for the rectangular branch, a regression here.
        G = self._cube()
        nt = G.shape[0]
        win = np.hanning(nt)
        kernel = np.exp(-2j * np.pi * (nt // 3) * np.arange(nt) / nt) * win
        ref = _full_fft_row(G, nt // 3, win)
        single = np.tensordot(kernel.astype(np.complex64), G, axes=1)
        assert _rel(ref, single) > 1e-8

    def test_chunking_the_wavenumber_axis_leaves_the_row_unchanged(self):
        # nk past the internal 4e6-element block, so the windowed path takes
        # more than one chunk and the block arithmetic is exercised.
        from uacpy.io.grn_reader import _dft_row
        G = self._cube(nt=64, nrd=8, nk=9000, seed=1)
        assert G.shape[0] * G.shape[1] * G.shape[2] > 4_000_000
        win = np.hanning(G.shape[0])
        f_idx = 21
        np.testing.assert_allclose(
            _dft_row(G, f_idx, win), _full_fft_row(G, f_idx, win),
            rtol=1e-12, atol=1e-12)

    def test_a_double_precision_cube_stays_in_double_precision(self):
        from uacpy.io.grn_reader import _dft_row
        G = self._cube().astype(np.complex128)
        ref = _full_fft_row(G, 9)
        got = _dft_row(G, 9)
        assert got.dtype == ref.dtype == np.complex128
        assert _rel(ref, got) < 1e-13


class TestSparcSnapshotFieldIsUnchangedByTheRowContraction:
    """End to end: the same snapshot through ``sparc_snapshot_to_field`` with
    the row contraction and with the full transform it replaced, including the
    ``/S(f0)`` deconvolution, the ``2/sum(win)`` estimator scale and the
    Hankel transform that follow it.
    """

    @staticmethod
    def _grn(nt=64, nrd=6, nk=48, freq=25.0, seed=3):
        rng = np.random.default_rng(seed)
        return {
            'is_sparc': True,
            'title': 'SPARC-  synthetic snapshot',
            'nfreq': nt, 'nsd': 1, 'nrd': nrd, 'nk': nk,
            'freq': freq,
            'freqVec': np.arange(nt) / (200.0 * freq),   # output TIMES
            'sd': np.array([20.0]),
            'rd': np.linspace(10.0, 200.0, nrd),
            'cVec': np.linspace(2500.0, 1400.0, nk),
            'atten': 0.0,
            'G': (rng.standard_normal((nt, 1, nrd, nk)) +
                  1j * rng.standard_normal((nt, 1, nrd, nk))
                  ).astype(np.complex64),
        }

    @staticmethod
    def _field(grn, ranges, normalize, pulse_type):
        from uacpy.io.grn_reader import sparc_snapshot_to_field
        # normalize='none' always announces its uncalibrated level.
        ctx = (pytest.warns(UserWarning, match='RAW field')
               if normalize == 'none' else contextlib.nullcontext())
        with ctx:
            return sparc_snapshot_to_field(
                grn, ranges, 25.0, normalize=normalize, pulse_type=pulse_type)

    @pytest.mark.parametrize('normalize,pulse_type', [('source', 'R'),
                                                      ('none', None)])
    def test_the_field_matches_the_full_fft_route(self, monkeypatch,
                                                  normalize, pulse_type):
        import uacpy.io.grn_reader as gr
        grn = self._grn()
        ranges = np.linspace(200.0, 5000.0, 12)
        new = self._field(grn, ranges, normalize, pulse_type)
        monkeypatch.setattr(gr, '_dft_row', _full_fft_row)
        ref = self._field(grn, ranges, normalize, pulse_type)
        # Single precision on the rectangular branch, double on the windowed
        # one — the same split the row helper is pinned to above.
        tol = 1e-6 if normalize == 'source' else 1e-12
        assert _rel(ref.data, new.data) < tol


def _write_grn(path, *, title, nsd, nfreq=2, nrd=1, nk=4):
    """A minimal well-formed ``.grn`` with ``nsd`` source depths.

    Built by hand rather than by running a binary: the point under test is how
    the reader responds to a header, and fabricating one reaches the check
    without a Fortran run. Layout follows ``RWSHDFile.f90`` — record 1 carries
    ``recl`` (in 4-byte words) and the 80-char title, record 3 the seven
    counts, record 10 the phase-speed vector, and the Green's-function slabs
    follow from record 11.
    """
    recl = 41                       # words; the file's own floor
    record_bytes = recl * 4
    buf = bytearray()

    def record(payload):
        assert len(payload) <= record_bytes, (len(payload), record_bytes)
        buf.extend(payload.ljust(record_bytes, b'\x00'))

    record(struct.pack('<i', recl) + title.encode().ljust(80, b' '))
    record(b'Green'.ljust(10, b' '))
    record(struct.pack('<7i', nfreq, 1, 1, 1, nsd, nrd, nk)
           + struct.pack('<dd', 100.0, 0.0))
    record(np.arange(nfreq, dtype='<f8').tobytes())        # freqVec / times
    record(np.zeros(1, dtype='<f8').tobytes())             # theta
    record(np.zeros(1, dtype='<f4').tobytes())             # Sx
    record(np.zeros(1, dtype='<f4').tobytes())             # Sy
    record(np.zeros(max(nsd, 1), dtype='<f4').tobytes())   # Sz
    record(np.zeros(nrd, dtype='<f4').tobytes())           # Rz
    record(np.full(nk, 1500.0, dtype='<f8').tobytes())     # cVec
    for _ in range(nfreq * nsd * nrd):
        record(np.zeros(2 * nk, dtype='<f4').tobytes())

    Path(path).write_bytes(bytes(buf))


class TestSparcSnapshotTakesOneSourceDepth:
    """``sparc.f90:286-287`` indexes a snapshot as ``iG = (Itout-1)*NRz + ir``
    — output time stands in for the frequency axis (``WriteHeaderSparc`` sets
    ``Nfreq = Ntout``, sparc.f90:318-320) and there is no source-depth factor,
    because a snapshot's ``Green`` carries no source-depth axis.
    ``WriteHeaderSparc`` never rewrites ``Pos%NSz``, though, so the header
    still reports whatever the deck asked for.

    ``SPARC.run`` refuses a multi-depth Source, which is what keeps the
    reader's sequential walk correct; this pins the reader's own half of that
    coupling, so relaxing the guard upstream cannot silently mis-shape a field.
    Without the check the walk would read time slot 2's slabs into time slot
    1's second source before running off the end of the file.
    """

    def test_a_single_source_depth_snapshot_reads(self, tmp_path):
        path = tmp_path / 'ok.grn'
        _write_grn(path, title='SPARC-   snapshot', nsd=1)
        out = read_grn_file(str(path))
        assert out['is_sparc'] is True
        assert out['G'].shape == (2, 1, 1, 4)

    def test_a_multi_source_depth_snapshot_is_refused(self, tmp_path):
        path = tmp_path / 'bad.grn'
        _write_grn(path, title='SPARC-   snapshot', nsd=2)
        with pytest.raises(FileFormatError, match='NSz=2'):
            read_grn_file(str(path))

    def test_a_scooter_header_may_carry_several_source_depths(self, tmp_path):
        # SCOOTER's records DO carry the NSz factor (scooter.f90:588), so the
        # same header is legal there and must keep working.
        path = tmp_path / 'scooter.grn'
        _write_grn(path, title='SCOOTER- test', nsd=2)
        out = read_grn_file(str(path))
        assert out['is_sparc'] is False
        assert out['G'].shape == (2, 2, 1, 4)


def _write_sparc_grn(path, *, title, nfreq=3, nsd=1, nrd=1, nk=4):
    """A minimal well-formed ``.grn`` in the ``RWSHDFile.f90`` layout.

    Record 1 carries ``recl`` (in 4-byte words) and the 80-char title,
    record 3 the seven counts, record 10 the phase-speed vector, and the
    Green's-function slabs follow from record 11. The title's prefix is what
    marks a SPARC snapshot, so it is the parameter under test.
    """
    recl = 41
    record_bytes = recl * 4
    buf = bytearray()

    def record(payload):
        assert len(payload) <= record_bytes, (len(payload), record_bytes)
        buf.extend(payload.ljust(record_bytes, b'\x00'))

    record(struct.pack('<i', recl) + title.encode().ljust(80, b' '))
    record(b'Green'.ljust(10, b' '))
    record(struct.pack('<7i', nfreq, 1, 1, 1, nsd, nrd, nk)
           + struct.pack('<dd', 100.0, 0.0))
    record(np.arange(1, nfreq + 1, dtype='<f8').tobytes())   # freqVec / times
    record(np.zeros(1, dtype='<f8').tobytes())               # theta
    record(np.zeros(1, dtype='<f4').tobytes())               # Sx
    record(np.zeros(1, dtype='<f4').tobytes())               # Sy
    record(np.zeros(max(nsd, 1), dtype='<f4').tobytes())     # Sz
    record(np.zeros(nrd, dtype='<f4').tobytes())             # Rz
    record(np.full(nk, 1500.0, dtype='<f8').tobytes())       # cVec
    for _ in range(nfreq * nsd * nrd):
        record(np.zeros(2 * nk, dtype='<f4').tobytes())

    Path(path).write_bytes(bytes(buf))


class TestBroadbandTransformRefusesASparcSnapshot:
    """``WriteHeaderSparc`` puts the output TIME vector in the ``.grn``'s
    frequency slot (``sparc.f90:317-319``), so transforming a snapshot as a
    multi-frequency Green's function labels seconds as hertz and reports a
    ``center_frequency`` that is a time. The two snapshot readers already
    refuse a non-SPARC ``.grn``; this is the same guard the other way round."""

    def test_a_sparc_grn_names_the_two_snapshot_readers(self, tmp_path):
        from uacpy.io.grn_reader import read_grn_file, grn_to_transfer_function
        path = tmp_path / 'snap.grn'
        _write_sparc_grn(path, title='SPARC-   snapshot')
        grn = read_grn_file(str(path))
        assert grn['is_sparc'] is True
        with pytest.raises(ConfigurationError,
                           match='sparc_snapshot_to_time_field'):
            grn_to_transfer_function(grn, np.array([1000.0]))

    def test_a_scooter_grn_transforms_onto_its_frequency_axis(self, tmp_path):
        from uacpy.io.grn_reader import read_grn_file, grn_to_transfer_function
        path = tmp_path / 'broad.grn'
        _write_sparc_grn(path, title='SCOOTER- broadband')
        grn = read_grn_file(str(path))
        field = grn_to_transfer_function(grn, np.array([1000.0]))
        assert np.array_equal(field.coords['frequency'], grn['freqVec'])
