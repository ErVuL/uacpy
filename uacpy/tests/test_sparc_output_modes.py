"""Tests for SPARC's run mode and output mode.

SPARC is a time-domain code: it marches a pulse and its product is ``p(t)``.
CW transmission loss used to be extracted from that pulse, but the extraction
is not quantitative — ``sparc.f90:313`` sets ``Atten = 0`` so the wavenumber
sum has no contour offset to move it off the real-axis modal poles (Scooter
uses ``Atten = Deltak``, ``scooter.f90:129``); ``Nk`` is sized across the whole
pulse band so only a fraction of the samples land in the analysis frequency's
window; and the default ``pulse_type='PN+B'`` band-pass is per-wavenumber while
``rts_to_pressure`` deconvolves a single scalar ``S(omega0)``. Measured on a
guide with exactly one propagating mode — where the exact TL is smooth and
monotone — SPARC gave 2.4 dB median error with 13 dB excursions against
Scooter's 0.07 dB, and did not converge under any grid setting.

``RunMode.COHERENT_TL`` is therefore withdrawn. ``output_mode`` is **not** a CW
concept: ``sparc.f90:579-609`` shows ``TopOpt(5:5)`` selecting three
*time-domain* outputs — ``'R'`` = RTS (horizontal array), ``'D'`` = RTS
(vertical array, ``RTSrz(ir,Itout)``), ``'S'`` = snapshot. ``'R'`` and ``'D'``
both return received time series. ``'S'`` writes a wavenumber-domain Green's function
(``Green(Itout, irz, ik)``); ``doc/sparc.htm`` says FIELDS must be run
afterwards to turn it into a pressure field, which uacpy now does in-tree with
one inverse Hankel transform per output time. All three modes return received
time series and are cross-checked against each other here.
"""

import pytest
import numpy as np

from uacpy import Environment, Source, Receiver, BoundaryProperties
from uacpy import Field
from uacpy.core.exceptions import ConfigurationError, UnsupportedFeatureError
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
        """sparc.f90:292 scales 'D' by 1/sqrt(pi*Rr) where 'R' carries
        1/sqrt(r), so without a sqrt(pi) correction the vertical array reads
        1/sqrt(pi) = 0.5642 of the horizontal one for the identical field."""
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
        -3.5449, std 0.0000), which the transform undoes."""
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
