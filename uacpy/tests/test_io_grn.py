"""The Green's-function files (``uacpy.io.grn_reader``) and the
``GreensFunction`` they are read into: the phase-speed taper, the Hankel
transform and the wavenumber axis.
"""

import numpy as np
import pytest
from uacpy.core.exceptions import ConfigurationError


class TestGrnPhaseSpeedTaper:
    """The cmin/cmax phase-speed taper: valid bands taper the spectrum edges;
    a band with no overlap with the file's phase-speed grid raises a typed
    ConfigurationError instead of a raw broadcast ValueError."""

    def _k(self, freq=100.0, c_lo=1400.0, c_hi=1700.0, nk=64):
        # Wavenumber grid spanning phase speeds [c_lo, c_hi] at ``freq``.
        omega = 2.0 * np.pi * freq
        return np.linspace(omega / c_hi, omega / c_lo, nk)

    def test_interior_band_tapers_edges(self):
        from uacpy.core.acoustics import wavenumber_taper
        win = wavenumber_taper(self._k(), 100.0, cmin=1450.0, cmax=1650.0)
        assert win.shape == (64,)
        assert np.all((win >= 0.0) & (win <= 1.0))
        assert win[0] < 1.0 and win[-1] < 1.0     # rolled off at both edges
        assert np.any(win == 1.0)                 # flat in the middle

    def test_generous_bounds_are_a_noop(self):
        from uacpy.core.acoustics import wavenumber_taper
        win = wavenumber_taper(self._k(), 100.0, cmin=1000.0, cmax=3000.0)
        assert np.all(win == 1.0)

    @pytest.mark.parametrize("cmin,cmax", [
        (None, 100.0),      # cmax below the grid's slowest phase speed
        (5000.0, None),     # cmin above the grid's fastest phase speed
        (1650.0, 1450.0),   # inverted band (cmin > cmax)
    ])
    def test_no_overlap_band_raises_typed(self, cmin, cmax):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.core.acoustics import wavenumber_taper
        with pytest.raises(ConfigurationError, match='phase-speed taper: '):
            wavenumber_taper(self._k(), 100.0, cmin=cmin, cmax=cmax)

    def test_greens_function_to_field_raises_typed(self):
        from uacpy.core.exceptions import ConfigurationError
        gf = _scooter_gf([100.0], np.ones((1, 1, 2, 32), dtype=np.complex64))
        with pytest.raises(ConfigurationError, match="no overlap"):
            gf.to_field(np.array([1000.0]), cmax=100.0)

    def test_to_field_transforms_the_frequency_it_is_given(self):
        """A multi-frequency file transforms the slice ``frequency=``
        names — the same numbers as a single-frequency file holding only
        that slice — and is labelled with it."""
        rng = np.random.default_rng(3)
        G = (rng.standard_normal((2, 1, 2, 32))
             + 1j * rng.standard_normal((2, 1, 2, 32))).astype(np.complex64)
        ranges = np.array([500.0, 1000.0])
        got = _scooter_gf([100.0, 200.0], G).to_field(ranges, frequency=190.0)
        alone = _scooter_gf([200.0], G[1:]).to_field(ranges)
        np.testing.assert_allclose(got.data, alone.data)
        assert float(np.atleast_1d(got.frequencies)[0]) == 200.0

    def test_to_field_refuses_to_pick_a_frequency_itself(self):
        from uacpy.core.exceptions import ConfigurationError
        G = np.ones((2, 1, 2, 32), dtype=np.complex64)
        with pytest.raises(ConfigurationError, match="2 frequencies"):
            _scooter_gf([100.0, 200.0], G).to_field(np.array([1000.0]))


def _scooter_gf(freqs, G, *, title="SCOOTER test", atten=0.0):
    """A frequency-domain GreensFunction on a decreasing phase-speed grid,
    as a ``.grn`` stores it."""
    from uacpy.core.results import GreensFunction
    nk = G.shape[-1]
    return GreensFunction(
        data=G, phase_speeds=np.linspace(1700.0, 1400.0, nk),
        receiver_depths=np.array([25.0, 75.0]), source_depths=[50.0],
        frequencies=freqs, stabilizing_attenuation=atten, title=title)


def _snapshot_gf(nt=8, nk=16, f0=50.0, dt=1e-3):
    from uacpy.core.results import GreensFunction
    rng = np.random.default_rng(5)
    G = (rng.standard_normal((nt, 1, 2, nk))
         + 1j * rng.standard_normal((nt, 1, 2, nk))).astype(np.complex64)
    return GreensFunction(
        data=G, phase_speeds=np.linspace(2500.0, 1400.0, nk),
        receiver_depths=np.array([25.0, 75.0]), source_depths=[50.0],
        frequencies=f0, times=np.arange(nt) * dt,
        title="SPARC-  snapshot", model='SPARC')


class TestGreensFunctionCarriesItsAxes:
    """``GreensFunction`` keeps frequencies and output times apart: a SCOOTER
    file has a frequency axis, a SPARC snapshot a time axis plus its one
    source frequency, and the shape of ``data`` must match them."""

    def test_values_must_match_the_axes(self):
        with pytest.raises(ConfigurationError, match="the axes give"):
            _scooter_gf([100.0, 200.0],
                        np.ones((1, 1, 2, 8), dtype=np.complex64))

    def test_values_must_be_four_dimensional(self):
        from uacpy.core.results import GreensFunction
        with pytest.raises(ConfigurationError, match="must be 4-D"):
            GreensFunction(data=np.ones((2, 8)), phase_speeds=np.ones(8),
                           receiver_depths=[1.0, 2.0], frequencies=100.0)

    def test_a_snapshot_carries_one_source_frequency(self):
        from uacpy.core.results import GreensFunction
        with pytest.raises(ConfigurationError, match="one source frequency"):
            GreensFunction(data=np.ones((3, 1, 1, 4)),
                           phase_speeds=np.ones(4), receiver_depths=[1.0],
                           source_depths=[5.0], frequencies=[50.0, 60.0],
                           times=[0.0, 1.0, 2.0])

    def test_a_snapshot_is_told_by_its_times(self):
        assert _snapshot_gf().is_snapshot
        assert not _scooter_gf([100.0], np.ones((1, 1, 2, 8))).is_snapshot

    def test_frequency_domain_transforms_refuse_a_snapshot(self):
        gf = _snapshot_gf()
        for call in (gf.to_field, gf.to_transfer_function):
            with pytest.raises(ConfigurationError, match="output TIMES"):
                call(np.array([100.0]))

    def test_snapshot_transforms_refuse_a_frequency_domain_file(self):
        gf = _scooter_gf([100.0], np.ones((1, 1, 2, 8), dtype=np.complex64))
        with pytest.raises(ConfigurationError, match="expects a SPARC"):
            gf.snapshot_to_field(np.array([100.0]), 100.0, normalize=None)
        with pytest.raises(ConfigurationError, match="expects a SPARC"):
            gf.snapshot_to_time_field(np.array([100.0]))

    def test_source_normalisation_needs_the_source_waveform(self):
        with pytest.raises(ConfigurationError, match="needs source_waveform"):
            _snapshot_gf().snapshot_to_field(np.array([100.0]), 50.0)

    def test_a_source_waveform_of_another_length_is_refused(self):
        gf = _snapshot_gf(nt=8)
        with pytest.raises(ConfigurationError, match="8 output times"):
            gf.snapshot_to_field(np.array([100.0]), 50.0,
                                 source_waveform=np.ones(7))
        gf.snapshot_to_field(np.array([100.0]), 50.0,
                             source_waveform=np.ones(8))

    @pytest.mark.parametrize("make", [
        lambda: _scooter_gf([100.0, 200.0],
                            np.full((2, 1, 2, 8), 1 + 2j, np.complex64),
                            atten=0.25),
        _snapshot_gf,
    ])
    def test_to_dict_round_trips_through_savez(self, make, tmp_path):
        from uacpy.core.results import GreensFunction
        gf = make()
        path = tmp_path / 'gf.npz'
        np.savez(path, **gf.to_dict())
        with np.load(path, allow_pickle=True) as f:
            back = GreensFunction.from_dict(dict(f))
        np.testing.assert_array_equal(back.data, gf.data)
        assert back.data.dtype == gf.data.dtype
        for name in ('phase_speeds', 'receiver_depths', 'frequencies',
                     'source_depths'):
            np.testing.assert_array_equal(getattr(back, name),
                                          getattr(gf, name))
        if gf.times is None:
            assert back.times is None
        else:
            np.testing.assert_array_equal(back.times, gf.times)
        assert back.stabilizing_attenuation == gf.stabilizing_attenuation
        assert back.title == gf.title and back.model == gf.model


class TestHankelScaledCylindrical:
    """fieldsco.m:133 — 'S' is 'R' with cylindrical spreading removed."""

    def test_scaled_equals_point_times_sqrt_r(self):
        from uacpy.core.acoustics import hankel_transform
        rng = np.random.default_rng(0)
        k = np.linspace(0.1, 1.0, 64)
        G = rng.normal(size=(2, 64)) + 1j * rng.normal(size=(2, 64))
        # Ranges inside the transform's alias bound r*dk <= 10
        # (fieldsco.m:109-111): dk ~ 0.0143 here, so r_max 400 m is safe.
        ranges = np.array([100.0, 200.0, 400.0])
        kw = dict(attenuation=0.0, spectrum='positive')
        p_point = hankel_transform(G, k, ranges, source_type='point', **kw)
        p_scaled = hankel_transform(G, k, ranges, source_type='scaled', **kw)
        np.testing.assert_allclose(
            p_scaled, p_point * np.sqrt(ranges)[None, :], rtol=1e-12)

    def test_line_is_the_plain_cartesian_transform(self):
        """'line' (fieldsco.m 'X') carries no sqrt(k) weight, no -pi/4 phase
        and no spreading: -(sum_k G(k) exp(-ikr)) dk / sqrt(2 pi)."""
        from uacpy.core.acoustics import hankel_transform
        rng = np.random.default_rng(1)
        k = np.linspace(0.1, 1.0, 64)
        G = rng.normal(size=(2, 64)) + 1j * rng.normal(size=(2, 64))
        ranges = np.array([100.0, 200.0, 400.0])
        p_line = hankel_transform(G, k, ranges, source_type='line',
                                  attenuation=0.0)
        dk = k[1] - k[0]
        want = -(G @ np.exp(-1j * np.outer(k, ranges))) * dk / np.sqrt(2 * np.pi)
        np.testing.assert_allclose(p_line, want, rtol=1e-12)

    def test_unknown_source_type_raises(self):
        from uacpy.core.acoustics import hankel_transform
        with pytest.raises(ConfigurationError,
                           match="source_type must be 'point', 'line', or 'scaled'"):
            hankel_transform(
                np.zeros((1, 4), dtype=complex), np.linspace(0.1, 1.0, 4),
                np.array([100.0]), attenuation=0.0, source_type='Q', spectrum='positive')


class TestGreensFunctionWavenumberAxis:
    """``GreensFunction.wavenumbers`` gives the ``k = 2πf/c`` axis a ``.grn``
    was sampled on, through ``wavenumbers_from_phase_speeds``: per frequency
    for Scooter, at the header's source frequency for SPARC (whose first
    axis holds times)."""

    C = np.array([2000.0, 1600.0, 1400.0])

    def _gf(self, *, snapshot, freqs=(50.0,)):
        from uacpy.core.results import GreensFunction
        n = 4 if snapshot else len(freqs)
        return GreensFunction(
            data=np.zeros((n, 1, 1, 3), np.complex64), phase_speeds=self.C,
            receiver_depths=[10.0], source_depths=[5.0],
            frequencies=list(freqs),
            times=np.arange(4) * 0.01 if snapshot else None)

    def test_the_function_is_two_pi_f_over_c(self):
        from uacpy.core.acoustics import wavenumbers_from_phase_speeds
        np.testing.assert_array_equal(
            wavenumbers_from_phase_speeds(self.C, 200.0),
            2.0 * np.pi * 200.0 / self.C)

    def test_scooter_axis_follows_the_frequency_asked_for(self):
        k = self._gf(snapshot=False, freqs=(50.0, 200.0)).wavenumbers(200.0)
        np.testing.assert_array_equal(k, 2.0 * np.pi * 200.0 / self.C)

    def test_a_single_frequency_file_needs_no_frequency(self):
        k = self._gf(snapshot=False).wavenumbers()
        np.testing.assert_array_equal(k, 2.0 * np.pi * 50.0 / self.C)

    def test_a_multi_frequency_file_needs_the_frequency(self):
        with pytest.raises(ConfigurationError, match="say which one"):
            self._gf(snapshot=False, freqs=(50.0, 200.0)).wavenumbers()

    def test_sparc_axis_uses_the_source_frequency(self):
        k = self._gf(snapshot=True).wavenumbers()
        np.testing.assert_array_equal(k, 2.0 * np.pi * 50.0 / self.C)

    def test_sparc_axis_takes_no_frequency(self):
        with pytest.raises(ConfigurationError, match="takes|without"):
            self._gf(snapshot=True).wavenumbers(0.37)


class TestGrnHeaderCountsAreBoundedEach:
    """``read_grn_file`` bounds every header count, and their product, by the
    file size (``_bound_counts``): a count past what the file can hold is
    refused even when another count is zero and the product is not."""

    RECORD_WORDS = 41                 # the file's own floor (RWSHDFile.f90:100)

    def _write(self, path, *, nfreq, nk):
        import struct
        record_bytes = 4 * self.RECORD_WORDS
        buf = bytearray()

        def record(payload):
            buf.extend(payload.ljust(record_bytes, b'\x00'))

        record(struct.pack('<i', self.RECORD_WORDS)
               + b'SCOOTER- test'.ljust(80, b' '))
        record(b'Green'.ljust(10, b' '))
        record(struct.pack('<7i', nfreq, 1, 1, 1, 1, 1, nk)
               + struct.pack('<dd', 100.0, 0.0))
        for _ in range(7):            # records 4-10
            record(b'')
        path.write_bytes(bytes(buf))
        return len(buf)

    def test_a_count_at_the_file_capacity_passes_the_bound(self, tmp_path):
        # The bound passes; the read then stops at the phase-speed record,
        # which one 41-word record cannot fill with size // 8 values.
        from uacpy.core.exceptions import FileFormatError
        from uacpy.io.grn_reader import read_grn_file
        path = tmp_path / 'at.grn'
        size = self._write(path, nfreq=0, nk=1)
        self._write(path, nfreq=0, nk=size // 8)
        with pytest.raises(FileFormatError,
                           match=f'phase-speed record 10 .expected '
                                 f'{size // 8} values'):
            read_grn_file(str(path))

    def test_a_count_that_fits_its_record_is_read(self, tmp_path):
        from uacpy.io.grn_reader import read_grn_file
        path = tmp_path / 'fits.grn'
        self._write(path, nfreq=0, nk=self.RECORD_WORDS // 2)
        gf = read_grn_file(str(path))
        assert len(gf.phase_speeds) == self.RECORD_WORDS // 2
        assert gf.data.shape == (0, 1, 1, self.RECORD_WORDS // 2)

    def test_a_count_one_past_the_file_capacity_is_refused(self, tmp_path):
        from uacpy.core.exceptions import FileFormatError
        from uacpy.io.grn_reader import read_grn_file
        path = tmp_path / 'past.grn'
        size = self._write(path, nfreq=0, nk=1)
        self._write(path, nfreq=0, nk=size // 8 + 1)
        with pytest.raises(FileFormatError, match='nk=.*implausible'):
            read_grn_file(str(path))
