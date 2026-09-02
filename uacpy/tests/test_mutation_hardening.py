"""Numeric-contract pins for formula-bearing helpers.

Each test anchors a computation to an independent reference — a published
check value, a closed-form identity, a symmetry, or a round-trip — chosen
so that any algebra slip in the implementation (a sign, a factor, a
boundary) moves the result well past the tolerance. Reference values are
cross-checked against the literature cited in the module docstrings
(Thorp 1967 / JKPS Eq. 1.47, Francois & Garrison 1982, AT AttenMod.f90,
Brown 1991, APL-UW TR 9407, UNESCO 1983, Del Grosso 1974).
"""

import io as _io
import math

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError, FileFormatError


# ─────────────────────────────────────────────────────────────────────────────
# core/absorption.py — bare formulas
# ─────────────────────────────────────────────────────────────────────────────


def _thorp_db_per_km_published(f_hz):
    """Thorp attenuation in dB/km, transcribed from the published polynomial.

    JKPS *Computational Ocean Acoustics* 2nd ed. Eq. (1.47) — four terms in
    ``f`` measured in **kHz**:

        alpha = 3.3e-3 + 0.11·f²/(1 + f²) + 44·f²/(4100 + f²) + 3e-4·f²

    AT's ``misc/AttenMod.f90`` carries the same four terms character for
    character (its comment numbers it Eq. 1.34, the 1st-edition numbering for
    the same expression). Written out here so a coefficient mis-transcribed
    into ``uacpy.core.absorption`` disagrees with something, rather than being
    frozen by the check values below — those were evaluated from these same
    coefficients and so cannot flag the transcription itself.
    """
    f = f_hz / 1000.0
    return (3.3e-3
            + 0.11 * f ** 2 / (1.0 + f ** 2)
            + 44.0 * f ** 2 / (4100.0 + f ** 2)
            + 3.0e-4 * f ** 2)


def _francois_garrison_db_per_km_published(f_hz, T, S, pH, z):
    """Francois–Garrison attenuation in dB/km, transcribed from the published
    formulas: Francois & Garrison (1982), JASA 72(6) 1879–1890, in the form AT
    codes in ``misc/AttenMod.f90``. ``f`` in **kHz**, ``T`` in °C, ``S`` in
    psu, ``z`` in m. Two chemical relaxations of the form
    ``A·P·f_r·f²/(f_r² + f²)`` plus pure-water viscosity, which has no
    relaxation and so enters as a plain ``f²``:

        c  = 1412 + 3.21·T + 1.19·S + 0.0167·z
        A1 = 8.86/c · 10^(0.78·pH − 5)              P1 = 1
        f1 = 2.8·sqrt(S/35) · 10^(4 − 1245/(T + 273))
        A2 = 21.44·S/c · (1 + 0.025·T)              P2 = 1 − 1.37e-4·z + 6.2e-9·z²
        f2 = 8.17·10^(8 − 1990/(T + 273)) / (1 + 0.0018·(S − 35))
                                                    P3 = 1 − 3.83e-5·z + 4.9e-10·z²
        A3 = 4.937e-4 − 2.59e-5·T + 9.11e-7·T² − 1.5e-8·T³      (T < 20)
             3.964e-4 − 1.146e-5·T + 1.45e-7·T² − 6.5e-10·T³    (otherwise)

        alpha = A1·P1·f1·f²/(f1² + f²) + A2·P2·f2·f²/(f2² + f²) + A3·P3·f²

    The published statement gives the two A3 fits for "T < 20" and "T > 20"
    and says nothing about T = 20 exactly; ``AttenMod.f90`` writes
    ``if (T < 20)`` with the warm fit in its ``else``, so T = 20.0 takes the
    **warm** branch there. This transcription writes the branch the same way,
    which is the behaviour uacpy matches.

    Same purpose as :func:`_thorp_db_per_km_published`: the 16-digit check
    values below were produced from these coefficients, so only an independent
    statement of the coefficients can catch a slip in them.
    """
    f = f_hz / 1000.0
    c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * z

    A1 = 8.86 / c * 10.0 ** (0.78 * pH - 5.0)
    P1 = 1.0
    f1 = 2.8 * math.sqrt(S / 35.0) * 10.0 ** (4.0 - 1245.0 / (T + 273.0))

    A2 = 21.44 * S / c * (1.0 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * z + 6.2e-9 * z ** 2
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / (T + 273.0)) / (1.0 + 0.0018 * (S - 35.0))

    P3 = 1.0 - 3.83e-5 * z + 4.9e-10 * z ** 2
    if T < 20.0:
        A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T ** 2 - 1.5e-8 * T ** 3
    else:
        A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T ** 2 - 6.5e-10 * T ** 3

    return (A1 * P1 * f1 * f ** 2 / (f1 ** 2 + f ** 2)
            + A2 * P2 * f2 * f ** 2 / (f2 ** 2 + f ** 2)
            + A3 * P3 * f ** 2)


class TestThorpReferenceValues:
    """Thorp's Eq. (1.47) is a fixed four-term polynomial in f²; these check
    values were evaluated from the published coefficients and agree with the
    textbook curve (~0.07 dB/km at 1 kHz, ~1.2 dB/km at 10 kHz)."""

    @pytest.mark.parametrize("f_hz, a_db_km", [
        (100.0, 0.004499425722313501),
        (1e3, 0.06932909046574005),
        (1e4, 1.1898299387081566),
        (5e4, 17.52992268425963),
    ])
    def test_matches_published_curve(self, f_hz, a_db_km):
        from uacpy.core.absorption import thorp_db_per_km
        assert float(thorp_db_per_km(f_hz)) == pytest.approx(
            a_db_km, rel=1e-12)

    @pytest.mark.parametrize("f_hz", [
        10.0, 100.0, 1e3, 1e4, 5e4, 1e5, 3e5,
    ])
    def test_matches_the_published_polynomial(self, f_hz):
        """The implementation against :func:`_thorp_db_per_km_published`, which
        states the coefficients independently of it. Evaluated on the four
        pinned frequencies plus three more, so a coefficient change that
        happened to leave the pinned values alone still has nowhere to hide.
        The 1e-12 is for float association, not for the algebra: the two
        expressions are the same polynomial and agree to a few ulp."""
        from uacpy.core.absorption import thorp_db_per_km
        assert float(thorp_db_per_km(f_hz)) == pytest.approx(
            _thorp_db_per_km_published(f_hz), rel=1e-12)

    def test_class_converts_db_per_km_to_db_per_m(self):
        """Thorp.alpha_db_per_m is the bare formula divided by 1000, flat in
        depth."""
        from uacpy.core.absorption import Thorp, thorp_db_per_km
        z = np.array([0.0, 500.0, 5000.0])
        a = Thorp().alpha_db_per_m(1e4, z)
        assert a.shape == z.shape
        np.testing.assert_allclose(
            a, float(thorp_db_per_km(1e4)) / 1000.0, rtol=1e-12)


class TestFrancoisGarrisonReferenceValues:
    """Francois & Garrison (1982) check values evaluated from the published
    coefficients (the AT AttenMod.f90 transcription); the set spans the
    boric-acid, MgSO4 and viscosity regimes, a deep cold-water case for the
    pressure corrections, and both sides of the 20 °C viscosity fit."""

    @pytest.mark.parametrize("f_hz, T, S, pH, z, a_db_km", [
        (1e3, 10.0, 35.0, 8.0, 0.0, 0.060126063391378846),
        (1e4, 10.0, 35.0, 8.0, 0.0, 0.962637291817115),
        (1e5, 10.0, 35.0, 8.0, 0.0, 33.63031641787127),
        (1e4, 4.0, 34.0, 7.9, 2000.0, 0.8372956055122321),
        # 20 °C sits on the A3 piecewise break: the warm fit applies there.
        (5e5, 20.0, 35.0, 8.0, 0.0, 146.64062876899342),
        (5e5, 10.0, 35.0, 8.0, 0.0, 124.67276253563452),
        (5e5, 25.0, 35.0, 8.0, 0.0, 169.7876059853789),
    ])
    def test_matches_published_curve(self, f_hz, T, S, pH, z, a_db_km):
        from uacpy.core.absorption import francois_garrison_db_per_km
        got = float(francois_garrison_db_per_km(f_hz, T, S, pH, z))
        assert got == pytest.approx(a_db_km, rel=1e-12)

    @pytest.mark.parametrize("f_hz, T, S, pH, z", [
        # Every point the check values above pin, so the two sets are tied
        # together rather than each standing alone.
        (1e3, 10.0, 35.0, 8.0, 0.0),
        (1e4, 10.0, 35.0, 8.0, 0.0),
        (1e5, 10.0, 35.0, 8.0, 0.0),
        (1e4, 4.0, 34.0, 7.9, 2000.0),
        (5e5, 20.0, 35.0, 8.0, 0.0),
        (5e5, 10.0, 35.0, 8.0, 0.0),
        (5e5, 25.0, 35.0, 8.0, 0.0),
        # The 63-kHz depth pair of test_depth_correction_attenuates_mgso4_term.
        (6.3e4, 10.0, 35.0, 8.0, 0.0),
        (6.3e4, 10.0, 35.0, 8.0, 4000.0),
        # Off the pinned grid: either side of the A3 break to a tenth of a
        # degree, the pH and salinity terms away from their nominal values, and
        # a deep cold case that leans on P2/P3.
        (2e5, 19.9, 35.0, 8.0, 0.0),
        (2e5, 20.1, 35.0, 8.0, 0.0),
        (3e3, 12.0, 35.0, 7.4, 0.0),
        (3e4, 12.0, 8.0, 8.2, 0.0),
        (1e5, 2.0, 34.7, 8.1, 5000.0),
    ])
    def test_matches_the_published_formula(self, f_hz, T, S, pH, z):
        """The implementation against
        :func:`_francois_garrison_db_per_km_published`, which states every
        coefficient independently of it. Same role as the Thorp transcription
        test, and the same reason for 1e-12: the two expressions differ only in
        how the products associate."""
        from uacpy.core.absorption import francois_garrison_db_per_km
        got = float(francois_garrison_db_per_km(f_hz, T, S, pH, z))
        assert got == pytest.approx(
            _francois_garrison_db_per_km_published(f_hz, T, S, pH, z),
            rel=1e-12)

    def test_the_a3_branch_at_exactly_20_degrees_is_the_warm_fit(self):
        """T = 20.0 exactly. The publication states the two A3 fits for
        "T < 20" and "T > 20" and says nothing about 20; ``AttenMod.f90``
        writes ``if (T < 20)`` with the warm fit in its ``else``, so AT takes
        the warm branch there and uacpy matches AT. Both the implementation and
        the transcription are held to that here, so the choice cannot drift on
        one side only.

        The two fits are built to nearly meet at the break — 2.2000e-4 against
        2.2010e-4, 4.5e-4 apart — which moves α at 500 kHz by only 1.7e-4
        relative. That is 8 orders above the 1e-12 the tests compare at and
        wholly invisible to a percent-level check, so the branch is worth
        pinning and only worth pinning tightly.
        """
        from uacpy.core.absorption import francois_garrison_db_per_km
        f_khz, T, z = 500.0, 20.0, 0.0
        A3_warm = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T ** 2 - 6.5e-10 * T ** 3
        A3_cold = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T ** 2 - 1.5e-8 * T ** 3
        P3 = 1.0 - 3.83e-5 * z + 4.9e-10 * z ** 2

        warm = _francois_garrison_db_per_km_published(f_khz * 1e3, T, 35.0, 8.0, z)
        # A3 is the only thing the branch changes, so swapping it is the whole
        # of the difference between the two readings at this T.
        cold = warm + (A3_cold - A3_warm) * P3 * f_khz ** 2
        assert abs(cold - warm) / warm > 1e-5

        assert float(francois_garrison_db_per_km(f_khz * 1e3, T, 35.0, 8.0, z)) \
            == pytest.approx(warm, rel=1e-12)

    def test_depth_correction_attenuates_mgso4_term(self):
        """P2 = 1 − 1.37e-4·z + 6.2e-9·z² cuts the 63-kHz (MgSO4-dominated)
        absorption to ~0.549 of its surface value at 4000 m."""
        from uacpy.core.absorption import francois_garrison_db_per_km
        a_surf = float(francois_garrison_db_per_km(6.3e4, 10., 35., 8., 0.))
        a_deep = float(francois_garrison_db_per_km(6.3e4, 10., 35., 8., 4000.))
        assert a_deep / a_surf == pytest.approx(0.54917617566523, rel=1e-10)

    def test_class_overrides_nominal_depth_with_the_depth_axis(self):
        """FrancoisGarrison.alpha_db_per_m re-evaluates the formula per depth
        (dB/m = dB/km / 1000), ignoring z_bar_m."""
        from uacpy.core.absorption import (
            FrancoisGarrison, francois_garrison_db_per_km)
        fg = FrancoisGarrison(temperature_c=10.0, salinity_psu=35.0,
                              pH=8.0, z_bar_m=100.0)
        z = np.array([0.0, 2000.0])
        got = fg.alpha_db_per_m(1e4, z)
        want = francois_garrison_db_per_km(
            1e4, temperature=10.0, salinity=35.0, pH=8.0, depth=z) / 1000.0
        np.testing.assert_allclose(got, want, rtol=1e-12)
        assert float(got[0]) == pytest.approx(0.000962637291817115, rel=1e-12)


class TestConvertAttenuationUnitsClosedForms:
    """Every unit is defined against the nepers/m attenuation ``a`` of
    ``exp(-a·x)`` at ``omega = 2πf``: dB/m = a·20/ln10, dB/wavelength =
    dB/m·(c/f), Q gives a = omega/(2cQ), L gives a = L·omega/c. Each path is
    checked against that definition written out independently, plus the
    round-trip back."""

    F = 100.0
    C = 1480.0
    NEPER_DB = 20.0 / np.log(10.0)

    def _conv(self, alpha, frm, to):
        from uacpy.core.absorption import convert_attenuation_units
        return float(convert_attenuation_units(
            alpha, self.F, frm, to, sound_speed=self.C))

    def test_db_km_is_db_m_times_1000(self):
        assert self._conv(3.0, 'dB/km', 'dB/m') == pytest.approx(
            3.0e-3, rel=1e-12)
        assert self._conv(3.0e-3, 'dB/m', 'dB/km') == pytest.approx(
            3.0, rel=1e-12)

    def test_db_per_wavelength_uses_lambda_c_over_f(self):
        lam = self.C / self.F
        assert self._conv(0.5, 'dB/wavelength', 'dB/m') == pytest.approx(
            0.5 / lam, rel=1e-12)
        assert self._conv(0.5 / lam, 'dB/m', 'dB/wavelength'
                          ) == pytest.approx(0.5, rel=1e-12)

    def test_nepers_use_20_over_ln10(self):
        assert self._conv(1.0, 'Nepers/m', 'dB/m') == pytest.approx(
            self.NEPER_DB, rel=1e-12)
        assert self._conv(self.NEPER_DB, 'dB/m', 'Nepers/m'
                          ) == pytest.approx(1.0, rel=1e-12)

    def test_quality_factor_definition(self):
        # a = omega/(2cQ) nepers/m, with omega = 2·pi·f, so a = pi·f/(c·Q).
        Q = 50.0
        a_nepers = np.pi * self.F / (self.C * Q)
        assert self._conv(Q, 'Q', 'dB/m') == pytest.approx(
            a_nepers * self.NEPER_DB, rel=1e-12)
        assert self._conv(a_nepers * self.NEPER_DB, 'dB/m', 'Q'
                          ) == pytest.approx(Q, rel=1e-12)

    def test_loss_tangent_definition(self):
        # a = L·omega/c nepers/m.
        L = 2e-3
        a_nepers = L * 2.0 * np.pi * self.F / self.C
        assert self._conv(L, 'L', 'dB/m') == pytest.approx(
            a_nepers * self.NEPER_DB, rel=1e-12)
        assert self._conv(a_nepers * self.NEPER_DB, 'dB/m', 'L'
                          ) == pytest.approx(L, rel=1e-12)

    def test_unknown_units_raise_in_both_positions(self):
        from uacpy.core.absorption import convert_attenuation_units
        with pytest.raises(ConfigurationError, match="Unknown unit"):
            convert_attenuation_units(1.0, self.F, 'furlongs', 'dB/m')
        with pytest.raises(ConfigurationError, match="Unknown unit"):
            convert_attenuation_units(1.0, self.F, 'dB/m', 'furlongs')


class TestBiologicalLorentzian:
    """The layer resonance is ``a0 / ((1 − f0²/f²)² + 1/Q²)`` dB/km
    (AttenMod.f90): the peak at f = f0 is exactly a0·Q², and one octave
    below, (1 − f0²/f²) = −3 so the denominator is 9 + 1/Q²."""

    def _bio(self):
        from uacpy.core.absorption import Biological
        return Biological(layers=[(10.0, 60.0, 1000.0, 5.0, 10.0)])

    def test_peak_is_a0_q_squared(self):
        a = self._bio().alpha_db_per_m(1000.0, np.array([30.0]))
        assert float(a[0]) == pytest.approx(10.0 * 25.0 / 1000.0, rel=1e-12)

    def test_off_resonance_denominator(self):
        a = self._bio().alpha_db_per_m(500.0, np.array([30.0]))
        want = 10.0 / (9.0 + 1.0 / 25.0) / 1000.0
        assert float(a[0]) == pytest.approx(want, rel=1e-12)

    def test_zero_frequency_raises_typed_error(self):
        """f = 0 must raise ConfigurationError, not divide by zero."""
        with pytest.raises(ConfigurationError, match="frequency must be > 0"):
            self._bio().alpha_db_per_m(0.0, np.array([30.0]))

    def test_zero_resonance_frequency_rejected(self):
        from uacpy.core.absorption import BiologicalLayer
        with pytest.raises(ConfigurationError, match="f0_hz must be positive"):
            BiologicalLayer(z_top_m=0.0, z_bottom_m=10.0, f0_hz=0.0,
                            Q=5.0, a0=1.0)


class TestConstantAbsorptionContract:
    """A zero baseline is the valid lossless limit; f = 0 has no wavelength
    to convert through and must raise the typed error."""

    def test_zero_value_is_a_valid_lossless_baseline(self):
        from uacpy.core.absorption import ConstantAbsorption
        c = ConstantAbsorption(value_db_per_wavelength=0.0)
        np.testing.assert_allclose(
            c.alpha_db_per_m(100.0, np.array([0.0, 50.0])), 0.0)

    def test_zero_frequency_raises_typed_error(self):
        from uacpy.core.absorption import ConstantAbsorption
        c = ConstantAbsorption(value_db_per_wavelength=0.5)
        with pytest.raises(ConfigurationError, match="frequency must be > 0"):
            c.alpha_db_per_m(0.0, np.array([10.0]))


# ─────────────────────────────────────────────────────────────────────────────
# io/_fortran_helpers.py — record framing and list-directed parsing boundaries
# ─────────────────────────────────────────────────────────────────────────────


class TestBoundCountsBoundary:
    """``_bound_counts`` rejects header counts whose *product* exceeds
    ``file_size // min_item_bytes`` — a product exactly equal to it is the
    densest legal file and must pass."""

    def test_product_exactly_at_capacity_is_accepted(self):
        from uacpy.io._fortran_helpers import _bound_counts
        # 64 bytes at 8 bytes/item -> at most 8 items; 2 x 4 = 8 exactly.
        _bound_counts('f.bin', 64, 8, n_rcv=2, n_freq=4)

    def test_product_one_past_capacity_is_rejected(self):
        from uacpy.io._fortran_helpers import _bound_counts
        with pytest.raises(FileFormatError, match="implausible"):
            _bound_counts('f.bin', 64, 8, n_rcv=3, n_freq=3)


class TestTakeTokensExhaustsTheStream:
    """A read of exactly the remaining tokens is satisfiable; only asking
    for one more is a truncation."""

    def test_consuming_the_whole_stream_succeeds(self):
        from uacpy.io._fortran_helpers import take_tokens
        vals, cursor = take_tokens(['1.0', '2.0'], 0, 2, 'amps', 'x.rts')
        assert vals == ['1.0', '2.0'] and cursor == 2

    def test_one_past_the_stream_raises(self):
        from uacpy.io._fortran_helpers import take_tokens
        with pytest.raises(FileFormatError, match="token stream ended"):
            take_tokens(['1.0', '2.0'], 0, 3, 'amps', 'x.rts')


class TestStripFortranQuotes:
    """AT writes titles as ``'…'`` character literals, often with trailing
    annotation; the quoted content — including an empty title — comes back
    bare."""

    def test_quoted_title_with_trailing_comment(self):
        from uacpy.io._fortran_helpers import strip_fortran_quotes
        assert strip_fortran_quotes("'PEKERIS' ! title\n") == 'PEKERIS'

    def test_empty_quoted_string_is_empty(self):
        from uacpy.io._fortran_helpers import strip_fortran_quotes
        assert strip_fortran_quotes("''\n") == ''


class TestFortranRecordFraming:
    """Record-marker edge cases of ``read_fortran_record``: the zero-length
    record a Fortran WRITE with an empty I/O list produces, and both sides
    of the 2**28-byte sanity cap."""

    def test_zero_length_record_is_valid(self):
        import struct
        from uacpy.io._fortran_helpers import read_fortran_record
        f = _io.BytesIO(struct.pack('<i', 0) + struct.pack('<i', 0))
        assert read_fortran_record(f, raw=True) == b''

    def test_length_at_the_cap_reads_rather_than_rejects(self):
        import struct
        from uacpy.io._fortran_helpers import read_fortran_record
        # A marker of exactly 2**28 is within the cap: the reader proceeds
        # and then reports the truncated payload, not an unreasonable length.
        f = _io.BytesIO(struct.pack('<i', 1 << 28) + b'xyz')
        with pytest.raises(FileFormatError, match="Short read"):
            read_fortran_record(f, raw=True)

    def test_length_past_the_cap_is_rejected_before_reading(self):
        import struct
        from uacpy.io._fortran_helpers import read_fortran_record
        f = _io.BytesIO(struct.pack('<i', (1 << 28) + 1) + b'xyz')
        with pytest.raises(FileFormatError, match="Unreasonable"):
            read_fortran_record(f, raw=True)


class TestDetectEndianResolution:
    """``detect_endian`` picks the byte order whose record marker is a
    plausible length, preferring little-endian on a tie (the CI-validated
    order), rejecting markers implausible both ways at the 2**28 cap, and
    warning only for big-endian files."""

    def test_marker_implausible_both_ways_raises(self):
        from uacpy.io._fortran_helpers import detect_endian
        # little: 0x10000080 >= 2**28; big: 0x80000010 < 0 — no valid order.
        with pytest.raises(FileFormatError, match="cannot resolve"):
            detect_endian(b'\x80\x00\x00\x10')

    def test_palindromic_marker_prefers_little_endian(self):
        from uacpy.io._fortran_helpers import detect_endian
        # Reads as 65792 in both orders; the documented tie-break is '<'.
        assert detect_endian(b'\x00\x01\x01\x00') == '<'

    def test_warns_for_big_endian_only(self):
        import struct
        import warnings as _w
        import uacpy.io._fortran_helpers as fh
        emitted = fh._ENDIAN_WARN_EMITTED
        try:
            fh._ENDIAN_WARN_EMITTED = False
            with _w.catch_warnings():
                _w.simplefilter('error', UserWarning)
                assert fh.detect_endian(struct.pack('<i', 128)) == '<'
            fh._ENDIAN_WARN_EMITTED = False
            with pytest.warns(UserWarning, match="big-endian"):
                assert fh.detect_endian(struct.pack('>i', 128)) == '>'
        finally:
            fh._ENDIAN_WARN_EMITTED = emitted


class TestReadVectorGeneratesAtMinimumLength:
    """SubTab's two-value (equally spaced) branch generates for every
    Nx >= 3 — including Nx = 3 itself, the shortest generatable vector."""

    def test_two_value_shorthand_at_nx_3(self):
        from uacpy.io._fortran_helpers import read_vector
        x, nx = read_vector(_io.StringIO("3\n0 100 /\n"))
        assert nx == 3
        np.testing.assert_allclose(x, [0.0, 50.0, 100.0])


# ─────────────────────────────────────────────────────────────────────────────
# Wave-5 mutation-campaign killing tests (2026-08-18 rerun): each class
# below pins a contract a surviving mutant showed to be untested.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# core/_beamforming.py — MVDR diagonal-loading magnitude
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadedInverseLoadingMagnitude:
    """``loaded_inverse`` adds ``loading`` *fractions of the average
    eigenvalue* ``tr(R)/N`` to the diagonal: for R = diag(2, 4) and
    loading = 0.5 the average eigenvalue is 3, so the loaded matrix is
    diag(3.5, 5.5) exactly."""

    def test_loading_is_a_fraction_of_the_average_eigenvalue(self):
        from uacpy.core._beamforming import loaded_inverse
        R = np.diag([2.0, 4.0]).astype(complex)
        got = loaded_inverse(R, loading=0.5)
        np.testing.assert_allclose(
            got, np.diag([1.0 / 3.5, 1.0 / 5.5]), rtol=1e-12)

    def test_zero_loading_is_the_plain_inverse(self):
        from uacpy.core._beamforming import loaded_inverse
        R = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=complex)
        np.testing.assert_allclose(
            loaded_inverse(R, loading=0.0) @ R, np.eye(2), atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# acoustic_signal/constant_q.py — frequency ladder, kernel, frames, hop, ppsd
# ─────────────────────────────────────────────────────────────────────────────


class TestConstantQFrequencyLadder:
    """``f_k = fmin·2**(k/B)`` with ``K = floor(B·log2(fmax/fmin)) + 1``
    bins; fmin below 1 Hz is legal, fmin = 0 and fmax <= fmin are not."""

    def test_exact_octave_ladder(self):
        from uacpy.acoustic_signal.constant_q import constant_q_transform
        x = np.sin(2 * np.pi * 200.0 * np.arange(4096) / 4096.0)
        r = constant_q_transform(x, 4096.0, fmin=100.0, fmax=400.0,
                                 bins_per_octave=1)
        np.testing.assert_allclose(r.frequencies, [100.0, 200.0, 400.0],
                                   rtol=1e-12)

    def test_sub_hertz_fmin_is_legal(self):
        from uacpy.acoustic_signal.constant_q import _cq_frequencies
        f = _cq_frequencies(0.5, 2.0, 1)
        np.testing.assert_allclose(f, [0.5, 1.0, 2.0], rtol=1e-12)

    def test_fmin_zero_raises(self):
        from uacpy.acoustic_signal.constant_q import _cq_frequencies
        with pytest.raises(ConfigurationError, match="0 < fmin < fmax"):
            _cq_frequencies(0.0, 100.0, 24)

    def test_fmax_equal_to_fmin_raises(self):
        from uacpy.acoustic_signal.constant_q import _cq_frequencies
        with pytest.raises(ConfigurationError, match="0 < fmin < fmax"):
            _cq_frequencies(100.0, 100.0, 24)


class TestConstantQKernelConstruction:
    """The kernel is ``w·exp(-2j·pi·f_k·n/fs)/Σw`` with a *periodic*
    (fftbins) window of ``N_k = max(1, ceil(Q·fs/f_k))`` samples — pinned
    against an independent reconstruction, phase included."""

    def test_kernel_matches_definition_exactly(self):
        from scipy.signal import get_window
        from uacpy.acoustic_signal.constant_q import _cq_kernels
        fs, fk, Q = 1000.0, 125.0, 16.817
        (Nk, ker, _), = _cq_kernels(np.array([fk]), Q, fs, "hann")
        assert Nk == int(np.ceil(Q * fs / fk))
        w = get_window("hann", Nk, fftbins=True)
        n = np.arange(Nk)
        want = (w * np.exp(-2j * np.pi * fk * n / fs)) / float(np.sum(w))
        np.testing.assert_allclose(ker, want, rtol=0, atol=1e-15)

    def test_window_floor_is_one_sample(self):
        from uacpy.acoustic_signal.constant_q import _cq_kernels
        # Q·fs/f_k = 0.5 -> ceil = 1: the floor keeps N_k = 1, not 2.
        (Nk, _, _), = _cq_kernels(np.array([2000.0]), 1.0, 1000.0, "hann")
        assert Nk == 1


class TestConstantQFrameGeometry:
    """Window centring, the exact-fit validity boundary, and the
    zero-padded edge path, pinned with unit impulses."""

    def test_window_is_centred_on_the_requested_sample(self):
        from uacpy.acoustic_signal.constant_q import _cq_frame, _cq_kernels
        kernels = _cq_kernels(np.array([100.0]), 8.0, 1000.0, "hann")
        Nk, ker, _ = kernels[0]
        x = np.zeros(4 * Nk)
        centre = 2 * Nk
        x[centre] = 1.0
        coeffs, valid = _cq_frame(x, centre, kernels)
        # The impulse sits at window index Nk//2, so the coefficient is
        # that single kernel sample.
        assert valid[0]
        np.testing.assert_allclose(coeffs[0], ker[Nk // 2], rtol=0,
                                   atol=1e-15)

    def test_exact_fit_window_is_valid_one_short_is_not(self):
        from uacpy.acoustic_signal.constant_q import _cq_frame, _cq_kernels
        kernels = _cq_kernels(np.array([100.0]), 8.0, 1000.0, "hann")
        Nk = kernels[0][0]
        x = np.ones(Nk)
        _, valid_fit = _cq_frame(x, Nk // 2, kernels)
        assert valid_fit[0]
        _, valid_short = _cq_frame(x[:-1], Nk // 2, kernels)
        assert not valid_short[0]

    def test_edge_padding_keeps_sample_zero(self):
        from uacpy.acoustic_signal.constant_q import _cq_frame, _cq_kernels
        kernels = _cq_kernels(np.array([100.0]), 8.0, 1000.0, "hann")
        Nk, ker, _ = kernels[0]
        x = np.zeros(Nk)
        x[0] = 1.0
        # Centre at 0: the window start is negative, so x[0] lands at
        # kernel index Nk//2 via the zero-padded path.
        coeffs, valid = _cq_frame(x, 0, kernels)
        assert not valid[0]
        np.testing.assert_allclose(coeffs[0], ker[Nk // 2], rtol=0,
                                   atol=1e-15)


class TestConstantQSpectrogramTimeAxis:
    """``times = arange(0, n, hop)/fs`` — starts at zero, in seconds."""

    def test_times_start_at_zero_in_seconds(self):
        from uacpy.acoustic_signal.constant_q import constant_q_spectrogram
        fs = 2000.0
        x = np.sin(2 * np.pi * 250.0 * np.arange(2048) / fs)
        r = constant_q_spectrogram(x, fs, fmin=125.0, fmax=500.0,
                                   bins_per_octave=2, hop=100)
        np.testing.assert_allclose(
            r.times, np.arange(0, 2048, 100) / fs, rtol=1e-12)


class TestConstantQHopResolution:
    """Default hop is ``max(1, min(n_lowest//8, max(1, n//8)))`` from the
    *lowest* bin's window; an explicit ``hop=1`` is legal."""

    def test_default_hop_follows_the_lowest_bin(self):
        from uacpy.acoustic_signal.constant_q import _resolve_hop
        kernels = [(100, None, None), (50, None, None)]
        assert _resolve_hop(None, kernels, 10000, "t") == 100 // 8

    def test_tiny_signal_floors_at_one(self):
        from uacpy.acoustic_signal.constant_q import _resolve_hop
        assert _resolve_hop(None, [(8, None, None)], 8, "t") == 1

    def test_explicit_hop_of_one_is_accepted(self):
        from uacpy.acoustic_signal.constant_q import _resolve_hop
        assert _resolve_hop(1, [(100, None, None)], 1000, "t") == 1
        with pytest.raises(ConfigurationError, match="hop must be >= 1"):
            _resolve_hop(0, [(100, None, None)], 1000, "t")


class TestConstantQSetupWarningBoundary:
    """The too-short-signal warning keys on the lowest bin needing *more*
    samples than the signal has — an exact fit stays silent."""

    def test_exact_fit_does_not_warn(self):
        import warnings as _w
        from uacpy.acoustic_signal.constant_q import (
            _cq_frequencies, _cq_kernels, _cq_quality, _cq_setup)
        fs, fmin, B = 1000.0, 100.0, 2
        n_lowest = _cq_kernels(
            _cq_frequencies(fmin, fs / 2, B), _cq_quality(B), fs,
            "hann")[0][0]
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            _cq_setup(np.zeros(n_lowest), fs, fmin, None, B, "hann", "t")
        # Only this warning is under test. `fmax=None` resolves to fs/2, so
        # the near-Nyquist image note fires on the same call by design.
        assert not any("lowest bin needs" in str(c.message) for c in caught)
        with pytest.warns(UserWarning, match="lowest bin needs"):
            _cq_setup(np.zeros(n_lowest - 1), fs, fmin, None, B, "hann", "t")


class TestConstantQTransformCentresTheFrame:
    """``constant_q_transform`` analyses one frame centred on ``n//2``."""

    def test_impulse_at_the_centre_sample(self):
        from uacpy.acoustic_signal.constant_q import (
            _cq_frequencies, _cq_kernels, _cq_quality, constant_q_transform)
        fs, fmin, fmax, B = 1000.0, 100.0, 200.0, 1
        n = 1024
        x = np.zeros(n)
        x[n // 2] = 1.0
        r = constant_q_transform(x, fs, fmin=fmin, fmax=fmax,
                                 bins_per_octave=B)
        kernels = _cq_kernels(_cq_frequencies(fmin, fmax, B),
                              _cq_quality(B), fs, "hann")
        for got, (Nk, ker, _) in zip(r.coefficients, kernels):
            np.testing.assert_allclose(got, ker[Nk // 2], rtol=0, atol=1e-15)


class TestProbabilisticConstantQContracts:
    """Level-edge defaults run ``lvlmin..lvlmax`` inclusive, and a bin
    with exactly one fully-inside frame is data — for the PSD average and
    the PPSD histogram both — not a NaN column."""

    def _one_frame_case(self):
        from uacpy.acoustic_signal.constant_q import (
            _cq_frequencies, _cq_kernels, _cq_quality)
        fs, fmin, B = 1000.0, 100.0, 2
        n_lowest = _cq_kernels(
            _cq_frequencies(fmin, fs / 2, B), _cq_quality(B), fs,
            "hann")[0][0]
        # 2*n_lowest+1 samples at hop = n_lowest: frame centres {0, n, 2n},
        # of which only the middle one fits the lowest bin's window
        # entirely -> exactly one valid frame for bin 0.
        x = np.sin(2 * np.pi * fmin * np.arange(2 * n_lowest + 1) / fs)
        return x, fs, fmin, B, n_lowest

    def test_level_edges_run_from_lvlmin_to_lvlmax_inclusive(self):
        from uacpy.acoustic_signal.constant_q import probabilistic_constant_q
        x, fs, fmin, B, n_lowest = self._one_frame_case()
        r = probabilistic_constant_q(x, fs, fmin=fmin, bins_per_octave=B,
                                     hop=n_lowest, ddB=1.0, lvlmin=0,
                                     lvlmax=150)
        assert r.level_edges[0] == 0.0
        assert r.level_edges[-1] == 150.0
        assert r.level_edges.size == 151

    def test_single_valid_frame_is_data(self):
        from uacpy.acoustic_signal.constant_q import (
            constant_q_psd, probabilistic_constant_q)
        x, fs, fmin, B, n_lowest = self._one_frame_case()
        psd = constant_q_psd(x, fs, fmin=fmin, bins_per_octave=B,
                             hop=n_lowest)
        assert np.isfinite(psd.power[0])
        ppsd = probabilistic_constant_q(x, fs, fmin=fmin, bins_per_octave=B,
                                        hop=n_lowest)
        assert np.isfinite(ppsd.pdf[:, 0]).any()


# ─────────────────────────────────────────────────────────────────────────────
# core/absorption.py — untested guard boundaries (wave-5 survivors)
# ─────────────────────────────────────────────────────────────────────────────


class TestConvertFromQualityFactorBoundaries:
    """Q in (0, 1] is a legal (heavily damped) quality factor; only
    Q <= 0 has no attenuation to convert."""

    def test_sub_unity_q_round_trips(self):
        from uacpy.core.absorption import convert_attenuation_units
        db_m = convert_attenuation_units(0.5, 100.0, 'Q', 'dB/m',
                                         sound_speed=1480.0)
        back = convert_attenuation_units(db_m, 100.0, 'dB/m', 'Q',
                                         sound_speed=1480.0)
        assert float(back) == pytest.approx(0.5, rel=1e-12)

    def test_zero_q_raises(self):
        from uacpy.core.absorption import convert_attenuation_units
        with pytest.raises(ConfigurationError, match="positive quality"):
            convert_attenuation_units(0.0, 100.0, 'Q', 'dB/m',
                                      sound_speed=1480.0)


class TestBiologicalLayerValidatorBoundaries:
    """Zero-thickness layers and exactly-zero Q/a0 are rejected;
    parameters in (0, 1] are legal."""

    def test_zero_thickness_layer_raises(self):
        from uacpy.core.absorption import BiologicalLayer
        with pytest.raises(ConfigurationError):
            BiologicalLayer(z_top_m=10.0, z_bottom_m=10.0, f0_hz=100.0,
                            Q=5.0, a0=1.0)

    def test_zero_q_raises(self):
        from uacpy.core.absorption import BiologicalLayer
        with pytest.raises(ConfigurationError):
            BiologicalLayer(z_top_m=0.0, z_bottom_m=10.0, f0_hz=100.0,
                            Q=0.0, a0=1.0)

    def test_zero_a0_raises(self):
        from uacpy.core.absorption import BiologicalLayer
        with pytest.raises(ConfigurationError):
            BiologicalLayer(z_top_m=0.0, z_bottom_m=10.0, f0_hz=100.0,
                            Q=5.0, a0=0.0)

    def test_sub_unity_parameters_are_legal(self):
        from uacpy.core.absorption import Biological
        b = Biological(layers=[(0.0, 10.0, 0.5, 0.5, 0.5)])
        a = b.alpha_db_per_m(0.5, np.array([5.0]))
        assert np.isfinite(a).all()


class TestSubHertzFrequenciesAreLegal:
    """Only f <= 0 has no wavelength; infrasonic f in (0, 1] converts."""

    def test_constant_absorption_at_half_a_hertz(self):
        from uacpy.core.absorption import ConstantAbsorption
        c = ConstantAbsorption(value_db_per_wavelength=0.5)
        assert np.isfinite(c.alpha_db_per_m(0.5, np.array([10.0]))).all()


# ─────────────────────────────────────────────────────────────────────────────
# acoustic_signal/transforms.py — radon/taup/fk contract holes (wave-5)
# ─────────────────────────────────────────────────────────────────────────────


class TestParabolicMoveoutCurve:
    """``_moveout_times('parabolic')`` is ``tau + q·x**2`` exactly."""

    def test_matches_the_closed_form(self):
        from uacpy.acoustic_signal.transforms import _moveout_times
        taus = np.array([0.1, 0.2])
        x = np.array([-2.0, 0.0, 3.0])
        got = _moveout_times("parabolic", taus[:, None], x[None, :], 0.05)
        np.testing.assert_allclose(
            got, taus[:, None] + 0.05 * x[None, :] ** 2, rtol=1e-15)


class TestRadonOffsetAxisOrigin:
    """The offset axis is ``arange(nx)·dx − x0``: with ``x0 = 2·dx`` the
    zero offset sits at trace index 2, where every moveout curve passes
    through ``t = tau`` — so a lone delta trace there contributes equally
    to every moveout."""

    def test_zero_offset_trace_is_moveout_invariant(self):
        from uacpy.acoustic_signal.transforms import radon_transform
        fs, nx, dx = 100.0, 5, 10.0
        data = np.zeros((64, nx))
        data[20, 2] = 1.0
        r = radon_transform(data, fs, dx, np.array([-1e-3, 0.0, 1e-3]),
                            kind="linear", x0=2 * dx)
        col = r.panel[:, 20]
        assert abs(col[0]) > 0
        np.testing.assert_allclose(col, col[0], rtol=1e-12)


class TestHyperbolicMoveoutGuardBoundaries:
    """Hyperbolic velocities in (0, 1] are legal; exactly zero is not."""

    def test_sub_unity_velocity_is_legal(self):
        from uacpy.acoustic_signal.transforms import radon_transform
        data = np.random.default_rng(0).normal(size=(32, 4))
        radon_transform(data, 100.0, 10.0, np.array([0.5]),
                        kind="hyperbolic")

    def test_zero_velocity_raises(self):
        from uacpy.acoustic_signal.transforms import radon_transform
        data = np.zeros((32, 4))
        with pytest.raises(ConfigurationError):
            radon_transform(data, 100.0, 10.0, np.array([0.0]),
                            kind="hyperbolic")


class TestTaupDefaultSlownessCeiling:
    """The default slowness axis tops out at 1/1000 s/m (the documented
    1000 m/s water-speed floor)."""

    def test_default_p_max(self):
        from uacpy.acoustic_signal.transforms import taup_transform
        data = np.random.default_rng(1).normal(size=(64, 8))
        r = taup_transform(data, 100.0, 25.0)
        assert float(np.max(np.abs(r.slownesses))) == pytest.approx(
            1.0 / 1000.0, rel=1e-12)


class TestFkTransformSegmentGeometry:
    """User nfft must cover the data in *each* dimension, an exact-fit
    nfft is legal, and one-sample segments are legal."""

    def _data(self, nt=16, nx=4):
        return np.random.default_rng(2).normal(size=(nt, nx))

    def test_nfft_smaller_than_data_raises_per_axis(self):
        from uacpy.acoustic_signal.transforms import fk_transform
        with pytest.raises(ConfigurationError):
            fk_transform(self._data(), 100.0, 10.0, nfft=(8, 4))
        with pytest.raises(ConfigurationError):
            fk_transform(self._data(), 100.0, 10.0, nfft=(16, 2))

    def test_exact_fit_nfft_is_legal(self):
        from uacpy.acoustic_signal.transforms import fk_transform
        fk_transform(self._data(), 100.0, 10.0, nfft=(16, 4))

    def test_single_sample_segments_are_legal(self):
        from uacpy.acoustic_signal.transforms import fk_transform
        fk_transform(self._data(), 100.0, 10.0, nperseg=1)

    def test_half_overlap_segmenting_covers_the_record(self):
        # nt=8, nperseg=4, noverlap=2: starts {0, 2, 4}; a start past
        # nt-nperseg would slice a short segment and fail loudly.
        from uacpy.acoustic_signal.transforms import fk_transform
        fk_transform(self._data(nt=8), 100.0, 10.0, nperseg=4, noverlap=2)


# ─────────────────────────────────────────────────────────────────────────────
# acoustic_signal/timefreq.py — lifter, smoothing windows, cwt, cepstra
# ─────────────────────────────────────────────────────────────────────────────


class TestCepstralLifterMask:
    """Scalar lifter L builds the mask 1 on quefrencies |q| <= |L| (head
    and mirrored tail), inverted for negative L; an array lifter
    multiplies element-wise after an exact shape check."""

    def test_low_pass_mask_exact(self):
        from uacpy.acoustic_signal.timefreq import _apply_lifter
        c = np.arange(1.0, 9.0)
        want = c * np.array([1, 1, 1, 0, 0, 0, 1, 1], dtype=float)
        np.testing.assert_allclose(_apply_lifter(c, 2), want, rtol=0)

    def test_zero_lifter_keeps_only_dc(self):
        from uacpy.acoustic_signal.timefreq import _apply_lifter
        c = np.arange(1.0, 9.0)
        want = np.zeros(8)
        want[0] = 1.0
        np.testing.assert_allclose(_apply_lifter(c, 0), want, rtol=0)

    def test_negative_lifter_is_the_complement(self):
        from uacpy.acoustic_signal.timefreq import _apply_lifter
        c = np.arange(1.0, 9.0)
        np.testing.assert_allclose(
            _apply_lifter(c, 2) + _apply_lifter(c, -2), c, rtol=0)

    def test_array_lifter_multiplies_and_checks_shape(self):
        from uacpy.acoustic_signal.timefreq import _apply_lifter
        c = np.arange(1.0, 9.0)
        np.testing.assert_allclose(_apply_lifter(c, 2.0 * np.ones(8)),
                                   2.0 * c, rtol=0)
        with pytest.raises(ConfigurationError, match="must match"):
            _apply_lifter(c, np.ones(7))


class TestWignerSmoothingWindowConstruction:
    """Both paths return an odd, symmetric, centre-peaked window: scalar
    even L generates hann(L-1); an even user array loses its centre
    sample. No smoothing is ``(None, 0)``."""

    def test_none_is_no_smoothing(self):
        from uacpy.acoustic_signal.timefreq import _smoothing_window
        assert _smoothing_window(None, "time") == (None, 0)

    def test_even_scalar_generates_length_l_minus_1(self):
        from uacpy.acoustic_signal.timefreq import _smoothing_window
        w, half = _smoothing_window(6, "time")
        assert w.size == 5 and half == 2
        np.testing.assert_allclose(w, w[::-1], rtol=0)
        assert w.argmax() == half

    def test_length_one_scalar_is_legal_zero_is_not(self):
        from uacpy.acoustic_signal.timefreq import _smoothing_window
        w, half = _smoothing_window(1, "time")
        assert w.size == 1 and half == 0
        with pytest.raises(ConfigurationError, match=">= 1"):
            _smoothing_window(0, "time")

    def test_even_array_loses_its_centre_sample(self):
        import scipy.signal as _sig
        from uacpy.acoustic_signal.timefreq import _smoothing_window
        w6 = _sig.get_window("hann", 6, fftbins=False)
        w, half = _smoothing_window(w6, "time")
        assert w.size == 5 and half == 2
        np.testing.assert_allclose(w, w[::-1], rtol=0, atol=1e-15)
        assert w.argmax() == half

    def test_two_dimensional_array_raises(self):
        from uacpy.acoustic_signal.timefreq import _smoothing_window
        with pytest.raises(ConfigurationError, match="1-D"):
            _smoothing_window(np.ones((3, 3)), "time")


class TestWignerVilleTimeAxis:
    """``times = arange(n)/fs`` — starts at zero, in seconds."""

    def test_times_axis(self):
        from uacpy.acoustic_signal.timefreq import wigner_ville
        fs = 500.0
        x = np.sin(2 * np.pi * 50.0 * np.arange(64) / fs)
        r = wigner_ville(x, fs)
        np.testing.assert_allclose(r.times, np.arange(64) / fs, rtol=1e-12)


class TestReconstructionConstantsPinned:
    """``(C_delta, psi0(0))`` pinned to this implementation's converged
    quadrature (deterministic grid), with ``psi0(0)`` agreeing with
    Torrence & Compo (1998) Table 2 to the tabulated digits."""

    @pytest.mark.parametrize("wavelet, w0, order, c_delta, psi0, tc_psi0", [
        ("morlet", 6.0, 2, 0.7784324428938577, 0.7511255437203238, 0.751),
        ("paul", 6.0, 4, 1.1330895139729555, 1.0789368501515262, 1.079),
        ("dog", 6.0, 2, 3.6162903894907927, 0.8673250705840745, 0.867),
    ])
    def test_constants(self, wavelet, w0, order, c_delta, psi0, tc_psi0):
        from uacpy.acoustic_signal.timefreq import _reconstruction_constants
        cd, p0 = _reconstruction_constants(wavelet, w0, order)
        assert cd == pytest.approx(c_delta, rel=1e-6)
        assert p0 == pytest.approx(psi0, rel=1e-6)
        assert p0 == pytest.approx(tc_psi0, abs=5e-4)


class TestCwtFrequencyContract:
    """Sub-hertz sampling rates are legal; the default grid tops out at
    Nyquist; a record too short for the default grid raises; explicit
    zero frequencies raise while sub-unity ones are legal."""

    def test_sub_hertz_sample_rate_is_legal(self):
        from uacpy.acoustic_signal.timefreq import cwt
        r = cwt(np.random.default_rng(3).normal(size=64), 0.5)
        assert np.isfinite(np.asarray(r.coefficients)).all()

    def test_default_grid_tops_out_at_nyquist(self):
        from uacpy.acoustic_signal.timefreq import cwt
        fs = 100.0
        r = cwt(np.random.default_rng(4).normal(size=128), fs)
        assert float(np.max(r.frequencies)) == pytest.approx(fs / 2.0,
                                                             rel=1e-9)

    def test_record_of_eight_samples_raises(self):
        # f_lo = 4*fs/n hits Nyquist exactly at n = 8 — the >= boundary.
        from uacpy.acoustic_signal.timefreq import cwt
        with pytest.raises(ConfigurationError):
            cwt(np.zeros(8), 100.0)

    def test_zero_frequency_rejected_subunity_legal(self):
        from uacpy.acoustic_signal.timefreq import cwt
        x = np.random.default_rng(5).normal(size=256)
        with pytest.raises(ConfigurationError):
            cwt(x, 100.0, frequencies=np.array([0.0, 10.0]))
        r = cwt(x, 100.0, frequencies=np.array([0.5, 10.0]))
        assert np.isfinite(np.asarray(r.coefficients)).all()


class TestSpectrogramSubHertzSampleRate:
    def test_legal(self):
        from uacpy.acoustic_signal.timefreq import spectrogram
        r = spectrogram(np.random.default_rng(6).normal(size=256), 0.5,
                        nperseg=64)
        assert np.isfinite(np.asarray(r.power)).all()


class TestCepstrumWindowConvention:
    """``window=`` multiplies by the *periodic* (fftbins) window before
    the spectrum."""

    def test_matches_prewindowed_signal(self):
        import scipy.signal as _sig
        from uacpy.acoustic_signal.timefreq import cepstrum
        x = np.random.default_rng(7).normal(size=64)
        w = _sig.get_window("hann", 64, fftbins=True)
        np.testing.assert_allclose(cepstrum(x, window="hann"),
                                   cepstrum(x * w), rtol=1e-12, atol=1e-12)


class TestComplexCepstrumDelayEstimator:
    """The linear-phase ramp is rounded to whole samples: a pure delayed
    impulse reports exactly its ramp; degenerate lengths report zero
    without dividing by zero."""

    def test_delayed_impulse_reports_the_ramp(self):
        from uacpy.acoustic_signal.timefreq import complex_cepstrum
        x = np.zeros(31)
        x[3] = 1.0
        r = complex_cepstrum(x)
        assert r.delay == 3            # three samples LATE: positive
        assert np.isfinite(r.cepstrum).all()

    def test_two_sample_signal_computes_the_ramp(self):
        from uacpy.acoustic_signal.timefreq import complex_cepstrum
        # Two samples: the one-sample delay sits at Nyquist, where +1 and -1
        # are the same ramp — only its magnitude is determined.
        assert abs(complex_cepstrum(np.array([0.0, 1.0])).delay) == 1

    def test_single_sample_signal_is_zero_delay(self):
        from uacpy.acoustic_signal.timefreq import complex_cepstrum
        r = complex_cepstrum(np.array([2.0]))
        assert r.delay == 0
        assert np.isfinite(r.cepstrum).all()


# ─────────────────────────────────────────────────────────────────────────────
# io/_fortran_helpers.py — endian probe values, raw records, SubTab replicate
# ─────────────────────────────────────────────────────────────────────────────


class TestBoundCountsUnitItemBytes:
    """``min_item_bytes=1`` means one byte per item — the capacity is the
    file size itself, not half of it."""

    def test_byte_sized_items_fill_the_file(self):
        from uacpy.io._fortran_helpers import _bound_counts
        _bound_counts('f.bin', 64, 1, n=64)


class TestDetectEndianUnitMarkers:
    """A record marker of exactly 1 is a plausible length on the side that
    reads it as 1 — not on the side that reads it as 16777216-and-warns."""

    def test_little_endian_one_is_little(self):
        import struct
        from uacpy.io._fortran_helpers import detect_endian
        assert detect_endian(struct.pack('<i', 1)) == '<'

    def test_big_endian_one_is_big(self):
        import struct
        import uacpy.io._fortran_helpers as fh
        emitted = fh._ENDIAN_WARN_EMITTED
        try:
            fh._ENDIAN_WARN_EMITTED = False
            with pytest.warns(UserWarning, match="big-endian"):
                assert fh.detect_endian(struct.pack('>i', 1)) == '>'
        finally:
            fh._ENDIAN_WARN_EMITTED = emitted


class TestEndianWarningFiresOnce:
    """The big-endian warning is emitted once per process, then latched."""

    def test_second_detect_is_silent(self):
        import struct
        import warnings as _w
        import uacpy.io._fortran_helpers as fh
        emitted = fh._ENDIAN_WARN_EMITTED
        try:
            fh._ENDIAN_WARN_EMITTED = False
            with pytest.warns(UserWarning, match="big-endian"):
                fh.detect_endian(struct.pack('>i', 128))
            with _w.catch_warnings():
                _w.simplefilter('error', UserWarning)
                fh.detect_endian(struct.pack('>i', 128))
        finally:
            fh._ENDIAN_WARN_EMITTED = emitted


class TestRawRecordIgnoresFmt:
    """``raw=True`` returns the payload bytes even when a ``fmt`` is also
    given."""

    def test_raw_with_fmt_returns_bytes(self):
        import struct
        from uacpy.io._fortran_helpers import read_fortran_record
        payload = struct.pack('<i', 7)
        f = _io.BytesIO(struct.pack('<i', 4) + payload + struct.pack('<i', 4))
        assert read_fortran_record(f, fmt='i', raw=True) == payload


class TestReadVectorReplicateBranch:
    """SubTab's replicate branch: one value with ``/`` at Nx >= 3 fills
    the vector with that value — including at Nx = 3 exactly."""

    def test_single_value_replicates_at_nx_3(self):
        from uacpy.io._fortran_helpers import read_vector
        x, nx = read_vector(_io.StringIO("3\n5.0 /\n"))
        assert nx == 3
        np.testing.assert_allclose(x, [5.0, 5.0, 5.0])


class TestReadVectorEmptyCount:
    """``Nx <= 0`` returns immediately without consuming further records."""

    def test_stream_position_is_preserved(self):
        from uacpy.io._fortran_helpers import read_vector
        f = _io.StringIO("0\nNEXT RECORD\n")
        x, nx = read_vector(f)
        assert nx == 0 and len(x) == 0
        assert f.readline().strip() == "NEXT RECORD"


class TestProbabilisticConstantQDefaultLevels:
    """The *default* level range is 0..150 dB — pinned without passing
    lvlmin/lvlmax explicitly."""

    def test_default_edges(self):
        from uacpy.acoustic_signal.constant_q import probabilistic_constant_q
        x = np.sin(2 * np.pi * 100.0 * np.arange(4096) / 1000.0)
        r = probabilistic_constant_q(x, 1000.0, fmin=100.0,
                                     bins_per_octave=2)
        assert r.level_edges[0] == 0.0
        assert r.level_edges[-1] == 150.0


class TestConstantQHopInnerFloor:
    """The signal-length term of the default hop, ``max(1, n//8)``, floors
    at one for signals under 16 samples even when the lowest window is
    longer."""

    def test_short_signal_with_long_window(self):
        from uacpy.acoustic_signal.constant_q import _resolve_hop
        assert _resolve_hop(None, [(24, None, None)], 8, "t") == 1


class TestWignerSmoothingWindowSingleSampleArray:
    """A one-sample user window is legal on the array path too."""

    def test_single_sample_array(self):
        from uacpy.acoustic_signal.timefreq import _smoothing_window
        w, half = _smoothing_window(np.array([0.7]), "time")
        assert w.size == 1 and half == 0


class TestWignerSmoothingWindowDeletesTheTrueCentre:
    """hann(8)'s two equal middles sit at indices 3 and 4: the deletion
    must take one of them, keeping the peak on-centre (hann(6) cannot
    discriminate — its index 6//3 = 2 is the other equal middle)."""

    def test_even_eight_array(self):
        import scipy.signal as _sig
        from uacpy.acoustic_signal.timefreq import _smoothing_window
        w8 = _sig.get_window("hann", 8, fftbins=False)
        w, half = _smoothing_window(w8, "time")
        assert w.size == 7 and half == 3
        np.testing.assert_allclose(w, w[::-1], rtol=0, atol=1e-15)
        assert w.argmax() == half


class TestInverseRadonOffsetAxisOrigin:
    """``inverse_radon`` builds the same ``arange(nx)·dx − x0`` offset
    axis: a single linear-slowness Radon sample back-projects onto
    ``t = tau`` exactly at the x = 0 trace."""

    def test_zero_offset_trace_takes_tau(self):
        from uacpy.acoustic_signal.transforms import inverse_radon
        fs, dx, nx = 100.0, 10.0, 5
        R = np.zeros((3, 64))
        R[2, 20] = 1.0  # moveout p = +1e-3 s/m, tau index 20
        out = inverse_radon(R, fs, dx, np.array([-1e-3, 0.0, 1e-3]), nx,
                            kind="linear", x0=2 * dx)
        col = out[:, 2]
        assert col.argmax() == 20

    def test_hyperbolic_guard_boundaries(self):
        from uacpy.acoustic_signal.transforms import inverse_radon
        R = np.zeros((1, 32))
        inverse_radon(R, 100.0, 10.0, np.array([0.5]), 4,
                      kind="hyperbolic")
        with pytest.raises(ConfigurationError):
            inverse_radon(R, 100.0, 10.0, np.array([0.0]), 4,
                          kind="hyperbolic")
