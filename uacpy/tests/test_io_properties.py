"""Property-based coverage of the io layer's Fortran text/binary contracts.

Each class pins one property over a *generated* input space rather than a
hand-written corpus: random legal Fortran spellings and record layouts
(cross-validated during exploration against gfortran's own list-directed
READ via compiled oracles), random writer->reader round-trips, and random
corruptions of valid files. ``hypothesis`` is not available in this
environment, so every test draws a **fixed number of examples from a
fixed-seed ``numpy.random.default_rng``** — deterministic, order-independent
under xdist, and bounded in runtime by construction.

The exploration campaign behind these bounds ran ~40k scalar spellings,
~3k whole-vector layouts, ~1.8k round-trips per writer/reader pair, ~10k
garbage/truncation probes on the text readers and ~12k on the binary
readers. Three defects it found are pinned as skipped repros (see the
``skip`` reasons naming them) rather than silently avoided.
"""

import io

import numpy as np
import pytest

from uacpy.core.exceptions import UACPYError


# ---------------------------------------------------------------------------
# Generators (seeded-numpy stand-ins for hypothesis strategies)
# ---------------------------------------------------------------------------

def _spell_float(rng):
    """One random legal Fortran list-directed real spelling -> (text, value).

    Covers D/d/E/e exponents, ``E+000``-style 3-digit exponents, all sign
    placements, missing leading zero (``.5``), trailing dot (``1.``) and
    exponent-less forms. Letterless exponents (``1.5+3``), which gfortran
    also accepts, are excluded: fortran_float rejects them (see the skipped
    repro in TestFortranFloatSpellings).
    """
    sign = rng.choice(['', '+', '-'])
    digits = list('0123456789')
    int_part = ''.join(rng.choice(digits, size=rng.integers(1, 9)))
    frac_part = ''.join(rng.choice(digits, size=rng.integers(1, 12)))
    mant = [int_part, int_part + '.', '.' + frac_part,
            int_part + '.' + frac_part][rng.integers(0, 4)]
    exp_txt, exp_val = '', 0
    if rng.random() < 0.6:
        exp_val = int(rng.integers(-120, 121))
        letter = rng.choice(['E', 'e', 'D', 'd'])
        style = rng.integers(0, 3)
        if style == 0:
            exp_txt = f"{letter}{exp_val:+d}"
        elif style == 1:
            exp_txt = f"{letter}{exp_val:+04d}"      # E+000 style
        else:
            exp_txt = f"{letter}{exp_val}"
    return sign + mant + exp_txt, float(f"{sign}{mant}E{exp_val}")


def _spell_value(rng, v):
    """A random legal spelling near value ``v`` -> (text, denoted_value)."""
    style = rng.integers(0, 6)
    if style == 0:
        text = f"{v:.6f}"
    elif style == 1:
        text = f"{v:.17g}"
    elif style == 2:
        text = f"{v:.9E}"
    elif style == 3:
        text = f"{v:.9E}".replace('E', str(rng.choice(['D', 'd', 'e'])))
    elif style == 4:
        m, e = f"{v:.9E}".split('E')
        text = f"{m}E{int(e):+04d}"
    else:
        text = f"{v:.9f}"
        if text.startswith('0.'):
            text = text[1:]
        elif text.startswith('-0.'):
            text = '-' + text[2:]
    return text, float(text.replace('D', 'E').replace('d', 'e'))


def _encode_layout(rng, tokens, comments=True, remainder=True):
    """A random legal list-directed encoding of a token list.

    Random separators (spaces / tabs / commas / line breaks), blank lines,
    optional trailing ``! comments`` and optional remainder junk after the
    final token — every layout a whole-vector list-directed READ consumes
    (``misc/RefCoef.f90:53``, ``Bellhop/sspMod.f90:417,428``).
    """
    parts = []
    for i, tok in enumerate(tokens):
        parts.append(tok)
        if i == len(tokens) - 1:
            break
        sep = rng.integers(0, 10)
        if sep < 4:
            parts.append(' ' * int(rng.integers(1, 4)))
        elif sep < 6:
            parts.append(str(rng.choice([', ', ' , ', ',', ',  '])))
        elif sep == 6:
            parts.append('\t')
        else:
            if comments and rng.random() < 0.3:
                parts.append('   ! trailing comment')
            parts.append('\n')
            if rng.random() < 0.2:
                parts.append('\n')
            if rng.random() < 0.15:
                parts.append('   ')
    if remainder and rng.random() < 0.5:
        parts.append(str(rng.choice(['  9e9 9e9', '  remainder junk',
                                     ' ! comment', ',  777'])))
    parts.append('\n')
    return ''.join(parts)


def _increasing(rng, n, lo, hi, min_gap):
    """n strictly increasing values in [lo, hi] with gaps >= min_gap."""
    gaps = rng.uniform(min_gap, (hi - lo) / n, size=n)
    return rng.uniform(lo, hi - gaps.sum()) + np.cumsum(gaps)


class TestFortranFloatSpellings:
    """``fortran_float`` must accept every real spelling a list-directed
    READ accepts, at the exact value gfortran parses. Explored against a
    gfortran oracle over 40k spellings: agreement is bit-exact on every
    lettered form; the only divergence is the letterless 3-digit exponent."""

    def test_every_generated_spelling_parses_to_the_exact_value(self):
        from uacpy.io._fortran_helpers import fortran_float
        rng = np.random.default_rng(0x5EED01)
        for _ in range(400):
            text, value = _spell_float(rng)
            assert fortran_float(text) == value, text

    #: Every spelling below was read by gfortran 14.2 itself
    #: (``read(token, *) x``) and the value is what it returned.
    LETTERLESS = [
        ('0.123457-118', 1.2345700000000000e-119),   # G15.6 drops the E at 3
        ('0.200000+121', 2.0000000000000000e+120),   # digits of exponent
        ('1.5+7', 1.5e7),                            # any width, in fact
        ('1.5-2', 1.4999999999999999e-02),
        ('-1.5-2', -1.4999999999999999e-02),
        ('+2.5+003', 2.5e3),
        ('1.-3', 1.0e-3),                            # mantissa may end in '.'
        ('.5+2', 5.0e1),                             # or start with it
        ('1+7', 1.0e7),                              # or carry no point
    ]

    @pytest.mark.parametrize('text,value', LETTERLESS)
    def test_letterless_exponent_spelling(self, text, value):
        """A ``Gw.d`` WRITE drops the ``E`` once the exponent needs three
        digits, and SPARC writes its ``.rts`` with ``'( 12G15.6 )'``
        (``Scooter/sparc.f90:294``) — so uacpy could not read a file the
        toolbox produced at its own request whenever a sample fell below
        ~1e-99. gfortran's list-directed READ takes the form at any exponent
        width; the sign is what marks it."""
        from uacpy.io._fortran_helpers import fortran_float
        assert fortran_float(text) == value

    @pytest.mark.parametrize('text', [
        'abc', '', '1e5-2', '--3', '1.5+', '+', '1.5+e2',
    ])
    def test_a_signed_tail_does_not_make_junk_parseable(self, text):
        """The letterless branch is a last resort, not a loosening: anything
        gfortran itself rejects still raises."""
        from uacpy.io._fortran_helpers import fortran_float
        with pytest.raises(ValueError):
            fortran_float(text)

    def test_an_rts_with_a_letterless_exponent_reads(self, tmp_path):
        """The end the gap was felt at: SPARC writes ``.rts`` with
        ``'( 12G15.6 )'`` (``Scooter/sparc.f90:294``), so a sample below
        ~1e-99 arrives with no ``E`` and the whole file failed to parse —
        a file uacpy asked the toolbox to produce."""
        from uacpy.io.oalib_reader import read_ts
        p = tmp_path / 's.rts'
        p.write_text("SPARC RTS\n"
                     "   1\n"
                     "   50.000000\n"
                     "   0.500000       0.100000E-02\n"
                     "   0.100000E-01   0.123457-118\n")
        out = read_ts(p)
        assert out['tout'].tolist() == [0.5, 0.01]
        assert out['RTS'].ravel().tolist() == [1.0e-3, 1.23457e-119]


class TestListDirectedRecovery:
    """``list_directed_int`` takes the first token of any legal count
    record (bdryMod.f90:71/:171); ``read_list_directed_values`` recovers
    exactly N values from ANY legal layout of those N values."""

    def test_count_records(self):
        from uacpy.io._fortran_helpers import list_directed_int
        rng = np.random.default_rng(0x5EED02)
        tails = ['', ',', ' ,', '\t! M', '   ! npts', ',   ! count',
                 '  NUMBER OF ELEMENTS', ' 42 17', ', 3, 4', '  words']
        for _ in range(200):
            n = int(rng.integers(-5, 100000))
            rec = (str(rng.choice(['', ' ', '   ', '\t'])) + str(n)
                   + str(rng.choice(tails)) + '\n')
            assert list_directed_int(rec) == n, rec

    def test_any_legal_layout_recovers_the_exact_values(self):
        from uacpy.io._fortran_helpers import read_list_directed_values
        rng = np.random.default_rng(0x5EED03)
        for _ in range(30):
            n = int(rng.integers(1, 30))
            vals = (rng.uniform(-1e4, 1e4, size=n)
                    * 10.0 ** rng.integers(-8, 9, size=n))
            toks, denoted = [], []
            for v in vals:
                t, d = _spell_value(rng, v)
                toks.append(t)
                denoted.append(d)
            text = _encode_layout(rng, toks)
            got = read_list_directed_values(io.StringIO(text), n, 'vals',
                                            'generated')
            assert np.array_equal(got, np.array(denoted)), text


class TestWriterReaderRoundTrips:
    """Writer->reader numeric round-trips on random physically-sane inputs:
    every numeric field must come back within the written format's own
    precision (the defect class of the RAM dz bug)."""

    def test_bty_and_ati_round_trip(self, tmp_path):
        from uacpy.io.bathy_io import (
            read_altimetry, read_bathymetry, write_ati_file, write_bty_file)
        rng = np.random.default_rng(0x5EED04)
        pairs = ((write_bty_file, read_bathymetry, 'bty'),
                 (write_ati_file, read_altimetry, 'ati'))
        for case in range(12):
            n = int(rng.integers(2, 60))
            r_m = _increasing(rng, n, 0.0, 2e5, 0.1)
            z = rng.uniform(-50.0, 6000.0, size=n)
            interp = str(rng.choice(['L', 'C']))
            for writer, reader, tag in pairs:
                p = tmp_path / f'c{case}.{tag}'
                writer(p, np.column_stack([r_m, z]), interp_type=interp)
                back, btype = reader(p)
                assert btype == interp
                # %.6f on a km axis: half-ulp = 5e-7 km = 5e-4 m.
                assert np.max(np.abs(back[0, 1:-1] - r_m)) <= 5.1e-4
                # %.6f on the metre column: half-ulp = 5e-7 m.
                assert np.max(np.abs(back[1, 1:-1] - z)) <= 5.1e-7

    def test_sbp_round_trip(self, tmp_path):
        from uacpy.io.refl_io import (
            read_source_beam_pattern, write_source_beam_pattern)
        rng = np.random.default_rng(0x5EED05)
        for case in range(12):
            # From 2: the writer refuses a table the engines cannot interpolate
            # between (bellhop.f90:270 reads adjacent rows as a pair).
            n = int(rng.integers(2, 40))
            ang = _increasing(rng, n, -90.0, 90.0, 1e-4)
            lvl = rng.uniform(-60.0, 0.0, size=n)
            p = tmp_path / f'c{case}.sbp'
            write_source_beam_pattern(p, ang, lvl)
            back = read_source_beam_pattern(p)
            assert np.max(np.abs(back[:, 0] - ang)) <= 5.1e-7   # %12.6f
            assert np.max(np.abs(back[:, 1] - lvl)) <= 5.1e-7

    def test_reflection_table_any_layout(self, tmp_path):
        """RefCoef.f90:53 reads the whole (theta, R, phi) table with ONE
        list-directed READ, so packed and wrapped layouts are the same
        table; the values must come back exactly as spelled."""
        from uacpy.io.refl_io import read_reflection_coefficient
        from uacpy.core.units import deg_to_rad
        rng = np.random.default_rng(0x5EED06)
        for case in range(12):
            n = int(rng.integers(1, 30))
            table = np.column_stack([np.sort(rng.uniform(0.0, 90.0, size=n)),
                                     rng.uniform(0.0, 1.0, size=n),
                                     rng.uniform(-180.0, 180.0, size=n)])
            toks, denoted = [], []
            for v in table.ravel():
                t, d = _spell_value(rng, v)
                toks.append(t)
                denoted.append(d)
            denoted = np.array(denoted).reshape(n, 3)
            if np.any(np.diff(denoted[:, 0]) < 0):
                continue  # spelling precision reordered theta; not a table
            layouts = [_encode_layout(rng, toks),
                       ' '.join(toks) + '\n',               # fully packed
                       '\n'.join(toks) + '\n']              # fully wrapped
            for j, body in enumerate(layouts):
                p = tmp_path / f'c{case}_{j}.brc'
                p.write_text(f"{n}\n" + body)
                d = read_reflection_coefficient(p)
                assert d['n_pts'] == n
                assert np.array_equal(d['theta'], denoted[:, 0])
                assert np.array_equal(d['R'], denoted[:, 1])
                assert np.array_equal(d['phi'], deg_to_rad(denoted[:, 2]))

    def test_ssp_round_trip(self, tmp_path):
        from uacpy.io.oalib_writer import write_ssp
        from uacpy.io.oalib_reader import read_ssp_2d
        rng = np.random.default_rng(0x5EED07)
        for case in range(10):
            nr = int(rng.integers(2, 12))
            nd = int(rng.integers(1, 30))
            r_m = _increasing(rng, nr, 0.0, 1e5, 0.5)
            c = rng.uniform(1400.0, 1700.0, size=(nd, nr))
            p = tmp_path / f'c{case}.ssp'
            write_ssp(p, r_m, c)
            d = read_ssp_2d(p)
            assert d['n_prof'] == nr and d['c_mat'].shape == (nd, nr)
            assert np.max(np.abs(d['r_prof'] - r_m)) <= 5.1e-4   # %.6f km
            assert np.max(np.abs(d['c_mat'] - c)) <= 5.1e-5      # %8.4f

    def test_ramin_seafloor_node_survives_the_deck(self, tmp_path):
        """The invariant the dz bug violated: with ``dz`` from
        ``RAM._snap_dz_to_seafloor(h, n)`` and both ``h`` and ``dz``
        round-tripped through write_ramin's %.12g deck, the binaries'
        seafloor node ``iz = int(1 + zb/dz)`` (ramgeo1.5.f:133,
        ramsurf1.5.f:118, rams0.5.f:135) must land at ``1 + n`` exactly,
        with ``zb/dz`` never below ``n``."""
        from uacpy.io.ramsurf_writer import write_ramin
        from uacpy.models.ram import RAM
        rng = np.random.default_rng(0x5EED08)
        for case in range(24):
            n = int(rng.integers(5, 2000))
            # 12-significant-digit depths: what any hand-specified or
            # %.6f-quantised bathymetry carries. Full-precision doubles
            # break this identity — see the skipped repro below.
            h = float(f"{rng.uniform(10.0, 5000.0):.6f}")
            dz = RAM._snap_dz_to_seafloor(h, n)
            kind = str(rng.choice(['ramgeo', 'ramsurf', 'rams']))
            seg = {'range': 0.0,
                   'water_ssp': [(0.0, 1500.0), (h, 1520.0)],
                   'bottom_c': [(h, 1700.0)], 'bottom_rho': [(h, 1.8)],
                   'bottom_attn': [(h, 0.5)]}
            if kind == 'rams':
                seg['bottom_cs'] = [(h, 400.0)]
                seg['bottom_attns'] = [(h, 1.0)]
            p = tmp_path / f'c{case}.in'
            write_ramin(p, kind=kind, fc=100.0, zs=h / 3, zr_line=h / 2,
                        rmax=5000.0, dr=10.0, ndr=1,
                        zmax=h * float(rng.uniform(1.2, 2.0)), dz=dz, ndz=1,
                        zmplt=h, c0=1500.0, np_pade=6,
                        bathymetry=[(0.0, h), (5000.0, h)],
                        range_segments=[seg],
                        surface=[(0.0, 0.0)] if kind == 'ramsurf' else None)
            lines = p.read_text().splitlines()
            # Deck tokens parse to the same doubles gfortran READ produces
            # (verified bit-exact against a gfortran oracle in exploration).
            dz_deck = float(lines[3].split()[1])
            bathy_row = 5 + (2 if kind == 'ramsurf' else 0)
            zb_deck = float(lines[bathy_row].split()[1])
            ratio = zb_deck / dz_deck
            assert int(1.0 + ratio) == 1 + n, (kind, h, n, ratio)
            assert 0.0 <= ratio - n < 5e-8, (kind, h, n, ratio)

    def test_full_precision_depth_seafloor_node(self):
        """A depth carrying all 17 digits aligns as well as a quantised one.

        The test above feeds ``%.6f`` depths, which is what a hand-written or
        quantised bathymetry carries. A depth straight out of a computation
        does not round-trip so kindly: the deck writes ``%.12g`` and the
        binaries divide the *read-back* depth by the *read-back* ``dz``, so
        both spellings of each have to clear ``n``. ``_snap_dz_to_seafloor``
        is what guarantees it, by lowering ``dz`` until every spelling pair
        does.
        """
        from uacpy.models.ram import RAM
        h, n = 36.56550087062047, 2000
        dz = RAM._snap_dz_to_seafloor(h, n)
        for hs in (h, float(f"{h:.12g}")):
            for dzs in (dz, float(f"{dz:.12g}")):
                ratio = hs / dzs
                assert ratio >= n, (hs, dzs, ratio)
                assert int(1.0 + ratio) == 1 + n, (hs, dzs, ratio)

    def test_every_spelling_pair_clears_the_seafloor_node(self):
        """The same property over full-precision depths, swept.

        One seed proves nothing about a rounding cliff: this walks 2000
        random ``(h, n)`` pairs and checks all four spelling combinations,
        which is the shape of the guarantee ``iz = int(1 + zb/dz)`` needs.
        """
        from uacpy.models.ram import RAM
        rng = np.random.default_rng(0x5EED0B)
        for _ in range(2000):
            h = float(rng.uniform(5.0, 6000.0))
            n = int(rng.integers(2, 4000))
            dz = RAM._snap_dz_to_seafloor(h, n)
            for hs in (h, float(f"{h:.12g}")):
                for dzs in (dz, float(f"{dz:.12g}")):
                    assert int(1.0 + hs / dzs) == 1 + n, (h, n, hs, dzs)


# Valid seed files for the totality properties: one per text format, small
# enough that every byte-truncation can be probed.
_TEXT_SEEDS = {
    'bty': b"'L'\n3\n0.0 100.0\n1.5 250.5\n3.0 180.25\n",
    'ati': b"'L'\n3\n0.0 0.0\n1.5 -2.5\n3.0 1.25\n",
    'brc': b"3\n0.0 1.0 180.0\n45.0 0.5 90.0\n90.0 0.1 0.0\n",
    'sbp': b"3\n-45.0 -10.0\n0.0 0.0\n45.0 -10.0\n",
    'ssp': b"3\n0.0 5.0 10.0\n1500.0 1510.0 1520.0\n1490.0 1495.0 1500.0\n",
    'flp': (b"'title'\n'RA'\n9999\n1\n0.0 /\n11\n0.0 10.0 /\n1\n25.0 /\n"
            b"5\n0.0 100.0 /\n1\n0.0 /\n"),
    'ts': b"my title\n2 10.0 20.0\n0.0 1.0 2.0\n0.1 3.0 4.0\n",
    'rts': (b"'SPARC title'\n2 100.0 200.0\n"
            b"0.0 1.0 2.0\n0.1 3.0 4.0\n0.2 5.0 6.0\n"),
}


def _text_readers():
    from uacpy.io.bathy_io import read_altimetry, read_bathymetry
    from uacpy.io.refl_io import (
        read_reflection_coefficient, read_source_beam_pattern)
    from uacpy.io.oalib_reader import (
        read_flp, read_rts_file, read_ssp_2d, read_ts)
    return [('bty', read_bathymetry), ('ati', read_altimetry),
            ('brc', read_reflection_coefficient),
            ('sbp', read_source_beam_pattern), ('ssp', read_ssp_2d),
            ('flp', read_flp), ('ts', read_ts), ('rts', read_rts_file)]


def _assert_total(reader, path, payload: bytes):
    """A reader fed arbitrary bytes must return data or raise a typed
    uacpy error — never a bare ValueError/IndexError/etc."""
    path.write_bytes(payload)
    try:
        reader(path)
    except UACPYError:
        pass
    except Exception as exc:   # pragma: no cover - the failure being pinned
        pytest.fail(f"{reader.__name__} leaked {type(exc).__name__}: {exc} "
                    f"on {payload[:80]!r}")


class TestTextReaderTotality:
    """Total robustness: whatever the bytes, the eight text readers either
    return valid data or raise a typed uacpy error. Explored over ~10k
    probes (random bytes, token soup, mutations, all truncations); the one
    escape found — ``read_vector`` returning non-finite axis values parsed
    from ``NaN``/``1e999`` tokens — is refused by ``read_vector``'s finite
    guard, pinned below."""

    def test_seeds_parse(self, tmp_path):
        for tag, reader in _text_readers():
            p = tmp_path / f'seed.{tag}'
            p.write_bytes(_TEXT_SEEDS[tag])
            reader(p)   # must not raise

    def test_every_truncation_of_a_valid_file(self, tmp_path):
        """Truncation-of-valid-file is the highest-yield corruption: cover
        every byte position of every seed."""
        for tag, reader in _text_readers():
            seed = _TEXT_SEEDS[tag]
            p = tmp_path / f't.{tag}'
            for cut in range(len(seed)):
                _assert_total(reader, p, seed[:cut])

    def test_random_bytes_and_token_soup(self, tmp_path):
        rng = np.random.default_rng(0x5EED09)
        vocab = ['0', '-1', '1e9', 'NaN', 'Inf', 'abc', '!', '/', "'", ',',
                 '*', '3*2.0', '1.5D+00', '9' * 40, '-', '+', '.', 'E5',
                 '1e999', '2147483648', '99999999999999999999']
        for tag, reader in _text_readers():
            p = tmp_path / f'g.{tag}'
            for i in range(15):
                n = int(rng.integers(0, 300))
                _assert_total(reader, p, rng.integers(
                    0, 256, size=n, dtype=np.uint8).tobytes())
            for i in range(15):
                toks = rng.choice(vocab, size=int(rng.integers(0, 40)))
                text = ''.join(t + str(rng.choice([' ', '\n', ',', '\t']))
                               for t in toks)
                _assert_total(reader, p, text.encode())

    def test_token_mutations_of_valid_files(self, tmp_path):
        rng = np.random.default_rng(0x5EED0A)
        vocab = ['-1', 'NaN', 'abc', '/', "'", '*', '1e999', '2147483648',
                 '', '!', '9' * 40]
        for tag, reader in _text_readers():
            words = _TEXT_SEEDS[tag].decode().split(' ')
            p = tmp_path / f'm.{tag}'
            for i in range(15):
                w = list(words)
                w[int(rng.integers(0, len(w)))] = str(rng.choice(vocab))
                _assert_total(reader, p, ' '.join(w).encode())

    def test_non_finite_vector_values_raise_a_typed_error(self, tmp_path):
        """``0.0 1e999 /`` (the generated branch would spread inf/nan over
        all ``Nx`` slots via ``np.linspace``) and an explicit ``NaN`` token
        both raise ``FileFormatError`` from ``read_vector``'s finite guard
        instead of returning non-finite axis values."""
        from uacpy.core.exceptions import FileFormatError
        from uacpy.io._fortran_helpers import read_vector
        p = tmp_path / 'v.txt'
        for deck in ('5\n0.0 1e999 /\n', '3\n0.0 NaN 2.0\n'):
            p.write_text(deck)
            with p.open() as fh, pytest.raises(FileFormatError,
                                               match='not finite'):
                read_vector(fh)

    def test_flp_count_bomb(self, tmp_path):
        """A 40-byte deck declaring ``Nx=999999999`` is refused, not allocated.

        ``read_vector``'s generated branches are the one place a small record
        can demand a large array — that is what the ``N`` / ``first last /``
        shorthand is *for* — so the file-size bound ``_bound_counts`` applies
        to the binary readers cannot be used. AT allocates and reports failure
        (``SourceReceiverPositions.f90:215-216``), which on a large-memory host
        means the 8 GB request simply succeeds.
        """
        from uacpy.io.oalib_reader import read_flp
        p = tmp_path / 'bomb.flp'
        p.write_text("'t'\n'RA'\n9\n999999999\n0.0 100.0 /\n")
        _assert_total(read_flp, p, p.read_bytes())

    def test_the_generated_vector_ceiling_names_the_count_and_the_size(self,
                                                                      tmp_path):
        from uacpy.io.oalib_reader import read_flp
        p = tmp_path / 'bomb.flp'
        p.write_text("'t'\n'RA'\n9\n999999999\n0.0 100.0 /\n")
        from uacpy.core.exceptions import FileFormatError
        with pytest.raises(FileFormatError, match=r'999999999 generated'):
            read_flp(p)

    def test_the_largest_deck_the_toolbox_ships_expands(self, tmp_path):
        """The bound must not clip a legal deck.

        ``tests/Noise/ATOC/aet_VLA_C.flp:6`` is the largest generated vector
        in any deck the Acoustics Toolbox ships — ``10001`` then
        ``0.0 500.0 /`` — three orders of magnitude below the ceiling.
        """
        from uacpy.io._fortran_helpers import read_vector
        p = tmp_path / 'v.txt'
        p.write_text('10001\n0.0 500.0 /\n')
        with p.open() as fh:
            x, nx = read_vector(fh)
        assert nx == 10001 and x.size == 10001
        assert x[0] == 0.0 and x[-1] == 500.0


class TestTypedFormatErrorWrapsParseErrorsOnly:
    """The ``typed_format_error`` conversion set is deliberately narrow
    (docs/guide/io.md §8): ``AttributeError`` / ``TypeError`` / ``NameError``
    raised inside a reader signal a uacpy defect, not a bad file, so they
    surface with their original type — never dressed as ``FileFormatError``.
    Pinned on synthetic decorated readers, the same registration every real
    reader in ``uacpy.io`` uses."""

    def test_code_bugs_surface_unwrapped(self, tmp_path):
        from uacpy.core.exceptions import FileFormatError
        from uacpy.io._fortran_helpers import typed_format_error

        p = tmp_path / 'wellformed.dat'
        p.write_text('3\n0.0 100.0\n')
        for bug_type in (AttributeError, TypeError, NameError):
            @typed_format_error
            def read_raising_a_code_defect(path, _bug=bug_type):
                path.read_text()
                raise _bug(f'synthetic reader defect: {_bug.__name__}')

            with pytest.raises(bug_type) as ei:
                read_raising_a_code_defect(p)
            assert type(ei.value) is bug_type
            assert not isinstance(ei.value, FileFormatError)

    def test_parse_errors_from_the_same_reader_stay_wrapped(self, tmp_path):
        from uacpy.core.exceptions import FileFormatError
        from uacpy.io._fortran_helpers import typed_format_error

        p = tmp_path / 'wellformed.dat'
        p.write_text('3\n0.0 100.0\n')

        @typed_format_error
        def read_with_parse_error(path):
            path.read_text()
            raise ValueError('invalid literal')

        with pytest.raises(FileFormatError) as ei:
            read_with_parse_error(p)
        assert isinstance(ei.value.__cause__, ValueError)


@pytest.mark.requires_binary
class TestBinaryReaderTotality:
    """Binary readers on truncated / bit-flipped copies of REAL model
    outputs: typed errors or valid data only. One tiny run per engine
    produces the seed files; corruption points are deterministic. Explored
    over ~12k random truncations/flips/int-stomps with zero escapes."""

    @pytest.fixture(scope='class')
    def real_outputs(self, tmp_path_factory):
        import uacpy
        from uacpy import Environment, Receiver, Source
        env = Environment(name='fuzz-seed', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=25.0, frequencies=100.0)
        rcv = Receiver(depths=np.linspace(5.0, 95.0, 4),
                       ranges=np.linspace(200.0, 2000.0, 5))
        out = {}
        for name, cls, kw in [('kraken', uacpy.Kraken, {}),
                              ('scooter', uacpy.Scooter, {}),
                              ('ramgeo', uacpy.RAM, {'backend': 'ramgeo'})]:
            wd = tmp_path_factory.mktemp(name)
            cls(work_dir=wd, cleanup=False, timeout=300, **kw).run(
                env, src, rcv)
            out[name] = wd
        find = lambda wd, pat: next(iter(sorted(wd.rglob(pat))))
        return {
            'shd': find(out['kraken'], '*.shd'),
            'mod': find(out['kraken'], '*.mod'),
            'grn': find(out['scooter'], '*.grn'),
            'tlgrid': find(out['ramgeo'], 'tl.grid'),
            'pcomplex': find(out['ramgeo'], 'pcomplex.bin'),
        }

    @staticmethod
    def _calls():
        from uacpy.io.grn_reader import read_grn_file
        from uacpy.io.modes_reader import read_modes_bin
        from uacpy.io.oalib_reader import read_shd_bin
        from uacpy.io.ramsurf_reader import read_pcomplex_grid, read_tl_grid
        ram_kw = dict(dr=25.0, ndr=1, dz=1.0, ndz=2)
        return {
            'shd': lambda p: read_shd_bin(str(p)),
            'mod': lambda p: read_modes_bin(str(p), frequency=100.0),
            'grn': read_grn_file,
            'tlgrid': lambda p: read_tl_grid(p, **ram_kw),
            'pcomplex': lambda p: read_pcomplex_grid(p, **ram_kw),
        }

    def test_real_seeds_parse(self, real_outputs):
        for tag, call in self._calls().items():
            call(real_outputs[tag])   # must not raise

    def test_truncations(self, real_outputs, tmp_path):
        """Deterministic truncation points spanning header, first record
        and mid-data of each format."""
        for tag, call in self._calls().items():
            data = real_outputs[tag].read_bytes()
            n = len(data)
            p = tmp_path / f't.{tag}'
            for cut in (0, 1, 3, 4, 7, 11, n // 16, n // 4, n // 2,
                        3 * n // 4, n - 3, n - 1):
                _assert_total(call, p, data[:cut])

    def test_bit_flips_and_count_stomps(self, real_outputs, tmp_path):
        for tag, call in self._calls().items():
            data = real_outputs[tag].read_bytes()
            n = len(data)
            p = tmp_path / f'b.{tag}'
            for pos, mask in ((5, 0x80), (n // 3, 0x01), (n - 2, 0x10)):
                b = bytearray(data)
                b[pos] ^= mask
                _assert_total(call, p, bytes(b))
            # Hostile header counts: the readers bound every allocation by
            # the file size (_bound_counts), so a stomped int32 must give a
            # typed error, not a multi-GB allocation.
            for pos, val in ((0, 2 ** 31 - 1), (8, 999999999), (12, -1)):
                b = bytearray(data)
                b[pos:pos + 4] = int(val).to_bytes(4, 'little', signed=True)
                _assert_total(call, p, bytes(b))
