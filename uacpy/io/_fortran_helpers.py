"""
Low-level Fortran helpers shared by the AT/Bellhop readers.

Two families, both private to ``uacpy.io`` and outside the public surface:

* binary record framing for the ``.shd``/``.mod`` formats — endianness
  detection and record-length-prefixed payload reads;
* list-directed text parsing for the ``.env``/``.flp``/``.bty`` decks —
  :func:`read_vector` and :func:`strip_fortran_comment`.

:func:`typed_format_error` and :data:`PARSE_ERRORS` give both families one
typed failure mode.
"""

import functools
import re
import struct
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from uacpy.core.exceptions import FileFormatError
from uacpy.core._warn_frames import USER_FRAME_SKIP


#: Exceptions a malformed or truncated file can legitimately raise out of a
#: reader's ``int()``/``float()``/``struct``/indexing operations. Deliberately
#: excludes ``AttributeError``/``TypeError``/``NameError``: those signal a uacpy
#: defect, and rewrapping them as :class:`FileFormatError` sends users to debug
#: a file that is fine.
PARSE_ERRORS = (ValueError, IndexError, KeyError, StopIteration, EOFError,
                ZeroDivisionError, OverflowError, struct.error)


def _bound_counts(filepath, file_size, min_item_bytes, **counts):
    """Reject header counts that cannot be satisfied by ``file_size``.

    Binary OASES readers size NumPy allocations directly off integer
    header fields (``n_rcv``, ``n_freq``, grid extents). A corrupt or
    hostile file with a garbage count (e.g. ``n_rcv = 0x7fffffff``) would
    otherwise drive a multi-GB/TB ``np.zeros`` before any data record is
    validated. The smallest a single data item can occupy on disk is
    ``min_item_bytes``, so no count — nor their product — can exceed
    ``file_size // min_item_bytes``. Raise :class:`FileFormatError` on
    overflow rather than attempting the allocation.
    """
    max_items = file_size // max(min_item_bytes, 1)
    product = 1
    for name, val in counts.items():
        if val < 0:
            raise FileFormatError(
                f"{filepath}: negative header count {name}={val}."
            )
        if val > max_items:
            raise FileFormatError(
                f"{filepath}: header count {name}={val} is implausible for "
                f"a {file_size}-byte file (max {max_items} items)."
            )
        product *= val
    if product > max_items:
        raise FileFormatError(
            f"{filepath}: header counts {dict(counts)} imply {product} data "
            f"items, implausible for a {file_size}-byte file "
            f"(max {max_items})."
        )


def list_directed_int(line: str) -> int:
    """First value of a list-directed integer READ.

    A list-directed scalar READ consumes the first separator-delimited token
    (separators: whitespace and commas) and ignores the record's remainder,
    so ``'9999,\t! M'`` (AT tests/sduct/sductK.flp:3) and
    ``'727  NUMBER OF ELEMENTS'`` (tests/3DAtlantic/lant.flp:414) are both
    valid records. ``int()`` on the raw line rejects both.
    """
    tokens = strip_fortran_comment(line).replace(',', ' ').split()
    if not tokens:
        raise FileFormatError(
            f"list-directed integer read on an empty record: {line!r}")
    # A scalar READ takes the first value of an ``r*c`` repeat group too
    # (``'2*5'`` reads as 5); see :func:`expand_repeat_counts`.
    return int(next(expand_repeat_counts(tokens)))


#: Mantissa followed by a *signed* exponent with no ``E``/``D`` letter —
#: ``'0.123457-118'``, ``'1.5+7'``, ``'.5+2'``, ``'1.-3'``. The sign is what
#: makes it unambiguous: without one, ``1 7`` would be two values.
_LETTERLESS_EXPONENT = re.compile(
    r'^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$')


def fortran_float(token) -> float:
    """``float()`` accepting the real spellings a list-directed READ accepts.

    Two forms Python's ``float()`` rejects:

    * **'D' exponents** (``'1.0D+00'``) — list-directed WRITEs emit them for
      double precision and list-directed READs accept them (RefCoef.f90:53,
      bdryMod.f90:195).
    * **Letterless exponents** (``'0.123457-118'`` = 1.23457e-119). A ``Gw.d``
      / ``Ew.d`` WRITE drops the ``E`` when the exponent needs three digits,
      so any value below ~1e-99 comes back in this form. It is a *foreign*
      file that carries one: uacpy's own OALIB build writes ``.rts`` from
      default-``REAL`` arrays (``Scooter/sparc.f90:40``, written
      ``'( 12G15.6 )'`` at ``:294``) and ``install.sh:999`` passes no
      ``-fdefault-real-8``, so single precision caps the exponent at the two
      digits of its ±38 range. The three-digit form therefore comes from a
      double-precision SPARC built elsewhere — or from a local build whose
      ``UACPY_FORTRAN_ARCH_FLAGS`` override widened the default real.
      gfortran's list-directed READ accepts it at any exponent width
      (``'1.5+7'`` reads as 1.5e7), so the sign alone marks the exponent and
      this does too.
    """
    text = str(token)
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return float(text.replace('D', 'E').replace('d', 'e'))
    except ValueError:
        match = _LETTERLESS_EXPONENT.match(text.strip())
        if match is None:
            raise
        return float(f"{match.group(1)}E{match.group(2)}")


#: Ceiling on the values one token stream's ``r*c`` groups may add. Lower
#: than :data:`_MAX_GENERATED_VECTOR`, which bounds a *numeric array*: these
#: are Python strings at ~50 bytes each, and the readers that index their
#: stream hold the whole expansion at once, so a short record must not be
#: able to claim hundreds of megabytes. No engine record repeats a value a
#: million times — a count that large is a corrupt file, not a compressed one.
_MAX_REPEAT_EXPANSION = 1_000_000

#: An unsigned nonzero-leading repeat count, a ``*``, and the constant —
#: the list-directed ``r*c`` form (``'2*0.5'`` = two 0.5s). The constant is
#: kept as text for the caller's own value parse, so ``'2*junk'`` expands
#: and then raises exactly where a bare ``'junk'`` token would.
_REPEAT_COUNT_TOKEN = re.compile(r'^(\d+)\*(.+)$')


def expand_repeat_counts(tokens):
    """Yield list-directed tokens with ``r*c`` repeat counts expanded.

    A list-directed READ accepts ``r*c`` as ``r`` copies of the constant
    ``c``. gfortran's own list-directed WRITEs never emit the form, but
    ifort's do for consecutive equal values — and RefCoef.f90:53 reads the
    ``.brc``, ArrMod.f90:99-118 writes the ``.arr``, list-directed, so a
    file from an ifort-built engine can carry it.

    Tokens the form does not cover pass through unchanged, so the caller's
    own ``int()``/``fortran_float`` raises for them: a zero repeat, a
    non-numeric constant, and Fortran's null-value form ``r*`` (which
    stands for ``r`` values *not* assigned — there is no value to hand a
    caller for it).

    Two ceilings raise :class:`FileFormatError`, both
    :data:`_MAX_REPEAT_EXPANSION`: one repeat count may not exceed it, and
    neither may the running total of copies one stream has added. A repeat
    count breaks the relation between file size and item count that
    :func:`_bound_counts` relies on — as SubTab's generated vectors do —
    and a per-token ceiling alone does not restore it, because a record may
    hold arbitrarily many groups just under the bar. The cumulative ceiling
    is what bounds the memory a caller that materialises the stream can be
    made to spend on a short file.

    The generator itself is lazy on both axes — tokens are consumed one at
    a time and each expansion is yielded copy by copy — so a consumer that
    walks it (``read_arr_file``, ``read_vector``,
    ``read_list_directed_values``) never materialises more than it asks
    for. The readers that index their token stream build a list from it,
    and for those the cumulative ceiling is the bound.
    """
    added = 0
    for tok in tokens:
        match = _REPEAT_COUNT_TOKEN.match(tok)
        if match is None or int(match.group(1)) == 0:
            yield tok
            continue
        count = int(match.group(1))
        added += count - 1
        if count > _MAX_REPEAT_EXPANSION or added > _MAX_REPEAT_EXPANSION:
            raise FileFormatError(
                f"list-directed repeat count {tok!r} asks for {count} "
                f"copies, taking this record's expansion to {added} values "
                f"past the {_MAX_REPEAT_EXPANSION} ceiling on one record's "
                f"repeat groups.",
                remediation="Check the record — a corrupt file can read a "
                            "data value into the repeat count. No real "
                            "engine output repeats a value this often.",
            )
        for _ in range(count):
            yield match.group(2)


def typed_format_error(reader):
    """Decorator: surface a reader's raw parse/truncation exceptions as a typed
    :class:`~uacpy.core.exceptions.FileFormatError`.

    The file-format readers do ``int()``/``float()`` on tokens, ``next()`` on
    line iterators, and index binary records — so a malformed or truncated file
    leaks a bare ``ValueError``/``IndexError``/``StopIteration``/``struct.error``
    instead of the typed, remediated error the rest of ``io`` raises. This
    converts :data:`PARSE_ERRORS` (and only those) while letting an
    already-typed ``FileFormatError``/``ConfigurationError`` pass through
    unchanged.

    An *absent* file is not in that set. Which exception a missing path
    deserves depends on who was supposed to produce it — ``ConfigurationError``
    when the user named it, ``FileFormatError`` when a model should have
    written it (:class:`~uacpy.core.exceptions.FileFormatError` states the
    rule) — and a decorator wrapped around fourteen readers of both kinds
    knows neither. Each reader therefore states its own provenance:
    :func:`require_model_output` for the readers of model output, an explicit
    ``ConfigurationError`` for the readers of user-authored decks.
    """
    @functools.wraps(reader)
    def wrapper(*args, **kwargs):
        try:
            return reader(*args, **kwargs)
        except PARSE_ERRORS as exc:
            target = args[0] if args else '<stream>'
            raise FileFormatError(
                f"{reader.__name__}: could not parse {target} — the file is "
                f"malformed, truncated, or not the expected format "
                f"({type(exc).__name__}: {exc}).",
                remediation="Verify the file was produced by the matching "
                            "model/writer and downloaded completely; a partial "
                            "file or a wrong format triggers this.",
            ) from exc
    return wrapper


def require_model_output(filepath, reader: str) -> None:
    """Raise :class:`FileFormatError` when a model output file is absent.

    The counterpart to the explicit ``ConfigurationError`` the readers of
    user-authored decks raise: this is the provenance statement for the
    readers whose file is one a model should have written, so an absent path
    means the run failed before writing it rather than that the caller named
    the wrong file.
    """
    if not Path(filepath).exists():
        raise FileFormatError(
            f"{reader}: file not found: {filepath}.",
            remediation="Check the path; for a model output, the run "
                        "may have failed before writing this file.",
        )


def strip_fortran_comment(line: str) -> str:
    """Drop a Fortran trailing comment (``! …``) and surrounding whitespace.

    A list-directed ``READ`` ignores the rest of a record once its I/O list is
    satisfied, so AT's example decks and uacpy's writers annotate scalar lines
    (``999999   ! Mlimit``); ``int()``/``float()`` do not. ``!`` is not a
    terminator to the Fortran runtime — reached while values are still
    expected it is a read error; only ``/`` ends a read early
    (see :func:`read_vector`).
    """
    return line.split('!', 1)[0].strip()


def strip_fortran_quotes(line: str) -> str:
    """Return the contents of a Fortran quoted string, or the bare line.

    AT writes titles and option words as ``'…'`` character literals, but its
    list-directed ``READ`` also accepts an unquoted token — so readers must
    handle both. Inside a literal, Fortran escapes an apostrophe by doubling
    it: ``'It''s'`` reads as ``It's``, and the doubled pair does not end the
    literal. Falls back to :func:`strip_fortran_comment` when the line
    carries no quotes or the literal never closes.
    """
    line = line.strip()
    start = line.find("'")
    if start >= 0:
        chars = []
        i = start + 1
        while i < len(line):
            if line[i] == "'":
                if i + 1 < len(line) and line[i + 1] == "'":
                    chars.append("'")
                    i += 2
                    continue
                return ''.join(chars)
            chars.append(line[i])
            i += 1
    return strip_fortran_comment(line)


#: Value separators a list-directed READ recognises between items: blanks,
#: tabs and commas. (A ``/`` terminates the whole list rather than one item
#: and is handled by the callers that can see it — see
#: :func:`read_list_directed_values`.)
_FORTRAN_VALUE_SEPARATORS = " \t\r\n,"


def split_fortran_tokens(line: str) -> list:
    """Split one list-directed record into items, keeping literals whole.

    ``str.split`` breaks ``'my modes'`` into two items, which silently
    truncates a CHARACTER value to its first word. A list-directed READ
    takes the whole delimited literal as one value — ``field3d.f90:192``
    reads a mode-file name that way, ``READ x( I ), y( I ),
    ModeFileName( I )`` — with a doubled delimiter standing for one
    embedded delimiter.

    Items keep their delimiters, so a caller can still tell a quoted
    literal from a bare word and numeric items are handed on unchanged to
    :func:`expand_repeat_counts` and :func:`fortran_float`. Use
    :func:`strip_fortran_quotes` on an item to get its undelimited text.
    An unterminated literal runs to the end of the record, which is where
    a Fortran runtime would also stop looking within one record.
    """
    tokens: list = []
    i, n = 0, len(line)
    while i < n:
        if line[i] in _FORTRAN_VALUE_SEPARATORS:
            i += 1
            continue
        if line[i] in "'\"":
            quote = line[i]
            j = i + 1
            while j < n:
                if line[j] == quote:
                    if j + 1 < n and line[j + 1] == quote:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            tokens.append(line[i:j])
            i = j
            continue
        j = i
        while j < n and line[j] not in _FORTRAN_VALUE_SEPARATORS:
            j += 1
        tokens.append(line[i:j])
        i = j
    return tokens


def take_tokens(tokens, cursor: int, n: int, what: str, source) -> Tuple[list, int]:
    """Consume ``n`` tokens of a flat token stream starting at ``cursor``.

    The reader for a free ``fscanf``-style token stream (MATLAB's
    ``read_ts.m``, the ``.rts`` payload): line breaks carry no meaning at
    all, so the caller tokenises the whole file once and this walks it.
    Returns ``(tokens[cursor:cursor + n], cursor + n)`` and raises
    :class:`FileFormatError` when the stream is too short.
    """
    end = cursor + n
    if end > len(tokens):
        raise FileFormatError(
            f"{source}: token stream ended while reading {what} — needed "
            f"{n} values, found {len(tokens) - cursor}.",
            remediation="The file is truncated or not the expected format; "
                        "verify it was written completely.",
        )
    return tokens[cursor:end], end


def read_list_directed_values(fid, n: int, what: str, source,
                              first_line: Optional[str] = None) -> np.ndarray:
    """Consume the ``n`` values of one list-directed ``READ(unit, *) x(1:n)``.

    A whole-vector list-directed READ keeps consuming records until ``n``
    values have been read, then skips the remainder of the final record —
    the next READ starts on a fresh record. Values may therefore wrap
    across any number of lines (``Bellhop/sspMod.f90:417,428``,
    ``misc/RefCoef.f90:53``), and several may share one line. Separators
    are whitespace and commas; ``D`` exponents are accepted; a trailing
    ``! …`` comment ends a record's values.

    Two deliberate departures from the Fortran runtime, both recorded here
    because each has been re-filed as a defect once:

    * **A trailing ``/`` is accepted, an early one is not.** Fortran's ``/``
      terminates the whole READ, leaving every item not yet assigned
      *undefined*. A record like ``0.0 100.0 /`` read with ``n = 2`` costs
      nothing: the loop stops the moment it holds ``n`` values, so the
      ``/`` is never converted — the ``.bty``/``.ati`` rows the AT decks
      write in exactly that form read back fine. The ``/`` only reaches
      :func:`fortran_float` when it arrives *before* the list is
      satisfied, and there the Fortran would hand the caller undefined
      values, so raising is the safer answer to a file uacpy cannot
      reproduce the semantics of. (It raises out of ``fortran_float`` as a
      ``ValueError``; every reader that calls this is wrapped in
      :func:`typed_format_error`, which retypes it.)
    * **``!`` leniency is a superset of gfortran.** ``!`` is a comment only
      in free-form *source*; to a list-directed READ it is just another
      token and would be a read error. Dropping it lets uacpy read the
      annotated decks its own writers emit. Nothing a real engine writes
      is rejected by the extra tolerance — but a record whose values are
      truncated by a ``!`` still fails, as the count is short (see
      :func:`strip_fortran_comment`).

    ``what`` and ``source`` name the record and file for the
    :class:`FileFormatError` raised when the file ends short of ``n``.
    ``first_line`` is a record the caller already read from ``fid`` (to
    detect EOF or a count); it is consumed before any further ``readline``.
    """
    values: list = []
    pending = first_line
    while len(values) < n:
        if pending is not None:
            line, pending = pending, None
        else:
            line = fid.readline()
        if line == '':
            raise FileFormatError(
                f"{source}: file ended while reading {what} — expected "
                f"{n} values, found {len(values)}.",
                remediation="The file is truncated or not the expected "
                            "format; verify it was written completely.",
            )
        for tok in expand_repeat_counts(
                strip_fortran_comment(line).replace(',', ' ').split()):
            if len(values) >= n:
                break
            values.append(fortran_float(tok))
    return np.array(values, dtype=float)


_ENDIAN_WARN_EMITTED = False


def _warn_non_little_endian(detected: str, source: str) -> None:
    """Log a one-shot notice the first time we decode a non-little-endian
    Fortran file. uacpy CI runs little-endian; big-endian decode works but
    is unvalidated."""
    global _ENDIAN_WARN_EMITTED
    if detected == 'big' and not _ENDIAN_WARN_EMITTED:
        warnings.warn(
            f"{source}: detected big-endian Fortran record framing; uacpy "
            "decodes it correctly but this byte order is not validated by CI.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        _ENDIAN_WARN_EMITTED = True


def detect_endian(first4: bytes, source: str = '_fortran_helpers') -> str:
    """Detect Fortran-record byte order from the first 4 bytes of a file.

    The Fortran framing puts a 4-byte record-length prefix at the head of
    every record. On a well-formed file that integer is much smaller than
    ``2**31`` in the correct endianness and absurdly large in the wrong
    one: reading a marker byte-reversed multiplies a short length by
    ``2**24``, so the smaller of the two candidates is the right one. The
    ``2**28`` cap (256 MiB) sits far above any record these formats write,
    and a one-shot notice is logged when the choice isn't little-endian.

    Returns ``'<'`` (little-endian) or ``'>'`` (big-endian).
    """
    if len(first4) < 4:
        raise FileFormatError("detect_endian: need 4 bytes to probe.")
    little = struct.unpack('<i', first4)[0]
    big = struct.unpack('>i', first4)[0]
    cap = 1 << 28
    little_ok = 0 < little < cap
    big_ok = 0 < big < cap
    if little_ok and not big_ok:
        chosen = '<'
    elif big_ok and not little_ok:
        chosen = '>'
    elif little_ok and big_ok:
        chosen = '<' if little <= big else '>'
    else:
        raise FileFormatError(
            f"detect_endian: cannot resolve byte order from first record "
            f"marker (little={little}, big={big}); file is probably corrupt."
        )
    _warn_non_little_endian('big' if chosen == '>' else 'little', source)
    return chosen


def read_fortran_record(f, fmt=None, raw=False, endian='<'):
    """Read a single Fortran UNFORMATTED sequential record.

    Layout::

        [4-byte length N][N bytes payload][4-byte length N]

    Both length markers must match; mismatch indicates file corruption or
    wrong endianness and raises :class:`~uacpy.core.exceptions.FileFormatError`.
    The markers are signed int32 of the compiler's default width — a file
    written with 8-byte record markers does not parse here — and the
    ``2**28``-byte cap rejects the inflated length a wrong-endianness read
    produces before it reaches ``f.read`` (see :func:`detect_endian`). A
    payload shorter than its own marker announces is a truncated file, and
    also raises.

    Parameters
    ----------
    f : file object (binary mode)
    fmt : str, optional
        struct format string for the payload (excluding endian prefix).
    raw : bool, optional
        If True, return raw bytes. Default False.
    endian : str, optional
        '<' (little-endian, x86 default) or '>' (big-endian).

    Returns
    -------
    tuple | bytes
        Unpacked payload (or raw bytes).
    """
    head = f.read(4)
    if len(head) < 4:
        raise FileFormatError(
            f"Unexpected EOF reading Fortran record head: "
            f"{getattr(f, 'name', '<stream>')} ended at byte {f.tell()} with "
            f"{len(head)} of the 4 marker bytes available. The file is "
            f"truncated: re-run the model and check it exited 0.")
    (nbytes,) = struct.unpack(endian + 'i', head)
    if nbytes < 0 or nbytes > (1 << 28):
        raise FileFormatError(
            f"Unreasonable Fortran record length: {nbytes} (wrong endianness?)"
        )
    payload = f.read(nbytes)
    if len(payload) < nbytes:
        raise FileFormatError(
            f"Short read: expected {nbytes} bytes, got {len(payload)}"
        )
    tail = f.read(4)
    if len(tail) < 4:
        raise FileFormatError(
            f"Unexpected EOF reading Fortran record tail: "
            f"{getattr(f, 'name', '<stream>')} ended at byte {f.tell()} with "
            f"{len(tail)} of the 4 marker bytes available, after a complete "
            f"{nbytes}-byte payload. The file is truncated: re-run the model "
            f"and check it exited 0.")
    (ntail,) = struct.unpack(endian + 'i', tail)
    if ntail != nbytes:
        raise FileFormatError(
            f"Fortran record marker mismatch: head={nbytes} tail={ntail} "
            "(wrong endianness or truncated file)"
        )
    if raw or fmt is None:
        return payload
    expected = struct.calcsize(endian + fmt)
    if expected != nbytes:
        raise FileFormatError(
            f"Fortran record payload {nbytes} != fmt '{fmt}' size {expected}"
        )
    return struct.unpack(endian + fmt, payload)


def _read_vector_values(fid, Nx: int) -> Tuple[np.ndarray, bool]:
    """Collect the value tokens of one AT ``ReadVector`` record.

    ``READ( ENVFile, * ) x( 1 : Nx )`` is list-directed: it keeps consuming
    records until ``Nx`` values have been read, or until a ``/`` terminates the
    read early. Returns the values collected and whether a ``/`` ended them.
    """
    def _take(dest, text):
        # Commas are value separators to a list-directed READ, and tokens
        # past the Nx-th are the record remainder the READ never looks at.
        for tok in expand_repeat_counts(text.replace(',', ' ').split()):
            if len(dest) >= Nx:
                break
            dest.append(fortran_float(tok))

    values = []
    while len(values) < Nx:
        line = fid.readline()
        if line == '':
            break
        line = strip_fortran_comment(line)
        if '/' in line:
            _take(values, line.split('/', 1)[0])
            return np.array(values, dtype=float), True
        _take(values, line)
    return np.array(values, dtype=float), False


#: Ceiling on a *generated* vector length (``80`` MB as float64).
#:
#: The generated branches are the only place a small record can demand a large
#: array: ``5`` then ``0 1000 /`` is five values, and the same two records with
#: ``999999999`` are a billion — so :func:`_bound_counts`' file-size test is the
#: wrong instrument here, since the shorthand exists precisely to break the
#: relation between file size and item count. AT itself just allocates and
#: reports failure (``ALLOCATE( x( MAX( 3, Nx ) ), Stat = IAllocStat )`` /
#: ``ERROUT( 'ReadVector', 'Too many ' … )``,
#: ``misc/SourceReceiverPositions.f90:215-216``), which on a large-memory
#: machine means a corrupt count succeeds in taking 8 GB before anything
#: notices. The largest generated vector in any deck the toolbox itself ships
#: is **10001** (``tests/Noise/ATOC/aet_VLA_C.flp:6``, ``10001`` then
#: ``0.0 500.0 /``), so this sits three orders of magnitude above real usage
#: while still bounding a hostile or truncated file.
_MAX_GENERATED_VECTOR = 10_000_000


def _generate(build, Nx: int) -> np.ndarray:
    """Run one of SubTab's generator branches under :data:`_MAX_GENERATED_VECTOR`.

    ``MemoryError`` is caught as well: it is not in :data:`PARSE_ERRORS` (it
    says nothing about the file in general), so without this a deck that is
    merely too large for *this* machine surfaces raw instead of typed.
    """
    if Nx > _MAX_GENERATED_VECTOR:
        raise FileFormatError(
            f"read_vector: the record asks for {Nx} generated values "
            f"({8 * Nx / 1e9:.2f} GB as float64), past the "
            f"{_MAX_GENERATED_VECTOR} ceiling on a generated vector.",
            remediation="Check the count line above the vector — a corrupt or "
                        "truncated deck reads a data value as the count. The "
                        "largest generated vector in any deck the "
                        "Acoustics Toolbox ships is 10001 values.",
        )
    try:
        return build()
    except MemoryError as exc:
        raise FileFormatError(
            f"read_vector: the record asks for {Nx} generated values "
            f"({8 * Nx / 1e9:.2f} GB as float64), which does not fit in "
            f"memory.",
            remediation="Check the count line above the vector — a corrupt or "
                        "truncated deck reads a data value as the count. AT "
                        "reports this as ERROUT('ReadVector', 'Too many …').",
        ) from exc


def read_vector(fid) -> Tuple[np.ndarray, int]:
    """
    Read a vector from an Acoustics Toolbox / Bellhop input file.

    Mirrors AT's ``ReadVector`` (``misc/SourceReceiverPositions.f90:221``)
    followed by ``SubTab`` (``misc/subtabulate.f90``): a list-directed
    ``READ`` of ``Nx`` values that continues across records until ``Nx`` are
    consumed, with ``/`` terminating early to trigger a generated vector.
    AT then calls ``Sort`` on the result
    (``SourceReceiverPositions.f90:224,268``), so the solver always computes
    on an ascending axis whatever order the deck listed. This function does
    **not** sort — callers that mirror an AT axis apply ``np.sort``
    themselves (see ``oalib_reader``'s position readers).
    ``ReadVector`` pre-fills ``x(2)`` and ``x(3)`` with ``-999.9``
    (``SourceReceiverPositions.f90:219-220``) and ``SubTab`` generates only
    where those sentinels survive the read — never for ``Nx < 3``, and never
    once three values have been given (``subtabulate.f90:3-5,24-28``), which
    is what separates the branches below.

    Parameters
    ----------
    fid : file object
        Open file handle positioned at vector specification

    Returns
    -------
    x : ndarray
        Vector of values
    Nx : int
        Number of values

    Notes
    -----
    Format 1 (linear spacing) — two values then ``/``::

        5
        0 1000 /

    Creates: [0, 250, 500, 750, 1000]

    Format 2 (explicit values) — ``Nx`` values, on one record or wrapped
    across several::

        5
        0 100 300
        700 1000

    Creates: [0, 100, 300, 700, 1000]

    Format 3 (replicate) — one value then ``/``::

        501
        0.0 /

    Creates: [0, 0, ..., 0] (501 zeros). ``SubTab`` reaches this through
    ``x(2) == x(1)``, so the generated spacing is zero.

    Examples
    --------
    The AT shorthand ``N`` then ``first last /`` expands to ``N`` evenly
    spaced values. Written to a temporary directory so running the example
    leaves nothing behind:

    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = os.path.join(d, 'vec.txt')
    ...     with open(path, 'w') as fh:
    ...         _ = fh.write('5\\n0 1000 /\\n')
    ...     with open(path) as fh:
    ...         x, Nx = read_vector(fh)
    >>> print(x)
    [   0.  250.  500.  750. 1000.]
    >>> Nx
    5
    """
    Nx = list_directed_int(fid.readline())
    if Nx <= 0:
        return np.array([]), Nx

    values, slash_terminated = _read_vector_values(fid, Nx)

    # Every call site hands the result to a caller as a geometry axis
    # (oalib_reader's read_flp position/offset vectors), and axis coordinates
    # must be finite — the same contract bathy_io enforces on its axes. A
    # non-finite value here poisons all three branches below: explicit values
    # return it as-is, and both SubTab generator branches spread it across
    # the whole vector (np.linspace(0, inf, N) is [0, inf, …, nan]). Checking
    # the parsed values, not the returned x, catches all three the same way.
    n_non_finite = int(values.size - np.count_nonzero(np.isfinite(values)))
    if n_non_finite:
        raise FileFormatError(
            f"read_vector: {n_non_finite} of {values.size} parsed values are "
            f"not finite (NaN or infinite); vector values must be finite "
            f"numbers.",
            remediation="Look for 'NaN'/'Infinity' tokens in the record, or "
                        "an exponent outside float64 range (a list-directed "
                        "READ parses 1e999 as Infinity, and gfortran does "
                        "the same).",
        )

    if len(values) == Nx:
        x = values
    elif not slash_terminated:
        # No '/' and the file ran out before Nx values: the record is truncated.
        raise FileFormatError(
            f"read_vector: expected {Nx} values but the file ended after "
            f"{len(values)}.",
            remediation="Verify the file was written completely by the "
                        "matching model/writer.",
        )
    elif Nx >= 3 and len(values) == 2:
        # SubTab's two-value branch: equally spaced from x(1) to x(2).
        x = _generate(lambda: np.linspace(values[0], values[1], Nx), Nx)
    elif Nx >= 3 and len(values) == 1:
        # SubTab's replicate branch (x(2) defaults to x(1) ⇒ zero spacing).
        x = _generate(lambda: np.full(Nx, values[0]), Nx)
    else:
        # A '/' short of Nx values that SubTab cannot generate from. SubTab
        # only generates for Nx >= 3 with 1 or 2 values given
        # (subtabulate.f90:3-5,24-28); every other slash-terminated shortfall
        # leaves AT slots at ReadVector's -999.9 pre-fill (x(2)/x(3),
        # SourceReceiverPositions.f90:219-220) or uninitialised (x(1),
        # x(4:Nx)), so there is no correct reading to recover.
        raise FileFormatError(
            f"read_vector: Nx={Nx} record was terminated by '/' after "
            f"{len(values)} values; only a 1-value (replicate) or 2-value "
            f"(equally spaced) record generates a vector, and only for "
            f"Nx >= 3.",
            remediation=f"Give all {Nx} values, or (for Nx >= 3) give 1 or 2 "
                        f"values before the '/'.",
        )

    return np.atleast_1d(x), Nx
