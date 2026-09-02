"""Shared HTTP layer for the on-demand external-data toolkit.

Stdlib-only (``urllib``) GET with uniform :class:`DataFetchError` wrapping,
so each data source (bathymetry, sound speed, …) parses bytes without
re-implementing network error handling. No third-party HTTP dependency.
"""

import errno
import http.client
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Union

import numpy as np

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._cache import staging_path
from uacpy._log import log_message

__all__ = ['http_get', 'curl_download', 'erddap_griddap_url', 'erddap_last_value',
           'checked_member_size', 'raise_substantive', 'MAX_MEMBER_BYTES']


def raise_substantive(errors):
    """Raise the most substantive error collected by a source-fallback chain.

    A :class:`DataFetchError` (no coverage / on land / live service failure)
    is raised in preference to a bare ``ConfigurationError`` ("cache not
    installed", missing prerequisite), so the caller sees the real cause
    rather than the last fallback's complaint. Ties keep the first
    ``DataFetchError``, else the last error.

    An empty ``errors`` means the chain ran no source at all — an empty source
    list, not a fetch failure — so it raises :class:`ConfigurationError`
    rather than an ``IndexError`` off the end of the list.
    """
    if not errors:
        raise ConfigurationError(
            "No data source was tried, so no fetch error explains the "
            "failure.",
            remediation="Pass at least one source: an empty sequence "
                        "(bottom_sources=(), source=(), …) selects none.",
        )
    data_errs = [e for e in errors if isinstance(e, DataFetchError)]
    raise (data_errs[0] if data_errs else errors[-1])

# HTTP status codes worth retrying (transient rate-limit / availability /
# gateway hiccups — the urllib3/requests default transient set).
_RETRY_CODES = (429, 500, 502, 503, 504)
# Transport-level flakes worth retrying with backoff: connection reset, DNS/
# socket timeout, a truncated body, a remote disconnect — the Python-side
# equivalent of the HTTP/2 stream cancels that bite long curl downloads.
_TRANSIENT_EXC = (
    urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout,
    http.client.IncompleteRead, http.client.RemoteDisconnected,
)
_MAX_RETRIES = 4
#: A refused TCP connection is the remote answering instantly with a
#: rejection, so the exponential ladder buys nothing over one quick second
#: attempt — and against a down host every sleep multiplies across every
#: grid of a multi-file build.
_REFUSED_RETRIES = 1
_REFUSED_WAIT_S = 1.0


def _is_permanent_dns_failure(exc) -> bool:
    """True when the failure chain ends in a resolver saying the name does
    not exist (``EAI_NONAME``) or failed for good (``EAI_FAIL``) — a typo in
    ``base_url`` or an offline host, which no retry ladder cures. A
    temporary resolver failure (``EAI_AGAIN``) stays transient."""
    seen = set()
    permanent = {getattr(socket, 'EAI_NONAME', -2), getattr(socket, 'EAI_FAIL', -4)}
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, socket.gaierror) and exc.errno in permanent:
            return True
        exc = getattr(exc, 'reason', None) or exc.__cause__
    return False


def _is_connection_refused(exc) -> bool:
    """True when the transport failure chain ends in ECONNREFUSED."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ConnectionRefusedError) or \
                getattr(exc, 'errno', None) == errno.ECONNREFUSED:
            return True
        exc = getattr(exc, 'reason', None) or exc.__cause__
    return False
_MAX_BACKOFF_S = 8.0
# Only network schemes — never ``file://`` / ``ftp://`` (which urlopen honours),
# so a user-supplied ``base_url=`` cannot turn into local-file disclosure / SSRF.
_ALLOWED_SCHEMES = ('http', 'https')
# Ceiling on a single response body, so a malicious / misdirected host cannot
# drive an unbounded allocation before the bytes ever reach a parser.
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024   # 512 MiB
# Ceiling on a single uncompressed archive member, so a tar/zip "decompression
# bomb" cannot exhaust memory when an install-time download extracts a member
# into memory. The largest legitimate member is the GLODAP pH grid (~100 MB
# uncompressed); the cap is generous but bounded.
MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024   # 2 GiB


def http_get(
    url: str,
    *,
    timeout: float = 30.0,
    verbose: Union[bool, str] = False,
    source: str = 'data',
    user_agent: str = 'uacpy',
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> bytes:
    """GET ``url`` and return the raw response body.

    Only ``http``/``https`` URLs are accepted (a non-network scheme such as
    ``file://`` raises rather than disclosing a local file), and the body is
    capped at ``max_bytes`` so a hostile or misdirected host cannot drive an
    unbounded allocation before the bytes reach a parser.

    Retries a bounded number of times (``_MAX_RETRIES``) on **both** transient
    HTTP responses (429 / 500 / 502 / 503 / 504 — honouring a ``Retry-After``
    header) and transient **transport** failures (connection reset, socket
    timeout, truncated body, remote disconnect — with exponential backoff). So a
    public host's rate limit (e.g. OpenTopoData ≤1 req/s) or a mid-stream network
    hiccup is ridden out rather than failing the whole fetch. Permanent failures
    (4xx other than 429, an over-size body, a blocked scheme) are not retried.

    Parameters
    ----------
    url : str
        Fully-formed request URL (caller is responsible for encoding).
    timeout : float, optional
        Network timeout in seconds.
    verbose : bool or str, optional
        Logging gate forwarded to ``log_message``.
    source : str, optional
        Short tag used in log lines.

    Raises
    ------
    DataFetchError
        On any HTTP or transport-level failure (after retries are exhausted).
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise DataFetchError(
            f"Refusing to fetch {url!r}: only http/https are allowed "
            f"(got scheme {scheme or '<none>'!r}).",
            remediation="Pass an http(s) base_url=; file://, ftp:// and other "
                        "schemes are blocked to avoid local-file disclosure.",
        )
    request = urllib.request.Request(url, headers={'User-Agent': user_agent})
    for attempt in range(_MAX_RETRIES + 1):
        log_message(source, f"GET {url}", verbose=verbose, level='debug')
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return _read_capped(response, url, max_bytes)
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRY_CODES and attempt < _MAX_RETRIES:
                wait = _retry_after(exc)
                log_message(source, f"HTTP {exc.code}; retrying in {wait:.1f}s "
                            f"(attempt {attempt + 1}/{_MAX_RETRIES})",
                            verbose=verbose, level='warning')
                time.sleep(wait)
                continue
            raise DataFetchError(
                f"Request to {url} failed: HTTP {exc.code} {exc.reason}.",
                remediation="Check the dataset/coordinate request. Public hosts "
                            "may be rate-limited — retry later or point at a "
                            "self-hosted instance via base_url=.",
            ) from exc
        except _TRANSIENT_EXC as exc:
            # Connection reset / timeout / truncated body / remote disconnect —
            # transient transport flakes (the Python-side analogue of an HTTP/2
            # stream cancel). Retry with exponential backoff before giving up.
            reason = getattr(exc, 'reason', None) or exc
            if _is_permanent_dns_failure(exc):
                raise DataFetchError(
                    f"Could not resolve the host of {url}: {reason}. The name "
                    f"does not exist to this machine's resolver, so retrying "
                    f"cannot help.",
                    remediation="Check base_url= for a typo, or the network "
                                "(offline hosts resolve nothing).",
                ) from exc
            refused = _is_connection_refused(exc)
            retries = _REFUSED_RETRIES if refused else _MAX_RETRIES
            if attempt < retries:
                wait = (_REFUSED_WAIT_S if refused
                        else min(_MAX_BACKOFF_S, 1.5 * (2 ** attempt)))
                log_message(source, f"transport error ({reason}); retrying in "
                            f"{wait:.1f}s (attempt {attempt + 1}/{retries})",
                            verbose=verbose, level='warning')
                time.sleep(wait)
                continue
            raise DataFetchError(
                f"Could not reach {url}: {reason} (after {retries} retries).",
                remediation="Check network connectivity, or pass base_url= for "
                            "a reachable service instance.",
            ) from exc


def _read_capped(response, url: str, max_bytes: int) -> bytes:
    """Read the response body, refusing anything larger than ``max_bytes``."""
    headers = getattr(response, 'headers', None)
    clen = headers.get('Content-Length') if headers else None
    try:
        if clen is not None and int(clen) > max_bytes:
            raise DataFetchError(
                f"Response from {url} is {int(clen)} bytes, over the "
                f"{max_bytes}-byte cap.",
                remediation="Raise max_bytes= if this is expected, narrow the "
                            "request, or point base_url= at a mirror.",
            )
    except ValueError:
        pass                                    # unparseable Content-Length
    # Read one byte past the cap so an over-size body is detected without
    # buffering the whole thing.
    data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise DataFetchError(
            f"Response from {url} exceeds the {max_bytes}-byte cap.",
            remediation="Raise max_bytes= if this is expected, narrow the "
                        "request, or point base_url= at a mirror.",
        )
    # http.client returns a short body on a mid-transfer disconnect without
    # raising ("might break compatibility"); surface it as the stdlib's own
    # IncompleteRead, which _TRANSIENT_EXC already lists, so the retry loop
    # re-fetches instead of a truncated file being cached as complete.
    try:
        expected = int(clen) if clen is not None else None
    except ValueError:
        expected = None
    if expected is not None and len(data) < expected:
        raise http.client.IncompleteRead(data, expected - len(data))
    return data


def curl_download(url: str, out, *, timeout: float, verbose: bool) -> bool:
    """Fetch ``url`` → ``out`` with curl; ``True`` on success.

    ``False`` when curl is absent or fails, so the caller can fall back to
    :func:`http_get`. The large static grid hosts (NCEI/Akamai, Zenodo, GLODAP)
    throttle Python urllib to a trickle but serve curl at full speed, so this is
    the preferred path for them. Downloads to a staging sibling and moves it
    into place only on success, so an interrupted transfer never leaves a
    truncated ``out`` for the cache to accept.

    The staging file is :func:`uacpy.data._cache.staging_path`'s, so the name
    is unguessable: a predictable name such as ``<out>.part`` lets a symlink be
    pre-placed there, and ``curl -o`` then writes through it before
    ``os.replace`` moves the link itself onto ``out``. It shares that helper
    rather than the whole ``atomic_write`` context manager because a curl
    failure is a ``False`` return, not an exception, and so must drop the
    staging file on a path that context manager treats as success.
    """
    curl = shutil.which('curl')
    if not curl:
        return False
    part = staging_path(out)
    try:
        subprocess.run(
            [curl, '-fL', '--retry', '3', '--max-time', str(int(timeout)),
             '-o', str(part), url],
            check=True, capture_output=not verbose)
    except (subprocess.SubprocessError, OSError):
        part.unlink(missing_ok=True)
        return False
    if not (part.exists() and part.stat().st_size > 0):
        part.unlink(missing_ok=True)
        return False
    os.replace(part, out)
    return True


def erddap_last_value(body: str) -> float:
    """Last numeric field of an ERDDAP griddap ``.csv`` point response.

    The body is a header row, a units row, then one data row whose final column
    is the requested variable. Returns ``NaN`` when the value is missing or the
    service returned no data row.
    """
    rows = [ln for ln in body.splitlines() if ln.strip()]
    if len(rows) < 3:
        return np.nan
    try:
        return float(rows[-1].split(',')[-1])
    except (ValueError, IndexError):
        return np.nan


def checked_member_size(
    declared_size: int, name: str, *, max_bytes: int = MAX_MEMBER_BYTES,
) -> int:
    """Validate an archive member's declared uncompressed size against a cap.

    Tar/zip headers carry the uncompressed size, so a decompression bomb can be
    rejected *before* it is read into memory. Returns the size on success;
    raises :class:`DataFetchError` if it exceeds ``max_bytes`` or is negative.
    """
    if declared_size is None or declared_size < 0 or declared_size > max_bytes:
        raise DataFetchError(
            f"Archive member {name!r} declares {declared_size} uncompressed "
            f"bytes, over the {max_bytes}-byte cap (possible decompression bomb).",
            remediation="The upstream archive may be corrupt or hostile; verify "
                        "the download source, or raise the cap if it is trusted.",
        )
    return declared_size


def _retry_after(exc: urllib.error.HTTPError) -> float:
    """Seconds to wait before a retry, from the ``Retry-After`` header.

    A missing or non-numeric header (it may also be an HTTP-date) falls back to
    1.5 s. The result is clamped into ``[1, _MAX_BACKOFF_S]``: the floor keeps a
    ``Retry-After: 0`` from becoming a hot retry loop, and the ceiling keeps a
    host's multi-hour cool-off from hanging the fetch.
    """
    header = exc.headers.get('Retry-After') if exc.headers else None
    try:
        wait = float(header)
    except (TypeError, ValueError):
        wait = 1.5
    return min(max(wait, 1.0), _MAX_BACKOFF_S)


def erddap_griddap_url(base_url: str, dataset: str, var: str, when,
                       lat: float, lon: float, *, level: float) -> str:
    """ERDDAP griddap CSV URL for ``var`` at the nearest time/level/lat/lon cell.

    Both served grids uacpy reads carry a singleton vertical axis between time
    and latitude — NBS winds at 10 m, WW3 at the 0 m surface — so ``level``
    names that node. The ``[(...)]`` value selectors snap each axis to its
    nearest node, and the longitude axis is [0, 360).
    """
    from uacpy.data._time import parse_date
    constraint = (f"{var}[({parse_date(when)}T00:00:00Z)][({level})]"
                  f"[({lat})][({lon % 360.0})]")
    query = urllib.parse.quote(constraint, safe='[]():.,-TZ')
    return f"{base_url}/{dataset}.csv?{query}"
