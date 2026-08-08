"""Shared HTTP layer for the on-demand external-data toolkit.

Stdlib-only (``urllib``) GET with uniform :class:`DataFetchError` wrapping,
so each data source (bathymetry, sound speed, …) parses bytes without
re-implementing network error handling. No third-party HTTP dependency.
"""

import http.client
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Union

import numpy as np

from uacpy.core.exceptions import DataFetchError
from uacpy._log import log_message

__all__ = ['http_get', 'curl_download', 'erddap_griddap_url', 'erddap_last_value',
           'checked_member_size', 'MAX_MEMBER_BYTES']

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
_MAX_BACKOFF_S = 8.0
# Only network schemes — never ``file://`` / ``ftp://`` (which urlopen honours),
# so a user-supplied ``base_url=`` cannot turn into local-file disclosure / SSRF.
_ALLOWED_SCHEMES = ('http', 'https')
# Ceiling on a single response body, so a malicious / misdirected host cannot
# drive an unbounded allocation before the bytes ever reach a parser.
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024   # 512 MiB
# Ceiling on a single uncompressed archive member, so a tar/zip "decompression
# bomb" cannot exhaust memory when an install-time download extracts a member
# into memory. The largest legitimate member is the Diesing raster (~hundreds of
# MB uncompressed); the cap is generous but bounded.
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
            if attempt < _MAX_RETRIES:
                wait = min(_MAX_BACKOFF_S, 1.5 * (2 ** attempt))
                log_message(source, f"transport error ({reason}); retrying in "
                            f"{wait:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})",
                            verbose=verbose, level='warning')
                time.sleep(wait)
                continue
            raise DataFetchError(
                f"Could not reach {url}: {reason} (after {_MAX_RETRIES} retries).",
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
    return data


def curl_download(url: str, out, *, timeout: float, verbose: bool) -> bool:
    """Fetch ``url`` → ``out`` with curl; ``True`` on success.

    ``False`` when curl is absent or fails, so the caller can fall back to
    :func:`http_get`. The large static grid hosts (NCEI/Akamai, Zenodo, GLODAP)
    throttle Python urllib to a trickle but serve curl at full speed, so this is
    the preferred path for them. Downloads to ``<out>.part`` and moves it into
    place only on success, so an interrupted transfer never leaves a truncated
    ``out`` for the cache to accept.
    """
    curl = shutil.which('curl')
    if not curl:
        return False
    part = Path(str(out) + '.part')
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
    """Seconds to wait before a retry, from the ``Retry-After`` header."""
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
    import urllib.parse
    from uacpy.data._time import parse_date
    constraint = (f"{var}[({parse_date(when)}T00:00:00Z)][({level})]"
                  f"[({lat})][({lon % 360.0})]")
    query = urllib.parse.quote(constraint, safe='[]():.,-TZ')
    return f"{base_url}/{dataset}.csv?{query}"
