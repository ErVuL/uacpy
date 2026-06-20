"""Shared HTTP layer for the on-demand external-data toolkit.

Stdlib-only (``urllib``) GET with uniform :class:`DataFetchError` wrapping,
so each data source (bathymetry, sound speed, …) parses bytes without
re-implementing network error handling. No third-party HTTP dependency.
"""

import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Union

from uacpy.core.exceptions import DataFetchError
from uacpy._log import log_message

__all__ = ['http_get']

# HTTP codes worth retrying (transient rate-limit / availability).
_RETRY_CODES = (429, 503)
_MAX_RETRIES = 2
_MAX_BACKOFF_S = 8.0
# Only network schemes — never ``file://`` / ``ftp://`` (which urlopen honours),
# so a user-supplied ``base_url=`` cannot turn into local-file disclosure / SSRF.
_ALLOWED_SCHEMES = ('http', 'https')
# Ceiling on a single response body, so a malicious / misdirected host cannot
# drive an unbounded allocation before the bytes ever reach a parser.
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024   # 512 MiB


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

    Retries a bounded number of times on transient rate-limit / availability
    responses (HTTP 429 / 503), honouring a ``Retry-After`` header when present
    — so public hosts (e.g. the OpenTopoData ≤1 req/s limit) are handled
    politely rather than failing the whole fetch.

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
        except urllib.error.URLError as exc:
            raise DataFetchError(
                f"Could not reach {url}: {exc.reason}.",
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


def _retry_after(exc: urllib.error.HTTPError) -> float:
    """Seconds to wait before a retry, from the ``Retry-After`` header."""
    header = exc.headers.get('Retry-After') if exc.headers else None
    try:
        wait = float(header)
    except (TypeError, ValueError):
        wait = 1.5
    return min(max(wait, 1.0), _MAX_BACKOFF_S)
