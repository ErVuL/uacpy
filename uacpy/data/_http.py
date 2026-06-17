"""Shared HTTP layer for the on-demand external-data toolkit.

Stdlib-only (``urllib``) GET with uniform :class:`DataFetchError` wrapping,
so each data source (bathymetry, sound speed, …) parses bytes without
re-implementing network error handling. No third-party HTTP dependency.
"""

import time
import urllib.error
import urllib.request
from typing import Union

from uacpy.core.exceptions import DataFetchError
from uacpy._log import log_message

__all__ = ['http_get']

# HTTP codes worth retrying (transient rate-limit / availability).
_RETRY_CODES = (429, 503)
_MAX_RETRIES = 2
_MAX_BACKOFF_S = 8.0


def http_get(
    url: str,
    *,
    timeout: float = 30.0,
    verbose: Union[bool, str] = False,
    source: str = 'data',
    user_agent: str = 'uacpy',
) -> bytes:
    """GET ``url`` and return the raw response body.

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
    request = urllib.request.Request(url, headers={'User-Agent': user_agent})
    for attempt in range(_MAX_RETRIES + 1):
        log_message(source, f"GET {url}", verbose=verbose, level='debug')
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
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


def _retry_after(exc: urllib.error.HTTPError) -> float:
    """Seconds to wait before a retry, from the ``Retry-After`` header."""
    header = exc.headers.get('Retry-After') if exc.headers else None
    try:
        wait = float(header)
    except (TypeError, ValueError):
        wait = 1.5
    return min(max(wait, 1.0), _MAX_BACKOFF_S)
