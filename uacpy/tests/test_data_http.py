"""Tests for the shared HTTP layer retry/backoff (uacpy.data._http)."""

import urllib.error

import pytest

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import _http
from uacpy.data._http import raise_substantive


class _FakeResp:
    headers = {}                      # mirror a real urllib response

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, amt=-1):           # real responses accept a byte count
        return b'OK'


def test_retries_on_429_then_succeeds(monkeypatch):
    calls = {'n': 0}

    def fake_urlopen(req, timeout=None):
        calls['n'] += 1
        if calls['n'] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, 'Too Many Requests',
                                         {'Retry-After': '1'}, None)
        return _FakeResp()

    monkeypatch.setattr(_http.urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setattr(_http.time, 'sleep', lambda s: None)   # no real wait
    assert _http.http_get('http://x') == b'OK'
    assert calls['n'] == 2                                     # retried once


def test_raises_after_retries_exhausted(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, 'Too Many', {}, None)

    monkeypatch.setattr(_http.urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setattr(_http.time, 'sleep', lambda s: None)
    with pytest.raises(DataFetchError, match='429'):
        _http.http_get('http://x')


def test_non_retry_code_fails_fast(monkeypatch):
    calls = {'n': 0}

    def fake_urlopen(req, timeout=None):
        calls['n'] += 1
        raise urllib.error.HTTPError(req.full_url, 400, 'Bad Request', {}, None)

    monkeypatch.setattr(_http.urllib.request, 'urlopen', fake_urlopen)
    with pytest.raises(DataFetchError, match='400'):
        _http.http_get('http://x')
    assert calls['n'] == 1                                     # no retry on 400


def test_retry_after_parsing():
    err = urllib.error.HTTPError('http://x', 429, 'x', {'Retry-After': '3'}, None)
    assert _http._retry_after(err) == 3.0
    err2 = urllib.error.HTTPError('http://x', 429, 'x', {}, None)
    assert _http._retry_after(err2) == 1.5                     # default
    err3 = urllib.error.HTTPError('http://x', 429, 'x', {'Retry-After': '999'}, None)
    assert _http._retry_after(err3) == _http._MAX_BACKOFF_S    # capped


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'ftp://h/x', '/etc/passwd'])
def test_non_http_scheme_rejected_before_urlopen(url, monkeypatch):
    # The guard must fire before urlopen is ever reached (no local-file read).
    def boom(*a, **k):                       # pragma: no cover - must not run
        raise AssertionError('urlopen should not be called')
    monkeypatch.setattr(_http.urllib.request, 'urlopen', boom)
    with pytest.raises(DataFetchError, match='http/https'):
        _http.http_get(url)


class _SizedResp(_FakeResp):
    def __init__(self, body, content_length=None):
        self._body = body
        self.headers = {} if content_length is None else {
            'Content-Length': str(content_length)}

    def read(self, amt=-1):
        return self._body[:amt] if amt and amt >= 0 else self._body


def test_size_cap_rejects_oversize_body(monkeypatch):
    monkeypatch.setattr(_http.urllib.request, 'urlopen',
                        lambda req, timeout=None: _SizedResp(b'x' * 50))
    with pytest.raises(DataFetchError, match='cap'):
        _http.http_get('http://x', max_bytes=10)


def test_size_cap_rejects_oversize_content_length(monkeypatch):
    monkeypatch.setattr(_http.urllib.request, 'urlopen',
                        lambda req, timeout=None: _SizedResp(b'x', content_length=10000))
    with pytest.raises(DataFetchError, match='cap'):
        _http.http_get('http://x', max_bytes=100)


def test_size_cap_passes_within_limit(monkeypatch):
    monkeypatch.setattr(_http.urllib.request, 'urlopen',
                        lambda req, timeout=None: _SizedResp(b'hello'))
    assert _http.http_get('http://x', max_bytes=100) == b'hello'


def test_checked_member_size_caps_bomb():
    assert _http.checked_member_size(100, 'a.tif', max_bytes=1000) == 100
    with pytest.raises(DataFetchError, match='decompression bomb'):
        _http.checked_member_size(2000, 'a.tif', max_bytes=1000)
    with pytest.raises(DataFetchError, match='decompression bomb'):
        _http.checked_member_size(-1, 'a.tif', max_bytes=1000)


def test_raise_substantive_rejects_an_empty_error_list():
    # errors[-1] on an empty chain raised IndexError with no remediation.
    with pytest.raises(ConfigurationError, match='No data source was tried'):
        raise_substantive([])


def test_a_refused_connection_gets_one_quick_retry_not_the_ladder():
    """ECONNREFUSED is the remote answering instantly with a rejection, so
    the fetch makes one quick second attempt and raises — the exponential
    ladder is for flakes that need time (resets, timeouts, 5xx), and
    against a down host its sleeps multiply across every grid of a
    multi-file build. Port 9 (discard) is closed on any test host, so the
    refusal is local and immediate."""
    import time as _time
    t0 = _time.monotonic()
    with pytest.raises(DataFetchError, match='after 1 retries'):
        _http.http_get('http://127.0.0.1:9/grid.tif', timeout=10.0)
    assert _time.monotonic() - t0 < 5.0


class TestAHostThatDoesNotResolveFailsAtOnce:
    def test_no_retry_ladder_on_a_permanent_dns_failure(self):
        import time
        from uacpy.data._http import http_get
        from uacpy.core.exceptions import DataFetchError
        t0 = time.monotonic()
        with pytest.raises(DataFetchError) as info:
            http_get('http://no-such-host.invalid/grid.nc', timeout=5.0,
                     source='test')
        elapsed = time.monotonic() - t0
        if 'retries' in str(info.value):
            pytest.skip("resolver reported a temporary failure (offline?), "
                        "which stays on the retry ladder")
        assert 'resolve' in str(info.value)
        assert elapsed < 5.0, elapsed


def test_chain_any_walks_reason_and_cause_and_survives_a_cycle():
    """Both classifiers share one walker now. It follows ``URLError.reason``
    and ``__cause__``, and the ``seen`` set stops a self-referential chain
    from looping forever."""
    import socket
    import urllib.error
    from uacpy.data._http import _chain_any

    inner = socket.gaierror(socket.EAI_NONAME, 'Name or service not known')
    via_reason = urllib.error.URLError(inner)
    assert _chain_any(via_reason, lambda e: isinstance(e, socket.gaierror))

    outer = RuntimeError('wrapped')
    outer.__cause__ = inner
    assert _chain_any(outer, lambda e: isinstance(e, socket.gaierror))
    assert not _chain_any(outer, lambda e: isinstance(e, ZeroDivisionError))

    loop = RuntimeError('self')
    loop.__cause__ = loop
    assert not _chain_any(loop, lambda e: isinstance(e, socket.gaierror))
    assert _chain_any(loop, lambda e: isinstance(e, RuntimeError))


def test_both_classifiers_discriminate_through_the_shared_walker():
    """EAI_AGAIN stays transient; a non-refused errno stays non-refused."""
    import errno as _errno
    import socket
    import urllib.error
    from uacpy.data._http import (_is_connection_refused,
                                  _is_permanent_dns_failure)

    permanent = urllib.error.URLError(
        socket.gaierror(socket.EAI_NONAME, 'no such host'))
    transient = urllib.error.URLError(
        socket.gaierror(socket.EAI_AGAIN, 'try again'))
    assert _is_permanent_dns_failure(permanent)
    assert not _is_permanent_dns_failure(transient)
    assert not _is_connection_refused(permanent)

    refused = urllib.error.URLError(ConnectionRefusedError(
        _errno.ECONNREFUSED, 'refused'))
    timed_out = urllib.error.URLError(OSError(_errno.ETIMEDOUT, 'timed out'))
    assert _is_connection_refused(refused)
    assert not _is_connection_refused(timed_out)
    assert not _is_permanent_dns_failure(refused)
