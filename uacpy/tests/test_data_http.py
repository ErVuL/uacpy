"""Tests for the shared HTTP layer retry/backoff (uacpy.data._http)."""

import urllib.error

import pytest

from uacpy.core.exceptions import DataFetchError
from uacpy.data import _http


class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
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
