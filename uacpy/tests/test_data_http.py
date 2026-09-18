"""Tests for the shared HTTP layer retry/backoff (uacpy.data._http)."""

import pathlib
import urllib.error
import urllib.request

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


# ── curl_download resumes a broken transfer ────────────────────────────────


def _fake_curl(script, log):
    """A ``subprocess.run`` that plays ``script`` and writes like curl would.

    Each entry is the number of bytes that attempt manages to write before it
    raises, or ``None`` for an attempt that completes. ``-C -`` appends, its
    absence truncates — which is the whole behaviour under test.
    """
    import subprocess as sp

    payload = bytes(range(256)) * 64

    def run(command, **kwargs):
        log.append(list(command))
        out = pathlib.Path(command[command.index('-o') + 1])
        resuming = '-C' in command
        have = out.stat().st_size if resuming and out.exists() else 0
        wrote = script.pop(0)
        chunk = payload[have:] if wrote is None else payload[have:have + wrote]
        with open(out, 'ab' if resuming else 'wb') as handle:
            handle.write(chunk)
        if wrote is not None:
            raise sp.CalledProcessError(18, command)
        return sp.CompletedProcess(command, 0)

    return run, payload


def test_a_broken_transfer_is_resumed_rather_than_restarted(tmp_path,
                                                            monkeypatch):
    """The bug: curl fixes the ``-C -`` offset when it starts, so an internal
    ``--retry`` restarts there and discards what the failed attempt wrote."""
    log = []
    run, payload = _fake_curl([4000, 6000, None], log)
    monkeypatch.setattr(_http.shutil, 'which', lambda _: '/usr/bin/curl')
    monkeypatch.setattr(_http.subprocess, 'run', run)
    out = tmp_path / 'grid.nc'

    assert _http.curl_download('https://example.invalid/g.nc', out,
                               timeout=30.0, verbose=False) is True
    assert out.read_bytes() == payload, "the grid arrived corrupt"
    assert len(log) == 3, "the transfer was not retried to completion"
    assert '-C' not in log[0], "the first attempt has nothing to resume onto"
    assert all('-C' in command for command in log[1:]), (
        "a later attempt restarted the transfer instead of resuming it")
    assert not any('--retry' in command for command in log), (
        "--retry is back: curl's internal retry discards the partial file")


def test_a_host_that_refuses_ranges_gets_the_file_from_byte_zero(
        tmp_path, monkeypatch):
    """curl exit 33 is "this server does not do byte ranges". Retrying the
    resume it just refused only repeats the refusal, so the file restarts."""
    import subprocess as sp
    log = []
    payload = b'x' * 500
    state = {'attempt': 0}

    def run(command, **kwargs):
        log.append(list(command))
        out = pathlib.Path(command[command.index('-o') + 1])
        state['attempt'] += 1
        if state['attempt'] == 1:
            out.write_bytes(payload[:100])
            raise sp.CalledProcessError(18, command)
        if state['attempt'] == 2:
            raise sp.CalledProcessError(33, command)   # refuses the range
        assert '-C' not in command, "resumed onto a host that refuses ranges"
        out.write_bytes(payload)
        return sp.CompletedProcess(command, 0)

    monkeypatch.setattr(_http.shutil, 'which', lambda _: '/usr/bin/curl')
    monkeypatch.setattr(_http.subprocess, 'run', run)
    out = tmp_path / 'grid.nc'
    assert _http.curl_download('https://example.invalid/g.nc', out,
                               timeout=30.0, verbose=False) is True
    assert out.read_bytes() == payload
    assert '-C' in log[1], "the second attempt should have tried to resume"


def test_two_attempts_without_progress_give_up_and_drop_the_partial(
        tmp_path, monkeypatch):
    """A resume that keeps failing at the same offset is not going to finish;
    spending the whole attempt budget on it only multiplies the timeout."""
    import subprocess as sp
    log = []

    def run(command, **kwargs):
        log.append(list(command))
        out = pathlib.Path(command[command.index('-o') + 1])
        if len(log) == 1:
            out.write_bytes(b'y' * 200)
        raise sp.CalledProcessError(18, command)

    monkeypatch.setattr(_http.shutil, 'which', lambda _: '/usr/bin/curl')
    monkeypatch.setattr(_http.subprocess, 'run', run)
    out = tmp_path / 'grid.nc'
    assert _http.curl_download('https://example.invalid/g.nc', out,
                               timeout=30.0, verbose=False) is False
    assert not out.exists(), "a failed download left a file for the cache"
    assert len(log) == 3, (
        f"gave up after {len(log)} attempts; expected the first, then two "
        f"that make no progress")
    assert not list(tmp_path.glob('*')), "the staging file was left behind"


# ── every static-grid downloader takes an address override ─────────────────


class _Recorded(Exception):
    """Raised by a stubbed transport once it has recorded its address."""


#: One row per cache filler — the nine ``download_*_db`` fetchers and the
#: coastline downloader behind the map plotters: the module that defines it,
#: the keyword it takes an address on, and the transport attributes to stub. A
#: fetcher that imports its transport inside the function body reaches it
#: through ``uacpy.data._http``, so that is where the stub goes; the rest
#: bind it at module scope and take the stub in their own namespace.
_ADDRESS_OVERRIDES = [
    ('uacpy.data.crust1_local', 'download_crust1_db', 'url',
     [('uacpy.data.crust1_local', 'http_get')]),
    ('uacpy.data.globsed_local', 'download_globsed_db', 'url',
     [('uacpy.data.globsed_local', 'download_grid_file')]),
    ('uacpy.data.graw_local', 'download_graw_db', 'url',
     [('uacpy.data._http', 'download_grid_file')]),
    ('uacpy.data.glodap_local', 'download_glodap_db', 'url',
     [('uacpy.data._http', 'curl_download')]),
    ('uacpy.data.diesing_local', 'download_diesing_db', 'url',
     [('uacpy.data.diesing_local', 'http_get')]),
    ('uacpy.data.sediment_db', 'download_sediment_db', 'url',
     [('uacpy.data.sediment_db', 'http_get')]),
    ('uacpy.data.wind_local', 'download_wind_db', 'url',
     [('uacpy.data.wind_local', 'curl_download')]),
    ('uacpy.data.emodnet_local', 'download_emodnet_db', 'base_url',
     [('uacpy.data.emodnet_local', 'http_get')]),
    ('uacpy.data.seaice_local', 'download_seaice_db', 'base_url',
     [('uacpy.data.seaice_local', 'http_get')]),
    # Not a ``_db`` name and not in ``uacpy.data``, but the same contract: it
    # fills the offline cache the map plotters read, so it is pinned with its
    # nine siblings rather than on its own where the next one would be missed.
    ('uacpy.visualization.basemap', 'download_coastline', 'url',
     [('uacpy.visualization.basemap', 'http_get')]),
]

_MIRROR = 'https://mirror.invalid/somewhere'


def _stub_transports(monkeypatch, targets, seen):
    import importlib

    def record(*args, **kwargs):
        # The address is the first argument that looks like one: these
        # transports take it first (``http_get``, ``curl_download``) or second
        # (``download_grid_file``, behind the dataset name).
        address = next((a for a in args if isinstance(a, str) and '://' in a),
                       kwargs.get('url'))
        seen.append(address)
        raise _Recorded(address)

    for module_name, attr in targets:
        monkeypatch.setattr(importlib.import_module(module_name), attr, record)


@pytest.mark.parametrize(
    'module_name,func_name,keyword,targets', _ADDRESS_OVERRIDES,
    ids=[row[1] for row in _ADDRESS_OVERRIDES])
def test_every_downloader_fetches_the_address_it_is_given(
        module_name, func_name, keyword, targets, tmp_path, monkeypatch):
    """The address a caller passes is the address that is requested.

    A publisher that moves, a site that 403s every path, a hung service: each
    is a fetcher pinned to one hostname with no way past it except editing the
    package. Every one of them therefore takes ``url=`` (or ``base_url=``
    where it builds many requests from a directory tree), and this drives all
    ten, because forwarding the keyword is exactly the step that is easy to
    add to a signature and forget in the body — ``download_coastline`` had the
    signature of the other nine and no override at all.
    """
    import importlib

    module = importlib.import_module(module_name)
    seen = []
    _stub_transports(monkeypatch, targets, seen)
    # tifffile/shapely are imported before the first fetch by two of these.
    if func_name == 'download_seaice_db':
        pytest.importorskip('tifffile')
    if func_name == 'download_emodnet_db':
        pytest.importorskip('shapely')

    with pytest.raises((_Recorded, DataFetchError)):
        getattr(module, func_name)(
            cache_dir=str(tmp_path / func_name),
            **{keyword: f'{_MIRROR}/{func_name}.bin'})

    assert seen, f"{func_name} reached no transport"
    assert seen[0].startswith(_MIRROR), (
        f"{func_name} ignored {keyword}= and fetched {seen[0]!r}")


def test_the_override_table_covers_every_downloader():
    """The sweep itself: a new cache filler must be listed here, or the
    parametrisation above silently stops covering the package.

    Two namespaces, because the cache is filled from two: the ``download_*_db``
    fetchers in ``uacpy.data``, and ``download_coastline`` in
    ``uacpy.visualization``, which caches the land polygons every map plotter
    draws. Looking only at ``uacpy.data`` is what let that one ship with no
    ``url=`` at all while its nine siblings had one.
    """
    import uacpy.data as data
    import uacpy.visualization as viz

    fetchers = {name for name in data.__all__
                if name.startswith('download_') and name.endswith('_db')}
    fetchers |= {name for name in viz.__all__ if name.startswith('download_')}
    assert fetchers == {row[1] for row in _ADDRESS_OVERRIDES}


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'ftp://h/x', '/etc/passwd'])
def test_curl_refuses_a_non_http_address_too(url, tmp_path, monkeypatch):
    """``curl -fL`` reads ``file://`` and speaks ftp as happily as urlopen, and
    the large grids take the curl path, so the guard that blocks local-file
    disclosure has to sit on both transports."""
    def boom(*a, **k):                    # pragma: no cover - must not run
        raise AssertionError('curl should not be launched')
    monkeypatch.setattr(_http.shutil, 'which', lambda _: '/usr/bin/curl')
    monkeypatch.setattr(_http.subprocess, 'run', boom)
    with pytest.raises(DataFetchError, match='http/https'):
        _http.curl_download(url, tmp_path / 'out.bin', timeout=5.0,
                            verbose=False)


@pytest.mark.parametrize('scheme', ['ftp', 'file'])
def test_a_redirect_that_lands_outside_http_returns_no_bytes(scheme,
                                                             monkeypatch):
    """The address is checked; so is where it ends up.

    Python's own ``HTTPRedirectHandler`` follows a 302 into ``ftp://`` (it
    excludes only schemes outside http/https/ftp), so a host at a permitted
    https address could answer with a redirect into the very scheme
    ``_ALLOWED_SCHEMES`` exists to forbid. The final URL is checked before any
    byte is read.
    """
    class _Redirected(_FakeResp):
        url = f'{scheme}://example.invalid/g.nc'

        def read(self, amt=-1):          # pragma: no cover - must not run
            raise AssertionError('the body was read from ' + self.url)

    monkeypatch.setattr(_http.urllib.request, 'urlopen',
                        lambda req, timeout=None: _Redirected())
    with pytest.raises(DataFetchError, match='http/https'):
        _http.http_get('https://example.invalid/g.nc')


def test_curl_is_told_which_schemes_a_redirect_may_use(tmp_path, monkeypatch):
    """The same rule on the other transport, where it is argv rather than a
    handler: without ``--proto-redir`` curl follows ftp on a redirect."""
    log = []

    def run(command, **kwargs):
        log.append(list(command))
        raise RuntimeError('stop after argv capture')

    monkeypatch.setattr(_http.shutil, 'which', lambda _: '/usr/bin/curl')
    monkeypatch.setattr(_http.subprocess, 'run', run)
    with pytest.raises(Exception):
        _http.curl_download('https://example.invalid/g.nc', tmp_path / 'g.nc',
                            timeout=5.0, verbose=False)
    argv = log[0]
    assert '--proto-redir' in argv
    assert argv[argv.index('--proto-redir') + 1] == '=http,https'
    assert argv[argv.index('--proto') + 1] == '=http,https'
