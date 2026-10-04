"""The source-fallback chain every fetched axis resolves by.

A chain is an ordered list of ``(source, backend)`` attempts. Within one
source the cached twin comes first; a cached twin that *read* an answer ("no
coverage", "on land") ends that source, since its live twin serves the same
dataset, and only an absent or unreadable cache
(:func:`uacpy.data._cache.is_cache_miss`) falls through to the live backend.
When no attempt answers, the most substantive error is raised.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import _cache

__all__ = ['CACHE_BACKEND', 'SourceProvider', 'SourceChain', 'first_answer',
           'raise_substantive', 'source_tuple']

#: The backend token of a source's cached twin.
CACHE_BACKEND = 'local'


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


def first_answer(attempts, call, *, cached=lambda backend:
                 backend == CACHE_BACKEND):
    """``(call(source, backend), (source, backend))`` for the first attempt
    that answers.

    ``attempts`` is an iterable of ``(source, backend)`` pairs, tried in
    order; ``call`` raises :class:`ConfigurationError` or
    :class:`DataFetchError` when an attempt yields nothing, and any other
    exception propagates. ``cached(backend)`` says whether a backend is its
    source's cached twin: when that twin fails with anything but a cache miss
    the rest of that source is skipped. When no attempt answers,
    :func:`raise_substantive` raises the most substantive error collected.
    """
    errors = []
    answered = set()                  # sources whose cached twin read an answer
    for source, backend in attempts:
        if source in answered:
            continue
        try:
            return call(source, backend), (source, backend)
        except (ConfigurationError, DataFetchError) as exc:
            errors.append(exc)
            if cached(backend) and not _cache.is_cache_miss(exc):
                answered.add(source)
    raise_substantive(errors)


def source_tuple(value):
    """A ``str``/sequence source spec as a tuple of lower-case source names:
    every axis matches names case-insensitively, so ``'GEBCO'`` and
    ``'gebco'`` select the same backend."""
    if isinstance(value, str):
        return (value.lower(),)
    return tuple(str(v).lower() for v in value)


@dataclass(frozen=True)
class SourceProvider:
    """One fetchable source of an axis.

    ``id`` is its catalogue id (the source keyword and the provenance id);
    ``backends`` the backend tokens tried for it, its cached twin
    (:data:`CACHE_BACKEND`) first. ``point(backend, point, **request)`` and
    ``transect(backend, start, end, **request)`` fetch it; ``request`` holds
    the axis's keywords, of which each fetcher reads its own. ``transect`` is
    ``None`` for a source with no transect fetch.
    """

    id: str
    backends: Tuple[str, ...]
    point: Optional[Callable] = None
    transect: Optional[Callable] = None

    def attempts(self, *, cache_only=False):
        """``(id, backend)`` for each backend, cached twin first;
        ``cache_only`` keeps the cached twin alone, so a source with no
        cached twin contributes none."""
        return [(self.id, backend) for backend in self.backends
                if not cache_only or backend == CACHE_BACKEND]


@dataclass(frozen=True)
class SourceChain:
    """The sources of one ``fetch_environment`` axis and its ``'auto'``
    order. ``'local'`` is the ``'auto'`` chain restricted to cached twins."""

    axis: str
    providers: Tuple[SourceProvider, ...]
    auto: Tuple[str, ...]

    def provider(self, source):
        """The provider whose id is ``source``."""
        return next(p for p in self.providers if p.id == source)

    def resolve(self, spec):
        """A ``*_sources`` spec as ``(sources, cache_only)``: ``'auto'`` →
        the axis's best-available chain; ``'local'`` → the same chain with
        cached backends only (no network); else the explicit list."""
        if spec == 'local':
            return self.auto, True
        if spec == 'auto':
            return self.auto, False
        return source_tuple(spec), False

    def attempts(self, sources, *, cache_only, who='fetch_environment',
                 keyword='*_sources='):
        """The ordered ``(source, backend)`` attempts of ``sources``,
        cache-first within each source; ``cache_only`` keeps the cached
        backends alone. An unknown source name is refused, naming
        ``who`` and the ``keyword`` the spec was passed as."""
        known = sorted(p.id for p in self.providers)
        attempts = []
        for src in sources:
            if src not in known:
                raise ConfigurationError(
                    f"{who}: unknown {self.axis} source {src!r}.",
                    remediation=(
                        f"{src!r} is a preset for the whole {keyword} value "
                        f"and is not valid inside a sequence; pass it alone, "
                        f"or list sources from {known}."
                        if src in ('auto', 'local')
                        else f"Use 'auto', 'local' or one of {known}."),
                )
            attempts.extend(self.provider(src).attempts(cache_only=cache_only))
        return attempts
