"""The source-fallback chain every fetched axis resolves by.

A chain is an ordered list of ``(source, backend)`` attempts. Within one
source the cached twin comes first; a cached twin that *read* an answer ("no
coverage", "on land") ends that source, since its live twin serves the same
dataset, and only an absent or unreadable cache
(:func:`uacpy.data._cache.is_cache_miss`) falls through to the live backend.
When no attempt answers, the most substantive error is raised.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (ConfigurationError, DataFetchError,
                                   FallbackWarning)
from uacpy.data import _cache

__all__ = ['CACHE_BACKEND', 'MissingPrerequisiteError', 'SourceProvider',
           'SourceChain', 'first_answer', 'lower_source_spec',
           'raise_substantive', 'source_tuple', 'unused_setting_notices']

#: The backend token of a source's cached twin.
CACHE_BACKEND = 'local'


class MissingPrerequisiteError(ConfigurationError):
    """A source asked for without an argument it needs (``date=``,
    ``transect_to=``).

    A :class:`ConfigurationError` to every caller. It is the caller's input,
    not a fetch that yielded nothing, so :func:`uacpy.data.fetch_environment`
    raises it even when a ``*=`` literal is given: the literal stands in only
    for data a complete request could not get (no coverage, a service down, a
    cache absent).
    """


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


def unused_setting_notices(who, unused):
    """One ``FallbackWarning`` per setting the caller gave that the run did
    not use, so it is never dropped in silence. ``unused``
    holds ``(name, value, why)`` triples; a ``None`` value is a setting left
    unset and gives none. ``why`` is the clause after "was not used:"."""
    for name, value, why in unused:
        if value is None:
            continue
        shown = (f"{name}={value!r}"
                 if isinstance(value, (str, int, float)) else f"{name}=")
        warnings.warn(f"{who}: {shown} was not used: {why}.",
                      FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)


def source_tuple(value):
    """A ``str``/sequence source spec as a tuple of lower-case source names:
    every axis matches names case-insensitively, so ``'GEBCO'`` and
    ``'gebco'`` select the same backend."""
    if isinstance(value, str):
        return (value.lower(),)
    return tuple(str(v).lower() for v in value)


def lower_source_spec(spec):
    """A ``source=`` / ``*_sources`` value lower-cased in its own shape: a
    preset or one name stays a ``str``, a sequence becomes a tuple of names,
    and ``None`` or a value that is no sequence passes as given (its
    validator refuses it) — so every preset check and validator downstream
    compares against one spelling."""
    if spec is None or isinstance(spec, str):
        return spec if spec is None else spec.lower()
    try:
        return source_tuple(spec)
    except TypeError:
        return spec


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
    """The sources of one fetched axis and its ``'auto'`` order.
    ``'local'`` is the ``'auto'`` chain restricted to cached twins, a preset
    where ``presets`` holds it (an axis with no cached source names only
    ``'auto'``)."""

    axis: str
    providers: Tuple[SourceProvider, ...]
    auto: Tuple[str, ...]
    presets: Tuple[str, ...] = ('auto', 'local')

    def provider(self, source):
        """The provider whose id is ``source``."""
        return next(p for p in self.providers if p.id == source)

    def resolve(self, spec):
        """A ``*_sources`` spec as ``(sources, cache_only)``: ``'auto'`` →
        the axis's best-available chain; ``'local'`` → the same chain with
        cached backends only (no network); else the explicit list. Names
        and presets match in any case."""
        spec = lower_source_spec(spec)
        if spec == 'local' and 'local' in self.presets:
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
                        if src in self.presets
                        else f"Use {self._preset_words()} or one of "
                             f"{known}."),
                )
            attempts.extend(self.provider(src).attempts(cache_only=cache_only))
        return attempts

    def _preset_words(self):
        """The presets as a refusal names them: ``'auto', 'local'``."""
        return ', '.join(repr(p) for p in self.presets)

    def answer(self, spec, call, *, who, keyword):
        """``(call(source, backend), (source, backend))`` for the first
        attempt of the ``spec`` chain that answers, by :func:`first_answer`;
        an unknown name is refused before any attempt."""
        sources, cache_only = self.resolve(spec)
        return first_answer(self.attempts(sources, cache_only=cache_only,
                                          who=who, keyword=keyword), call)

    def fetch(self, source, backend, where, request, *, who, keyword):
        """One attempt: ``source``'s point fetch at ``where = (point,)``, or
        its transect fetch when ``where`` holds both ends; a source with no
        transect fetch is refused (``ConfigurationError``, so a chain moves
        on), naming ``who`` and the ``keyword`` it was asked for by."""
        provider = self.provider(source)
        if len(where) == 1:
            return provider.point(backend, *where, **request)
        if provider.transect is None:
            usable = ' or '.join(repr(p.id) for p in self.providers
                                 if p.transect is not None)
            raise ConfigurationError(
                f"{who}: {keyword}{source!r} has no transect fetch, so it "
                f"gives no range-dependent {self.axis}.",
                remediation=f"Use {usable}.",
            )
        return provider.transect(backend, *where, **request)
