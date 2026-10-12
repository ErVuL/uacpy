"""One provenance notice per fetch: the decorator that folds every notice on
where a fetch's data come from — the source, how far and how stale the data
are, a waypoint filled from its neighbour, a profile extended to the seafloor,
a licence — into ONE ``ProvenanceWarning`` naming each.

A transect samples a point fetcher at every waypoint, and each sample gives
its own notice; the decorator turns those into one list per call. A fetch
decorated with it that runs inside another one passes its notices up, so the
outermost call gives the one summary.

The data layer gives each such notice through :func:`provenance_notice` or
:func:`provenance_fallback`, which append it to the summary of the decorated
fetch running in the caller's context (a ``ContextVar``, so one thread's
fetch never collects another's) and warn at once outside one. No
``warnings.catch_warnings`` window is opened: it edits the process-global
filter list, so it would mute and collect every thread's warnings while it
is held. Every other warning reaches the user as it is raised.
"""

import contextvars
import functools
import re
import warnings

from uacpy.core.exceptions import FallbackWarning, ProvenanceWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP

__all__ = ['one_provenance_notice', 'provenance_notice',
           'provenance_fallback']

#: The end of a notice's first sentence: a full stop after a letter or a
#: bracket, so "62.08 N" and "1.5 m" do not end it, and not the one of a
#: citation's "et al.".
_SENTENCE_END = re.compile(r'(?<=[A-Za-z)\]])(?<!et al)\.\s')

#: The notice list of the decorated fetch running in this context, or
#: ``None`` outside one; a decorated fetch called inside another appends to
#: the outer one's list.
_COLLECTED = contextvars.ContextVar('uacpy_provenance_collected',
                                    default=None)


def _first_sentence(text: str) -> str:
    """``text`` up to its first sentence end, whitespace collapsed."""
    text = ' '.join(text.split())
    match = _SENTENCE_END.search(text)
    return text[:match.start()] if match else text.rstrip('.')


def provenance_notice(message: str) -> None:
    """Give a ``ProvenanceWarning`` on where data come from (an offset, a
    stale date, a licence): one item of the summary of the decorated fetch
    running in this context, else a warning at once."""
    collected = _COLLECTED.get()
    if collected is not None:
        collected.append(message)
        return
    warnings.warn(message, ProvenanceWarning,
                  skip_file_prefixes=USER_FRAME_SKIP)


def provenance_fallback(message: str) -> None:
    """Give a ``FallbackWarning`` that says where a substituted value came
    from (a transect waypoint filled from its neighbour, a profile extended
    to the seafloor): one item of the summary of the decorated fetch running
    in this context, else a warning at once."""
    collected = _COLLECTED.get()
    if collected is not None:
        collected.append(message)
        return
    warnings.warn(message, FallbackWarning,
                  skip_file_prefixes=USER_FRAME_SKIP)


def one_provenance_notice(*, subject: str, record: str):
    """Decorate a fetch so one call gives at most one ``ProvenanceWarning``.

    The notice reads "``<fetch>``: N note(s) on where ``subject`` come
    from: (1) ...; (2) .... Each source's full record is in ``record``.",
    each item the first sentence of a notice given through
    :func:`provenance_notice` or :func:`provenance_fallback` during the call,
    repeats dropped. Every other warning reaches the user as it is raised.
    """
    def decorate(fetch):
        @functools.wraps(fetch)
        def fetch_with_one_provenance_notice(*args, **kwargs):
            if _COLLECTED.get() is not None:
                return fetch(*args, **kwargs)
            collected = []
            token = _COLLECTED.set(collected)
            try:
                result = fetch(*args, **kwargs)
            finally:
                _COLLECTED.reset(token)
            items = [_first_sentence(message) for message in collected]
            if items:
                items = list(dict.fromkeys(items))
                listed = '; '.join(f"({i}) {item}"
                                   for i, item in enumerate(items, start=1))
                warnings.warn(
                    f"{fetch.__name__}: {len(items)} note(s) on where "
                    f"{subject} come from: {listed}. Each source's full "
                    f"record is in {record}.",
                    ProvenanceWarning, skip_file_prefixes=USER_FRAME_SKIP)
            return result
        return fetch_with_one_provenance_notice
    return decorate
