"""One provenance notice per fetch: the decorator that folds every notice on
where a fetch's data come from — the source, how far and how stale the data
are, a waypoint filled from its neighbour, a profile extended to the seafloor,
a licence — into ONE ``ProvenanceWarning`` naming each.

A transect samples a point fetcher at every waypoint, and each sample gives
its own notice; the decorator turns those into one list per call. A fetch
decorated with it that runs inside another one passes its notices up, so the
outermost call gives the one summary.
"""

import contextvars
import functools
import re
import warnings

from uacpy.core.exceptions import FallbackWarning, ProvenanceWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP

__all__ = ['one_provenance_notice']

#: The FallbackWarnings of the fetchers that say where a value came from
#: rather than that a request failed: a transect waypoint filled from its
#: nearest covered neighbour (``data.sediment``) and a profile extended to the
#: seafloor (``data.sound_speed.extend_ssp_below_data``). They are folded into
#: the one provenance summary.
_PROVENANCE_FALLBACK_PHRASES = (
    'were filled from the nearest covered waypoint',
    "extrapolated the last",
)

#: The end of a notice's first sentence: a full stop after a letter or a
#: bracket, so "62.08 N" and "1.5 m" do not end it, and not the one of a
#: citation's "et al.".
_SENTENCE_END = re.compile(r'(?<=[A-Za-z)\]])(?<!et al)\.\s')

#: True while a decorated fetch collects its notices; a decorated fetch
#: called inside it passes its warnings up as they come.
_COLLECTING = contextvars.ContextVar('uacpy_provenance_collecting',
                                     default=False)


def _first_sentence(text: str) -> str:
    """``text`` up to its first sentence end, whitespace collapsed."""
    text = ' '.join(text.split())
    match = _SENTENCE_END.search(text)
    return text[:match.start()] if match else text.rstrip('.')


def _is_provenance(warning) -> bool:
    """Whether a caught warning says where the data come from."""
    return (issubclass(warning.category, ProvenanceWarning)
            or (issubclass(warning.category, FallbackWarning)
                and any(phrase in str(warning.message)
                        for phrase in _PROVENANCE_FALLBACK_PHRASES)))


def one_provenance_notice(*, subject: str, record: str):
    """Decorate a fetch so one call gives at most one ``ProvenanceWarning``.

    The notice reads "``<fetch>``: N note(s) on where ``subject`` come
    from: (1) ...; (2) .... Each source's full record is in ``record``.",
    each item the first sentence of the notice it replaces, repeats dropped.
    Every other warning is passed on as it came.
    """
    def decorate(fetch):
        @functools.wraps(fetch)
        def fetch_with_one_provenance_notice(*args, **kwargs):
            if _COLLECTING.get():
                return fetch(*args, **kwargs)
            token = _COLLECTING.set(True)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    result = fetch(*args, **kwargs)
            finally:
                _COLLECTING.reset(token)
            items = []
            for warning in caught:
                if _is_provenance(warning):
                    items.append(_first_sentence(str(warning.message)))
                else:
                    category = warning.category
                    warnings.warn(str(warning.message), category,
                                  skip_file_prefixes=USER_FRAME_SKIP)
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
