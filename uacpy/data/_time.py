"""Shared calendar-date parsing for the data layer."""

import datetime as _dt
from typing import Union

from uacpy.core.exceptions import ConfigurationError

__all__ = ['parse_date']


def parse_date(date: Union[str, _dt.date]) -> _dt.date:
    """Parse a calendar date into a :class:`datetime.date`.

    Accepts an ISO-8601 string (``'YYYY-MM-DD'`` or a full datetime string) or a
    ``datetime.date`` / ``datetime.datetime``. Raises a typed
    :class:`ConfigurationError` on a malformed value rather than letting a bare
    ``ValueError`` escape.
    """
    if isinstance(date, _dt.datetime):
        return date.date()
    if isinstance(date, _dt.date):
        return date
    if isinstance(date, str):
        for parse in (_dt.date.fromisoformat,
                      lambda s: _dt.datetime.fromisoformat(s).date()):
            try:
                return parse(date)
            except ValueError:
                continue
        raise ConfigurationError(
            f"could not parse date {date!r}; expected ISO-8601 (YYYY-MM-DD).",
            remediation="Pass e.g. '2026-06-14' or a datetime.date.",
        )
    raise ConfigurationError(
        f"date must be an ISO-8601 string or datetime.date; "
        f"got {type(date).__name__}.",
        remediation="Pass e.g. '2026-06-14' or a datetime.date.",
    )
