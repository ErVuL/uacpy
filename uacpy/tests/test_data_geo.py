"""Tests for the shared geographic helpers (uacpy.data._geo)."""

import pytest

from uacpy.core.exceptions import ConfigurationError
from uacpy.data._geo import as_coordinate, normalize_lon


@pytest.mark.parametrize('given,expected', [
    (0.0, 0.0),
    (-140.0, -140.0),
    (220.0, -140.0),     # 0–360 convention
    (360.0, 0.0),
    (180.0, -180.0),     # +180 wraps to -180
    (190.0, -170.0),     # just past dateline
    (-200.0, 160.0),
])
def test_normalize_lon(given, expected):
    assert normalize_lon(given) == pytest.approx(expected)


def test_as_coordinate_accepts_pairs():
    assert as_coordinate((43.2, 7.5)) == (43.2, 7.5)
    assert as_coordinate([1, 2]) == (1.0, 2.0)        # list, ints → floats


@pytest.mark.parametrize('bad', [43.2, (1.0,), (1, 2, 3), 'ab', None])
def test_as_coordinate_rejects_non_pairs(bad):
    with pytest.raises(ConfigurationError, match='coordinate pair'):
        as_coordinate(bad)


def test_parse_date_accepts_iso_and_objects():
    import datetime as dt
    from uacpy.data._time import parse_date
    assert parse_date('2026-06-14') == dt.date(2026, 6, 14)
    assert parse_date('2026-06-14T12:30:00') == dt.date(2026, 6, 14)
    assert parse_date(dt.date(2026, 6, 14)) == dt.date(2026, 6, 14)
    assert parse_date(dt.datetime(2026, 6, 14, 5)) == dt.date(2026, 6, 14)


@pytest.mark.parametrize('bad', ['2026-13-99', 'June 2026', '', 20260614, None])
def test_parse_date_rejects_bad(bad):
    from uacpy.data._time import parse_date
    with pytest.raises(ConfigurationError):
        parse_date(bad)
