"""``equally_spaced`` (``uacpy.core._validate``), the uniform-axis test the
io deck writers make before writing a compact axis: its tolerance and its
return type."""

import numpy as np
import pytest


class TestEquallySpacedToleranceBoundary:
    """io.md §10: ``equally_spaced(x, tol=1e-9)`` decides compact vs
    explicit axis encoding by comparing against the linspace rebuilt from
    the endpoints — a single interior sample's jitter is the deviation."""

    def test_uniform_axis_passes(self):
        from uacpy.core._validate import equally_spaced
        assert equally_spaced(np.linspace(0.0, 10.0, 11))

    def test_jitter_below_the_default_tolerance_passes(self):
        from uacpy.core._validate import equally_spaced
        x = np.linspace(0.0, 10.0, 11)
        x[5] += 5e-10
        assert equally_spaced(x)

    def test_jitter_above_the_default_tolerance_fails(self):
        from uacpy.core._validate import equally_spaced
        x = np.linspace(0.0, 10.0, 11)
        x[5] += 2e-9
        assert not equally_spaced(x)

    def test_explicit_tolerance_moves_the_boundary(self):
        from uacpy.core._validate import equally_spaced
        x = np.linspace(0.0, 10.0, 11)
        x[5] += 1e-6
        assert equally_spaced(x, tol=1e-5)
        assert not equally_spaced(x, tol=1e-7)

    @pytest.mark.parametrize('x', [[], [3.0]])
    def test_degenerate_axes_are_trivially_uniform(self, x):
        from uacpy.core._validate import equally_spaced
        assert equally_spaced(np.asarray(x, dtype=float))


class TestEquallySpacedReturnsARealBool:
    """``equally_spaced`` is annotated ``-> bool`` and documents "is_equal :
    bool", but returned ``np.bool_`` from its main path — ``np.max(delta) <
    tol`` is a numpy scalar. The distinction is not cosmetic:
    ``isinstance(np.True_, bool)`` is False, ``np.True_ is True`` is False, and
    ``json.dumps`` raises ``TypeError: Object of type bool is not JSON
    serializable`` on it.

    It was also internally inconsistent: the ``n <= 1`` early return gave a
    real ``True``, so the return type depended on the input length. Found by
    running the module's own docstring examples, which expected ``True`` and
    got ``np.True_``.
    """

    @staticmethod
    def _cases():
        import numpy as np
        return [(np.linspace(0.0, 10.0, 11), True),
                (np.array([0.0, 1.0, 3.0, 7.0, 10.0]), False),
                (np.array([1.0]), True),          # the n <= 1 early return
                (np.array([]), True)]

    def test_every_path_returns_a_python_bool(self):
        from uacpy.core._validate import equally_spaced
        for x, _ in self._cases():
            got = equally_spaced(x)
            assert isinstance(got, bool), f"{type(got)} for size {x.size}"
            assert got is True or got is False

    def test_each_case_returns_its_expected_verdict(self):
        from uacpy.core._validate import equally_spaced
        for x, expected in self._cases():
            assert equally_spaced(x) == expected

    def test_the_result_is_json_serialisable(self):
        import json
        import numpy as np
        from uacpy.core._validate import equally_spaced
        assert json.dumps({'ok': equally_spaced(np.linspace(0.0, 10.0, 11))}) \
            == '{"ok": true}'
