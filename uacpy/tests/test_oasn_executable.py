"""
Tests specific to OASN executable resolution and error paths.

OASN instantiation and run-time tests (covariance + replicas) live in
test_oases_comprehensive.py; this file covers only the things unique to
the OASN wrapper:
  - the missing-executable error message
  - the ``_oasn_available`` probe used by the wrapper to decide whether
    to expose its hydrophone-array products.
"""

from pathlib import Path

import pytest

from uacpy.core.exceptions import ExecutableNotFoundError
from uacpy.models.oases import OASN

pytestmark = pytest.mark.requires_oases


class TestOASNExecutableResolution:
    """OASN-specific executable-path handling."""

    @pytest.mark.requires_binary
    def test_oasn_executable_detection(self):
        """If OASN reports availability, the resolved path must actually exist."""
        oasn = OASN()
        if hasattr(oasn, "_oasn_available") and oasn._oasn_available:
            assert oasn.oasn_executable is not None
            assert oasn.oasn_executable.exists()

    def test_oasn_missing_executable(self):
        """Pointing OASN at a nonexistent path raises ExecutableNotFoundError."""
        with pytest.raises(ExecutableNotFoundError, match="OASN executable not found"):
            OASN(executable=Path("/nonexistent/oast"))


class TestExecutableSearchOrder:
    """oases.md §9: uacpy prefers the ``<name>_bash`` wrapper OASES ships and
    falls back to the bare binary, searching ``uacpy/bin/oases/``,
    ``uacpy/bin/oalib/``, ``uacpy/third_party/oases/bin/`` and finally PATH.
    Both tests stub the filesystem — no binary is needed or launched."""

    def test_oasn_asks_for_the_bash_wrapper_first(self, monkeypatch, tmp_path):
        calls = []
        # _resolve_executable verifies the found path is a file this process
        # can exec, so the stub has to hand back a real executable one.
        wrapper = tmp_path / 'oasn2_bin_bash'
        wrapper.touch()
        wrapper.chmod(0o755)

        def fake_find(self, names, bin_subdirs=None, dev_subdir=None,
                      try_exe_suffix=True):
            names = [names] if isinstance(names, str) else list(names)
            calls.append((names, list(bin_subdirs or []), dev_subdir))
            return tmp_path / names[0]

        monkeypatch.setattr(OASN, '_find_executable_in_paths', fake_find)
        OASN(verbose=False)
        assert (['oasn2_bin_bash', 'oasn2_bin'], ['oases', 'oalib'],
                'oases') in calls

    def test_bash_wrapper_outranks_the_bare_binary_across_dirs(
            self, monkeypatch):
        import pathlib
        import uacpy.models.base as base_mod

        base_dir = pathlib.Path(base_mod.__file__).parent.parent
        bash_oases = base_dir / 'bin' / 'oases' / 'oasn2_bin_bash'
        bare_oalib = base_dir / 'bin' / 'oalib' / 'oasn2_bin'
        runnable = set()
        # The search selects a candidate the OS would actually exec, so the
        # stub stands in for that predicate rather than for bare existence.
        monkeypatch.setattr(base_mod, '_is_runnable',
                            lambda path: str(path) in runnable)
        monkeypatch.setattr(base_mod.shutil, 'which', lambda name: None)

        model = OASN.__new__(OASN)      # only the search helper is exercised
        model.model_name = 'OASN'

        def find():
            return model._find_executable_in_paths(
                ['oasn2_bin_bash', 'oasn2_bin'],
                bin_subdirs=['oases', 'oalib'], dev_subdir='oases')

        # Both present: the _bash wrapper wins even though the bare binary
        # exists in an earlier-tried directory for its own name.
        runnable.update({str(bash_oases), str(bare_oalib)})
        assert find() == bash_oases

        # Only the bare binary in bin/oalib: found by falling through every
        # _bash candidate first.
        runnable.clear()
        runnable.add(str(bare_oalib))
        assert find() == bare_oalib

        # Nothing anywhere: the error lists the candidates in search order,
        # _bash first.
        runnable.clear()
        with pytest.raises(ExecutableNotFoundError) as err:
            find()
        msg = str(err.value)
        assert msg.index(str(bash_oases)) < msg.index(str(bare_oalib))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
