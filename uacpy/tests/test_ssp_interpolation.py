"""SSP-interpolation method-focused tests."""

import pytest
import numpy as np

from uacpy.core.environment import SoundSpeedProfile
from uacpy import Field
from uacpy.models import Bellhop
from uacpy.core import Environment, Receiver, Source

pytestmark = pytest.mark.requires_binary


class TestSSPInterpolationMethods:
    """End-to-end acceptance of each ``interp_ssp`` scheme.

    Each case writes a deck with a different ``TopOpt(1)`` character and runs
    the binary, so a scheme that produced an unwritable deck (or one AT
    rejects) fails here. Nothing compares the fields, so these do **not**
    show that the chosen scheme changed the interpolation — only that the
    whole path is wired. The character mapping itself is pinned in
    ``test_ssp_presets.py``.
    """

    @pytest.fixture
    def receiver(self):
        return Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0])
        )

    @pytest.mark.requires_binary
    def test_ssp_isovelocity(self, source, receiver):
        """An isovelocity env forces ``TopOpt(1)='C'`` whatever the model's
        ``interp_ssp`` says, so this is the shortcut path."""
        env = Environment(
            name="iso_test",
            bathymetry=100.0,
            ssp=1500.0
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)

    @staticmethod
    def _deck_topopt_char(work_dir):
        """``TopOpt(1)`` from the staged ``.env`` — the binary-free
        discriminator for which connection scheme the deck asked for. Line 0
        is the quoted title; the next quoted line is TopOpt."""
        env_file = next(iter(work_dir.rglob('*.env')))
        topopt = next(ln for ln in env_file.read_text().splitlines()[1:]
                      if ln.startswith("'"))
        return topopt[1]

    @pytest.mark.requires_binary
    def test_ssp_linear(self, source, receiver, tmp_path):
        """Bellhop with linear-connected SSP samples writes AT's C-linear
        ``TopOpt(1)='C'`` (``_AT_INTERP_TO_CODE`` in io/oalib_writer.py)."""
        depths = np.array([0, 50, 100])
        speeds = np.array([1500, 1490, 1480])

        env = Environment(
            name="linear_test",
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(np.column_stack([depths, speeds])),
        )

        bellhop = Bellhop(verbose=False, interp_ssp='linear',
                          work_dir=tmp_path, cleanup=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)
        assert self._deck_topopt_char(tmp_path) == 'C'

    @pytest.mark.requires_binary
    def test_ssp_cubic(self, source, receiver, tmp_path):
        """Bellhop with cubic-spline-connected SSP samples writes AT's
        spline ``TopOpt(1)='S'``."""
        depths = np.array([0, 25, 50, 75, 100])
        speeds = np.array([1500, 1495, 1490, 1485, 1480])

        env = Environment(
            name="cubic_test",
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(np.column_stack([depths, speeds])),
        )

        bellhop = Bellhop(verbose=False, interp_ssp='cubic',
                          work_dir=tmp_path, cleanup=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)
        assert self._deck_topopt_char(tmp_path) == 'S'

    @pytest.mark.requires_binary
    def test_linear_and_cubic_fields_differ_on_a_curved_profile(self):
        """The scheme must reach the physics, not just the deck token: over
        a genuinely curved profile (a duct — the spline overshoots between
        samples where the C-linear connection cannot) the two schemes bend
        rays differently, and by 400 Hz x 8 km the interference pattern has
        moved. The profiles in the two smokes above cannot show this: the
        cubic one's samples are collinear, so its spline IS the line."""
        depths = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        speeds = np.array([1500.0, 1490.0, 1487.0, 1490.0, 1500.0])
        env = Environment(
            name="curved_test",
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(np.column_stack([depths, speeds])),
        )
        src = Source(depths=50.0, frequencies=400.0)
        rcv = Receiver(depths=np.array([50.0]),
                       ranges=np.linspace(500.0, 8000.0, 30))
        tl = {}
        for scheme in ('linear', 'cubic'):
            field = Bellhop(verbose=False, interp_ssp=scheme).compute_tl(
                env=env, source=src, receiver=rcv)
            tl[scheme] = np.asarray(field.db).ravel()
        both = np.isfinite(tl['linear']) & np.isfinite(tl['cubic'])
        assert both.sum() > tl['linear'].size // 2
        # Identical decks reproduce bit-identically, so any real difference
        # proves the TopOpt letter reached the binary; 0.1 dB keeps the
        # assertion above numeric trivia.
        assert np.max(np.abs(tl['linear'][both] - tl['cubic'][both])) > 0.1
