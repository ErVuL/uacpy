"""Provenance + licence catalogue for the native propagation engines.

The model layer wraps third-party Fortran/C programs, each with its own
authorship and licence — exactly the kind of metadata the ``data/`` layer
already catalogues for ocean datasets (:mod:`uacpy.data.sources`). This is the
model-side equivalent: one :class:`ModelProvenance` per upstream program, keyed in
:data:`MODEL_PROVENANCE`. A :class:`~uacpy.models._spec.ModelSpec` references an
entry by ``id`` via its ``source`` field, so ``PropagationModel`` can surface
the citation (``model.source`` / ``model.citation``) and — mirroring the
non-commercial fetch warning in ``data/`` — emit a one-time ``ProvenanceWarning`` when
a licence-restricted engine (OASES) is instantiated.

Licence facts are taken from the vendored ``third_party/<engine>/LICENSE`` /
``README`` files, not invented; see ``third_party/MODIFICATIONS.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._repr import FieldsRepr

__all__ = ['ModelProvenance', 'MODEL_PROVENANCE', 'model_provenance']


@dataclass(frozen=True)
class ModelProvenance(FieldsRepr):
    """Provenance + licence metadata for one native propagation engine.

    Mirrors :class:`uacpy.data.sources.DataSource` for the model layer. The
    two booleans drive policy: ``commercial_use=False`` triggers the one-time
    instantiation warning (as CRUST1.0 does at fetch time), and
    ``redistributable=False`` records why an engine is a user-side install
    (``install.sh``) rather than bundled in the wheel/sdist.
    """

    id: str
    name: str
    authors: str
    license: str
    citation: str
    url: str
    commercial_use: bool
    redistributable: bool
    note: str = ''
    #: ``(engine class name, citation)`` pairs for an entry shared by
    #: several engines (the Acoustics Toolbox): each engine is cited by its
    #: own reference, and ``citation`` answers for any engine not listed.
    engine_citations: Tuple[Tuple[str, str], ...] = ()

    _REPR_FIELDS = ('name', 'authors', 'license')

    def citation_for(self, engine: str) -> str:
        """The reference to cite for the engine class named ``engine``:
        its own when ``engine_citations`` lists it, else ``citation``."""
        return dict(self.engine_citations).get(engine, self.citation)

    @property
    def attribution(self) -> str:
        """Short credit ``"<primary author>, <name>"`` — the model-side
        counterpart of :attr:`uacpy.data.sources.DataSource.attribution`, so a
        single credit path can render either kind. The primary author is the
        first author group (before any ``';'``) with its parenthetical
        affiliation stripped.
        """
        primary = self.authors.split(';')[0]
        primary = re.sub(r'\([^)]*\)', '', primary).strip().rstrip(',').strip()
        return f"{primary}, {self.name}"


MODEL_PROVENANCE: Dict[str, ModelProvenance] = {
    'acoustics_toolbox': ModelProvenance(
        id='acoustics_toolbox',
        name='Acoustics Toolbox',
        authors='Michael B. Porter (HLS Research)',
        license='GPL-3.0-or-later',
        citation='M. B. Porter, "The KRAKEN Normal Mode Program", '
                 'SACLANTCEN SM-245 (1991); Acoustics Toolbox.',
        url='http://oalib.hlsresearch.com/AcousticsToolbox/',
        commercial_use=True,      # GPL permits commercial use (copyleft applies)
        redistributable=True,
        note='Bellhop (Fortran backend), Kraken, Scooter, SPARC and Bounce. '
             'GPL-3.0 copyleft: derivatives that redistribute must stay GPL.',
        engine_citations=(
            ('Bellhop',
             'M. B. Porter, "The BELLHOP Manual and User\'s Guide: '
             'Preliminary Draft", Heat, Light, and Sound Research, Inc., '
             'La Jolla, CA (2011); Acoustics Toolbox.'),
            ('Kraken',
             'M. B. Porter, "The KRAKEN Normal Mode Program", '
             'SACLANTCEN SM-245 (1991); Acoustics Toolbox.'),
            ('SPARC',
             'M. B. Porter, "The time-marched fast-field program (FFP) for '
             'modeling acoustic pulse propagation", J. Acoust. Soc. Am. 87, '
             '2013-2023 (1990); Acoustics Toolbox.'),
            ('Scooter',
             'F. B. Jensen, W. A. Kuperman, M. B. Porter and H. Schmidt, '
             'Computational Ocean Acoustics, 2nd ed., Springer (2011), '
             'ch. 4 (wavenumber integration); Acoustics Toolbox (SCOOTER), '
             'M. B. Porter.'),
            ('Bounce',
             'M. B. Porter, Acoustics Toolbox (BOUNCE, plane-wave '
             'reflection coefficients of a layered seabed), '
             'http://oalib.hlsresearch.com/AcousticsToolbox/.'),
        ),
    ),
    'bellhopcxx': ModelProvenance(
        id='bellhopcxx',
        name='bellhopcxx / bellhopcuda',
        authors='Marine Physical Lab at Scripps Oceanography (The Regents '
                'of the University of California)',
        license='GPL-3.0-or-later',
        citation='bellhopcxx / bellhopcuda, C++/CUDA port of BELLHOP / '
                 'BELLHOP3D, Copyright (C) 2021-2026 The Regents of the '
                 'University of California, Marine Physical Lab at Scripps '
                 'Oceanography; based on BELLHOP / BELLHOP3D, Copyright (C) '
                 '1983-2022 Michael B. Porter.',
        url='https://github.com/A-New-BellHope/bellhopcuda',
        commercial_use=True,      # GPL permits commercial use (copyleft applies)
        redistributable=True,
        note='The C++ and CUDA ports of Bellhop, which Bellhop auto-selects '
             'ahead of the Fortran binary (CUDA > C++ > Fortran). A different '
             'codebase and a different copyright holder from the Acoustics '
             'Toolbox Fortran Bellhop, under the same GPL-3.0 copyleft.',
    ),
    'oases': ModelProvenance(
        id='oases',
        name='OASES',
        authors='Henrik Schmidt (Massachusetts Institute of Technology)',
        license='Academic — no formal open licence; not redistributable',
        citation='H. Schmidt, OASES Version 3.1 User Guide and Reference '
                 'Manual, Dept. of Ocean Engineering, MIT.',
        url='https://acoustics.mit.edu/faculty/henrik/oases.html',
        commercial_use=False,
        redistributable=False,
        note='OAST / OASN / OASR / OASP / OASS / OASSP. Academic licence: '
             'the user installs it separately (install.sh --oases yes); '
             'uacpy never bundles or redistributes the binary. Verify terms '
             'before commercial use.',
    ),
    'collins_ram': ModelProvenance(
        id='collins_ram',
        name="Collins' RAM parabolic-equation family",
        authors='Michael D. Collins (US Naval Research Laboratory); vendored '
                'backends by B. D. Dushaw (mpiramS) and D. C. Calvo / '
                'S. Guelton (RAMSurf)',
        license='Public domain (Collins/NRL, US Government work); backends: '
                'mpiramS CC-BY-4.0, RAMSurf tree (rams0.5, ramsurf1.5) BSD',
        citation='M. D. Collins, "A split-step Pade solution for the parabolic '
                 'equation method", J. Acoust. Soc. Am. 93, 1736-1742 (1993).',
        url='https://oalib-acoustics.org/',
        commercial_use=True,
        redistributable=True,
        note='RAM dispatches at run() time to ramgeo (public domain), '
             'rams0.5 / ramsurf1.5 (BSD) or mpiramS (CC-BY-4.0); see '
             'third_party/mpiramS/LICENSE, third_party/ramsurf/LICENSE '
             '(which covers rams0.5 and ramsurf1.5) and '
             'third_party/ramgeo/README.md (ramgeo ships no LICENSE file).',
    ),
}


def model_provenance(source_id: Optional[str]) -> Optional[ModelProvenance]:
    """The :class:`ModelProvenance` for ``source_id``, or ``None`` if unset.

    An id absent from :data:`MODEL_PROVENANCE` raises
    :class:`~uacpy.core.exceptions.ConfigurationError` naming it and the
    catalogued ids.
    """
    if source_id is None:
        return None
    try:
        return MODEL_PROVENANCE[source_id]
    except KeyError:
        raise ConfigurationError(
            f"Unknown model source id {source_id!r}. Catalogued ids: "
            f"{', '.join(sorted(MODEL_PROVENANCE))}."
        ) from None
