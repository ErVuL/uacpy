"""Catalogue of the external data sources, their licences and citations.

Single source of truth for provenance/attribution across the data layer (the
README licensing table mirrors this). :func:`uacpy.data.fetch_environment`
records the sources it actually used on the returned ``Environment`` as
``env.data_sources``; :func:`citations` renders the required attribution text
for an environment, a list of source ids, or the whole catalogue.
"""

from dataclasses import dataclass
from typing import Dict, List

from uacpy.core.exceptions import ConfigurationError

__all__ = ['DataSource', 'SOURCES', 'citations']


@dataclass(frozen=True)
class DataSource:
    """Provenance + licence metadata for one external dataset."""
    id: str
    name: str
    used_for: str
    license: str
    attribution: str
    citation: str
    url: str
    commercial_use: bool


SOURCES: Dict[str, DataSource] = {
    'gebco': DataSource(
        id='gebco',
        name='GEBCO grid (served via OpenTopoData)',
        used_for='bathymetry',
        license='Public domain (attribution requested)',
        attribution='GEBCO Compilation Group, GEBCO Grid',
        citation='GEBCO Compilation Group, GEBCO Grid — cite the grid DOI for '
                 'the vintage used (GEBCO 2025 offline; gebco.net).',
        url='https://www.gebco.net/',
        commercial_use=True,
    ),
    'gmrt': DataSource(
        id='gmrt',
        name='Global Multi-Resolution Topography (GMRT) synthesis',
        used_for='bathymetry (multibeam, higher-res)',
        license='CC-BY 4.0',
        attribution='Global Multi-Resolution Topography (GMRT) Synthesis '
                    '(Ryan et al. 2009)',
        citation='Ryan, W.B.F., et al. (2009). Global Multi-Resolution '
                 'Topography synthesis. Geochem. Geophys. Geosyst. 10, Q03014. '
                 'doi:10.1029/2008GC002332',
        url='https://www.gmrt.org/',
        commercial_use=True,
    ),
    'woa23': DataSource(
        id='woa23',
        name='World Ocean Atlas 2023 (NOAA NCEI)',
        used_for='sound speed (climatology), absorption',
        license='U.S. Government work — public domain',
        attribution='NOAA World Ocean Atlas 2023 (NCEI)',
        citation='Reagan, J.R., et al. (2024). World Ocean Atlas 2023. '
                 'NOAA National Centers for Environmental Information.',
        url='https://www.ncei.noaa.gov/products/world-ocean-atlas',
        commercial_use=True,
    ),
    'argo': DataSource(
        id='argo',
        name='Argo float profiles (Ifremer ERDDAP)',
        used_for='sound speed (real in-situ profiles)',
        license='Free and unrestricted (Argo data policy)',
        attribution='These data were collected and made freely available by the '
                    'International Argo Program and the national programmes that '
                    'contribute to it (https://argo.ucsd.edu).',
        citation='Argo (2024). Argo float data and metadata from Global Data '
                 'Assembly Centre (Argo GDAC). SEANOE. doi:10.17882/42182.',
        url='https://argo.ucsd.edu/',
        commercial_use=True,
    ),
    'copernicus': DataSource(
        id='copernicus',
        name='Copernicus Marine Service',
        used_for='sound speed (operational)',
        license='Copernicus Marine License (free; commercial use allowed)',
        attribution='Generated using E.U. Copernicus Marine Service '
                    'Information; <product DOI>',
        citation='E.U. Copernicus Marine Service Information; cite the '
                 'product DOI from the Copernicus Marine catalogue.',
        url='https://marine.copernicus.eu/',
        commercial_use=True,
    ),
    'seaice': DataSource(
        id='seaice',
        name='NSIDC Sea Ice Index sea-ice concentration (NOAA@NSIDC)',
        used_for='sea-ice concentration (surface; monthly climatology)',
        license='U.S. Government work — public domain',
        attribution='Sea ice: NSIDC Sea Ice Index (G02135), NOAA@NSIDC',
        citation='Fetterer, F., Knowles, K., Meier, W.N., Savoie, M. & '
                 'Windnagel, A.K. (2017, updated). Sea Ice Index. NSIDC. '
                 'doi:10.7265/N5K072F8.',
        url='https://nsidc.org/data/g02135',
        commercial_use=True,
    ),
    'emodnet': DataSource(
        id='emodnet',
        name='EMODnet Geology — seabed substrate',
        used_for='sediment (European seas)',
        license='CC-BY 4.0',
        attribution='EMODnet Geology seabed substrate '
                    '(emodnet.ec.europa.eu), CC-BY 4.0',
        citation='EMODnet Geology seabed substrate (1:1M).',
        url='https://emodnet.ec.europa.eu/en/geology',
        commercial_use=True,
    ),
    'globsed': DataSource(
        id='globsed',
        name='GlobSed total sediment thickness (NOAA NCEI)',
        used_for='sediment thickness (low-frequency seabed)',
        license='U.S. Government work — public domain',
        attribution='Sediment thickness: GlobSed v3 (NOAA NCEI)',
        citation='Straume, E.O., et al. (2019). GlobSed: Updated total sediment '
                 'thickness in the world\'s oceans. Geochem. Geophys. Geosyst. '
                 '20, 1756-1772. doi:10.1029/2018GC008115',
        url='https://www.ncei.noaa.gov/products/total-sediment-thickness-oceans-seas',
        commercial_use=True,
    ),
    'crust1': DataSource(
        id='crust1',
        name='CRUST1.0 global crustal model (UCSD)',
        used_for='layered seabed geoacoustics (Vp/Vs/density)',
        license='No formal licence — cite Laske et al. 2013; verify terms '
                'before commercial use',
        attribution='Crustal structure: CRUST1.0 (Laske, Masters, Ma & Pasyanos '
                    '2013)',
        citation='Laske, G., Masters, G., Ma, Z. & Pasyanos, M. (2013). Update '
                 'on CRUST1.0 — a 1-degree global model of Earth\'s crust. '
                 'Geophys. Res. Abstracts 15, EGU2013-2658.',
        url='https://igppweb.ucsd.edu/~gabi/crust1.html',
        commercial_use=False,
    ),
    'diesing': DataSource(
        id='diesing',
        name='Diesing 2020 global deep-sea seafloor lithology',
        used_for='sediment (global deep-sea, measured/modelled)',
        license='CC-BY 4.0',
        attribution='Seafloor lithology: Diesing (2020), CC-BY 4.0',
        citation='Diesing, M. (2020). Deep-sea sediments of the global ocean. '
                 'Earth Syst. Sci. Data 12, 3367-3381. '
                 'doi:10.5194/essd-12-3367-2020. Data: PANGAEA '
                 'doi:10.1594/PANGAEA.911692.',
        url='https://doi.org/10.1594/PANGAEA.911692',
        commercial_use=True,
    ),
    'pelagic': DataSource(
        id='pelagic',
        name='Pelagic sediment model (depth/latitude classifier)',
        used_for='sediment (global open-ocean fallback, modelled)',
        license='Public domain (first-principles model)',
        attribution='Open-ocean sediment: pelagic depth/latitude model '
                    '(after Diesing 2020 / Berger 1974)',
        citation='Diesing, M. (2020). Deep-sea sediments of the global ocean. '
                 'Earth Syst. Sci. Data 12, 3367-3381. '
                 'doi:10.5194/essd-12-3367-2020 (CC-BY 4.0). '
                 'After Berger, W.H. (1974), Deep-sea sedimentation.',
        url='https://essd.copernicus.org/articles/12/3367/2020/',
        commercial_use=True,
    ),
    'grainsize': DataSource(
        id='grainsize',
        name='NCEI Seafloor Sediment Grain-Size Database (NOAA, G00127)',
        used_for='sediment (global, local samples)',
        license='U.S. Government work — public domain',
        attribution='NOAA NCEI Seafloor Sediment Grain-Size Database (G00127)',
        citation='National Geophysical Data Center (1976): The NGDC Seafloor '
                 'Sediment Grain Size Database. NOAA NCEI. doi:10.7289/V5G44N6W',
        url='https://www.ngdc.noaa.gov/mgg/geology/data/g00127/',
        commercial_use=True,
    ),
}


def _resolve(obj) -> List[DataSource]:
    if obj is None:
        return list(SOURCES.values())
    if hasattr(obj, 'data_sources'):       # an Environment from fetch_environment
        return list(obj.data_sources)
    if isinstance(obj, (str, DataSource)):  # a single id / source, not an iterable
        obj = [obj]
    resolved = []
    for s in obj:
        if isinstance(s, DataSource):
            resolved.append(s)
        elif s in SOURCES:
            resolved.append(SOURCES[s])
        else:
            raise ConfigurationError(
                f"citations: unknown source id {s!r}; valid ids: "
                f"{sorted(SOURCES)}"
            )
    return resolved


def citations(obj=None) -> str:
    """Render the required attribution / citation text.

    Parameters
    ----------
    obj : optional
        ``None`` → the whole catalogue; an ``Environment`` returned by
        :func:`fetch_environment` → the sources it used (``env.data_sources``);
        or an iterable of source ids / :class:`DataSource`.

    Returns
    -------
    str
        One block per source (name, licence, attribution, citation), with a
        flag where commercial use is not confirmed.
    """
    blocks = []
    for s in _resolve(obj):
        lines = [f"{s.name}  [{s.license}]",
                 f"  Attribution: {s.attribution}",
                 f"  Cite:        {s.citation}"]
        if not s.commercial_use:
            lines.append("  ⚠ Commercial use not confirmed — verify licence terms")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
