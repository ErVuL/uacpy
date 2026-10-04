"""The provenance records of the data layers a carrier holds, and the
catalogue they name.

:class:`DataSource` is one **catalogue entry** per dataset (GEBCO, WOA23, …):
its identity, licence, attribution and citation text, shared by every fetch of
that dataset. :class:`DataProvenance` is one **fetch**: a ``DataSource``
(``.source``) plus the actual date and coordinates that fetch returned.
:data:`SOURCES` is the catalogue, the single source of truth (the README
licensing table mirrors it).

Pure records, in core because a carrier's ``data_sources`` holds them: a saved
carrier names each record's source by id and is rebuilt from this catalogue,
so a loaded environment holds the very records a fetch attaches.
:mod:`uacpy.data.sources` is their public home in the data layer, with
:func:`~uacpy.data.sources.citations`.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from uacpy.core._repr import FieldsRepr

__all__ = ['DataSource', 'DataProvenance', 'SOURCES', 'POINT_KINDS',
           'fetch_summary_line', 'point_in_words']

#: What a record's ``data_point`` can be: the centre of a grid ``'cell'``,
#: the position of an Argo ``'cast'``, or of a seabed ``'sample'``.
POINT_KINDS = ('cell', 'cast', 'sample')


@dataclass(frozen=True)
class DataSource(FieldsRepr):
    """Provenance + licence metadata for one external dataset.

    Saves and loads through :meth:`to_dict` / :meth:`from_dict`.
    """
    id: str
    name: str
    used_for: str
    license: str
    attribution: str
    citation: str
    url: str
    commercial_use: bool

    _REPR_FIELDS = ('id', 'name', 'license')

    def to_dict(self) -> Dict[str, Any]:
        """Every field as plain data; :meth:`from_dict` reads it back."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DataSource':
        """The record :meth:`to_dict` wrote: the catalogue's own entry
        when it is one of :data:`SOURCES` unchanged, as a copy of one is,
        else a new record.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output.
        """
        record = cls(**dict(d))
        known = SOURCES.get(record.id)
        return known if known == record else record

    # A catalogue entry is one shared, immutable record: a copy of an
    # environment keeps pointing at it, so a loaded or copied record is the
    # catalogue's own object.
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True)
class DataProvenance(FieldsRepr):
    """One fetched data layer's provenance: a reference to the catalogue
    :class:`DataSource` (``.source``) plus the **actual** date and coordinates
    the returned data corresponds to.

    The actual values can differ from what was requested — a WOA23 climatology
    snaps to a grid cell (and to a month, not a day), a daily Copernicus mean
    snaps to the nearest day, an Argo float is the nearest cast (tens of km and
    days away). Recording them closes the "did I actually get data where/when I
    asked?" gap.

    The two levels are kept distinct: read the dataset's identity/licence/
    citation through ``prov.source`` (a shared catalogue entry), and this
    fetch's specifics off ``prov`` directly. Carriers and environments hold
    these uniformly as ``data_sources`` tuples."""
    source: DataSource
    data_date: Optional[str] = None                    # actual date/period represented
    data_point: Optional[Tuple[float, float]] = None   # actual (lat, lon) returned
    requested_point: Optional[Tuple[float, float]] = None
    requested_date: Optional[str] = None
    #: The product or dataset id read, where one catalogue source serves
    #: several (Copernicus Marine: reanalysis vs analysis/forecast; Argo:
    #: real-time vs adjusted values), so the attribution can name the
    #: product whose DOI the licence asks for and the values that were read.
    product: Optional[str] = None
    #: What ``data_point`` is, one of :data:`POINT_KINDS`; ``None`` when the
    #: value was read at the point itself (a point query, a map polygon, a
    #: model) or the kind is not known.
    point_kind: Optional[str] = None
    #: The grid spacing (degrees) of a ``'cell'``, when the grid is regular
    #: in latitude and longitude; ``None`` otherwise.
    cell_size_deg: Optional[float] = None
    #: ``True`` when the cell read is not the requested point's own cell
    #: (that one held no data, so the nearest wet one was read), ``False``
    #: when it is, ``None`` when the fetcher cannot tell.
    from_neighbour_cell: Optional[bool] = None
    #: The along-track range (m) of the column this record supplied, on a
    #: range-dependent carrier; ``None`` for a single point.
    range_m: Optional[float] = None
    #: Depth (m) below which the column holds another period than
    #: ``data_date`` — a WOA23 monthly column (which ends at 1500 m) continued
    #: by the annual mean; ``None`` when the whole column is one period.
    split_depth_m: Optional[float] = None
    #: The period below ``split_depth_m`` (``'annual mean'``).
    period_below: Optional[str] = None

    def __post_init__(self):
        if self.point_kind is not None and self.point_kind not in POINT_KINDS:
            from uacpy.core.exceptions import ConfigurationError
            raise ConfigurationError(
                f"DataProvenance: point_kind must be one of {POINT_KINDS} or "
                f"None; got {self.point_kind!r}.")

    def to_dict(self) -> Dict[str, Any]:
        """This record as plain types: the catalogue ``source`` by its id,
        the points as ``[lat, lon]`` lists; :meth:`from_dict` reads it
        back."""
        return {'source': self.source.id, 'data_date': self.data_date,
                'data_point': (None if self.data_point is None
                               else [float(v) for v in self.data_point]),
                'requested_point': (None if self.requested_point is None
                                    else [float(v)
                                          for v in self.requested_point]),
                'requested_date': self.requested_date,
                'product': self.product,
                'point_kind': self.point_kind,
                'cell_size_deg': (None if self.cell_size_deg is None
                                  else float(self.cell_size_deg)),
                'from_neighbour_cell': (None if self.from_neighbour_cell is None
                                        else bool(self.from_neighbour_cell)),
                'range_m': None if self.range_m is None else float(self.range_m),
                'split_depth_m': (None if self.split_depth_m is None
                                  else float(self.split_depth_m)),
                'period_below': self.period_below}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'DataProvenance':
        """The record :meth:`to_dict` wrote, its source looked up in the
        catalogue :data:`SOURCES` by id."""
        fields = {k: v for k, v in dict(d).items() if v is not None}
        source = SOURCES[fields.pop('source')]
        for key in ('data_point', 'requested_point'):
            if key in fields:
                fields[key] = tuple(float(v) for v in fields[key])
        for key in ('cell_size_deg', 'range_m', 'split_depth_m'):
            if key in fields:
                fields[key] = float(fields[key])
        if 'from_neighbour_cell' in fields:
            fields['from_neighbour_cell'] = bool(fields['from_neighbour_cell'])
        return cls(source=source, **fields)

    @property
    def offset_km(self) -> Optional[float]:
        """Great-circle distance (km) from the requested point to the actual
        ``data_point`` — derived, not stored. ``None`` if either is unknown."""
        if self.data_point is None or self.requested_point is None:
            return None
        from uacpy.core.geo import great_circle_km
        la, lo = self.requested_point
        da, do = self.data_point
        return float(great_circle_km(la, lo, da, do))

    @property
    def offset_days(self) -> Optional[int]:
        """Days from the data's date to the requested date (requested minus
        data; positive when the data are older) — derived, not stored.
        ``None`` when either date is unknown or the data are a climatology
        period rather than a day."""
        requested, data = _day(self.requested_date), _day(self.data_date)
        if requested is None or data is None:
            return None
        return (requested - data).days

    def describe_point(self) -> Optional[str]:
        """``data_point`` in words: ``'1° cell centred 42.50 N, 4.50 E'``,
        ``'nearest wet 1° cell, centred 42.50 N, 4.50 E'``, ``'cast at …'``,
        ``'sample at …'``, or the bare position when the kind is not
        known; ``None`` without a data point."""
        if self.data_point is None:
            return None
        where = point_in_words(*self.data_point)
        if self.point_kind == 'cell':
            size = ('' if self.cell_size_deg is None
                    else f"{_degrees_in_words(self.cell_size_deg)} ")
            if self.from_neighbour_cell:
                return f"nearest wet {size}cell, centred {where}"
            return f"{size}cell centred {where}"
        if self.point_kind in ('cast', 'sample'):
            return f"{self.point_kind} at {where}"
        return where

    def _date_bits(self) -> list:
        bits = []
        if self.product is not None:
            bits.append(f"product {self.product}")
        if self.data_date is not None:
            days = self.offset_days
            if days is not None:
                req = (f", {abs(days)} day{'s' if abs(days) != 1 else ''} "
                       f"from requested {self.requested_date}")
            elif self.requested_date and self.requested_date != self.data_date:
                req = f", requested {self.requested_date}"
            else:
                req = ""
            split = ("" if self.split_depth_m is None else
                     f" above {self.split_depth_m:g} m, {self.period_below} "
                     f"below")
            bits.append(f"date {self.data_date}{split}{req}")
        return bits

    def _fetch_line(self) -> Optional[str]:
        """The ``Fetched:`` provenance line (date, the point the data came
        from in words, its offset), or ``None``."""
        bits = self._date_bits()
        if self.data_point is not None:
            off_km = self.offset_km
            off = (f", {off_km:.1f} km from requested"
                   if off_km is not None else "")
            prefix = {'cell': 'at the ', 'cast': '', 'sample': ''}.get(
                self.point_kind, 'at ')
            bits.append(f"{prefix}{self.describe_point()}{off}")
        return ("  Fetched:     " + "; ".join(bits)) if bits else None


def fetch_summary_line(records: Sequence[DataProvenance]) -> Optional[str]:
    """The ``Fetched:`` line of one source's records: the record's own line
    for one, else a summary — the dates, how many points (along the transect
    when they carry ``range_m``), the span of their offsets, and where the
    largest one is."""
    records = list(records)
    if len(records) == 1:
        return records[0]._fetch_line()
    bits = []
    for name in ('product', 'data_date'):
        values = list(dict.fromkeys(getattr(r, name) for r in records
                                    if getattr(r, name) is not None))
        if values:
            bits.append(f"{'product' if name == 'product' else 'date'} "
                        f"{', '.join(values)}")
    days = [abs(r.offset_days) for r in records if r.offset_days is not None]
    if days and bits:
        bits[-1] += f", {min(days)}–{max(days)} days from requested"
    located = [r for r in records if r.offset_km is not None]
    if located:
        offsets = [r.offset_km for r in located]
        far = located[offsets.index(max(offsets))]
        along = (" along the transect"
                 if all(r.range_m is not None for r in located) else "")
        at = (f"range {far.range_m / 1000.0:g} km: " if far.range_m is not None
              else "")
        if max(offsets) == 0.0:
            bits.append(f"{len(located)} points{along}, each read at its "
                        f"requested point")
        else:
            bits.append(f"{len(located)} points{along}, {min(offsets):.1f}–"
                        f"{max(offsets):.1f} km away (largest at {at}"
                        f"{far.describe_point()})")
    return ("  Fetched:     " + "; ".join(bits)) if bits else None


def _day(text):
    """The calendar day an ISO date (or datetime) string names, else
    ``None`` (a climatology period such as ``'month 07 (climatology)'``)."""
    if not text:
        return None
    import datetime
    try:
        return datetime.date.fromisoformat(str(text)[:10])
    except ValueError:
        return None


def point_in_words(lat: float, lon: float) -> str:
    """``(42.5, -4.5)`` as ``'42.50 N, 4.50 W'``."""
    return (f"{abs(lat):.2f} {'N' if lat >= 0 else 'S'}, "
            f"{abs(lon):.2f} {'E' if lon >= 0 else 'W'}")


def _degrees_in_words(deg: float) -> str:
    """A grid spacing: ``'1°'``, ``'0.25°'``, or ``'1/12°'`` for a spacing
    that is a whole fraction of a degree with no short decimal."""
    deg = float(deg)
    if round(deg, 4) == deg:
        return f"{deg:g}°"
    n = 1.0 / deg
    if abs(n - round(n)) < 1e-6:
        return f"1/{round(n)}°"
    return f"{deg:.4g}°"


SOURCES: Dict[str, DataSource] = {
    'gebco': DataSource(
        id='gebco',
        name='GEBCO grid',
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
    'emodnet_dtm': DataSource(
        id='emodnet_dtm',
        name='EMODnet Bathymetry DTM 2024',
        used_for='bathymetry (European seas + Caribbean, ~115 m)',
        license='CC-BY 4.0',
        attribution='EMODnet Bathymetry (emodnet.ec.europa.eu), CC-BY 4.0',
        citation='EMODnet Bathymetry Consortium (2024). EMODnet Digital '
                 'Bathymetry (DTM 2024). EMODnet Bathymetry.',
        url='https://emodnet.ec.europa.eu/en/bathymetry',
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
                    'Information',
        citation='E.U. Copernicus Marine Service Information; cite the DOI '
                 'the Copernicus Marine catalogue gives the product named in '
                 'the fetch record.',
        url='https://marine.copernicus.eu/',
        commercial_use=True,
    ),
    'glodap': DataSource(
        id='glodap',
        name='GLODAPv2.2016b Mapped Climatology (GLODAP)',
        used_for='seawater pH (absorption)',
        license='CC-BY 4.0',
        attribution='GLODAPv2.2016b Mapped Climatology (glodap.info), CC-BY 4.0',
        citation='Lauvset, S.K., et al. (2016). A new global interior ocean '
                 'mapped climatology: the 1x1 GLODAP version 2. Earth Syst. '
                 'Sci. Data 8, 325-340. doi:10.5194/essd-8-325-2016',
        url='https://glodap.info/index.php/mapped-data-product/',
        commercial_use=True,
    ),
    'copernicus_bgc': DataSource(
        id='copernicus_bgc',
        name='Copernicus Marine global biogeochemistry reanalysis (pH)',
        used_for='seawater pH (absorption; operational)',
        license='Copernicus Marine License (free; commercial use allowed)',
        attribution='pH: E.U. Copernicus Marine Service Information '
                    '(GLOBAL_MULTIYEAR_BGC_001_029)',
        citation='E.U. Copernicus Marine Service Information; '
                 'GLOBAL_MULTIYEAR_BGC_001_029. Cite the DOI the Copernicus '
                 'Marine catalogue gives the product named in the fetch '
                 'record.',
        url='https://marine.copernicus.eu/',
        commercial_use=True,
    ),
    'nbs': DataSource(
        id='nbs',
        name='NOAA/NCEI Blended Seawinds (NBS)',
        used_for='10 m wind speed (ambient noise, sea surface)',
        license='U.S. Government work — public domain',
        attribution='Wind: NOAA/NCEI Blended Seawinds (NBS)',
        citation='Saha, K. & Zhang, H.-M. (2022). Hurricane and typhoon storm '
                 'wind resolving NOAA NCEI Blended Sea Surface Wind (NBS) '
                 'product. Front. Mar. Sci. 9, 935549. '
                 'doi:10.3389/fmars.2022.935549 (NBS v2, the files read). '
                 'Method: Zhang, H.-M., Bates, J.J. & Reynolds, R.W. (2006). '
                 'Assessment of composite global sampling: Sea surface wind '
                 'speed. Geophys. Res. Lett. 33, L17714.',
        url='https://www.ncei.noaa.gov/products/blended-sea-winds',
        commercial_use=True,
    ),
    'ww3': DataSource(
        id='ww3',
        name='NOAA WaveWatch III',
        used_for='significant wave height (sea surface; recent)',
        license='U.S. Government work — public domain',
        attribution='Waves: NOAA WaveWatch III',
        citation='Tolman, H.L. (2009). User manual and system documentation of '
                 'WAVEWATCH III. NOAA/NWS/NCEP Technical Note 276.',
        url='https://polar.ncep.noaa.gov/waves/',
        commercial_use=True,
    ),
    'waverys': DataSource(
        id='waverys',
        name='Copernicus Marine WAVERYS global wave reanalysis',
        used_for='significant wave height (sea surface; reanalysis)',
        license='Copernicus Marine License (free; commercial use allowed)',
        attribution='Waves: E.U. Copernicus Marine Service Information '
                    '(WAVERYS)',
        citation='E.U. Copernicus Marine Service Information; GLOBAL_MULTIYEAR_'
                 'WAV_001_032 (WAVERYS). Cite the DOI the Copernicus Marine '
                 'catalogue gives the product named in the fetch record.',
        url='https://marine.copernicus.eu/',
        commercial_use=True,
    ),
    'seaice': DataSource(
        id='seaice',
        name='NSIDC Sea Ice Index sea-ice concentration (NOAA@NSIDC)',
        used_for='sea-ice concentration (surface; monthly climatology)',
        license='U.S. Government work — public domain',
        attribution='Sea ice: NSIDC Sea Ice Index (G02135 v4), NOAA@NSIDC',
        citation='Fetterer, F., Knowles, K., Meier, W. N., Savoie, M., '
                 'Windnagel, A. K. & Stafford, T. (2025). Sea Ice Index, '
                 'Version 4 (G02135). NSIDC. doi:10.7265/a98x-0f50.',
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
    'mars': DataSource(
        id='mars',
        name='AusSeabed Marine Sediments (MARS) database (Geoscience Australia)',
        used_for='sediment (Australian margin, point samples)',
        license='CC-BY 4.0',
        attribution='Sediment samples: AusSeabed MARS database (Geoscience '
                    'Australia), CC-BY 4.0',
        citation='Geoscience Australia (2020). Marine Sediments (MARS) '
                 'Database. AusSeabed data portal.',
        url='https://portal.ga.gov.au/persona/marine',
        commercial_use=True,
    ),
    'graw': DataSource(
        id='graw',
        name='Graw 2021 predicted global seabed bulk density (NRL)',
        used_for='seabed bulk density (measured-density bottom)',
        license='CC-BY 4.0',
        attribution='Seabed density: Graw, Wood & Phrampus (2021), CC-BY 4.0',
        citation='Graw, J.H., Wood, W.T. & Phrampus, B.J. (2021). Predicting '
                 'global marine sediment density using the random forest '
                 'regressor machine learning algorithm. J. Geophys. Res. Solid '
                 'Earth 126. Data: Zenodo doi:10.5281/zenodo.3762390.',
        url='https://doi.org/10.5281/zenodo.3762390',
        commercial_use=True,
    ),
    'deck41': DataSource(
        id='deck41',
        name='DECK41 Surficial Seafloor Sediment Description Database (NOAA)',
        used_for='sediment (global, dominant-lithology descriptions)',
        license='U.S. Government work — public domain',
        attribution='NOAA NCEI DECK41 Surficial Seafloor Sediment Descriptions',
        citation='Bershad, S. & Weiss, M. (1976). Deck41 Surficial Seafloor '
                 'Sediment Description Database. NOAA NCEI (G02094). '
                 'doi:10.7289/V5VD6WCZ',
        url='https://www.ngdc.noaa.gov/mgg/geology/deck41.html',
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
