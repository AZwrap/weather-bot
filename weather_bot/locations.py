"""Polymarket weather market locations and the active market list.

Two structures live here:

  * `STATIONS_BY_ID` — every distinct observation station that resolves at
    least one Polymarket market. Coordinates are the station's actual
    location (not the city centre); the unit is what Polymarket uses for
    that station's markets.

  * `MARKETS` — every active market on Polymarket, as (station_id, target)
    pairs where `target` is `"highest"` or `"lowest"`. 59 rows (2026-05-07).

Source: spreadsheet provided by the user, cross-checked against
polymarket.com/markets/weather event metadata. City spellings normalised
(spreadsheet contained typos like "Shangha", "hong Kong").

Bucket widths on Polymarket: **1 °C for °C markets, 2 °F for °F markets.**
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .forecast.fetcher import Location
from .units import Unit

MarketTarget = Literal["highest", "lowest"]


@dataclass(frozen=True)
class Station:
    name: str            # city display name (normalised)
    station_id: str      # ICAO airport code, or station ID for non-airport (e.g. "HKO")
    station_label: str   # human-readable station name as it appears on Polymarket
    latitude: float
    longitude: float
    timezone: str
    unit: Unit           # market resolution unit (1°C or 2°F bucket)

    def to_location(self) -> Location:
        return Location(
            name=self.name,
            latitude=self.latitude,
            longitude=self.longitude,
            timezone=self.timezone,
        )

    @property
    def bucket_width(self) -> float:
        return 1.0 if self.unit == "C" else 2.0


# All distinct stations covered by Polymarket weather markets. Coordinates
# are the actual airport / observatory positions, accurate to ≤1 km.
STATIONS_BY_ID: dict[str, Station] = {
    s.station_id: s
    for s in [
        # United States (°F, 2°F buckets)
        Station("Denver",        "KBKF", "Buckley Space Force Base",                39.7017, -104.7517, "America/Denver",   "F"),
        Station("Chicago",       "KORD", "Chicago O'Hare International Airport",    41.9742,  -87.9073, "America/Chicago",  "F"),
        Station("Miami",         "KMIA", "Miami International Airport",             25.7959,  -80.2870, "America/New_York", "F"),
        Station("Los Angeles",   "KLAX", "Los Angeles International Airport",       33.9425, -118.4081, "America/Los_Angeles","F"),
        Station("Dallas",        "KDAL", "Dallas Love Field",                       32.8471,  -96.8517, "America/Chicago",  "F"),
        Station("NYC",           "KLGA", "LaGuardia Airport",                       40.7772,  -73.8726, "America/New_York", "F"),
        Station("Houston",       "KHOU", "William P. Hobby Airport",                29.6454,  -95.2789, "America/Chicago",  "F"),
        Station("Seattle",       "KSEA", "Seattle-Tacoma International Airport",    47.4502, -122.3088, "America/Los_Angeles","F"),
        Station("Austin",        "KAUS", "Austin-Bergstrom International Airport",  30.1944,  -97.6700, "America/Chicago",  "F"),
        Station("Atlanta",       "KATL", "Hartsfield-Jackson International Airport",33.6407,  -84.4277, "America/New_York", "F"),
        Station("San Francisco", "KSFO", "San Francisco International Airport",     37.6189, -122.3750, "America/Los_Angeles","F"),

        # Europe (°C, 1°C buckets)
        Station("London",        "EGLC", "London City Airport",                     51.5050,    0.0553, "Europe/London",    "C"),
        Station("Madrid",        "LEMD", "Adolfo Suárez Madrid-Barajas Airport",    40.4936,   -3.5668, "Europe/Madrid",    "C"),
        Station("Paris",         "LFPB", "Paris-Le Bourget Airport",                48.9694,    2.4414, "Europe/Paris",     "C"),
        Station("Warsaw",        "EPWA", "Warsaw Chopin Airport",                   52.1657,   20.9671, "Europe/Warsaw",    "C"),
        Station("Munich",        "EDDM", "Munich Airport",                          48.3538,   11.7861, "Europe/Berlin",    "C"),
        Station("Helsinki",      "EFHK", "Helsinki Vantaa Airport",                 60.3172,   24.9633, "Europe/Helsinki",  "C"),
        Station("Amsterdam",     "EHAM", "Amsterdam Airport Schiphol",              52.3086,    4.7639, "Europe/Amsterdam", "C"),
        Station("Milan",         "LIMC", "Milan Malpensa Airport",                  45.6306,    8.7281, "Europe/Rome",      "C"),
        Station("Moscow",        "UUWW", "Vnukovo International Airport",           55.5915,   37.2615, "Europe/Moscow",    "C"),
        Station("Istanbul",      "LTFM", "Istanbul Airport",                        41.2606,   28.7427, "Europe/Istanbul",  "C"),
        Station("Ankara",        "LTAC", "Esenboğa International Airport",          40.1281,   32.9951, "Europe/Istanbul",  "C"),

        # Asia (°C, 1°C buckets)
        Station("Tokyo",         "RJTT", "Tokyo Haneda Airport",                    35.5494,  139.7798, "Asia/Tokyo",       "C"),
        Station("Beijing",       "ZBAA", "Beijing Capital International Airport",   40.0801,  116.5846, "Asia/Shanghai",    "C"),
        Station("Shanghai",      "ZSPD", "Shanghai Pudong International Airport",   31.1443,  121.8083, "Asia/Shanghai",    "C"),
        Station("Wuhan",         "ZHHH", "Wuhan Tianhe International Airport",      30.7838,  114.2081, "Asia/Shanghai",    "C"),
        Station("Chengdu",       "ZUUU", "Chengdu Shuangliu International Airport", 30.5785,  103.9472, "Asia/Shanghai",    "C"),
        Station("Chongqing",     "ZUCK", "Chongqing Jiangbei International Airport",29.7192,  106.6417, "Asia/Shanghai",    "C"),
        Station("Guangzhou",     "ZGGG", "Guangzhou Baiyun International Airport",  23.3924,  113.2988, "Asia/Shanghai",    "C"),
        Station("Shenzhen",      "ZGSZ", "Shenzhen Bao'an International Airport",   22.6393,  113.8108, "Asia/Shanghai",    "C"),
        Station("Qingdao",       "ZSQD", "Qingdao Jiaodong International Airport",  36.2611,  120.0867, "Asia/Shanghai",    "C"),
        # Hong Kong (HKO) excluded — Hong Kong Observatory is not on the
        # Iowa State ASOS network. Without METAR truth we'd be trading on
        # ERA5 reanalysis, which can disagree with HKO's own data by 1–2°C
        # at coastal subtropical stations. Re-add only with a dedicated
        # HKO scraper or paid station-level feed.
        Station("Taipei",        "RCSS", "Taipei Songshan Airport",                 25.0697,  121.5526, "Asia/Taipei",      "C"),
        Station("Seoul",         "RKSI", "Incheon International Airport",           37.4602,  126.4407, "Asia/Seoul",       "C"),
        Station("Busan",         "RKPK", "Gimhae International Airport",            35.1795,  128.9382, "Asia/Seoul",       "C"),
        Station("Singapore",     "WSSS", "Singapore Changi Airport",                 1.3592,  103.9894, "Asia/Singapore",   "C"),
        Station("Kuala Lumpur",  "WMKK", "Kuala Lumpur International Airport",       2.7456,  101.7099, "Asia/Kuala_Lumpur","C"),
        Station("Jakarta",       "WIHH", "Halim Perdanakusuma International Airport",-6.2667, 106.8911, "Asia/Jakarta",     "C"),
        Station("Manila",        "RPLL", "Ninoy Aquino International Airport",      14.5086,  121.0194, "Asia/Manila",      "C"),
        Station("Lucknow",       "VILK", "Chaudhary Charan Singh International Airport", 26.7606, 80.8893, "Asia/Kolkata",  "C"),
        # NOTE: Polymarket's market metadata is INCONSISTENT for Karachi.
        # The event `description` says "Masroor Airbase Station" (twice).
        # The `resolutionSource` URL points at OPKC (Jinnah Intl). The two
        # stations are ~30 km apart. UMA resolvers read the description first,
        # so OPMR is the resolving station; the URL is a misconfigured link.
        # Excluded from MARKETS below (2026-05-11): Iowa State ASOS does not
        # publish OPMR/OPKC data on a usable schedule — observed 100% of
        # records pending on May 8/9/10 2026. Without a truth source we
        # cannot calibrate, so we don't trade Karachi. Registry entry kept
        # for historical record interpretation.
        Station("Karachi",       "OPMR", "Masroor Airbase",                         24.8936,   66.9388, "Asia/Karachi",     "C"),
        Station("Tel Aviv",      "LLBG", "Ben Gurion International Airport",        32.0114,   34.8866, "Asia/Jerusalem",   "C"),
        Station("Jeddah",        "OEJN", "King Abdulaziz International Airport",    21.6797,   39.1565, "Asia/Riyadh",      "C"),

        # Oceania
        Station("Wellington",    "NZWN", "Wellington International Airport",       -41.3272,  174.8050, "Pacific/Auckland", "C"),

        # Africa
        Station("Lagos",         "DNMM", "Murtala Muhammad International Airport",   6.5774,    3.3210, "Africa/Lagos",     "C"),
        Station("Cape Town",     "FACT", "Cape Town International Airport",        -33.9648,   18.6017, "Africa/Johannesburg","C"),

        # Latin America
        Station("Mexico City",   "MMMX", "Benito Juárez International Airport",     19.4361,  -99.0719, "America/Mexico_City","C"),
        Station("Panama City",   "MPMG", "Marcos A. Gelabert International Airport", 8.9750,  -79.5556, "America/Panama",   "C"),
        Station("Buenos Aires",  "SAEZ", "Ministro Pistarini International Airport",-34.8222,  -58.5358, "America/Argentina/Buenos_Aires","C"),
        Station("Sao Paulo",     "SBGR", "São Paulo-Guarulhos International Airport",-23.4356, -46.4731, "America/Sao_Paulo","C"),

        # North America (Canada)
        Station("Toronto",       "CYYZ", "Toronto Pearson International Airport",   43.6777,  -79.6248, "America/Toronto",  "C"),
    ]
}


# All currently-listed Polymarket weather markets. 59 rows.
MARKETS: list[tuple[str, MarketTarget]] = [
    ("KBKF", "highest"),  # Denver
    ("RKSI", "highest"),  # Seoul
    ("WSSS", "highest"),  # Singapore
    ("EGLC", "highest"),  # London
    ("KORD", "highest"),  # Chicago
    ("RJTT", "highest"),  # Tokyo
    ("LEMD", "highest"),  # Madrid
    ("ZSPD", "highest"),  # Shanghai
    ("LFPB", "highest"),  # Paris
    ("NZWN", "highest"),  # Wellington
    ("LLBG", "highest"),  # Tel Aviv
    ("ZBAA", "highest"),  # Beijing
    ("WIHH", "highest"),  # Jakarta
    ("EPWA", "highest"),  # Warsaw
    ("LTFM", "highest"),  # Istanbul
    ("UUWW", "highest"),  # Moscow
    ("KMIA", "highest"),  # Miami
    ("MPMG", "highest"),  # Panama City
    ("KLAX", "highest"),  # Los Angeles
    ("ZGSZ", "highest"),  # Shenzhen
    ("LTAC", "highest"),  # Ankara
    ("KDAL", "highest"),  # Dallas
    ("RCSS", "highest"),  # Taipei
    ("MMMX", "highest"),  # Mexico City
    ("KLGA", "highest"),  # NYC
    ("WMKK", "highest"),  # Kuala Lumpur
    ("ZHHH", "highest"),  # Wuhan
    ("ZUUU", "highest"),  # Chengdu
    ("RKSI", "lowest"),   # Seoul
    ("ZUCK", "highest"),  # Chongqing
    ("VILK", "highest"),  # Lucknow
    ("KHOU", "highest"),  # Houston
    ("KSEA", "highest"),  # Seattle
    ("SAEZ", "highest"),  # Buenos Aires
    ("OEJN", "highest"),  # Jeddah
    ("ZGGG", "highest"),  # Guangzhou
    ("EDDM", "highest"),  # Munich
    ("EFHK", "highest"),  # Helsinki
    ("SBGR", "highest"),  # Sao Paulo
    ("RJTT", "lowest"),   # Tokyo
    ("DNMM", "highest"),  # Lagos
    ("KAUS", "highest"),  # Austin
    ("ZSPD", "lowest"),   # Shanghai
    ("KLGA", "lowest"),   # NYC
    ("KMIA", "lowest"),   # Miami
    ("RPLL", "highest"),  # Manila
    ("EHAM", "highest"),  # Amsterdam
    ("RKPK", "highest"),  # Busan
    ("KATL", "highest"),  # Atlanta
    ("LIMC", "highest"),  # Milan
    ("CYYZ", "highest"),  # Toronto
    ("ZSQD", "highest"),  # Qingdao
    ("KSFO", "highest"),  # San Francisco
    ("FACT", "highest"),  # Cape Town
    ("LFPB", "lowest"),   # Paris
    ("EGLC", "lowest"),   # London
]


# Stations that have at least one max market (for backtest scope)
STATIONS_WITH_MAX = sorted(
    {sid for sid, t in MARKETS if t == "highest"}
)
STATIONS_WITH_MIN = sorted(
    {sid for sid, t in MARKETS if t == "lowest"}
)


# Backwards compatibility for the demo script
STATIONS: list[Station] = list(STATIONS_BY_ID.values())

# City-name index for resolving events that have no resolutionSource URL
# (Tel Aviv, Moscow, Istanbul, HKO — those use NOAA/HKO sources rather than
# Wunderground). Lookup is case-insensitive.
STATION_BY_CITY: dict[str, Station] = {
    s.name.lower(): s for s in STATIONS_BY_ID.values()
}


# Cities/stations Polymarket lists but we have intentionally chosen NOT to
# trade. The unmatched-events notifier suppresses these to avoid perpetual
# WARN alerts on cities we've already decided to skip.
#
# To re-include one in the future: remove it from these sets, add a Station
# entry, add it to MARKETS, run train_bias.py, restart the dashboard.
EXCLUDED_CITIES_LOWER: frozenset[str] = frozenset({
    "hong kong",  # HKO Observatory not on Iowa State ASOS network → would
                  # fall back to ERA5, which disagrees by 1–2°C. Re-add only
                  # with a dedicated HKO scraper or paid station-level feed.
})

EXCLUDED_STATION_IDS: frozenset[str] = frozenset({
    "HKO",  # Same reasoning. Excluded by ICAO too in case the URL is parsed
            # differently than the city name in some future Polymarket event.
})

