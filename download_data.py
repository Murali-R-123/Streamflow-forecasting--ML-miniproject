"""Download the two raw datasets and build data/streamflow_data.csv.

Streamflow : USGS NWIS daily mean discharge (00060, cfs), site 02472000
             'Leaf River near Collins, MS'  (https://waterdata.usgs.gov/monitoring-location/USGS-02472000/)
Rainfall   : Daymet v4 daily precipitation (mm/day), single grid cell at the gauge location
             (https://daymet.ornl.gov/)  -- gridded, NOT a rain-gauge record.

Period: 2003-01-01 to 2012-12-31 (same years as the Stanford reference project).
Needs internet. If a download fails, see the 'Manual download' section of README.md.
"""
from pathlib import Path

import requests

from src.data_loader import build_master, parse_daymet_csv, parse_usgs_rdb

SITE, LAT, LON = "02472000", 31.7069444, -89.4069444   # coordinates from USGS site metadata
START, END = "2003-01-01", "2012-12-31"
RAW = Path("data/raw")

USGS_URL = ("https://waterservices.usgs.gov/nwis/dv/?format=rdb"
            f"&sites={SITE}&parameterCd=00060&statCd=00003&startDT={START}&endDT={END}")
DAYMET_URL = ("https://daymet.ornl.gov/single-pixel/api/data"
              f"?lat={LAT}&lon={LON}&vars=prcp&start={START}&end={END}&format=csv")


def fetch(url: str) -> str:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.text


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    print("Downloading USGS discharge ...")
    q_txt = fetch(USGS_URL)
    (RAW / f"usgs_{SITE}_daily_discharge_raw.tsv").write_text(q_txt)
    print("Downloading Daymet precipitation ...")
    p_txt = fetch(DAYMET_URL)
    (RAW / "daymet_precip_raw.csv").write_text(p_txt)

    master = build_master(parse_usgs_rdb(q_txt), parse_daymet_csv(p_txt))
    master = master[(master["date"] >= START) & (master["date"] <= END)]
    master.to_csv("data/streamflow_data.csv", index=False)
    print(f"Wrote data/streamflow_data.csv  ({len(master)} rows, "
          f"{master['date'].min().date()} to {master['date'].max().date()})")
    print(master.isna().sum().to_string())


if __name__ == "__main__":
    main()
