# Data

The pipeline reads three CSVs from `data/raw/`:

| File | Period | Rows | Use |
|---|---|---:|---|
| `earthquakeq_train.csv` | 1990–2021 | 271,207 | fit |
| `earthquakeq_test.csv` | 2022–2024 | 48,328 | calibration / test history |
| `earthquakeq_judge.csv` | Jan 2025 – Mar 2026 | 17,979 | one held-out check only |

They were supplied by the SC Quantathon V3 Earthquake-Q challenge (SRNL & Clemson) and are not redistributed
here. All three share one long format:

```
source,time,latitude,longitude,depth_km,magnitude,station_id,parameter,value
```

`source` is one of `seismic` (USGS ComCat events, M ≥ 1.0), `geomagnetic` (ap, Kp), `solar` (F10.7, sunspot
number), `tidal_ephemeris` (tidal potential, lunar phase) or `gps` (daily station positions, 18 stations).
The pipeline never modifies these files.
