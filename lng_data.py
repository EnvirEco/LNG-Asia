"""
Build a monthly panel for Japan, South Korea, China to study how LNG behaves
in destination power markets.

Data pulled
1) UN Comtrade API: monthly LNG imports (HS 271111) by reporter country
2) Ember API: monthly electricity generation by fuel (coal, gas, solar, wind, nuclear)
3) World Bank Pink Sheet: monthly coal price proxy (Newcastle)

Output
- data/panel.parquet
- data/panel.csv
- data/qa_missingness.csv

Before running
pip install pandas numpy requests pyarrow openpyxl statsmodels linearmodels

Notes
- Comtrade API sometimes enforces rate limits and row limits. This script batches by year.
- Ember API endpoints may change slightly by version. If an endpoint fails,
  open https://api.ember-energy.org/docs in a browser and adjust the endpoint strings below.
"""

from __future__ import annotations

import os
import time
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

COUNTRIES = {
    "JPN": {"name": "Japan", "comtrade_reporter_code": 392, "ember_country": "Japan"},
    "KOR": {"name": "South Korea", "comtrade_reporter_code": 410, "ember_country": "South Korea"},
    "CHN": {"name": "China", "comtrade_reporter_code": 156, "ember_country": "China"},
}

HS_LNG = "271111"  # LNG
FLOW_IMPORT = "M"  # imports in Comtrade API (varies by API version; we use "flowCode" below)

START_YEAR = 2015
END_YEAR = 2025  # set to current year as needed

# World Bank Pink Sheet monthly xlsx (historical, nominal USD)
WORLD_BANK_PINK_SHEET_XLSX = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"

# Comtrade API base and path pattern
# Example seen in official wrappers: https://comtradeapi.un.org/data/v1/get/C/A/HS?... (annual)
# For monthly, path is typically C/M/HS.
COMTRADE_BASE = "https://comtradeapi.un.org/data/v1/get"

# Ember API base
EMBER_BASE = "https://api.ember-energy.org"

# Sleep between API calls to be polite
SLEEP_SECONDS = 0.35


def _get_json(url: str, params: Dict, headers: Optional[Dict] = None, timeout: int = 60) -> Dict:
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"GET failed {r.status_code} for {r.url}\n{r.text[:500]}")
    return r.json()


def _safe_to_datetime(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=int(month), day=1)


def fetch_comtrade_monthly_lng_imports(
    reporter_code: int,
    start_year: int,
    end_year: int,
    cmd_code: str = HS_LNG,
    include_desc: bool = False,
) -> pd.DataFrame:
    """
    Returns monthly LNG imports for one reporter country.

    Output columns
    - period (Timestamp, first day of month)
    - reporter_code
    - net_wgt_kg (float)
    - qty (float, optional)
    - trade_value_usd (float)
    """
    rows: List[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        url = f"{COMTRADE_BASE}/C/M/HS"
        params = {
            "reporterCode": reporter_code,
            "partnerCode": 0,         # world
            "partner2Code": 0,
            "cmdCode": cmd_code,
            "flowCode": "M",          # imports
            "customsCode": "C00",
            "motCode": 0,
            "period": year,           # API returns all months for the year in one call (often)
            "includeDesc": "true" if include_desc else "false",
        }

        js = _get_json(url, params=params)
        data = js.get("data", []) or js.get("dataset", []) or []

        if not data:
            continue

        df = pd.DataFrame(data)

        # Comtrade field names can vary by API version. Try common options.
        # period often comes as yyyymm int or string.
        if "period" in df.columns:
            period_raw = df["period"]
        elif "refPeriodId" in df.columns:
            period_raw = df["refPeriodId"]
        else:
            raise RuntimeError(f"Could not find period column in Comtrade response columns: {df.columns.tolist()}")

        period_raw = period_raw.astype(str)
        df["year"] = period_raw.str.slice(0, 4).astype(int)
        df["month"] = period_raw.str.slice(4, 6).astype(int)
        df["period"] = [
            _safe_to_datetime(y, m) for y, m in zip(df["year"].values, df["month"].values)
        ]

        # Net weight
        if "netWgt" in df.columns:
            df["net_wgt_kg"] = pd.to_numeric(df["netWgt"], errors="coerce")
        elif "NetWeight" in df.columns:
            df["net_wgt_kg"] = pd.to_numeric(df["NetWeight"], errors="coerce")
        else:
            df["net_wgt_kg"] = np.nan

        # Quantity
        qty_col = None
        for c in ["qty", "qty1", "Qty", "Quantity"]:
            if c in df.columns:
                qty_col = c
                break
        df["qty"] = pd.to_numeric(df[qty_col], errors="coerce") if qty_col else np.nan

        # Trade value
        val_col = None
        for c in ["primaryValue", "tradeValue", "TradeValue"]:
            if c in df.columns:
                val_col = c
                break
        df["trade_value_usd"] = pd.to_numeric(df[val_col], errors="coerce") if val_col else np.nan

        keep = df[["period", "net_wgt_kg", "qty", "trade_value_usd"]].copy()
        keep["reporter_code"] = reporter_code
        rows.append(keep)

        time.sleep(SLEEP_SECONDS)

    if not rows:
        return pd.DataFrame(columns=["period", "reporter_code", "net_wgt_kg", "qty", "trade_value_usd"])

    out = pd.concat(rows, ignore_index=True)

    # Aggregate to period just in case the API returns multiple rows per month
    out = (
        out.groupby(["reporter_code", "period"], as_index=False)[["net_wgt_kg", "qty", "trade_value_usd"]]
        .sum(min_count=1)
    )
    return out


def fetch_ember_monthly_generation(country_name: str) -> pd.DataFrame:
    """
    Pull monthly generation by fuel for one country from Ember API.

    Output columns
    - period (Timestamp)
    - fuel (standardized)
    - generation_gwh (float)

    If the endpoint shape differs, adjust in one place here.
    """
    # Endpoint based on Ember docs. If this fails, open Ember swagger:
    # https://api.ember-energy.org/docs
    url = f"{EMBER_BASE}/v1/electricity-generation/monthly"

    params = {
        "country": country_name,
    }

    js = _get_json(url, params=params)
    data = js.get("data", js)

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["period", "fuel", "generation_gwh"])

    # Try common column names
    # Expect something like: year, month, fuel, value (GWh)
    for ycol in ["year", "Year"]:
        if ycol in df.columns:
            df["year"] = pd.to_numeric(df[ycol], errors="coerce")
            break
    for mcol in ["month", "Month"]:
        if mcol in df.columns:
            df["month"] = pd.to_numeric(df[mcol], errors="coerce")
            break
    fuel_col = "fuel" if "fuel" in df.columns else ("Fuel" if "Fuel" in df.columns else None)
    val_col = None
    for c in ["generation", "value", "Generation", "Value", "gwh", "GWh"]:
        if c in df.columns:
            val_col = c
            break
    if fuel_col is None or val_col is None or "year" not in df.columns or "month" not in df.columns:
        raise RuntimeError(f"Unexpected Ember response columns: {df.columns.tolist()}")

    df["period"] = [
        _safe_to_datetime(int(y), int(m)) for y, m in zip(df["year"].values, df["month"].values)
    ]
    df["fuel"] = df[fuel_col].astype(str).str.lower().str.strip()
    df["generation_gwh"] = pd.to_numeric(df[val_col], errors="coerce")

    # Standardize fuel labels
    fuel_map = {
        "coal": "coal",
        "gas": "gas",
        "natural gas": "gas",
        "solar": "solar",
        "wind": "wind",
        "nuclear": "nuclear",
    }
    df["fuel"] = df["fuel"].map(lambda x: fuel_map.get(x, x))

    keep = df[["period", "fuel", "generation_gwh"]].copy()
    return keep


def fetch_worldbank_coal_price_newcastle() -> pd.DataFrame:
    """
    Loads World Bank Pink Sheet monthly data and extracts a coal series.

    Output columns
    - period (Timestamp)
    - coal_price_usd (float)

    The workbook has a Description tab that names series.
    We try a few likely column labels and fall back to searching.
    """
    xlsx_path = DATA_DIR / "worldbank_pink_sheet_monthly.xlsx"
    if not xlsx_path.exists():
        r = requests.get(WORLD_BANK_PINK_SHEET_XLSX, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download Pink Sheet xlsx: {r.status_code}")
        xlsx_path.write_bytes(r.content)

    # The monthly series are typically in a sheet named "Monthly Prices"
    # but it can vary. We'll try to find the first sheet that looks like monthly time series.
    xls = pd.ExcelFile(xlsx_path)

    sheet_name = None
    for candidate in ["Monthly Prices", "Monthly_Prices", "Data", "Monthly"]:
        if candidate in xls.sheet_names:
            sheet_name = candidate
            break
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Try to locate date column
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in ["date", "time", "month", "period"]:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df = df.rename(columns={date_col: "period"})
    df["period"] = pd.to_datetime(df["period"], errors="coerce")

    # Find a coal series column (Newcastle/Australian thermal coal)
    cols_lower = {c: str(c).lower() for c in df.columns}
    coal_candidates = []
    for c, cl in cols_lower.items():
        if "coal" in cl and ("australia" in cl or "newcastle" in cl or "thermal" in cl):
            coal_candidates.append(c)

    if not coal_candidates:
        # fallback: any coal column
        for c, cl in cols_lower.items():
            if "coal" in cl and "price" not in cl:
                coal_candidates.append(c)

    if not coal_candidates:
        raise RuntimeError(
            "Could not find a coal price column in the Pink Sheet workbook. "
            "Open the xlsx and pick the right column name, then set it in the script."
        )

    coal_col = coal_candidates[0]
    out = df[["period", coal_col]].copy()
    out = out.rename(columns={coal_col: "coal_price_usd"})
    out["coal_price_usd"] = pd.to_numeric(out["coal_price_usd"], errors="coerce")
    out = out.dropna(subset=["period"])
    out = out.sort_values("period")
    return out


def build_panel() -> pd.DataFrame:
    # 1) LNG imports
    lng_parts = []
    for iso3, meta in COUNTRIES.items():
        df_lng = fetch_comtrade_monthly_lng_imports(
            reporter_code=meta["comtrade_reporter_code"],
            start_year=START_YEAR,
            end_year=END_YEAR,
            cmd_code=HS_LNG,
        )
        df_lng["iso3"] = iso3
        lng_parts.append(df_lng)

    lng = pd.concat(lng_parts, ignore_index=True)
    lng = lng.rename(columns={"net_wgt_kg": "lng_imports_kg"})
    lng = lng[["iso3", "period", "lng_imports_kg", "trade_value_usd"]]

    # 2) Ember generation by fuel
    gen_parts = []
    for iso3, meta in COUNTRIES.items():
        df_gen = fetch_ember_monthly_generation(meta["ember_country"])
        df_gen["iso3"] = iso3
        gen_parts.append(df_gen)
        time.sleep(SLEEP_SECONDS)

    gen_long = pd.concat(gen_parts, ignore_index=True)

    # Pivot fuels wide
    gen_wide = (
        gen_long.pivot_table(
            index=["iso3", "period"],
            columns="fuel",
            values="generation_gwh",
            aggfunc="sum",
        )
        .reset_index()
    )
    gen_wide.columns = [c if isinstance(c, str) else str(c) for c in gen_wide.columns]

    # Keep key fuels if present
    for c in ["coal", "gas", "solar", "wind", "nuclear"]:
        if c not in gen_wide.columns:
            gen_wide[c] = np.nan

    gen_wide["renew_gwh"] = gen_wide[["solar", "wind"]].sum(axis=1, min_count=1)
    gen_wide["thermal_gwh"] = gen_wide[["coal", "gas"]].sum(axis=1, min_count=1)

    # 3) Prices
    coal_price = fetch_worldbank_coal_price_newcastle()

    # Merge panel
    panel = (
        lng.merge(gen_wide, on=["iso3", "period"], how="outer")
        .merge(coal_price, on="period", how="left")
        .sort_values(["iso3", "period"])
    )

    # Derived shares (guard against zero)
    panel["total_known_gwh"] = panel[["coal", "gas", "renew_gwh", "nuclear"]].sum(axis=1, min_count=1)
    panel["coal_share"] = panel["coal"] / panel["total_known_gwh"]
    panel["gas_share"] = panel["gas"] / panel["total_known_gwh"]

    # Simple lags
    panel["lng_imports_kg_l1"] = panel.groupby("iso3")["lng_imports_kg"].shift(1)
    panel["lng_imports_kg_l3"] = panel.groupby("iso3")["lng_imports_kg"].shift(3)

    return panel


def qa_missingness(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "lng_imports_kg",
        "coal",
        "gas",
        "renew_gwh",
        "nuclear",
        "coal_price_usd",
    ]
    out = (
        panel.groupby("iso3")[cols]
        .apply(lambda d: d.isna().mean())
        .reset_index()
        .rename(columns={c: f"missing_rate_{c}" for c in cols})
    )
    return out


def main():
    panel = build_panel()

    # Trim to analysis window
    panel = panel[(panel["period"] >= pd.Timestamp(START_YEAR, 1, 1)) & (panel["period"] <= pd.Timestamp(END_YEAR, 12, 1))]

    panel_path_parquet = DATA_DIR / "panel.parquet"
    panel_path_csv = DATA_DIR / "panel.csv"
    qa_path = DATA_DIR / "qa_missingness.csv"

    panel.to_parquet(panel_path_parquet, index=False)
    panel.to_csv(panel_path_csv, index=False)

    qa = qa_missingness(panel)
    qa.to_csv(qa_path, index=False)

    print("Saved")
    print(panel_path_parquet)
    print(panel_path_csv)
    print(qa_path)
    print()
    print("Panel shape:", panel.shape)
    print("Missingness by country")
    print(qa.to_string(index=False))


if __name__ == "__main__":
    main()
