import csv
import pandas as pd
from pathlib import Path

# ---- files -------------------------------------------------
FILES = [
    "TradeData_1_15_2026_20_5_39.csv",
    "TradeData_1_15_2026_20_5_15.csv",
    "TradeData_1_15_2026_20_4_27.csv",
    "TradeData_1_15_2026_20_3_58.csv",
    "TradeData_1_15_2026_20_3_38.csv",
    "TradeData_1_15_2026_20_3_25.csv",
]

OUT_PANEL = "lng_imports_monthly_panel.csv"
OUT_QA = "lng_imports_monthly_qa.csv"

# ---- helper: robust CSV reader -----------------------------
def read_trade_csv(path: str) -> pd.DataFrame:
    # Try tab-separated first (your sample is TSV)
    try:
        return pd.read_csv(
            path,
            sep="\t",
            engine="python",         # more tolerant of malformed rows
            encoding="utf-8",
            on_bad_lines="skip",      # pandas >= 1.3
            quoting=csv.QUOTE_NONE,   # treat quotes as normal characters
            escapechar="\\",
        )
    except Exception:
        # Fallback: comma-separated
        return pd.read_csv(
            path,
            sep=",",
            engine="python",
            encoding="utf-8",
            on_bad_lines="skip",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )

# ---- load and stack ----------------------------------------
dfs = []
for f in FILES:
    df = read_trade_csv(f)
    dfs.append(df)

raw = pd.concat(dfs, ignore_index=True)

# ---- keep LNG imports only --------------------------------
raw = raw[
    (raw["cmdCode"].astype(str) == "271111") &
    (raw["flowCode"].astype(str).str.lower().isin(["m", "import"]))
].copy()

# ---- date --------------------------------------------------
raw["date"] = pd.to_datetime(
    dict(year=raw["refYear"], month=raw["refMonth"], day=1)
)

# ---- quantities -------------------------------------------
raw["netWgt"] = pd.to_numeric(raw["netWgt"], errors="coerce")
raw["qty"] = pd.to_numeric(raw["qty"], errors="coerce")

# Use netWgt first, fall back to qty
raw["lng_imports_kg"] = raw["netWgt"].fillna(raw["qty"])

raw["lng_value_usd"] = pd.to_numeric(raw["primaryValue"], errors="coerce")

# ---- aggregate to monthly panel ---------------------------
panel = (
    raw.groupby(["reporterISO", "date"], as_index=False)
       .agg(
           lng_imports_kg=("lng_imports_kg", "sum"),
           lng_value_usd=("lng_value_usd", "sum"),
       )
       .sort_values(["reporterISO", "date"])
)

# ---- QA summary -------------------------------------------
qa = (
    panel.groupby("reporterISO")
         .agg(
             start=("date", "min"),
             end=("date", "max"),
             months=("date", "count"),
             total_kg=("lng_imports_kg", "sum"),
         )
         .reset_index()
)

# ---- save --------------------------------------------------
panel.to_csv(OUT_PANEL, index=False)
qa.to_csv(OUT_QA, index=False)

print("Saved:")
print(" ", OUT_PANEL)
print(" ", OUT_QA)
print("\nCoverage:")
print(qa)
