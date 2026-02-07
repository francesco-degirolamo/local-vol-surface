# Quick Reference: Available Data Dates

## ⏰ OptionMetrics Data Availability

**Important**: WRDS OptionMetrics has a **3-6 month delay**. As of February 2026, the most recent available data is likely from **mid-2025** or earlier.

### How to Find Latest Available Date

```python
import wrds

db = wrds.Connection()

# Check latest date in OptionMetrics
query = """
SELECT MAX(date) as latest_date
FROM optionm.opprcd2025
WHERE secid = (SELECT secid FROM optionm.secnmd WHERE ticker = 'AAPL' LIMIT 1)
"""

result = db.raw_sql(query)
print(f"Latest AAPL option data: {result.iloc[0, 0]}")
```

### Recommended Date Ranges (Safe Choices)

| Period | Start Date | End Date | VIX Level | Use Case |
|--------|-----------|----------|-----------|----------|
| **Recent Normal** | 2024-12-01 | 2025-01-31 | ~14-16 | Current market |
| **Stable 2024** | 2024-01-01 | 2024-02-29 | ~13-15 | Low volatility |
| **Stable 2023** | 2023-06-01 | 2023-07-31 | ~12-14 | Very low vol |
| **COVID Crash** | 2020-03-01 | 2020-04-30 | >70 | High volatility |
| **2022 Selloff** | 2022-06-01 | 2022-07-31 | ~30 | Medium-high vol |
| **Pre-COVID** | 2019-01-01 | 2019-02-28 | ~15-17 | Pre-crisis |

### Common Errors

#### ❌ `relation "optionm.opprcd2026" does not exist`
**Cause**: Trying to access future data  
**Fix**: Use dates from 2024-2025

#### ❌ `No data found in year 2025 tables`
**Cause**: Tables exist but no data for your date range  
**Fix**: Use earlier dates (e.g., 2024 instead of late 2025)

#### ❌ `Insufficient data after cleaning`
**Cause**: Too few liquid options in date range  
**Fix**: Widen date range or use more liquid ticker (SPY, AAPL)

### Quick Test Script

```python
from local_vol_builder import LocalVolSurfaceBuilder

# Test different date ranges
test_dates = [
    ('2024-12-01', '2025-01-31'),  # Try this first
    ('2024-06-01', '2024-07-31'),  # Fallback 1
    ('2024-01-01', '2024-02-29'),  # Fallback 2
]

builder = LocalVolSurfaceBuilder()

for start, end in test_dates:
    try:
        print(f"\nTrying {start} to {end}...")
        df = builder.fetch_option_data('AAPL', start, end)
        if len(df) > 100:
            print(f"✓ Success! Found {len(df)} options")
            print(f"Use: START_DATE = '{start}', END_DATE = '{end}'")
            break
    except Exception as e:
        print(f"✗ Failed: {e}")
```

### Data Vintage Notes

- **1996-2019**: Complete historical data
- **2020**: COVID period (extreme volatility)
- **2021-2023**: Post-COVID recovery
- **2024**: Most recent stable data
- **2025**: Partial data (check availability)
- **2026**: Not yet available

### For Academic Research

**Time-Series Studies**: Use complete years
- Good: `2023-01-01` to `2023-12-31`
- Avoid: Incomplete years like `2025-01-01` to `2025-12-31`

**Event Studies**: Center around known events
- Earnings: Check company calendar
- Fed announcements: FOMC dates
- Market crashes: 2020-03 (COVID), 2022-06 (selloff)

**Cross-Sectional**: Use same snapshot date for all stocks
- Good: `2024-06-15` (single date)
- Avoid: Different dates for different stocks

---

**Last Updated**: February 7, 2026  
**Next Update**: Check when 2025 Q4 data becomes available (~March-April 2026)
