# Local Volatility Surface Builder
## Academic Research Tool for Quantitative Finance

**Author**: Francesco De Girolamo  
**Date**: February 2026  
**Institution**: Columbia University  

Complete research pipeline for building **Local Volatility surfaces** from WRDS OptionMetrics data using the **Dupire (1994) formula**.

---

## 📚 Academic Context

This project implements the complete local volatility calibration framework for equity options pricing. The pipeline transforms market-observed implied volatilities into arbitrage-free local volatility surfaces suitable for:

- **Options pricing** under stochastic volatility
- **Risk management** and Greeks calculation
- **Model calibration** for exotic derivatives
- **Academic research** in volatility modeling

### Theoretical Foundation

Local volatility models (Dupire, 1994; Derman & Kani, 1994) provide a deterministic volatility function σ(S,t) that reproduces market option prices while maintaining no-arbitrage conditions.

---

## 🎯 Features

### Data Pipeline
- ✅ **WRDS OptionMetrics integration** - Institutional-grade options data
- ✅ **Robust filtering** - Liquidity, bid-ask spread, moneyness constraints
- ✅ **Data quality checks** - Volume thresholds, IV sanity checks

### Mathematical Implementation
- ✅ **RBF interpolation** - Smooth 2D surface construction
- ✅ **Dupire formula** - ∂C/∂T and ∂²C/∂K² numerical differentiation
- ✅ **Arbitrage checks** - Monotonicity and convexity validation
- ✅ **Gaussian smoothing** - Noise reduction for stable derivatives

### Visualization & Analysis
- ✅ **Interactive 3D plots** - Implied vol and local vol surfaces
- ✅ **Jupyter notebook workflow** - Step-by-step analysis
- ✅ **Export capabilities** - HTML plots, CSV data, NumPy arrays

## 📊 Research Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. Data Acquisition (WRDS OptionMetrics)                  │
│     • Fetch option prices, IVs, Greeks                      │
│     • Query by ticker, date range                           │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Data Cleaning & Feature Engineering                    │
│     • Bid-ask spread filter (<10%)                          │
│     • Moneyness normalization (K/S)                         │
│     • Time to maturity (years)                              │
│     • Liquidity filter (volume ≥ 10)                        │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Implied Volatility Surface (Market View)               │
│     • RBF 2D interpolation (cubic kernel)                   │
│     • Gaussian smoothing (σ=0.8)                            │
│     • Regular grid (50×30 default)                          │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Call Price Surface (Black-Scholes)                     │
│     • IV → Call prices via BS formula                       │
│     • Moneyness × Spot = Absolute strikes                   │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Local Volatility (Dupire Formula)                      │
│     • ∂C/∂T: Time derivative (forward diff)                 │
│     • ∂C/∂K: Strike derivative (central diff)               │
│     • ∂²C/∂K²: Second derivative (convexity)                │
│     • σ²ₗₒcₐₗ = [∂C/∂T + (r-q)K∂C/∂K + qC] / [½K²∂²C/∂K²]  │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Arbitrage Validation                                    │
│     • Monotonicity: ∂C/∂K < 0                               │
│     • Convexity: ∂²C/∂K² > 0                                │
│     • Violation threshold: <5% acceptable                   │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Visualization & Export                                  │
│     • Interactive 3D Plotly surfaces                        │
│     • HTML export for presentations                         │
│     • CSV/NPZ data for further analysis                     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Method 1: Interactive Notebook (Recommended for Research)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure WRDS (one-time setup)
python test_wrds_connection.py
# Follow Duo Mobile authentication prompts

# 3. Launch Jupyter
jupyter lab

# 4. Open local_vol_notebook.ipynb
# Run cells step-by-step to understand the pipeline
```

### Method 2: Batch Processing Script

```bash
# Set your WRDS username
export WRDS_USERNAME='your_username'

# Run pipeline for multiple tickers
python main.py
```

**Output files**:
- `{TICKER}_IV_surface_{DATE}.html` - Interactive implied volatility surface
- `{TICKER}_LocalVol_surface_{DATE}.html` - Interactive local volatility surface  
- `{TICKER}_options_data.csv` - Cleaned option data
- `{TICKER}_surfaces.npz` - NumPy arrays for further analysis

### Method 3: Python API

```python
from local_vol_builder import LocalVolSurfaceBuilder

# Initialize
builder = LocalVolSurfaceBuilder(wrds_username='your_user')

# Build full pipeline
results = builder.build_full_pipeline(
    ticker='AAPL',
    start_date='2025-01-01',
    end_date='2025-12-31'
)

# Access results
print(f"Spot: ${results['spot']:.2f}")
print(f"Arbitrage violations: {results['arbitrage_check']}")
results['fig_iv'].show()  # Display IV surface
results['fig_lv'].show()  # Display local vol surface
```

## 📖 Usage

### Basic Example

```python
from local_vol_builder import LocalVolSurfaceBuilder

# Initialize
builder = LocalVolSurfaceBuilder(wrds_username='your_username')

# Build surfaces for a single stock
results = builder.build_full_pipeline(
    ticker='AAPL',
    start_date='2025-01-01',
    end_date='2025-12-31'
)

# Access results
iv_surface = results['IV_surface']
lv_surface = results['LV_surface']
spot_price = results['spot']
data = results['data']
```

### Custom Analysis

```python
# Fetch and clean data
df_raw = builder.fetch_option_data('TSLA', '2025-01-01', '2025-12-31')
df_clean = builder.clean_and_prepare(df_raw)

# Build IV surface
K_grid, T_grid, IV_grid, S0 = builder.build_iv_surface(df_clean)

# Convert to call prices
C_grid = builder.iv_to_call_prices(K_grid, T_grid, IV_grid, S0)

# Compute local volatility
LV_grid = builder.compute_local_vol(K_grid, T_grid, C_grid, S0)

# Check arbitrage
arb_violations = builder.check_arbitrage(C_grid, K_grid)
print(f"Arbitrage violations: {arb_violations}")
```

## 🔧 Configuration

### Risk-Free Rate

Update the risk-free rate in the constructor:

```python
builder = LocalVolSurfaceBuilder(
    wrds_username='your_username',
    risk_free_rate=0.045  # 4.5%
)
```

For dynamic rates, integrate with FRED API:

```python
import pandas_datareader as pdr

# Fetch 3-month Treasury rate
rf_rate = pdr.get_data_fred('DGS3MO').iloc[-1].values[0] / 100
builder = LocalVolSurfaceBuilder(wrds_username='user', risk_free_rate=rf_rate)
```

### Data Filters

Adjust filters in `clean_and_prepare()`:

```python
df = df[
    (df['moneyness'] >= 0.80) &  # Min moneyness
    (df['moneyness'] <= 1.20) &  # Max moneyness
    (df['tau'] >= 0.02) &        # Min maturity (years)
    (df['tau'] <= 1.0) &         # Max maturity (years)
    (df['volume'] >= 10) &       # Min volume
    (df['spread_pct'] < 0.10)    # Max bid-ask spread (10%)
]
```

### Grid Resolution

Change interpolation grid size:

```python
K_grid, T_grid, IV_grid, S0 = builder.build_iv_surface(
    df_clean, 
    grid_size=(100, 50)  # (K_points, T_points)
)
```

## 📐 Mathematical Background

### Dupire Formula

The local volatility is computed using Dupire's formula:

$$\sigma_{local}^2(K,T) = \frac{\frac{\partial C}{\partial T} + (r-q)K\frac{\partial C}{\partial K} + qC}{\frac{1}{2}K^2\frac{\partial^2 C}{\partial K^2}}$$

Where:
- $C(K,T)$ = Call price surface
- $K$ = Strike price
- $T$ = Time to maturity
- $r$ = Risk-free rate
- $q$ = Continuous dividend yield

### No-Arbitrage Conditions

The pipeline checks:
1. **Monotonicity**: $\frac{\partial C}{\partial K} < 0$ (calls decrease with strike)
2. **Convexity**: $\frac{\partial^2 C}{\partial K^2} > 0$ (butterfly spread arbitrage)

## 📊 Output Files

### HTML Plots

Interactive 3D surfaces with:
- Zoom, rotate, pan controls
- Hover tooltips with exact values
- Customizable camera angles

### CSV Data

Cleaned option data with columns:
- `date`, `exdate`, `strike_price`, `spot_price`
- `moneyness`, `tau` (time to maturity)
- `impl_volatility`, `delta`, `gamma`, `vega`
- `mid_price`, `spread_pct`, `volume`, `open_interest`

## 🔬 Advanced Features

### 1. Put-Call Parity Integration

Currently uses only calls. To include puts:

```python
# In clean_and_prepare(), remove the call-only filter
# df = df[df['cp_flag'] == 'C']  # Comment this out

# Then use put-call parity to synthesize missing data
```

### 2. SVI Calibration

Replace RBF interpolation with SVI parametric model:

```python
from scipy.optimize import minimize

def svi_model(params, k, tau):
    """SVI parameterization for IV surface"""
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

# Calibrate to market data
# ... (implementation left as exercise)
```

### 3. Real-Time Updates

Schedule daily updates with cron or Airflow:

```python
# airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator

def run_local_vol_pipeline():
    builder = LocalVolSurfaceBuilder(wrds_username='user')
    builder.build_full_pipeline('SPY')

dag = DAG('local_vol_daily', schedule_interval='0 18 * * 1-5')
task = PythonOperator(task_id='build_surfaces', python_callable=run_local_vol_pipeline, dag=dag)
```

## 🐛 Troubleshooting

### "Insufficient data after cleaning"

- Widen the date range (`start_date` to `end_date`)
- Relax filters (moneyness range, volume threshold)
- Check if the stock has active options

### High arbitrage violations

- Increase smoothing: `gaussian_filter(IV_grid, sigma=1.5)`
- Use finer grid: `grid_size=(100, 60)`
- Filter out illiquid options more aggressively

### WRDS connection issues

```bash
# Reset WRDS credentials
wrds-config
```

## 📚 Academic References

### Primary Literature

1. **Dupire, B. (1994)**. "Pricing with a Smile". *Risk Magazine*, 7(1), 18-20.  
   *Original local volatility framework*

2. **Derman, E., & Kani, I. (1994)**. "Riding on a Smile". *Risk Magazine*, 7(2), 32-39.  
   *Binomial tree approach to local volatility*

3. **Andersen, L., & Brotherton-Ratcliffe, R. (1998)**. "The Equity Option Volatility Smile: An Implicit Finite-Difference Approach". *Journal of Computational Finance*, 1(2), 5-37.  
   *Numerical methods for Dupire PDE*

4. **Gatheral, J. (2006)**. *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.  
   *Comprehensive reference for volatility modeling*

### Data & Methodology

5. **OptionMetrics Ivy DB Documentation**. WRDS.  
   [https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/)

6. **Fengler, M. R. (2009)**. "Arbitrage-free smoothing of the implied volatility surface". *Quantitative Finance*, 9(4), 417-428.  
   *RBF interpolation for volatility surfaces*

### Related Work

7. **Hagan, P. S., et al. (2002)**. "Managing Smile Risk". *Wilmott Magazine*, September.  
   *SVI and SABR models for comparison*

8. **Cont, R., & da Fonseca, J. (2002)**. "Dynamics of implied volatility surfaces". *Quantitative Finance*, 2(1), 45-60.  
   *Time-series properties of volatility surfaces*

## 📝 License

MIT License - Free to use and modify

## 👤 Author

Francesco De Girolamo  
Date: 2026-02-06
