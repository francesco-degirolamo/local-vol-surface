# Project Structure

## 📁 Directory Organization

```
Vol Surface/
├── 📘 Core Research Files
│   ├── local_vol_builder.py          # Main Python module (18KB)
│   ├── local_vol_notebook.ipynb      # Interactive research notebook (15KB)
│   └── main.py                        # Batch processing script (2KB)
│
├── 📚 Documentation
│   ├── README.md                      # Project overview & quick start (14KB)
│   ├── METHODOLOGY.md                 # Detailed technical documentation (12KB)
│   ├── WRDS_SETUP.md                  # WRDS authentication guide (2KB)
│   └── SECURITY.md                    # Security best practices (2KB)
│
├── 🔧 Setup & Configuration
│   ├── requirements.txt               # Python dependencies
│   ├── .env.example                   # Environment variable template
│   └── .gitignore                     # Git ignore rules
│
├── 🧪 Testing & Utilities
│   ├── test_wrds_connection.py        # WRDS connection test
│   ├── test_wrds_auth.py              # Authentication test with Duo
│   ├── setup_wrds.py                  # One-time WRDS setup
│   ├── simple_test.py                 # Quick connection test
│   └── find_tables.py                 # OptionMetrics table explorer
│
└── 🗑️ Legacy (not needed)
    ├── vol_surface_builder.ipynb      # Old notebook (can delete)
    └── wrds_config.py                 # ⚠️ Contains credentials - DO NOT COMMIT
```

---

## 📄 File Descriptions

### Core Research Files

#### `local_vol_builder.py`
**Main Python module** implementing the complete pipeline:
- `LocalVolSurfaceBuilder` class
- WRDS data fetching
- Data cleaning & filtering
- RBF interpolation for IV surface
- Dupire local volatility calculation
- Arbitrage validation
- Plotly 3D visualization

**Key Methods**:
- `fetch_option_data()`: Query WRDS OptionMetrics
- `clean_and_prepare()`: Data quality filters
- `build_iv_surface()`: RBF interpolation
- `compute_local_vol()`: Dupire formula
- `check_arbitrage()`: No-arbitrage validation
- `build_full_pipeline()`: End-to-end execution

#### `local_vol_notebook.ipynb`
**Interactive Jupyter notebook** for step-by-step analysis:
- Academic documentation with formulas
- Research notes and interpretations
- Code cells for each pipeline step
- 3D surface visualizations
- Arbitrage checks and comparisons
- Export capabilities (HTML, CSV, NPZ)

**Best for**: Learning, debugging, presentation, thesis chapters

#### `main.py`
**Batch processing script** for multiple tickers:
- Loops over ticker list (AAPL, TSLA, SPY, NVDA)
- Generates surfaces for each
- Saves HTML plots and CSV data
- Summary statistics

**Best for**: Production runs, time-series studies

---

### Documentation

#### `README.md`
- Project overview
- Academic context
- Pipeline architecture
- Quick start (3 methods)
- Example code
- References

#### `METHODOLOGY.md`
**Comprehensive technical documentation**:
- Mathematical foundations
- Data pipeline details
- RBF interpolation theory
- Numerical differentiation
- Arbitrage validation
- Computational complexity
- Parameter tuning guide
- Academic references

**Best for**: Thesis methodology chapter, paper submissions

#### `WRDS_SETUP.md`
- WRDS authentication flow
- Duo Mobile 2FA setup
- Credential caching
- Troubleshooting

#### `SECURITY.md`
- Security best practices
- Environment variables
- Production deployment
- What to never commit

---

### Setup & Configuration

#### `requirements.txt`
Python dependencies with **pinned versions**:
```
wrds==3.1.7
pandas==2.2.0
numpy==1.26.4
scipy==1.12.0
plotly==5.18.0
jupyterlab==4.0.11
```

Install: `pip install -r requirements.txt`

#### `.env.example`
Template for environment variables:
```bash
WRDS_USERNAME=your_username
RISK_FREE_RATE=0.045
```

Usage: `cp .env.example .env` then edit

#### `.gitignore`
Protects sensitive files:
- `wrds_config.py` (credentials)
- `.env` (environment vars)
- Output files (*.html, *.csv)
- Jupyter checkpoints
- Python cache

---

### Testing & Utilities

#### `test_wrds_connection.py`
**First test to run**: Validates WRDS access
- Tests database connection
- Checks credentials
- Verifies OptionMetrics access

Run: `python test_wrds_connection.py`

#### `test_wrds_auth.py`
**Duo Mobile authentication test**:
- Interactive prompt
- Tests 2FA flow
- Saves credentials to `~/.pgpass`

#### `setup_wrds.py`
**One-time setup wizard**:
- Runs `wrds-config` command
- Guides through authentication
- Tests connection

#### `find_tables.py`
**OptionMetrics table explorer**:
- Lists all tables in `optionm` library
- Searches for specific table patterns
- Helps debug table names by year

---

## 🎯 Recommended Workflow

### First-Time Setup

```bash
# 1. Clone/download project
cd "Vol Surface"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure WRDS (one-time)
python test_wrds_connection.py
# Follow prompts for username/password/Duo

# 4. Set environment
cp .env.example .env
nano .env  # Add your WRDS_USERNAME
```

### Interactive Research (Recommended)

```bash
# Launch Jupyter
jupyter lab

# Open local_vol_notebook.ipynb
# Run cells step-by-step
# Experiment with parameters
# Export results
```

### Batch Processing

```bash
# Edit ticker list in main.py
nano main.py

# Run pipeline
python main.py

# Results saved as:
# - AAPL_IV_surface_2026-02-07.html
# - AAPL_LocalVol_surface_2026-02-07.html
# - AAPL_options_data.csv
```

### Python API

```python
from local_vol_builder import LocalVolSurfaceBuilder

builder = LocalVolSurfaceBuilder()
results = builder.build_full_pipeline(
    ticker='AAPL',
    start_date='2025-12-01',
    end_date='2026-02-07'
)

# Access results
print(results['spot'])
print(results['arbitrage_check'])
results['fig_iv'].show()
```

---

## 📊 Output Files

### Generated Automatically

```
AAPL_IV_surface_2026-02-07.html        # Interactive 3D IV surface
AAPL_LocalVol_surface_2026-02-07.html  # Interactive 3D local vol
AAPL_options_data.csv                  # Cleaned option data
AAPL_surfaces.npz                      # NumPy arrays (optional)
```

All outputs are in `.gitignore` (not committed)

---

## 🔬 For Academic Research

### Thesis/Paper Workflow

1. **Methodology Chapter**: Copy from `METHODOLOGY.md`
2. **Results**: Run notebook, export plots as PNG
3. **Tables**: Use pandas to export summary stats
4. **Reproducibility**: Include `requirements.txt` + notebook

### Extending the Project

**Common Extensions**:
- Time-series: Loop over dates, save surfaces, analyze dynamics
- Cross-sectional: Compare across stocks (tech vs finance)
- Model comparison: Add SABR, SVI, Heston calibration
- Risk management: Compute vega, Greeks, VaR

**Files to Modify**:
- `local_vol_builder.py`: Add new methods
- `local_vol_notebook.ipynb`: Add analysis cells
- `main.py`: Add loops/comparisons

---

## ⚠️ Security Checklist

Before committing to Git:

- [ ] `wrds_config.py` is in `.gitignore`
- [ ] `.env` is in `.gitignore`
- [ ] No credentials in code
- [ ] `requirements.txt` has exact versions
- [ ] Output files (*.html, *.csv) ignored

Before sharing:

- [ ] Remove `wrds_config.py` file
- [ ] Change WRDS password if exposed
- [ ] Check `git log` for leaked credentials

---

## 📚 Quick Reference

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Test WRDS | `python test_wrds_connection.py` |
| Notebook | `jupyter lab` |
| Batch | `python main.py` |
| Find tables | `python find_tables.py` |

| File Extension | Purpose |
|---------------|---------|
| `.py` | Python module/script |
| `.ipynb` | Jupyter notebook |
| `.md` | Documentation (Markdown) |
| `.txt` | Requirements, config |
| `.html` | Interactive 3D plots |
| `.csv` | Data export |
| `.npz` | NumPy array archive |

---

**Last Updated**: February 7, 2026  
**Project Version**: 1.0 (Academic Focus)
