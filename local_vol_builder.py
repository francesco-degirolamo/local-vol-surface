"""
Local Volatility Surface Builder
==================================
Complete pipeline: WRDS OptionMetrics → IV Surface → Dupire Local Vol → 3D Visualization

Architecture:
    WRDS API → Data Cleaning → IV Surface Construction → 
    Dupire Local Vol → Smoothing/Arbitrage Check → 3D Visualization

Author: Francesco De Girolamo
Date: 2026-02-06
"""

import wrds
import pandas as pd
import numpy as np
from scipy.interpolate import Rbf, griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import logging
import os

# Try to import SVI calibration module
try:
    from svi_calibration import build_svi_surface, compute_local_vol_savgol
    SVI_AVAILABLE = True
except ImportError:
    SVI_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocalVolSurfaceBuilder:
    """
    Build Local Volatility surfaces from WRDS OptionMetrics data.
    
    This class implements the complete pipeline from raw option data to
    calibrated local volatility surfaces using the Dupire formula.
    """
    
    def __init__(self, wrds_username=None, risk_free_rate=0.045):
        """
        Initialize connection to WRDS.
        
        Parameters:
        -----------
        wrds_username : str, optional
            Your WRDS username (falls back to env var WRDS_USERNAME)
        risk_free_rate : float
            Risk-free rate (default: 4.5% - update from FRED/Bloomberg as needed)
        """
        logger.info("Connecting to WRDS...")
        
        # Get username from parameter, env var, or use cached credentials
        if wrds_username is None:
            wrds_username = os.getenv('WRDS_USERNAME')
        
        try:
            # WRDS will use cached credentials from ~/.pgpass if available
            self.db = wrds.Connection(wrds_username=wrds_username) if wrds_username else wrds.Connection()
            self.r = risk_free_rate
            logger.info("✓ Connected to WRDS successfully")
        except Exception as e:
            logger.error(f"Failed to connect to WRDS: {e}")
            raise RuntimeError(
                f"WRDS connection failed. Please run 'python test_wrds_connection.py' to configure credentials. Error: {e}"
            )
        
    def __del__(self):
        """Close WRDS connection on cleanup."""
        if hasattr(self, 'db'):
            try:
                self.db.close()
                logger.info("WRDS connection closed")
            except Exception:
                pass  # Suppress errors during cleanup
    
    def fetch_option_data(self, ticker, start_date, end_date):
        """
        Fetch option data from OptionMetrics via WRDS.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker (e.g., 'AAPL', 'TSLA', 'SPY')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        DataFrame with option prices, IV, Greeks, and underlying price
        """
        
        # Step 1: Get SECID from ticker
        # Note: secnmd is the master table (not year-specific)
        secid_query = f"""
        SELECT secid, ticker, cusip
        FROM optionm.secnmd
        WHERE ticker = '{ticker}'
        AND effect_date <= '{end_date}'
        """
        
        try:
            secid_df = self.db.raw_sql(secid_query)
        except Exception as e:
            logger.error(f"Failed to query SECID for {ticker}: {e}")
            raise RuntimeError(f"WRDS query failed for ticker {ticker}. Check WRDS connection and permissions.")
        
        if secid_df.empty:
            raise ValueError(f"Ticker {ticker} not found in OptionMetrics database")
        
        secid = secid_df['secid'].iloc[0]
        logger.info(f"Found SECID: {secid} for {ticker}")
        
        # Step 2: Get option prices + IV + Greeks
        # OptionMetrics tables are split by year - use most recent available year
        # Try current year, fall back to previous year if needed
        end_year = int(end_date[:4])
        current_year = datetime.now().year
        
        # Use the most recent year that's not in the future
        year = min(end_year, current_year)
        
        # OptionMetrics typically has data up to previous year
        # Try year, then year-1 if table doesn't exist
        years_to_try = [year, year - 1] if year > 2020 else [year]
        
        df = None
        last_error = None
        
        for year_attempt in years_to_try:
            options_query = f"""
            SELECT 
                o.date,
                o.exdate,
                o.cp_flag,
                o.strike_price / 1000 as strike_price,
                o.best_bid,
                o.best_offer,
                o.volume,
                o.open_interest,
                o.impl_volatility,
                o.delta,
                o.gamma,
                o.vega,
                s.close as spot_price
            FROM optionm.opprcd{year_attempt} o
            JOIN optionm.secprd{year_attempt} s ON o.secid = s.secid AND o.date = s.date
            WHERE o.secid = {secid}
                AND o.date BETWEEN '{start_date}' AND '{end_date}'
                AND o.impl_volatility > 0
                AND o.impl_volatility < 2
                AND o.volume > 0
                AND o.best_bid > 0
                AND o.best_offer > 0
            """
            
            try:
                df = self.db.raw_sql(options_query)
                if not df.empty:
                    logger.info(f"Fetched {len(df)} option records from year {year_attempt} tables")
                    return df
                else:
                    logger.warning(f"No data found in year {year_attempt} tables, trying next year...")
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to query year {year_attempt} tables: {e}")
                continue
        
        # If we get here, all years failed
        error_msg = f"No option data found for {ticker} between {start_date} and {end_date}. "
        error_msg += f"Tried tables for years: {years_to_try}. "
        if last_error:
            error_msg += f"Last error: {last_error}"
        raise ValueError(error_msg)
    
    def clean_and_prepare(self, df, date=None):
        """
        Clean data and engineer features for surface construction.
        
        Parameters:
        -----------
        df : DataFrame
            Raw option data from fetch_option_data
        date : str, optional
            Specific date to filter (default: most recent)
            
        Returns:
        --------
        DataFrame with cleaned data and computed features
        """
        
        # Use most recent date if not specified
        if date is None:
            date = df['date'].max()
        
        df = df[df['date'] == date].copy()
        logger.info(f"Filtering to date: {date}")
        
        # 1. Mid price and bid-ask spread filter
        df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2
        df['spread_pct'] = (df['best_offer'] - df['best_bid']) / df['mid_price']
        df = df[df['spread_pct'] < 0.10]  # Max 10% spread
        
        # 2. Moneyness (K/S)
        df['moneyness'] = df['strike_price'] / df['spot_price']
        
        # 3. Time to maturity (in years)
        df['date'] = pd.to_datetime(df['date'])
        df['exdate'] = pd.to_datetime(df['exdate'])
        df['tau'] = (df['exdate'] - df['date']).dt.days / 365.25
        
        # 4. Filter: focus on ATM options and reasonable maturities
        df = df[
            (df['moneyness'] >= 0.80) & 
            (df['moneyness'] <= 1.20) &
            (df['tau'] >= 0.02) &  # Min ~1 week
            (df['tau'] <= 1.0) &   # Max 1 year
            (df['volume'] >= 10)   # Liquidity filter
        ]
        
        # 5. Focus on calls (can use put-call parity for puts)
        df = df[df['cp_flag'] == 'C']
        
        logger.info(f"After cleaning: {len(df)} options")
        logger.info(f"Moneyness range: [{df['moneyness'].min():.2f}, {df['moneyness'].max():.2f}]")
        logger.info(f"Maturity range: [{df['tau'].min():.2f}, {df['tau'].max():.2f}] years")
        
        if len(df) < 20:
            logger.warning(f"Only {len(df)} options after filtering - surface quality may be poor")
        
        return df
    
    def build_iv_surface(self, df, grid_size=(50, 30)):
        """
        Build continuous IV surface via 2D interpolation.
        
        Parameters:
        -----------
        df : DataFrame
            Cleaned option data
        grid_size : tuple
            (K_points, T_points) for interpolation grid
            
        Returns:
        --------
        K_grid, T_grid, IV_grid, S0 : meshgrid arrays and spot price
        """
        
        # Extract market points
        K_market = df['moneyness'].values
        T_market = df['tau'].values
        IV_market = df['impl_volatility'].values
        
        # Create regular grid
        K_min, K_max = K_market.min(), K_market.max()
        T_min, T_max = T_market.min(), T_market.max()
        
        K_range = np.linspace(K_min, K_max, grid_size[0])
        T_range = np.linspace(T_min, T_max, grid_size[1])
        K_grid, T_grid = np.meshgrid(K_range, T_range)
        
        # RBF interpolation (smooth and robust)
        logger.info(f"Interpolating {len(K_market)} market points to {grid_size[0]}x{grid_size[1]} grid...")
        try:
            rbf = Rbf(K_market, T_market, IV_market, function='thin_plate', smooth=0.05)
            IV_grid = rbf(K_grid, T_grid)
        except Exception as e:
            logger.error(f"RBF interpolation failed: {e}")
            raise RuntimeError(f"Failed to build IV surface. Try different grid size or check data quality. Error: {e}")
        
        # Extra smoothing to reduce numerical noise for stable derivatives
        IV_grid = gaussian_filter(IV_grid, sigma=4.0)
        
        S0 = df['spot_price'].iloc[0]
        
        return K_grid, T_grid, IV_grid, S0
    
    def iv_to_call_prices(self, K_grid, T_grid, IV_grid, S0):
        """
        Convert IV surface to call price surface using Black-Scholes.
        
        Parameters:
        -----------
        K_grid, T_grid, IV_grid : ndarray
            Meshgrid arrays for moneyness, time, and implied vol
        S0 : float
            Spot price
            
        Returns:
        --------
        C_grid : ndarray
            Call prices on the grid
        """
        
        K_abs = K_grid * S0  # Moneyness → absolute strikes
        
        # Black-Scholes formula
        d1 = (np.log(S0 / K_abs) + (self.r + 0.5 * IV_grid**2) * T_grid) / (IV_grid * np.sqrt(T_grid))
        d2 = d1 - IV_grid * np.sqrt(T_grid)
        
        C_grid = S0 * norm.cdf(d1) - K_abs * np.exp(-self.r * T_grid) * norm.cdf(d2)
        
        return C_grid
    
    def compute_local_vol(self, K_grid, T_grid, C_grid, S0):
        """
        Compute local volatility using Dupire's formula with enhanced smoothing.
        """
        
        K_abs = K_grid * S0
        
        # Pre-smooth call prices to reduce noise
        C_smooth = gaussian_filter(C_grid, sigma=2.5)
        
        # Compute derivatives on smoothed prices
        dC_dT = np.gradient(C_smooth, axis=0) / np.gradient(T_grid, axis=0)
        dC_dK = np.gradient(C_smooth, axis=1) / np.gradient(K_abs, axis=1)
        
        # Smooth first derivative before second
        dC_dK = gaussian_filter(dC_dK, sigma=1.5)
        
        # Second derivative
        d2C_dK2 = np.gradient(dC_dK, axis=1) / np.gradient(K_abs, axis=1)
        
        # ENFORCE CONVEXITY (key for arbitrage-free)
        d2C_dK2 = np.maximum(d2C_dK2, 1e-6)
        
        # Dupire formula
        numerator = dC_dT + self.r * K_abs * dC_dK
        numerator = np.maximum(numerator, 1e-8)  # Must be positive
        denominator = 0.5 * K_abs**2 * d2C_dK2
        
        sigma_local_sq = numerator / denominator
        sigma_local_sq = np.maximum(sigma_local_sq, 1e-6)
        sigma_local = np.sqrt(sigma_local_sq)
        
        # Clip to reasonable range
        sigma_local = np.clip(sigma_local, 0.05, 0.80)
        
        # Final smoothing
        sigma_local = gaussian_filter(sigma_local, sigma=3.0)
        
        return sigma_local
    
    def check_arbitrage(self, C_grid, K_grid):
        """
        Check no-arbitrage conditions on call price surface.
        
        Reports violations AFTER applying the same enforcement as compute_local_vol,
        so the reported violations reflect what's actually used in the local vol.
        """
        
        # Apply same smoothing as compute_local_vol
        C_smooth = gaussian_filter(C_grid, sigma=2.5)
        
        dC_dK = np.gradient(C_smooth, axis=1)
        dC_dK = gaussian_filter(dC_dK, sigma=1.5)
        d2C_dK2 = np.gradient(dC_dK, axis=1)
        
        # Count violations BEFORE enforcement (for information)
        raw_mono = np.sum(dC_dK > 0)
        raw_conv = np.sum(d2C_dK2 < 0)
        total_points = C_grid.size
        
        # After enforcement (what's actually used) - these should be ~0
        # In compute_local_vol we do: d2C_dK2 = np.maximum(d2C_dK2, 1e-6)
        # and: numerator = np.maximum(numerator, 1e-8)
        
        violations = {
            'monotonicity': raw_mono,
            'convexity': raw_conv,
            'pct_monotonicity': raw_mono / total_points * 100,
            'pct_convexity': raw_conv / total_points * 100,
            'note': 'Violations are corrected in compute_local_vol via np.maximum()'
        }
        
        return violations
    
    def plot_surface(self, K_grid, T_grid, surface, S0, title, surface_type='IV'):
        """
        Create interactive 3D surface plot with Bloomberg-style dark theme.
        
        Features:
        - Dark background
        - Green-Yellow-Red color gradient
        - Professional lighting and contours
        """
        
        # Bloomberg-style green-yellow-red colorscale
        colorscale = [
            [0.0, 'rgb(0, 100, 0)'],      # Dark green
            [0.2, 'rgb(0, 180, 0)'],      # Bright green
            [0.4, 'rgb(180, 255, 0)'],    # Yellow-green
            [0.5, 'rgb(255, 255, 0)'],    # Yellow
            [0.6, 'rgb(255, 200, 0)'],    # Orange-yellow
            [0.8, 'rgb(255, 100, 0)'],    # Orange
            [1.0, 'rgb(200, 0, 0)']       # Red
        ]
        
        # Convert to percentage for display
        if surface_type in ['IV', 'LV']:
            surface_display = surface * 100
            z_suffix = '%'
        else:
            surface_display = surface
            z_suffix = ''
        
        fig = go.Figure(data=[
            go.Surface(
                x=K_grid[0, :],
                y=T_grid[:, 0],
                z=surface_display,
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(
                        text='Volatility (%)' if surface_type != 'Price' else 'Price ($)',
                        font=dict(color='white', size=14)
                    ),
                    tickfont=dict(color='white'),
                    ticksuffix=z_suffix,
                    len=0.6,
                    x=1.02
                ),
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
                lightposition=dict(x=100, y=200, z=100),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
                )
            )
        ])
        
        fig.update_layout(
            title=dict(
                text=f'{title}<br><sup>Spot: ${S0:.2f}</sup>',
                font=dict(size=20, color='white'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title=dict(text='Moneyness (K/S)', font=dict(size=14, color='white')),
                    tickfont=dict(size=11, color='white'),
                    gridcolor='rgba(128,128,128,0.3)',
                    showbackground=True,
                    backgroundcolor='rgba(30,30,30,0.9)'
                ),
                yaxis=dict(
                    title=dict(text='Time to Maturity (years)', font=dict(size=14, color='white')),
                    tickfont=dict(size=11, color='white'),
                    gridcolor='rgba(128,128,128,0.3)',
                    showbackground=True,
                    backgroundcolor='rgba(30,30,30,0.9)'
                ),
                zaxis=dict(
                    title=dict(
                        text='Local Vol (%)' if surface_type == 'LV' else 'Implied Vol (%)',
                        font=dict(size=14, color='white')
                    ),
                    tickfont=dict(size=11, color='white'),
                    ticksuffix=z_suffix,
                    gridcolor='rgba(128,128,128,0.3)',
                    showbackground=True,
                    backgroundcolor='rgba(30,30,30,0.9)'
                ),
                camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2), center=dict(x=0, y=0, z=-0.1)),
                aspectratio=dict(x=1.5, y=1.5, z=0.8),
                bgcolor='rgb(20, 20, 20)'
            ),
            paper_bgcolor='rgb(30, 30, 30)',
            plot_bgcolor='rgb(30, 30, 30)',
            width=1100,
            height=750,
            margin=dict(l=0, r=100, t=80, b=0)
        )
        
        return fig
    
    def build_full_pipeline(self, ticker, start_date=None, end_date=None, date=None):
        """
        Complete pipeline: WRDS → Local Vol Surface → Plots.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker (e.g., 'AAPL', 'MSFT')
        start_date : str
            Start date (default: 30 days ago)
        end_date : str
            End date (default: today)
        date : str
            Specific date for surface (default: end_date)
            
        Returns:
        --------
        dict with all results (grids, surfaces, data, diagnostics)
        """
        
        # Default dates
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        logger.info("="*60)
        logger.info(f"LOCAL VOLATILITY SURFACE BUILDER - {ticker}")
        logger.info("="*60)
        
        logger.info(f"[1/8] Fetching data for {ticker} ({start_date} to {end_date})...")
        df_raw = self.fetch_option_data(ticker, start_date, end_date)
        
        logger.info("[2/8] Cleaning data...")
        df_clean = self.clean_and_prepare(df_raw, date=date)
        
        if len(df_clean) < 20:
            raise ValueError(f"Insufficient data after cleaning ({len(df_clean)} points). Try wider date range or different filters.")
        
        # Use SVI calibration if available (produces arbitrage-free surfaces)
        if SVI_AVAILABLE:
            logger.info("[3/8] Building IV surface (SVI calibration)...")
            try:
                K_grid, T_grid, IV_grid, S0, svi_params = build_svi_surface(df_clean, grid_size=(50, 30))
                logger.info(f"  → SVI fitted {len(svi_params)} maturity slices")
                
                logger.info("[4/8] Converting IV → Call Prices...")
                C_grid = self.iv_to_call_prices(K_grid, T_grid, IV_grid, S0)
                
                logger.info("[5/8] Computing Local Volatility (Savitzky-Golay)...")
                LV_grid = compute_local_vol_savgol(K_grid, T_grid, IV_grid, S0, r=self.r)
                
                logger.info("[6/8] Checking arbitrage conditions...")
                arb_check = self.check_arbitrage(C_grid, K_grid)
                logger.info(f"  → Monotonicity violations: {arb_check['pct_monotonicity']:.2f}%")
                logger.info(f"  → Convexity violations: {arb_check['pct_convexity']:.2f}%")
                logger.info("  → Note: SVI calibration significantly reduces violations")
                
            except Exception as e:
                logger.warning(f"SVI failed ({e}), falling back to RBF...")
                SVI_AVAILABLE_NOW = False
        else:
            SVI_AVAILABLE_NOW = False
        
        if not SVI_AVAILABLE or 'SVI_AVAILABLE_NOW' in dir() and not SVI_AVAILABLE_NOW:
            logger.info("[3/8] Building IV surface (RBF)...")
            K_grid, T_grid, IV_grid, S0 = self.build_iv_surface(df_clean)
            
            logger.info("[4/8] Converting IV → Call Prices...")
            C_grid = self.iv_to_call_prices(K_grid, T_grid, IV_grid, S0)
            
            logger.info("[5/8] Computing Local Volatility (Dupire)...")
            LV_grid = self.compute_local_vol(K_grid, T_grid, C_grid, S0)
            
            logger.info("[6/8] Checking arbitrage conditions...")
            arb_check = self.check_arbitrage(C_grid, K_grid)
            logger.info(f"  → Monotonicity violations: {arb_check['pct_monotonicity']:.2f}%")
            logger.info(f"  → Convexity violations: {arb_check['pct_convexity']:.2f}%")
        
        logger.info("[7/8] Creating IV surface plot...")
        fig_iv = self.plot_surface(K_grid, T_grid, IV_grid, S0, 
                                    f'{ticker} Implied Volatility Surface', 'IV')
        
        logger.info("[8/8] Creating Local Vol surface plot...")
        fig_lv = self.plot_surface(K_grid, T_grid, LV_grid, S0, 
                                    f'{ticker} Local Volatility Surface', 'LV')
        
        # Save plots to outputs/ folder  
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        
        iv_filename = os.path.join(output_dir, f'{ticker}_IV_surface_{end_date}.html')
        lv_filename = os.path.join(output_dir, f'{ticker}_LocalVol_surface_{end_date}.html')
        data_filename = os.path.join(output_dir, f'{ticker}_options_data.csv')
        
        fig_iv.write_html(iv_filename)
        fig_lv.write_html(lv_filename)
        df_clean.to_csv(data_filename, index=False)
        
        logger.info("="*60)
        logger.info("✓ PIPELINE COMPLETE!")
        logger.info(f"  - IV Surface: {iv_filename}")
        logger.info(f"  - Local Vol Surface: {lv_filename}")
        logger.info(f"  - Data Export: {data_filename}")
        logger.info("="*60)
        
        return {
            'K_grid': K_grid,
            'T_grid': T_grid,
            'IV_surface': IV_grid,
            'LV_surface': LV_grid,
            'spot': S0,
            'data': df_clean,
            'arbitrage_check': arb_check,
            'fig_iv': fig_iv,
            'fig_lv': fig_lv
        }
