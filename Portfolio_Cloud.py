# Portfolio_5.py — Enhanced EVM Project Management Suite with targeted fixes
# Requirements:
#   pip install streamlit pandas matplotlib python-dateutil requests
#
# Notes:
# - Data now stored in session state (no database files needed)
# - S-curve uses Beta(2,2) closed-form CDF (no SciPy needed)
# - Charts use matplotlib; solid to Data Date, dotted to BAC/EAC at Forecast Completion
# - Enhanced UX with reorganized sidebar and professional styling

from __future__ import annotations
import io
import json
import logging
import math
import os
import re
import textwrap
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import streamlit as st
from dateutil import parser as date_parser

# =============================================================================
# CONSTANTS
# =============================================================================

# Application constants
APP_TITLE = "Project Portfolio Intelligence Suite 📊"
DEFAULT_DATASET_TABLE = "dataset"
RESULTS_TABLE = "evm_results"
CONFIG_TABLE = "app_config"

# Calculation constants
DAYS_PER_MONTH = 30.44
INTEGRATION_STEPS = 200
MAX_TIMEOUT_SECONDS = 120
MIN_TIMEOUT_SECONDS = 10
EXCEL_ORDINAL_BASE = datetime(1899, 12, 30)

# Config file paths for local storage
CONFIG_DIR = Path.home() / ".portfolio_suite"
MODEL_CONFIG_FILE = CONFIG_DIR / "model_config.json"

# Validation patterns
VALID_TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
VALID_COLUMN_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\s-]+$')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page config with professional styling
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .section-header {
        font-weight: 600;
        font-size: 1.1rem;
        color: #495057;
        margin-bottom: 0.5rem;
        padding-bottom: 0.2rem;
        border-bottom: 2px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SECURITY AND VALIDATION UTILITIES
# =============================================================================

def validate_table_name(table_name: str) -> bool:
    """Validate table name to prevent SQL injection."""
    if not table_name or len(table_name) > 50:
        return False
    return bool(VALID_TABLE_NAME_PATTERN.match(table_name))

def validate_column_name(column_name: str) -> bool:
    """Validate column name to prevent SQL injection."""
    if not column_name or len(column_name) > 100:
        return False
    return bool(VALID_COLUMN_NAME_PATTERN.match(column_name))

def sanitize_sql_identifier(identifier: str) -> str:
    """Sanitize SQL identifier by quoting it properly."""
    if not identifier:
        raise ValueError("Empty identifier")
    cleaned = identifier.replace('"', '').replace('[', '').replace(']', '')
    return f'"{cleaned}"'

def validate_numeric_input(value: Any, field_name: str, min_val: float = None, max_val: float = None) -> float:
    """Validate numeric input with proper error handling."""
    try:
        num_val = float(value)
        if math.isnan(num_val) or math.isinf(num_val):
            raise ValueError(f"{field_name} cannot be NaN or infinite")
        if min_val is not None and num_val < min_val:
            raise ValueError(f"{field_name} must be >= {min_val}")
        if max_val is not None and num_val > max_val:
            raise ValueError(f"{field_name} must be <= {max_val}")
        return num_val
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid {field_name}: {e}")

# =============================================================================
# LOCAL STORAGE UTILITIES
# =============================================================================

def ensure_config_dir():
    """Ensure config directory exists."""
    CONFIG_DIR.mkdir(exist_ok=True)


def save_model_config(provider: str, model: str):
    """Save model configuration."""
    try:
        ensure_config_dir()
        config = {"provider": provider, "model": model}
        with open(MODEL_CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        logger.error(f"Failed to save model config: {e}")

def load_model_config() -> Dict[str, str]:
    """Load model configuration."""
    try:
        if MODEL_CONFIG_FILE.exists():
            with open(MODEL_CONFIG_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load model config: {e}")
    return {"provider": "OpenAI", "model": "gpt-4o-mini"}

# =============================================================================
# ENHANCED UTILITIES
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, avoiding division by zero."""
    try:
        if abs(denominator) < 1e-10:
            return default
        result = numerator / denominator
        if math.isinf(result) or math.isnan(result):
            return default
        return result
    except (ZeroDivisionError, TypeError, ValueError):
        return default

def is_valid_finite_number(value: Any) -> bool:
    """Check if value is a valid finite number."""
    try:
        num_val = float(value)
        return math.isfinite(num_val)
    except (ValueError, TypeError):
        return False

def parse_date_any(x):
    """Robust date parser with improved error handling."""
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime(x.year, x.month, x.day)

    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            if x < 1 or x > 2958465:
                raise ValueError(f"Excel ordinal {x} out of valid range")
            return datetime.fromordinal(EXCEL_ORDINAL_BASE.toordinal() + int(x))
        except (ValueError, OverflowError) as e:
            logger.warning(f"Failed to parse Excel ordinal {x}: {e}")
            raise ValueError(f"Invalid Excel date ordinal: {x}")

    if not isinstance(x, str):
        raise ValueError(f"Unrecognized date format: {x!r}")

    s = x.strip()
    if not s:
        raise ValueError("Empty date string")

    s = s.replace("–", "-").replace("—", "-").replace(",", " ")
    s = re.sub(r"\s+", " ", s)

    date_formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
        "%d-%m-%y", "%d/%m/%y", "%m/%d/%y", "%d-%b-%Y", "%d %b %Y",
        "%b %d %Y", "%d-%b-%y", "%d %b %y", "%b %d %y",
        "%d-%B-%Y", "%d %B %Y", "%B %d %Y", "%d-%B-%y", "%d %B %y", "%B %d %y",
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    try:
        return date_parser.parse(s, dayfirst=True, yearfirst=False)
    except Exception as e:
        raise ValueError(f"Unable to parse date '{x}': {e}")

def format_currency(amount: float, symbol: str, postfix: str = "", decimals: int = 2) -> str:
    """Enhanced currency formatting with comma separators and postfix options.
    
    The postfix indicates the unit the user's figures are already in.
    For example, if user enters 1200 with postfix 'Million', we display '1,200 M'
    because the user's figure is already in millions.
    
    Optimized for displaying up to 9-digit numbers with proper formatting.
    """
    if not is_valid_finite_number(amount):
        return "—"
    
    try:
        # Handle postfix labels (no scaling - user's figures are already in specified units)
        if postfix.lower() == "thousand":
            postfix_label = "K"
        elif postfix.lower() == "million":
            postfix_label = "M"
        elif postfix.lower() == "billion":
            postfix_label = "B"
        else:
            postfix_label = ""
        
        # Format with commas for better readability of large numbers (up to 9 digits)
        formatted_amount = f"{amount:,.{decimals}f}"
        
        # Ensure we don't exceed screen width - compact display for very long numbers
        if len(formatted_amount) > 12 and decimals > 0:
            # Try with 0 decimals if too long
            formatted_amount = f"{amount:,.0f}"
            
        result = f"{symbol} {formatted_amount}"
        if postfix_label:
            result += f" {postfix_label}"
            
        return result
    except (ValueError, OverflowError):
        return "—"

def format_percentage(value: float) -> str:
    """Format percentage values consistently with 1 decimal place."""
    if not is_valid_finite_number(value):
        return "—"
    return f"{value:.1f}%"

def format_performance_index(value: float) -> str:
    """Format performance indices (CPI, SPI, SPIe) consistently with 2 decimal places."""
    if not is_valid_finite_number(value):
        return "N/A"
    return f"{value:.2f}"

def maybe(val, default="—"):
    """Return default if value is None or invalid."""
    if val is None:
        return default
    if isinstance(val, (int, float)) and not is_valid_finite_number(val):
        return default
    return val

def create_gauge_chart(value: float, title: str, min_val: float = 0.4, max_val: float = 1.1) -> None:
    """Create a simple gauge chart for performance indices with scale 0.4 to 1.1."""
    try:
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Create semicircular gauge
        theta1, theta2 = 0, 180  # Semicircle from 0 to 180 degrees
        
        # Background arc
        arc = patches.Arc((0.5, 0), 1, 1, angle=0, theta1=theta1, theta2=theta2, linewidth=20, color='lightgray')
        ax.add_patch(arc)
        
        # Color zones based on new scale (0.4 to 1.1)
        range_val = max_val - min_val
        
        # Red zone (0.4-0.9)
        red_start = 0
        red_end = ((0.9 - min_val) / range_val) * 180
        red_arc = patches.Arc((0.5, 0), 1, 1, angle=0, theta1=red_start, theta2=red_end, linewidth=20, color='#ff4444')
        ax.add_patch(red_arc)
        
        # Yellow zone (0.9-1.0)
        yellow_start = red_end
        yellow_end = ((1.0 - min_val) / range_val) * 180
        yellow_arc = patches.Arc((0.5, 0), 1, 1, angle=0, theta1=yellow_start, theta2=yellow_end, linewidth=20, color='#ffaa00')
        ax.add_patch(yellow_arc)
        
        # Green zone (1.0-1.1)
        green_start = yellow_end
        green_end = 180
        green_arc = patches.Arc((0.5, 0), 1, 1, angle=0, theta1=green_start, theta2=green_end, linewidth=20, color='#44aa44')
        ax.add_patch(green_arc)
        
        # Value indicator
        if is_valid_finite_number(value):
            # Clamp value to our scale range
            value_clamped = max(min_val, min(value, max_val))
            # Calculate angle based on position within our range
            value_position = (value_clamped - min_val) / range_val
            value_angle = value_position * 180
            value_rad = np.radians(value_angle)
            
            # Pointer
            ax.arrow(0.5, 0, 0.4 * np.cos(value_rad), 0.4 * np.sin(value_rad),
                    head_width=0.03, head_length=0.02, fc='black', ec='black', linewidth=2)
        
        # Labels
        ax.text(0.5, -0.2, title, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0.5, -0.35, f'{format_performance_index(value)}' if is_valid_finite_number(value) else 'N/A', 
               ha='center', va='center', fontsize=14, fontweight='bold', color='blue')
        
        # Scale labels based on new range
        ax.text(0, -0.05, f'{min_val}', ha='center', va='center', fontsize=10)
        ax.text(0.25, 0.25, f'{(min_val + range_val*0.25):.1f}', ha='center', va='center', fontsize=8)
        ax.text(0.5, 0.45, '1.0', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(0.75, 0.25, f'{(min_val + range_val*0.75):.1f}', ha='center', va='center', fontsize=8)
        ax.text(1, -0.05, f'{max_val}', ha='center', va='center', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.4, 0.6)
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig
        
    except Exception as e:
        logger.error(f"Gauge chart creation failed: {e}")
        return None

# =============================================================================
# PORTFOLIO ANALYTICS FUNCTIONS
# =============================================================================

def calculate_portfolio_summary(batch_results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive portfolio analytics from batch results."""
    if batch_results_df.empty:
        return {}
    
    # Filter out error rows - safe boolean indexing
    if 'error' in batch_results_df.columns:
        valid_results = batch_results_df[batch_results_df['error'].isna()].copy()
    else:
        valid_results = batch_results_df.copy()
    
    if valid_results.empty:
        return {"error": "No valid project results found"}
    
    try:
        # Portfolio totals
        total_bac = valid_results['bac'].sum() if 'bac' in valid_results else 0
        total_ac = valid_results['ac'].sum() if 'ac' in valid_results else 0
        total_pv = valid_results['planned_value'].sum() if 'planned_value' in valid_results else 0
        total_ev = valid_results['earned_value'].sum() if 'earned_value' in valid_results else 0
        
        # Portfolio performance indices (correct weighted calculation)
        portfolio_cpi = safe_divide(total_ev, total_ac, 0.0)
        portfolio_spi = safe_divide(total_ev, total_pv, 0.0)
        
        # Calculate weighted average CPI and SPI (CPI*BAC / Sum BAC, SPI*BAC / Sum BAC)
        total_cpi_weighted = 0
        total_spi_weighted = 0
        for _, row in valid_results.iterrows():
            project_bac = row.get('bac', 0)
            project_cpi = row.get('cost_performance_index', 0)
            project_spi = row.get('schedule_performance_index', 0)
            
            if is_valid_finite_number(project_cpi) and is_valid_finite_number(project_bac) and project_bac > 0:
                total_cpi_weighted += project_cpi * project_bac
            if is_valid_finite_number(project_spi) and is_valid_finite_number(project_bac) and project_bac > 0:
                total_spi_weighted += project_spi * project_bac
        
        weighted_avg_cpi = safe_divide(total_cpi_weighted, total_bac, 0.0)
        weighted_avg_spi = safe_divide(total_spi_weighted, total_bac, 0.0)
        
        # Performance quadrant analysis
        quadrants = {
            'on_budget_on_schedule': 0,
            'on_budget_behind_schedule': 0,
            'over_budget_on_schedule': 0,
            'over_budget_behind_schedule': 0
        }
        
        quadrant_budgets = {
            'on_budget_on_schedule': 0,
            'on_budget_behind_schedule': 0,
            'over_budget_on_schedule': 0,
            'over_budget_behind_schedule': 0
        }
        
        for _, row in valid_results.iterrows():
            cpi = row.get('cost_performance_index', 0)
            spi = row.get('schedule_performance_index', 0)
            project_bac = row.get('bac', 0)
            
            if cpi >= 1.0 and spi >= 1.0:
                quadrants['on_budget_on_schedule'] += 1
                quadrant_budgets['on_budget_on_schedule'] += project_bac
            elif cpi >= 1.0 and spi < 1.0:
                quadrants['on_budget_behind_schedule'] += 1
                quadrant_budgets['on_budget_behind_schedule'] += project_bac
            elif cpi < 1.0 and spi >= 1.0:
                quadrants['over_budget_on_schedule'] += 1
                quadrant_budgets['over_budget_on_schedule'] += project_bac
            else:
                quadrants['over_budget_behind_schedule'] += 1
                quadrant_budgets['over_budget_behind_schedule'] += project_bac
        
        total_projects = len(valid_results)
        
        # Calculate percentages
        quadrant_percentages = {
            key: safe_divide(count, total_projects) * 100 
            for key, count in quadrants.items()
        }
        
        budget_percentages = {
            key: safe_divide(budget, total_bac) * 100 
            for key, budget in quadrant_budgets.items()
        }
        
        return {
            'total_projects': total_projects,
            'total_bac': total_bac,
            'total_ac': total_ac,
            'total_pv': total_pv,
            'total_ev': total_ev,
            'portfolio_cpi': portfolio_cpi,
            'portfolio_spi': portfolio_spi,
            'weighted_avg_cpi': weighted_avg_cpi,
            'weighted_avg_spi': weighted_avg_spi,
            'quadrants': quadrants,
            'quadrant_percentages': quadrant_percentages,
            'quadrant_budgets': quadrant_budgets,
            'budget_percentages': budget_percentages,
            'average_progress': valid_results['percent_complete'].mean() if 'percent_complete' in valid_results else 0
        }
    
    except Exception as e:
        logger.error(f"Portfolio summary calculation failed: {e}")
        return {"error": str(e)}

def is_project_table(table_name: str) -> bool:
    """Check if a table contains project data by examining required columns."""
    try:
        df = load_table(table_name)
        required_columns = ['Project ID', 'BAC', 'AC', 'Plan Start', 'Plan Finish']
        
        # Check if table has project-like columns (case insensitive)
        df_columns_lower = [col.lower() for col in df.columns]
        
        project_indicators = 0
        for req_col in required_columns:
            for col in df_columns_lower:
                if req_col.lower().replace(' ', '').replace('_', '') in col.replace(' ', '').replace('_', ''):
                    project_indicators += 1
                    break
        
        # If it has at least 3 out of 5 required indicators, consider it a project table
        return project_indicators >= 3
        
    except Exception:
        return False

def save_column_mapping(table_name: str, mapping: Dict[str, str]):
    """Save column mapping for a specific table persistently."""
    try:
        ensure_config_dir()
        mapping_file = CONFIG_DIR / f"mapping_{table_name}.json"
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f)
    except Exception as e:
        logger.error(f"Failed to save column mapping for {table_name}: {e}")

def load_column_mapping(table_name: str) -> Dict[str, str]:
    """Load column mapping for a specific table."""
    try:
        mapping_file = CONFIG_DIR / f"mapping_{table_name}.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load column mapping for {table_name}: {e}")
    return {}

# =============================================================================
# SESSION STATE DATA FUNCTIONS (formerly database functions)
# =============================================================================

def ensure_db():
    """Ensure session state is initialized (replaces database initialization)."""
    try:
        # Initialize session state data structures if not already done
        if "data_df" not in st.session_state:
            st.session_state.data_df = None
        if "config_dict" not in st.session_state:
            st.session_state.config_dict = {}
        if "data_loaded" not in st.session_state:
            st.session_state.data_loaded = False
        if "original_filename" not in st.session_state:
            st.session_state.original_filename = None
        if "file_type" not in st.session_state:
            st.session_state.file_type = None
        
        logger.info("Session state initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize session state: {e}")
        raise

@st.cache_data(show_spinner=False)
def list_session_tables() -> List[str]:
    """List available tables from session state."""
    try:
        tables = []
        
        # Add main data table if exists and has data
        if (st.session_state.data_df is not None and 
            not st.session_state.data_df.empty):
            tables.append(DEFAULT_DATASET_TABLE)
        
        # Add tables from config dict
        if ("tables" in st.session_state.config_dict and 
            st.session_state.config_dict["tables"]):
            config_tables = [name for name in st.session_state.config_dict["tables"].keys() 
                           if validate_table_name(name)]
            tables.extend(config_tables)
        
        # Remove duplicates and sort
        return sorted(list(set(tables)))
        
    except Exception as e:
        logger.error(f"Failed to list session tables: {e}")
        return []

@st.cache_data(show_spinner=False)
def load_table(table_name: str) -> pd.DataFrame:
    """Load table from session state (replaces SQLite loading)."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    try:
        # Load main dataset table
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is not None:
                return st.session_state.data_df.copy()
            else:
                return pd.DataFrame()
        
        # Load from config tables
        if ("tables" in st.session_state.config_dict and 
            table_name in st.session_state.config_dict["tables"]):
            table_records = st.session_state.config_dict["tables"][table_name]
            return pd.DataFrame(table_records)
        
        # Table not found
        logger.warning(f"Table {table_name} not found in session state")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Failed to load session table {table_name}: {e}")
        raise

def save_table_replace(df: pd.DataFrame, table_name: str):
    """Save table to session state (replaces SQLite saving)."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    try:
        # Save to appropriate session location
        if table_name == DEFAULT_DATASET_TABLE:
            st.session_state.data_df = df.copy()
            st.session_state.data_loaded = True
        else:
            # Save to config tables
            if "tables" not in st.session_state.config_dict:
                st.session_state.config_dict["tables"] = {}
            
            st.session_state.config_dict["tables"][table_name] = df.to_dict('records')
        
        logger.info(f"Saved {len(df)} rows to session table {table_name}")
        
    except Exception as e:
        logger.error(f"Failed to save session table {table_name}: {e}")
        raise

def delete_table(table_name: str):
    """Delete a table from session state (replaces SQLite table deletion)."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    try:
        table_deleted = False
        
        # Handle main dataset table
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is not None:
                st.session_state.data_df = None
                st.session_state.data_loaded = False
                table_deleted = True
        else:
            # Handle tables in config_dict
            if ("tables" in st.session_state.config_dict and 
                table_name in st.session_state.config_dict["tables"]):
                del st.session_state.config_dict["tables"][table_name]
                table_deleted = True
        
        if not table_deleted:
            raise ValueError(f"Table '{table_name}' not found")
        
        logger.info(f"Deleted session table {table_name}")
        
    except Exception as e:
        logger.error(f"Failed to delete session table {table_name}: {e}")
        raise

def insert_project_record(project_data: Dict[str, Any], table_name: str = DEFAULT_DATASET_TABLE):
    """Insert a new project record into session state."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    try:
        # Initialize session data if needed
        if st.session_state.data_df is None:
            st.session_state.data_df = pd.DataFrame({
                "Project ID": [], "Project": [], "Organization": [], "Project Manager": [],
                "BAC": [], "AC": [], "Plan Start": [], "Plan Finish": [],
                "Use_Manual_PV": [], "Manual_PV": []
            })
        
        # Get the target DataFrame
        if table_name == DEFAULT_DATASET_TABLE:
            existing_df = st.session_state.data_df
            # Ensure all expected columns exist
            expected_cols = ["Project ID", "Project", "Organization", "Project Manager",
                           "BAC", "AC", "Plan Start", "Plan Finish", "Use_Manual_PV", "Manual_PV"]
            for col in expected_cols:
                if col not in existing_df.columns:
                    existing_df[col] = None
        else:
            # Handle other tables from config_dict
            if "tables" not in st.session_state.config_dict:
                st.session_state.config_dict["tables"] = {}
            
            if table_name in st.session_state.config_dict["tables"]:
                existing_df = pd.DataFrame(st.session_state.config_dict["tables"][table_name])
            else:
                existing_df = pd.DataFrame({
                    "Project ID": [], "Project": [], "Organization": [], "Project Manager": [],
                    "BAC": [], "AC": [], "Plan Start": [], "Plan Finish": [],
                    "Use_Manual_PV": [], "Manual_PV": []
                })
        
        # Check for duplicate Project ID
        if not existing_df.empty:
            if str(project_data['Project ID']) in existing_df['Project ID'].astype(str).values:
                raise ValueError("Project ID already exists")
        
        # Create new record DataFrame
        new_record_df = pd.DataFrame([project_data])
        
        # Append new record using concat
        updated_df = pd.concat([existing_df, new_record_df], ignore_index=True)
        
        # Save back to session state
        if table_name == DEFAULT_DATASET_TABLE:
            st.session_state.data_df = updated_df
            st.session_state.data_loaded = True
        else:
            st.session_state.config_dict["tables"][table_name] = updated_df.to_dict('records')
        
        logger.info(f"Inserted project {project_data['Project ID']} into session")
        
    except Exception as e:
        logger.error(f"Failed to insert project into session: {e}")
        raise

def update_project_record(project_id: str, project_data: Dict[str, Any], table_name: str = DEFAULT_DATASET_TABLE):
    """Update an existing project record in session state."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    try:
        # Load DataFrame from session state
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is None or st.session_state.data_df.empty:
                raise ValueError("No data available to update")
            df = st.session_state.data_df.copy()
            # Ensure all expected columns exist
            expected_cols = ["Project ID", "Project", "Organization", "Project Manager",
                           "BAC", "AC", "Plan Start", "Plan Finish", "Use_Manual_PV", "Manual_PV"]
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = None
        else:
            # Handle other tables from config
            if ("tables" not in st.session_state.config_dict or 
                table_name not in st.session_state.config_dict["tables"]):
                raise ValueError(f"Table {table_name} not found")
            
            table_records = st.session_state.config_dict["tables"][table_name]
            df = pd.DataFrame(table_records)
        
        # Create mask for WHERE clause equivalent
        mask = df['Project ID'].astype(str) == str(project_id)
        
        # Check if project exists
        if not mask.any():
            raise ValueError(f"Project ID '{project_id}' not found")
        
        # Update records using .loc
        for key, value in project_data.items():
            if key in df.columns:
                df.loc[mask, key] = value
            else:
                logger.warning(f"Column '{key}' not found in table, skipping update")
        
        # Save updated DataFrame back to session state
        if table_name == DEFAULT_DATASET_TABLE:
            st.session_state.data_df = df
        else:
            st.session_state.config_dict["tables"][table_name] = df.to_dict('records')
        
        logger.info(f"Updated project {project_id} in session state")
        
    except Exception as e:
        logger.error(f"Failed to update project in session: {e}")
        raise

def delete_project_record(project_id: str, table_name: str = DEFAULT_DATASET_TABLE):
    """Delete a project record from session state."""
    if not validate_table_name(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    try:
        # Load DataFrame from session state
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is None or st.session_state.data_df.empty:
                raise ValueError("No data available")
            df = st.session_state.data_df.copy()
            # Ensure Project ID column exists for deletion
            if "Project ID" not in df.columns:
                raise ValueError("Project ID column not found in data")
        else:
            # Handle other tables from config
            if ("tables" not in st.session_state.config_dict or 
                table_name not in st.session_state.config_dict["tables"]):
                raise ValueError(f"Table {table_name} not found")
            
            table_records = st.session_state.config_dict["tables"][table_name]
            df = pd.DataFrame(table_records)
        
        # Find rows to delete
        mask = df['Project ID'].astype(str) == str(project_id)
        matching_indices = df.index[mask]
        
        if len(matching_indices) == 0:
            raise ValueError(f"Project ID '{project_id}' not found")
        
        # Perform deletion using .drop()
        df_after_delete = df.drop(matching_indices, axis=0).reset_index(drop=True)
        
        # Save updated DataFrame back to session state
        if table_name == DEFAULT_DATASET_TABLE:
            st.session_state.data_df = df_after_delete
        else:
            st.session_state.config_dict["tables"][table_name] = df_after_delete.to_dict('records')
        
        logger.info(f"Deleted project {project_id} from session ({len(matching_indices)} record(s))")
        
    except Exception as e:
        logger.error(f"Failed to delete project from session: {e}")
        raise

def get_project_record(project_id: str, table_name: str = DEFAULT_DATASET_TABLE) -> Optional[Dict]:
    """Get a specific project record from session state."""
    try:
        # Load DataFrame from session state
        if table_name == DEFAULT_DATASET_TABLE:
            if st.session_state.data_df is None or st.session_state.data_df.empty:
                return None
            df = st.session_state.data_df
        else:
            # Handle other tables from config
            if ("tables" not in st.session_state.config_dict or 
                table_name not in st.session_state.config_dict["tables"]):
                return None
            
            table_records = st.session_state.config_dict["tables"][table_name]
            df = pd.DataFrame(table_records)
        
        if df.empty:
            return None
        
        # Filter using .loc for WHERE clause equivalent
        mask = df['Project ID'].astype(str) == str(project_id)
        matching_rows = df.loc[mask]
        
        # Check if any matches found
        if matching_rows.empty:
            return None
        
        # Return first match as dictionary
        return matching_rows.iloc[0].to_dict()
        
    except Exception as e:
        logger.error(f"Failed to get project {project_id} from session: {e}")
        return None

def create_demo_data():
    """Create empty demo data with proper schema at startup."""
    try:
        ensure_db()
        
        # Check if demo data already exists and has content
        try:
            existing_df = load_table(DEFAULT_DATASET_TABLE)
            if len(existing_df) > 0:
                logger.info("Demo data already exists")
                return  # Don't overwrite existing data
        except:
            pass  # Table doesn't exist yet
        
        # Create empty demo data with proper schema
        demo_data = {
            "Project ID": [],
            "Project": [],
            "Organization": [],
            "Project Manager": [],
            "BAC": [],
            "AC": [],
            "Plan Start": [],
            "Plan Finish": [],
            "Use_Manual_PV": [],
            "Manual_PV": []
        }
        
        demo_df = pd.DataFrame(demo_data)
        save_table_replace(demo_df, DEFAULT_DATASET_TABLE)
        logger.info("Demo data created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create demo data: {e}")
        # Don't raise - application should continue even if demo creation fails

# =============================================================================
# EVM CALCULATION ENGINE (UNCHANGED)
# =============================================================================

def scurve_cdf(x: float, alpha: float = 2.0, beta: float = 2.0) -> float:
    """S-curve CDF with improved error handling and validation."""
    try:
        x = max(0.0, min(1.0, float(x)))
        alpha = max(0.1, float(alpha))
        beta = max(0.1, float(beta))
        
        # Closed-form for Beta(2,2)
        if abs(alpha - 2.0) < 1e-9 and abs(beta - 2.0) < 1e-9:
            return 3 * x * x - 2 * x * x * x
        
        # Improved numeric integration for other parameters
        if x == 0.0:
            return 0.0
        if x == 1.0:
            return 1.0
            
        n = max(INTEGRATION_STEPS, 100)
        xs = np.linspace(0, x, n + 1)
        pdf_vals = (xs**(alpha-1)) * ((1 - xs)**(beta-1))
        
        # Use gamma function with error handling
        try:
            B = math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
        except (ValueError, OverflowError):
            logger.warning(f"Gamma function overflow for alpha={alpha}, beta={beta}")
            return 3 * x * x - 2 * x * x * x  # Fallback to Beta(2,2)
            
        result = float(np.trapz(pdf_vals, xs) / B)
        return max(0.0, min(1.0, result))
        
    except Exception as e:
        logger.error(f"S-curve CDF calculation failed: {e}")
        return 3 * x * x - 2 * x * x * x  # Safe fallback

def calculate_durations(plan_start, plan_finish, data_date) -> tuple[float, float]:
    """Calculate durations with improved error handling."""
    try:
        ps = parse_date_any(plan_start)
        pf = parse_date_any(plan_finish)
        dd = parse_date_any(data_date)
        
        duration_to_date = max(((dd - ps).days / DAYS_PER_MONTH), 0.0)
        original_duration = max(((pf - ps).days / DAYS_PER_MONTH), 0.0)
        
        return round(duration_to_date, 2), round(original_duration, 2)
        
    except Exception as e:
        logger.error(f"Duration calculation failed: {e}")
        return 0.0, 0.0

def calculate_present_value(ac, duration_months, annual_inflation_rate) -> float:
    """Calculate present value with improved validation."""
    try:
        ac = validate_numeric_input(ac, "AC", min_val=0.0)
        duration_months = validate_numeric_input(duration_months, "Duration", min_val=0.0)
        annual_inflation_rate = validate_numeric_input(annual_inflation_rate, "Inflation Rate", min_val=0.0, max_val=1.0)
        
        if annual_inflation_rate == 0 or duration_months == 0:
            return round(ac, 2)
            
        monthly_rate = annual_inflation_rate / 12.0
        pmt = ac / max(duration_months, 1e-9)
        
        try:
            factor = (1 - (1 + monthly_rate) ** (-duration_months)) / monthly_rate
            return round(max(pmt * factor, 0.0), 2)
        except (OverflowError, ValueError):
            logger.warning("PV calculation overflow, returning AC")
            return round(ac, 2)
            
    except ValueError as e:
        logger.error(f"Present value calculation failed: {e}")
        return 0.0

def calculate_pv_linear(bac, current_duration, total_duration) -> float:
    """Calculate planned value (linear) with validation."""
    try:
        bac = validate_numeric_input(bac, "BAC", min_val=0.0)
        current_duration = validate_numeric_input(current_duration, "Current Duration", min_val=0.0)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)
        
        if total_duration <= 0:
            return round(bac, 2) if current_duration > 0 else 0.0
        if current_duration >= total_duration:
            return round(bac, 2)
            
        progress_ratio = max(min(current_duration / total_duration, 1.0), 0.0)
        return round(bac * progress_ratio, 2)
        
    except ValueError as e:
        logger.error(f"Linear PV calculation failed: {e}")
        return 0.0

def calculate_pv_scurve(bac, current_duration, total_duration, alpha=2.0, beta=2.0) -> float:
    """Calculate planned value (S-curve) with validation."""
    try:
        bac = validate_numeric_input(bac, "BAC", min_val=0.0)
        current_duration = validate_numeric_input(current_duration, "Current Duration", min_val=0.0)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)
        
        if total_duration <= 0:
            return round(bac, 2) if current_duration > 0 else 0.0
        if current_duration >= total_duration:
            return round(bac, 2)
            
        progress_ratio = max(min(current_duration / total_duration, 1.0), 0.0)
        return round(bac * scurve_cdf(progress_ratio, alpha, beta), 2)
        
    except ValueError as e:
        logger.error(f"S-curve PV calculation failed: {e}")
        return 0.0

def calculate_evm_metrics(bac, ac, present_value, planned_value) -> Dict[str, float]:
    """Calculate EVM metrics with improved error handling."""
    try:
        bac = validate_numeric_input(bac, "BAC", min_val=0.0)
        ac = validate_numeric_input(ac, "AC", min_val=0.0)
        present_value = validate_numeric_input(present_value, "Present Value", min_val=0.0)
        planned_value = validate_numeric_input(planned_value, "Planned Value", min_val=0.0)
        
        percent_complete = safe_divide(present_value, bac, 0.0)
        earned_value = bac * percent_complete
        cost_variance = earned_value - ac
        schedule_variance = earned_value - planned_value
        
        cpi = safe_divide(earned_value, ac, 0.0)
        spi = safe_divide(earned_value, planned_value, 0.0)
        
        eac = safe_divide(bac, cpi, float("inf")) if cpi > 1e-10 else float("inf")
        etc = eac - ac if is_valid_finite_number(eac) else float("inf")
        vac = bac - eac if is_valid_finite_number(eac) else float("-inf")
        
        return {
            'percent_complete': round(percent_complete * 100, 2),
            'earned_value': round(earned_value, 2),
            'cost_variance': round(cost_variance, 2),
            'schedule_variance': round(schedule_variance, 2),
            'cost_performance_index': round(cpi, 3) if is_valid_finite_number(cpi) else 0.0,
            'schedule_performance_index': round(spi, 3) if is_valid_finite_number(spi) else 0.0,
            'estimate_at_completion': round(eac, 2) if is_valid_finite_number(eac) else float("inf"),
            'estimate_to_complete': round(etc, 2) if is_valid_finite_number(etc) else float("inf"),
            'variance_at_completion': round(vac, 2) if is_valid_finite_number(vac) else float("-inf"),
        }
        
    except ValueError as e:
        logger.error(f"EVM metrics calculation failed: {e}")
        return {
            'percent_complete': 0.0, 'earned_value': 0.0, 'cost_variance': 0.0,
            'schedule_variance': 0.0, 'cost_performance_index': 0.0,
            'schedule_performance_index': 0.0, 'estimate_at_completion': float("inf"),
            'estimate_to_complete': float("inf"), 'variance_at_completion': float("-inf")
        }

def find_earned_schedule_linear(earned_value, bac, total_duration) -> float:
    """Find earned schedule (linear) with validation."""
    try:
        earned_value = validate_numeric_input(earned_value, "Earned Value", min_val=0.0)
        bac = validate_numeric_input(bac, "BAC", min_val=0.01)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)
        
        es = safe_divide(earned_value, bac) * total_duration
        return round(max(min(es, total_duration), 0.0), 2)
        
    except ValueError as e:
        logger.error(f"Linear earned schedule calculation failed: {e}")
        return 0.0

def find_earned_schedule_scurve(earned_value, bac, total_duration, alpha=2.0, beta=2.0) -> float:
    """Find earned schedule (S-curve) with improved algorithm."""
    try:
        earned_value = validate_numeric_input(earned_value, "Earned Value", min_val=0.0)
        bac = validate_numeric_input(bac, "BAC", min_val=0.01)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)
        
        if total_duration <= 0:
            return 0.0
            
        target = max(min(earned_value / bac, 1.0), 0.0)
        
        if target == 0.0:
            return 0.0
        if target == 1.0:
            return total_duration
        
        # Binary search for more accurate results
        low, high = 0.0, total_duration
        tolerance = 0.01
        max_iterations = 100
        
        for _ in range(max_iterations):
            mid = (low + high) / 2
            ratio = mid / total_duration
            cdf = scurve_cdf(ratio, alpha, beta)
            
            if abs(cdf - target) < tolerance / total_duration:
                return round(mid, 2)
            elif cdf < target:
                low = mid
            else:
                high = mid
        
        return round((low + high) / 2, 2)
        
    except ValueError as e:
        logger.error(f"S-curve earned schedule calculation failed: {e}")
        return 0.0

def add_months_approx(start_dt: datetime, months: int) -> datetime:
    """Add months with improved calendar handling."""
    try:
        months = int(months)
        y = start_dt.year + (start_dt.month - 1 + months) // 12
        m = (start_dt.month - 1 + months) % 12 + 1
        
        # Handle leap years properly
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if m == 2 and ((y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)):
            days_in_month[1] = 29
            
        d = min(start_dt.day, days_in_month[m-1])
        return datetime(y, m, d)
    except (ValueError, OverflowError) as e:
        logger.error(f"Error adding {months} months to {start_dt}: {e}")
        return start_dt

def calculate_earned_schedule_metrics(earned_schedule, actual_duration, total_duration, plan_start) -> Dict[str, Union[float, int, str]]:
    """Calculate earned schedule metrics with validation."""
    try:
        earned_schedule = validate_numeric_input(earned_schedule, "Earned Schedule", min_val=0.0)
        actual_duration = max(validate_numeric_input(actual_duration, "Actual Duration", min_val=0.0), 1e-9)
        total_duration = validate_numeric_input(total_duration, "Total Duration", min_val=0.0)
        ps = parse_date_any(plan_start)
        
        spie = safe_divide(earned_schedule, actual_duration, 1.0)
        forecast_duration = safe_divide(total_duration, spie, float("inf")) if spie > 1e-10 else float("inf")
        
        if is_valid_finite_number(forecast_duration):
            forecast_duration_rounded = max(1, math.ceil(forecast_duration))
            fcomp = add_months_approx(ps, forecast_duration_rounded)
            fcomp_str = fcomp.strftime('%Y-%m-%d')
        else:
            forecast_duration_rounded = None
            fcomp_str = "N/A"
        
        return {
            'earned_schedule': round(earned_schedule, 2),
            'spie': round(spie, 3),
            'forecast_duration': int(forecast_duration_rounded) if forecast_duration_rounded is not None else None,
            'forecast_completion': fcomp_str
        }
        
    except (ValueError, OverflowError) as e:
        logger.error(f"Earned schedule metrics calculation failed: {e}")
        return {
            'earned_schedule': 0.0,
            'spie': 1.0,
            'forecast_duration': None,
            'forecast_completion': "N/A"
        }

@st.cache_data(show_spinner=False)
def perform_complete_evm_analysis(bac, ac, plan_start, plan_finish, data_date,
                                  annual_inflation_rate, curve_type='linear', alpha=2.0, beta=2.0, 
                                  manual_pv=None, use_manual_pv=False) -> Dict[str, Any]:
    """Perform complete EVM analysis with comprehensive error handling."""
    try:
        actual_duration, original_duration = calculate_durations(plan_start, plan_finish, data_date)
        present_value = calculate_present_value(ac, actual_duration, annual_inflation_rate)
        
        # Use manual PV if specified, otherwise calculate automatically
        if use_manual_pv and manual_pv is not None and is_valid_finite_number(manual_pv):
            planned_value = float(manual_pv)
        else:
            if curve_type.lower() == 's-curve':
                planned_value = calculate_pv_scurve(bac, actual_duration, original_duration, alpha, beta)
            else:
                planned_value = calculate_pv_linear(bac, actual_duration, original_duration)
        
        evm_metrics = calculate_evm_metrics(bac, ac, present_value, planned_value)
        
        if curve_type.lower() == 's-curve':
            earned_schedule = find_earned_schedule_scurve(evm_metrics['earned_value'], bac, original_duration, alpha, beta)
        else:
            earned_schedule = find_earned_schedule_linear(evm_metrics['earned_value'], bac, original_duration)
        
        es_metrics = calculate_earned_schedule_metrics(earned_schedule, actual_duration, original_duration, plan_start)
        
        return {
            'bac': float(bac),
            'ac': float(ac),
            'plan_start': parse_date_any(plan_start).strftime('%Y-%m-%d'),
            'plan_finish': parse_date_any(plan_finish).strftime('%Y-%m-%d'),
            'data_date': parse_date_any(data_date).strftime('%Y-%m-%d'),
            'inflation_rate': round(annual_inflation_rate * 100.0, 3),
            'curve_type': curve_type,
            'alpha': alpha,
            'beta': beta,
            'actual_duration_months': actual_duration,
            'original_duration_months': original_duration,
            'present_value': present_value,
            'planned_value': planned_value,
            'use_manual_pv': use_manual_pv,
            'manual_pv': manual_pv if use_manual_pv else None,
            **evm_metrics,
            **es_metrics
        }
        
    except Exception as e:
        logger.error(f"Complete EVM analysis failed: {e}")
        raise

# =============================================================================
# CSV HANDLING
# =============================================================================

def try_read_csv(file, header_option: str) -> pd.DataFrame:
    """Read CSV with improved error handling."""
    encodings = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            file.seek(0)
            if header_option == "First row has headers":
                return pd.read_csv(file, encoding=encoding)
            elif header_option == "No header (assign generic)":
                return pd.read_csv(file, header=None, encoding=encoding)
            else:  # Auto-detect
                return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning(f"CSV read failed with encoding {encoding}: {e}")
            continue
    
    raise ValueError("Unable to read CSV file with any supported encoding")

# =============================================================================
# BATCH CALCULATION FUNCTIONS
# =============================================================================

def perform_batch_calculation(df: pd.DataFrame, column_mapping: Dict[str, str], 
                            curve_type: str, alpha: float, beta: float, 
                            data_date: date, inflation_rate: float) -> pd.DataFrame:
    """Perform EVM calculations on entire dataset."""
    results_list = []
    
    # Ensure we have a clean copy of the dataframe
    df_clean = df.copy().reset_index(drop=True)
    
    for idx, row in df_clean.iterrows():
        try:
            # Safely extract values using column mapping
            pid_col = column_mapping.get('pid_col')
            bac_col = column_mapping.get('bac_col')
            ac_col = column_mapping.get('ac_col')
            st_col = column_mapping.get('st_col')
            fn_col = column_mapping.get('fn_col')
            
            if not all([pid_col, bac_col, ac_col, st_col, fn_col]):
                raise ValueError("Missing required column mappings")
            
            # Extract values with proper error handling
            project_id = str(row[pid_col]) if pid_col in row.index else f"Row_{idx}"
            
            # Stop processing if we encounter blank/empty Project ID
            if pd.isna(row[pid_col]) or str(row[pid_col]).strip() == "":
                logger.info(f"Stopping CSV import at row {idx} due to blank Project ID")
                break
                
            bac = float(row[bac_col]) if bac_col in row.index else 0.0
            ac = float(row[ac_col]) if ac_col in row.index else 0.0
            plan_start = row[st_col] if st_col in row.index else None
            plan_finish = row[fn_col] if fn_col in row.index else None
            
            if plan_start is None or plan_finish is None:
                raise ValueError("Missing plan start or finish date")
            
            # Optional fields - safe extraction
            pname_col = column_mapping.get('pname_col')
            org_col = column_mapping.get('org_col')
            pm_col = column_mapping.get('pm_col')
            
            project_name = str(row[pname_col]) if pname_col and pname_col in row.index else ""
            organization = str(row[org_col]) if org_col and org_col in row.index else ""
            project_manager = str(row[pm_col]) if pm_col and pm_col in row.index else ""
            
            # Clean up nan/null values
            if project_name == 'nan' or pd.isna(project_name):
                project_name = ""
            if organization == 'nan' or pd.isna(organization):
                organization = ""
            if project_manager == 'nan' or pd.isna(project_manager):
                project_manager = ""
            
            # Check for manual PV settings
            use_manual_pv = bool(row.get('Use_Manual_PV', False)) if 'Use_Manual_PV' in row.index else False
            manual_pv_val = row.get('Manual_PV', None) if 'Manual_PV' in row.index else None
            
            # Perform EVM analysis
            results = perform_complete_evm_analysis(
                bac=bac, ac=ac, plan_start=plan_start, plan_finish=plan_finish,
                data_date=data_date, annual_inflation_rate=inflation_rate/100.0,
                curve_type=curve_type, alpha=alpha, beta=beta,
                manual_pv=manual_pv_val, use_manual_pv=use_manual_pv
            )
            
            # Add project info to results
            results['project_id'] = project_id
            results['project_name'] = project_name
            results['organization'] = organization
            results['project_manager'] = project_manager
            results['calculation_date'] = datetime.now().isoformat()
            results['row_index'] = idx
            
            results_list.append(results)
            
        except Exception as e:
            logger.error(f"Batch calculation failed for row {idx}: {e}")
            # Add error record with safe project ID extraction
            try:
                error_pid = str(row[column_mapping.get('pid_col', 'Project ID')]) if column_mapping.get('pid_col') else f"Row_{idx}"
            except:
                error_pid = f"Row_{idx}"
                
            error_result = {
                'project_id': error_pid,
                'error': str(e),
                'calculation_date': datetime.now().isoformat(),
                'row_index': idx
            }
            results_list.append(error_result)
    
    # Convert to DataFrame with better error handling
    try:
        return pd.DataFrame(results_list)
    except Exception as e:
        logger.error(f"Failed to create results DataFrame: {e}")
        # Return minimal error DataFrame
        return pd.DataFrame([{
            'error': 'Failed to process batch calculation results',
            'calculation_date': datetime.now().isoformat()
        }])

# =============================================================================
# LLM INTEGRATION
# =============================================================================


def create_portfolio_executive_summary(portfolio_summary: Dict[str, Any], controls: Dict[str, Any]) -> str:
    """Create executive summary prompt for portfolio health assessment."""
    try:
        currency = controls['currency_symbol']
        postfix = controls['currency_postfix']
        
        # Format key financial metrics
        total_budget = format_currency(portfolio_summary['total_bac'], currency, postfix)
        total_ac = format_currency(portfolio_summary['total_ac'], currency, postfix)
        total_ev = format_currency(portfolio_summary['total_ev'], currency, postfix)
        total_pv = format_currency(portfolio_summary['total_pv'], currency, postfix)
        
        # Calculate key ratios
        portfolio_cpi = portfolio_summary['portfolio_cpi']
        portfolio_spi = portfolio_summary['portfolio_spi']
        weighted_avg_cpi = portfolio_summary['weighted_avg_cpi']
        weighted_avg_spi = portfolio_summary['weighted_avg_spi']
        avg_progress = portfolio_summary['average_progress']
        
        # Budget utilization
        budget_used_pct = safe_divide(portfolio_summary['total_ac'], portfolio_summary['total_bac']) * 100
        value_earned_pct = safe_divide(portfolio_summary['total_ev'], portfolio_summary['total_bac']) * 100
        
        # Performance quadrants
        quadrants = portfolio_summary['quadrants']
        total_projects = portfolio_summary['total_projects']
        
        prompt = f"""
As a senior project portfolio management consultant, analyze this portfolio performance data and provide a comprehensive executive briefing.

PORTFOLIO OVERVIEW:
- Total Projects: {total_projects}
- Total Portfolio Budget: {total_budget}
- Total Actual Cost: {total_ac}
- Total Planned Value: {total_pv}  
- Total Earned Value: {total_ev}
- Average Project Progress: {avg_progress:.1f}%

PERFORMANCE INDICES:
- Portfolio Cost Performance Index (CPI): {portfolio_cpi:.2f}
- Portfolio Schedule Performance Index (SPI): {portfolio_spi:.2f}  
- Weighted Average CPI: {weighted_avg_cpi:.2f}
- Weighted Average SPI: {weighted_avg_spi:.2f}

FINANCIAL UTILIZATION:
- Budget Utilized: {budget_used_pct:.1f}%
- Value Earned: {value_earned_pct:.1f}%

PROJECT PERFORMANCE DISTRIBUTION:
- On Budget & On Schedule: {quadrants['on_budget_on_schedule']} projects ({(quadrants['on_budget_on_schedule']/total_projects)*100:.1f}%)
- On Budget & Behind Schedule: {quadrants['on_budget_behind_schedule']} projects ({(quadrants['on_budget_behind_schedule']/total_projects)*100:.1f}%)
- Over Budget & On Schedule: {quadrants['over_budget_on_schedule']} projects ({(quadrants['over_budget_on_schedule']/total_projects)*100:.1f}%)
- Over Budget & Behind Schedule: {quadrants['over_budget_behind_schedule']} projects ({(quadrants['over_budget_behind_schedule']/total_projects)*100:.1f}%)

Please provide:

1. **EXECUTIVE SUMMARY** (2-3 key bullets on overall portfolio health)

2. **FINANCIAL PERFORMANCE ASSESSMENT** - Cost performance analysis
   - Budget utilization efficiency
   - Value realization status

3. **SCHEDULE PERFORMANCE ASSESSMENT**
   - Schedule adherence analysis
   - Timeline risk assessment

4. **PORTFOLIO HEALTH RATING** (Excellent/Good/Fair/Poor with rationale)

5. **KEY RISKS & CONCERNS** (Top 3 portfolio risks)

6. **STRATEGIC RECOMMENDATIONS** (Top 5 actionable recommendations for portfolio improvement)

7. **IMMEDIATE ACTIONS REQUIRED** (Next 30-60 days priority actions)

Focus on strategic insights and actionable recommendations for senior leadership. Use professional language appropriate for C-suite executives and portfolio sponsors.
"""
        return prompt.strip()
        
    except Exception as e:
        logger.error(f"Portfolio executive summary creation failed: {e}")
        return "Error creating portfolio summary prompt"

def safe_llm_request(provider: str, model: str, api_key: str, 
                    temperature: float, timeout_sec: int, prompt: str) -> str:
    """Make LLM request with comprehensive error handling."""
    try:
        if not prompt.strip():
            return "Error: Empty prompt provided"
        
        if not api_key.strip():
            return "No API key available"
        
        timeout_sec = max(MIN_TIMEOUT_SECONDS, min(timeout_sec, MAX_TIMEOUT_SECONDS))
        temperature = max(0.0, min(temperature, 2.0))
        
        if provider == "OpenAI":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model.strip(),
                "messages": [
                    {"role": "system", "content": "You are a project controls expert specializing in earned value management."},
                    {"role": "user", "content": prompt.strip()}
                ],
                "temperature": float(temperature),
                "max_tokens": 2000
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            response.raise_for_status()
            data = response.json()
            
            if 'choices' not in data or not data['choices']:
                return "Error: Invalid response from OpenAI API"
                
            return data["choices"][0]["message"]["content"].strip()
            
        elif provider == "Gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model.strip()}:generateContent"
            headers = {
                "Content-Type": "application/json"
            }
            params = {
                "key": api_key.strip()
            }
            
            # System message is included in the user prompt for Gemini
            full_prompt = f"You are a project controls expert specializing in earned value management.\n\n{prompt.strip()}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": full_prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": float(temperature),
                    "maxOutputTokens": 2000
                }
            }
            
            response = requests.post(url, headers=headers, params=params, json=payload, timeout=timeout_sec)
            response.raise_for_status()
            data = response.json()
            
            if 'candidates' not in data or not data['candidates']:
                return "Error: Invalid response from Gemini API"
            
            candidate = data['candidates'][0]
            if 'content' not in candidate or 'parts' not in candidate['content']:
                return "Error: Unexpected response format from Gemini API"
                
            return candidate['content']['parts'][0]['text'].strip()
        
        else:
            return f"Error: Unsupported provider '{provider}'"
                
    except requests.exceptions.Timeout:
        return f"Error: Request timed out after {timeout_sec} seconds"
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to the API endpoint"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} - {e.response.text[:200]}"
    except json.JSONDecodeError:
        return "Error: Invalid JSON response from API"
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return f"Error: {str(e)[:200]}"

# =============================================================================
# ENHANCED SIDEBAR COMPONENTS
# =============================================================================

def normalize_dataframe_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Normalize DataFrame columns to standard names for consistent manual entry operations."""
    if df.empty or not column_mapping:
        return df
    
    # Standard column mapping for manual entry compatibility
    standard_mapping = {
        'pid_col': 'Project ID',
        'pname_col': 'Project', 
        'org_col': 'Organization',
        'pm_col': 'Project Manager',
        'bac_col': 'BAC',
        'ac_col': 'AC', 
        'st_col': 'Plan Start',
        'fn_col': 'Plan Finish'
    }
    
    # Create a copy to avoid modifying original
    normalized_df = df.copy()
    
    # Rename columns based on mapping
    rename_dict = {}
    for map_key, original_col in column_mapping.items():
        if map_key in standard_mapping and original_col in normalized_df.columns:
            standard_name = standard_mapping[map_key]
            if original_col != standard_name:  # Only rename if different
                rename_dict[original_col] = standard_name
    
    if rename_dict:
        normalized_df = normalized_df.rename(columns=rename_dict)
        logger.info(f"Normalized column names: {rename_dict}")
    
    return normalized_df

def render_data_source_section():
    """Render A. Data Source section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">A. Data Source</div>', unsafe_allow_html=True)
    
    # Simple data source selection
    data_source = st.radio(
        "Select Data Source",
        options=["Load JSON", "Load CSV"],
        index=0,
        key="data_source_radio",
        help="Choose JSON for comprehensive data with configuration, or CSV for data-only imports"
    )
    
    selected_table = None
    column_mapping = {}
    df = None
    
    # Dynamic file upload info based on selection
    if data_source == "Load JSON":
        st.info("💡 Upload JSON files exported from this application or compatible EVM tools")
        file_types = ["json"]
        help_text = "JSON files contain both data and configuration settings"
    else:  # Load CSV
        st.info("💡 Upload CSV files containing project data. Configuration will be set to defaults")
        file_types = ["csv"]
        help_text = "CSV files contain data only - you'll need to map columns to EVM fields"
    
    # Single unified file uploader
    uploaded_file = st.file_uploader(
        "Choose file",
        type=file_types,
        key="unified_file_uploader",
        help=help_text,
        label_visibility="visible"
    )
    
    # Process uploaded file based on selection
    if uploaded_file is not None:
        # Create a unique identifier for the current file to avoid reprocessing
        current_file_info = (uploaded_file.name, uploaded_file.size)

        # Process only if it's a new file
        if st.session_state.get('processed_file_info') != current_file_info:
            try:
                if data_source == "Load JSON":
                    # JSON processing - direct read
                    df, config_data, filename = load_json_file(uploaded_file)
                    st.success(f"✅ JSON loaded: {filename}")
                    
                    if not df.empty:
                        st.session_state.data_df = df
                        st.session_state.data_loaded = True
                        st.session_state.config_dict.update(config_data)
                        st.session_state.original_filename = filename
                        st.session_state.file_type = 'json'
                    else:
                        st.warning("⚠️ No project data found in JSON file")
                        
                else:  # CSV processing
                    df, filename = load_csv_file(uploaded_file)
                    st.success(f"✅ CSV loaded: {filename}")
                    
                    st.session_state.data_df = df
                    st.session_state.data_loaded = True
                    st.session_state.original_filename = filename
                    st.session_state.file_type = 'csv'
                    st.session_state.config_dict.update({
                        "import_metadata": {
                            "import_date": datetime.now().isoformat(),
                            "source": "csv_import",
                            "original_filename": filename
                        }
                    })
                
                # After successful processing, store the file's info
                st.session_state.processed_file_info = current_file_info
                # Clear cache to ensure UI updates with new data
                st.cache_data.clear()
                st.rerun()

            except Exception as e:
                st.error(f"Failed to load file: {e}")
                # Clear the info if processing fails
                if 'processed_file_info' in st.session_state:
                    del st.session_state['processed_file_info']

    # Use data from session state if available
    if st.session_state.data_df is not None and not st.session_state.data_df.empty:
        df = st.session_state.data_df
        selected_table = DEFAULT_DATASET_TABLE

        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"{len(df)} rows loaded.")
            
        # Get stored column mappings from session state
        stored_mappings = st.session_state.config_dict.get('column_mappings', {})
        column_mapping = render_column_mapping_section(df.columns.tolist(), stored_mappings)
        
        # Store column mappings in session state
        if 'column_mappings' not in st.session_state.config_dict:
            st.session_state.config_dict['column_mappings'] = {}
        st.session_state.config_dict['column_mappings'] = column_mapping
        
        # Normalize DataFrame for manual entry compatibility
        if column_mapping:
            normalized_df = normalize_dataframe_columns(df, column_mapping)
            if st.session_state.data_df is not None:
                st.session_state.data_df = normalized_df

    st.markdown('</div>', unsafe_allow_html=True)
    return df, selected_table, column_mapping

def render_column_mapping_section(columns: List[str], stored_mapping: Dict = None) -> Dict[str, str]:
    """Render column mapping interface."""
    st.markdown("**Column Mapping**")
    
    stored_mapping = stored_mapping or {}
    
    def smart_guess(target: str, keywords: List[str]) -> str:
        # Exact match first
        for col in columns:
            if col.lower().strip() == target.lower():
                return col
        # Contains match
        for keyword in keywords:
            for col in columns:
                if keyword.lower() in col.lower():
                    return col
        return columns[0] if columns else ""

    # Required mappings
    mapping = {}
    mapping['pid_col'] = st.selectbox(
        "🔹 Project ID *",
        columns,
        index=columns.index(stored_mapping.get('pid_col', smart_guess("project id", ["project", "id", "proj"]))) 
        if stored_mapping.get('pid_col') in columns else columns.index(smart_guess("project id", ["project", "id", "proj"])) 
        if smart_guess("project id", ["project", "id", "proj"]) in columns else 0,
        key="map_pid"
    )
    
    mapping['bac_col'] = st.selectbox(
        "🔹 BAC (Budget) *",
        columns,
        index=columns.index(stored_mapping.get('bac_col', smart_guess("bac", ["bac", "budget", "total"])))
        if stored_mapping.get('bac_col') in columns else columns.index(smart_guess("bac", ["bac", "budget", "total"]))
        if smart_guess("bac", ["bac", "budget", "total"]) in columns else 0,
        key="map_bac"
    )
    
    mapping['ac_col'] = st.selectbox(
        "🔹 AC (Actual Cost) *",
        columns,
        index=columns.index(stored_mapping.get('ac_col', smart_guess("ac", ["ac", "actual", "cost", "spent"])))
        if stored_mapping.get('ac_col') in columns else columns.index(smart_guess("ac", ["ac", "actual", "cost", "spent"]))
        if smart_guess("ac", ["ac", "actual", "cost", "spent"]) in columns else 0,
        key="map_ac"
    )
    
    mapping['st_col'] = st.selectbox(
        "🔹 Plan Start *",
        columns,
        index=columns.index(stored_mapping.get('st_col', smart_guess("plan start", ["start", "begin", "plan start"])))
        if stored_mapping.get('st_col') in columns else columns.index(smart_guess("plan start", ["start", "begin", "plan start"]))
        if smart_guess("plan start", ["start", "begin", "plan start"]) in columns else 0,
        key="map_start"
    )
    
    mapping['fn_col'] = st.selectbox(
        "🔹 Plan Finish *",
        columns,
        index=columns.index(stored_mapping.get('fn_col', smart_guess("plan finish", ["finish", "end", "complete", "plan finish"])))
        if stored_mapping.get('fn_col') in columns else columns.index(smart_guess("plan finish", ["finish", "end", "complete", "plan finish"]))
        if smart_guess("plan finish", ["finish", "end", "complete", "plan finish"]) in columns else 0,
        key="map_finish"
    )
    
    # Optional mappings
    st.markdown("_Optional Fields:_")
    none_option = "— None —"
    
    optional_cols = [none_option] + columns
    
    mapping['pname_col'] = st.selectbox(
        "Project",
        optional_cols,
        index=optional_cols.index(stored_mapping.get('pname_col', none_option)) 
        if stored_mapping.get('pname_col') in optional_cols else 0,
        key="map_pname"
    )
    
    mapping['org_col'] = st.selectbox(
        "Organization",
        optional_cols,
        index=optional_cols.index(stored_mapping.get('org_col', none_option))
        if stored_mapping.get('org_col') in optional_cols else 0,
        key="map_org"
    )
    
    mapping['pm_col'] = st.selectbox(
        "Project Manager",
        optional_cols,
        index=optional_cols.index(stored_mapping.get('pm_col', none_option))
        if stored_mapping.get('pm_col') in optional_cols else 0,
        key="map_pm"
    )
    
    # Clean up None values
    for key in ['pname_col', 'org_col', 'pm_col']:
        if mapping[key] == none_option:
            mapping[key] = None
    
    return mapping

def render_manual_entry_section(selected_table: str):
    """Render B. Manual Entry section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">B. Manual Entry</div>', unsafe_allow_html=True)
    
    has_data = (st.session_state.data_df is not None and not st.session_state.data_df.empty) or selected_table
    if not has_data:
        st.warning("Select a database from Data Source section first.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    entry_mode = st.radio(
        "Manual Entry Options",
        ["Add New Project", "Edit Existing", "Delete Project"],
        key="manual_entry_mode"
    )
    
    if entry_mode == "Add New Project":
        st.markdown("**Add New Project**")
        st.markdown("_Fill in the project details below:_")

        new_pid = st.text_input("🔹 **Project ID** (Required)", key="add_pid", max_chars=50,
                               placeholder="e.g., PROJ-001")
        new_pname = st.text_input("📋 **Project**", key="add_pname", max_chars=200,
                                 placeholder="e.g., Project Name")

        col1, col2 = st.columns(2)
        with col1:
            new_bac = st.number_input("💰 **BAC (Total Budget)** (Required)",
                                    min_value=0.01, value=1000.0, key="add_bac",
                                    help="Budget at Completion - total planned project cost")
        with col2:
            new_ac = st.number_input("💸 **AC (Actual Cost)** (Required)",
                                   min_value=0.0, value=0.0, key="add_ac",
                                   help="Actual Cost to date - amount spent so far")

        col3, col4 = st.columns(2)
        with col3:
            new_start = st.date_input("📅 **Plan Start** (Required)",
                                    value=date.today(), key="add_start")
        with col4:
            new_finish = st.date_input("🏁 **Plan Finish** (Required)",
                                     value=date.today() + timedelta(days=365), key="add_finish")

        st.markdown("**Planned Value (PV) Configuration**")
        use_manual_pv = st.checkbox("📊 Enter Planned Value Manually", key="add_use_manual_pv",
                                   help="Check this to enter PV manually instead of automatic calculation")

        manual_pv = None
        if use_manual_pv:
            manual_pv = st.number_input("💹 **Planned Value (PV)** (Manual)",
                                      min_value=0.0, value=0.0, key="add_manual_pv",
                                      help="Manual Planned Value - will override automatic PV calculation")

        st.markdown("_Optional Information:_")
        new_org = st.text_input("🏢 Organization", key="add_org", max_chars=200,
                               placeholder="e.g., Organization Name")
        new_pm = st.text_input("👤 Project Manager", key="add_pm", max_chars=200,
                              placeholder="e.g., Name of PM")

        if st.button("➕ Add Project", type="primary"):
            if not new_pid.strip():
                st.error("Project ID is required!")
            elif new_finish <= new_start:
                st.error("Plan Finish must be after Plan Start!")
            else:
                try:
                    project_data = {
                        "Project ID": new_pid.strip(),
                        "Project": new_pname.strip(),
                        "Organization": new_org.strip(),
                        "Project Manager": new_pm.strip(),
                        "BAC": new_bac,
                        "AC": new_ac,
                        "Plan Start": new_start.strftime("%Y-%m-%d"),
                        "Plan Finish": new_finish.strftime("%Y-%m-%d"),
                        "Use_Manual_PV": use_manual_pv,
                        "Manual_PV": manual_pv if use_manual_pv else None
                    }
                    insert_project_record(project_data, DEFAULT_DATASET_TABLE)
                    st.success(f"✅ Project '{new_pid}' added successfully!")

                    # Manually clear form fields by resetting session state keys
                    keys_to_clear = ["add_pid", "add_pname", "add_org", "add_pm"]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            st.session_state[key] = ""
                    st.session_state.add_bac = 1000000.0
                    st.session_state.add_ac = 0.0
                    st.session_state.add_start = date.today()
                    st.session_state.add_finish = date.today() + timedelta(days=365)
                    st.session_state.add_use_manual_pv = False
                    st.session_state.add_manual_pv = 0.0

                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to add project: {e}")

    elif entry_mode == "Edit Existing":
        st.markdown("**Edit Existing Project**")
        
        edit_pid = st.text_input("Enter Project ID to edit", value=st.session_state.get("edit_pid_input", ""), key="edit_pid_input", max_chars=50)
        
        if edit_pid and st.button("🔍 Load Project", key="load_project_btn"):
            project = get_project_record(edit_pid, DEFAULT_DATASET_TABLE)
            if project:
                st.session_state.edit_project = project
                st.success(f"Project '{edit_pid}' loaded for editing")
            else:
                st.error(f"Project ID '{edit_pid}' not found")
        
        if "edit_project" in st.session_state:
            project = st.session_state.edit_project
            
            st.markdown("**Current Project Details:**")
            
            edit_pname = st.text_input("Project", value=str(project.get("Project", "")), key="edit_pname")
            edit_org = st.text_input("Organization", value=str(project.get("Organization", "")), key="edit_org")
            edit_pm = st.text_input("Project Manager", value=str(project.get("Project Manager", "")), key="edit_pm")
            
            col1, col2 = st.columns(2)
            with col1:
                edit_bac = st.number_input("BAC (Budget)", min_value=0.01, value=float(project.get("BAC", 1000)), key="edit_bac")
                edit_ac = st.number_input("AC (Actual Cost)", min_value=0.0, value=float(project.get("AC", 0)), key="edit_ac")
            with col2:
                try:
                    edit_start = st.date_input("Plan Start", value=parse_date_any(project.get("Plan Start", date.today())).date(), key="edit_start")
                    edit_finish = st.date_input("Plan Finish", value=parse_date_any(project.get("Plan Finish", date.today())).date(), key="edit_finish")
                except:
                    edit_start = st.date_input("Plan Start", value=date.today(), key="edit_start")
                    edit_finish = st.date_input("Plan Finish", value=date.today() + timedelta(days=365), key="edit_finish")
            
            st.markdown("**Planned Value (PV) Configuration**")
            current_use_manual = bool(project.get("Use_Manual_PV", False))
            current_manual_pv = float(project.get("Manual_PV", 0)) if project.get("Manual_PV") else 0.0
            
            edit_use_manual_pv = st.checkbox("📊 Enter Planned Value Manually", 
                                           value=current_use_manual, key="edit_use_manual_pv",
                                           help="Check this to enter PV manually instead of automatic calculation")
            
            edit_manual_pv = None
            if edit_use_manual_pv:
                edit_manual_pv = st.number_input("💹 **Planned Value (PV)** (Manual)", 
                                                min_value=0.0, value=current_manual_pv, key="edit_manual_pv",
                                                help="Manual Planned Value - will override automatic PV calculation")
            
            if st.button("💾 Update Project", type="primary", key="update_project_btn"):
                if edit_finish <= edit_start:
                    st.error("Plan Finish must be after Plan Start!")
                else:
                    try:
                        updated_data = {
                            "Project": edit_pname.strip(),
                            "Organization": edit_org.strip(),
                            "Project Manager": edit_pm.strip(),
                            "BAC": edit_bac,
                            "AC": edit_ac,
                            "Plan Start": edit_start.strftime("%Y-%m-%d"),
                            "Plan Finish": edit_finish.strftime("%Y-%m-%d"),
                            "Use_Manual_PV": edit_use_manual_pv,
                            "Manual_PV": edit_manual_pv if edit_use_manual_pv else None
                        }
                        update_project_record(edit_pid, updated_data, DEFAULT_DATASET_TABLE)
                        st.success(f"✅ Project '{edit_pid}' updated successfully!")
                        del st.session_state.edit_project
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update project: {e}")
    
    else:  # Delete Project
        st.markdown("**Delete Project**")
        
        delete_pid = st.text_input("Enter Project ID to delete", value=st.session_state.get("delete_pid_input", ""), key="delete_pid_input", max_chars=50)
        
        if delete_pid and st.button("🔍 Show Project Details", key="show_delete_project"):
            project = get_project_record(delete_pid, DEFAULT_DATASET_TABLE)
            if project:
                st.session_state.delete_project = project
                st.info("Project found. Review details below:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.text_input("Project ID", value=str(project.get('Project ID', '')), disabled=True)
                    st.text_input("Project", value=str(project.get('Project', '')), disabled=True)
                with col2:
                    st.text_input("BAC", value=str(project.get('BAC', '')), disabled=True)
                    st.text_input("AC", value=str(project.get('AC', '')), disabled=True)
            else:
                st.error(f"Project ID '{delete_pid}' not found")
        
        if "delete_project" in st.session_state:
            st.warning("⚠️ This action cannot be undone!")
            if st.button("🗑️ DELETE PROJECT", type="secondary", key="delete_project_btn"):
                if st.session_state.get("confirm_delete_project", False):
                    try:
                        delete_project_record(delete_pid, DEFAULT_DATASET_TABLE)
                        st.success(f"✅ Project '{delete_pid}' deleted successfully!")
                        del st.session_state.delete_project
                        st.session_state.confirm_delete_project = False
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete project: {e}")
                else:
                    st.session_state.confirm_delete_project = True
                    st.warning("Click DELETE PROJECT again to confirm")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
def render_controls_section():
    """Render C. Controls section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">C. Controls</div>', unsafe_allow_html=True)
    
    # Curve settings
    curve_type = st.selectbox(
        "Curve Type (PV)",
        ["Linear", "S-Curve"],
        key="curve_type_select"
    )
    
    if curve_type == "S-Curve":
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.number_input("S-Curve α", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="s_alpha")
        with col2:
            beta = st.number_input("S-Curve β", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="s_beta")
    else:
        alpha, beta = 2.0, 2.0
    
    # Date and financial settings
    data_date = st.date_input("Data Date", value=date.today(), key="controls_data_date")
    inflation_rate = st.number_input(
        "Inflation Rate (% APR)", 
        min_value=0.0, max_value=100.0, value=12.0, step=0.1,
        key="controls_inflation"
    )
    
    # Currency settings
    st.markdown("**Currency Settings**")
    col1, col2 = st.columns(2)
    with col1:
        currency_symbol = st.text_input("Currency Symbol", value="PKR", max_chars=10, key="currency_symbol")
    with col2:
        currency_postfix = st.selectbox(
            "Currency Postfix",
            ["", "Thousand", "Million", "Billion"],
            key="currency_postfix"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'curve_type': curve_type.lower(),
        'alpha': alpha,
        'beta': beta,
        'data_date': data_date,
        'inflation_rate': inflation_rate,
        'currency_symbol': currency_symbol,
        'currency_postfix': currency_postfix
    }

def render_llm_provider_section():
    """Render D. LLM Provider section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">D. LLM Provider (Executive Brief)</div>', unsafe_allow_html=True)
    
    # Load saved configuration
    model_config = load_model_config()
    
    provider = st.radio(
        "Choose Provider",
        ["OpenAI", "Gemini"],
        index=0 if model_config.get("provider") == "OpenAI" else 1,
        key="llm_provider"
    )
    
    # Clear API key from session state when provider changes
    if 'previous_provider' in st.session_state and st.session_state['previous_provider'] != provider:
        if 'api_key' in st.session_state:
            del st.session_state['api_key']
        st.info("📋 Provider changed. Please upload a new API key file.")
    st.session_state['previous_provider'] = provider
    
    # API Key file upload
    uploaded_file = st.file_uploader(
        "Upload API Key File",
        type=["txt", "key"],
        help="Upload a text file containing your API key",
        key="api_key_uploader"
    )
    
    api_key = ""
    if uploaded_file is not None:
        # Read the uploaded file content
        api_key = str(uploaded_file.read(), "utf-8").strip()
        # Store in session state temporarily
        st.session_state['api_key'] = api_key
        st.success("✅ API key loaded successfully")
    elif 'api_key' in st.session_state:
        # Use previously uploaded key from session state
        api_key = st.session_state['api_key']
        st.info("ℹ️ Using previously uploaded API key")
    
    if provider == "OpenAI":
        # Model selection for OpenAI
        openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        default_model = model_config.get("model", "gpt-4o-mini") if model_config.get("provider") == "OpenAI" else "gpt-4o-mini"
        if default_model not in openai_models:
            openai_models.append(default_model)
        
        selected_model = st.selectbox(
            "OpenAI Model",
            openai_models,
            index=openai_models.index(default_model) if default_model in openai_models else 0,
            key="openai_model"
        )
        
    else:  # Gemini
        # Model selection for Gemini
        gemini_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        default_model = model_config.get("model", "gemini-1.5-flash") if model_config.get("provider") == "Gemini" else "gemini-1.5-flash"
        if default_model not in gemini_models:
            gemini_models.append(default_model)
        
        selected_model = st.selectbox(
            "Gemini Model",
            gemini_models,
            index=gemini_models.index(default_model) if default_model in gemini_models else 0,
            key="gemini_model"
        )
    
    # Save model configuration
    save_model_config(provider, selected_model)
    
    # LLM settings
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.0, value=0.2, step=0.01,
        key="llm_temperature"
    )
    
    timeout = st.slider(
        "Timeout (seconds)",
        min_value=MIN_TIMEOUT_SECONDS, max_value=MAX_TIMEOUT_SECONDS, value=60, step=5,
        key="llm_timeout"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'provider': provider,
        'model': selected_model,
        'api_key': api_key,
        'temperature': temperature,
        'timeout': timeout
    }

def render_batch_calculation_section():
    """Render E. Batch Calculation section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">E. Batch Calculation</div>', unsafe_allow_html=True)
    
    enable_batch = st.checkbox(
        "Enable Batch Calculation",
        value=False,
        help="Calculate EVM for all projects in the selected dataset",
        key="enable_batch"
    )
    
    if enable_batch:
        st.info("🔄 Batch mode enabled. EVM calculations will be performed for the entire dataset.")
        
        # Batch results management
        if st.button("🔍 Check Previous Results", key="check_batch_results"):
            try:
                # Check if batch results exist
                results_df = load_table(f"{RESULTS_TABLE}_batch")
                st.success(f"Found {len(results_df)} previous batch results")
                if st.button("📥 Download Previous Results"):
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            except:
                st.info("No previous batch results found")
    else:
        st.info("💡 Enable batch calculation to process multiple projects simultaneously")
    
    # Application exit button
    st.markdown("---")
    if st.button("🚪 Exit Application", type="secondary", help="Save settings and close application properly"):
        st.info("👋 Goodbye! Your settings have been saved.")
        st.stop()
    
    st.markdown('</div>', unsafe_allow_html=True)
    return enable_batch

def render_help_section():
    """Render F. Help & Information section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">F. Help & Information</div>', unsafe_allow_html=True)
    
    # Collapsible help sections
    with st.expander("📚 Data Source Options"):
        st.markdown("""
        ### 🗃️ Data Source
        - **Session Data**: Load from current session or uploaded files
        - **Load CSV**: Upload new CSV files and map columns to EVM fields
        
        ### ✏️ Manual Entry
        - Add new projects directly to session
        - Edit existing projects
        - Delete projects
        
        ### 📊 Analysis Options
        - Single project EVM analysis
        - Batch processing for multiple projects
        - AI-powered executive reporting
        """)
    
    with st.expander("🎯 Quick Start Guide"):
        st.markdown("""
        **For New Users:**
        1. Session data is automatically initialized for you
        2. Use "Manual Entry" → "Add New Project" to start
        3. Or upload JSON/CSV data using the sidebar
        
        **Key Fields Required:**
        - Project ID (unique identifier)
        - BAC (Budget at Completion)
        - AC (Actual Cost to date)
        - Plan Start & Finish dates
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_save_download_section():
    """Render G. Save & Download section."""
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">G. Save & Download</div>', unsafe_allow_html=True)
    
    # Check if there's data to save
    has_data = False
    data_sources = []
    
    # Check session state data
    if st.session_state.data_df is not None and not st.session_state.data_df.empty:
        has_data = True
        data_sources.append(f"Session Data ({len(st.session_state.data_df)} rows)")
    
    # Check batch results
    if st.session_state.batch_results is not None and not st.session_state.batch_results.empty:
        has_data = True
        data_sources.append(f"Batch Results ({len(st.session_state.batch_results)} projects)")
    
    # Check config data
    config_items = len(st.session_state.config_dict) if st.session_state.config_dict else 0
    if config_items > 0:
        data_sources.append(f"Configuration ({config_items} settings)")
    
    if has_data or config_items > 0:
        # Show what will be saved
        st.markdown("**[Package] Available for Export:**")
        for source in data_sources:
            st.markdown(f"• {source}")
        
        # Custom filename input for JSON export
        if hasattr(st.session_state, 'original_filename') and st.session_state.original_filename:
            original_name = st.session_state.original_filename
            base_name = original_name.rsplit('.', 1)[0] if '.' in original_name else original_name
            default_filename = f"{base_name}_updated"
        else:
            default_filename = f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        custom_filename = st.text_input(
            "[File] JSON Filename (without .json extension):",
            value=default_filename,
            help="Enter a custom filename for your JSON export",
            key="json_filename_input"
        )
        
        # Main save button
        if st.button("Save JSON", 
                    type="primary", 
                    help="Save data and configuration as JSON file",
                    use_container_width=True):
            if not custom_filename.strip():
                st.error("X Please enter a filename")
            else:
                try:
                    with st.spinner("Creating JSON file..."):
                        # Determine what data to include
                        if st.session_state.data_df is not None and not st.session_state.data_df.empty:
                            export_df = st.session_state.data_df
                        elif st.session_state.batch_results is not None:
                            export_df = st.session_state.batch_results
                        else:
                            # Create empty DataFrame with schema
                            export_df = pd.DataFrame({
                                "Project ID": [],
                                "Project": [],
                                "Organization": [],
                                "Project Manager": [],
                                "BAC": [],
                                "AC": [],
                                "Plan Start": [],
                                "Plan Finish": [],
                                "Use_Manual_PV": [],
                                "Manual_PV": []
                            })
                        
                        # Convert DataFrame to list of dictionaries
                        data_records = export_df.to_dict('records') if not export_df.empty else []
                        
                        # Create comprehensive config for export
                        export_config = st.session_state.config_dict.copy()
                        export_config.update({
                            "export_metadata": {
                                "export_date": datetime.now().isoformat(),
                                "app_version": "Project Portfolio Intelligence Suite",
                                "data_rows": len(export_df),
                                "data_columns": len(export_df.columns) if not export_df.empty else 0,
                                "has_batch_results": st.session_state.batch_results is not None,
                                "config_items": config_items
                            },
                            "data": data_records
                        })
                        
                        # Create JSON content
                        json_content = json.dumps(export_config, indent=2, default=str)
                        
                        # Clean filename and add .json extension
                        clean_filename = custom_filename.strip().replace('.json', '') + '.json'
                        
                        # Provide download button
                        st.download_button(
                            label="Download JSON File",
                            data=json_content,
                            file_name=clean_filename,
                            mime="application/json",
                            use_container_width=True,
                            help=f"Save as: {clean_filename}"
                        )
                        
                        st.success(f"✓ JSON file ready for download: {clean_filename}")
                        st.info("[Info] You can choose where to save this file using your browser's download settings")
                    
                except Exception as e:
                    st.error(f"X Export failed: {e}")
                    logger.error(f"Export error: {e}")
        
        # Exit button
        if st.button("Exit", 
                    type="secondary", 
                    help="Clear session and restart application",
                    use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            st.success("[Wave] Session cleared! Restarting application...")
            st.rerun()
        
    else:
        st.info("[Info] **No data to export**\n\nAdd projects via Manual Entry or upload JSON/CSV files to enable downloads.")
        
        # Quick action buttons when no data
        st.markdown("**[Action] Quick Actions:**")
        if st.button("[File] Load Demo Data", 
                    use_container_width=True,
                    help="Load sample data for testing"):
            try:
                # Create demo data
                demo_data = {
                    "Project ID": ["DEMO001", "DEMO002", "DEMO003"],
                    "Project": ["Website Redesign", "Mobile App", "Data Migration"],
                    "Organization": ["IT", "Product", "Operations"],
                    "Project Manager": ["Alice Smith", "Bob Jones", "Carol Davis"],
                    "BAC": [150000, 250000, 100000],
                    "AC": [120000, 180000, 85000],
                    "Plan Start": ["2024-01-01", "2024-02-15", "2024-03-01"],
                    "Plan Finish": ["2024-06-30", "2024-08-15", "2024-05-30"],
                    "Use_Manual_PV": [False, False, True],
                    "Manual_PV": [None, None, 75000]
                }
                st.session_state.data_df = pd.DataFrame(demo_data)
                st.session_state.data_loaded = True
                st.session_state.original_filename = "demo_data"
                st.session_state.file_type = "demo"
                st.success("✓ Demo data loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load demo data: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
# =============================================================================
# ENHANCED RESULTS DISPLAY
# =============================================================================

def render_enhanced_inputs_tab(project_data: Dict, results: Dict, controls: Dict):
    """Render enhanced inputs tab with better organization."""
    st.markdown("### 📋 Project Information")
    
    # Project Details - Organized in multiple rows
    st.markdown("#### Project Details")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Project ID", value=project_data.get('project_id', ''), disabled=True)
        st.text_input("Project", value=project_data.get('project_name', ''), disabled=True)
    with col2:
        st.text_input("Organization", value=project_data.get('organization', ''), disabled=True)
        st.text_input("Project Manager", value=project_data.get('project_manager', ''), disabled=True)
    
    # Financial Summary - Optimized for large numbers
    st.markdown("#### Financial Summary")
    
    # Use smaller font and better layout for large numbers
    st.markdown("""
    <style>
    .financial-metric {
        font-size: 0.85rem;
        line-height: 1.2;
    }
    .financial-value {
        font-size: 0.9rem;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Row 1: Budget and Actual Cost
    col1, col2 = st.columns(2)
    with col1:
        bac_formatted = format_currency(results['bac'], controls['currency_symbol'], controls['currency_postfix'])
        st.markdown(f'<div class="financial-metric">💰 **Budget (BAC)**<br><span class="financial-value">{bac_formatted}</span></div>', unsafe_allow_html=True)
    with col2:
        ac_formatted = format_currency(results['ac'], controls['currency_symbol'], controls['currency_postfix'])
        st.markdown(f'<div class="financial-metric">💸 **Actual Cost (AC)**<br><span class="financial-value">{ac_formatted}</span></div>', unsafe_allow_html=True)
    
    # Row 2: Planned and Earned Value
    col3, col4 = st.columns(2)
    with col3:
        pv_formatted = format_currency(results['planned_value'], controls['currency_symbol'], controls['currency_postfix'])
        st.markdown(f'<div class="financial-metric">📊 **Planned Value (PV)**<br><span class="financial-value">{pv_formatted}</span></div>', unsafe_allow_html=True)
    with col4:
        ev_formatted = format_currency(results['earned_value'], controls['currency_symbol'], controls['currency_postfix'])
        st.markdown(f'<div class="financial-metric">💎 **Earned Value (EV)**<br><span class="financial-value">{ev_formatted}</span></div>', unsafe_allow_html=True)
    
    # Performance Indicators - Multi-row layout
    st.markdown("#### Performance Indicators")
    
    # Row 1: CPI and SPI
    col1, col2 = st.columns(2)
    with col1:
        cpi = results['cost_performance_index']
        delta_cpi = f"{cpi - 1:.2f}" if is_valid_finite_number(cpi) else "N/A"
        status_cpi = "normal" if cpi >= 1.0 else "inverse"
        st.metric("Cost Performance Index (CPI)", format_performance_index(cpi), 
                 delta=delta_cpi, delta_color=status_cpi)
    
    with col2:
        spi = results['schedule_performance_index']
        delta_spi = f"{spi - 1:.2f}" if is_valid_finite_number(spi) else "N/A"
        status_spi = "normal" if spi >= 1.0 else "inverse"
        st.metric("Schedule Performance Index (SPI)", format_performance_index(spi),
                 delta=delta_spi, delta_color=status_spi)
    
    # Row 2: Progress metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        budget_used = safe_divide(results['ac'], results['bac']) * 100
        st.metric("% Budget Used", f"{budget_used:.1f}%")
    
    with col2:
        st.metric("% Physical Progress", format_percentage(results['percent_complete']))
    
    with col3:
        time_used = safe_divide(results['actual_duration_months'], results['original_duration_months']) * 100
        st.metric("% Time Used", f"{time_used:.1f}%")
    
    # Project Timeline
    st.markdown("#### Project Timeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input("Plan Start", value=results['plan_start'], disabled=True)
    with col2:
        st.text_input("Plan Finish", value=results['plan_finish'], disabled=True)
    with col3:
        st.text_input("Data Date", value=results['data_date'], disabled=True)

def build_enhanced_results_table(results: dict, controls: dict, project_data: dict) -> pd.DataFrame:
    """Build enhanced results table with better formatting."""
    currency = controls['currency_symbol']
    postfix = controls['currency_postfix']
    
    # Helper function for currency formatting
    def fmt_curr(amount):
        if amount is None or not is_valid_finite_number(amount):
            return "—"
        return format_currency(amount, currency, postfix)
    
    # Calculate derived metrics
    bac = results.get('bac', 0)
    ac = results.get('ac', 0)
    pv = results.get('planned_value', 0)
    ev = results.get('earned_value', 0)
    cpi = results.get('cost_performance_index', 0)
    spi = results.get('schedule_performance_index', 0)
    spie = results.get('spie', 0)
    cv = results.get('cost_variance', 0)
    sv = results.get('schedule_variance', 0)
    
    # Status indicators
    def get_status_color(value, good_threshold=1.0):
        if not is_valid_finite_number(value):
            return "⚪"
        return "🟢" if value >= good_threshold else "🟡" if value >= 0.9 else "🔴"
    
    # Build comprehensive results table
    rows = [
        # Project Information
        ("📋 PROJECT INFORMATION", "", "", ""),
        ("Project ID", "", project_data.get('project_id', ''), ""),
        ("Project", "", project_data.get('project_name', ''), ""),
        ("Organization", "", project_data.get('organization', ''), ""),
        ("Project Manager", "", project_data.get('project_manager', ''), ""),
        ("", "", "", ""),  # Spacer
        
        # Financial Overview
        ("💰 FINANCIAL OVERVIEW", "", "", ""),
        ("Budget at Completion (BAC)", "Total planned budget", fmt_curr(bac), ""),
        ("Actual Cost (AC)", "Total spent to date", fmt_curr(ac), ""),
        ("Planned Value (PV)", "Value of work planned", fmt_curr(pv), ""),
        ("Earned Value (EV)", "Value of work completed", fmt_curr(ev), ""),
        ("", "", "", ""),  # Spacer
        
        # Performance Metrics
        ("📊 PERFORMANCE METRICS", "", "", ""),
        ("Cost Performance Index (CPI)", "EV ÷ AC", f"{get_status_color(cpi, 1.0)} {format_performance_index(cpi)}" if is_valid_finite_number(cpi) else "N/A", 
         "Excellent" if cpi > 1.1 else "Good" if cpi > 1.0 else "Poor" if cpi < 0.9 else "Fair"),
        ("Schedule Performance Index (SPI)", "EV ÷ PV", f"{get_status_color(spi, 1.0)} {format_performance_index(spi)}" if is_valid_finite_number(spi) else "N/A",
         "Excellent" if spi > 1.1 else "Good" if spi > 1.0 else "Poor" if spi < 0.9 else "Fair"),
        ("Schedule Performance Index (SPIe)", "ES ÷ AT", f"{get_status_color(spie, 1.0)} {format_performance_index(spie)}" if is_valid_finite_number(spie) else "N/A",
         "Excellent" if spie > 1.1 else "Good" if spie > 1.0 else "Poor" if spie < 0.9 else "Fair"),
        ("", "", "", ""),  # Spacer
        
        # Variances
        ("📈 VARIANCES", "", "", ""),
        ("Cost Variance (CV)", "EV - AC", fmt_curr(cv), "Under budget" if cv > 0 else "Over budget" if cv < 0 else "On budget"),
        ("Schedule Variance (SV)", "EV - PV", fmt_curr(sv), "Ahead of schedule" if sv > 0 else "Behind schedule" if sv < 0 else "On schedule"),
        ("", "", "", ""),  # Spacer
        
        # Forecasts
        ("🔮 FORECASTS", "", "", ""),
        ("Estimate at Completion (EAC)", "BAC ÷ CPI", fmt_curr(results.get('estimate_at_completion')) if is_valid_finite_number(results.get('estimate_at_completion')) else "Cannot determine", ""),
        ("Variance at Completion (VAC)", "BAC - EAC", fmt_curr(results.get('variance_at_completion')) if is_valid_finite_number(results.get('variance_at_completion')) else "Cannot determine", ""),
        ("Estimate to Complete (ETC)", "EAC - AC", fmt_curr(results.get('estimate_to_complete')) if is_valid_finite_number(results.get('estimate_to_complete')) else "Cannot determine", ""),
        ("", "", "", ""),  # Spacer
        
        # Progress Analysis
        ("⏱️ PROGRESS ANALYSIS", "", "", ""),
        ("Physical Progress", "EV ÷ BAC × 100", format_percentage(results['percent_complete']), ""),
        ("Budget Utilization", "AC ÷ BAC × 100", format_percentage(safe_divide(ac, bac) * 100), ""),
        ("Time Utilization", "AT ÷ OD × 100", format_percentage(safe_divide(results['actual_duration_months'], results['original_duration_months']) * 100), ""),
        ("", "", "", ""),  # Spacer
        
        # Schedule Analysis
        ("📅 SCHEDULE ANALYSIS", "", "", ""),
        ("Original Duration", "Planned months", f"{results['original_duration_months']:.1f} months", ""),
        ("Elapsed Duration", "Months to date", f"{results['actual_duration_months']:.1f} months", ""),
        ("Earned Schedule", "Time where PV = EV", f"{results['earned_schedule']:.1f} months", ""),
        ("Expected Duration", "OD ÷ SPIe", f"{results.get('forecast_duration', 0)} months" if results.get('forecast_duration') else "Cannot determine", ""),
        ("Expected Finish Date", "Based on current performance", results.get('forecast_completion', 'N/A'), ""),
    ]
    
    # Add currency note if postfix is used
    if postfix:
        note = f"Note: User figures entered in {postfix.lower()}s (as indicated by {postfix.upper()[0]} postfix)"
        rows.insert(-1, ("", "", note, ""))
    
    return pd.DataFrame(rows, columns=["Category", "Description", "Value", "Status"])

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application with enhanced UX."""
    st.title("📊 Project Portfolio Intelligence Suite")
    st.markdown(
        """
        <div style='color:#003366; font-size:16px; line-height:1.4;'>
            Smarter Projects and Portfolios with Earned Value Analysis 
            and AI-Powered Executive Reporting
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "Developed by Dr. Khalid Ahmad Khan – "
        "[LinkedIn](https://www.linkedin.com/in/khalidahmadkhan/)"
    )
    
    
    try:
        # Initialize session state for first run
        if "session_initialized" not in st.session_state:
            create_demo_data()
            st.session_state.batch_results = None
            st.session_state.data_df = None
            st.session_state.config_dict = {}
            st.session_state.data_loaded = False
            st.session_state.original_filename = None
            st.session_state.file_type = None
            st.session_state.processed_file_info = None
            st.session_state.session_initialized = True

        # Sidebar configuration
        with st.sidebar:
            st.markdown("## ⚙️ Configuration")
            
            # Render all sidebar sections
            df, selected_table, column_mapping = render_data_source_section()
            render_manual_entry_section(selected_table)
            controls = render_controls_section()
            llm_config = render_llm_provider_section()
            enable_batch = render_batch_calculation_section()
            render_help_section()
            render_save_download_section()
        
        
        # Main content area - prioritize session state data
        session_has_data = (st.session_state.data_df is not None and 
                           not st.session_state.data_df.empty)
        
        if not session_has_data:
            st.info("🚀 **Ready to start!** Session initialized automatically. Use **'Manual Entry'** in the sidebar to add your first project, or **'Load JSON/CSV'** to import data.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                ### 🎯 Quick Start
                1. **Add Project**: Use Manual Entry → Add New Project
                2. **Load Data**: Use JSON or CSV Import in sidebar
                3. **Analyze**: Select project for EVM analysis
                """)
            
            with col2:
                st.markdown("""
                ### ✏️ Manual Entry
                - Add new projects directly
                - Edit existing projects
                - Delete projects
                - No files required!
                """)
            
            with col3:
                st.markdown("""
                ### 🔧 Controls & Help
                - Configure curve types & inflation
                - Customize currency display
                - Check Help section for guidance
                """)
            
            return
        
        # Use session state data for display
        display_df = st.session_state.data_df
        
        st.markdown("### 🎯 Single Project Analysis")
        
        with st.expander("📊 Data Preview", expanded=False):
            st.dataframe(display_df.head(20), use_container_width=True)
            st.caption(f"Showing first 20 of {len(display_df)} total projects.")
        
        
        # Batch calculation mode
        if enable_batch:
            st.markdown("### 🔄 Batch Calculation Mode")
            
            if st.button("🚀 Run Batch EVM Calculation", type="primary"):
                # Validate required column mappings exist
                required_keys = ['pid_col', 'bac_col', 'ac_col', 'st_col', 'fn_col']
                missing_mappings = []
                
                for key in required_keys:
                    if not column_mapping.get(key):
                        missing_mappings.append(key.replace('_col', '').upper())
                    elif column_mapping[key] not in display_df.columns:
                        missing_mappings.append(f"{key.replace('_col', '').upper()} (column '{column_mapping[key]}' not found)")
                
                if missing_mappings:
                    st.error(f"Missing required column mappings: {', '.join(missing_mappings)}")
                else:
                    with st.spinner("Processing batch calculations..."):
                        try:
                            batch_results = perform_batch_calculation(
                                display_df, column_mapping, 
                                controls['curve_type'], controls['alpha'], controls['beta'],
                                controls['data_date'], controls['inflation_rate']
                            )
                            
                            st.session_state.batch_results = batch_results
                            save_table_replace(batch_results, f"{RESULTS_TABLE}_batch")
                            st.success(f"✅ Batch calculation completed! Processed {len(batch_results)} projects.")
                            
                        except Exception as e:
                            st.error(f"Batch calculation failed: {e}")
            
            # Display batch results
            if st.session_state.batch_results is not None:
                st.markdown("### 📋 Batch Results")
                
                batch_df = st.session_state.batch_results
                
                if 'error' in batch_df.columns:
                    error_mask = batch_df['error'].notna()
                    valid_count = len(batch_df[~error_mask])
                else:
                    valid_count = len(batch_df)
                
                portfolio_summary = calculate_portfolio_summary(batch_df)
                if "error" not in portfolio_summary:
                    avg_cpi = portfolio_summary.get('weighted_avg_cpi', 0)
                    avg_spi = portfolio_summary.get('weighted_avg_spi', 0)
                else:
                    avg_cpi, avg_spi = 0, 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Projects", len(batch_df))
                with col2:
                    st.metric("Valid Results", valid_count)
                with col3:
                    cpi_status = "🟢" if avg_cpi >= 1.0 else "🟡" if avg_cpi >= 0.9 else "🔴"
                    st.metric("Avg CPI", f"{cpi_status} {format_performance_index(avg_cpi)}" if avg_cpi else "N/A", 
                             f"{avg_cpi - 1:.2f}" if avg_cpi else None)
                with col4:
                    spi_status = "🟢" if avg_spi >= 1.0 else "🟡" if avg_spi >= 0.9 else "🔴"
                    st.metric("Avg SPI", f"{spi_status} {format_performance_index(avg_spi)}" if avg_spi else "N/A",
                             f"{avg_spi - 1:.2f}" if avg_spi else None)
                
                if valid_count > 0 and "error" not in portfolio_summary:
                    st.markdown("### 📊 Portfolio Performance Analysis")
                    
                    st.markdown("#### 💰 Portfolio Financial Summary")
                    st.markdown("""
                    <style>
                    .portfolio-financial-metric { font-size: 0.85rem; line-height: 1.2; }
                    .portfolio-financial-value { font-size: 0.9rem; font-weight: bold; color: #1f77b4; }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        bac_formatted = format_currency(portfolio_summary['total_bac'], controls['currency_symbol'], controls['currency_postfix'])
                        st.markdown(f'<div class="portfolio-financial-metric">💰 **Total Budget (BAC)**<br><span class="portfolio-financial-value">{bac_formatted}</span></div>', unsafe_allow_html=True)
                    with col2:
                        ac_formatted = format_currency(portfolio_summary['total_ac'], controls['currency_symbol'], controls['currency_postfix'])
                        st.markdown(f'<div class="portfolio-financial-metric">💸 **Total Actual Cost (AC)**<br><span class="portfolio-financial-value">{ac_formatted}</span></div>', unsafe_allow_html=True)
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        pv_formatted = format_currency(portfolio_summary['total_pv'], controls['currency_symbol'], controls['currency_postfix'])
                        st.markdown(f'<div class="portfolio-financial-metric">📊 **Total Planned Value (PV)**<br><span class="portfolio-financial-value">{pv_formatted}</span></div>', unsafe_allow_html=True)
                    with col4:
                        ev_formatted = format_currency(portfolio_summary['total_ev'], controls['currency_symbol'], controls['currency_postfix'])
                        st.markdown(f'<div class="portfolio-financial-metric">💎 **Total Earned Value (EV)**<br><span class="portfolio-financial-value">{ev_formatted}</span></div>', unsafe_allow_html=True)
                    
                    st.markdown("#### 📈 Portfolio Progress Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        budget_used = safe_divide(portfolio_summary['total_ac'], portfolio_summary['total_bac']) * 100
                        st.metric("% Budget Used", format_percentage(budget_used))
                    with col2:
                        earned_value_pct = safe_divide(portfolio_summary['total_ev'], portfolio_summary['total_bac']) * 100
                        st.metric("% Value Earned", format_percentage(earned_value_pct))
                    with col3:
                        avg_progress = portfolio_summary['average_progress']
                        st.metric("Avg Progress", format_percentage(avg_progress))
                    
                    st.markdown("#### 🎯 Performance Quadrants Analysis")
                    quadrant_data = []
                    labels = {
                        'on_budget_on_schedule': '🟢 On Budget & On Schedule',
                        'on_budget_behind_schedule': '🟡 On Budget & Behind Schedule', 
                        'over_budget_on_schedule': '🟡 Over Budget & On Schedule',
                        'over_budget_behind_schedule': '🔴 Over Budget & Behind Schedule'
                    }
                    
                    for key, label in labels.items():
                        count = portfolio_summary['quadrants'][key]
                        count_pct = portfolio_summary['quadrant_percentages'][key]
                        budget = portfolio_summary['quadrant_budgets'][key]
                        budget_pct = portfolio_summary['budget_percentages'][key]
                        quadrant_data.append({
                            'Performance Category': label,
                            'Project Count': count,
                            'Count %': format_percentage(count_pct),
                            'Total Budget': format_currency(budget, controls['currency_symbol'], controls['currency_postfix']),
                            'Budget %': format_percentage(budget_pct)
                        })
                    st.dataframe(pd.DataFrame(quadrant_data), use_container_width=True, hide_index=True)
                    
                    st.markdown("#### 📊 Portfolio Performance Charts")
                    try:
                        fig = plt.figure(figsize=(20, 16))
                        plt.style.use('seaborn-v0_8-whitegrid')
                        
                        # Portfolio Performance Matrix
                        ax1 = plt.subplot(2, 2, 1)
                        ax1.set_xlim(0.5, 1.5); ax1.set_ylim(0.5, 1.5)
                        ax1.fill([1.0, 1.5, 1.5, 1.0], [1.0, 1.0, 1.5, 1.5], alpha=0.2, color='green')
                        ax1.fill([0.5, 1.0, 1.0, 0.5], [1.0, 1.0, 1.5, 1.5], alpha=0.2, color='yellow')
                        ax1.fill([0.5, 1.0, 1.0, 0.5], [0.5, 0.5, 1.0, 1.0], alpha=0.2, color='red')
                        ax1.fill([1.0, 1.5, 1.5, 1.0], [0.5, 0.5, 1.0, 1.0], alpha=0.2, color='yellow')
                        ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
                        ax1.axvline(1.0, color="gray", linestyle="--", alpha=0.7)
                        p_cpi = portfolio_summary['portfolio_cpi']; p_spi = portfolio_summary['portfolio_spi']
                        port_color = 'green' if (p_cpi >= 1.0 and p_spi >= 1.0) else 'orange' if (p_cpi >= 0.9 or p_spi >= 0.9) else 'red'
                        ax1.scatter([p_spi], [p_cpi], s=200, c=port_color, alpha=0.8, edgecolors='black', linewidth=2, label='Portfolio')
                        if 'cost_performance_index' in batch_df.columns and 'schedule_performance_index' in batch_df.columns:
                            ax1.scatter(batch_df['schedule_performance_index'].fillna(0), batch_df['cost_performance_index'].fillna(0), s=30, alpha=0.5, c='blue', label='Projects')
                        ax1.set_xlabel('Schedule Performance Index (SPI)'); ax1.set_ylabel('Cost Performance Index (CPI)')
                        ax1.set_title('Portfolio Performance Matrix', fontweight='bold'); ax1.legend(); ax1.grid(True, alpha=0.3)
                        
                        # Portfolio Progress Summary
                        ax2 = plt.subplot(2, 2, 2)
                        categories = ['Budget Used', 'Earned Value', 'Average Progress']
                        values = [
                            safe_divide(portfolio_summary['total_ac'], portfolio_summary['total_bac']) * 100,
                            safe_divide(portfolio_summary['total_ev'], portfolio_summary['total_bac']) * 100,
                            portfolio_summary['average_progress']
                        ]
                        bars = ax2.barh(categories, values, color=['#ff6b6b', '#45b7d1', '#2E8B57'], alpha=0.8)
                        ax2.axvline(100, color='gray', linestyle='--', alpha=0.7)
                        ax2.set_xlim(0, max(120, max(values) + 10))
                        ax2.set_xlabel('Percentage (%)'); ax2.set_title('Portfolio Progress Summary', fontweight='bold')
                        for bar, value in zip(bars, values):
                            ax2.text(value + 2, bar.get_y() + bar.get_height()/2, f'{value:.1f}%', va='center', fontweight='bold')
                        
                        # Portfolio Time/Budget Performance Curve
                        ax3 = plt.subplot(2, 1, 2)
                        T = np.linspace(0, 1, 101)
                        blue_curve = -0.794*T**3 + 0.632*T**2 + 1.162*T
                        red_curve = -0.387*T**3 + 1.442*T**2 - 0.055*T
                        ax3.plot(T, blue_curve, 'b-', linewidth=2, label='Blue Curve', alpha=0.8)
                        ax3.plot(T, red_curve, 'r-', linewidth=2, label='Red Curve', alpha=0.8)
                        if 'actual_duration_months' in batch_df.columns and 'original_duration_months' in batch_df.columns:
                            portfolio_norm_time = (batch_df['actual_duration_months'] / batch_df['original_duration_months']).fillna(0).mean()
                        else:
                            portfolio_norm_time = 0.5
                        ax3.axvline(portfolio_norm_time, color='yellow', linestyle='--', alpha=0.8, linewidth=2, label='Portfolio Data Date')
                        if 'actual_duration_months' in batch_df.columns and 'earned_value' in batch_df.columns and 'bac' in batch_df.columns:
                            project_norm_times = batch_df['actual_duration_months'] / batch_df['original_duration_months']
                            project_norm_evs = batch_df['earned_value'] / batch_df['bac']
                            cpis = batch_df.get('cost_performance_index', pd.Series([1.0] * len(batch_df))).fillna(1.0)
                            colors = ['green' if cpi >= 1.0 else 'orange' if cpi >= 0.9 else 'red' for cpi in cpis]
                            ax3.scatter(project_norm_times, project_norm_evs, c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=1, label='Projects')
                        ax3.set_xlim(0, 1); ax3.set_ylim(0, 1.2); ax3.set_xlabel('Time (Normalized)'); ax3.set_ylabel('PV (Normalized)')
                        ax3.set_title('Portfolio Time/Budget Performance Curve', fontweight='bold'); ax3.legend(loc='upper left'); ax3.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Chart generation failed: {e}")

                    st.markdown("#### 📋 Portfolio Executive Report")
                    if st.button("🚀 Generate Portfolio Executive Report", type="primary"):
                        portfolio_prompt = create_portfolio_executive_summary(portfolio_summary, controls)
                        with st.spinner("Generating comprehensive portfolio executive report..."):
                            executive_report = safe_llm_request(llm_config['provider'], llm_config['model'], llm_config['api_key'],
                                                                llm_config['temperature'], llm_config['timeout'],
                                                                portfolio_prompt)
                            st.session_state.portfolio_executive_report = executive_report
                    
                    if "portfolio_executive_report" in st.session_state:
                        portfolio_report = st.session_state.portfolio_executive_report
                        if portfolio_report.startswith("Error:") or portfolio_report == "No API key available":
                            if portfolio_report == "No API key available":
                                st.warning("⚠️ No API key available. Please upload an API key file in the LLM Provider section.")
                            else:
                                st.error(portfolio_report)
                        else:
                            st.markdown("#### 📄 Portfolio Executive Assessment")
                            st.markdown(portfolio_report)
                            st.download_button("📥 Download Report", portfolio_report, file_name=f"portfolio_executive_report.md", mime="text/markdown")
                
                st.markdown("### 📋 Detailed Results")
                st.dataframe(batch_df, use_container_width=True, height=400)
                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Batch Results", csv_data, file_name="batch_evm_results.csv", mime="text/csv", type="primary")
            return
        
        if not all(column_mapping.get(key) for key in ['pid_col', 'bac_col', 'ac_col', 'st_col', 'fn_col']):
            st.warning("⚠️ Please complete the column mapping in the sidebar first.")
            return
        
        try:
            pid_col = column_mapping['pid_col']
            pname_col = column_mapping.get('pname_col')
            project_ids = display_df[pid_col].astype(str).tolist()
            project_names = display_df[pname_col].astype(str).fillna("").tolist() if pname_col else [""] * len(project_ids)
            display_options = [f"{pid} — {pname}" if pname and pname != "nan" else pid for pid, pname in zip(project_ids, project_names)]
            
            selected_idx = st.selectbox("Select Project for Analysis", range(len(display_options)), format_func=lambda i: display_options[i])
            selected_project_id = project_ids[selected_idx]
            
        except Exception as e:
            st.error(f"Error processing project list: {e}")
            return
        
        project_row = display_df[display_df[pid_col].astype(str) == selected_project_id].iloc[0]
        
        try:
            project_data = {
                'project_id': selected_project_id,
                'project_name': str(project_row.get(column_mapping.get('pname_col'), "")),
                'organization': str(project_row.get(column_mapping.get('org_col'), "")),
                'project_manager': str(project_row.get(column_mapping.get('pm_col'), "")),
                'bac': float(project_row[column_mapping['bac_col']]),
                'ac': float(project_row[column_mapping['ac_col']]),
                'plan_start': project_row[column_mapping['st_col']],
                'plan_finish': project_row[column_mapping['fn_col']]
            }
        except Exception as e:
            st.error(f"Error extracting project data: {e}")
            return
        
        try:
            with st.spinner("Calculating EVM metrics..."):
                use_manual_pv = bool(project_row.get('Use_Manual_PV', False))
                manual_pv_val = project_row.get('Manual_PV')
                results = perform_complete_evm_analysis(
                    bac=project_data['bac'], ac=project_data['ac'],
                    plan_start=project_data['plan_start'], plan_finish=project_data['plan_finish'],
                    data_date=controls['data_date'], annual_inflation_rate=controls['inflation_rate'] / 100.0,
                    curve_type=controls['curve_type'], alpha=controls['alpha'], beta=controls['beta'],
                    manual_pv=manual_pv_val, use_manual_pv=use_manual_pv
                )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                cpi = results['cost_performance_index']
                cpi_status = "🟢" if cpi >= 1.0 else "🟡" if cpi >= 0.9 else "🔴"
                st.metric("Cost Performance", f"{cpi_status} {format_performance_index(cpi)}", f"{cpi - 1:.2f}")
            with col2:
                spi = results['schedule_performance_index']
                spi_status = "🟢" if spi >= 1.0 else "🟡" if spi >= 0.9 else "🔴"
                st.metric("Schedule Performance", f"{spi_status} {format_performance_index(spi)}", f"{spi - 1:.2f}")
            with col3:
                st.metric("Progress", format_percentage(results['percent_complete']))
            
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Inputs", "📊 Results", "🤖 Executive Brief", "📈 Charts"])
            
            with tab1:
                render_enhanced_inputs_tab(project_data, results, controls)
            
            with tab2:
                st.markdown("### 📊 Detailed EVM Analysis")
                results_table = build_enhanced_results_table(results, controls, project_data)
                st.dataframe(results_table, use_container_width=True, height=600)
                
                csv_buffer = io.StringIO()
                results_table.to_csv(csv_buffer, index=False)
                st.download_button("📥 Download Results (CSV)", csv_buffer.getvalue(), file_name=f"evm_analysis_{selected_project_id}.csv")
            
            with tab3:
                st.markdown("### 🤖 Executive Brief")
                
                project_name = project_data['project_name'] or selected_project_id
                combined_text = f"{project_name} {project_data['organization']}".lower()
                
                project_context = ""
                if any(w in combined_text for w in ['bridge', 'construction', 'road']): project_context = "This is likely a construction project."
                elif any(w in combined_text for w in ['software', 'it', 'system', 'app']): project_context = "This is likely a technology project."
                
                prompt = textwrap.dedent(f"""
                Create an executive-level project status report for the following Earned Value Management analysis.
                Structure for C-suite and senior stakeholders who need quick decision-making information.

                PROJECT: {project_name}
                ORGANIZATION: {project_data.get('organization', '')}
                PROJECT MANAGER: {project_data.get('project_manager', '')}
                ANALYSIS DATE: {results['data_date']}
                CURRENCY: {controls['currency_symbol']} {controls.get('currency_postfix', '')}

                === PROJECT DATA ===
                FINANCIALS:
                - Budget (BAC): {format_currency(results['bac'], controls['currency_symbol'], controls['currency_postfix'])}
                - Actual Cost (AC): {format_currency(results['ac'], controls['currency_symbol'], controls['currency_postfix'])}
                - Earned Value (EV): {format_currency(results['earned_value'], controls['currency_symbol'], controls['currency_postfix'])}
                - Planned Value (PV): {format_currency(results['planned_value'], controls['currency_symbol'], controls['currency_postfix'])}

                SCHEDULE:
                - Plan Start: {results['plan_start']}
                - Plan Finish: {results['plan_finish']}
                - Expected Finish: {results['forecast_completion']}
                - Physical Completion: {format_percentage(results['percent_complete'])}

                PERFORMANCE INDICES:
                - Cost Performance (CPI): {format_performance_index(results['cost_performance_index'])}
                - Schedule Performance (SPI): {format_performance_index(results['schedule_performance_index'])}
                - Budget Utilization: {format_percentage(safe_divide(results['ac'], results['bac']) * 100)}
                - Time Utilization: {format_percentage(safe_divide(results['actual_duration_months'], results['original_duration_months']) * 100)}

                FORECASTS:
                - Estimate at Completion (EAC): {format_currency(results['estimate_at_completion'], controls['currency_symbol'], controls['currency_postfix']) if is_valid_finite_number(results['estimate_at_completion']) else 'Cannot determine'}
                - Variance at Completion (VAC): {format_currency(results['variance_at_completion'], controls['currency_symbol'], controls['currency_postfix']) if is_valid_finite_number(results['variance_at_completion']) else 'Cannot determine'}

                STRUCTURE YOUR RESPONSE AS FOLLOWS:

                # PROJECT STATUS REPORT

                **Project:** {project_name}  
                **Organization:** {project_data.get('organization', 'N/A')}  
                **Project Manager:** {project_data.get('project_manager', 'N/A')}  
                **Report Date:** {results['data_date']}  
                **Currency:** {controls['currency_symbol']} {controls.get('currency_postfix', '')}

                ---

                ### PROJECT DASHBOARD
                Create a formatted table with these key metrics:
                | Metric | Value | Status |
                |--------|--------|---------|
                | Data Date | [date] | |
                | Original Budget | [amount] | |
                | Budget Used % | [percentage] | [Good/Warning/Critical] |
                | Time Elapsed % | [percentage] | [Good/Warning/Critical] |
                | Physical Progress % | [percentage] | |
                | Cost Performance (CPI) | [value] | [Good/Warning/Critical] |
                | Schedule Performance (SPI) | [value] | [Good/Warning/Critical] |
                | Target Finish | [original date] | |
                | Expected Finish | [forecast date] | [On Time/Delayed] |
                | Budget Variance | [amount over/under] | [Good/Warning/Critical] |

                ### EXECUTIVE SUMMARY
                - Provide 3-4 bullet points with the most critical information
                - Include overall project health rating (GREEN/YELLOW/RED)
                - Highlight immediate concerns requiring executive attention
                - Quantify financial and schedule impacts

                ### PERFORMANCE ANALYSIS
                **Cost Performance:**
                - Analysis of spending efficiency and budget variance
                - Root cause assessment if CPI < 0.95

                **Schedule Performance:** 
                - Timeline analysis and delay impact
                - Critical path concerns if SPI < 0.90

                ### RISK ASSESSMENT & IMPACT
                - Rate risk level as LOW/MEDIUM/HIGH for each category:
                  - Financial risk (cost overrun probability)
                  - Schedule risk (delivery delay probability) 
                  - Stakeholder risk (reputation/contract penalties)
                - Quantify potential impacts where possible

                ### CORRECTIVE ACTIONS (Priority Ranked)
                Provide specific, measurable actions with:
                1. **Immediate Actions (0-30 days)** - What must be done now
                2. **Short-term Actions (30-90 days)** - Recovery measures
                3. **Long-term Monitoring** - Ongoing controls

                For each action, specify:
                - Specific deliverable/outcome
                - Success metric
                - Responsibility/ownership
                - Timeline

                ### DECISION POINTS
                List any decisions requiring executive/sponsor approval:
                - Budget increase authorizations
                - Scope change considerations  
                - Resource reallocation needs
                - Contract modification requirements

                Focus on being actionable, specific, and quantified. Avoid generic project management advice.
                Use professional tone appropriate for executive leadership.
                """)
                
                if st.button("🚀 Generate Executive Brief", type="primary"):
                    with st.spinner("Generating executive brief..."):
                        brief_response = safe_llm_request(
                            llm_config['provider'], llm_config['model'], llm_config['api_key'],
                            llm_config['temperature'], llm_config['timeout'], prompt
                        )
                        st.session_state.executive_brief = brief_response
                
                if "executive_brief" in st.session_state:
                    brief = st.session_state.executive_brief
                    if brief.startswith("Error:") or brief == "No API key available":
                        if brief == "No API key available":
                            st.warning("⚠️ No API key available. Please upload an API key file in the LLM Provider section.")
                        else:
                            st.error(brief)
                    else:
                        st.markdown("#### 📄 Executive Summary Report")
                        st.markdown(brief)
                        st.download_button("📥 Download Brief", brief, file_name=f"executive_brief_{selected_project_id}.md", mime="text/markdown")
            
            with tab4:
                st.markdown("### 📈 Performance Visualization")
                try:
                    fig = plt.figure(figsize=(20, 16))
                    plt.style.use('seaborn-v0_8-whitegrid')
                    
                    # Performance Matrix
                    ax1 = plt.subplot(2, 2, 1)
                    cpi, spi = results['cost_performance_index'], results['schedule_performance_index']
                    ax1.fill([1.0, 1.5, 1.5, 1.0], [1.0, 1.0, 1.5, 1.5], alpha=0.2, color='green')
                    ax1.fill([0.5, 1.0, 1.0, 0.5], [1.0, 1.0, 1.5, 1.5], alpha=0.2, color='yellow')
                    ax1.fill([0.5, 1.0, 1.0, 0.5], [0.5, 0.5, 1.0, 1.0], alpha=0.2, color='red')
                    ax1.fill([1.0, 1.5, 1.5, 1.0], [0.5, 0.5, 1.0, 1.0], alpha=0.2, color='yellow')
                    color = 'green' if (cpi >= 1.0 and spi >= 1.0) else 'orange' if (cpi >= 0.9 or spi >= 0.9) else 'red'
                    ax1.scatter([spi], [cpi], s=150, c=color, alpha=0.8, edgecolors='black', linewidth=2)
                    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
                    ax1.axvline(1.0, color='gray', linestyle='--', alpha=0.7)
                    ax1.set_xlim(0.5, 1.5); ax1.set_ylim(0.5, 1.5)
                    ax1.set_xlabel('Schedule Performance Index (SPI)'); ax1.set_ylabel('Cost Performance Index (CPI)')
                    ax1.set_title('Performance Matrix', fontweight='bold')
                    
                    # Progress Comparison
                    ax2 = plt.subplot(2, 2, 2)
                    metrics = ['Budget Used', 'Time Used', 'Work Complete']
                    values = [
                        safe_divide(results['ac'], results['bac']) * 100,
                        safe_divide(results['actual_duration_months'], results['original_duration_months']) * 100,
                        results['percent_complete']
                    ]
                    bars = ax2.barh(metrics, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'], alpha=0.8)
                    ax2.axvline(100, color='gray', linestyle='--', alpha=0.7)
                    ax2.set_xlim(0, max(120, max(values) + 10)); ax2.set_xlabel('Percentage (%)')
                    ax2.set_title('Progress Comparison', fontweight='bold')
                    for bar, value in zip(bars, values):
                        ax2.text(value + 2, bar.get_y() + bar.get_height()/2, f'{value:.1f}%', va='center', fontweight='bold')

                    # EVM Curves
                    ax3 = plt.subplot(2, 1, 2)
                    total_months = results['original_duration_months']
                    actual_months = results['actual_duration_months']
                    timeline = np.linspace(0, total_months, 100)
                    actual_timeline = np.linspace(0, actual_months, 100) if actual_months > 0 else [0]
                    
                    if controls['curve_type'] == 's-curve':
                        pv_values = [calculate_pv_scurve(results['bac'], t, total_months, controls['alpha'], controls['beta']) for t in timeline]
                    else:
                        pv_values = [calculate_pv_linear(results['bac'], t, total_months) for t in timeline]
                    
                    ax3.plot(timeline, pv_values, 'b-', linewidth=2, label='Planned Value (PV)')
                    if actual_months > 0:
                        ax3.plot(actual_timeline, np.linspace(0, results['ac'], 100), 'r-', linewidth=2, label='Actual Cost (AC)')
                        ax3.plot(actual_timeline, np.linspace(0, results['earned_value'], 100), 'g-', linewidth=2, label='Earned Value (EV)')
                    
                    ax3.axvline(actual_months, color='orange', linestyle='--', linewidth=2, label='Data Date')
                    ax3.set_xlabel('Time (Months)'); ax3.set_ylabel(f'Value ({controls["currency_symbol"]})')
                    ax3.set_title('EVM Performance Curves', fontweight='bold', fontsize=14)
                    ax3.legend(); ax3.grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart generation failed: {e}")
        
        except Exception as e:
            st.error(f"EVM calculation failed: {e}")
            logger.error(f"EVM calculation error: {e}")
    
    except Exception as e:
        st.error(f"An unexpected application error occurred: {e}")
        logger.error(f"Application error: {e}", exc_info=True)


# =============================================================================
# FILE HANDLING FUNCTIONS
# =============================================================================
def load_json_file(uploaded_file) -> tuple[pd.DataFrame, Dict[str, Any], str]:
    """Load JSON file and return DataFrame, config, and filename."""
    try:
        json_content = uploaded_file.getvalue()
        filename = uploaded_file.name
        config_data = json.loads(json_content.decode('utf-8'))
        
        project_data = []
        if 'data' in config_data and isinstance(config_data['data'], list):
            project_data = config_data['data']
        elif isinstance(config_data, list):
            project_data = config_data
        
        df = pd.DataFrame(project_data) if project_data else pd.DataFrame()
        logger.info(f"Loaded JSON: {len(df)} rows.")
        return df, config_data, filename
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        raise


def load_csv_file(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Load CSV file."""
    try:
        filename = uploaded_file.name
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                if not df.empty:
                    logger.info(f"Loaded CSV with {encoding}: {len(df)} rows.")
                    return df, filename
            except Exception:
                continue
        
        raise ValueError("Could not load CSV file with any supported encoding.")
        
    except Exception as e:
        logger.error(f"Failed to load CSV file: {e}")
        raise

if __name__ == "__main__":
    main()