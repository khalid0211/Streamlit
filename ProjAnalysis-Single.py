import streamlit as st
from datetime import datetime
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

# Function to calculate the values
def calculate_values(BC, SD, CD, DD, AC, PPE):
    # Convert dates to datetime objects
    SD_dt = datetime.strptime(SD, "%d/%m/%Y")
    CD_dt = datetime.strptime(CD, "%d/%m/%Y")
    DD_dt = datetime.strptime(DD, "%d/%m/%Y")
    
    # Check if dates are valid
    if SD_dt >= CD_dt:
        st.error("Start Date must be before Finish Date.")
        return None, None, None, None, None, None, None
    if DD_dt < SD_dt or DD_dt > CD_dt:
        st.error("Data Date must be between Start Date and Finish Date.")
        return None, None, None, None, None, None, None
    
    # Calculate PD: Project Duration (in days)
    PD = (CD_dt - SD_dt).days
    
    # Calculate T: % Time used
    T = (DD_dt - SD_dt).days / PD
    
    # Calculate PV: Planned Disbursement (Back Loaded)
    PV = 0.65 * T**2 + 0.35 * T
    
    # Calculate PV(l): Planned Disbursement (Range-Low)
    PV_l = -0.387 * T**3 + 1.442 * T**2 - 0.055 * T
    
    # Calculate PV(h): Planned Disbursement (Range-High)
    PV_h = -0.794 * T**3 + 0.632 * T**2 + 1.162 * T
    
    # For TL and TE, we need to solve for T in the equations
    # EV is the PPE (Physical Progress Estimate %) converted to decimal
    EV = PPE / 100
    
    # Function to find TL: Latest planned time to achieve EV
    def equation_TL(T):
        return -0.387 * T**3 + 1.442 * T**2 - 0.055 * T - EV
    
    # Function to find TE: Earliest planned time to achieve EV
    def equation_TE(T):
        return -0.794 * T**3 + 0.632 * T**2 + 1.162 * T - EV
    
    # Solve for TL and TE
    try:
        # Try multiple initial guesses to find a valid solution
        TL_candidates = []
        for guess in np.linspace(0.01, 0.99, 10):
            TL_candidate = fsolve(equation_TL, guess)[0]
            if 0 <= TL_candidate <= 1 and abs(equation_TL(TL_candidate)) < 1e-6:
                TL_candidates.append(TL_candidate)
        
        if TL_candidates:
            TL = max(TL_candidates)  # Take the maximum as the "latest" time
        else:
            TL = None
    except:
        TL = None
    
    try:
        # Try multiple initial guesses to find a valid solution
        TE_candidates = []
        for guess in np.linspace(0.01, 0.99, 10):
            TE_candidate = fsolve(equation_TE, guess)[0]
            if 0 <= TE_candidate <= 1 and abs(equation_TE(TE_candidate)) < 1e-6:
                TE_candidates.append(TE_candidate)
        
        if TE_candidates:
            TE = min(TE_candidates)  # Take the minimum as the "earliest" time
        else:
            TE = None
    except:
        TE = None
    
    return PD, T, PV, PV_l, PV_h, TL, TE

# Function to create the plot
def create_plot(T, PPE):
    # Create figure and axis with adjusted size to match table height
    fig, ax = plt.subplots(figsize=(5, 7))  # Adjusted height to match table
    
    # Generate T values from 0 to 1
    T_values = np.linspace(0, 1, 100)
    
    # Calculate PV(l) and PV(h) for all T values
    PV_l_values = -0.387 * T_values**3 + 1.442 * T_values**2 - 0.055 * T_values
    PV_h_values = -0.794 * T_values**3 + 0.632 * T_values**2 + 1.162 * T_values
    
    # Plot the curves
    ax.plot(T_values, PV_l_values, 'b-', label='PV(l) - Range-Low')
    ax.plot(T_values, PV_h_values, 'r-', label='PV(h) - Range-High')
    
    # Add vertical line at current T
    ax.axvline(x=T, color='g', linestyle='--', label=f'T = {T:.2f}')
    
    # Add point at (T, PPE/100)
    ax.plot(T, PPE/100, 'ko', markersize=8, label=f'PPE = {PPE}%')
    
    # Set grid with lines every 0.1
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('% Time Used (T)')
    ax.set_ylabel('Planned Disbursement')
    ax.set_title('Project Progress Analysis')
    ax.legend()
    
    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    return fig

# Streamlit app
st.title("Project Performance Analysis")

# Initialize session state for popup
if 'show_popup' not in st.session_state:
    st.session_state.show_popup = False

# Input Section
st.header("Input Section (1-6)")

BC = st.number_input("BC: Budgeted Cost (currency)", min_value=0.0, value=10000.0)
SD = st.date_input("SD: Start Date").strftime("%d/%m/%Y")
CD = st.date_input("CD: Finish Date").strftime("%d/%m/%Y")
DD = st.date_input("DD: Data Date").strftime("%d/%m/%Y")
AC = st.number_input("AC: Actual Cost (currency)", min_value=0.0, value=5000.0)
PPE = st.number_input("PPE: Physical Progress Estimate %", min_value=0.0, max_value=100.0, value=15.0)

# Button to calculate
if st.button("Calculate"):
    # Calculate values
    PD, T, PV, PV_l, PV_h, TL, TE = calculate_values(BC, SD, CD, DD, AC, PPE)
    
    # If any of the values is None, there was an error
    if PD is None:
        st.error("Please correct the errors in the input dates and try again.")
    else:
        # Determine status for T (variable 8)
        if TL is not None and TE is not None:
            if T < TE:
                t_status = "Below"
            elif T > TL:
                t_status = "Above"
            else:
                t_status = "Normal"
        else:
            t_status = "N/A"
        
        # Determine status for PPE (variable 6)
        PPE_decimal = PPE / 100
        if PPE_decimal < PV_l:
            ppe_status = "Below"
        elif PPE_decimal > PV_h:
            ppe_status = "Above"
        else:
            ppe_status = "Normal"
        
        # Create a DataFrame for the calculated values (7-13)
        data = {
            'Description': [
                'Project Duration',
                '% Time used',
                'Planned Disbursement (Back Loaded)',
                'Planned Disbursement (Range-Low)',
                'Planned Disbursement (Range-High)',
                'Latest planned time to achieve EV',
                'Earliest planned time to achieve EV'
            ],
            'Value': [
                f"{PD} days",
                t_status,  # Abbreviated status for T
                f"{PV:.2f}",
                f"{PV_l:.2f}",
                f"{PV_h:.2f}",
                f"{TL:.2f}" if TL is not None else "N/A",
                f"{TE:.2f}" if TE is not None else "N/A"
            ]
        }
        
        # Create analysis data (14-15)
        analysis_data = {
            'Description': [
                'Physical Progress Estimate status',
                '% Time used status'
            ],
            'Value': [
                ppe_status,  # Abbreviated status for PPE
                t_status     # Abbreviated status for T
            ]
        }
        
        # Combine both dataframes
        df_calculated = pd.DataFrame(data)
        df_analysis = pd.DataFrame(analysis_data)
        combined_df = pd.concat([df_calculated, df_analysis], ignore_index=True)
        
        # Store the dataframe in session state for popup
        st.session_state.df = combined_df
        st.session_state.T = T
        st.session_state.PPE = PPE
        st.session_state.show_popup = True

# Create the popup window using Streamlit components
if st.session_state.show_popup and 'df' in st.session_state:
    # Create a full-width container for the popup background
    popup_bg = st.container()
    
    with popup_bg:
        # Create a container for the popup content
        popup_container = st.container()
        
        with popup_container:
            # Add a header for the popup
            st.markdown("## Project Performance Analysis Results")
            
            # Create two columns for the popup content (more space for table)
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Add custom CSS to reduce font size
                st.markdown("""
                <style>
                .dataframe {
                    font-size: 0.8rem !important;
                }
                .dataframe th {
                    font-size: 0.9rem !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Display the combined table (7-15) with only Description and Value columns
                st.dataframe(st.session_state.df[['Description', 'Value']], use_container_width=True)
            
            with col2:
                # Plot Section
                fig = create_plot(st.session_state.T, st.session_state.PPE)
                st.pyplot(fig)
            
            # Add a divider and close button
            st.divider()
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("Close Results", use_container_width=True):
                    st.session_state.show_popup = False
                    st.experimental_rerun()