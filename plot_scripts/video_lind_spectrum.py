import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- CONFIGURATION ---
DATA_FILE = '../rust/eigenvalues_8.parquet' 

# --- 1. LOAD DATA ---
@st.cache_data
def load_and_prep_data():
    """Loads data and calculates global limits ONCE."""
    # Load file
    if DATA_FILE.endswith('.parquet'):
        df = pd.read_parquet(DATA_FILE)
    else:
        df = pd.read_csv(DATA_FILE)

    # --- CALCULATE GLOBAL LIMITS (FIXED AXIS) ---
    # We scan the entire dataset now so we never have to recalculate.
    # Columns 0,1 are Gamma/Omega.
    # Reals: Cols 2, 4, 6... | Imags: Cols 3, 5, 7...
    
    reals = df.iloc[:, 2::2]
    imags = df.iloc[:, 3::2]

    # Get absolute min/max across ALL rows and ALL eigenvalue columns
    g_x_min, g_x_max = reals.min().min(), reals.max().max()
    g_y_min, g_y_max = imags.min().min(), imags.max().max()

    # Add 10% padding so points don't touch the border
    pad_x = (g_x_max - g_x_min) * 0.1 if g_x_max != g_x_min else 1.0
    pad_y = (g_y_max - g_y_min) * 0.1 if g_y_max != g_y_min else 1.0

    fixed_limits = {
        'x': [g_x_min - pad_x, g_x_max + pad_x],
        'y': [g_y_min - pad_y, g_y_max + pad_y]
    }
    
    return df, fixed_limits

# Load Data & Limits
try:
    df, global_limits = load_and_prep_data()
except Exception as e:
    st.error(f"Error: {e}. Please ensure you ran the parquet conversion script.")
    st.stop()

# --- 2. CONTROLS (SLIDERS) ---
st.sidebar.header("Parameters")

# Gamma Slider
unique_gammas = sorted(df.iloc[:, 0].unique())
sel_gamma = st.sidebar.select_slider("Gamma (γ)", options=unique_gammas)

# Omega Slider (Updates based on Gamma)
available_omegas = sorted(df[df.iloc[:, 0] == sel_gamma].iloc[:, 1].unique())
sel_omega = st.sidebar.select_slider("Omega (ω)", options=available_omegas)

# --- 3. FILTER & PLOT ---
# Get the specific row for these parameters
row = df[(df.iloc[:, 0] == sel_gamma) & (df.iloc[:, 1] == sel_omega)]

if not row.empty:
    # Extract data (Skip first 2 cols)
    data = row.iloc[0, 2:].values
    current_reals = data[::2]
    current_imags = data[1::2]

    # Build Plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=current_reals, 
        y=current_imags,
        mode='markers',
        marker=dict(size=8, color='royalblue', line=dict(width=1, color='black')),
        name='Eigenvalues'
    ))

    # --- APPLY FIXED LIMITS ---
    # This block forces the camera to stay still
    fig.update_layout(
        title=f"Spectrum at γ={sel_gamma:.4f}, ω={sel_omega:.4f}",
        xaxis=dict(
            range=global_limits['x'],  # <--- LOCKED X RANGE
            title="Real Part (Re)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black'
        ),
        yaxis=dict(
            range=global_limits['y'],  # <--- LOCKED Y RANGE
            title="Imaginary Part (Im)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black',
            scaleanchor="x",           # Optional: Forces 1:1 aspect ratio
            scaleratio=1
        ),
        width=700,
        height=700,
        template="plotly_white"
    )

    st.plotly_chart(fig)
else:
    st.error("No data found for this combination.")