import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

# --- 1. Data Loading ---
print("Loading data...")
# 'header=None' is critical here because your file starts with data immediately (0.1, 0...)
# If omitted, the first row would disappear into column headers.
df = pd.read_csv('../rust/occupation_time_8.csv', header=None)

# Columns 0 and 1 are parameters (Gamma, Omega)
gamma_vals = sorted(df[0].unique())
omega_vals = sorted(df[1].unique())

# Calculate number of time steps
# Total columns minus 2 parameter columns, divided by 2 (pairs of Occ, Corr)
n_timesteps = (df.shape[1] - 2) // 2
time_axis = np.arange(n_timesteps)

print(f"Loaded {len(gamma_vals)} γ values, {len(omega_vals)} Ω values.")
print(f"Detected {n_timesteps} time steps per parameter pair.")

# --- 2. Indexing for Performance ---
# Create a mapping for sliders to look up rows instantly
df['gamma_idx'] = df[0].map({v: i for i, v in enumerate(gamma_vals)})
df['omega_idx'] = df[1].map({v: i for i, v in enumerate(omega_vals)})

# Set MultiIndex for fast .loc[] access
df.set_index(['gamma_idx', 'omega_idx'], inplace=True)

# Drop original parameter columns to keep only the time-series data
# The dataframe now contains ONLY: occ1, corr1, occ2, corr2...
df_data = df.drop(columns=[0, 1])

app = Dash(__name__)

# --- 3. App Layout ---
app.layout = html.Div([
    html.H3("System Dynamics: Occupation, Correlation & Δ", style={'textAlign': 'center'}),
    
    # Plot Area
    dcc.Graph(id='dynamics-plot', style={'height': '80vh'}),
    
    # Control Area
    html.Div([
        # Gamma Slider
        html.Div([
            html.Label('γ (Gamma)', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='gamma-slider',
                min=0,
                max=len(gamma_vals)-1,
                value=0, # Start at first index
                step=1,
                marks={i: f'{gamma_vals[i]:.2f}' 
                       for i in range(0, len(gamma_vals), max(1, len(gamma_vals)//10))},
                updatemode='drag'
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px'}),
        
        # Omega Slider
        html.Div([
            html.Label('Ω (Omega)', style={'font-weight': 'bold'}),
            dcc.Slider(
                id='omega-slider',
                min=0,
                max=len(omega_vals)-1,
                value=0, # Start at first index
                step=1,
                marks={i: f'{omega_vals[i]:.2f}' 
                       for i in range(0, len(omega_vals), max(1, len(omega_vals)//10))},
                updatemode='drag'
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'padding': '10px', 'float': 'right'}),
    ], style={'background-color': '#f9f9f9', 'border-top': '1px solid #ddd'})
])

# --- 4. Plotting Logic ---
@app.callback(
    Output('dynamics-plot', 'figure'),
    Input('gamma-slider', 'value'),
    Input('omega-slider', 'value')
)
def update_plot(gamma_idx, omega_idx):
    # Retrieve the specific row for these parameters
    try:
        row_series = df_data.loc[(gamma_idx, omega_idx)]
    except KeyError:
        return go.Figure() # Return empty if indices mismatch (safety)

    # Get the scalar values for display/calculation
    g_val = gamma_vals[gamma_idx]
    w_val = omega_vals[omega_idx]

    # --- Slicing the Data ---
    # The row contains: [occ0, corr0, occ1, corr1, occ2, corr2, ...]
    # values[::2]  -> takes indices 0, 2, 4... (Occupations)
    # values[1::2] -> takes indices 1, 3, 5... (Correlations)
    occupations = row_series.values[::2]
    correlations = row_series.values[1::2]
    
    # --- Calculation ---
    # Formula: g*delta = g * correlation - (3g + 1) * occupation + g
    # This is a vectorized operation (fast)
    g_delta = (g_val * correlations) - ((3 * g_val + 1) * occupations) + g_val

    # --- Create Subplots ---
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f"Occupation (n)", 
            f"Correlation (c)", 
            r"$g \cdot \Delta = g c - (3g+1)n + g$"
        )
    )

    # Trace 1: Occupation
    fig.add_trace(go.Scatter(
        x=time_axis, y=occupations, 
        mode='lines', name='Occupation', line=dict(color='blue', width=2)
    ), row=1, col=1)

    # Trace 2: Correlation
    fig.add_trace(go.Scatter(
        x=time_axis, y=correlations, 
        mode='lines', name='Correlation', line=dict(color='red', width=2)
    ), row=2, col=1)

    # Trace 3: g*Delta
    fig.add_trace(go.Scatter(
        x=time_axis, y=g_delta, 
        mode='lines', name='g·Δ', line=dict(color='green', width=2)
    ), row=3, col=1)

    # Layout Polish
    fig.update_layout(
        height=700,
        title_text=f"Parameters: γ={g_val:.4f}, Ω={w_val:.4f}",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    
    fig.update_yaxes(title_text="n(t)", row=1, col=1)
    fig.update_yaxes(title_text="c(t)", row=2, col=1)
    fig.update_yaxes(title_text="g·Δ(t)", row=3, col=1)
    fig.update_xaxes(title_text="Time Step", row=3, col=1)

    return fig

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)