import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# Load data efficiently
print("Loading data...")
df = pd.read_csv('../rust/eigenvalues_10.csv', header=None)

gamma_vals = sorted(df[0].unique())
omega_vals = sorted(df[1].unique())
n_eigvals = (df.shape[1] - 2) // 2

print(f"Found {len(gamma_vals)} gamma values, {len(omega_vals)} omega values")
print(f"Number of eigenvalues per point: {n_eigvals}")

# Pre-compute axis limits (skip first 2 columns which are gamma, omega)
all_real = df.iloc[:, 2::2].values.flatten()
all_imag = df.iloc[:, 3::2].values.flatten()
x_range = [np.nanmin(all_real) * 1.05, np.nanmax(all_real) * 1.05]
y_range = [np.nanmin(all_imag) * 1.05, np.nanmax(all_imag) * 1.05]

print(f"Axis ranges: x={x_range}, y={y_range}")

# Index for faster lookup
df['gamma_idx'] = df[0].map({v: i for i, v in enumerate(gamma_vals)})
df['omega_idx'] = df[1].map({v: i for i, v in enumerate(omega_vals)})
df.drop(columns=[0, 1], inplace=True)
df.set_index(['gamma_idx', 'omega_idx'], inplace=True)



app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='spectrum-plot', style={'height': '70vh'}),
    html.Div([
        html.Label('γ (Gamma)', style={'font-weight': 'bold'}),
        dcc.Slider(
            id='gamma-slider',
            min=0,
            max=len(gamma_vals)-1,
            value=len(gamma_vals)//2,
            step=1,
            marks={i: f'{gamma_vals[i]:.3f}' 
                   for i in range(0, len(gamma_vals), max(1, len(gamma_vals)//10))},
            updatemode='drag'  # Real-time updates while dragging
        ),
    ], style={'padding': '20px'}),
    html.Div([
        html.Label('ω (Omega)', style={'font-weight': 'bold'}),
        dcc.Slider(
            id='omega-slider',
            min=0,
            max=len(omega_vals)-1,
            value=len(omega_vals)//2,
            step=1,
            marks={i: f'{omega_vals[i]:.3f}'
                   for i in range(0, len(omega_vals), max(1, len(omega_vals)//10))},
            updatemode='drag'  # Real-time updates while dragging
        ),
    ], style={'padding': '20px'}),
])

@app.callback(
    Output('spectrum-plot', 'figure'),
    Input('gamma-slider', 'value'),
    Input('omega-slider', 'value')
)
def update_plot(gamma_idx, omega_idx):
    row = df.loc[(gamma_idx, omega_idx)]
    
    # Extract eigenvalues (gamma and omega are now in index, not in row.iloc)
    real_parts = [row.iloc[2*i] for i in range(n_eigvals)]
    imag_parts = [row.iloc[2*i+1] for i in range(n_eigvals)]
    
    for r, i in zip(real_parts, imag_parts):
        if r == gamma_vals[gamma_idx] and i == omega_vals[omega_idx]:
            real_parts.remove(r)
            imag_parts.remove(i)

    
    # real_parts.remove(gamma_vals[gamma_idx])
    # imag_parts.remove(omega_vals[omega_idx])
    
    # print(real_parts.index(gamma_vals[gamma_idx]), imag_parts.index(omega_vals[omega_idx]))
    
    fig = go.Figure(data=go.Scattergl(
        x=real_parts,
        y=imag_parts,
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(width=0.5, color='black')
        )
    ))
    
    fig.update_layout(
        title = r'Lindbladian Spectrum: g={g:.4f}, $\Omega$={o:.4f}'.format(g=gamma_vals[gamma_idx],o=omega_vals[omega_idx]),
        xaxis=dict(
            title='Re(λ)',
            range=x_range,
            showgrid=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        yaxis=dict(
            title='Im(λ)',
            range=y_range,
            showgrid=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black'
        ),
        height=600,
        hovermode='closest',
        transition={'duration': 0}  # No animation lag
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, dev_tools_hot_reload=False, use_reloader=False)
