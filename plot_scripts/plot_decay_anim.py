import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# ---------------------------------------------------------
# 1. DATA LOADING (Unchanged)
# ---------------------------------------------------------
def load_data(filename):
    data_points = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if len(parts) < 3: continue
                try:
                    gp    = float(parts[0])
                    gm    = float(parts[1])
                    omega = float(parts[2])
                    raw_data = parts[3:]
                    num_modes = len(raw_data) // 5
                    for i in range(num_modes):
                        idx = i * 5
                        data_points.append({
                            'gp':                 gp,
                            'gm':                 gm,
                            'omega':              omega,
                            'decay_rate':         float(raw_data[idx]),
                            'energy_oscillation': float(raw_data[idx+1]),
                            'overlap':            float(raw_data[idx+2]),
                            'occupation':         float(raw_data[idx+3]),
                            'size':               int(float(raw_data[idx+4])),
                        })
                except ValueError: continue
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found."); return pd.DataFrame()
    return pd.DataFrame(data_points)


# ---------------------------------------------------------
# 2. INTERACTIVE PLOTTER
# ---------------------------------------------------------
def interactive_plot(df):
    if df.empty: print("No data."); return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True, background='w', foreground='k')

    unique_gps = sorted(df['gp'].unique())
    unique_gms = sorted(df['gm'].unique())
    unique_ws  = sorted(df['omega'].unique())

    # ── Main window ──────────────────────────────────────
    win = QtWidgets.QWidget()
    win.setWindowTitle("Spectrum Viewer")
    root = QtWidgets.QVBoxLayout(win)

    # Title label at the top
    title_lbl = QtWidgets.QLabel("")
    title_lbl.setAlignment(QtCore.Qt.AlignCenter)
    title_lbl.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px;")
    root.addWidget(title_lbl)

    # Two stacked PyQtGraph plots
    glw = pg.GraphicsLayoutWidget()
    root.addWidget(glw)

    plot1 = glw.addPlot(row=0, col=0)
    plot2 = glw.addPlot(row=1, col=0)

    plot1.setLabel('left',   'Overlap Magnitude')
    plot1.setLabel('bottom', 'Energy Oscillation  Im[λ]')
    plot2.setLabel('left',   'Overlap Magnitude')
    plot2.setLabel('bottom', 'Decay Rate  Re[λ]')

    # Compute global limits from full dataset
    e_min, e_max = df['energy_oscillation'].min(), df['energy_oscillation'].max()
    d_min, d_max = df['decay_rate'].min(),         df['decay_rate'].max()
    y_min, y_max = df[df['overlap'] > 0]['overlap'].min(), df['overlap'].max()
    pad_e = 0.1 * (abs(e_max) if e_max != 0 else 1.0)
    pad_d = 0.1 * (abs(d_max) if d_max != 0 else 1.0)
    pad_y = 0.1 * (abs(y_max) if y_max != 0 else 1.0)

    # Lock axes — disables all auto-rescaling
    plot1.setXRange(e_min - pad_e, e_max + pad_e, padding=0)
    plot1.setYRange(y_min * 0.5,   y_max + pad_y, padding=0)
    plot2.setXRange(d_min - pad_d, d_max + pad_d, padding=0)
    plot2.setYRange(y_min * 0.5,   y_max + pad_y, padding=0)

    for p in (plot1, plot2):
        p.enableAutoRange(False)
        p.setMouseEnabled(x=True, y=True) 
        p.showGrid(x=True, y=True, alpha=0.3)
        p.setLogMode(x=False, y=False)

    # ── Scatter items ─────────────────────────────────────
    # Normal modes  (blue circles)
    BRUSH_NORMAL = pg.mkBrush(65, 105, 225, 170)
    PEN_NONE     = pg.mkPen(None)
    PEN_BLACK    = pg.mkPen('k', width=0.5)
    PEN_RED      = pg.mkPen('r', width=2)
    BRUSH_NONE   = pg.mkBrush(None)

    sn1 = pg.ScatterPlotItem(size=8,  pen=PEN_NONE,  brush=BRUSH_NORMAL)
    sn2 = pg.ScatterPlotItem(size=8,  pen=PEN_NONE,  brush=BRUSH_NORMAL)
    # Jordan blocks (stars, colored by block size)
    sj1 = pg.ScatterPlotItem(size=15, symbol='star', pen=PEN_BLACK)
    sj2 = pg.ScatterPlotItem(size=15, symbol='star', pen=PEN_BLACK)
    # Selection overlay (hollow red rings)
    ss1 = pg.ScatterPlotItem(size=18, pen=PEN_RED, brush=BRUSH_NONE)
    ss2 = pg.ScatterPlotItem(size=18, pen=PEN_RED, brush=BRUSH_NONE)

    for plot, n, j, s in [(plot1, sn1, sj1, ss1), (plot2, sn2, sj2, ss2)]:
        plot.addItem(n); plot.addItem(j); plot.addItem(s)

    # Legend (manual, via dummy items)
    leg = plot1.addLegend(offset=(10, 10))
    leg.addItem(pg.ScatterPlotItem(size=8,  pen=PEN_NONE, brush=BRUSH_NORMAL),          'Normal (k=1)')
    leg.addItem(pg.ScatterPlotItem(size=15, symbol='star', pen=PEN_BLACK,
                                   brush=pg.mkBrush(220, 50, 50, 200)),                  'Jordan (k>1)')
    leg.addItem(pg.ScatterPlotItem(size=18, pen=PEN_RED, brush=BRUSH_NONE),              'Selected')

    # ── State ────────────────────────────────────────────
    selected_indices = set()          # stable df integer indices
    state = {
        'subset':           pd.DataFrame(),
        'normal_df_idx':    np.array([], dtype=int),
        'jordan_df_idx':    np.array([], dtype=int),
    }

    # ── Helpers ───────────────────────────────────────────
    def _jordan_colors(sizes):
        """Map Jordan block sizes → RGBA tuples (reddish gradient)."""
        norm = np.clip((sizes - 1) / 4.0, 0, 1)
        r = 255
        g = (80  * (1 - norm)).astype(int)
        b = (80  * (1 - norm)).astype(int)
        a = 200
        return [pg.mkBrush(int(r), int(g[i]), int(b[i]), a) for i in range(len(norm))]

    def update_selection_overlay():
        subset = state['subset']
        if subset.empty or not selected_indices:
            ss1.setData([], []); ss2.setData([], [])
            return
        sel = subset[subset.index.isin(selected_indices)]
        if sel.empty:
            ss1.setData([], []); ss2.setData([], [])
        else:
            ss1.setData(x=sel['energy_oscillation'].values, y=sel['overlap'].values)
            ss2.setData(x=sel['decay_rate'].values,         y=sel['overlap'].values)

    def update():
        gp = unique_gps[sl_gp.value()]
        gm = unique_gms[sl_gm.value()]
        w  = unique_ws [sl_w .value()]

        lbl_gp.setText(f"γ₊ = {gp:.4f}")
        lbl_gm.setText(f"γ₋ = {gm:.4f}")
        lbl_w .setText(f"ω  = {w:.4f}")
        title_lbl.setText(f"γ₊ = {gp:.4f}   γ₋ = {gm:.4f}   ω = {w:.4f}")

        subset = df[(df['gp'] == gp) & (df['gm'] == gm) & (df['omega'] == w)]
        normal = subset[subset['size'] == 1]
        jordan = subset[subset['size'] > 1]

        state['subset']        = subset
        state['normal_df_idx'] = normal.index.values
        state['jordan_df_idx'] = jordan.index.values

        # Normal modes — fast array path
        if not normal.empty:
            sn1.setData(x=normal['energy_oscillation'].values, y=normal['overlap'].values)
            sn2.setData(x=normal['decay_rate'].values,         y=normal['overlap'].values)
        else:
            sn1.setData([], []); sn2.setData([], [])

        # Jordan blocks — per-point color by size
        if not jordan.empty:
            brushes = _jordan_colors(jordan['size'].values)
            spots1 = [{'pos': (x, y), 'brush': b}
                      for x, y, b in zip(jordan['energy_oscillation'], jordan['overlap'], brushes)]
            spots2 = [{'pos': (x, y), 'brush': b}
                      for x, y, b in zip(jordan['decay_rate'], jordan['overlap'], brushes)]
            sj1.setData(spots1); sj2.setData(spots2)
        else:
            sj1.setData([]); sj2.setData([])

        update_selection_overlay()

    # ── Pick / click events ───────────────────────────────
    def make_click_handler(is_normal):
        """Returns a handler that maps clicked SpotItems back to df indices."""
        def handler(scatter_item, points, *_):
            idx_arr = state['normal_df_idx'] if is_normal else state['jordan_df_idx']
            for pt in points:
                local = pt.index()
                if local < len(idx_arr):
                    df_idx = idx_arr[local]
                    if df_idx in selected_indices:
                        selected_indices.discard(df_idx)
                    else:
                        selected_indices.add(df_idx)
            update_selection_overlay()
        return handler

    sn1.sigClicked.connect(make_click_handler(True))
    sn2.sigClicked.connect(make_click_handler(True))
    sj1.sigClicked.connect(make_click_handler(False))
    sj2.sigClicked.connect(make_click_handler(False))

    # ── Slider panel ──────────────────────────────────────
    slider_panel = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(slider_panel)
    grid.setContentsMargins(10, 6, 10, 6)

    def make_slider(values):
        s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        s.setMinimum(0); s.setMaximum(len(values) - 1); s.setValue(0)
        return s

    sl_gp = make_slider(unique_gps)
    sl_gm = make_slider(unique_gms)
    sl_w  = make_slider(unique_ws)

    lbl_gp = QtWidgets.QLabel(f"γ₊ = {unique_gps[0]:.4f}")
    lbl_gm = QtWidgets.QLabel(f"γ₋ = {unique_gms[0]:.4f}")
    lbl_w  = QtWidgets.QLabel(f"ω  = {unique_ws[0]:.4f}")

    for lbl in (lbl_gp, lbl_gm, lbl_w):
        lbl.setMinimumWidth(130)
        lbl.setStyleSheet("font-family: monospace; font-size: 12px;")

    for row, (lbl, sl) in enumerate([(lbl_gp, sl_gp), (lbl_gm, sl_gm), (lbl_w, sl_w)]):
        grid.addWidget(lbl, row, 0)
        grid.addWidget(sl,  row, 1)

    sl_gp.valueChanged.connect(lambda _: update())
    sl_gm.valueChanged.connect(lambda _: update())
    sl_w .valueChanged.connect(lambda _: update())

    root.addWidget(slider_panel)

    # Clear selection button
    btn = QtWidgets.QPushButton("Clear Selection")
    btn.clicked.connect(lambda: (selected_indices.clear(), update_selection_overlay()))
    root.addWidget(btn)

    win.resize(1000, 950)
    win.show()
    update()

    if sys.flags.interactive == 0:
        app.exec_()


df = load_data('../rust/decay.csv')
interactive_plot(df)