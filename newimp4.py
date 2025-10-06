import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math

# Constants
bullet_diameter = 5.6
series_size = 10
ring_radii = [2.5, 5.2, 13.2, 21.2, 29.2, 37.2, 45.2, 53.2, 61.2, 69.2, 77.2]
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import io

# ========================
# Configuration
# ========================
BULLET_DIAMETER = 5.6
SERIES_SIZE = 10
RING_RADII = [2.5, 5.2, 13.2, 21.2, 29.2, 37.2, 45.2, 53.2, 61.2, 69.2, 77.2]

# ========================
# Page config
# ========================
st.set_page_config(
    page_title="Target Plotter - Dual Scoring",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================
# Session state init
# ========================
if 'coords' not in st.session_state:
    st.session_state.coords = []
if 'series_index' not in st.session_state:
    st.session_state.series_index = 0
if 'show_optimal' not in st.session_state:
    st.session_state.show_optimal = False
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 25

# ========================
# Scoring functions
# ========================
def score_shot(x, y):
    d = math.hypot(x, y)
    for score, r in zip(range(10, 0, -1), RING_RADII[1:]):
        if d - BULLET_DIAMETER/2 <= r:
            return score
    return 0

def score_shot_decimal(x, y):
    d = math.hypot(x, y)
    inner_edge = RING_RADII[1] + BULLET_DIAMETER/2
    step = inner_edge / 10.0
    if d <= inner_edge:
        k = math.ceil(d/step) if step>0 else 10
        k = min(max(k,1),10)
        return round(10.9 - 0.1*(k-1), 2)
    for i in range(1, len(RING_RADII)-1):
        ir = RING_RADII[i] + BULLET_DIAMETER/2
        or_ = RING_RADII[i+1] + BULLET_DIAMETER/2
        if d <= or_:
            band = or_ - ir
            if band <= 0:
                return float(max(0,10-(i-1)))
            stp = band/10.0
            k = math.ceil((d-ir)/stp)
            k = min(max(k,1),10)
            base = 10-(i-1)
            return round(base - 0.1*k, 2)
    return 0.0

def is_inner_ten(x, y):
    return score_shot_decimal(x, y) >= 10.3

def find_optimal_center(coords):
    if not coords:
        return 0, 0, 0, 0
    xs, ys = zip(*coords)
    cx_init = sum(xs) / len(xs)
    cy_init = sum(ys) / len(ys)
    actual_score = sum(score_shot_decimal(x, y) for x, y in coords)
    search_radius = 15
    grid_step = 1.0
    best_cx, best_cy = cx_init, cy_init
    best_score = sum(score_shot_decimal(x, y) for x, y in coords)
    for _ in range(4):
        improved = False
        steps = int(search_radius / grid_step)
        for dx_step in range(-steps, steps + 1):
            for dy_step in range(-steps, steps + 1):
                test_cx = best_cx + dx_step * grid_step
                test_cy = best_cy + dy_step * grid_step
                total = sum(score_shot_decimal(x - test_cx, y - test_cy) for x, y in coords)
                if total > best_score:
                    best_score = total
                    best_cx, best_cy = test_cx, test_cy
                    improved = True
        if improved:
            search_radius *= 0.6
            grid_step *= 0.5
        else:
            break
    return best_cx, best_cy, actual_score, best_score

# ========================
# Plot function
# ========================
def plot_target(coords, idx, show_optimal, zoom):
    fig, ax = plt.subplots(figsize=(8,8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    num_series = (len(coords) + SERIES_SIZE - 1) // SERIES_SIZE
    is_all = (idx == num_series)
    if is_all:
        sel = coords
        start = 0
    else:
        start = idx * SERIES_SIZE
        sel = coords[start:start+SERIES_SIZE]
    if show_optimal and sel:
        cx, cy, actual_score, best_score = find_optimal_center(sel)
        disp = [(x - cx, y - cy) for x, y in sel]
    else:
        cx, cy = 0, 0
        disp = sel
    dec_tot = 0
    int_tot = 0
    inner_count = 0
    for j, ((sx, sy), (dx, dy)) in enumerate(zip(sel, disp), start=start+1):
        ax.add_patch(Circle((dx, dy), BULLET_DIAMETER/2, facecolor='grey', edgecolor='white'))
        px, py = (dx, dy) if show_optimal else (sx, sy)
        ds = score_shot_decimal(px, py)
        nd = score_shot(px, py)
        dec_tot += ds
        int_tot += nd
        if is_inner_ten(px, py):
            inner_count += 1
        if not is_all:
            ax.text(dx, dy, str(j),
                    color='yellow' if is_inner_ten(px, py) else 'white',
                    ha='center', va='center', weight='bold')
    for r in RING_RADII:
        ax.add_patch(Circle((0, 0), r, fill=False, color='white', linewidth=0.5))
    shots = len(sel)
    txt = f"Non decimal: {int_tot}/{shots*10}  Decimal: {dec_tot:.1f}/{shots*10}  Inner tens: {inner_count}/{shots}"
    ax.text(0.5, 0.05, txt, transform=ax.transAxes,
            ha='center', color='yellow', weight='bold')
    title = "All Shots Combined" if is_all else f"Series {idx+1} ({start+1}-{start+shots})"
    if show_optimal:
        title += " (Centered)"
    ax.set_title(title, color='white')
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)
    ax.set_xlabel("X (mm)", color='white')
    ax.set_ylabel("Y (mm)", color='white')
    ax.grid(True, color='white', alpha=0.2, linestyle='--')
    return fig, int_tot, dec_tot, inner_count, shots, is_all, start, (cx, cy)

# ========================
# UI: Data input
# ========================
st.title("🎯 Target Plotter")
tab1, tab2 = st.tabs(["📁 Upload Excel","⌨️ Manual Input"])
with tab1:
    up = st.file_uploader("Excel file", type=['xlsx','xls'])
    if up:
        df = pd.read_excel(up)
        nums = df.select_dtypes(include='number').columns.tolist()
        if len(nums) < 2:
            st.error("Need at least two numeric columns")
        else:
            xcol = st.selectbox("X column", nums, index=0)
            ycol = st.selectbox("Y column", nums, index=1 if len(nums) > 1 else 0)
            xs = pd.to_numeric(df[xcol], errors='coerce').dropna()
            ys = pd.to_numeric(df[ycol], errors='coerce').dropna()
            n = min(len(xs), len(ys))
            if n > 0:
                st.session_state.coords = list(zip(xs[:n], ys[:n]))
                st.session_state.series_index = 0
                st.success(f"Loaded {n} shots")
            else:
                st.error("No valid data")
with tab2:
    x_in = st.text_area("X values", height=200)
    y_in = st.text_area("Y values", height=200)
    if st.button("Load"):
        try:
            xs = [float(x) for x in x_in.split()]
            ys = [float(y) for y in y_in.split()]
            if len(xs) == len(ys) and len(xs) > 0:
                st.session_state.coords = list(zip(xs, ys))
                st.session_state.series_index = 0
                st.success(f"Loaded {len(xs)} shots")
            else:
                st.error("Counts mismatch or empty")
        except:
            st.error("Invalid numbers")

# ========================
# UI: Plot & navigation
# ========================
if st.session_state.coords:
    coords = st.session_state.coords
    num_series = (len(coords) + SERIES_SIZE - 1) // SERIES_SIZE
    total = num_series + 1  # includes "All Shots Combined"
    series_labels = [f"Series {i+1}" for i in range(num_series)] + ["All Shots Combined"]
    selected_slider = st.slider(
        "Select Series",
        min_value=1,
        max_value=total,
        value=st.session_state.series_index + 1,
        format="%d"
    )
    if selected_slider == total:
        st.session_state.series_index = num_series  # All Shots combined index
    else:
        st.session_state.series_index = selected_slider - 1  # zero based index for series
    st.write(f"**Viewing:** {series_labels[st.session_state.series_index]}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🔍 Zoom In", key="zoom_in_btn"):
            st.session_state.zoom_level = max(10, st.session_state.zoom_level * 0.8)
            st.rerun()
    with c2:
        if st.button("🔍 Zoom Out", key="zoom_out_btn"):
            st.session_state.zoom_level = min(50, st.session_state.zoom_level * 1.25)
            st.rerun()
    with c3:
        if st.button("🏠 Reset Zoom", key="reset_btn"):
            st.session_state.zoom_level = 25
            st.rerun()
    with c4:
        if st.button("Show Optimal" if not st.session_state.show_optimal else "Show Raw", key="toggle_optimal"):
            st.session_state.show_optimal = not st.session_state.show_optimal
            st.rerun()
    fig, nd, dd, inn, shots, is_all, start, center = plot_target(
        coords,
        st.session_state.series_index,
        st.session_state.show_optimal,
        st.session_state.zoom_level
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    sel = coords if is_all else coords[start:start+shots]
    if st.session_state.show_optimal and sel:
        cx, cy = center
        sel_shifted = [(x - cx, y - cy) for x, y in sel]
    else:
        sel_shifted = sel
    df2 = pd.DataFrame([
        {"idx": i + start + 1, "x": x, "y": y,
         "int": score_shot(x, y), "dec": score_shot_decimal(x, y),
         "inner": int(is_inner_ten(x, y))}
        for i, (x, y) in enumerate(sel_shifted)
    ])
    st.download_button(
        "Download CSV",
        data=df2.to_csv(index=False).encode(),
        file_name="all_shots.csv" if is_all else f"series_{st.session_state.series_index+1}.csv"
    )
    buf = io.BytesIO()
    fig2, *_ = plot_target(
        coords,
        st.session_state.series_index,
        st.session_state.show_optimal,
        st.session_state.zoom_level
    )
    fig2.savefig(buf, format='png', facecolor='black', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig2)
    st.download_button(
        "Download Plot",
        data=buf,
        file_name="all_shots.png" if is_all else f"series_{st.session_state.series_index+1}.png",
        mime="image/png"
    )
    if st.button("Download All Series Data"):
        all_data = []
        for series_idx in range(num_series):
            start_idx = series_idx * SERIES_SIZE
            end_idx = min(start_idx + SERIES_SIZE, len(coords))
            series_coords = coords[start_idx:end_idx]
            if st.session_state.show_optimal and series_coords:
                cx, cy, _, _ = find_optimal_center(series_coords)
                series_coords = [(x - cx, y - cy) for x, y in series_coords]
            for i, (x, y) in enumerate(series_coords):
                all_data.append({
                    "Series": series_idx + 1,
                    "Shot": start_idx + i + 1,
                    "X": x,
                    "Y": y,
                    "Non decimal": score_shot(x, y),
                    "Decimal": score_shot_decimal(x, y),
                    "Inner tens": int(is_inner_ten(x, y))
                })
        df_all = pd.DataFrame(all_data)
        csv_all = df_all.to_csv(index=False).encode()
        st.download_button(
            "Download All Series CSV",
            data=csv_all,
            file_name="all_series_shots.csv",
            mime="text/csv"
        )
        st.markdown("### Download All Series Images")
        for series_idx in range(num_series):
            fig, *_ = plot_target(
                coords,
                series_idx,
                st.session_state.show_optimal,
                st.session_state.zoom_level
            )
            buf_img = io.BytesIO()
            fig.savefig(buf_img, format='png', facecolor='black', dpi=150, bbox_inches='tight')
            buf_img.seek(0)
            plt.close(fig)
            st.download_button(
                f"Download Series {series_idx+1} Image",
                data=buf_img,
                file_name=f"series_{series_idx+1}.png",
                mime="image/png"
            )
        fig_all, *_ = plot_target(
            coords,
            num_series,
            st.session_state.show_optimal,
            st.session_state.zoom_level
        )
        buf_all = io.BytesIO()
        fig_all.savefig(buf_all, format='png', facecolor='black', dpi=150, bbox_inches='tight')
        buf_all.seek(0)
        plt.close(fig_all)
        st.download_button(
            "Download All Shots Image",
            data=buf_all,
            file_name="all_shots.png",
            mime="image/png"
        )
else:
    st.info("Upload data or enter coordinates to begin plotting")

st.caption("🎯 Target Plotter v2.4")

def score_shot_decimal(x, y, shot_no=None):
    d = math.hypot(x, y)
    inner_ten_radius = ring_radii[1] + 2.8
    step_inner = inner_ten_radius / 10
    for k in range(10):
        if d <= (k + 1) * step_inner:
            return round(10.9 - 0.1 * k, 2)
    for i in range(1, len(ring_radii) - 1):
        inner_r = ring_radii[i] + 2.8
        outer_r = ring_radii[i + 1] + 2.8
        if d <= outer_r:
            step = (outer_r - inner_r) / 10
            for k in range(10):
                if d <= inner_r + (k + 1) * step:
                    return round(10 - (i - 1) - 0.1 * k, 2)
    return 0

def is_inner_ten(x, y):
    return math.hypot(x, y) + bullet_diameter / 2 <= ring_radii[0]

def create_target_plot(coords, series_start=0, series_end=None, title="Target Plot"):
    if series_end is None:
        series_end = len(coords)

    fig = go.Figure()

    # Draw full rings without clipping
    for r in ring_radii:
        fig.add_shape(
            type="circle", xref="x", yref="y",
            x0=-r, y0=-r, x1=r, y1=r,
            line=dict(color="white", width=2),
            fillcolor="rgba(0,0,0,0)",
            cliponaxis=False
        )

    # Place ring numbers
    for i, num in enumerate(range(8, 0, -1), start=3):
        if i < len(ring_radii) - 1:
            mid = (ring_radii[i] + ring_radii[i - 1]) / 2 - 1
            for x_off, y_off in [(mid,0),(-mid,0),(0,mid),(0,-mid)]:
                fig.add_annotation(
                    x=x_off, y=y_off,
                    text=str(num),
                    showarrow=False,
                    font=dict(color="white", size=12, family="Arial Black")
                )

    total_score = 0
    inner_tens = 0
    shot_slice = coords[series_start:series_end]
    for idx, (x, y) in enumerate(shot_slice, start=1+series_start):
        sc = score_shot_decimal(x, y)
        total_score += sc
        if is_inner_ten(x, y):
            inner_tens += 1

        # Bullet hole
        fig.add_shape(
            type="circle", xref="x", yref="y",
            x0=x - bullet_diameter/2, y0=y - bullet_diameter/2,
            x1=x + bullet_diameter/2, y1=y + bullet_diameter/2,
            line=dict(color="white", width=1),
            fillcolor="grey",
            cliponaxis=False
        )
        fig.add_annotation(
            x=x, y=y,
            text=str(idx),
            showarrow=False,
            font=dict(color="white", size=10, family="Arial Black")
        )

    # Hide all axis lines and grid
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, visible=False,
                     scaleanchor="x", scaleratio=1)

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=16)),
        plot_bgcolor="black",
        paper_bgcolor="black",
        width=600, height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    if series_end - series_start <= series_size:
        score_text = f"Series Decimal Score: {total_score:.2f}"
    else:
        score_text = f"Total Decimal Score: {total_score:.2f}   Inner Tens: {inner_tens}"

    fig.add_annotation(
        x=0, y=-23,
        text=score_text,
        showarrow=False,
        font=dict(color="yellow", size=14, family="Arial Black")
    )

    return fig, total_score, inner_tens

def main():
    st.set_page_config(page_title="Target Plotter", page_icon="🎯", layout="wide")
    st.title("🎯 Target Plotter - ISSF 50m Rifle Scoring")
    st.sidebar.header("📊 Input Method")
    method = st.sidebar.radio("Choose input method:", ["Manual Input", "Excel File Upload"])

    if 'coords' not in st.session_state:
        st.session_state.coords = []

    if method == "Manual Input":
        with st.sidebar.form("manual_form"):
            x_txt = st.text_area("X coords (one per line):", key="x_txt")
            y_txt = st.text_area("Y coords (one per line):", key="y_txt")
            if st.form_submit_button("Load Coordinates"):
                try:
                    xs = [float(v) for v in x_txt.split() if v]
                    ys = [float(v) for v in y_txt.split() if v]
                    if len(xs) != len(ys):
                        st.sidebar.error("X/Y count mismatch!")
                    else:
                        st.session_state.coords = list(zip(xs, ys))
                        st.sidebar.success(f"Loaded {len(xs)} shots")
                except:
                    st.sidebar.error("Invalid numbers")
    else:
        with st.sidebar.form("excel_form"):
            up = st.file_uploader("Upload Excel", type=['xlsx','xls'])
            if up:
                df = pd.read_excel(up)
                nums = df.select_dtypes(include=[float,int]).columns
                if len(nums) < 2:
                    st.sidebar.error("Need ≥2 numeric columns")
                else:
                    xcol = st.selectbox("X column", nums, key="xcol")
                    ycol = st.selectbox("Y column", nums, key="ycol")
            if st.form_submit_button("Load from Excel") and up:
                xs = df[xcol].dropna().tolist()
                ys = df[ycol].dropna().tolist()
                n = min(len(xs), len(ys))
                st.session_state.coords = list(zip(xs[:n], ys[:n]))
                st.sidebar.success(f"Loaded {n} shots")

    coords = st.session_state.coords
    if coords:
        nseries = (len(coords) + series_size - 1) // series_size
        opts = ["All Shots Combined"] + [
            f"Series {i+1} (shots {i*series_size+1}-{min((i+1)*series_size,len(coords))})"
            for i in range(nseries)
        ]
        if 'view' not in st.session_state:
            st.session_state.view = opts[0]
        sel = st.selectbox("📈 Select View:", opts, index=opts.index(st.session_state.view))
        st.session_state.view = sel

        if sel == opts[0]:
            start, end, ttl = 0, len(coords), "All Shots Combined"
        else:
            sn = int(sel.split()[1]) - 1
            start = sn * series_size
            end = min(start + series_size, len(coords))
            ttl = f"Shot Group Series {sn+1} (shots {start+1}-{end})"

        fig, tot, xt = create_target_plot(coords, start, end, ttl)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 Total Shots", len(coords))
        c2.metric("📊 Score", f"{tot:.2f}")
        c3.metric("⭐ Inner Tens", xt)

        with st.expander("🔍 Shot-by-Shot Analysis"):
            data = []
            for i, (x, y) in enumerate(coords[start:end], start+1):
                sc = score_shot_decimal(x, y)
                data.append({
                    "Shot": i,
                    "X": f"{x:.2f}",
                    "Y": f"{y:.2f}",
                    "Score": f"{sc:.2f}",
                    "Inner Ten": "X" if is_inner_ten(x, y) else ""
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.info("Enter or upload coordinates to begin.")

if __name__ == "__main__":
    main()
