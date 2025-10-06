import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import io
from datetime import datetime

# ========================
# Configuration
# ========================
BULLET_DIAMETER = 5.6
SERIES_SIZE = 10
# RING_RADII[0] is center (0). RING_RADII[1] is inner-10 boundary (10-ring outer edge),
# then outward to 1-ring boundary.
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
# Initialize session state
# ========================
if 'coords' not in st.session_state:
    st.session_state.coords = []
if 'series_index' not in st.session_state:
    st.session_state.series_index = 0
if 'show_optimal' not in st.session_state:
    st.session_state.show_optimal = False
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 25
if 'show_all_downloads' not in st.session_state:
    st.session_state.show_all_downloads = False

# ========================
# Scoring functions
# ========================
def score_shot(x, y):
    """Calculate non-decimal score using touch rule; avoid double rounding."""
    d = math.hypot(x, y)
    # Rings 10..1 correspond to RING_RADII[1:] as outer edges
    for score, r in zip(range(10, 0, -1), RING_RADII[1:]):
        if d - BULLET_DIAMETER / 2 <= r:
            return score
    return 0

def score_shot_decimal(x, y):
    """
    Calculate decimal score with 0.1 steps within each ring band.
    Assumptions:
      - 10.9 at exact center; 10.0 at the inner 10 ring boundary (RING_RADII[1]).
      - The bullet counts if its edge reaches into the sub-band (touch rule).
    """
    d = math.hypot(x, y)

    # Inner 10 ring handling: split radius [0 .. RING_RADII[1]+BD/2] into 10 bands
    inner_edge = RING_RADII[1]
    inner_eff = inner_edge + BULLET_DIAMETER / 2
    step = inner_eff / 10.0

    if d <= inner_eff:
        # Bands: (0..step] => 10.9, (step..2*step] => 10.8, ..., (9*step..10*step] => 10.0
        k = math.ceil(d / step) if step > 0 else 10
        k = min(max(k, 1), 10)
        return round(10.9 - 0.1 * (k - 1), 2)

    # Outer rings: each ring band subdivided into 10 equal sub-bands
    # RING_RADII[i]..RING_RADII[i+1] maps to nominal scores 10-(i-1) downwards
    for i in range(1, len(RING_RADII) - 1):
        ir = RING_RADII[i]
        or_ = RING_RADII[i + 1]
        ir_eff = ir + BULLET_DIAMETER / 2
        or_eff = or_ + BULLET_DIAMETER / 2
        if d <= or_eff:
            band = or_eff - ir_eff
            if band <= 0:
                # Safety fallback: return nominal integer score at inner boundary
                return float(max(0, 10 - (i - 1)))
            stp = band / 10.0
            k = math.ceil((d - ir_eff) / stp)
            k = min(max(k, 1), 10)
            base = 10 - (i - 1)  # integer score at inner boundary of this ring
            return round(base - 0.1 * k, 2)

    return 0.0

def is_inner_ten(x, y):
    """Inner 10 criterion as >= 10.3."""
    return score_shot_decimal(x, y) >= 10.3

# ========================
# Helper: draw rings and optional labels
# ========================
def draw_rings(ax, show_labels=False):
    # Draw ring circles
    for r in RING_RADII:
        ax.add_patch(Circle((0, 0), r, fill=False, color='white', linewidth=0.5))
    if show_labels:
        for k, r in zip(range(10, 0, -1), RING_RADII[1:]):
            ax.text(0, r + 1.5, f"{k}", color='white', fontsize=8, ha='center', va='bottom')

# ========================
# Plot function
# ========================
def plot_target(coords, series_idx, show_optimal, zoom_level, show_ring_labels=False):
    """Create target plot. Scores are computed from original coordinates; centering affects only display."""
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(colors='white', direction='in', labelsize=10)
    ax.set_aspect('equal')

    # Determine selection
    num_series = (len(coords) + SERIES_SIZE - 1) // SERIES_SIZE
    is_combined = (series_idx == num_series)

    if is_combined:
        start = 0
        end = len(coords)
        sel = coords
    else:
        start = series_idx * SERIES_SIZE
        end = min(start + SERIES_SIZE, len(coords))
        sel = coords[start:end]

    # Display coords (possibly centered)
    display_sel = sel
    if show_optimal and sel:
        xs, ys = zip(*sel)
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        display_sel = [(x - cx, y - cy) for x, y in sel]

    # Compute scores from ORIGINAL sel; draw using display_sel
    decimal_total = 0.0
    non_decimal_total = 0
    inner_tens = 0

    for j, ((sx, sy), (dx, dy)) in enumerate(zip(sel, display_sel), start=start + 1):
        # draw shot
        ax.add_patch(Circle((dx, dy), BULLET_DIAMETER / 2,
                            facecolor='grey', edgecolor='white', linewidth=1))

        # scoring
        ds = score_shot_decimal(sx, sy)
        nd = score_shot(sx, sy)
        decimal_total += ds
        non_decimal_total += nd
        if is_inner_ten(sx, sy):
            inner_tens += 1
        color = 'yellow' if is_inner_ten(sx, sy) else 'white'

        # annotate shot numbers for individual series
        if not is_combined:
            ax.text(dx, dy, f"{j}", fontsize=11, ha='center', va='center',
                    color=color, weight='bold')

    # Draw rings and optional labels
    draw_rings(ax, show_labels=show_ring_labels)

    # Score display anchored to axes coords (stable under zoom)
    num_shots = len(sel)
    score_text = f"Non-Decimal: {int(non_decimal_total)}/{num_shots*10}   Decimal: {decimal_total:.1f}/{num_shots*10}   Inner-10s: {inner_tens}/{num_shots}"
    ax.text(0.5, 0.04, score_text, transform=ax.transAxes,
            ha='center', va='bottom', color='yellow', fontsize=13, weight='bold')

    # Title
    if is_combined:
        title = "All Shots Combined"
        if show_optimal:
            title += " (Centered)"
    else:
        title = f"Series {series_idx + 1} (shots {start + 1}-{end})"
        if show_optimal:
            title += " (Centered)"

    ax.set_xlim(-zoom_level, zoom_level)
    ax.set_ylim(-zoom_level, zoom_level)
    ax.set_title(title, color='white', fontsize=16, pad=15)
    ax.set_xlabel("X (mm)", color='white', fontsize=12)
    ax.set_ylabel("Y (mm)", color='white', fontsize=12)
    ax.grid(True, alpha=0.2, color='white', linestyle='--', linewidth=0.5)

    return fig, non_decimal_total, decimal_total, inner_tens, (start, end), len(sel), is_combined

# ========================
# Data utilities
# ========================
def clean_numeric_series(s):
    """Coerce to numeric, drop NaN/inf."""
    s = pd.to_numeric(s, errors='coerce')
    s = s.replace([float('inf'), -float('inf')], pd.NA).dropna()
    return s

def series_dataframe(sel, start_idx):
    """Return a DataFrame with scoring per shot for the current selection."""
    rows = []
    for j, (sx, sy) in enumerate(sel, start=start_idx + 1):
        rows.append({
            "shot_index": j,
            "x_mm": float(sx),
            "y_mm": float(sy),
            "score_decimal": float(score_shot_decimal(sx, sy)),
            "score_int": int(score_shot(sx, sy)),
            "inner_ten": int(is_inner_ten(sx, sy)),
        })
    return pd.DataFrame(rows)

# ========================
# UI
# ========================
st.title("🎯 Target Plotter - Dual Scoring + Optimal")
st.markdown("---")

# Tabs for input
tab1, tab2 = st.tabs(["📁 Excel File Upload", "⌨️ Manual Coordinate Input"])

with tab1:
    st.markdown("### Upload your shooting data Excel file")
    uploaded_file = st.file_uploader("Choose an Excel file (.xlsx or .xls)", type=['xlsx', 'xls'])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)

            # Find numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_cols) < 2:
                st.error("❌ Need at least 2 numeric columns in the Excel file")
            else:
                # Let user select columns for X, Y for robustness
                x_col = st.selectbox("Select X column", numeric_cols, index=0, key="xsel_upload")
                # Default Y to next numeric col if available
                y_default_idx = 1 if len(numeric_cols) > 1 else 0
                y_col = st.selectbox("Select Y column", numeric_cols, index=y_default_idx, key="ysel_upload")

                xs = clean_numeric_series(df[x_col]).tolist()
                ys = clean_numeric_series(df[y_col]).tolist()

                n = min(len(xs), len(ys))
                if n > 0:
                    st.session_state.coords = list(zip(xs[:n], ys[:n]))
                    st.session_state.series_index = 0
                    st.session_state.show_all_downloads = False
                    st.success(f"✅ Loaded {n} coordinate pairs from '{x_col}' and '{y_col}'")
                else:
                    st.error("❌ No valid numeric data found in the selected columns")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

with tab2:
    st.markdown("### Enter coordinates manually (one value per line)")
    col1, col2 = st.columns(2)

    with col1:
        x_input = st.text_area("X coordinates", height=250, placeholder="1.5\n-2.3\n0.8\n...")
    with col2:
        y_input = st.text_area("Y coordinates", height=250, placeholder="2.1\n-1.5\n3.2\n...")

    if st.button("📥 Load Coordinates", type="primary"):
        try:
            xs = [float(x.strip()) for x in x_input.strip().split('\n') if x.strip()]
            ys = [float(y.strip()) for y in y_input.strip().split('\n') if y.strip()]

            if len(xs) != len(ys):
                st.error(f"❌ Mismatched counts: {len(xs)} X values vs {len(ys)} Y values")
            elif len(xs) == 0:
                st.error("❌ No coordinates entered")
            else:
                st.session_state.coords = list(zip(xs, ys))
                st.session_state.series_index = 0
                st.session_state.show_all_downloads = False
                st.success(f"✅ Loaded {len(xs)} coordinate pairs")
        except ValueError:
            st.error("❌ Invalid input - please enter only numeric values")

# ========================
# Main plot and controls
# ========================
if st.session_state.coords:
    st.markdown("---")

    coords = st.session_state.coords
    num_series = (len(coords) + SERIES_SIZE - 1) // SERIES_SIZE
    total_views = num_series + 1  # includes "All Shots Combined"

    # Top summary of current state
    status_cols = st.columns([2, 1, 1, 1])
    with status_cols[0]:
        st.caption(f"Total shots: {len(coords)} | Series size: {SERIES_SIZE}")
    with status_cols[1]:
        st.caption(f"Optimal: {'On' if st.session_state.show_optimal else 'Off'}")
    with status_cols[2]:
        st.caption(f"Zoom: {st.session_state.zoom_level:.0f}")
    with status_cols[3]:
        st.caption(f"Views: {total_views}")

    # FIXED: Navigation controls - Handle button presses before slider
    # Check for button presses and update series_index BEFORE rendering the slider
    col3, col4, col5, col6 = st.columns(4)

    # with col1:
    #     if st.button("⬅️ Previous", key="prev_btn", use_container_width=True):
    #         st.session_state.series_index = (st.session_state.series_index - 1) % total_views

    # with col2:
    #     if st.button("Next ➡️", key="next_btn", use_container_width=True):
    #         st.session_state.series_index = (st.session_state.series_index + 1) % total_views

    with col3:
        if st.button("🔍 Zoom In", key="zoom_in_btn", use_container_width=True):
            st.session_state.zoom_level = max(10, st.session_state.zoom_level * 0.8)

    with col4:
        if st.button("🔍 Zoom Out", key="zoom_out_btn", use_container_width=True):
            st.session_state.zoom_level = min(50, st.session_state.zoom_level * 1.25)

    with col5:
        if st.button("🏠 Reset View", key="reset_btn", use_container_width=True):
            st.session_state.zoom_level = 25
            st.session_state.show_optimal = False

    with col6:
        btn_text = "Show Raw" if st.session_state.show_optimal else "Show Optimal"
        if st.button(btn_text, key="optimal_btn", use_container_width=True):
            st.session_state.show_optimal = not st.session_state.show_optimal

    # with col7:
    #     show_ring_labels = st.checkbox("Ring Labels", value=False)

    # Series selector slider - This will now properly reflect the button changes
    slider_options = [f"Series {i + 1}" for i in range(num_series)] + ["All Shots Combined"]
    
    # Use session state value and update it when slider changes
    current_series = st.select_slider(
        "Select View",
        options=range(total_views),
        value=st.session_state.series_index,
        format_func=lambda x: slider_options[x],
        key="series_slider"
    )
    
    # Update session state if slider changed (but don't override button changes)
    if current_series != st.session_state.series_index:
        st.session_state.series_index = current_series

    # Plot
    fig, nd_score, d_score, inner_10s, (start, end), num_shots, is_combined = plot_target(
        coords,
        st.session_state.series_index,
        st.session_state.show_optimal,
        st.session_state.zoom_level,
        # show_ring_labels
    )

    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Summary metrics
    st.markdown("### Series Statistics")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Non-Decimal Score", f"{int(nd_score)}/{num_shots*10}")
    with m2:
        st.metric("Decimal Score", f"{d_score:.1f}/{num_shots*10}")
    with m3:
        st.metric("Inner Tens", f"{inner_10s}/{num_shots}")

    # CSV export of current view
    # Build sel again similarly to plot_target to make CSV match the current selection
    if is_combined:
        sel = coords
        csv_start = 0
    else:
        csv_start = start
        sel = coords[start:end]
    df_series = series_dataframe(sel, csv_start)
    csv_bytes = df_series.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download Series CSV",
        data=csv_bytes,
        file_name=("series_scores_all.csv" if is_combined else f"series_{st.session_state.series_index + 1}_scores.csv"),
        mime="text/csv",
        use_container_width=True
    )

    # Export options
    st.markdown("---")
    st.markdown("### Download Options")

    exp1, exp2 = st.columns(2)

    with exp1:
        # Single plot download (current view)
        buf = io.BytesIO()
        # Regenerate current fig to ensure clean buffer (fig already closed)
        fig_dl, nd_dl, d_dl, x_dl, *_ = plot_target(
            coords,
            st.session_state.series_index,
            st.session_state.show_optimal,
            st.session_state.zoom_level,
            # show_ring_labels
        )
        fig_dl.savefig(buf, format='png', facecolor='black', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig_dl)

        if is_combined:
            filename = f"target_all_shots_combined_nd{int(nd_dl)}_d{d_dl:.1f}_x{x_dl}.png"
        else:
            filename = f"target_series_{st.session_state.series_index + 1}_nd{int(nd_dl)}_d{d_dl:.1f}_x{x_dl}.png"

        st.download_button(
            label="Download Current Plot",
            data=buf,
            file_name=filename,
            mime="image/png",
            use_container_width=True
        )

    with exp2:
        if st.button("Prepare All Plots for Download", use_container_width=True):
            st.session_state.show_all_downloads = True

    # Show all download buttons if requested
    if st.session_state.show_all_downloads:
        st.markdown("#### Download Individual Plots")

        num_series = (len(coords) + SERIES_SIZE - 1) // SERIES_SIZE
        total_views = num_series + 1
        cols_per_row = 3

        for i in range(0, total_views, cols_per_row):
            row_cols = st.columns(cols_per_row)

            for j in range(cols_per_row):
                idx = i + j
                if idx < total_views:
                    with row_cols[j]:
                        plot_fig, nd_i, d_i, x_i, *_ = plot_target(
                            coords,
                            idx,
                            st.session_state.show_optimal,
                            st.session_state.zoom_level,
                            # show_ring_labels
                        )

                        img_buffer = io.BytesIO()
                        plot_fig.savefig(img_buffer, format='png', facecolor='black',
                                         dpi=150, bbox_inches='tight')
                        img_buffer.seek(0)
                        plt.close(plot_fig)

                        if idx == num_series:
                            label = "All Shots"
                            filename = f"all_shots_combined_nd{int(nd_i)}_d{d_i:.1f}_x{x_i}.png"
                        else:
                            label = f"Series {idx + 1}"
                            filename = f"series_{idx + 1}_nd{int(nd_i)}_d{d_i:.1f}_x{x_i}.png"

                        st.download_button(
                            label=label,
                            data=img_buffer,
                            file_name=filename,
                            mime="image/png",
                            key=f"download_{idx}",
                            use_container_width=True
                        )

        if st.button("Hide Download Options", use_container_width=True):
            st.session_state.show_all_downloads = False

else:
    st.info("Please upload an Excel file or enter coordinates manually to begin plotting")

# Footer
st.markdown("---")
st.caption("🎯 Target Plotter v2.2 | Fixed navigation buttons, accurate scoring, stable exports")
