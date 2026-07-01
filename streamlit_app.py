"""
Streamlit app — GPX Equivalent Flat Distance (EFD)
====================================================
Upload a GPX trace and compute the bioenergetic Equivalent Flat Distance,
based on Minetti et al. (2002)'s slope-dependent energy cost of running,
following the logic in Veronique Billat's article on why
"1000 m ascent = 10 km flat" is a poor approximation.

Run with:
    streamlit run streamlit_app.py
"""

import io
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="GPX Equivalent Flat Distance", layout="wide")


# ---------------------------------------------------------------------------
# Energy cost of running as a function of slope (Minetti et al. 2002)
# ---------------------------------------------------------------------------
def cost_of_running(slope: np.ndarray) -> np.ndarray:
    """
    Mass-specific energy cost of running, J/(kg*m), as a function of slope
    (decimal fraction). Vectorized. Clamped to +/-45% since the polynomial
    fit isn't empirically supported beyond that range.
    """
    i = np.clip(slope, -0.45, 0.45)
    return (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
            + 46.3 * i**2 + 19.5 * i + 3.6)


FLAT_COST = float(cost_of_running(np.array([0.0]))[0])  # 3.6 J/kg/m


# ---------------------------------------------------------------------------
# GPX parsing
# ---------------------------------------------------------------------------
def parse_gpx(file_obj) -> pd.DataFrame:
    tree = ET.parse(file_obj)
    root = tree.getroot()

    ns_uri = root.tag[1:root.tag.index("}")] if root.tag.startswith("{") else ""
    ns = {"gpx": ns_uri} if ns_uri else {}

    def findall(tag):
        return root.findall(f".//gpx:{tag}", ns) if ns else root.findall(f".//{tag}")

    def find_child(el, tag):
        return el.find(f"gpx:{tag}", ns) if ns else el.find(tag)

    rows = []
    for trkpt in findall("trkpt"):
        lat = float(trkpt.get("lat"))
        lon = float(trkpt.get("lon"))
        ele_el = find_child(trkpt, "ele")
        ele = float(ele_el.text) if ele_el is not None and ele_el.text else 0.0
        rows.append((lat, lon, ele))

    if not rows:
        for tag in ("rtept", "wpt"):
            for pt in findall(tag):
                lat = float(pt.get("lat"))
                lon = float(pt.get("lon"))
                ele_el = find_child(pt, "ele")
                ele = float(ele_el.text) if ele_el is not None and ele_el.text else 0.0
                rows.append((lat, lon, ele))

    return pd.DataFrame(rows, columns=["lat", "lon", "ele"])


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in meters between arrays of points."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Core pipeline: smooth -> resample onto uniform distance grid -> cost
# ---------------------------------------------------------------------------
def process_track(df: pd.DataFrame, smooth_window: int, resample_step_m: float):
    lat, lon, ele = df["lat"].to_numpy(), df["lon"].to_numpy(), df["ele"].to_numpy()

    # Smooth elevation (rolling mean, vectorized via pandas)
    ele_smooth = (pd.Series(ele)
                  .rolling(window=smooth_window, center=True, min_periods=1)
                  .mean()
                  .to_numpy())

    # Cumulative distance along the raw trace (vectorized haversine)
    seg_dist = haversine_vec(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_dist)])
    total_dist = cum_dist[-1]

    # Resample onto a uniform horizontal-distance grid via interpolation
    n_steps = max(2, int(total_dist // resample_step_m))
    grid = np.linspace(0.0, total_dist, n_steps)
    ele_grid = np.interp(grid, cum_dist, ele_smooth)
    lat_grid = np.interp(grid, cum_dist, lat)
    lon_grid = np.interp(grid, cum_dist, lon)

    dx = np.diff(grid)                 # horizontal distance per segment (uniform)
    dz = np.diff(ele_grid)             # vertical change per segment
    dist3d = np.hypot(dx, dz)
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=dx > 0)

    cost = cost_of_running(slope)      # J/(kg*m)
    energy = cost * dist3d             # J/kg per segment

    d_plus = float(dz[dz > 0].sum())
    d_minus = float(-dz[dz < 0].sum())
    total_energy = float(energy.sum())
    efd_m = total_energy / FLAT_COST
    naive_efd_m = total_dist + d_plus * 10.0

    segments = pd.DataFrame({
        "distance_m": grid[1:],
        "distance_km": grid[1:] / 1000,
        "elevation_m": ele_grid[1:],
        "lat": lat_grid[1:],
        "lon": lon_grid[1:],
        "slope_pct": slope * 100,
        "cost_j_per_kg_m": cost,
        "energy_j_per_kg": energy,
        "cumulative_efd_km": np.cumsum(energy) / FLAT_COST / 1000,
    })

    summary = {
        "horizontal_distance_m": total_dist,
        "d_plus_m": d_plus,
        "d_minus_m": d_minus,
        "total_energy_j_per_kg": total_energy,
        "efd_m": efd_m,
        "naive_efd_m": naive_efd_m,
    }
    return segments, summary


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏔️ GPX Equivalent Flat Distance")
st.caption(
    "Bioenergetic alternative to the '1000 m D+ = 10 km flat' rule, "
    "using Minetti et al. (2002)'s slope-dependent cost of running."
)

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload a GPX file", type=["gpx"])
    smooth_window = st.slider("Elevation smoothing window (points)", 1, 31, 9, step=2)
    resample_step = st.slider("Resample step (m)", 5, 100, 20, step=5)
    st.caption(
        "Smoothing denoises raw GPS/barometric elevation before slope is computed. "
        "Resample step controls the horizontal resolution used to integrate energy."
    )

if uploaded is None:
    st.info("Upload a GPX file to begin.")
    st.stop()

df_points = parse_gpx(io.BytesIO(uploaded.getvalue()))
if len(df_points) < 2:
    st.error("Not enough track points found in this GPX file.")
    st.stop()

segments, summary = process_track(df_points, smooth_window, resample_step)
overestimation_pct = 100 * (summary["naive_efd_m"] - summary["efd_m"]) / summary["efd_m"]

# --- Metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Horizontal distance", f"{summary['horizontal_distance_m']/1000:.2f} km")
c2.metric("D+ / D−", f"{summary['d_plus_m']:.0f} m / {summary['d_minus_m']:.0f} m")
c3.metric("Equivalent Flat Distance", f"{summary['efd_m']/1000:.2f} km")
c4.metric(
    "Naive '1000m=10km' estimate",
    f"{summary['naive_efd_m']/1000:.2f} km",
    delta=f"{overestimation_pct:+.1f}% vs EFD",
    delta_color="inverse",
)

st.divider()

# --- Elevation profile + cost intensity ---
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
    vertical_spacing=0.06,
    subplot_titles=("Elevation profile", "Energy cost per meter along the route"),
)
fig.add_trace(
    go.Scatter(
        x=segments["distance_km"], y=segments["elevation_m"],
        mode="lines", fill="tozeroy", name="Elevation",
        line=dict(color="#6b7280"),
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(
        x=segments["distance_km"], y=segments["cost_j_per_kg_m"],
        mode="lines", name="Cost (J/kg/m)",
        line=dict(color="#dc2626"),
    ),
    row=2, col=1,
)
fig.add_hline(y=FLAT_COST, line_dash="dot", line_color="gray",
              annotation_text="flat cost (3.6)", row=2, col=1)
fig.update_layout(height=550, showlegend=False, margin=dict(t=40, b=10))
fig.update_xaxes(title_text="Distance (km)", row=2, col=1)
fig.update_yaxes(title_text="Elevation (m)", row=1, col=1)
fig.update_yaxes(title_text="J/kg/m", row=2, col=1)
st.plotly_chart(fig, use_container_width=True)

# --- Cumulative EFD vs naive linear distance ---
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=segments["distance_km"], y=segments["cumulative_efd_km"],
    mode="lines", name="Cumulative EFD", line=dict(color="#2563eb"),
))
fig2.add_trace(go.Scatter(
    x=segments["distance_km"], y=segments["distance_km"],
    mode="lines", name="Raw horizontal distance", line=dict(color="#9ca3af", dash="dash"),
))
fig2.update_layout(
    title="Cumulative Equivalent Flat Distance vs. raw distance covered",
    xaxis_title="Actual distance (km)", yaxis_title="Equivalent flat km",
    height=350, margin=dict(t=40, b=10),
)
st.plotly_chart(fig2, use_container_width=True)

# --- Route map colored by cost ---
st.subheader("Route colored by energy cost")
fig3 = go.Figure(go.Scattermapbox(
    lat=segments["lat"], lon=segments["lon"],
    mode="markers+lines",
    marker=dict(size=5, color=segments["cost_j_per_kg_m"], colorscale="RdYlGn_r",
                colorbar=dict(title="J/kg/m")),
    line=dict(width=2, color="#3b82f6"),
))
fig3.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=segments["lat"].mean(), lon=segments["lon"].mean()),
        zoom=10,
    ),
    height=500, margin=dict(t=0, b=0, l=0, r=0),
)
st.plotly_chart(fig3, use_container_width=True)

# --- Data export ---
st.divider()
csv_bytes = segments.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download segment data (CSV)",
    data=csv_bytes,
    file_name=f"{uploaded.name.rsplit('.', 1)[0]}_efd_segments.csv",
    mime="text/csv",
)

with st.expander("Segment data preview"):
    st.dataframe(segments.head(50), use_container_width=True)