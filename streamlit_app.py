"""
Streamlit app — .fit Equivalent Flat Distance / Equivalent Flat Speed per zona FC
==================================================================================
Carica una o più tracce .fit e calcola, per ciascun segmento:
  - EFD (Equivalent Flat Distance), da Minetti et al. (2002)
  - tempo del segmento
  - EFS (Equivalent Flat Speed) = EFD_segmento / tempo_segmento
  - zona cardiaca prevalente del segmento

In output: velocità media EFS per ogni zona cardiaca, per ciascun file, più un
riepilogo finale aggregato su tutti i file caricati.

Include inoltre un modulo opzionale di "Lap Analysis" manuale: l'utente
definisce dei lap (in km) su un file a scelta e ottiene, per ciascun lap,
distanza, EFD, tempo e velocità equivalente pianeggiante media.

Run with:
    streamlit run streamlit_app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
from fitparse import FitFile

st.set_page_config(page_title="FIT — EFD/EFS per zona FC", layout="wide")

MIN_FILE_DURATION_S = 10 * 60  # file sotto questa durata esclusi dal riepilogo finale


# ---------------------------------------------------------------------------
# Energy cost of running as a function of slope (Minetti et al. 2002)
# ---------------------------------------------------------------------------
def cost_of_running(slope: np.ndarray) -> np.ndarray:
    """
    Mass-specific energy cost of running, J/(kg*m), as a function of slope
    (decimal fraction). Vectorized. Clamped to +/-45%.
    """
    i = np.clip(slope, -0.45, 0.45)
    return (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
            + 46.3 * i**2 + 19.5 * i + 3.6)


FLAT_COST = float(cost_of_running(np.array([0.0]))[0])  # 3.6 J/kg/m
SEMICIRCLE_TO_DEG = 180.0 / (2 ** 31)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seconds_to_hhmm(seconds) -> str:
    """Format a duration in seconds as HH:MM:SS (or MM:SS if under an hour)."""
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "-"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# .fit parsing
# ---------------------------------------------------------------------------
def parse_fit(file_obj) -> pd.DataFrame:
    """Extract lat, lon, elevation, heart rate, elapsed time (s) per record."""
    fitfile = FitFile(file_obj)
    rows = []
    for record in fitfile.get_messages("record"):
        data = {f.name: f.value for f in record}

        lat_raw = data.get("position_lat")
        lon_raw = data.get("position_long")
        if lat_raw is None or lon_raw is None:
            continue

        lat = lat_raw * SEMICIRCLE_TO_DEG
        lon = lon_raw * SEMICIRCLE_TO_DEG
        ele = data.get("enhanced_altitude", data.get("altitude"))
        hr = data.get("heart_rate")
        ts = data.get("timestamp")

        rows.append((lat, lon, ele, hr, ts))

    df = pd.DataFrame(rows, columns=["lat", "lon", "ele", "hr", "timestamp"])
    df = df.dropna(subset=["lat", "lon", "timestamp"]).reset_index(drop=True)

    if df.empty:
        return df

    df["ele"] = df["ele"].astype(float).ffill().bfill().fillna(0.0)
    df["hr"] = df["hr"].astype(float)  # may contain NaN, handled later
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in meters between arrays of points."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def hr_to_zone(hr, z1, z2, z3, z4, z5):
    if np.isnan(hr):
        return None
    if hr <= z1:
        return "Z1"
    elif hr <= z2:
        return "Z2"
    elif hr <= z3:
        return "Z3"
    elif hr <= z4:
        return "Z4"
    else:
        return "Z5"


# ---------------------------------------------------------------------------
# Core pipeline: smooth -> resample onto uniform distance grid -> cost -> EFS/zone
# ---------------------------------------------------------------------------
def process_track(df: pd.DataFrame, smooth_window: int, resample_step_m: float,
                   zones: tuple):
    lat, lon, ele = df["lat"].to_numpy(), df["lon"].to_numpy(), df["ele"].to_numpy()
    time_s = df["elapsed_s"].to_numpy()
    hr_raw = df["hr"].to_numpy()

    # Smooth elevation (rolling mean)
    ele_smooth = (pd.Series(ele)
                  .rolling(window=smooth_window, center=True, min_periods=1)
                  .mean()
                  .to_numpy())

    # Cumulative horizontal distance along the raw trace
    seg_dist = haversine_vec(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_dist)])
    total_dist = cum_dist[-1]

    if total_dist <= 0:
        return None, None

    # Resample onto a uniform horizontal-distance grid
    n_steps = max(2, int(total_dist // resample_step_m))
    grid = np.linspace(0.0, total_dist, n_steps)
    ele_grid = np.interp(grid, cum_dist, ele_smooth)
    time_grid = np.interp(grid, cum_dist, time_s)

    # HR: interpolate ignoring NaNs (fill gaps first so np.interp has valid data)
    hr_series = pd.Series(hr_raw).interpolate(limit_direction="both").to_numpy()
    hr_grid = np.interp(grid, cum_dist, hr_series)

    dx = np.diff(grid)
    dz = np.diff(ele_grid)
    dt = np.diff(time_grid)
    dist3d = np.hypot(dx, dz)
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=dx > 0)

    cost = cost_of_running(slope)          # J/(kg*m)
    energy = cost * dist3d                 # J/kg per segment
    efd_m = energy / FLAT_COST             # equivalent flat meters per segment
    efs_ms = np.divide(efd_m, dt, out=np.full_like(efd_m, np.nan), where=dt > 0)

    hr_seg = (hr_grid[:-1] + hr_grid[1:]) / 2.0   # avg HR across the segment
    zone = np.array([hr_to_zone(h, *zones) for h in hr_seg])

    d_plus = float(dz[dz > 0].sum())
    d_minus = float(-dz[dz < 0].sum())
    total_energy = float(energy.sum())
    total_efd_m = float(efd_m.sum())
    total_time_s = float(dt.sum())

    segments = pd.DataFrame({
        "distance_km": grid[1:] / 1000,
        "elevation_m": ele_grid[1:],
        "slope_pct": slope * 100,
        "dt_s": dt,
        "efd_m": efd_m,
        "efs_ms": efs_ms,
        "hr_avg": hr_seg,
        "zone": zone,
    })

    summary = {
        "horizontal_distance_m": total_dist,
        "d_plus_m": d_plus,
        "d_minus_m": d_minus,
        "total_energy_j_per_kg": total_energy,
        "efd_m": total_efd_m,
        "total_time_s": total_time_s,
    }
    return segments, summary


def zone_efs_table(segments: pd.DataFrame) -> pd.DataFrame:
    """Average EFS per HR zone = total EFD in zone / total time in zone (time-weighted)."""
    valid = segments.dropna(subset=["zone"])
    rows = []
    for z in ["Z1", "Z2", "Z3", "Z4", "Z5"]:
        zdf = valid[valid["zone"] == z]
        if zdf.empty:
            continue
        total_efd = zdf["efd_m"].sum()
        total_time = zdf["dt_s"].sum()
        if total_time <= 0:
            continue
        efs_ms = total_efd / total_time
        rows.append({
            "Zona": z,
            "Tempo (min)": total_time / 60,
            "EFD (km)": total_efd / 1000,
            "EFS media (km/h)": efs_ms * 3.6,
        })
    return pd.DataFrame(rows)


def compute_lap_result(segments: pd.DataFrame, name: str, start_km: float, end_km: float,
                        max_distance_km: float, overall_efs_kmh: float) -> dict | None:
    """
    Given a segments dataframe (output of process_track) and a [start_km, end_km)
    range, aggregate EFD and time in that range and return a result row, or None
    if no data falls in the range.

    Also computes:
      - % avanzamento: end_km / max_distance_km * 100 (percentuale di avanzamento
        del file raggiunta alla fine del lap)
      - scostamento EFS: (EFS lap - EFS media totale del file) / EFS media totale * 100
    """
    mask = (segments["distance_km"] > start_km) & (segments["distance_km"] <= end_km)
    lap_segments = segments.loc[mask]
    if lap_segments.empty:
        return None

    lap_time_s = float(lap_segments["dt_s"].sum())
    lap_efd_km = float(lap_segments["efd_m"].sum()) / 1000
    lap_efs_kmh = (lap_efd_km / (lap_time_s / 3600)) if lap_time_s > 0 else np.nan

    progress_pct = (end_km / max_distance_km * 100) if max_distance_km > 0 else np.nan

    if not np.isnan(lap_efs_kmh) and overall_efs_kmh and overall_efs_kmh > 0:
        efs_deviation_pct = (lap_efs_kmh - overall_efs_kmh) / overall_efs_kmh * 100
    else:
        efs_deviation_pct = np.nan

    return {
        "Lap": name,
        "Distanza (km)": round(end_km - start_km, 2),
        "% Avanzamento": round(progress_pct, 1) if not np.isnan(progress_pct) else None,
        "EFD (km)": round(lap_efd_km, 2),
        "Tempo": seconds_to_hhmm(lap_time_s),
        "Tempo (s)": round(lap_time_s, 1),
        "EFS media (km/h)": round(lap_efs_kmh, 2) if not np.isnan(lap_efs_kmh) else None,
        "Scostamento EFS (%)": round(efs_deviation_pct, 1) if not np.isnan(efs_deviation_pct) else None,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏃‍♂️ FIT — Equivalent Flat Speed per zona cardiaca")
st.caption(
    "Analisi bioenergetica basata su Minetti et al. (2002): per ogni segmento calcola "
    "distanza equivalente pianeggiante (EFD), tempo, velocità equivalente pianeggiante "
    "(EFS) e zona cardiaca prevalente."
)

# --- HR Zones module ---
default_zones = {'z1': 140, 'z2': 160, 'z3': 170, 'z4': 180, 'z5': 200}
for zone, val in default_zones.items():
    if zone not in st.session_state:
        st.session_state[zone] = val

st.subheader("❤️ Zone cardiache dell'atleta")

input_method = st.radio("Metodo di input:", ["Manuale", "Importa CSV"], horizontal=True)

if input_method == "Manuale":
    with st.form("hr_zones_form"):
        st.caption("Inserisci il **limite superiore (in bpm)** di ciascuna zona, poi salva:")
        col1, col2, col3, col4, col5 = st.columns(5)
        z1_in = col1.number_input("Zona 1 fino a:", min_value=60, value=st.session_state['z1'])
        z2_in = col2.number_input("Zona 2 fino a:", min_value=60, value=st.session_state['z2'])
        z3_in = col3.number_input("Zona 3 fino a:", min_value=60, value=st.session_state['z3'])
        z4_in = col4.number_input("Zona 4 fino a:", min_value=60, value=st.session_state['z4'])
        z5_in = col5.number_input("Zona 5 fino a:", min_value=60, value=st.session_state['z5'])
        submitted = st.form_submit_button("💾 Salva zone")

    if submitted:
        if not (z1_in < z2_in < z3_in < z4_in < z5_in):
            st.error("⚠️ Le soglie FC non sono coerenti (devono essere crescenti). Zone NON salvate.")
        else:
            st.session_state.update({'z1': z1_in, 'z2': z2_in, 'z3': z3_in, 'z4': z4_in, 'z5': z5_in})
            st.success("✅ Zone salvate!")
else:
    uploaded_hr_csv = st.file_uploader("Carica CSV zone FC:", type=["csv"], key="hr_zones_csv")
    if uploaded_hr_csv is not None:
        hr_df = pd.read_csv(uploaded_hr_csv)
        required_cols = ['z1', 'z2', 'z3', 'z4', 'z5']
        if all(col in hr_df.columns for col in required_cols):
            z1c, z2c, z3c, z4c, z5c = hr_df.loc[0, required_cols]
            if not (z1c < z2c < z3c < z4c < z5c):
                st.error("⚠️ Le soglie FC nel CSV non sono coerenti (devono essere crescenti). Zone NON importate.")
            else:
                st.session_state.update({col: hr_df.loc[0, col] for col in required_cols})
                athlete = hr_df.loc[0, 'athlete_name'] if 'athlete_name' in hr_df.columns else 'atleta'
                st.success(f"✅ Zone importate correttamente per {athlete}")
        else:
            st.error("⚠️ Il CSV deve contenere le colonne: z1, z2, z3, z4, z5")

# Le zone effettivamente usate nell'elaborazione sono sempre quelle salvate in session_state
z1, z2, z3, z4, z5 = (st.session_state[c] for c in ["z1", "z2", "z3", "z4", "z5"])
zones = (z1, z2, z3, z4, z5)

with st.expander("📋 Zone attualmente in uso"):
    st.write(f"""
    - 🩵 Zona 1 (Aerobica bassa): ≤ {z1} bpm
    - 💚 Zona 2 (Aerobica alta): {z1+1} - {z2} bpm
    - 💛 Zona 3 (Resistenza aerobica): {z2+1} - {z3} bpm
    - 🧡 Zona 4 (Sub soglia): {z3+1} - {z4} bpm
    - ❤️ Zona 5 (Sopra soglia): > {z4} bpm
    """)

    athlete_name = st.session_state.get('athlete_name', 'atleta')
    export_df = pd.DataFrame([{
        'athlete_name': athlete_name, 'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5
    }])
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Esporta zone in CSV",
        data=csv_data,
        file_name=f"{str(athlete_name).replace(' ', '_')}_HR_Zones.csv",
        mime='text/csv'
    )

st.divider()

# --- File upload & processing settings ---
with st.sidebar:
    st.header("Impostazioni")
    uploaded_files = st.file_uploader(
        "Carica uno o più file .fit", type=["fit"], accept_multiple_files=True
    )
    smooth_window = st.slider("Finestra di smoothing quota (punti)", 1, 31, 9, step=2)
    resample_step = st.slider("Passo di ricampionamento (m)", 5, 100, 20, step=5)
    st.caption(
        "Lo smoothing ripulisce la quota grezza GPS/barometrica prima di calcolare la pendenza. "
        "Il passo di ricampionamento controlla la risoluzione orizzontale usata per integrare l'energia."
    )

if not uploaded_files:
    st.info("Carica uno o più file .fit per iniziare.")
    st.stop()

per_file_zone_tables = []  # list of (filename, zone_table, total_time_s)
per_file_segments = {}     # filename -> segments dataframe (needed for Lap Analysis below)

for uploaded in uploaded_files:
    st.header(f"📄 {uploaded.name}")

    df_points = parse_fit(io.BytesIO(uploaded.getvalue()))
    if len(df_points) < 2:
        st.error("Traccia non valida: mancano punti GPS/timestamp validi in questo file.")
        continue
    if df_points["hr"].isna().all():
        st.warning("⚠️ Nessun dato di frequenza cardiaca trovato in questo file: impossibile assegnare le zone.")
        continue

    segments, summary = process_track(df_points, smooth_window, resample_step, zones)
    if segments is None:
        st.error("Impossibile calcolare la distanza percorsa (traccia degenere).")
        continue

    per_file_segments[uploaded.name] = segments

    c1, c2, c3 = st.columns(3)
    c1.metric("Distanza orizzontale", f"{summary['horizontal_distance_m']/1000:.2f} km")
    c2.metric("D+ / D−", f"{summary['d_plus_m']:.0f} m / {summary['d_minus_m']:.0f} m")
    c3.metric("EFD totale", f"{summary['efd_m']/1000:.2f} km")

    st.subheader("EFS media per zona cardiaca")
    zone_table = zone_efs_table(segments)
    if zone_table.empty:
        st.warning("Nessun segmento assegnabile a una zona (dati FC insufficienti).")
    else:
        display_table = zone_table.copy()
        display_table["Tempo (min)"] = display_table["Tempo (min)"].round(0).astype(int)
        st.dataframe(
            display_table.style.format({
                "EFD (km)": "{:.2f}",
                "EFS media (km/h)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            label="📥 Scarica tabella zone (CSV)",
            data=zone_table.to_csv(index=False).encode('utf-8'),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_zone_efs.csv",
            mime='text/csv',
            key=f"dl_{uploaded.name}",
        )
        per_file_zone_tables.append((uploaded.name, zone_table, summary["total_time_s"]))

    with st.expander("Dettaglio segmenti (debug)"):
        st.dataframe(segments, use_container_width=True)

    st.divider()

# ---------------------------------------------------------------------------
# Riepilogo finale aggregato su tutti i file
# ---------------------------------------------------------------------------
included = [(name, tbl) for name, tbl, total_t in per_file_zone_tables if total_t >= MIN_FILE_DURATION_S]
excluded = [name for name, tbl, total_t in per_file_zone_tables if total_t < MIN_FILE_DURATION_S]

if included:
    st.header("📊 Riepilogo finale (media tra i file)")
    if excluded:
        st.caption(
            f"File esclusi dalla media perché più corti di {MIN_FILE_DURATION_S // 60} minuti: "
            + ", ".join(excluded)
        )

    all_tables = pd.concat(
        [tbl.assign(file=name) for name, tbl in included], ignore_index=True
    )

    summary_rows = []
    for z in ["Z1", "Z2", "Z3", "Z4", "Z5"]:
        zdf = all_tables[all_tables["Zona"] == z]
        if zdf.empty:
            continue
        total_time_min = zdf["Tempo (min)"].sum()
        total_efd_km = zdf["EFD (km)"].sum()
        # media pesata: EFS = EFD totale / tempo totale (non media delle medie)
        efs_weighted = (total_efd_km / (total_time_min / 60)) if total_time_min > 0 else np.nan
        summary_rows.append({
            "Zona": z,
            "Tempo totale (min)": int(round(total_time_min)),
            "EFD totale (km)": total_efd_km,
            "EFS media (km/h)": efs_weighted,
            "N. file": len(zdf),
        })

    final_summary = pd.DataFrame(summary_rows)
    st.dataframe(
        final_summary.style.format({
            "EFD totale (km)": "{:.2f}",
            "EFS media (km/h)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        label="📥 Scarica riepilogo finale (CSV)",
        data=final_summary.to_csv(index=False).encode('utf-8'),
        file_name="riepilogo_finale_zone_efs.csv",
        mime='text/csv',
    )
elif per_file_zone_tables:
    st.info(f"Tutti i file caricati sono più corti di {MIN_FILE_DURATION_S // 60} minuti: nessun riepilogo finale calcolato.")


# ===========================================================
# 🏁 LAP ANALYSIS BLOCK (manuale, basato su EFD/EFS)
# ===========================================================
st.divider()
st.header("🏁 Lap Analysis")

do_lap = st.checkbox("Attiva Lap Analysis", value=False, key="do_lap_analysis")

if do_lap:
    if not per_file_segments:
        st.warning("Nessun file elaborato correttamente: la Lap Analysis non è disponibile.")
    else:
        file_names = list(per_file_segments.keys())
        target_file = st.selectbox(
            "Seleziona il file su cui eseguire la Lap Analysis:",
            file_names,
            key="lap_target_file",
        )
        segments = per_file_segments[target_file]
        max_distance = round(float(segments["distance_km"].max()), 2)

        st.caption(f"Distanza totale del file selezionato: **{max_distance} km**")

        # --- CSV import (start_km / end_km) ---
        st.markdown("#### 📥 Importa Lap da CSV")
        imported_lap_csv = st.file_uploader(
            "Importa un CSV di lap esportato in precedenza (colonne: name, start_km, end_km):",
            type=["csv"],
            key="import_lap_csv",
        )
        if imported_lap_csv is not None:
            try:
                imported_df = pd.read_csv(imported_lap_csv)
                required_import_cols = {"name", "start_km", "end_km"}
                if not required_import_cols.issubset(set(imported_df.columns)):
                    st.error("⚠️ Il CSV deve contenere le colonne: name, start_km, end_km")
                else:
                    imported_df = imported_df[["name", "start_km", "end_km"]].copy()
                    imported_df["name"] = imported_df["name"].fillna("").astype(str)
                    st.session_state["manual_lap_table"] = imported_df.reset_index(drop=True)
                    st.success(f"✅ Importati {len(imported_df)} lap dal CSV!")
            except Exception as e:
                st.error(f"⚠️ Errore durante la lettura del CSV: {e}")

        # --- Editable lap table (start_km / end_km only, no chart) ---
        table_key = "manual_lap_table"
        if table_key not in st.session_state:
            st.session_state[table_key] = pd.DataFrame({
                "name": pd.Series([], dtype="str"),
                "start_km": pd.Series([], dtype="float"),
                "end_km": pd.Series([], dtype="float"),
            })

        st.subheader("Aggiungi / Modifica Lap")
        st.info(f"Inserisci la distanza di inizio e fine (in km) per ciascun lap (max: {max_distance} km)")

        with st.form("manual_lap_form"):
            edited = st.data_editor(
                st.session_state[table_key],
                num_rows="dynamic",
                key="manual_lap_editor",
                column_config={
                    "name": st.column_config.TextColumn("Nome Lap"),
                    "start_km": st.column_config.NumberColumn(
                        "Inizio (km)", min_value=0.0, max_value=float(max_distance), step=0.1, format="%.2f"
                    ),
                    "end_km": st.column_config.NumberColumn(
                        "Fine (km)", min_value=0.0, max_value=float(max_distance), step=0.1, format="%.2f"
                    ),
                },
            )
            submit_laps = st.form_submit_button("💾 Salva e calcola Lap")

        if submit_laps:
            edited = edited.reset_index(drop=True)

            # Auto-name any unnamed rows
            edited["name"] = [
                row["name"] if pd.notna(row.get("name")) and str(row.get("name", "")).strip() != ""
                else f"Lap {i+1}"
                for i, row in edited.iterrows()
            ]

            # Validation
            problems = []
            for i, row in edited.iterrows():
                if pd.isna(row.get("start_km")):
                    problems.append(f"Riga {i+1} ({row['name']}): manca l'inizio (km)")
                if pd.isna(row.get("end_km")):
                    problems.append(f"Riga {i+1} ({row['name']}): manca la fine (km)")
                elif not pd.isna(row.get("start_km")) and float(row["end_km"]) <= float(row["start_km"]):
                    problems.append(f"Riga {i+1} ({row['name']}): la fine deve essere maggiore dell'inizio")
                elif not pd.isna(row.get("end_km")) and float(row["end_km"]) > max_distance:
                    problems.append(
                        f"Riga {i+1} ({row['name']}): la fine ({row['end_km']}) supera la distanza del file ({max_distance} km)"
                    )

            if problems:
                for msg in problems:
                    st.error(f"⚠️ {msg}")
                st.stop()

            st.session_state[table_key] = edited

            # EFS media dell'intero file selezionato, usata come riferimento per lo scostamento
            total_efd_km_file = float(segments["efd_m"].sum()) / 1000
            total_time_s_file = float(segments["dt_s"].sum())
            overall_efs_kmh = (total_efd_km_file / (total_time_s_file / 3600)) if total_time_s_file > 0 else np.nan

            lap_results = []
            for i, row in edited.iterrows():
                res = compute_lap_result(
                    segments, row["name"], float(row["start_km"]), float(row["end_km"]),
                    max_distance_km=max_distance, overall_efs_kmh=overall_efs_kmh,
                )
                if res is None:
                    st.warning(f"⚠️ Nessun dato trovato per il lap '{row['name']}' nell'intervallo richiesto.")
                    continue
                lap_results.append(res)

            if lap_results:
                results_df = pd.DataFrame(lap_results).drop(columns=["Tempo (s)"])
                st.session_state["lap_results_df"] = results_df
                st.session_state["lap_export_df"] = edited[["name", "start_km", "end_km"]].copy()
                st.session_state["lap_results_file"] = target_file
                st.session_state["lap_overall_efs"] = overall_efs_kmh if not np.isnan(overall_efs_kmh) else None
                st.success(f"✅ {len(lap_results)} lap calcolati!")
            else:
                st.session_state["lap_results_df"] = None
                st.error("⚠️ Nessun lap valido calcolato. Controlla i valori di distanza inseriti.")

        # --- Results table + export ---
        if st.session_state.get("lap_results_df") is not None:
            st.subheader("📈 Risultati Lap")
            if st.session_state.get("lap_overall_efs") is not None:
                st.caption(f"EFS media dell'intero file (riferimento per lo scostamento): "
                           f"**{st.session_state['lap_overall_efs']:.2f} km/h**")
            results_df = st.session_state["lap_results_df"]
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            col_a, col_b = st.columns(2)
            with col_a:
                results_csv = results_df.to_csv(index=False).encode("utf-8")
                safe_name = st.session_state.get("lap_results_file", target_file).rsplit(".", 1)[0]
                st.download_button(
                    label="📥 Scarica risultati Lap (CSV)",
                    data=results_csv,
                    file_name=f"{safe_name}_lap_results.csv",
                    mime="text/csv",
                    key="download_lap_results_csv",
                )
            with col_b:
                if "lap_export_df" in st.session_state:
                    defs_csv = st.session_state["lap_export_df"].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Scarica definizione Lap (CSV)",
                        data=defs_csv,
                        file_name=f"{safe_name}_lap_definitions.csv",
                        mime="text/csv",
                        key="download_lap_defs_csv",
                    )