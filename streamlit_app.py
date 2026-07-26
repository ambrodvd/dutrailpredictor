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

Include infine un modulo "Race Plan": definiti dei lap sul percorso gara
(km reali), calcola per ciascuno il tempo di passaggio previsto tenendo
conto del decadimento dell'EFS in funzione dell'EFD accumulato (equazione
cubica calibrata su dati storici), calibrando i tempi lap-per-lap in modo
che il totale torni esatto rispetto alla previsione di sezione 4.

Run with:
    streamlit run streamlit_app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
from fitparse import FitFile
import xml.etree.ElementTree as ET

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

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏃‍♂️ DU Trail race predictor")
st.caption(
    "La previsione di gara si basa sull'analisi degli allenamenti e delle gare di avvicinamento effettuate dall'atleta." \
    "L'analisi è basata sullo studio bioenergentico di Minetti et al. (2002): per ogni segmento calcola "
    "distanza equivalente pianeggiante (EFD), tempo, velocità equivalente pianeggiante "
    "(EFS) e zona cardiaca prevalente."
)

# --- HR Zones module ---
default_zones = {'z1': 140, 'z2': 160, 'z3': 170, 'z4': 180, 'z5': 200}
for zone, val in default_zones.items():
    if zone not in st.session_state:
        st.session_state[zone] = val

st.subheader("Per l'analisi degli allenamenti carica i file .fit a sinistra e inserisci le zone cardiache dell'atleta")

st.markdown(':red[Imposta le zone cardiache per una corretta analisi degli allenamenti:]')

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
    st.session_state['final_summary_df'] = final_summary
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

# ---------------------------------------------------------------------------
# GPX parsing (nessuna dipendenza esterna, solo xml.etree della stdlib)
# ---------------------------------------------------------------------------
def parse_gpx(file_obj) -> pd.DataFrame:
    """Estrae lat, lon, elevazione da un file GPX (percorso gara, no HR/tempo)."""
    tree = ET.parse(file_obj)
    root = tree.getroot()
    ns_uri = root.tag.split('}')[0].strip('{') if root.tag.startswith('{') else ''
 
    def tag(name):
        return f'{{{ns_uri}}}{name}' if ns_uri else name
 
    rows = []
    for trkpt in root.iter(tag('trkpt')):
        lat_attr = trkpt.get('lat')
        lon_attr = trkpt.get('lon')
        if lat_attr is None or lon_attr is None:
            continue
        ele_el = trkpt.find(tag('ele'))
        ele = float(ele_el.text) if ele_el is not None and ele_el.text else np.nan
        rows.append((float(lat_attr), float(lon_attr), ele))
 
    df = pd.DataFrame(rows, columns=["lat", "lon", "ele"])
    if df.empty:
        return df
    df["ele"] = df["ele"].ffill().bfill().fillna(0.0)
    return df
 
 
# ---------------------------------------------------------------------------
# Calcolo EFD totale di un percorso (nessuna HR/zona richiesta)
# ---------------------------------------------------------------------------
def compute_race_efd(df: pd.DataFrame, smooth_window: int, resample_step_m: float):
    """
    Calcola EFD totale (m), distanza orizzontale, D+/D- per un percorso gara.
    Ritorna (summary, profile) dove profile è un DataFrame con la mappa
    km percorso (distanza orizzontale cumulata, "km" reali) -> EFD cumulato
    (in km), usata poi dal Race Plan (sezione 5) per convertire un punto
    del percorso nel corrispondente EFD accumulato.
    """
    lat, lon, ele = df["lat"].to_numpy(), df["lon"].to_numpy(), df["ele"].to_numpy()
 
    ele_smooth = (pd.Series(ele)
                  .rolling(window=smooth_window, center=True, min_periods=1)
                  .mean()
                  .to_numpy())
 
    seg_dist = haversine_vec(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_dist)])
    total_dist = cum_dist[-1]
    if total_dist <= 0:
        return None, None
 
    n_steps = max(2, int(total_dist // resample_step_m))
    grid = np.linspace(0.0, total_dist, n_steps)
    ele_grid = np.interp(grid, cum_dist, ele_smooth)
 
    dx = np.diff(grid)
    dz = np.diff(ele_grid)
    dist3d = np.hypot(dx, dz)
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=dx > 0)
 
    cost = cost_of_running(slope)
    energy = cost * dist3d
    efd_m = energy / FLAT_COST
 
    d_plus = float(dz[dz > 0].sum())
    d_minus = float(-dz[dz < 0].sum())

    cum_efd_m = np.concatenate([[0.0], np.cumsum(efd_m)])
    profile = pd.DataFrame({
        "km": grid / 1000,
        "cum_efd_km": cum_efd_m / 1000,
    })
 
    summary = {
        "horizontal_distance_m": total_dist,
        "d_plus_m": d_plus,
        "d_minus_m": d_minus,
        "efd_m": float(efd_m.sum()),
    }
    return summary, profile
 
 
# ===========================================================
# UI — Previsione Tempo Gara
# ===========================================================
st.divider()
st.header("🎯 Previsione Tempo Gara")

# --- 1. Percorso gara --------------------------------------------------
st.subheader("1️⃣ Percorso gara (GPX o FIT)")
race_files = st.file_uploader(
    "Carica uno o più file del percorso gara (.gpx o .fit)",
    type=["gpx", "fit"],
    accept_multiple_files=True,
    key="race_course_files",
)

if race_files:
    race_names = [f.name for f in race_files]
    selected_race_name = st.selectbox(
        "Seleziona il file gara da usare per la previsione:",
        race_names,
        key="selected_race_file",
    )
    selected_race_file = next(f for f in race_files if f.name == selected_race_name)
    ext = selected_race_file.name.rsplit(".", 1)[-1].lower()

    try:
        if ext == "gpx":
            df_race = parse_gpx(io.BytesIO(selected_race_file.getvalue()))
        else:
            df_race_full = parse_fit(io.BytesIO(selected_race_file.getvalue()))
            df_race = df_race_full[["lat", "lon", "ele"]] if not df_race_full.empty else df_race_full

        if df_race is None or len(df_race) < 2:
            st.error("⚠️ Traccia gara non valida: mancano punti GPS sufficienti.")
        else:
            race_summary, race_profile = compute_race_efd(df_race, smooth_window, resample_step)
            if race_summary is None:
                st.error("⚠️ Impossibile calcolare l'EFD per questo percorso (traccia degenere).")
            else:
                st.session_state["race_efd_m"] = race_summary["efd_m"]
                st.session_state["race_profile"] = race_profile
                st.session_state["race_total_km"] = race_summary["horizontal_distance_m"] / 1000
                c1, c2, c3 = st.columns(3)
                c1.metric("Distanza orizzontale", f"{race_summary['horizontal_distance_m']/1000:.2f} km")
                c2.metric("D+ / D−", f"{race_summary['d_plus_m']:.0f} m / {race_summary['d_minus_m']:.0f} m")
                c3.metric("EFD totale gara", f"{race_summary['efd_m']/1000:.2f} km")
    except Exception as e:
        st.error(f"⚠️ Errore durante la lettura del file gara: {e}")

race_efd_m = st.session_state.get("race_efd_m")

st.divider()

# --- 2. Dati EFS di riferimento -----------------------------------------
st.subheader("2️⃣ Dati EFS di riferimento (allenamento)")

efs_source = st.radio(
    "Fonte EFS per zona:",
    ["Usa il riepilogo calcolato in questa sessione", "Importa CSV riepilogo EFS"],
    key="efs_source_race",
)

zone_efs_ref = {}

if efs_source == "Usa il riepilogo calcolato in questa sessione":
    final_summary_df = st.session_state.get("final_summary_df")
    if final_summary_df is not None and not final_summary_df.empty:
        for _, row in final_summary_df.iterrows():
            zone_efs_ref[row["Zona"]] = float(row["EFS media (km/h)"])
        st.success(
            "✅ EFS di riferimento: " +
            ", ".join(f"{z} = {v:.2f} km/h" for z, v in zone_efs_ref.items())
        )
    else:
        st.warning(
            "⚠️ Nessun riepilogo finale disponibile in questa sessione: carica dei file .fit "
            "più in alto nella pagina, oppure importa un CSV."
        )
else:
    uploaded_efs_csv = st.file_uploader(
        "Carica CSV riepilogo EFS per zona (colonne: Zona, EFS media (km/h))",
        type=["csv"],
        key="efs_ref_csv",
    )
    if uploaded_efs_csv is not None:
        try:
            ref_df = pd.read_csv(uploaded_efs_csv)
            zona_col = next((c for c in ref_df.columns if c.strip().lower() == "zona"), None)
            efs_col = next((c for c in ref_df.columns if "efs" in c.lower()), None)
            if zona_col is None or efs_col is None:
                st.error(
                    "⚠️ Il CSV deve contenere una colonna 'Zona' e una colonna EFS "
                    "(es. 'EFS media (km/h)')."
                )
            else:
                for _, row in ref_df.iterrows():
                    zone_efs_ref[row[zona_col]] = float(row[efs_col])
                st.success(
                    "✅ EFS di riferimento importate: " +
                    ", ".join(f"{z} = {v:.2f} km/h" for z, v in zone_efs_ref.items())
                )
        except Exception as e:
            st.error(f"⚠️ Errore durante la lettura del CSV: {e}")

st.divider()

# --- 3. Vincoli orari per zona -------------------------------------------
st.subheader("3️⃣ Vincoli orari per zona (Z2 - Z5)")

PRESET_VALUES = {
    "Cauto":        {"t_z2": 0.0, "t_z3": 0.0, "t_z4": 0.0, "t_z5": 0.0},
    "Conservativo": {"t_z2": 0.0, "t_z3": 0.0, "t_z4": 0.0, "t_z5": 0.0},
    "Coraggioso":   {"t_z2": 3.0, "t_z3": 0.0, "t_z4": 0.0, "t_z5": 0.0},
    "Competitivo":  {"t_z2": 6.0, "t_z3": 1.5, "t_z4": 0.0, "t_z5": 0.0},
}

CAUTO_Z1_FACTOR = 0.9  # EFS Zona 1 ridotta del 10% quando il profilo è "Cauto"


def render_profile_selector(key_prefix: str, title: str | None = None):
    """
    Disegna il selettore di profilo + slider per un singolo profilo di gara.
    key_prefix distingue i session_state key (es. "race", "race_a", "race_b")
    così più profili possono coesistere in pagina senza sovrascriversi.
    Ritorna: t_z2, t_z3, t_z4, t_z5, use_cauto, preset
    """
    if title:
        st.markdown(f"**{title}**")

    preset = st.radio(
        "Profilo di gara:",
        ["Cauto", "Conservativo", "Coraggioso", "Competitivo", "Personalizzato"],
        horizontal=True,
        key=f"{key_prefix}_profile_preset",
    )

    # Applica i valori preset SOLO quando il profilo cambia, così l'utente può
    # comunque modificare gli slider liberamente dopo la selezione (anche in
    # "Personalizzato", dove semplicemente non si sovrascrive nulla).
    last_preset_key = f"_last_{key_prefix}_preset"
    if st.session_state.get(last_preset_key) != preset:
        if preset in PRESET_VALUES:
            st.session_state[f"{key_prefix}_t_z2"] = PRESET_VALUES[preset]["t_z2"]
            st.session_state[f"{key_prefix}_t_z3"] = PRESET_VALUES[preset]["t_z3"]
            st.session_state[f"{key_prefix}_t_z4"] = PRESET_VALUES[preset]["t_z4"]
            st.session_state[f"{key_prefix}_t_z5"] = PRESET_VALUES[preset]["t_z5"]
        st.session_state[last_preset_key] = preset

    if preset == "Conservativo":
        st.info("ℹ️ L'atleta correrà unicamente in Zona 1")
    elif preset == "Cauto":
        st.info(
            "ℹ️ L'atleta correrà unicamente in Zona 1, con EFS Zona 1 ridotta del "
            f"{int((1 - CAUTO_Z1_FACTOR) * 100)}% per margine di sicurezza"
        )

    col1, col2, col3, col4 = st.columns(4)
    t_z2 = col1.slider("Ore in Zona 2", min_value=0.0, max_value=15.0, step=0.25, key=f"{key_prefix}_t_z2")
    t_z3 = col2.slider("Ore in Zona 3", min_value=0.0, max_value=15.0, step=0.25, key=f"{key_prefix}_t_z3")
    t_z4 = col3.slider("Ore in Zona 4", min_value=0.0, max_value=15.0, step=0.25, key=f"{key_prefix}_t_z4")
    t_z5 = col4.slider("Ore in Zona 5", min_value=0.0, max_value=15.0, step=0.25, key=f"{key_prefix}_t_z5")

    use_cauto = (preset == "Cauto")
    return t_z2, t_z3, t_z4, t_z5, use_cauto, preset


compare_mode = st.checkbox("🔀 Confronta due profili (calcola un range)", key="race_compare_mode")

if not compare_mode:
    t_z2, t_z3, t_z4, t_z5, use_cauto, preset = render_profile_selector("race")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        t_z2_a, t_z3_a, t_z4_a, t_z5_a, use_cauto_a, preset_a = render_profile_selector("race_a", "Profilo A")
    with col_b:
        t_z2_b, t_z3_b, t_z4_b, t_z5_b, use_cauto_b, preset_b = render_profile_selector("race_b", "Profilo B")

st.divider()

# --- 4. Calcolo previsione -----------------------------------------------
st.subheader("4️⃣ Previsione tempo gara")


def compute_race_time(t_z2, t_z3, t_z4, t_z5, zone_efs_ref, race_efd_m, use_cauto):
    """
    Calcola il tempo di gara previsto per un singolo profilo (vincoli orari
    per zona + EFS di riferimento). Se use_cauto=True, l'EFS di Zona 1 usata
    per il tempo residuo viene ridotta di CAUTO_Z1_FACTOR.
    Ritorna un dict con "error" (None se ok) e i dettagli del calcolo.
    """
    total_efd_km = race_efd_m / 1000
    zone_times_h = {"Z5": t_z5, "Z4": t_z4, "Z3": t_z3, "Z2": t_z2}

    remaining_efd_km = total_efd_km
    capped = False
    capped_zone = None
    breakdown = []

    # Si procede dalla zona più alta (Z5) alla più bassa (Z2), sottraendo
    # via via l'EFD "consumata" da ciascun vincolo orario. Se in un
    # qualunque momento l'EFD residua andrebbe sotto zero, si tronca
    # quella zona a ciò che resta e ci si ferma (le zone successive
    # -incluse eventuali zone più basse e la Zona 1- non ricevono EFD).
    for zname in ["Z5", "Z4", "Z3", "Z2"]:
        t_h = zone_times_h[zname]
        efs = zone_efs_ref.get(zname)

        if t_h > 0 and (efs is None or efs <= 0):
            return {"error": f"Manca l'EFS di riferimento per la {zname}: impossibile calcolare."}

        contrib_km = t_h * (efs or 0.0)

        if remaining_efd_km - contrib_km <= 0:
            breakdown.append({
                "Zona": zname, "Tempo (h)": t_h,
                "EFD coperta (km)": round(remaining_efd_km, 2),
            })
            remaining_efd_km = 0.0
            capped = True
            capped_zone = zname
            break
        else:
            breakdown.append({
                "Zona": zname, "Tempo (h)": t_h,
                "EFD coperta (km)": round(contrib_km, 2),
            })
            remaining_efd_km -= contrib_km

    efs_z1 = zone_efs_ref.get("Z1")
    if efs_z1 is None or efs_z1 <= 0:
        return {"error": "Manca l'EFS di riferimento per la Zona 1: impossibile calcolare il tempo restante."}

    efs_z1_used = efs_z1 * CAUTO_Z1_FACTOR if use_cauto else efs_z1

    time_z1_h = (remaining_efd_km / efs_z1_used) if remaining_efd_km > 0 else 0.0
    race_time_h = time_z1_h + t_z2 + t_z3 + t_z4 + t_z5

    return {
        "error": None,
        "race_time_h": race_time_h,
        "capped": capped,
        "capped_zone": capped_zone,
        "breakdown": breakdown,
        "remaining_efd_km": remaining_efd_km,
        "time_z1_h": time_z1_h,
        "efs_z1_used": efs_z1_used,
        "total_efd_km": total_efd_km,
    }


def display_result(result, label: str | None = None):
    suffix = f" — {label}" if label else ""
    st.metric(f"⏱️ Tempo di gara previsto{suffix}", seconds_to_hhmm(result["race_time_h"] * 3600))

    if result["capped"]:
        st.warning(
            f"⚠️ I vincoli orari inseriti richiedono più EFD di quanta ne offra il percorso "
            f"gara: la {result['capped_zone']} è stata troncata e la Zona 1 non contribuisce "
            f"(tempo = 0h). Il profilo scelto non è sostenibile su questa distanza."
        )

    with st.expander(f"Dettaglio calcolo{suffix}"):
        st.write(f"EFD totale gara: **{result['total_efd_km']:.2f} km**")
        st.dataframe(
            pd.DataFrame(result["breakdown"]), use_container_width=True, hide_index=True
        )
        st.write(f"EFD residua per Zona 1: **{result['remaining_efd_km']:.2f} km**")
        st.write(f"Tempo in Zona 1: **{seconds_to_hhmm(result['time_z1_h'] * 3600)}**")
        st.write(f"EFS Zona 1 usata: **{result['efs_z1_used']:.2f} km/h**")


if st.button("🎯 Calcola previsione", key="compute_race_time"):
    if race_efd_m is None:
        st.error("⚠️ Carica prima un file gara valido (sezione 1).")
    elif not zone_efs_ref:
        st.error("⚠️ Nessun dato EFS di riferimento disponibile (sezione 2).")
    elif not compare_mode:
        result = compute_race_time(t_z2, t_z3, t_z4, t_z5, zone_efs_ref, race_efd_m, use_cauto)
        if result["error"]:
            st.error(f"⚠️ {result['error']}")
            st.session_state["prediction_ready"] = False
        else:
            display_result(result)
            st.session_state["prediction_ready"] = True
            st.session_state["prediction_compare_mode"] = False
            st.session_state["result_single"] = result
    else:
        result_a = compute_race_time(t_z2_a, t_z3_a, t_z4_a, t_z5_a, zone_efs_ref, race_efd_m, use_cauto_a)
        result_b = compute_race_time(t_z2_b, t_z3_b, t_z4_b, t_z5_b, zone_efs_ref, race_efd_m, use_cauto_b)

        if result_a["error"]:
            st.error(f"⚠️ Profilo A: {result_a['error']}")
        if result_b["error"]:
            st.error(f"⚠️ Profilo B: {result_b['error']}")

        if not result_a["error"] and not result_b["error"]:
            times_h = [result_a["race_time_h"], result_b["race_time_h"]]
            low_h, high_h = min(times_h), max(times_h)
            st.metric(
                "⏱️ Range tempo di gara previsto",
                f"{seconds_to_hhmm(low_h * 3600)} – {seconds_to_hhmm(high_h * 3600)}",
            )
            st.caption(
                f"Profilo A ({preset_a}): {seconds_to_hhmm(result_a['race_time_h'] * 3600)}  ·  "
                f"Profilo B ({preset_b}): {seconds_to_hhmm(result_b['race_time_h'] * 3600)}"
            )
            st.session_state["prediction_ready"] = True
            st.session_state["prediction_compare_mode"] = True
            st.session_state["result_a"] = result_a
            st.session_state["result_b"] = result_b
            st.session_state["preset_a_label"] = preset_a
            st.session_state["preset_b_label"] = preset_b
        else:
            st.session_state["prediction_ready"] = False

        if not result_a["error"]:
            display_result(result_a, label="Profilo A")
        if not result_b["error"]:
            display_result(result_b, label="Profilo B")


# ===========================================================
# UI — Race Plan (tempi di passaggio ai lap)
# ===========================================================
st.divider()
st.header("🗺️ Race Plan")

prediction_ready = st.session_state.get("prediction_ready", False)

if not prediction_ready:
    st.info(
        "ℹ️ Calcola prima una previsione tempo gara (sezione 4, pulsante "
        "'🎯 Calcola previsione') per poter costruire il race plan."
    )
else:
    want_plan = st.radio(
        "Vuoi costruire un race plan con i tempi di passaggio intermedi?",
        ["No", "Sì"],
        horizontal=True,
        key="want_race_plan",
    )

    if want_plan == "Sì":
        race_profile = st.session_state.get("race_profile")
        race_total_km = st.session_state.get("race_total_km")

        if race_profile is None or race_total_km is None:
            st.error("⚠️ Profilo del percorso non disponibile: ricarica il file gara in sezione 1.")
        else:
            st.caption(
                f"Distanza totale percorso: **{race_total_km:.2f} km**. "
                "Definisci i segmenti di tuo interesse con km di partenza e arrivo"
            )

            lap_input_method = st.radio(
                "Come vuoi inserire i segmenti?", ["Manualmente", "Importa CSV"], horizontal=True, key="lap_input_method"
            )

            laps_df = None

            if lap_input_method == "Manualmente":
                default_laps = pd.DataFrame({
                    "name": ["Lap 1"],
                    "start_km": [0.0],
                    "end_km": [round(min(10.0, race_total_km), 2)],
                })
                laps_edit = st.data_editor(
                    st.session_state.get("laps_editor_data", default_laps),
                    num_rows="dynamic",
                    use_container_width=True,
                    key="laps_editor",
                    column_config={
                        "name": st.column_config.TextColumn("Nome lap"),
                        "start_km": st.column_config.NumberColumn(
                            "Km partenza", min_value=0.0, max_value=float(race_total_km), step=0.1
                        ),
                        "end_km": st.column_config.NumberColumn(
                            "Km arrivo", min_value=0.0, max_value=float(race_total_km), step=0.1
                        ),
                    },
                )
                st.session_state["laps_editor_data"] = laps_edit
                laps_df = laps_edit.copy()
            else:
                uploaded_laps_csv = st.file_uploader(
                    "Carica CSV lap (colonne: name, start_km, end_km)", type=["csv"], key="laps_csv"
                )
                if uploaded_laps_csv is not None:
                    try:
                        laps_df = pd.read_csv(uploaded_laps_csv)
                        required = {"name", "start_km", "end_km"}
                        if not required.issubset(set(laps_df.columns)):
                            st.error("⚠️ Il CSV deve contenere le colonne: name, start_km, end_km")
                            laps_df = None
                        else:
                            laps_df = laps_df[["name", "start_km", "end_km"]]
                    except Exception as e:
                        st.error(f"⚠️ Errore durante la lettura del CSV: {e}")
                        laps_df = None

            if laps_df is not None and not laps_df.empty:
                laps_df = laps_df.dropna(subset=["name", "start_km", "end_km"]).reset_index(drop=True)
                laps_df["start_km"] = laps_df["start_km"].astype(float)
                laps_df["end_km"] = laps_df["end_km"].astype(float)

                errors = []
                if (laps_df["start_km"] >= laps_df["end_km"]).any():
                    errors.append("Ogni lap deve avere km partenza < km arrivo.")
                if (laps_df["end_km"] > race_total_km + 1e-6).any():
                    errors.append(f"Nessun lap può superare i {race_total_km:.2f} km del percorso.")
                if (laps_df["start_km"] < -1e-6).any():
                    errors.append("I km di partenza non possono essere negativi.")

                if errors:
                    for err in errors:
                        st.error(f"⚠️ {err}")
                else:
                    def cum_efd_at_km(km_value):
                        return float(np.interp(km_value, race_profile["km"], race_profile["cum_efd_km"]))

                    def build_plan_for_profile(result, label):
                        """
                        Per ciascun lap:
                          - e = EFD cumulato (valore assoluto) al km di ARRIVO del lap
                          - ΔEFS(%) dall'equazione cubica (Minetti/decadence fit)
                          - EFS lap = baseline_efs * (1 + ΔEFS%/100), baseline_efs =
                            EFD totale gara / tempo di gara previsto (EFS media pesata
                            del profilo)
                          - tempo lap grezzo = EFD del lap / EFS lap
                        I tempi grezzi vengono infine calibrati con un fattore unico in
                        modo che la somma torni esatta al tempo di gara previsto.
                        """
                        baseline_efs = result["total_efd_km"] / result["race_time_h"]
                        rows = []
                        raw_times_h = []
                        for _, lap in laps_df.iterrows():
                            efd_start = cum_efd_at_km(lap["start_km"])
                            efd_end = cum_efd_at_km(lap["end_km"])
                            efd_lap_km = max(efd_end - efd_start, 0.0)
                            e = abs(efd_end)  # EFD accumulato a fine lap
                            dist_lap_km = lap["end_km"] - lap["start_km"]
                            delta_pct = (38.204 - 1.2831 * e + 0.01336 * e**2 - 0.000047 * e**3)
                            efs_lap = baseline_efs * (1 + delta_pct / 100)
                            if efs_lap <= 0:
                                efs_lap = baseline_efs * 0.05  # floor di sicurezza
                            time_h = efd_lap_km / efs_lap
                            raw_times_h.append(time_h)
                            rows.append({
                                "Lap": lap["name"],
                                "Km partenza": lap["start_km"],
                                "Km arrivo": lap["end_km"],
                                "Distanza (km)": dist_lap_km,
                                "EFD lap (km)": efd_lap_km,
                                "EFD accumulato (km)": e,
                                "EFS lap (km/h)": efs_lap,
                            })

                        raw_times_h = np.array(raw_times_h)
                        total_raw_h = raw_times_h.sum()
                        calib_factor = (result["race_time_h"] / total_raw_h) if total_raw_h > 0 else 1.0
                        final_times_h = raw_times_h * calib_factor
                        cum_times_h = np.cumsum(final_times_h)

                        out = pd.DataFrame(rows)
                        out["Velocità (km/h)"]= np.divide(
                            out["distanza (km)"].to_numpy(), final_times_h,
                            out=np.zeros_like(final_times_h), where=final_times_h > 0
                        )
                        out[f"Tempo lap ({label})"] = [seconds_to_hhmm(t * 3600) for t in final_times_h]
                        out[f"Arrivo cumulato ({label})"] = [seconds_to_hhmm(t * 3600) for t in cum_times_h]
                        return out, calib_factor

                    compare = st.session_state.get("prediction_compare_mode", False)

                    if not compare:
                        result = st.session_state.get("result_single")
                        plan_df, calib = build_plan_for_profile(result, "gara")
                        st.caption(f"Fattore di calibrazione applicato: ×{calib:.3f}")
                        display_cols = ["Lap", "Km partenza", "Km arrivo", "Distanza (km)",
                                         "Velocità (km/h)"
                                         "Tempo lap (gara)", "Arrivo cumulato (gara)"]
                        st.dataframe(
                            plan_df[display_cols].style.format({
                                "Km partenza": "{:.2f}", "Km arrivo": "{:.2f}",
                                "Distanza (km)": "{:.2f}",
                                "Velocità (km/h)": "{:.1f}"
                            }),
                            use_container_width=True, hide_index=True,
                        )
                        st.download_button(
                            "📥 Scarica race plan (CSV)",
                            data=plan_df[display_cols].to_csv(index=False).encode("utf-8"),
                            file_name="race_plan.csv",
                            mime="text/csv",
                        )
                    else:
                        result_a = st.session_state.get("result_a")
                        result_b = st.session_state.get("result_b")
                        label_a = st.session_state.get("preset_a_label", "A")
                        label_b = st.session_state.get("preset_b_label", "B")

                        plan_a, calib_a = build_plan_for_profile(result_a, label_a)
                        plan_b, calib_b = build_plan_for_profile(result_b, label_b)

                        merged = plan_a[["Lap", "Km partenza", "Km arrivo", "Distanza (km)"]].copy()
                        merged[f"Velocità ({label_a}) km/h"] = plan_a["Velocità (km/h)"]
                        merged[f"Tempo segmento ({label_a})"] = plan_a[f"Tempo segmento ({label_a})"]
                        merged[f"Tempo di gara ({label_a})"] = plan_a[f"Arrivo cumulato ({label_a})"]

                        merged[f"Velocità ({label_b}) km/h"] = plan_b["Velocità (km/h)"]
                        merged[f"Tempo segmento ({label_b})"] = plan_b[f"Tempo lap ({label_b})"]
                        merged[f"Tempo di gara ({label_b})"] = plan_b[f"Arrivo cumulato ({label_b})"]

                        st.caption(
                            f"Fattore di calibrazione: {label_a} ×{calib_a:.3f} · {label_b} ×{calib_b:.3f}"
                        )
                        st.dataframe(
                            merged.style.format({
                                "Km partenza": "{:.2f}", 
                                "Km arrivo": "{:.2f}",
                                "Distanza (km)": "{:.2f}",
                                f"Velocità ({label_a}) km/h": "{:.1f}",
                                f"Velocità ({label_b}) km/h": "{:.1f}",
                            }),
                            use_container_width=True, hide_index=True,
                        )
                        st.download_button(
                            "📥 Scarica race plan (CSV)",
                            data=merged.to_csv(index=False).encode("utf-8"),
                            file_name="race_plan_confronto.csv",
                            mime="text/csv",
                        )