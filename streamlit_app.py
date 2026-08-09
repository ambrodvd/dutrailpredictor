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
conto del decadimento dell'EFS in funzione dell'EFD accumulato (modello
a bucket per lunghezza gara, calibrato su dati storici, con interpolazione
tra bucket adiacenti), calibrando i tempi lap-per-lap in modo che il totale
torni esatto rispetto alla previsione di sezione 4.

--------------------------------------------------------------------------
NOTA SULLE PERFORMANCE (importante se modifichi il file)
--------------------------------------------------------------------------
Streamlit ri-esegue l'intero script a ogni interazione con un qualsiasi
widget (slider, radio, bottone...), anche uno lontano dalla sezione in
questione. Senza caching questo significa ri-parsare TUTTI i file .fit
caricati e ri-fare tutto il resampling/integrazione ogni singola volta,
anche se l'utente ha solo mosso uno slider nella sezione Race Plan.

Per questo il parsing + processing di ogni traccia (.fit o .gpx) è
racchiuso in funzioni decorate con @st.cache_data, con chiave = bytes
del file + parametri numerici che influenzano il risultato (smoothing,
passo di resampling, zone FC). Se questi non cambiano, la chiamata
successiva è una lookup istantanea invece di un recompute completo.

Regola pratica quando si aggiunge nuova logica di parsing/elaborazione:
la funzione pesante (fitparse, haversine su array grandi, resampling)
va SEMPRE dietro una cache_data che riceve bytes/primitivi come argomenti,
mai un file-like object o un DataFrame già "vivo" come argomento di cache,
altrimenti la chiave di cache è instabile o l'hashing diventa esso stesso
il collo di bottiglia.
--------------------------------------------------------------------------

Run with:
    streamlit run streamlit_app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fitparse import FitFile
import xml.etree.ElementTree as ET
import warnings
import fitdecode

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
# Modello di decadimento EFS a bucket per lunghezza gara (EFD totale)
# ---------------------------------------------------------------------------
# Calibrato separatamente nell'app "Speed Decadence" (dutrailpredictor), che
# fitta un'equazione per ciascun bucket di EFD totale su "p" = EFD accumulato
# in % del totale del file (non EFD assoluto: cosi' file/gare di lunghezza
# diversa restano confrontabili sulla stessa posizione relativa di gara).
# Qui i coefficienti sono hardcoded: quando si ricalibra su nuovi dati in
# Speed Decadence, aggiornare i coefficienti sotto di conseguenza.
#
# NOTA: il bucket "< 40 km" non ha ancora dati/equazione propria (nessun file
# idoneo al momento della calibrazione): per gare piu' corte del centro del
# bucket piu' basso disponibile si usa quella stessa curva (clamp), invece
# di estrapolare un'equazione mai validata su quel range.
def _delta_efs_40_60(p):
    return 23.860 - 0.5984 * p + 0.00263 * p**2


def _delta_efs_60_100(p):
    return 36.833 - 0.7490 * p - 0.00277 * p**2 + 0.000057 * p**3


def _delta_efs_100_plus(p):
    return 41.599 - 1.3125 * p + 0.01342 * p**2 - 0.000064 * p**3


BUCKET_DELTA_EFS_FN = {
    "40-60 km": _delta_efs_40_60,
    "60-100 km": _delta_efs_60_100,
    ">100 km": _delta_efs_100_plus,
}

# Centro di ciascun bucket in km di EFD totale, usato per l'interpolazione.
# Il centro del bucket ">100 km" non è vincolato da un limite superiore: va
# aggiornato con la mediana reale dei file di quel bucket (visibile nel
# pannello diagnostico di Speed Decadence) quando disponibile. Il valore
# sotto è una stima di partenza, da correggere.
BUCKET_CENTERS_KM = {
    "40-60 km": 50.0,
    "60-100 km": 80.0,
    ">100 km": 130.0,  # <-- aggiornare con la mediana reale dei file >100 km
}

_BUCKET_ORDER_BY_CENTER = sorted(BUCKET_CENTERS_KM, key=lambda k: BUCKET_CENTERS_KM[k])


def predict_delta_efs_blended(efd_pct: float, race_total_efd_km: float) -> float:
    """
    Scostamento % dell'EFS previsto a una posizione di gara (efd_pct, 0-100),
    interpolando linearmente tra le curve dei due bucket più vicini per
    centro (in km di EFD totale) in base alla lunghezza della gara target.
    Questo evita salti bruschi tra bucket adiacenti (es. gara da 99 km vs
    101 km danno una previsione quasi identica, non un salto netto).
    """
    centers = [(name, BUCKET_CENTERS_KM[name]) for name in _BUCKET_ORDER_BY_CENTER]

    if race_total_efd_km <= centers[0][1]:
        return float(BUCKET_DELTA_EFS_FN[centers[0][0]](efd_pct))
    if race_total_efd_km >= centers[-1][1]:
        return float(BUCKET_DELTA_EFS_FN[centers[-1][0]](efd_pct))

    for (name_lo, c_lo), (name_hi, c_hi) in zip(centers[:-1], centers[1:]):
        if c_lo <= race_total_efd_km <= c_hi:
            w_hi = (race_total_efd_km - c_lo) / (c_hi - c_lo)
            w_lo = 1.0 - w_hi
            pred_lo = BUCKET_DELTA_EFS_FN[name_lo](efd_pct)
            pred_hi = BUCKET_DELTA_EFS_FN[name_hi](efd_pct)
            return float(w_lo * pred_lo + w_hi * pred_hi)

    return float("nan")


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
# .fit / .gpx parsing (pure functions — no Streamlit calls inside, so they
# are safe to call from cached wrappers below)
# ---------------------------------------------------------------------------

def parse_fit(file_obj) -> pd.DataFrame:
    """Extract lat, lon, elevation, heart rate, elapsed time (s) per record.

    Uses fitdecode instead of fitparse: some devices (e.g. COROS) write
    malformed field-size declarations in definition messages (typically in
    developer-data fields). fitparse treats this as fatal and aborts the
    whole file; fitdecode logs a warning and skips just that field, so the
    rest of the file still parses normally.
    """
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with fitdecode.FitReader(file_obj) as fit:
            for frame in fit:
                if frame.frame_type != fitdecode.FIT_FRAME_DATA or frame.name != "record":
                    continue
                data = {f.name: f.value for f in frame.fields}

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
    df["hr"] = df["hr"].astype(float)
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df


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


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in meters between arrays of points."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def hr_to_zone_vec(hr_arr: np.ndarray, z1, z2, z3, z4, z5) -> np.ndarray:
    """
    Vectorized HR -> zone assignment (replaces a per-point Python loop).
    z5 is unused as an upper bound (Z5 = "above z4") but kept in the
    signature for symmetry with the rest of the app / CSV zone format.
    Returns an object array with values in {"Z1".."Z5"} or None where hr is NaN.
    """
    bins = [z1, z2, z3, z4]  # upper bounds for Z1..Z4; anything above z4 is Z5
    idx = np.digitize(hr_arr, bins)  # 0..4
    labels = np.array(["Z1", "Z2", "Z3", "Z4", "Z5"], dtype=object)
    zones_arr = labels[np.clip(idx, 0, 4)]
    zones_arr = np.where(np.isnan(hr_arr), None, zones_arr)
    return zones_arr


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
    zone = hr_to_zone_vec(hr_seg, *zones)         # vectorized, was a per-point loop

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

    # Cumulative D+/D- at each grid point, used later to compute gain/loss
    # per Race Plan lap (via interpolation) and to draw the elevation chart.
    dplus_step = np.where(dz > 0, dz, 0.0)
    dminus_step = np.where(dz < 0, -dz, 0.0)
    cum_dplus_m = np.concatenate([[0.0], np.cumsum(dplus_step)])
    cum_dminus_m = np.concatenate([[0.0], np.cumsum(dminus_step)])

    profile = pd.DataFrame({
        "km": grid / 1000,
        "ele_m": ele_grid,
        "cum_efd_km": cum_efd_m / 1000,
        "cum_dplus_m": cum_dplus_m,
        "cum_dminus_m": cum_dminus_m,
    })

    summary = {
        "horizontal_distance_m": total_dist,
        "d_plus_m": d_plus,
        "d_minus_m": d_minus,
        "efd_m": float(efd_m.sum()),
    }
    return summary, profile


# ---------------------------------------------------------------------------
# CACHED WRAPPERS
# ---------------------------------------------------------------------------
# These are the ONLY entry points the UI code below should call for parsing
# .fit/.gpx files. Cache key = raw file bytes + the numeric params that
# actually affect the output. As long as the user doesn't change the file
# or those settings, any unrelated widget interaction elsewhere on the page
# hits the cache instead of re-parsing/re-integrating from scratch.
@st.cache_data(show_spinner="Elaborazione file .fit...")
def load_and_process_fit(file_bytes: bytes, smooth_window: int,
                          resample_step_m: float, zones: tuple):
    """Parse + process a training .fit file. Returns a status dict."""
    df_points = parse_fit(io.BytesIO(file_bytes))
    if len(df_points) < 2:
        return {"status": "invalid"}
    if df_points["hr"].isna().all():
        return {"status": "no_hr"}
    segments, summary = process_track(df_points, smooth_window, resample_step_m, zones)
    if segments is None:
        return {"status": "degenerate"}
    return {"status": "ok", "segments": segments, "summary": summary}


@st.cache_data(show_spinner="Elaborazione percorso gara...")
def load_and_process_race(file_bytes: bytes, ext: str, smooth_window: int,
                           resample_step_m: float):
    """Parse + process a race-course .gpx or .fit file. Returns (summary, profile) or (None, None)."""
    if ext == "gpx":
        df_race = parse_gpx(io.BytesIO(file_bytes))
    else:
        df_full = parse_fit(io.BytesIO(file_bytes))
        df_race = df_full[["lat", "lon", "ele"]] if not df_full.empty else df_full

    if df_race is None or len(df_race) < 2:
        return None, None

    return compute_race_efd(df_race, smooth_window, resample_step_m)


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

    result = load_and_process_fit(uploaded.getvalue(), smooth_window, resample_step, zones)

    if result["status"] == "invalid":
        st.error("Traccia non valida: mancano punti GPS/timestamp validi in questo file.")
        continue
    if result["status"] == "no_hr":
        st.warning("⚠️ Nessun dato di frequenza cardiaca trovato in questo file: impossibile assegnare le zone.")
        continue
    if result["status"] == "degenerate":
        st.error("Impossibile calcolare la distanza percorsa (traccia degenere).")
        continue

    segments, summary = result["segments"], result["summary"]
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
        max_time_min = zdf["Tempo (min)"].max()  # massimo in un singolo file
        # media pesata: EFS = EFD totale / tempo totale (non media delle medie)
        efs_weighted = (total_efd_km / (total_time_min / 60)) if total_time_min > 0 else np.nan
        summary_rows.append({
            "Zona": z,
            "Tempo totale (min)": int(round(total_time_min)),
            "Durata massima (min)": int(round(max_time_min)),
            "EFD totale (km)": total_efd_km,
            "EFS media (km/h)": efs_weighted,
            "N. file": len(zdf),
        })

    final_summary = pd.DataFrame(summary_rows)
    st.session_state['final_summary_df'] = final_summary
    st.session_state['max_minutes_by_zone'] = dict(
        zip(final_summary["Zona"], final_summary["Durata massima (min)"])
    )
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
        race_summary, race_profile = load_and_process_race(
            selected_race_file.getvalue(), ext, smooth_window, resample_step
        )
        if race_summary is None:
            st.error("⚠️ Traccia gara non valida o degenere: mancano punti GPS sufficienti.")
        else:
            st.session_state["race_efd_m"] = race_summary["efd_m"]
            st.session_state["race_profile"] = race_profile
            st.session_state["race_total_km"] = race_summary["horizontal_distance_m"] / 1000
            c1, c2, c3 = st.columns(3)
            c1.metric("Distanza orizzontale", f"{race_summary['horizontal_distance_m']/1000:.2f} km")
            c2.metric("D+ / D−", f"{race_summary['d_plus_m']:.0f} m / {race_summary['d_minus_m']:.0f} m")
            c3.metric("EFD totale gara", f"{race_summary['efd_m']/1000:.2f} km")

            _race_efd_km = race_summary["efd_m"] / 1000
            _centers = sorted(BUCKET_CENTERS_KM.items(), key=lambda kv: kv[1])
            if _race_efd_km <= _centers[0][1]:
                st.caption(f"📐 Modello decadimento: curva bucket «{_centers[0][0]}» (nessun bucket più corto disponibile).")
            elif _race_efd_km >= _centers[-1][1]:
                st.caption(f"📐 Modello decadimento: curva bucket «{_centers[-1][0]}» (nessun bucket più lungo disponibile).")
            else:
                for (n_lo, c_lo), (n_hi, c_hi) in zip(_centers[:-1], _centers[1:]):
                    if c_lo <= _race_efd_km <= c_hi:
                        w_hi = (_race_efd_km - c_lo) / (c_hi - c_lo)
                        st.caption(
                            f"📐 Modello decadimento: interpolazione {(1-w_hi)*100:.0f}% «{n_lo}» "
                            f"+ {w_hi*100:.0f}% «{n_hi}»."
                        )
                        break
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

max_minutes_by_zone = st.session_state.get("max_minutes_by_zone")
if max_minutes_by_zone:
    st.caption(
        "📏 Minutaggio massimo per sessione"
        "(considerati i file caricati per l'analisi):"
    )
    max_rows = [
        {"Zona": z, "Durata massima (min)": max_minutes_by_zone[z]}
        for z in ["Z1", "Z2", "Z3", "Z4", "Z5"]
        if z in max_minutes_by_zone
    ]
    st.dataframe(pd.DataFrame(max_rows), use_container_width=True, hide_index=True)
else:
    st.caption(
        "ℹ️ Carica ed elabora i file di allenamento più in alto per vedere qui i minutaggi "
        "massimi per zona registrati dall'atleta."
    )

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
    Gli slider sono in minuti in UI, ma la funzione ritorna sempre le ore
    (t_z2, t_z3, t_z4, t_z5), unità usata dal resto del calcolo.
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
            st.session_state[f"{key_prefix}_t_z2"] = int(round(PRESET_VALUES[preset]["t_z2"] * 60))
            st.session_state[f"{key_prefix}_t_z3"] = int(round(PRESET_VALUES[preset]["t_z3"] * 60))
            st.session_state[f"{key_prefix}_t_z4"] = int(round(PRESET_VALUES[preset]["t_z4"] * 60))
            st.session_state[f"{key_prefix}_t_z5"] = int(round(PRESET_VALUES[preset]["t_z5"] * 60))
        st.session_state[last_preset_key] = preset

    if preset == "Conservativo":
        st.info("ℹ️ L'atleta correrà unicamente in Zona 1")
    elif preset == "Cauto":
        st.info(
            "ℹ️ L'atleta correrà unicamente in Zona 1, con EFS Zona 1 ridotta del "
            f"{int((1 - CAUTO_Z1_FACTOR) * 100)}% per margine di sicurezza"
        )

    col1, col2, col3, col4 = st.columns(4)
    t_z2_min = col1.slider("Minuti in Zona 2", min_value=0, max_value=900, step=15, key=f"{key_prefix}_t_z2")
    t_z3_min = col2.slider("Minuti in Zona 3", min_value=0, max_value=900, step=15, key=f"{key_prefix}_t_z3")
    t_z4_min = col3.slider("Minuti in Zona 4", min_value=0, max_value=900, step=15, key=f"{key_prefix}_t_z4")
    t_z5_min = col4.slider("Minuti in Zona 5", min_value=0, max_value=900, step=15, key=f"{key_prefix}_t_z5")

    t_z2, t_z3, t_z4, t_z5 = t_z2_min / 60, t_z3_min / 60, t_z4_min / 60, t_z5_min / 60

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
    time_used_h_total = 0.0  # tempo EFFETTIVAMENTE speso in Z2-Z5 (puo' essere < somma dei t_h impostati se si tronca)

    # Si procede dalla zona più alta (Z5) alla più bassa (Z2), sottraendo
    # via via l'EFD "consumata" da ciascun vincolo orario. Se in un
    # qualunque momento l'EFD residua andrebbe sotto zero, si tronca
    # quella zona al tempo effettivamente necessario a coprire l'EFD
    # rimasta (non al tempo pieno richiesto dall'utente) e ci si ferma:
    # le zone successive (incluse eventuali zone più basse e la Zona 1)
    # non ricevono EFD né tempo.
    for zname in ["Z5", "Z4", "Z3", "Z2"]:
        t_h = zone_times_h[zname]
        efs = zone_efs_ref.get(zname)

        if t_h > 0 and (efs is None or efs <= 0):
            return {"error": f"Manca l'EFS di riferimento per la {zname}: impossibile calcolare."}

        contrib_km = t_h * (efs or 0.0)

        if remaining_efd_km - contrib_km <= 0:
            efd_covered = remaining_efd_km
            # tempo davvero necessario a coprire l'EFD residua a questo EFS,
            # NON il t_h pieno impostato dall'utente
            time_used_h = (efd_covered / efs) if (efs and efs > 0) else 0.0
            breakdown.append({
                "Zona": zname, "Tempo (h)": time_used_h,
                "EFD coperta (km)": round(efd_covered, 2),
            })
            remaining_efd_km = 0.0
            time_used_h_total += time_used_h
            capped = True
            capped_zone = zname
            break
        else:
            breakdown.append({
                "Zona": zname, "Tempo (h)": t_h,
                "EFD coperta (km)": round(contrib_km, 2),
            })
            remaining_efd_km -= contrib_km
            time_used_h_total += t_h

    efs_z1 = zone_efs_ref.get("Z1")
    if efs_z1 is None or efs_z1 <= 0:
        return {"error": "Manca l'EFS di riferimento per la Zona 1: impossibile calcolare il tempo restante."}

    efs_z1_used = efs_z1 * CAUTO_Z1_FACTOR if use_cauto else efs_z1

    time_z1_h = (remaining_efd_km / efs_z1_used) if remaining_efd_km > 0 else 0.0
    race_time_h = time_z1_h + time_used_h_total

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
            f"⚠️ La somma dei vincoli orari inseriti supera il tempo di gara totale previsto. "
            f"La {result['capped_zone']} è stata utilizzata solo parzialmente e la Zona 1 non contribuisce "
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
        "ℹ️ Calcola una previsione di gara per poter costruire il race plan."
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
            )

            n_segments = st.number_input(
                "Quanti segmenti vuoi creare?",
                min_value=1, max_value=50, step=1,
                key="n_segments_input",
            )

            with st.form("segments_form"):

                seg_names = []
                seg_ends = []
                for i in range(n_segments):
                    is_last = (i == n_segments - 1)
                    default_end = (
                        race_total_km if is_last
                        else round(race_total_km * (i + 1) / n_segments, 2)
                    )
                    c1, c2 = st.columns([2, 1])
                    name_i = c1.text_input(
                        f"Nome segmento {i + 1}",
                        value=f"Segmento {i + 1}",
                        key=f"seg_name_{i}",
                    )
                    end_i = c2.number_input(
                        f"Km finale segmento {i + 1}",
                        min_value=0.0, max_value=float(race_total_km),
                        value=float(default_end), step=0.1,
                        disabled=is_last,
                        key=f"seg_end_{i}",
                    )
                    seg_names.append(name_i)
                    seg_ends.append(race_total_km if is_last else end_i)

                submitted_segments = st.form_submit_button("💾 Salva segmenti")

            if submitted_segments:
                errors = []
                for i in range(1, n_segments):
                    if seg_ends[i] <= seg_ends[i - 1] + 1e-9:
                        errors.append(
                            f"Il km finale del segmento {i + 1} deve essere maggiore "
                            f"di quello del segmento {i}."
                        )
                if errors:
                    for err in errors:
                        st.error(f"⚠️ {err}")
                    st.session_state["laps_df_confirmed"] = None
                else:
                    start_list = [0.0] + seg_ends[:-1]
                    final_names = [
                        n.strip() if n.strip() else f"Segmento {i + 1}"
                        for i, n in enumerate(seg_names)
                    ]
                    st.session_state["laps_df_confirmed"] = pd.DataFrame({
                        "name": final_names,
                        "start_km": start_list,
                        "end_km": seg_ends,
                    })
                    st.success(f"✅ {n_segments} segmenti salvati.")

            laps_df = st.session_state.get("laps_df_confirmed")

            if laps_df is not None and not laps_df.empty:
                st.dataframe(
                    laps_df.rename(columns={
                        "name": "Nome", "start_km": "Km partenza", "end_km": "Km arrivo",
                    }).style.format({"Km partenza": "{:.2f}", "Km arrivo": "{:.2f}"}),
                    use_container_width=True, hide_index=True,
                )

            if laps_df is not None and not laps_df.empty:
                if st.button("🎯 Calcola Race Plan definitivo", key="compute_race_plan_button"):
                    def cum_efd_at_km(km_value):
                        return float(np.interp(km_value, race_profile["km"], race_profile["cum_efd_km"]))

                    def cum_dplus_at_km(km_value):
                        return float(np.interp(km_value, race_profile["km"], race_profile["cum_dplus_m"]))

                    def cum_dminus_at_km(km_value):
                        return float(np.interp(km_value, race_profile["km"], race_profile["cum_dminus_m"]))

                    LAP_COLORS = [
                        "#1f4e79", "#f2c14e", "#8fb339", "#c1440e",
                        "#5b8c85", "#a05195", "#2a9d8f", "#e76f51",
                    ]

                    def plot_elevation_profile_with_laps(race_profile: pd.DataFrame, laps_df: pd.DataFrame):
                        """Profilo altimetrico del percorso con ciascun segmento del Race Plan
                        evidenziato con una banda colorata e il proprio nome."""
                        fig, ax = plt.subplots(figsize=(10, 3.2))
                        ax.plot(race_profile["km"], race_profile["ele_m"], color="#333333", linewidth=1.2, zorder=3)
                        ax.fill_between(
                            race_profile["km"], race_profile["ele_m"], race_profile["ele_m"].min(),
                            color="#333333", alpha=0.08, zorder=1,
                        )

                        ele_min = race_profile["ele_m"].min()
                        ele_max = race_profile["ele_m"].max()
                        y_span = max(ele_max - ele_min, 1.0)
                        label_y = ele_max + 0.06 * y_span

                        for i, lap in laps_df.iterrows():
                            color = LAP_COLORS[i % len(LAP_COLORS)]
                            ax.axvspan(lap["start_km"], lap["end_km"], color=color, alpha=0.22, zorder=0)
                            ax.axvline(lap["start_km"], color=color, alpha=0.5, linewidth=0.8, zorder=2)
                            mid_km = (lap["start_km"] + lap["end_km"]) / 2
                            ax.text(
                                mid_km, label_y, lap["name"], ha="center", va="bottom",
                                fontsize=8, rotation=0, color=color, fontweight="bold",
                            )

                        ax.set_xlabel("Distanza (km)")
                        ax.set_ylabel("Quota (m)")
                        ax.set_ylim(ele_min - 0.05 * y_span, label_y + 0.12 * y_span)
                        ax.set_xlim(race_profile["km"].min(), race_profile["km"].max())
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        fig.tight_layout()
                        return fig

                    def build_plan_for_profile(result, label):
                        """
                        Per ciascun lap:
                          - efd_pct = EFD cumulato a fine lap, in % dell'EFD totale gara
                          - ΔEFS(%) previsto tramite predict_delta_efs_blended, che
                            interpola tra i bucket di lunghezza gara più vicini (evita
                            salti bruschi tra bucket adiacenti)
                          - EFS lap = baseline_efs * (1 + ΔEFS%/100), baseline_efs =
                            EFD totale gara / tempo di gara previsto (EFS media pesata
                            del profilo)
                          - tempo lap grezzo = EFD del lap / EFS lap
                        I tempi grezzi vengono infine calibrati con un fattore unico in
                        modo che la somma torni esatta al tempo di gara previsto.
                        """
                        baseline_efs = result["total_efd_km"] / result["race_time_h"]
                        total_efd_km_race = result["total_efd_km"]
                        rows = []
                        raw_times_h = []
                        for _, lap in laps_df.iterrows():
                            efd_start = cum_efd_at_km(lap["start_km"])
                            efd_end = cum_efd_at_km(lap["end_km"])
                            efd_lap_km = max(efd_end - efd_start, 0.0)
                            e = abs(efd_end)  # EFD accumulato a fine lap (km)
                            efd_pct = (e / total_efd_km_race * 100.0) if total_efd_km_race > 0 else 0.0
                            dist_lap_km = lap["end_km"] - lap["start_km"]
                            dplus_lap_m = cum_dplus_at_km(lap["end_km"]) - cum_dplus_at_km(lap["start_km"])
                            dminus_lap_m = cum_dminus_at_km(lap["end_km"]) - cum_dminus_at_km(lap["start_km"])
                            delta_pct = predict_delta_efs_blended(efd_pct, total_efd_km_race)
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
                                "D+ (m)": dplus_lap_m,
                                "D- (m)": dminus_lap_m,
                                "EFD lap (km)": efd_lap_km,
                                "EFD accumulato (km)": e,
                                "EFD accumulato (%)": efd_pct,
                                "EFS lap (km/h)": efs_lap,
                            })

                        raw_times_h = np.array(raw_times_h)
                        total_raw_h = raw_times_h.sum()
                        calib_factor = (result["race_time_h"] / total_raw_h) if total_raw_h > 0 else 1.0
                        final_times_h = raw_times_h * calib_factor
                        cum_times_h = np.cumsum(final_times_h)

                        out = pd.DataFrame(rows)
                        out["Velocità (km/h)"] = np.divide(
                            out["Distanza (km)"].to_numpy(), final_times_h,
                            out=np.zeros_like(final_times_h), where=final_times_h > 0
                        )
                        out[f"Tempo segmento ({label})"] = [seconds_to_hhmm(t * 3600) for t in final_times_h]
                        out[f"Tempo di gara ({label})"] = [seconds_to_hhmm(t * 3600) for t in cum_times_h]
                        return out, calib_factor

                    compare = st.session_state.get("prediction_compare_mode", False)

                    if not compare:
                        result = st.session_state.get("result_single")
                        plan_df, calib = build_plan_for_profile(result, "gara")
                        plan_df = plan_df.rename(columns={
                            "Tempo segmento (gara)": "Tempo segmento",
                            "Tempo di gara (gara)": "Tempo di gara",
                        })
                        st.caption(f"Fattore di calibrazione applicato: ×{calib:.3f}")
                        display_cols = ["Lap", "Km partenza", "Km arrivo", "Distanza (km)",
                                         "D+ (m)", "D- (m)",
                                         "Velocità (km/h)",
                                         "Tempo segmento", "Tempo di gara"]
                        st.dataframe(
                            plan_df[display_cols].style.format({
                                "Km partenza": "{:.2f}", "Km arrivo": "{:.2f}",
                                "Distanza (km)": "{:.2f}",
                                "D+ (m)": "{:.0f}", "D- (m)": "{:.0f}",
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

                        st.pyplot(plot_elevation_profile_with_laps(race_profile, laps_df))
                    else:
                        result_a = st.session_state.get("result_a")
                        result_b = st.session_state.get("result_b")
                        label_a = st.session_state.get("preset_a_label", "A")
                        label_b = st.session_state.get("preset_b_label", "B")

                        plan_a, calib_a = build_plan_for_profile(result_a, label_a)
                        plan_b, calib_b = build_plan_for_profile(result_b, label_b)

                        merged = plan_a[["Lap", "Km partenza", "Km arrivo", "Distanza (km)", "D+ (m)", "D- (m)"]].copy()
                        merged[f"Velocità ({label_a}) km/h"] = plan_a["Velocità (km/h)"]
                        merged[f"Tempo segmento ({label_a})"] = plan_a[f"Tempo segmento ({label_a})"]
                        merged[f"Tempo di gara ({label_a})"] = plan_a[f"Tempo di gara ({label_a})"]

                        merged[f"Velocità ({label_b}) km/h"] = plan_b["Velocità (km/h)"]
                        merged[f"Tempo segmento ({label_b})"] = plan_b[f"Tempo segmento ({label_b})"]
                        merged[f"Tempo di gara ({label_b})"] = plan_b[f"Tempo di gara ({label_b})"]

                        st.caption(
                            f"Fattore di calibrazione: {label_a} ×{calib_a:.3f} · {label_b} ×{calib_b:.3f}"
                        )
                        st.dataframe(
                            merged.style.format({
                                "Km partenza": "{:.2f}", 
                                "Km arrivo": "{:.2f}",
                                "Distanza (km)": "{:.2f}",
                                "D+ (m)": "{:.0f}", "D- (m)": "{:.0f}",
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

                        st.pyplot(plot_elevation_profile_with_laps(race_profile, laps_df))