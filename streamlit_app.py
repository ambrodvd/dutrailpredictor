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
NOTA SUGLI UPLOAD (.zip e file senza estensione)
--------------------------------------------------------------------------
Gli uploader accettano anche archivi .zip e file .gz (es. export
TrainingPeaks, che consegna uno .zip contenente file .fit.gz, o file
privi di estensione / con estensione sbagliata, da scompattare e
rinominare a mano in .fit per essere leggibili).

Pipeline di normalizzazione (expand_uploads):
  1. gunzip  — se i bytes iniziano con 1f 8b vengono scompattati (anche
     compressioni ripetute, max 3 giri);
  2. unzip   — se sono uno zip, viene attraversato in memoria, e ogni
     contenuto ripassa dal punto 1 (gli zip annidati sono gestiti);
  3. sniff   — il formato si riconosce dai magic bytes, non dall'estensione:
     un file FIT porta la stringa ASCII ".FIT" ai byte 8-11 dell'header,
     un GPX è XML con root <gpx>.

Nessuna scrittura su disco. In pratica: si trascina lo zip così com'è e
non serve né scompattare né rinominare nulla.

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

Nota: l'estrazione dagli zip avviene PRIMA delle funzioni cached e
produce dei bytes, quindi la chiave di cache resta stabile esattamente
come per un file caricato direttamente.
--------------------------------------------------------------------------

Run with:
    streamlit run streamlit_app.py
"""

import io
import gzip
import zlib
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fitparse import FitFile
import xml.etree.ElementTree as ET
import warnings
import fitdecode
import plotly.graph_objects as go

st.set_page_config(page_title=" 🔮 DU Trail Predictor", layout="wide")

MIN_FILE_DURATION_S = 10 * 60  # file sotto questa durata esclusi dal riepilogo finale

# Soglie minime di file per abilitare modelli con più parametri (evita overfitting).
# Usate da fit_candidates nella scheda 🆚 Confronto.
MIN_FILES_FOR_QUADRATIC = 10
MIN_FILES_FOR_CUBIC = 15

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
GIORNI_IT = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
MESI_IT = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
           "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]


def format_date_it(dt, with_time: bool = True) -> str:
    """'mercoledì 12 agosto 2026, 06:42'. Nomi hardcoded invece di usare
    locale.setlocale(): il locale it_IT non è garantito sui container di
    Streamlit Cloud e fallirebbe silenziosamente in inglese."""
    if dt is None or pd.isna(dt):
        return "-"
    s = f"{GIORNI_IT[dt.weekday()]} {dt.day} {MESI_IT[dt.month - 1]} {dt.year}"
    if with_time:
        s += f", {dt.hour:02d}:{dt.minute:02d}"
    return s


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


def hhmm_to_seconds(t):
    """Converte 'HH:MM:SS' o 'MM:SS' in secondi. Non parsabile -> NaN.
    Inverso di seconds_to_hhmm: serve a rileggere i tempi del race plan CSV."""
    if pd.isna(t) or not isinstance(t, str) or not t.strip():
        return np.nan
    try:
        parts = [int(p) for p in t.strip().split(":")]
    except ValueError:
        return np.nan
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        return np.nan
    return h * 3600 + m * 60 + s

def slugify_name(text: str) -> str:
    """'Davide Grazielli' -> 'Davide Grazielli'. Compone nomi file sicuri: tiene solo
    alfanumerici, spazi, trattini e underscore, poi collassa gli spazi."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in " -_" else "_" for ch in (text or "").strip()
    )
    return "_".join(cleaned.split()) or "report"

# ---------------------------------------------------------------------------
# Riconoscimento formato + espansione archivi .zip
# ---------------------------------------------------------------------------
# TrainingPeaks (e altri export) consegnano un .zip in cui i file sono privi
# di estensione o ne hanno una sbagliata. Invece di chiedere all'utente di
# rinominare a mano, si guardano i magic bytes:
#   - FIT : l'header contiene la stringa ASCII ".FIT" ai byte 8-11
#   - GPX : XML con root <gpx>
# Tutto avviene in memoria, nessuna scrittura su disco.

ZIP_MAGIC = b"PK\x03\x04"
GZIP_MAGIC = b"\x1f\x8b"

# Estensioni "di trasporto" da togliere dal nome quando si ricostruisce
# un nome pulito: TrainingPeaks consegna tipicamente <id>.fit.gz.
_STRIPPABLE_EXT = {"gz", "gzip", "zip", "fit", "gpx", "dat", "txt", "bin"}




def gunzip_if_needed(data: bytes, max_depth: int = 3) -> bytes:
    """Scompatta i bytes se sono gzip (TrainingPeaks consegna .fit.gz).
    Itera perché capita di trovare doppie compressioni; se la decompressione
    fallisce si restituiscono i bytes così com'erano."""
    if data is None:
        return data
    depth = 0
    while data[:2] == GZIP_MAGIC and depth < max_depth:
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError, zlib.error):
            break
        depth += 1
    return data


def sniff_ext(data: bytes) -> str | None:
    """Ritorna 'fit' o 'gpx' guardando il contenuto, None se non riconosciuto.
    I bytes devono essere già scompattati (vedi gunzip_if_needed)."""
    if data is None or len(data) < 16:
        return None

    if data[8:12] == b".FIT":
        return "fit"

    head = data[:2048].lstrip().lower()
    if head.startswith(b"<?xml") or head.startswith(b"<"):
        if b"<gpx" in head:
            return "gpx"

    return None


def clean_base_name(name: str) -> str:
    """'workouts/12345.fit.gz' -> '12345'. Toglie path e le estensioni di
    trasporto impilate, così il nome finale non diventa '12345.fit.fit'."""
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    for _ in range(3):
        stem, dot, suffix = base.rpartition(".")
        if dot and suffix.lower() in _STRIPPABLE_EXT:
            base = stem
        else:
            break
    return base or name


def _iter_zip_blobs(data: bytes, depth: int = 0):
    """Restituisce (nome_interno, bytes) per ogni file dentro lo zip.
    Ogni contenuto viene scompattato se gzip (.fit.gz) e attraversato se a
    sua volta uno zip (max 2 livelli, per non finire in ricorsioni
    patologiche su archivi malformati)."""
    if depth > 2:
        return
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            blob = gunzip_if_needed(z.read(info))
            if blob[:4] == ZIP_MAGIC:
                yield from _iter_zip_blobs(blob, depth + 1)
            else:
                yield info.filename, blob


def expand_uploads(uploaded_files, accept=("fit",)):
    """
    Normalizza la lista di file caricati in [(nome, bytes, ext), ...].

    - un file caricato direttamente viene riconosciuto dai magic bytes
      (con fallback sull'estensione del nome);
    - un .zip viene aperto in memoria e ogni contenuto riconosciuto come
      formato in `accept` viene restituito come voce a sé stante;
    - i nomi duplicati vengono resi univoci con un suffisso _2, _3, ...
      perché sono usati come chiave in per_file_segments e nelle key dei
      widget Streamlit.

    Ritorna una lista vuota se non viene riconosciuto nulla.
    """
    out = []
    seen = set()

    def _add(name: str, blob: bytes, ext: str):
        base = clean_base_name(name)
        candidate = f"{base}.{ext}"
        n = 1
        while candidate in seen:
            n += 1
            candidate = f"{base}_{n}.{ext}"
        seen.add(candidate)
        out.append((candidate, blob, ext))

    for up in uploaded_files or []:
        # scompatta subito un eventuale .gz caricato direttamente
        raw = gunzip_if_needed(up.getvalue())

        if raw[:4] == ZIP_MAGIC:
            found = 0
            try:
                for inner_name, blob in _iter_zip_blobs(raw):
                    ext = sniff_ext(blob)
                    if ext in accept:
                        _add(inner_name, blob, ext)
                        found += 1
            except zipfile.BadZipFile:
                st.error(f"⚠️ {up.name}: archivio zip non valido o corrotto.")
                continue
            if found == 0:
                st.warning(
                    f"⚠️ {up.name}: nessun file "
                    f"{'/'.join('.' + a for a in accept)} riconosciuto nell'archivio."
                )
            else:
                st.caption(
                    f"📦 {up.name}: {found} file estratti dall'archivio."
                )
            continue

        ext = sniff_ext(raw)
        if ext is None:
            ext = up.name.rsplit(".", 1)[-1].lower() if "." in up.name else ""
        if ext in accept:
            _add(up.name, raw, ext)
        else:
            st.warning(f"⚠️ {up.name}: formato non riconosciuto, file ignorato.")

    return out


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
    utc_offset_s = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with fitdecode.FitReader(file_obj) as fit:
            for frame in fit:
                if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                    continue

                # Il messaggio "activity" porta sia timestamp (UTC) sia
                # local_timestamp (ora dell'orologio dell'atleta): la loro
                # differenza è l'offset del fuso in cui l'attività è stata
                # svolta. Serve per mostrare la data giusta: un'uscita
                # serale delle 22:30 in Italia è salvata come 20:30 UTC, e
                # una partenza dopo le 02:00 finirebbe sul giorno prima.
                if frame.name == "activity":
                    adata = {f.name: f.value for f in frame.fields}
                    ts_utc = adata.get("timestamp")
                    ts_loc = adata.get("local_timestamp")
                    if ts_utc is not None and ts_loc is not None:
                        try:
                            a = ts_utc.replace(tzinfo=None)
                            b = ts_loc.replace(tzinfo=None)
                            # gli offset reali sono multipli di 15 minuti
                            utc_offset_s = round((b - a).total_seconds() / 900) * 900
                        except (AttributeError, TypeError):
                            utc_offset_s = None
                    continue

                if frame.name != "record":
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

    # offset del fuso orario dell'attività (secondi), None se il file non
    # contiene un messaggio "activity" con local_timestamp
    df.attrs["utc_offset_s"] = utc_offset_s

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
# Race Plan: grafico del profilo e costruzione del piano
# ---------------------------------------------------------------------------
# Queste funzioni stavano dentro il blocco `if st.button(...)` della sezione
# Race Plan. Sono state portate a livello di modulo perché quel blocco viene
# eseguito solo nel rerun in cui il bottone è premuto: qualunque widget
# aggiunto lì dentro (es. le checkbox di scelta colonne per l'export PDF)
# sparirebbe al rerun successivo. Ora il piano si calcola una volta, si salva
# in session_state e la UI di export vive fuori dal bottone.

def cum_efd_at_km(race_profile, km_value):
    return float(np.interp(km_value, race_profile["km"], race_profile["cum_efd_km"]))


def cum_dplus_at_km(race_profile, km_value):
    return float(np.interp(km_value, race_profile["km"], race_profile["cum_dplus_m"]))


def cum_dminus_at_km(race_profile, km_value):
    return float(np.interp(km_value, race_profile["km"], race_profile["cum_dminus_m"]))

LAP_COLORS = [
    "#1f4e79", "#f2c14e", "#8fb339", "#c1440e",
    "#5b8c85", "#a05195", "#2a9d8f", "#e76f51",
]

# Colori dei tempi di passaggio: profilo A / profilo B.
# Scelti scuri e distinti dai colori delle bande dei lap,
# che sono al 22% di alpha e quindi molto chiari.
PROFILE_TIME_COLORS = ["#11324d", "#8c2f0d"]

def plot_elevation_profile_with_laps(race_profile: pd.DataFrame, laps_df: pd.DataFrame,
                                      time_labels=None, legend_labels=None):
    """Profilo altimetrico del percorso con ciascun segmento del Race Plan
    evidenziato con una banda colorata e il proprio nome.

    time_labels (opzionale): lista lunga quanto laps_df, dove ogni voce
    è una lista di 1 o 2 stringhe — il tempo di gara cumulato a fine
    segmento, uno per profilo. Viene scritta in verticale sotto il
    confine di fine segmento. legend_labels dà i nomi dei profili
    corrispondenti (per la legenda dei colori)."""
    fig, ax = plt.subplots(figsize=(10, 4.2 if time_labels else 3.2))
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

    # --- tempi di gara ai confini di fine segmento ---
    # Testo ruotato di 90°: con molti segmenti le etichette
    # orizzontali si sovrapporrebbero. Con due profili si
    # scrivono due colonne affiancate, distinte per colore.
    bottom = ele_min - 0.05 * y_span
    if time_labels:
        x_span = race_profile["km"].max() - race_profile["km"].min()
        dx = 0.009 * x_span
        time_y = ele_min - 0.09 * y_span

        for i, lap in laps_df.iterrows():
            labels_i = time_labels[i]
            ax.axvline(lap["end_km"], color="#999999", alpha=0.6,
                       linewidth=0.7, linestyle=":", zorder=2)
            if len(labels_i) == 1:
                offsets = [0.0]
            else:
                offsets = [-dx, dx]
            for txt, off, col in zip(labels_i, offsets, PROFILE_TIME_COLORS):
                ax.text(
                    lap["end_km"] + off, time_y, txt,
                    ha="center", va="top", fontsize=7,
                    rotation=90, color=col, fontweight="bold",
                )

        bottom = ele_min - 0.45 * y_span

        if legend_labels and len(legend_labels) > 1:
            # sotto l'asse x: in alto si scontrerebbe con i
            # nomi dei segmenti
            for name, col in zip(legend_labels, PROFILE_TIME_COLORS):
                ax.plot([], [], color=col, linewidth=3, label=name)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                      fontsize=8, frameon=False, ncol=len(legend_labels))

    ax.set_xlabel("Distanza (km)")
    ax.set_ylabel("Quota (m)")
    ax.set_ylim(bottom, label_y + 0.12 * y_span)
    # margine a destra: l'etichetta del tempo finale sta
    # esattamente sull'ultimo km e verrebbe tagliata a metà
    x_lo = race_profile["km"].min()
    x_hi = race_profile["km"].max()
    x_pad = 0.012 * (x_hi - x_lo) if time_labels else 0.0
    ax.set_xlim(x_lo, x_hi + x_pad)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

def build_plan_for_profile(result, label, race_profile, laps_df):
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
        efd_start = cum_efd_at_km(race_profile, lap["start_km"])
        efd_end = cum_efd_at_km(race_profile, lap["end_km"])
        efd_lap_km = max(efd_end - efd_start, 0.0)
        e = abs(efd_end)  # EFD accumulato a fine lap (km)
        efd_pct = (e / total_efd_km_race * 100.0) if total_efd_km_race > 0 else 0.0
        dist_lap_km = lap["end_km"] - lap["start_km"]
        dplus_lap_m = cum_dplus_at_km(race_profile, lap["end_km"]) - cum_dplus_at_km(race_profile, lap["start_km"])
        dminus_lap_m = cum_dminus_at_km(race_profile, lap["end_km"]) - cum_dminus_at_km(race_profile, lap["start_km"])
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


# ---------------------------------------------------------------------------
# Export PDF del Race Plan
# ---------------------------------------------------------------------------
# reportlab è importato in modo "soffice": se manca, l'app continua a
# funzionare e la sezione di export mostra le istruzioni per installarlo,
# invece di far esplodere l'intera pagina all'avvio.
try:
    from reportlab.lib import colors as _rl_colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Image as RLImage, Table, TableStyle)
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


# Formati numerici per l'export: stessa resa della tabella a schermo, così
# il PDF non mostra 12.340000000000002 dove la pagina mostra 12.34.
PLAN_COL_FORMATS = {
    "Km partenza": "{:.2f}",
    "Km arrivo": "{:.2f}",
    "Distanza (km)": "{:.2f}",
    "D+ (m)": "{:.0f}",
    "D- (m)": "{:.0f}",
    "EFD lap (km)": "{:.2f}",
    "EFD accumulato (km)": "{:.2f}",
    "EFD accumulato (%)": "{:.1f}",
    "EFS lap (km/h)": "{:.2f}",
}


def format_plan_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Converte il piano in stringhe già formattate, pronte per la tabella PDF."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        fmt = None
        for key, f in PLAN_COL_FORMATS.items():
            if col == key or col.startswith(key):
                fmt = f
                break
        if fmt is None and "Velocità" in col:
            fmt = "{:.1f}"
        if fmt is not None:
            out[col] = [fmt.format(v) if pd.notna(v) else "-" for v in df[col]]
        else:
            out[col] = df[col].astype(str)
    return out


def fig_to_png_bytes(fig, dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def build_race_plan_pdf(table_df: pd.DataFrame, chart_png: bytes | None,
                         title: str, meta_lines: list[str],
                         comment: str | None = None,
                         note: str | None = None) -> bytes:
    """Compone il PDF del race plan: intestazione, dati di sintesi, profilo
    altimetrico e tabella dei segmenti. Ritorna i bytes del file.

    Formato orizzontale A4: in modalità confronto la tabella arriva a una
    dozzina di colonne e in verticale sarebbe illeggibile."""
    buf = io.BytesIO()
    from xml.sax.saxutils import escape as _xml_escape
    page_w, page_h = landscape(A4)
    margin = 12 * mm
    doc = SimpleDocTemplate(
        buf, pagesize=landscape(A4),
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title=title, author="DU Trail Predictor",
    )
    avail_w = page_w - 2 * margin

    styles = getSampleStyleSheet()
    h_style = ParagraphStyle("H", parent=styles["Heading1"], fontSize=16,
                             spaceAfter=2, textColor=_rl_colors.HexColor("#1f4e79"))
    meta_style = ParagraphStyle("M", parent=styles["Normal"], fontSize=8.5,
                                leading=12, textColor=_rl_colors.HexColor("#444444"))
    note_style = ParagraphStyle("N", parent=styles["Normal"], fontSize=7.5,
                                leading=10, textColor=_rl_colors.HexColor("#777777"))

    story = [Paragraph(_xml_escape(title), h_style), Spacer(1, 3)]
    if meta_lines:
        story.append(Paragraph(
            "&nbsp;&nbsp;·&nbsp;&nbsp;".join(_xml_escape(m) for m in meta_lines),
            meta_style,
        ))
    story.append(Spacer(1, 8))

    if comment and comment.strip():
        comment_style = ParagraphStyle(
            "C", parent=styles["Normal"], fontSize=9, leading=13,
            textColor=_rl_colors.HexColor("#222222"),
            backColor=_rl_colors.HexColor("#f4f6f8"),
            borderColor=_rl_colors.HexColor("#1f4e79"),
            borderWidth=0.6, borderPadding=6,
        )
        # gli a capo scritti dal coach vanno preservati
        safe_comment = _xml_escape(comment.strip()).replace("\n", "<br/>")
        story.append(Paragraph(
            f"<b>Note del coach</b><br/>{safe_comment}", comment_style))
        story.append(Spacer(1, 10))

    if chart_png:
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(io.BytesIO(chart_png)).getSize()
        img_w = avail_w
        img_h = img_w * ih / iw
        max_img_h = 78 * mm          # lascia spazio alla tabella nella prima pagina
        if img_h > max_img_h:
            img_h = max_img_h
            img_w = img_h * iw / ih
        img = RLImage(io.BytesIO(chart_png), width=img_w, height=img_h)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Spacer(1, 10))

    # --- tabella ---
    from reportlab.lib.enums import TA_CENTER

    header = list(table_df.columns)
    body_rows = table_df.astype(str).values.tolist()

    n_cols = len(header)
    font_size = 8 if n_cols <= 8 else (7 if n_cols <= 11 else 6)

    # Larghezze proporzionali al contenuto dei DATI, non delle intestazioni:
    # in modalità confronto le colonne tipo "Tempo di gara"
    # hanno intestazioni lunghissime e, pesandole, schiaccerebbero "D+ (m)"
    # sotto la larghezza della sua stessa intestazione. Le intestazioni vanno
    # invece a capo (sono Paragraph), con un minimo pari alla parola più
    # lunga così nessuna parola resta tagliata.
    widths = []
    for j, col in enumerate(header):
        data_len = max([len(str(r[j])) for r in body_rows] or [0])
        longest_word = max((len(w) for w in str(col).split()), default=4)
        widths.append(max(data_len, longest_word, 4))
    total = float(sum(widths))
    col_widths = [avail_w * w / total for w in widths]

    th_style = ParagraphStyle(
        "th", fontName="Helvetica-Bold", fontSize=font_size,
        leading=font_size + 1.5, alignment=TA_CENTER,
        textColor=_rl_colors.white,
    )
    header_cells = [Paragraph(_xml_escape(str(c)), th_style) for c in header]
    data = [header_cells] + body_rows

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), _rl_colors.HexColor("#1f4e79")),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), font_size),
        ("LEADING", (0, 1), (-1, -1), font_size + 3),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.25, _rl_colors.HexColor("#cccccc")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, _rl_colors.HexColor("#1f4e79")),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i),
                          _rl_colors.HexColor("#f4f6f8")))
    tbl.setStyle(TableStyle(style))
    story.append(tbl)

    if note:
        story.append(Spacer(1, 8))
        story.append(Paragraph(note, note_style))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fit di forme candidate (usato dalla scheda 🆚 Confronto)
# ---------------------------------------------------------------------------
# Stessa libreria di forme dell'app Speed Decadence: si fittano più equazioni
# e si sceglie quella con adj. R² migliore. n_files limita i gradi di libertà
# quando il campione è piccolo, perché i bin di uno stesso file NON sono
# osservazioni indipendenti.
def fit_candidates(x: np.ndarray, y: np.ndarray, n_files: int) -> dict:
    n = len(x)
    ss_tot = np.sum((y - y.mean()) ** 2)

    def r2_of(pred):
        ss_res = np.sum((y - pred) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    def adj_r2_of(r2, k):
        if n - k - 1 <= 0 or np.isnan(r2):
            return np.nan
        return 1 - (1 - r2) * (n - 1) / (n - k - 1)

    candidates = {}

    c = np.polyfit(x, y, 1)
    pred = np.polyval(c, x)
    r2 = r2_of(pred)
    candidates["Lineare"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, xx),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot p",
    }

    x_log = np.log1p(x)
    c = np.polyfit(x_log, y, 1)
    pred = np.polyval(c, x_log)
    r2 = r2_of(pred)
    candidates["Logaritmica"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, np.log1p(xx)),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot \ln(p+1)",
    }

    x_sqrt = np.sqrt(x)
    c = np.polyfit(x_sqrt, y, 1)
    pred = np.polyval(c, x_sqrt)
    r2 = r2_of(pred)
    candidates["Radice quadrata"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, np.sqrt(xx)),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot \sqrt{{p}}",
    }

    if n_files >= MIN_FILES_FOR_QUADRATIC:
        c = np.polyfit(x, y, 2)
        pred = np.polyval(c, x)
        r2 = r2_of(pred)
        candidates["Quadratica"] = {
            "coefs": c, "k": 3, "r2": r2, "adj_r2": adj_r2_of(r2, 3),
            "predict": lambda xx, c=c: np.polyval(c, xx),
            "equation": rf"\Delta EFS(\%) = {c[2]:.3f} + {c[1]:.4f} \cdot p + {c[0]:.5f} \cdot p^2",
        }

    if n_files >= MIN_FILES_FOR_CUBIC:
        c = np.polyfit(x, y, 3)
        pred = np.polyval(c, x)
        r2 = r2_of(pred)
        candidates["Cubica"] = {
            "coefs": c, "k": 4, "r2": r2, "adj_r2": adj_r2_of(r2, 4),
            "predict": lambda xx, c=c: np.polyval(c, xx),
            "equation": (rf"\Delta EFS(\%) = {c[3]:.3f} + {c[2]:.4f} \cdot p + "
                         rf"{c[1]:.5f} \cdot p^2 + {c[0]:.6f} \cdot p^3"),
        }

    return candidates


def best_candidate(candidates: dict):
    return max(candidates, key=lambda k: (candidates[k]["adj_r2"]
                                           if not np.isnan(candidates[k]["adj_r2"]) else -np.inf))


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

    # Data/ora di inizio: primo timestamp valido dei record (UTC), riportato
    # all'ora locale dell'atleta se il file dichiara l'offset del fuso.
    start_utc = df_points["timestamp"].iloc[0]
    offset_s = df_points.attrs.get("utc_offset_s")
    if offset_s is not None:
        start_local = start_utc + pd.Timedelta(seconds=offset_s)
    else:
        start_local = start_utc
    summary["start_time"] = start_local.tz_localize(None) if start_local.tzinfo else start_local
    summary["start_time_is_local"] = offset_s is not None

    return {"status": "ok", "segments": segments, "summary": summary}


@st.cache_data(show_spinner="Elaborazione gara reale...")
def load_and_process_real_race(file_bytes: bytes, smooth_window: int,
                                resample_step_m: float, zones: tuple):
    """Parse + process del .fit della gara realmente corsa (scheda 🆚 Confronto).
    A differenza di load_and_process_fit NON pretende dati cardiaci: qui servono
    solo tempo, distanza ed EFD per segmento. Se l'HR manca, la colonna 'zone'
    resta a None e nessuno la legge."""
    df = parse_fit(io.BytesIO(file_bytes))
    if df.empty or len(df) < 2:
        return None, None
    return process_track(df, smooth_window, resample_step_m, zones)


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

# ---------------------------------------------------------------------------
# Schede (tab)
# ---------------------------------------------------------------------------
# Nota su Streamlit: il corpo di TUTTE le schede viene eseguito a ogni rerun
# (la scheda non attiva è solo nascosta via CSS, non "saltata"). Quindi le
# variabili definite in una scheda restano visibili in quelle successive,
# purché i blocchi `with` siano nell'ordine giusto: zone FC -> allenamenti ->
# gara -> race plan -> export. Non serve duplicare nulla in session_state per
# il semplice passaggio di dati tra schede.

tab_hr, tab_training, tab_race, tab_plan, tab_export, tab_compare = st.tabs([
    "❤️ Zone FC",
    "📊 Allenamenti",
    "🎯 Gara e profilo",
    "🗺️ Race Plan",
    "📄 Export",
    "🆚 Confronto",
])

# --- File upload & processing settings ---
with st.sidebar:
    st.header("Impostazioni")
    uploaded_files = st.file_uploader(
        "Carica uno o più file .fit (anche .gz, o un .zip che li contiene)",
        type=["fit", "gz", "zip"], accept_multiple_files=True
    )
    smooth_window = st.slider("Finestra di smoothing quota (punti)", 1, 31, 9, step=2)
    resample_step = st.slider("Passo di ricampionamento (m)", 5, 100, 20, step=5)
    st.caption(
        "Puoi caricare direttamente lo .zip di export (es. TrainingPeaks): i file al suo interno "
        "vengono scompattati (.gz) e riconosciuti dal contenuto, non serve rinominarli in .fit."
    )
    st.caption(
        "Lo smoothing ripulisce la quota grezza GPS/barometrica prima di calcolare la pendenza. "
        "Il passo di ricampionamento controlla la risoluzione orizzontale usata per integrare l'energia."
    )

with tab_hr:
    # --- HR Zones module ---
    default_zones = {'z1': 140, 'z2': 160, 'z3': 170, 'z4': 180, 'z5': 200}
    for zone, val in default_zones.items():
        if zone not in st.session_state:
            st.session_state[zone] = val

    st.subheader("Zone cardiache dell'atleta")

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

        # Il campo "Nome dell'atleta" vive nella scheda 📄 Export: qui si legge
        # soltanto, se già compilato, per intestare il CSV delle zone.
        athlete_name = (st.session_state.get('athlete_name') or '').strip()
        export_df = pd.DataFrame([{
            'athlete_name': athlete_name or 'atleta',
            'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5
        }])
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Esporta zone in CSV",
            data=csv_data,
            file_name=(f"{athlete_name.replace(' ', '_')}_HR_Zones.csv"
                       if athlete_name else "HR_Zones.csv"),
            mime='text/csv'
        )


with tab_training:
    per_file_zone_tables = []  # list of (filename, zone_table, total_time_s)
    per_file_segments = {}     # filename -> segments dataframe (needed for Lap Analysis below)

    fit_uploads = expand_uploads(uploaded_files, accept=("fit",))

    # --- esclusione manuale di singoli file ------------------------------------
    # I nomi prodotti da expand_uploads sono deterministici a parità di upload,
    # quindi si possono usare come identificativo stabile tra un rerun e l'altro.
    if "excluded_fit_files" not in st.session_state:
        st.session_state["excluded_fit_files"] = set()
    excluded_files = st.session_state["excluded_fit_files"]

    # Ripulisce i nomi non più presenti tra i file caricati: senza questo, un
    # file rimosso e ricaricato più tardi resterebbe escluso a sorpresa.
    excluded_files &= {name for name, _b, _e in fit_uploads}

    active_uploads = [u for u in fit_uploads if u[0] not in excluded_files]


    # --- controllo duplicati ----------------------------------------------------
    def find_duplicate_groups(uploads, tolerance_s: int = 120):
        """Raggruppa i file la cui partenza dista meno di `tolerance_s` secondi e
        restituisce solo i gruppi con più di un file.

        Due export della stessa attività — es. lo stesso allenamento scaricato sia
        da Garmin sia da TrainingPeaks — hanno bytes diversi ma partenza a pochi
        secondi di distanza (il ricampionamento o il trim possono spostare il primo
        record), quindi il confronto va fatto sulla data e con una tolleranza, non
        sull'hash del file.

        Si raggruppa per vicinanza e non arrotondando al minuto: due file a 06:45:29
        e 06:45:31 distano 2 secondi ma cadrebbero in due minuti diversi, e il
        duplicato sfuggirebbe. Due attività realmente distinte che partono a meno di
        due minuti l'una dall'altra non esistono nella pratica."""
        entries = []
        for name, blob, _ext in uploads:
            res = load_and_process_fit(blob, smooth_window, resample_step, zones)
            if res.get("status") != "ok":
                continue
            start = res["summary"].get("start_time")
            if start is None:
                continue
            entries.append((pd.Timestamp(start), name))

        entries.sort(key=lambda e: e[0])

        groups = {}
        current = []
        for ts, name in entries:
            if current and (ts - current[-1][0]).total_seconds() <= tolerance_s:
                current.append((ts, name))
            else:
                if len(current) > 1:
                    groups[current[0][0]] = [n for _t, n in current]
                current = [(ts, name)]
        if len(current) > 1:
            groups[current[0][0]] = [n for _t, n in current]

        return groups


    dup_groups = find_duplicate_groups(active_uploads) if len(active_uploads) > 1 else {}
    duplicated_names = {n for names in dup_groups.values() for n in names}

    if dup_groups:
        dettaglio = "\n".join(
            f"- **{format_date_it(start)}** → {', '.join(names)}"
            for start, names in dup_groups.items()
        )
        st.warning(
            "⚠️ **Possibili allenamenti duplicati.** Questi file hanno la stessa data e "
            "ora di partenza. Escludi i doppioni con il "
            "bottone accanto al file.\n\n" + dettaglio
        )

    for fit_name, fit_bytes, _fit_ext in fit_uploads:
        is_excluded = fit_name in excluded_files

        head_col, btn_col = st.columns([5, 1])
        with head_col:
            st.header(f"📄 {fit_name}")
        with btn_col:
            if is_excluded:
                if st.button("↩️ Reincludi", key=f"incl_{fit_name}",
                             use_container_width=True):
                    excluded_files.discard(fit_name)
                    st.rerun()
            else:
                if st.button("🚫 Escludi", key=f"excl_{fit_name}",
                             use_container_width=True):
                    excluded_files.add(fit_name)
                    st.rerun()

        if is_excluded:
            st.caption("⛔ File escluso dall'analisi e dal riepilogo finale.")
            st.divider()
            continue

        result = load_and_process_fit(fit_bytes, smooth_window, resample_step, zones)

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
        per_file_segments[fit_name] = segments

        start_dt = summary.get("start_time")
        if start_dt is not None:
            date_label = f"🗓️ {format_date_it(start_dt)}"
            if not summary.get("start_time_is_local", False):
                date_label += "  ·  ora UTC (il file non dichiara il fuso)"
            if fit_name in duplicated_names:
                gemelli = [
                    n for names in dup_groups.values() if fit_name in names
                    for n in names if n != fit_name
                ]
                date_label += f"  ·  ⚠️ stessa partenza di: {', '.join(gemelli)}"
            st.caption(date_label)

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
            per_file_zone_tables.append((fit_name, zone_table, summary["total_time_s"]))

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
        if excluded_files:
            st.caption(
                "File esclusi manualmente: " + ", ".join(sorted(excluded_files))
            )
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




with tab_race:
    st.header("🎯 Gara e profilo di corsa")

    # --- 1. Percorso gara --------------------------------------------------
    st.subheader("1️⃣ Percorso gara (GPX o FIT)")
    race_files = st.file_uploader(
        "Carica uno o più file del percorso gara (.gpx, .fit, .gz o .zip)",
        type=["gpx", "fit", "gz", "zip"],
        accept_multiple_files=True,
        key="race_course_files",
    )

    race_uploads = expand_uploads(race_files, accept=("fit", "gpx"))

    if race_uploads:
        race_names = [name for name, _b, _e in race_uploads]
        selected_race_name = st.selectbox(
            "Seleziona il file gara da usare per la previsione:",
            race_names,
            key="selected_race_file",
        )
        _, selected_race_bytes, ext = next(
            r for r in race_uploads if r[0] == selected_race_name
        )

        # Il nome gara si preimposta dal nome del file e smette di seguirlo appena
        # l'utente lo personalizza (stessa logica del nome profilo in sezione 4).
        _last_race_file_key = "_last_race_file_for_name"
        _prev_file = st.session_state.get(_last_race_file_key)
        if _prev_file != selected_race_name:
            _cur = (st.session_state.get("race_name") or "").strip()
            _prev_stem = (_prev_file or "").rsplit(".", 1)[0]
            if _cur in ("", _prev_stem):
                st.session_state["race_name"] = selected_race_name.rsplit(".", 1)[0]
            st.session_state[_last_race_file_key] = selected_race_name

        try:
            race_summary, race_profile = load_and_process_race(
                selected_race_bytes, ext, smooth_window, resample_step
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

    # --- 2. Dati di riferimento: EFS + minutaggi massimi per zona -----------
    st.subheader("2️⃣ Dati di riferimento (EFS + minutaggi massimi per zona)")

    efs_source = st.radio(
        "Fonte dati di riferimento:",
        ["Usa il riepilogo calcolato in questa sessione", "Importa CSV riepilogo"],
        key="efs_source_race",
    )

    zone_efs_ref = {}
    max_minutes_by_zone = {}

    if efs_source == "Usa il riepilogo calcolato in questa sessione":
        final_summary_df = st.session_state.get("final_summary_df")
        if final_summary_df is not None and not final_summary_df.empty:
            for _, row in final_summary_df.iterrows():
                zone_efs_ref[row["Zona"]] = float(row["EFS media (km/h)"])
                max_minutes_by_zone[row["Zona"]] = float(row["Durata massima (min)"])
            st.success(
                "✅ I dati di riferimento per il calcolo della previsione sono quelli calcolati in questa sessione"
            )

            ref_export_df = pd.DataFrame([
                {
                    "Zona": z,
                    "EFS media (km/h)": zone_efs_ref[z],
                    "Durata massima (min)": max_minutes_by_zone[z],
                }
                for z in ["Z1", "Z2", "Z3", "Z4", "Z5"] if z in zone_efs_ref
            ])
            st.download_button(
                label="📥 Scarica dati di riferimento (EFS + minutaggi) in CSV",
                data=ref_export_df.to_csv(index=False).encode('utf-8'),
                file_name="riferimento_efs_minutaggi.csv",
                mime='text/csv',
                key="dl_ref_efs_minutaggi",
            )
        else:
            st.warning(
                "⚠️ Nessun riepilogo finale disponibile in questa sessione: carica dei file .fit "
                "nella scheda **📊 Allenamenti**, oppure importa un CSV."
            )
    else:
        uploaded_ref_csv = st.file_uploader(
            "Carica CSV riferimento (colonne: Zona, EFS media (km/h), Durata massima (min))",
            type=["csv"],
            key="efs_ref_csv",
        )
        if uploaded_ref_csv is not None:
            try:
                ref_df = pd.read_csv(uploaded_ref_csv)
                zona_col = next((c for c in ref_df.columns if c.strip().lower() == "zona"), None)
                efs_col = next((c for c in ref_df.columns if "efs" in c.lower()), None)
                max_col = next(
                    (c for c in ref_df.columns if "durata" in c.lower() or "massim" in c.lower()), None
                )
                if zona_col is None or efs_col is None:
                    st.error(
                        "⚠️ Il CSV deve contenere almeno una colonna 'Zona' e una colonna EFS "
                        "(es. 'EFS media (km/h)'). La colonna 'Durata massima (min)' è opzionale."
                    )
                else:
                    for _, row in ref_df.iterrows():
                        zone_efs_ref[row[zona_col]] = float(row[efs_col])
                        if max_col is not None and pd.notna(row[max_col]):
                            max_minutes_by_zone[row[zona_col]] = float(row[max_col])
                    msg = "✅ Dati di riferimento importati: " + ", ".join(
                        f"{z} = {v:.2f} km/h" for z, v in zone_efs_ref.items()
                    )
                    if max_col is None:
                        msg += " (nessuna colonna di minutaggio massimo trovata nel CSV: sezione 3 vuota)"
                    st.success(msg)
            except Exception as e:
                st.error(f"⚠️ Errore durante la lettura del CSV: {e}")

    st.divider()

    # --- 3. Vincoli orari per zona -------------------------------------------
    st.subheader("3️⃣ Tempi massimi per zona (Z1 - Z5)")

    if max_minutes_by_zone:
        st.caption(
            "📏 Minutaggio massimo per zona, in base alla fonte dati selezionata al punto 2️⃣:"
        )
        max_rows = [
            {"Zona": z, "Durata massima (min)": max_minutes_by_zone[z]}
            for z in ["Z1", "Z2", "Z3", "Z4", "Z5"]
            if z in max_minutes_by_zone
        ]
        st.dataframe(pd.DataFrame(max_rows), use_container_width=True, hide_index=True)
    else:
        st.caption(
            "ℹ️ Nessun minutaggio massimo disponibile: carica ed elabora i file di allenamento "
            "nella scheda **📊 Allenamenti**, oppure importa un CSV con la colonna 'Durata massima (min)'."
        )

    st.divider()

    # --- 4. Selezione profilo -------------------------------------------
    st.subheader("4️⃣ Seleziona il profilo di corsa dell'atleta")

    PRESET_VALUES = {
        "Cauto":        {"t_z2": 0.0, "t_z3": 0.0, "t_z4": 0.0, "t_z5": 0.0},
        "Conservativo": {"t_z2": 0.0, "t_z3": 0.0, "t_z4": 0.0, "t_z5": 0.0},
        "Coraggioso":   {"t_z2": 3.0, "t_z3": 0.0, "t_z4": 0.0, "t_z5": 0.0},
        "Competitivo":  {"t_z2": 6.0, "t_z3": 1.5, "t_z4": 0.0, "t_z5": 0.0},
    }

    CAUTO_Z1_FACTOR = 0.9  # EFS Zona 1 ridotta del 10% quando il profilo è "Cauto"


    ZONE_SLIDER_SPECS = [
        ("Z2", "t_z2", "Minuti in Zona 2"),
        ("Z3", "t_z3", "Minuti in Zona 3"),
        ("Z4", "t_z4", "Minuti in Zona 4"),
        ("Z5", "t_z5", "Minuti in Zona 5"),
    ]


    def render_profile_selector(key_prefix: str, title: str | None = None,
                                 max_minutes_by_zone: dict | None = None):
        """
        Disegna il selettore di profilo + slider per un singolo profilo di gara.
        key_prefix distingue i session_state key (es. "race", "race_a", "race_b")
        così più profili possono coesistere in pagina senza sovrascriversi.
        Gli slider sono in minuti in UI, ma la funzione ritorna sempre le ore
        (t_z2, t_z3, t_z4, t_z5), unità usata dal resto del calcolo.
        max_minutes_by_zone (opzionale, da sezione 2️⃣/3️⃣) alimenta sia il preset
        "Massimo" sia il bottone "Massimo" affiancato a ciascuno slider.
        Ritorna: t_z2, t_z3, t_z4, t_z5, use_cauto, preset, profile_name
        (preset è il nome tecnico che governa la logica, profile_name è
        l'etichetta scelta dall'utente e usata solo per la visualizzazione)
        """
        if title:
            st.markdown(f"**{title}**")

        max_minutes_by_zone = max_minutes_by_zone or {}

        preset = st.radio(
            "Profilo di gara:",
            ["Personalizzato", "Cauto", "Conservativo", "Coraggioso", "Competitivo", "Massimo"],
            horizontal=True,
            key=f"{key_prefix}_profile_preset",
        )

        # Applica i valori preset SOLO quando il profilo cambia, così l'utente può
        # comunque modificare gli slider liberamente dopo la selezione (anche in
        # "Personalizzato", dove semplicemente non si sovrascrive nulla).
        last_preset_key = f"_last_{key_prefix}_preset"
        name_key = f"{key_prefix}_profile_name"
        prev_preset = st.session_state.get(last_preset_key)

        if prev_preset != preset:
            # Il nome segue il preset finché l'utente non lo personalizza: se il
            # nome corrente è ancora quello del preset precedente (o è vuoto) lo
            # si aggiorna, altrimenti si rispetta quello scritto a mano.
            current_name = (st.session_state.get(name_key) or "").strip()
            if current_name in ("", prev_preset or ""):
                st.session_state[name_key] = preset

            if preset in PRESET_VALUES:
                st.session_state[f"{key_prefix}_t_z2"] = int(round(PRESET_VALUES[preset]["t_z2"] * 60))
                st.session_state[f"{key_prefix}_t_z3"] = int(round(PRESET_VALUES[preset]["t_z3"] * 60))
                st.session_state[f"{key_prefix}_t_z4"] = int(round(PRESET_VALUES[preset]["t_z4"] * 60))
                st.session_state[f"{key_prefix}_t_z5"] = int(round(PRESET_VALUES[preset]["t_z5"] * 60))
            elif preset == "Massimo":
                # Zona 1 esclusa di proposito: il tempo in Z1 non è un vincolo
                # orario impostabile, è ciò che resta dopo aver coperto Z2-Z5.
                for zona, state_suffix, _ in ZONE_SLIDER_SPECS:
                    st.session_state[f"{key_prefix}_{state_suffix}"] = int(
                        round(max_minutes_by_zone.get(zona, 0))
                    )
            st.session_state[last_preset_key] = preset

        # Il campo editabile vive nella scheda 📄 Export: qui si legge soltanto il
        # valore in session_state. tab_export viene eseguita DOPO questa scheda
        # nell'ordine dello script, quindi la scrittura sopra (nome che segue il
        # preset) avviene sempre prima che il widget venga istanziato: nessun
        # conflitto con la protezione di Streamlit sulle chiavi dei widget.
        profile_name = (st.session_state.get(name_key) or "").strip() or preset
        st.caption(f"Nome profilo: **{profile_name}** — rinominabile nella scheda 📄 Export.")

        if preset == "Conservativo":
            st.info("ℹ️ L'atleta correrà unicamente in Zona 1")
        elif preset == "Cauto":
            st.info(
                "ℹ️ L'atleta correrà unicamente in Zona 1, con EFS Zona 1 ridotta del "
                f"{int((1 - CAUTO_Z1_FACTOR) * 100)}% per margine di sicurezza"
            )
        elif preset == "Massimo":
            if max_minutes_by_zone:
                st.info("ℹ️ Ogni zona (Z2-Z5) è impostata al minutaggio massimo registrato dall'atleta.")
            else:
                st.warning(
                    "⚠️ Nessun minutaggio massimo disponibile (sezione 2️⃣/3️⃣ vuota): gli slider "
                    "restano a 0. Carica dei file .fit o importa un CSV di riferimento."
                )

        cols = st.columns(4)
        minutes_by_zone = {}
        for (zona, state_suffix, label), col in zip(ZONE_SLIDER_SPECS, cols):
            state_key = f"{key_prefix}_{state_suffix}"
            max_min = max_minutes_by_zone.get(zona)

            if col.button(f"⏫ Massimo {zona}", key=f"{state_key}_max_btn",
                          disabled=(max_min is None), use_container_width=True):
                st.session_state[state_key] = int(round(max_min))

            slider_max = max(900, int(round(max_min)) if max_min else 0)
            minutes_by_zone[zona] = col.slider(
                label, min_value=0, max_value=slider_max, step=15, key=state_key
            )

        t_z2, t_z3, t_z4, t_z5 = (
            minutes_by_zone["Z2"] / 60, minutes_by_zone["Z3"] / 60,
            minutes_by_zone["Z4"] / 60, minutes_by_zone["Z5"] / 60,
        )

        use_cauto = (preset == "Cauto")
        return t_z2, t_z3, t_z4, t_z5, use_cauto, preset, profile_name


    compare_mode = st.checkbox("🔀 Confronta due profili (calcola un range)", key="race_compare_mode")

    if not compare_mode:
        t_z2, t_z3, t_z4, t_z5, use_cauto, preset, name_single = render_profile_selector(
            "race", max_minutes_by_zone=max_minutes_by_zone
        )
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            t_z2_a, t_z3_a, t_z4_a, t_z5_a, use_cauto_a, preset_a, name_a = render_profile_selector(
                "race_a", "Profilo A", max_minutes_by_zone=max_minutes_by_zone
            )
        with col_b:
            t_z2_b, t_z3_b, t_z4_b, t_z5_b, use_cauto_b, preset_b, name_b = render_profile_selector(
                "race_b", "Profilo B", max_minutes_by_zone=max_minutes_by_zone
            )

        # I nomi finiscono come suffisso nei nomi di colonna del race plan: se
        # coincidono, le colonne di B sovrascriverebbero quelle di A nel merge.
        if name_a == name_b:
            name_a, name_b = f"{name_a} (A)", f"{name_b} (B)"

    st.divider()

    # --- 5. Calcolo previsione -----------------------------------------------
    st.subheader("5️⃣ Previsione tempo gara")


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
            st.error("⚠️ Carica prima un file gara valido (punto 1️⃣ di questa scheda).")
        elif not zone_efs_ref:
            st.error("⚠️ Nessun dato EFS di riferimento disponibile (punto 2️⃣ di questa scheda).")
        elif not compare_mode:
            result = compute_race_time(t_z2, t_z3, t_z4, t_z5, zone_efs_ref, race_efd_m, use_cauto)
            if result["error"]:
                st.error(f"⚠️ {result['error']}")
                st.session_state["prediction_ready"] = False
            else:
                display_result(result, label=name_single)
                st.session_state["prediction_ready"] = True
                st.session_state["prediction_compare_mode"] = False
                st.session_state["result_single"] = result
                st.session_state["profile_label_single"] = name_single
                # il piano eventualmente già calcolato si riferisce alla previsione
                # precedente: va buttato, altrimenti resterebbe a schermo incoerente
                st.session_state.pop("race_plan_data", None)
        else:
            result_a = compute_race_time(t_z2_a, t_z3_a, t_z4_a, t_z5_a, zone_efs_ref, race_efd_m, use_cauto_a)
            result_b = compute_race_time(t_z2_b, t_z3_b, t_z4_b, t_z5_b, zone_efs_ref, race_efd_m, use_cauto_b)

            if result_a["error"]:
                st.error(f"⚠️ {name_a}: {result_a['error']}")
            if result_b["error"]:
                st.error(f"⚠️ {name_b}: {result_b['error']}")

            if not result_a["error"] and not result_b["error"]:
                times_h = [result_a["race_time_h"], result_b["race_time_h"]]
                low_h, high_h = min(times_h), max(times_h)
                st.metric(
                    "⏱️ Range tempo di gara previsto",
                    f"{seconds_to_hhmm(low_h * 3600)} – {seconds_to_hhmm(high_h * 3600)}",
                )
                st.caption(
                    f"{name_a}: {seconds_to_hhmm(result_a['race_time_h'] * 3600)}  ·  "
                    f"{name_b}: {seconds_to_hhmm(result_b['race_time_h'] * 3600)}"
                )
                st.session_state["prediction_ready"] = True
                st.session_state["prediction_compare_mode"] = True
                st.session_state["result_a"] = result_a
                st.session_state["result_b"] = result_b
                st.session_state["preset_a_label"] = name_a
                st.session_state["preset_b_label"] = name_b
                st.session_state.pop("race_plan_data", None)
            else:
                st.session_state["prediction_ready"] = False

            if not result_a["error"]:
                display_result(result_a, label=name_a)
            if not result_b["error"]:
                display_result(result_b, label=name_b)




with tab_plan:
    st.header("🗺️ Race Plan")

    prediction_ready = st.session_state.get("prediction_ready", False)

    if not prediction_ready:
        st.info(
            "ℹ️ Calcola prima una previsione nella scheda **🎯 Gara e profilo**, "
        "poi qui potrai costruire il race plan."
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
                st.error("⚠️ Profilo del percorso non disponibile: ricarica il file gara nella scheda **🎯 Gara e profilo**.")
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
                        # i segmenti sono cambiati: il piano precedente non vale più
                        st.session_state.pop("race_plan_data", None)
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
                    # Il piano si calcola quando si preme il bottone, ma il
                    # risultato va in session_state: la UI di export (checkbox di
                    # scelta colonne + download) deve sopravvivere ai rerun, cosa
                    # impossibile se vive dentro il blocco del bottone.
                    if st.button("🎯 Calcola Race Plan definitivo", key="compute_race_plan_button"):
                        compare = st.session_state.get("prediction_compare_mode", False)

                        if not compare:
                            result = st.session_state.get("result_single")
                            label_single = st.session_state.get("profile_label_single", "gara")
                            plan_df, calib = build_plan_for_profile(
                                result, "gara", race_profile, laps_df
                            )
                            plan_df = plan_df.rename(columns={
                                "Tempo segmento (gara)": "Tempo segmento",
                                "Tempo di gara (gara)": "Tempo di gara",
                            })
                            display_cols = ["Lap", "Km partenza", "Km arrivo", "Distanza (km)",
                                            "D+ (m)", "D- (m)", "Velocità (km/h)",
                                            "Tempo segmento", "Tempo di gara"]
                            fig = plot_elevation_profile_with_laps(
                                race_profile, laps_df,
                                time_labels=[[t] for t in plan_df["Tempo di gara"]],
                            )
                            st.session_state["race_plan_data"] = {
                                "compare": False,
                                "plan_df": plan_df,
                                "display_cols": display_cols,
                                "chart_png": fig_to_png_bytes(fig),
                                "calib_caption": f"Fattore di calibrazione applicato: ×{calib:.3f}",
                                "meta": [
                                    f"Profilo: {label_single}",
                                    f"Tempo previsto: {seconds_to_hhmm(result['race_time_h'] * 3600)}",
                                    f"EFD totale: {result['total_efd_km']:.2f} km",
                                ],
                                "file_stem": "race_plan",
                            }
                            plt.close(fig)
                        else:
                            result_a = st.session_state.get("result_a")
                            result_b = st.session_state.get("result_b")
                            label_a = st.session_state.get("preset_a_label", "A")
                            label_b = st.session_state.get("preset_b_label", "B")

                            plan_a, calib_a = build_plan_for_profile(
                                result_a, label_a, race_profile, laps_df
                            )
                            plan_b, calib_b = build_plan_for_profile(
                                result_b, label_b, race_profile, laps_df
                            )

                            merged = plan_a[["Lap", "Km partenza", "Km arrivo",
                                             "Distanza (km)", "D+ (m)", "D- (m)"]].copy()
                            merged[f"Velocità ({label_a}) km/h"] = plan_a["Velocità (km/h)"]
                            merged[f"Tempo segmento ({label_a})"] = plan_a[f"Tempo segmento ({label_a})"]
                            merged[f"Tempo di gara ({label_a})"] = plan_a[f"Tempo di gara ({label_a})"]
                            merged[f"Velocità ({label_b}) km/h"] = plan_b["Velocità (km/h)"]
                            merged[f"Tempo segmento ({label_b})"] = plan_b[f"Tempo segmento ({label_b})"]
                            merged[f"Tempo di gara ({label_b})"] = plan_b[f"Tempo di gara ({label_b})"]

                            fig = plot_elevation_profile_with_laps(
                                race_profile, laps_df,
                                time_labels=[
                                    [ta, tb] for ta, tb in zip(
                                        plan_a[f"Tempo di gara ({label_a})"],
                                        plan_b[f"Tempo di gara ({label_b})"],
                                    )
                                ],
                                legend_labels=[label_a, label_b],
                            )
                            st.session_state["race_plan_data"] = {
                                "compare": True,
                                "plan_df": merged,
                                "display_cols": list(merged.columns),
                                "chart_png": fig_to_png_bytes(fig),
                                "calib_caption": (
                                    f"Fattore di calibrazione: {label_a} ×{calib_a:.3f} · "
                                    f"{label_b} ×{calib_b:.3f}"
                                ),
                                "meta": [
                                    f"{label_a}: {seconds_to_hhmm(result_a['race_time_h'] * 3600)}",
                                    f"{label_b}: {seconds_to_hhmm(result_b['race_time_h'] * 3600)}",
                                    f"EFD totale: {result_a['total_efd_km']:.2f} km",
                                ],
                                "file_stem": "race_plan_confronto",
                            }
                            plt.close(fig)

    # ---------------------------------------------------------------
    # Risultato del Race Plan + export (fuori dal blocco del bottone)
    # ---------------------------------------------------------------
    plan_state = st.session_state.get("race_plan_data")

    if plan_state:
        plan_df = plan_state["plan_df"]
        display_cols = plan_state["display_cols"]

        st.caption(plan_state["calib_caption"])
        st.dataframe(
            plan_df[display_cols].style.format({
                c: f for c, f in PLAN_COL_FORMATS.items() if c in display_cols
            } | {
                c: "{:.1f}" for c in display_cols if "Velocità" in c
            }),
            use_container_width=True, hide_index=True,
        )

        st.image(plan_state["chart_png"], use_container_width=True)

with tab_export:
    st.header("📄 Esporta il report")

    # --- Intestazione ---------------------------------------------------------
    # Questi campi stanno FUORI dal controllo su plan_state di proposito: i nomi
    # dei profili vengono letti da render_profile_selector (scheda 🎯 Gara e
    # profilo), che gira prima di questa, quindi devono essere compilabili anche
    # quando un race plan non esiste ancora. Lo stesso vale per il nome atleta,
    # usato anche dal CSV delle zone nella scheda ❤️ Zone FC.
    st.subheader("Intestazione del report")

    c_ath, c_race = st.columns(2)
    c_ath.text_input(
        "Nome dell'atleta *",
        key="athlete_name",
        max_chars=60,
        placeholder="es. Davide Grazielli",
        help="Obbligatorio: compare nell'intestazione del report e nel nome dei file.",
    )
    c_race.text_input(
        "Nome della gara *",
        key="race_name",
        max_chars=80,
        placeholder="es. UTMB 2026",
        help="Obbligatorio: titolo del report e parte del nome dei file.",
    )

    _profile_help = (
        "Il nome entra nelle colonne del piano: dopo averlo cambiato ricalcola "
        "il race plan per vederlo aggiornato."
    )
    if st.session_state.get("race_compare_mode", False):
        c_pa, c_pb = st.columns(2)
        c_pa.text_input("Nome del profilo A", key="race_a_profile_name",
                        max_chars=40, help=_profile_help)
        c_pb.text_input("Nome del profilo B", key="race_b_profile_name",
                        max_chars=40, help=_profile_help)
    else:
        st.text_input("Nome del profilo", key="race_profile_name",
                      max_chars=40, help=_profile_help)

    athlete = (st.session_state.get("athlete_name") or "").strip()
    race_title = (st.session_state.get("race_name") or "").strip()

    st.divider()

    plan_state = st.session_state.get("race_plan_data")

    if not plan_state:
        st.info(
            "ℹ️ Nessun race plan da esportare: costruiscine uno nella scheda "
            "**🗺️ Race Plan**."
        )
    elif not athlete or not race_title:
        mancanti = [
            etichetta for etichetta, valore in
            (("Nome dell'atleta", athlete), ("Nome della gara", race_title))
            if not valore
        ]
        st.warning(
            "⚠️ Compila i campi obbligatori per generare gli export: "
            + ", ".join(f"**{m}**" for m in mancanti) + "."
        )
    else:
        plan_df = plan_state["plan_df"]
        display_cols = plan_state["display_cols"]

        # Nome file unico per CSV e PDF: atleta_gara_raceplan
        file_stem = f"{slugify_name(athlete)}_{slugify_name(race_title)}_raceplan"

        # --- CSV --------------------------------------------------------------
        st.subheader("Dati (CSV)")
        st.download_button(
            "📥 Scarica race plan (CSV)",
            data=plan_df[display_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"{file_stem}.csv",
            mime="text/csv",
            key="dl_race_plan_csv",
        )
        st.caption(
            "Scarica il CSV per utilizzarlo in futuro per analisi e piani nutrizionali"
        )

        st.divider()

        # --- PDF --------------------------------------------------------------
        st.subheader("Report (PDF)")

        if not REPORTLAB_OK:
            st.warning(
                "⚠️ Manca la libreria `reportlab`, necessaria per generare il PDF. "
                "Installala con `uv add reportlab` (oppure aggiungila a "
                "`pyproject.toml`) e riavvia l'app."
            )
        else:
            st.caption(
                "Scegli quali colonne finiscono nella tabella del PDF: "
                "meno colonne significa caratteri più grandi e un foglio "
                "più leggibile da tasca."
            )

            checkbox_cols = st.columns(3)
            selected_cols = []
            for i, col_name in enumerate(display_cols):
                # "Lap" preselezionata e sempre utile come riferimento
                default = True
                with checkbox_cols[i % 3]:
                    if st.checkbox(col_name, value=default,
                                   key=f"pdfcol_{col_name}"):
                        selected_cols.append(col_name)

            include_chart = st.checkbox(
                "Includi il profilo altimetrico", value=True,
                key="pdf_include_chart",
            )

            coach_comment = st.text_area(
                "Commento del coach",
                key="coach_comment",
                height=120,
                max_chars=1500,
                placeholder="Note di strategia, alimentazione, punti critici, "
                            "cambi scarpe ai ristori…",
                help="Testo libero: finisce in un riquadro in cima al report. "
                     "Gli a capo vengono mantenuti.",
            )

            pdf_meta = [f"Atleta: {athlete}"] + list(plan_state["meta"])

            if not selected_cols:
                st.info("Seleziona almeno una colonna per generare il PDF.")
            else:
                try:
                    pdf_bytes = build_race_plan_pdf(
                        table_df=format_plan_for_export(plan_df[selected_cols]),
                        chart_png=plan_state["chart_png"] if include_chart else None,
                        title=race_title,
                        meta_lines=pdf_meta,
                        comment=coach_comment,
                        note=plan_state["calib_caption"],
                    )
                    st.download_button(
                        "📄 Scarica Race Plan (PDF)",
                        data=pdf_bytes,
                        file_name=f"{file_stem}.pdf",
                        mime="application/pdf",
                        key="dl_race_plan_pdf",
                        type="primary",
                    )
                except Exception as e:
                    st.error(f"⚠️ Errore durante la generazione del PDF: {e}")

with tab_compare:
    st.header("🆚 Confronto Race Plan previsto vs Gara reale")
    st.caption(
        "Carica il race plan esportato dalla scheda **📄 Export** (CSV) e il file .fit della "
        "gara realmente corsa: confronta i tempi di passaggio previsti vs reali ai lap, e lo "
        "scostamento EFS realmente osservato in gara rispetto alla curva teorica che ha "
        "generato il piano."
    )

    bin_width_pct = st.slider(
        "Ampiezza fetta (% dell'EFD totale della gara)",
        1.0, 20.0, 5.0, step=1.0, key="compare_bin_width",
    )
    st.caption(
        "Il decadimento si analizza per fette di EFD **relativo** (% del totale della gara), "
        "la stessa scala su cui è calibrata la curva del modello."
    )

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        plan_csv = st.file_uploader(
            "📋 Race plan previsto (CSV esportato dalla scheda Export)",
            type=["csv"], key="compare_plan_csv",
        )
    with col_up2:
        real_fit_files = st.file_uploader(
            "🏁 File .fit della gara realmente corsa",
            type=["fit", "gz", "zip"], accept_multiple_files=False,
            key="compare_real_fit",
        )

    real_uploads = expand_uploads(
        [real_fit_files] if real_fit_files is not None else [], accept=("fit",)
    )

    if plan_csv is not None and real_uploads:
        _real_name, real_bytes, _real_ext = real_uploads[0]

        try:
            plan_df_cmp = pd.read_csv(plan_csv)
        except Exception as e:
            st.error(f"⚠️ Errore nella lettura del CSV: {e}")
            plan_df_cmp = None

        if plan_df_cmp is not None:
            # Il nome del profilo tra parentesi varia (Competitivo, Profilo A...),
            # e in modalità singola la colonna è semplicemente "Tempo di gara":
            # in entrambi i casi startswith("Tempo di gara") intercetta tutto.
            tempo_gara_cols = [c for c in plan_df_cmp.columns if c.startswith("Tempo di gara")]
            required_base_cols = {"Lap", "Km partenza", "Km arrivo", "Distanza (km)"}
            missing = required_base_cols - set(plan_df_cmp.columns)

            if missing or not tempo_gara_cols:
                st.error(
                    "⚠️ Il CSV non sembra un race plan valido. "
                    + (f"Colonne mancanti: {missing}. " if missing else "")
                    + ("Nessuna colonna 'Tempo di gara (...)' trovata." if not tempo_gara_cols else "")
                )
            else:
                if len(tempo_gara_cols) > 1:
                    tempo_gara_col = st.selectbox(
                        "Più profili trovati nel CSV: quale usare come riferimento?",
                        tempo_gara_cols, key="tempo_gara_col_select",
                    )
                else:
                    tempo_gara_col = tempo_gara_cols[0]

                try:
                    segments_real, summary_real = load_and_process_real_race(
                        real_bytes, smooth_window, resample_step, zones
                    )
                except Exception as e:
                    st.error(f"⚠️ File .fit illeggibile o corrotto: {e}")
                    segments_real = None

                if segments_real is None:
                    st.error("⚠️ Nessun dato GPS valido nel file .fit della gara reale (traccia degenere).")
                else:
                    cum_km_real = segments_real["distance_km"].to_numpy()
                    cum_t_real = np.cumsum(segments_real["dt_s"].to_numpy())
                    total_efd_real_km = float(segments_real["efd_m"].sum() / 1000)
                    total_time_real_s = float(cum_t_real[-1])
                    total_km_real = float(cum_km_real[-1])

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Distanza reale", f"{total_km_real:.2f} km")
                    c2.metric("EFD reale", f"{total_efd_real_km:.2f} km")
                    c3.metric("Tempo reale totale", seconds_to_hhmm(total_time_real_s))

                    # --- tabella di confronto ai lap ---
                    rows_cmp = []
                    prev_t_prev_s, prev_t_real_s = 0.0, 0.0
                    for _, lap in plan_df_cmp.iterrows():
                        km_end = float(lap["Km arrivo"])
                        t_prev_s = hhmm_to_seconds(lap[tempo_gara_col])

                        if km_end <= total_km_real:
                            t_real_s = float(np.interp(km_end, cum_km_real, cum_t_real))
                            oltre_percorso = False
                        else:
                            # l'atleta non ha (ancora) raggiunto questo km nel file reale
                            t_real_s = np.nan
                            oltre_percorso = True

                        delta_s = (t_real_s - t_prev_s) if not np.isnan(t_real_s) else np.nan
                        seg_prev_s = t_prev_s - prev_t_prev_s
                        seg_real_s = (t_real_s - prev_t_real_s) if not np.isnan(t_real_s) else np.nan

                        rows_cmp.append({
                            "Lap": lap["Lap"],
                            "Km arrivo": km_end,
                            "Tempo previsto (cum.)": seconds_to_hhmm(t_prev_s),
                            "Tempo reale (cum.)": seconds_to_hhmm(t_real_s) if not oltre_percorso else "—",
                            "Δ cumulato": (
                                f"{'+' if delta_s >= 0 else '−'}{seconds_to_hhmm(abs(delta_s))}"
                                if not np.isnan(delta_s) else "—"
                            ),
                            "Tempo segmento previsto": seconds_to_hhmm(seg_prev_s),
                            "Tempo segmento reale": seconds_to_hhmm(seg_real_s) if not oltre_percorso else "—",
                            "_delta_s": delta_s,
                        })
                        prev_t_prev_s = t_prev_s
                        if not oltre_percorso:
                            prev_t_real_s = t_real_s

                    cmp_table = pd.DataFrame(rows_cmp)

                    st.subheader("📋 Tempi di passaggio: previsto vs reale")
                    display_cmp = cmp_table.drop(columns=["_delta_s"])
                    st.dataframe(display_cmp, use_container_width=True, hide_index=True)
                    st.download_button(
                        "📥 Scarica confronto (CSV)",
                        data=display_cmp.to_csv(index=False).encode("utf-8"),
                        file_name="confronto_race_plan_vs_reale.csv",
                        mime="text/csv",
                        key="dl_confronto_csv",
                    )

                    valid_delta = cmp_table.dropna(subset=["_delta_s"])
                    if not valid_delta.empty:
                        final_delta_s = valid_delta["_delta_s"].iloc[-1]
                        if final_delta_s > 0:
                            st.warning(
                                f"🐢 L'atleta è arrivato **{seconds_to_hhmm(final_delta_s)} più lento** "
                                "del previsto (all'ultimo lap raggiunto)."
                            )
                        elif final_delta_s < 0:
                            st.success(
                                f"🚀 L'atleta è arrivato **{seconds_to_hhmm(abs(final_delta_s))} più veloce** "
                                "del previsto (all'ultimo lap raggiunto)."
                            )
                        else:
                            st.info("⏱️ Tempo esattamente in linea con la previsione.")

                        gap_min = valid_delta["_delta_s"].to_numpy() / 60.0
                        x_gap = valid_delta["Km arrivo"].to_numpy()
                        lap_names_gap = valid_delta["Lap"].to_numpy()
                        delta_s_gap = valid_delta["_delta_s"].to_numpy()
                        bar_labels = [
                            f"{'+' if s >= 0 else '−'}{seconds_to_hhmm(abs(s))}" for s in delta_s_gap
                        ]
                        bar_colors = ["#c1440e" if v >= 0 else "#2a9d8f" for v in gap_min]

                        fig_gap = go.Figure()
                        fig_gap.add_trace(go.Bar(
                            x=x_gap,
                            y=gap_min,
                            marker_color=bar_colors,
                            text=bar_labels,
                            textposition="outside",
                            customdata=lap_names_gap,
                            hovertemplate=(
                                "<b>%{customdata}</b><br>Km %{x:.1f}<br>Scarto: %{text}<extra></extra>"
                            ),
                        ))
                        fig_gap.add_hline(y=0, line_color="#333333", line_width=1)
                        fig_gap.update_layout(
                            title="Scarto cumulato: tempo reale vs tempo previsto",
                            xaxis_title="Km",
                            yaxis_title="Scarto (min)",
                            showlegend=False,
                            height=420,
                            margin=dict(t=60, b=40, l=40, r=20),
                            uniformtext_minsize=10,
                        )
                        st.caption("🟠 rosso = più lento del previsto · 🟢 verde = più veloce del previsto")
                        st.plotly_chart(fig_gap, use_container_width=True)

                    # --- modello teorico vs decadimento osservato ---
                    st.divider()
                    st.subheader("📐 Decadimento EFS: modello teorico vs gara reale")
                    st.caption(
                        "EFD accumulato in % del totale della gara reale, scostamento dell'EFS di "
                        "fetta rispetto all'EFS medio dell'INTERA gara reale, sovrapposto alla curva "
                        "del modello a bucket (`predict_delta_efs_blended`) — la stessa che ha "
                        "generato i tempi di passaggio del piano."
                    )

                    if total_time_real_s <= 0 or total_efd_real_km <= 0:
                        st.info("Dati reali insufficienti per calcolare il decadimento.")
                    else:
                        dt_arr_r = segments_real["dt_s"].to_numpy()
                        efd_arr_r = segments_real["efd_m"].to_numpy()
                        avg_efs_real_ms = efd_arr_r.sum() / dt_arr_r.sum()

                        cum_efd_before_r = np.cumsum(efd_arr_r) - efd_arr_r
                        mid_efd_r = cum_efd_before_r + efd_arr_r / 2.0
                        efd_pct_arr_r = mid_efd_r / efd_arr_r.sum() * 100.0
                        bin_idx_r = (efd_pct_arr_r // bin_width_pct).astype(int)

                        bin_df_r = pd.DataFrame({"bin_idx": bin_idx_r, "dt_s": dt_arr_r, "efd_m": efd_arr_r})
                        grouped_r = bin_df_r.groupby("bin_idx", as_index=False).sum()
                        grouped_r = grouped_r[grouped_r["dt_s"] > 0]

                        real_decadence_rows = []
                        for _, row in grouped_r.iterrows():
                            bin_efs_ms_r = row["efd_m"] / row["dt_s"]
                            dev_pct_r = (bin_efs_ms_r - avg_efs_real_ms) / avg_efs_real_ms * 100.0
                            efd_pct_center_r = (row["bin_idx"] + 0.5) * bin_width_pct
                            real_decadence_rows.append({
                                "efd_pct_accum": efd_pct_center_r,
                                "deviation_pct_reale": dev_pct_r,
                            })
                        real_decadence_df = pd.DataFrame(real_decadence_rows)

                        x_line_r = np.linspace(0, 100, 200)
                        theoretical_line_r = np.array([
                            predict_delta_efs_blended(p, total_efd_real_km) for p in x_line_r
                        ])

                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(
                            x=x_line_r, y=theoretical_line_r, mode="lines",
                            name=f"Teorica (modello, {total_efd_real_km:.0f} km EFD)",
                            line=dict(color="#1f4e79", width=3),
                        ))
                        fig3.add_trace(go.Scatter(
                            x=real_decadence_df["efd_pct_accum"],
                            y=real_decadence_df["deviation_pct_reale"],
                            mode="markers", name="Reale (fette di gara)",
                            marker=dict(size=9, color="#c1440e"),
                        ))

                        # Fit esplorativo sui punti di QUESTA gara. n_files è forzato
                        # a MIN_FILES_FOR_CUBIC per sbloccare tutte le forme: non è il
                        # vero numero di file indipendenti (è 1), ma qui si vuole
                        # lasciare al fit la libertà di descrivere la forma osservata.
                        real_best_fit = None
                        if len(real_decadence_df) >= 6:
                            real_fit_candidates = fit_candidates(
                                real_decadence_df["efd_pct_accum"].to_numpy(),
                                real_decadence_df["deviation_pct_reale"].to_numpy(),
                                n_files=MIN_FILES_FOR_CUBIC,
                            )
                            real_best_name = best_candidate(real_fit_candidates)
                            real_best_fit = real_fit_candidates[real_best_name]
                            real_fitted_line = real_best_fit["predict"](x_line_r)

                            fig3.add_trace(go.Scatter(
                                x=x_line_r, y=real_fitted_line, mode="lines",
                                name=f"Fit reale — {real_best_name}",
                                line=dict(color="#c1440e", width=2, dash="dash"),
                            ))

                        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig3.update_layout(
                            title="Modello del piano vs decadimento realmente osservato",
                            xaxis_title="EFD accumulato (% della gara reale)",
                            yaxis_title="Scostamento EFS (%)",
                            height=450,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        )
                        st.plotly_chart(fig3, use_container_width=True)

                        if real_best_fit is not None:
                            st.caption(
                                f"Equazione fittata sui dati reali di questa gara — "
                                f"miglior candidato: **{real_best_name}** "
                                f"(R² = {real_best_fit['r2']:.3f} · adj. R² = {real_best_fit['adj_r2']:.3f}). "
                                "Con un solo file i bin non sono osservazioni indipendenti: leggi il fit "
                                "come descrizione della forma osservata in QUESTA gara, non come modello "
                                "statisticamente robusto — un adj. R² molto alto è spesso solo "
                                "overfitting sui pochi punti disponibili."
                            )
                            st.latex(real_best_fit["equation"])
                            with st.expander("Vedi tutte le equazioni candidate fittate sui dati reali"):
                                ranked_real = sorted(
                                    real_fit_candidates.items(),
                                    key=lambda kv: (kv[1]["adj_r2"] if not np.isnan(kv[1]["adj_r2"]) else -np.inf),
                                    reverse=True,
                                )
                                for cname, c in ranked_real:
                                    star = " 🏆 **migliore**" if cname == real_best_name else ""
                                    st.markdown(f"{cname}{star}")
                                    st.latex(c["equation"])
                                    st.caption(f"R² = {c['r2']:.3f} · adj. R² = {c['adj_r2']:.3f}")
                        else:
                            st.caption(
                                "Troppo pochi bin per fittare un'equazione affidabile sui dati reali "
                                "(servono almeno 6 fette: riduci l'ampiezza fetta o carica un file più lungo)."
                            )

                        with st.expander("Dati decadimento reale (debug/export)"):
                            st.dataframe(real_decadence_df, use_container_width=True, hide_index=True)
                            st.download_button(
                                "📥 Scarica decadimento reale (CSV)",
                                data=real_decadence_df.to_csv(index=False).encode("utf-8"),
                                file_name="decadimento_efs_reale.csv",
                                mime="text/csv",
                                key="dl_decadimento_reale_csv",
                            )
    elif plan_csv is not None or real_fit_files is not None:
        st.info("Carica sia il CSV del race plan sia il file .fit della gara reale per procedere.")