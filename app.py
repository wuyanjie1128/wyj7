import json
import math
from datetime import datetime, timedelta, date

import pandas as pd
import requests
import plotly.express as px
import pytz
import streamlit as st
from dateutil import tz

# --------------- App Config ---------------
st.set_page_config(
    page_title="Open-Meteo Interactive Weather Dashboard",
    page_icon="⛅",
    layout="wide",
)

# --------------- Utilities ---------------
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_OPTIONS = {
    "Temperature (2m)": "temperature_2m",
    "Relative Humidity (2m)": "relative_humidity_2m",
    "Precipitation": "precipitation",
    "Wind Speed (10m)": "wind_speed_10m",
}
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "sunrise",
    "sunset",
    "wind_speed_10m_max",
]

def _safe_get(d, *keys, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d

@st.cache_data(show_spinner=False, ttl=60 * 30)
def geocode_place(name: str, count: int = 10):
    """Geocode a place name using Open-Meteo's free geocoding API."""
    if not name:
        return []
    try:
        resp = requests.get(
            OPEN_METEO_GEOCODE_URL,
            params={"name": name, "count": count, "language": "en", "format": "json"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", []) or []
        # Normalize into a list of options
        normalized = []
        for r in results:
            label_bits = [r.get("name", "")]
            admin1 = r.get("admin1")
            country = r.get("country")
            if admin1:
                label_bits.append(admin1)
            if country:
                label_bits.append(country)
            label = ", ".join([b for b in label_bits if b])
            normalized.append(
                {
                    "label": label,
                    "latitude": r.get("latitude"),
                    "longitude": r.get("longitude"),
                    "timezone": r.get("timezone", "UTC"),
                }
            )
        return normalized
    except Exception as e:
        # Return empty rather than breaking the UI
        st.warning(f"Geocoding failed: {e}")
        return []

@st.cache_data(show_spinner=False, ttl=60 * 10)
def fetch_forecast(
    latitude: float,
    longitude: float,
    timezone_str: str = "auto",
    start_date: date | None = None,
    end_date: date | None = None,
):
    """Fetch forecast from Open-Meteo; returns dict or None."""
    if latitude is None or longitude is None:
        return None
    # Open-Meteo can auto-detect timezone with "auto"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone_str if timezone_str else "auto",
        "hourly": ",".join(HOURLY_OPTIONS.values()),
        "daily": ",".join(DAILY_VARS),
    }

    # Bound date range (Open-Meteo supports forecast for upcoming days; past data via archive API)
    if start_date:
        params["start_date"] = start_date.isoformat()
    if end_date:
        params["end_date"] = end_date.isoformat()

    try:
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data
    except Exception as e:
        st.error(f"Forecast request failed: {e}")
        return None

def c_to_f(c):
    return c * 9 / 5 + 32

def mm_to_in(mm):
    return mm / 25.4

def kmh_to_mph(kmh):
    return kmh * 0.621371

def build_hourly_df(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for label, key in HOURLY_OPTIONS.items():
        if key in hourly:
            df[key] = hourly.get(key)
    return df

def build_daily_df(payload: dict) -> pd.DataFrame:
    daily = payload.get("daily", {})
    times = daily.get("time", [])
    df = pd.DataFrame({"date": pd.to_datetime(times).date})
    for k in DAILY_VARS:
        if k in daily:
            df[k] = daily.get(k)
    return df

def user_tz():
    # Fallback to system tz if Streamlit cannot infer
    try:
        return tz.tzlocal()
    except Exception:
        return tz.gettz("UTC")

# --------------- Sidebar Controls ---------------
with st.sidebar:
    st.title("🔧 Controls")

    st.write("Search for a place or enter coordinates:")
    place_query = st.text_input("Place name (city, landmark, etc.)", value="Seoul")
    found = geocode_place(place_query) if place_query else []

    selected = None
    if found:
        labels = [opt["label"] for opt in found]
        choice = st.selectbox("Choose a match", labels, index=0)
        selected = next((o for o in found if o["label"] == choice), None)

    st.write("Or set coordinates manually:")
    lat = st.number_input(
        "Latitude", value=float(selected["latitude"]) if selected else 37.5665, format="%.6f"
    )
    lon = st.number_input(
        "Longitude", value=float(selected["longitude"]) if selected else 126.9780, format="%.6f"
    )
    tz_guess = selected["timezone"] if selected and selected.get("timezone") else "auto"

    st.markdown("---")
    st.subheader("Date Range")
    today = datetime.now().date()
    default_start = today
    default_end = today + timedelta(days=6)  # a week span
    date_range = st.date_input(
        "Select start and end dates (forecast horizon only)",
        value=(default_start, default_end),
        min_value=today,
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = date_range, date_range

    st.markdown("---")
    st.subheader("Units")
    use_fahrenheit = st.checkbox("Use °F instead of °C", value=False)
    use_mph = st.checkbox("Use mph instead of km/h", value=False)
    use_inches = st.checkbox("Use inches instead of mm (precip.)", value=False)

    st.markdown("---")
    st.subheader("Hourly Chart")
    hourly_label = st.selectbox("Variable", list(HOURLY_OPTIONS.keys()), index=0)
    smooth = st.checkbox("Apply moving average (3-hour)", value=True)

# --------------- Header ---------------
st.title("Open-Meteo Interactive Weather Dashboard")
st.caption(
    "Live forecast via the free Open-Meteo API. Search for a location, tweak date range and units, and explore hourly & daily data."
)

# --------------- Data Fetch ---------------
with st.spinner("Fetching forecast..."):
    payload = fetch_forecast(
        latitude=lat,
        longitude=lon,
        timezone_str=tz_guess,
        start_date=start_date,
        end_date=end_date,
    )

if not payload:
    st.stop()

# --------------- Location Summary ---------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Location**")
    st.write(f"{place_query if place_query else 'Custom'}, {lat:.4f}, {lon:.4f}")
with col2:
    st.markdown("**Timezone**")
    st.write(_safe_get(payload, "timezone", default="auto"))
with col3:
    st.markdown("**Elevation (m)**")
    st.write(_safe_get(payload, "elevation", default="—"))
with col4:
    st.markdown("**Model**")
    st.write(_safe_get(payload, "generationtime_ms", default="—"))

# --------------- Map ---------------
st.subheader("Map")
st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), size=100)

# --------------- Daily Overview ---------------
st.subheader("Daily Overview")
daily_df = build_daily_df(payload)
if daily_df.empty:
    st.info("No daily data available for the selected range.")
else:
    # Convert units
    if "temperature_2m_max" in daily_df and use_fahrenheit:
        daily_df["temperature_2m_max"] = daily_df["temperature_2m_max"].apply(c_to_f)
        daily_df["temperature_2m_min"] = daily_df["temperature_2m_min"].apply(c_to_f)
    if "precipitation_sum" in daily_df and use_inches:
        daily_df["precipitation_sum"] = daily_df["precipitation_sum"].apply(mm_to_in)
    if "wind_speed_10m_max" in daily_df and use_mph:
        daily_df["wind_speed_10m_max"] = daily_df["wind_speed_10m_max"].apply(kmh_to_mph)

    # Small KPI cards
    k1, k2, k3, k4 = st.columns(4)
    try:
        today_row = daily_df.iloc[0]
        tmax = today_row.get("temperature_2m_max", None)
        tmin = today_row.get("temperature_2m_min", None)
        p_sum = today_row.get("precipitation_sum", None)
        wmax = today_row.get("wind_speed_10m_max", None)

        unit_t = "°F" if use_fahrenheit else "°C"
        unit_p = "in" if use_inches else "mm"
        unit_w = "mph" if use_mph else "km/h"

        with k1:
            st.metric("Today's High", f"{tmax:.1f}{unit_t}" if tmax is not None else "—")
        with k2:
            st.metric("Today's Low", f"{tmin:.1f}{unit_t}" if tmin is not None else "—")
        with k3:
            st.metric("Today's Precip.", f"{p_sum:.2f} {unit_p}" if p_sum is not None else "—")
        with k4:
            st.metric("Max Wind", f"{wmax:.1f} {unit_w}" if wmax is not None else "—")
    except Exception:
        pass

    # Daily table
    nicer = daily_df.copy()
    nicer = nicer.rename(
        columns={
            "temperature_2m_max": f"Tmax ({'°F' if use_fahrenheit else '°C'})",
            "temperature_2m_min": f"Tmin ({'°F' if use_fahrenheit else '°C'})",
            "precipitation_sum": f"Precip. ({'in' if use_inches else 'mm'})",
            "wind_speed_10m_max": f"Wind Max ({'mph' if use_mph else 'km/h'})",
        }
    )
    st.dataframe(nicer, use_container_width=True, hide_index=True)

# --------------- Hourly Exploration ---------------
st.subheader("Hourly Exploration")
hourly_df = build_hourly_df(payload)

if hourly_df.empty:
    st.info("No hourly data available for the selected range.")
else:
    key = HOURLY_OPTIONS[hourly_label]
    plot_df = hourly_df[["time", key]].copy()

    # Convert units
    y_label = hourly_label
    if key == "temperature_2m" and use_fahrenheit:
        plot_df[key] = plot_df[key].apply(c_to_f)
        y_label = "Temperature (°F)"
    elif key == "precipitation" and use_inches:
        plot_df[key] = plot_df[key].apply(mm_to_in)
        y_label = "Precipitation (in)"
    elif key == "wind_speed_10m" and use_mph:
        plot_df[key] = plot_df[key].apply(kmh_to_mph)
        y_label = "Wind Speed (mph)"
    elif key == "temperature_2m":
        y_label = "Temperature (°C)"
    elif key == "precipitation":
        y_label = "Precipitation (mm)"
    elif key == "wind_speed_10m":
        y_label = "Wind Speed (km/h)"

    # Optional smoothing (simple centered moving avg)
    if smooth:
        plot_df["Smoothed"] = plot_df[key].rolling(window=3, min_periods=1, center=True).mean()
        fig = px.line(
            plot_df,
            x="time",
            y=["Smoothed", key],
            labels={"value": y_label, "time": "Time", "variable": "Series"},
            title=f"Hourly {hourly_label}",
        )
    else:
        fig = px.line(
            plot_df,
            x="time",
            y=key,
            labels={key: y_label, "time": "Time"},
            title=f"Hourly {hourly_label}",
        )

    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), height=420)
    st.plotly_chart(fig, use_container_width=True)

# --------------- Raw JSON (Advanced) ---------------
with st.expander("Raw API Response (JSON)", expanded=False):
    st.code(json.dumps(payload, indent=2)[:20000])  # truncate to keep UI snappy

# --------------- Footer ---------------
st.caption(
    "Powered by Open-Meteo (https://open-meteo.com/). Note: Only forecast windows are supported here; "
    "historical data would require the Open-Meteo archive API."
)
