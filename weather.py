import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_FIELDS = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,"
    "precipitation,weather_code,wind_speed_10m,wind_gusts_10m"
)
HOURLY_FIELDS = "precipitation_probability,uv_index"

TIMEOUT_SECONDS = 10


def geocode(location_query: str):

    if not location_query or not location_query.strip():
        return None, None, None, "No location was given."

    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"name": location_query.strip(), "count": 5},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return None, None, None, f"Geocoding service unreachable: {e}"

    results = data.get("results")
    if not results:
        return None, None, None, f"No location found matching '{location_query}'."

    top = results[0]
    return top["latitude"], top["longitude"], top.get("name", location_query), None


def fetch_weather(latitude: float, longitude: float):

    if latitude is None or longitude is None:
        return None, "No coordinates to fetch weather for."

    try:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": CURRENT_FIELDS,
                "hourly": HOURLY_FIELDS,
                "timezone": "auto",
            },
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return None, f"Weather service unreachable: ({type(e).__name__})"

    current = data.get("current")
    if not current:
        return None, "Weather service returned no current conditions."

    hourly = data.get("hourly") or {}
    current_time = current.get("time")
    precip_prob = None
    uv_index = None
    hourly_times = hourly.get("time", [])
    current_hour = current_time[:13] + ":00" if current_time else None
    if current_hour in hourly_times:
        idx = hourly_times.index(current_hour)
        probs = hourly.get("precipitation_probability", [])
        uvs = hourly.get("uv_index", [])
        if idx < len(probs):
            precip_prob = probs[idx]
        if idx < len(uvs):
            uv_index = uvs[idx]

    facts = {
        "time": current_time,
        "temperature_c": current.get("temperature_2m"),
        "apparent_temperature_c": current.get("apparent_temperature"),
        "relative_humidity_pct": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation"),
        "precipitation_probability_pct": precip_prob,
        "weather_code": current.get("weather_code"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "wind_gusts_kmh": current.get("wind_gusts_10m"),
        "uv_index": uv_index,
    }
    return facts, None