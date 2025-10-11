import numpy as np
import requests
import streamlit as st

def haversine_deg(lat1, lon1, lat2, lon2):
    # all inputs arrays or pandas Series in degrees
    R = 6371.0  # Earth radius in km
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # distance in km


def map_code_to_category(code):
    # A simplified mapping — adjust or expand as needed
    if code == 0:
        return "Sunny"
    elif code in [1, 2, 3]:
        return "Cloudy"
    elif code in [45, 48]:
        return "Fog"
    elif code in [95, 96, 99]:
        return "Stormy"
    # You may define codes for sand, windy, etc., depending on data
    else:
        return "Unknown"

def get_weather_category(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    data = resp.json()
    cw = data.get("current_weather", {})
    code = cw.get("weathercode")
    return map_code_to_category(code)

