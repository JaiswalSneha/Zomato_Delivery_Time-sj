# map_center = [18.97220285334854, 72.82052102265148]

import streamlit as st
from streamlit_folium import st_folium
import folium
from geopy.distance import geodesic

def location_selector(default_location=[18.97220285334854, 72.82052102265148], max_distance_km=20):
    st.subheader("🗺️ Distance and Traffic Calculation")
    st.text("Step 1: Select Source & Destination")
    st.text("Step 2: View Google Maps Directions with Traffic")

    # Initialize session state
    if 'source' not in st.session_state:
        st.session_state.source = None
    if 'destination' not in st.session_state:
        st.session_state.destination = None

    distance = None

    st.markdown("""
    ##### 🚦 How it works:
    1. Click to select a **Source** location on the map.
    2. Click again to select the **Destination**.
    3. If distance ≤ 20 km, a Google Maps link will show the route **with traffic**.
    """)

    # Create map
    m = folium.Map(location=default_location, zoom_start=12)

    # Add existing markers
    if st.session_state.source:
        folium.Marker(st.session_state.source, tooltip="Source", icon=folium.Icon(color='green')).add_to(m)

    if st.session_state.destination:
        folium.Marker(st.session_state.destination, tooltip="Destination", icon=folium.Icon(color='red')).add_to(m)

    # Show map and capture clicks
    map_data = st_folium(m, height=500, width=800)

    if map_data and map_data.get("last_clicked"):
        clicked = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])

        if not st.session_state.source:
            st.session_state.source = clicked
            st.success(f"✅ Source selected at: {clicked}")
        elif not st.session_state.destination:
            distance = geodesic(st.session_state.source, clicked).km
            if distance <= max_distance_km:
                st.session_state.destination = clicked
                st.success(f"✅ Destination selected at: {clicked}")
                st.text(f"📏 Distance: {distance:.2f} km")
            else:
                st.error(f"❌ Distance is {distance:.2f} km (> {max_distance_km} km). Please reset and try again.")

    # Show selected coordinates
    if st.session_state.source:
        st.write("🟢 **Source Coordinates:**", f"Lat: `{st.session_state.source[0]}`, Lon: `{st.session_state.source[1]}`")
    if st.session_state.destination:
        st.write("🔴 **Destination Coordinates:**", f"Lat: `{st.session_state.destination[0]}`, Lon: `{st.session_state.destination[1]}`")

    # Show Google Maps route
    if st.session_state.source and st.session_state.destination:
        src_lat, src_lon = st.session_state.source
        dst_lat, dst_lon = st.session_state.destination

        gmaps_link = (
            f"https://www.google.com/maps/dir/?api=1"
            f"&origin={src_lat},{src_lon}"
            f"&destination={dst_lat},{dst_lon}"
            f"&travelmode=driving&layer=t"
        )

        st.text("Google Maps Route with Traffic")
        st.markdown(f"[Open in Google Maps]({gmaps_link})", unsafe_allow_html=True)

    # Reset
    if st.button("Reset"):
        st.session_state.source = None
        st.session_state.destination = None
        st.rerun()

    # Safe return
    source_lat, source_lon = st.session_state.source if st.session_state.source else (None, None)
    destination_lat, destination_lon = st.session_state.destination if st.session_state.destination else (None, None)

    return distance, source_lat, source_lon, destination_lat, destination_lon
