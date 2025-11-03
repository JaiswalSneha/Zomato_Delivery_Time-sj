import pandas as pd
import numpy as np
import streamlit as st
st.set_page_config(layout="wide")
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(__file__))
# from pages import loc_cal

# from loc_cal import haversine_deg,get_weather_category
from pages.loc_cal import haversine_deg, get_weather_category
from location import location_selector
import time
from sklearn.metrics import mean_squared_error,r2_score
import joblib

best_model_object = None
y_test = None
flag = 0

if 'model_name' in st.session_state and 'y_test' in st.session_state:
    # best_model_object = st.session_state["model_object"]
    st.header(f"Hello {st.session_state['model_name']}, let's train a model!")
    y_test = st.session_state["y_test"]
else:
    st.header("Please Train the Best Model and then we can proceed with Prediction !")
    flag = 1

if 'model_name' in st.session_state and 'y_test' in st.session_state:
    best_model_object = joblib.load('best_delivery_model.pkl')

data_dir = Path(__file__).parent.parent / 'data'
clean_data_zomato = data_dir / 'Cleaned_Data.csv'
df = pd.read_csv(clean_data_zomato)

if flag == 0:

    cat_col = ['weather_group', 'vehicle_cond', 'vehicle', 'festival', 'city', 'rating_bin']
    num_col = ['multiple_deliveries', 'age', 'distance_bin_map', 'traffic_bin', 'speed']

    st.subheader('Predicting on basis the Best Model')

    min_age = int(df['age'].min())
    max_age = int(df['age'].max())

    col1,col2 = st.columns(2)
    with col1:
        age = st.number_input('Age of the Delivery person', min_value=df['age'].min().astype(int), max_value=df['age'].max().astype(int))
    with col2:
        ratings = st.selectbox('Ratings of the Delivery person', df['rating_bin'].unique())

    col1, col2 = st.columns(2)
    with col1:
        vehicle = st.selectbox('Vehicle', df['vehicle'].unique())
    with col2:
        multiple_deliveries = st.selectbox('Number of deliveries at a time', np.unique(df['multiple_deliveries']).astype(int))

    col1, col2 = st.columns(2)
    with col1:
        vehicle_cond = st.selectbox('Vehicle condition',np.unique(df['vehicle_cond']))
    with col2:
        festival = st.selectbox('If there is a Festival or not', np.unique(df['festival']))

    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox('Type of City', np.unique(df['city']))
    with col2:
        weather_options = np.unique(df['weather_group'])
        default_index = 0
        weather = st.selectbox('Weather', options=weather_options, index=default_index)
        safe_weather = weather

    # distance = 0
    col1, col2 = st.columns(2)
    
    with col1:
        traffic_map_dic = {'Low': 1, 'Medium': 2, 'High': 3,'Jam':4}
        traffic = st.selectbox('Traffic flow', ['Low','Medium','High','Jam'],index=0)
        traffic = traffic_map_dic[traffic]
    with col2:
        bike_speed = st.number_input('Speed of the Bike', min_value=20.00, max_value=60.00)


    col1, col2 = st.columns(2)
    with col1:
        distance_yes_no = st.selectbox('Do you have the distance?',['Default', 'Yes', 'No', 'Wish to enter the Longitude and Latitude'], )
        distance = 0  # default value
    with col2:
        pass

    col11,col12,col21,col22 = st.columns(4)

    if distance_yes_no == 'Yes':
        distance_bin_labels = ['0-4', '4-8', '8-12', '12-16', '16+']
        distance_bin_map = {'0-4': 0, '4-8': 1, '8-12': 2, '12-16': 3, '16+': 4}
        distance_mapping = st.selectbox('Distance of the Delivery person',distance_bin_labels)
        distance = distance_bin_map[distance_mapping]

    if distance_yes_no == 'No':
        distance_from_loc, src_lat, src_lon, dst_lat, dst_lon = location_selector()
        if distance_from_loc is None:
            distance_from_loc=0
            
        detected_weather = get_weather_category(src_lat, src_lon)
        if detected_weather in weather_options:
            weather = detected_weather
        else:
            weather = safe_weather

        
        if distance_from_loc<4:
            distance=0
        elif distance_from_loc<8:
            distance=1
        elif distance_from_loc<12:
            distance=2
        elif distance_from_loc<16:
            distance=3        
        else:
            distance=4
        weather = get_weather_category(src_lat,src_lon)
        if weather is not None:
            st.text(f"The weather is {weather} in your area")
            st.text(f"Please select the weather else your default weather is {safe_weather}")
            st.text(f"The distance is {distance}")
    
    if distance_yes_no == 'Wish to enter the Longitude and Latitude':
        with col11:
            src_lat = st.number_input('Source latitude', min_value=-90, max_value=90)
        with col12:
            src_lon = st.number_input('Source longitude', min_value=-180, max_value=180)
        with col21:
            des_lat = st.number_input('Destination latitude', min_value=-90, max_value=90)
        with col22:
            des_lon = st.number_input('Destination longitude', min_value=-180, max_value=180)
        distance_cal = loc_cal.haversine_deg(src_lat,src_lon,des_lat,des_lon)
        if distance_cal<4:
            distance=0
        elif distance_cal<8:
            distance=1
        elif distance_cal<12:
            distance=2
        elif distance_cal<16:
            distance=3        
        else:
            distance=4
        st.text(f"The distance is {distance}")

    input_data = pd.DataFrame([{
        'age':age,
        'rating_bin':ratings,
        'weather_group': weather,
        'traffic_bin': traffic,
        'vehicle_cond': vehicle_cond,
        'vehicle': vehicle,
        'multiple_deliveries': multiple_deliveries,
        'festival': festival,
        'city': city,
        'distance_bin_map': distance,
        'speed':bike_speed
    }])

    

    predict_button = st.button('📈 Predict')
    if predict_button:
        st.dataframe(input_data)
        with st.spinner("Predicting..."):
            time.sleep(2)  # simulate delay
            y_pred = best_model_object.predict(input_data)
            lower_y_pred = y_pred[0]-2
            upper_y_pred = y_pred[0]+2
            st.success(f"The predicted delivery time is **{np.round(lower_y_pred,2)} - {np.round(upper_y_pred,2)} minutes**")
