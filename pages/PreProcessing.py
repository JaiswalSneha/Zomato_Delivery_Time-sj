import pandas as pd
import numpy as np
import streamlit as st
import io
from pages import loc_cal
import matplotlib.pyplot as plt
from pathlib import Path
import io
st.set_page_config(layout="wide")
import seaborn as sns
import os


project_root = Path(__file__).resolve().parent.parent
gif_path = project_root / 'pages' / 'zomato_del.gif'


col1, col_merged, _, _ = st.columns([1, 3, 0.0001, 0.0001])
with col1:
    st.image(gif_path)
with col_merged:
    st.subheader('Description')
    st.text('Description: Delve into the world of food delivery with the Zomato Delivery Dataset. This dataset provides a comprehensive view of delivery operations, including delivery person details, order timestamps, weather conditions, and more. Explore patterns, optimize delivery routes, and enhance customer satisfaction with insights from this dataset.')
    st.subheader('Kaggle Dataset')
    st.text('Zomato Delivery Operations Analytics Dataset')
    st.write('Dataset Link: https://www.kaggle.com/datasets/saurabhbadole/zomato-delivery-operations-analytics-dataset')

st.markdown("---")

st.subheader("About the data")
st.markdown("""
**ID:** Unique identifier for each delivery.<br>
**Delivery_person_ID:** Unique identifier for each delivery person.<br>
**Delivery_person_Age:** Age of the delivery person.<br>
**Delivery_person_Ratings:** Ratings assigned to the delivery person.<br>
**Restaurant_latitude:** Latitude of the restaurant.<br>
**Restaurant_longitude:** Longitude of the restaurant.<br>
**Delivery_location_latitude:** Latitude of the delivery location.<br>
**Delivery_location_longitude:** Longitude of the delivery location.<br>
**Order_Date:** Date of the order.<br>
**Time_Ordered:** Time the order was placed.<br>
**Time_Order_picked:** Time the order was picked up for delivery.<br>
**Weather_conditions:** Weather conditions at the time of delivery.<br>
**Road_traffic_density:** Density of road traffic during delivery.<br>
**Vehicle_condition:** Condition of the delivery vehicle.<br>
**Type_of_order:** Type of order (e.g., dine-in, takeaway, delivery).<br>
**Type_of_vehicle:** Type of vehicle used for delivery.<br>
**Multiple_deliveries:** Indicator of whether multiple deliveries were made in the same trip.<br>
**Festival:** Indicator of whether the delivery coincided with a festival.<br>
**City:** City where the delivery took place.<br>
**Time_taken (min):** Time taken for delivery in minutes.
""", unsafe_allow_html=True)



st.markdown("---")

project_root = Path(__file__).resolve().parent.parent

csv_path = project_root / 'data' / 'Zomato_Delivery.csv'

# st.text(os.getcwd())
# st.text(csv_path.resolve())
# st.text(csv_path.exists())

df = pd.read_csv(csv_path)


st.subheader('Zomato Delivery Dataset - raw data')
st.dataframe(df.head())
st.markdown("---")

df['distance_km'] = loc_cal.haversine_deg(df['Restaurant_latitude'], df['Restaurant_longitude'], df['Delivery_location_latitude'], df['Delivery_location_longitude'])
df = df.rename(columns = {'Delivery_person_Age':'age','Delivery_person_Ratings':'ratings','Type_of_vehicle':'vehicle','Road_traffic_density':'traffic'})
df = df.rename(columns = {'Weather_conditions':'weather','Type_of_order':'order_type','Time_taken (min)':'time','Vehicle_condition':'vehicle_cond'})
df = df.rename(columns = {'Time_Orderd':'order_time','Order_Date':'date','Time_Order_picked':'order_pick','Festival':'festival','City':'city'})

df['city_name'] = df['Delivery_person_ID'].str.replace('RES','-').str.split('-').str.get(0)
df =df.drop(columns = ['ID','Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude','Delivery_person_ID'])
df =df.drop(columns = ['date', 'order_time', 'order_pick'])


st.subheader('Distance Computation with Lat/Lon, Column Renaming, and Column Removal')
with st.expander("▶️ Show Operations Performed"):
    st.text('1. Calculated the distance in Km from the latitude and longitude of the restaurant and the delivery location')
    st.text('2. Calculated the City Initials from the Restaurant ID')
    st.text('3. Renamed the columns to shorter names')
    st.text('4. Dropped the ID column')
    st.text("5. Other columns dropped : Restaurant_latitude, Restaurant_longitude, Delivery_location_latitude, Delivery_location_longitude, Delivery_person_ID, date, order_time & order_pick")

st.dataframe(df.head())
st.markdown("---")

st.subheader('Checking the distribution and spread of data')
st.dataframe(df.describe())
st.markdown("---")

st.subheader('Type of Features')
st.text(""" Categorical Features:
weather, traffic, vehicle_cond, order_type, vehicle, festival, city & city_name
""")
st.text(""" Numerical Features:
age, ratings, multiple_deliveries, time & distance_km
""")
st.markdown("---")

st.subheader('Target Feature :')
st.text(""" Time Taken (min)""")


st.markdown("---")

st.subheader('Distribution of Categorical Features - Frequency of data')
col1, col2, col3, col4= st.columns(4)
categorical_columns_plot = {'weather':'Weather Frequency', 'traffic':'Traffic type', 'vehicle':'Vehicle Type', 'vehicle_cond':'Vehicle Condition'}
col_div_list = [col1,col2,col3,col4]

for (col_name, label), col_div in zip(categorical_columns_plot.items(), col_div_list):
    with col_div:
        fig, ax = plt.subplots()
        df[col_name].value_counts().plot(kind='bar', ax=ax)
        st.text(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        st.pyplot(fig)

# ===========================================================================================================================

col1, col2, col3, col4= st.columns(4)
categorical_columns_plot = {'order_type':'Order Type', 'festival':'Festival', 'city':'City Type', 'city_name':'City Name'}
col_div_list = [col1,col2,col3,col4]

for (col_name, label), col_div in zip(categorical_columns_plot.items(), col_div_list):
    with col_div:
        fig, ax = plt.subplots()
        df[col_name].value_counts().plot(kind='bar', ax=ax)
        st.text(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        st.pyplot(fig)

st.markdown("---")

# ===========================================================================================================================

st.subheader('Distribution of Numerical Categories - Frequency of data')
col1, col2, col3, col4 = st.columns(4)
numerical_columns = {'age':'Age', 'ratings':'Ratings',  'multiple_deliveries':'Multiple Deliveries','distance_km':'Distance (Km)'}
col_div_list = [col1,col2,col3,col4]

for (col_name, label), col_div in zip(numerical_columns.items(), col_div_list):
    with col_div:
        fig, ax = plt.subplots()
        sns.histplot(df[col_name], kde=True, bins=30)
        st.text(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        st.pyplot(fig)


# =================================================== Removing Invalid Values =========================================================

st.markdown("---")
data = df.copy()
st.subheader('Removing Invalid Values')
st.text('It is physically not possible to cover 2,000 km on a bike in 10 or 60 minutes, hence introducing the Speed column.')
st.text('Hence, eliminating the rows where speed > 60 km/hr')
st.text('The Zomato ratings should lie between 0-5')
st.text('Hence, discarding the ratings above 5')
st.markdown("---")

df['speed'] = df['distance_km']/(df['time']/60)
df = df[df['speed']<61]
df = df[df['ratings']<6]

col1, col2, col3, col4 = st.columns(4)
with col1:
    fig, ax = plt.subplots()
    sns.histplot(df['speed'], kde=True, bins=30, color = 'green')
    st.text("Distribution of Speed")
    ax.set_xlabel('Speed')
    ax.set_ylabel("Count")
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    ax.scatter(df['speed'], df['time'], color = 'green')
    st.text("Speed v/s Time(min)")
    ax.set_xlabel('Speed')
    ax.set_ylabel("Time(min)")
    ax.legend()
    st.pyplot(fig)

st.text('Adding the Speed column to shift the importance to the Distance column as without Speed the Distance in km column had no impact on the target variable Time')

st.markdown("---")

# ===========================================================================================================================

st.subheader('Checking the distribution and spread of data')
col1, col2 = st.columns(2)
with col1:
#     st.text('With Invalid data')
#     st.dataframe(data.describe())
# with col2:
    st.text('After removal of Invalid data')
    st.dataframe(df.describe())
    st.markdown("<small>After removal of invalid rows the mean distance reduced from 99km to 9km, and the standard deviation reduced from 1099km to 5km</small>", unsafe_allow_html=True)
st.markdown("---")

# ===========================================================================================================================

st.subheader('Data with significant impact')
col1, col2, col3, col4 = st.columns(4)
with col1:
    fig, ax = plt.subplots()
    sns.histplot(data['distance_km'], kde=True, bins=30, color = 'red')
    st.text('Invalid Distance')
    ax.set_xlabel('Invalid Distance')
    ax.set_ylabel("Count")
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    sns.histplot(df['distance_km'], kde=True, bins=30, color = 'green')
    st.text('Valid Distance')
    ax.set_xlabel('Valid Distance')
    ax.set_ylabel("Count")
    st.pyplot(fig)
with col3:
    fig, ax = plt.subplots()
    sns.histplot(data['ratings'], kde=True, bins=30, color = 'red')
    st.text('Invalid Ratings')
    ax.set_xlabel('Invalid Ratings')
    ax.set_ylabel("Count")
    st.pyplot(fig)
with col4:
    fig, ax = plt.subplots()
    sns.histplot(df['ratings'], kde=True, bins=30, color = 'green')
    st.text('Valid Ratings')
    ax.set_xlabel('Valid Ratings')
    ax.set_ylabel("Count")
    st.pyplot(fig)


st.markdown("---")

# ===========================================================================================================================

st.subheader('Checking for Null/ NA values')
shape1 = df.shape[0]
shape2 = df.dropna().shape[0]
percentage = ((shape1-shape2) / shape1)*100
st.text(f"Removing the NA values as they contribute to only {np.round(percentage,2)}  % of data")

col1,col2 = st.columns(2)
with col1:
    st.table(df.isna().sum())
    st.text(f"# rows: {df.shape[0]}, # cols: {df.shape[1]}")
with col2:
    df = df.dropna()
    st.table(df.isna().sum())
    st.text(f"# rows: {df.shape[0]}, # cols: {df.shape[1]}")

df['age'] = df['age'].astype(int)
df['multiple_deliveries'] = df['multiple_deliveries'].astype(int)

st.markdown("---")

# ===========================================================================================================================

st.subheader('Distribution of Categorical Features - With Time (in minutes)')
import seaborn as sns

col1, col2, col3, col4= st.columns(4)
categorical_columns_plot = {'weather':'Weather Frequency', 'traffic':'Traffic type', 'vehicle':'Vehicle Type', 'vehicle_cond':'Vehicle Condition'}
col_div_list = [col1,col2,col3,col4]

for (col_name, label), col_div in zip(categorical_columns_plot.items(), col_div_list):
    with col_div:
        fig, ax = plt.subplots()
        sns.boxplot(x=col_name, y='time', data=df, color = 'orange')
        st.text(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Time")
        st.pyplot(fig)

col1, col2, col3, col4= st.columns(4)
categorical_columns_plot = {'order_type':'Order Type', 'festival':'Festival', 'city':'City Type', 'city_name':'City Name'}
col_div_list = [col1,col2,col3,col4]

for (col_name, label), col_div in zip(categorical_columns_plot.items(), col_div_list):
    with col_div:
        fig, ax = plt.subplots()
        sns.boxplot(x=col_name, y='time', data=df, color = 'orange')
        st.text(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Time")
        st.pyplot(fig)


st.markdown("---")

# ===========================================================================================================================

st.subheader('Distribution of Numerical Columns with Time (in minutes)')

col1, col2, col3, col4 = st.columns(4)
numerical_columns = {'age':'Age', 'ratings':'Ratings',  'multiple_deliveries':'Multiple Deliveries','distance_km':'Distance (Km)'}
col_div_list = [col1,col2,col3,col4]

for (col_name, label), col_div in zip(numerical_columns.items(), col_div_list):
    with col_div:
        fig, ax = plt.subplots()
        ax.scatter(df[col_name], df['time'])
        st.text(f"{label} v/s Time(min)")
        ax.set_xlabel(label)
        ax.set_ylabel("Time(min)")
        ax.legend()
        st.pyplot(fig)

st.markdown('---')

# ===========================================================================================================================
# ================================Binning=======================================

# cat_col = ['**weather_group','**vehicle_cond', **'vehicle','**festival','**city','**rating_bin']
# num_col = ['multiple_deliveries','age','distance_bin_map','traffic_bin','speed']

# Distance
df['distance_bin'] = pd.cut(df['distance_km'],bins=[0, 4, 8, 12, 16, 25],labels=['0-4', '4-8', '8-12', '12-16', '16+'],include_lowest=True)
distance_bin_map = {'0-4': 0, '4-8': 1, '8-12': 2, '12-16': 3, '16+': 4}
df['distance_bin_map'] = df['distance_bin'].map(distance_bin_map).astype(int)

# Rating column bin
bins = [0, 4.0, 4.5, 5.1]
labels = ['Rating_Problematic', 'Rating_Good', 'Rating_Excellent']
df['rating_bin'] = pd.cut(df['ratings'], bins=bins, labels=labels, right=False, include_lowest=True)

# Map traffic categories to integers (ordinal encoding)
traffic_map = {'Low': 1, 'Medium': 2, 'High': 3,'Jam':4}
df['traffic_bin'] = df['traffic'].map(traffic_map)

#weather mapping
weather_group_map = { 'Cloudy': 'Weather_Max_Delay', 'Fog': 'Weather_Max_Delay', 'Sandstorms': 'Weather_Moderate_Delay', 'Stormy': 'Weather_Moderate_Delay', 'Windy': 'Weather_Moderate_Delay', 'Sunny': 'Weather_Benefit_Sunny' }
df['weather_group'] = df['weather'].map(weather_group_map)

df['vehicle_old'] =  df['vehicle']
df['vehicle'] = df['vehicle'].str.replace('electric_','')

df['vehicle_cond_old'] =  df['vehicle_cond']
df['vehicle_cond'] = df['vehicle_cond'].replace(2,1)


# ===========================================================================================================================

st.subheader('Impact of Delivery Conditions on Time')
col1,col2 = st.columns(2)

with col1:
    st.text('City type')
    st.dataframe(df.groupby('city')['time'].describe())
    st.markdown("<small>Keeping the city column because it captures critical geographical variation that strongly affects delivery times and model accuracy.</small>", unsafe_allow_html=True)
with col2:
    st.text('Festival')
    st.dataframe(df.groupby('festival')['time'].describe())
    st.markdown("<small>Keeping the festival column because it introduces a strong and consistent shift in delivery time distribution, making it a high-signal binary predictor.</small>",unsafe_allow_html=True)

st.markdown('---')

col1,col2 = st.columns(2)
with col1:
    st.text('Weather')
    st.dataframe(df.groupby('weather')['time'].describe())
    st.markdown("<small>Weather is a high-variance categorical feature that contributes significantly to target variability</small>",unsafe_allow_html=True)
with col2:
    st.text('Grouping Weather Conditions to Enhance Model Performance')
    st.dataframe(df.groupby('weather_group')['time'].describe())
    st.markdown("<small>Cloudy and foggy weather cause maximum delays; sandstorms, storms, and wind cause moderate delays; sunny weather is beneficial. </small>",unsafe_allow_html=True)
    st.markdown("<small>Grouping weather conditions in the dataset helps reduce variability, highlight delay patterns, and create meaningful features like weather impact levels to improve model accuracy. </small>",unsafe_allow_html=True)

st.markdown('---')

col1,col2 = st.columns(2)
with col1:
    st.text('Vehicle Condition')
    st.dataframe(df.groupby('vehicle_cond_old')['time'].describe())
    st.markdown("<small>Keeping the traffic column to capture complex and possibly non-linear effects of traffic flow on delivery time.</small>", unsafe_allow_html=True)
with col2:
    st.text('Grouping Vehicle Condition to Enhance Model Performance')
    st.dataframe(df.groupby('vehicle_cond')['time'].describe())
    st.markdown("<small>Replacing vehicle_cond 2 with 1 is justified because both show similar delivery performance, allowing for feature simplification without losing predictive power.</small>", unsafe_allow_html=True)

st.markdown('---')
col1,col2 = st.columns(2)
with col1:
    st.text('Vehicle Type')
    st.dataframe(df.groupby('vehicle_old')['time'].describe())
    st.markdown("<small>Keeping the vehicle type column because it explains significant variance in delivery time due to performance and operational differences across vehicle types.</small>", unsafe_allow_html=True)
with col2:
    st.text('Vehicle Type after grouping')
    st.dataframe(df.groupby('vehicle')['time'].describe())
    st.markdown("<small>Electric scooters were grouped with scooters due to their similar delivery performance, reducing category complexity without sacrificing accuracy.</small>", unsafe_allow_html=True)


st.markdown('---')
col1,col2 = st.columns(2)
with col1:
    st.text('Vehicle Type')
    st.dataframe(df.groupby('ratings')['time'].describe())
    st.markdown("<small>To clean up scattered data points and group them into clearer, more useful categories for easier understanding and better accuracy.</small>", unsafe_allow_html=True)
with col2:
    st.text('Vehicle Type after grouping')
    st.dataframe(df.groupby('rating_bin')['time'].describe())
    st.markdown("<small>To make the data easier to understand and more reliable by combining small, scattered groups into a few clear and useful categories.</small>", unsafe_allow_html=True)

st.markdown('---')

col1,col2 = st.columns(2)
with col1:
    st.text('Traffic')
    st.dataframe(df.groupby('traffic')['time'].describe())
    st.markdown("<small>Clubbing the traffic high and medium values did not give good results with the accuracy of the model, hence no merging of categories.</small>", unsafe_allow_html=True)
with col2:
    st.text('Ranking Traffic Levels for Model Understanding')
    st.dataframe(df.groupby('traffic_bin')['time'].describe())
    st.markdown("<small>Organizing the traffic data in a ranked way so the model can tell the difference between light, heavy, and jammed traffic and understand the order they come in.</small>", unsafe_allow_html=True)

st.markdown('---')

col1, col2 = st.columns(2)
with col1:
    st.text('Order Type')
    st.dataframe(df.groupby('order_type')['time'].describe())
    st.markdown("<small>The order_type feature has nearly identical delivery times across categories, offering minimal predictive value</small>", unsafe_allow_html=True)
with col2:
    st.text('City name')
    st.dataframe(df.groupby('city_name')['time'].describe())
    st.markdown("<small>The City name shows almost no variation in delivery time across categories, indicating minimal predictive value for the model.</small>", unsafe_allow_html=True)

st.text('Dropping the Order Type and City Name columns')

st.markdown('---')

#=======================================================================================================

# cat_col = ['**weather_group','**vehicle_cond', **'vehicle','**festival','**city','**rating_bin']
# num_col = ['multiple_deliveries','age','distance_bin_map','traffic_bin','speed']

st.subheader('Before and After Transformation - Final')

st.subheader('Distribution of Categorical Data after Grouping')

categorical_columns_plot = {'weather':['Weather','Before'], 'weather_group':['Weather','After'], 'ratings':['Rating','Before'],'rating_bin':['Rating','After']}
col1, col2, col3, col4= st.columns(4)
col_div_list = [col1,col2,col3,col4]

for (col_name, label), col_div in zip(categorical_columns_plot.items(), col_div_list):
    with col_div:
        fig, ax = plt.subplots()
        if label[1]=='Before':
            sns.boxplot(x=col_name, y='time', data=df, color = 'red')
        if label[1]=='After':
            sns.boxplot(x=col_name, y='time', data=df, color = 'green')
        st.text(f"{label[0]} - {label[1]}")
        ax.set_xlabel(label[0])
        ax.set_ylabel("Time")
        st.pyplot(fig)

st.markdown('---')

categorical_columns_plot = {'vehicle_cond_old':['Vehicle Condition','Before'],'vehicle_cond':['Vehicle Condition','After'], 'vehicle_old':['Vehicle','Before'], 'vehicle':['Vehicle','After']}
col1, col2, col3, col4= st.columns(4)
col_div_list = [col1,col2,col3,col4]
for (col_name, label), col_div in zip(categorical_columns_plot.items(), col_div_list):
    with col_div:
        fig, ax = plt.subplots()
        if label[1]=='Before':
            sns.boxplot(x=col_name, y='time', data=df, color = 'red')
        if label[1]=='After':
            sns.boxplot(x=col_name, y='time', data=df, color = 'green')
        st.text(f"{label[0]} - {label[1]}")
        ax.set_xlabel(label[0])
        ax.set_ylabel("Time")
        st.pyplot(fig)

st.markdown('---')

#=======================================================================================================

# cat_col = ['**weather_group','**vehicle_cond', **'vehicle','**festival','**city','**rating_bin']
# num_col = ['multiple_deliveries','age','distance_bin_map','traffic_bin','speed']

st.subheader('Distribution of Numerical Data after Grouping')

categorical_columns_plot = {'distance_km':['Distance','Before'],'distance_bin_map':['Distance','After'], 'traffic':['Traffic','Before'], 'traffic_bin':['Traffic','After']}
col1, col2, col3, col4= st.columns(4)
col_div_list = [col1,col2,col3,col4]
for (col_name, label), col_div in zip(categorical_columns_plot.items(), col_div_list):
    with col_div:
        fig, ax = plt.subplots()
        if label[1]=='Before':
            ax.scatter(df[col_name], df['time'],color = 'red')
        if label[1]=='After':
            sns.boxplot(x=col_name, y='time', data=df, color = 'green')
        st.text(f"{label[0]} - {label[1]}")
        ax.set_xlabel(label[0])
        ax.set_ylabel("Time")
        st.pyplot(fig)

st.markdown('---')

#=======================================================================================================
df.to_csv(project_root / 'data' / 'Cleaned_Data.csv')

#=======================================================================================================

df = df.drop(columns = ['ratings','weather','traffic','order_type','city_name','distance_bin_map','vehicle_old','vehicle_cond_old'])

#=======================================================================================================

st.subheader('Final Data Information')

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.code(s)

st.markdown('---')
st.dataframe(df)

st.success('Data Cleaning is completed, please proceed with Model Training and Prediction')
