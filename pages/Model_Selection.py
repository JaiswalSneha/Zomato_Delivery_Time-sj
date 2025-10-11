import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
import sys
import os
sys.path.append(os.path.dirname(__file__))
import time
from pathlib import Path
st.set_page_config(layout="wide")

data_dir = Path(__file__).parent.parent / 'data'
clean_data_zomato = data_dir / 'Cleaned_Data.csv'
df = pd.read_csv(clean_data_zomato)

algo = ['','Linear Regression','Decision Tree','Random Forest','Gradient Boosting','XGBoost', 'SVM', 'KNeighbors']

st.subheader('Select the Algorithm for Regression')
algo_selected = st.selectbox('',algo)

cat_col = ['weather_group', 'vehicle_cond', 'vehicle', 'festival', 'city', 'rating_bin']
num_col = ['multiple_deliveries', 'age', 'distance_bin_map', 'traffic_bin', 'speed']
X = df[['weather_group', 'vehicle_cond', 'vehicle', 'festival', 'city', 'rating_bin','multiple_deliveries', 'age', 'distance_bin_map', 'traffic_bin', 'speed']]
y = df['time']


if algo_selected != "":
    st.title(algo_selected)

    # Split data
    test_size_selected = st.slider('Test size', min_value=0.1, max_value=1.0, value=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_selected)

    # Preprocessing selection
    ohe_selection = st.selectbox('One Hot Encoding for Categorical Columns', ['Yes'])
    scale_selection = st.selectbox('Standard Scaling for Numerical Columns', ['Yes', 'No'])

    # Build preprocessor
    if ohe_selection == 'Yes' and scale_selection == 'Yes':
        preprocessor = ColumnTransformer([
            ('categorical', OneHotEncoder(drop='first'), cat_col),
            ('numerical', StandardScaler(), num_col)
        ], remainder='passthrough')
    elif ohe_selection == 'Yes' and scale_selection == 'No':
        preprocessor = ColumnTransformer([
            ('categorical', OneHotEncoder(drop='first'), cat_col)
        ], remainder='passthrough')
    else:
        preprocessor = 'passthrough'  # fallback

    # Select model
    if algo_selected == 'Linear Regression':
        model = ('regressor', LinearRegression())
    elif algo_selected == 'Decision Tree':
        model = ('regressor', DecisionTreeRegressor())
    elif algo_selected == 'Random Forest':
        model = ('regressor', RandomForestRegressor())
    elif algo_selected == 'Gradient Boosting':
        model = ('regressor', GradientBoostingRegressor())
    elif algo_selected == 'XGBoost':
        model = ('regressor', XGBRegressor())
    elif algo_selected == 'SVM':
        model = ('regressor', SVR())
    elif algo_selected == 'KNeighbors':
        model = ('regressor', KNeighborsRegressor())
    else:
        st.error("Invalid algorithm selected")
        model = None

    if st.button('Start Training and Testing') and model:
        start = time.time()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            model
        ])

        final_pipeline = pipeline.fit(X_train, y_train)
        y_pred = final_pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.markdown('---')
        st.subheader('Evaluation Metrics')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Mean Squared Error**")
            st.text(np.round(mse,2))
        with col2:
            st.write(f"**Root Mean Squared Error**")
            st.text(np.round(rmse,2))
        with col3:
            st.write(f"**R2 Score**")
            st.text((np.round(r2,2)))

        st.text('')
        st.text('')
        st.markdown(f"*Model Time : {time.time() - start}*")
        st.markdown('---')

        # ----------------Plotting the test and predict data -----------------------
        df_plot = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred
        }).reset_index(drop=True)

        # Plot line chart
        fig, ax = plt.subplots(figsize=(10, 5))
        # ax.scatter(df_plot['Actual'], label='Actual', color='blue')
        # ax.scatter(df_plot['Predicted'], label='Predicted', color='orange', linestyle='--')
        ax.scatter(df_plot.index, df_plot['Actual'], label='Actual', color='blue')
        ax.scatter(df_plot.index, df_plot['Predicted'], label='Predicted', color='orange')

        ax.set_title("Test vs Predicted")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Target Value")
        ax.legend()
        st.pyplot(fig)

# st.dataframe(py())
# st.text(df)



