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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os
sys.path.append(os.path.dirname(__file__))
from loc_cal import haversine_deg,get_weather_category
from location import location_selector
import time
from pathlib import Path
from sklearn.model_selection import GridSearchCV
st.set_page_config(layout="wide")

data_dir = Path(__file__).parent.parent / 'data'
clean_data_zomato = data_dir / 'Cleaned_Data.csv'
df = pd.read_csv(clean_data_zomato)


start = time.time()

# --- 1. Define model dictionary ---
models_and_params = {
    'Linear Regression': (LinearRegression(), {}),
     'Decision Tree Regressor': ( DecisionTreeRegressor(), {'regressor__max_depth': [None, 10, 20]} ),
    #  'Random Forest Regressor': (  RandomForestRegressor(random_state=42), {'regressor__n_estimators': [100], 'regressor__max_depth': [None, 10]}),
     'Gradient Boosting Regressor': ( GradientBoostingRegressor(random_state=42,verbose =1), {'regressor__learning_rate': [0.05, 0.1], 'regressor__n_estimators': [100,200]}    ),
     'XGB Regressor': ( XGBRegressor(random_state=42, verbosity=1),  {'regressor__n_estimators': [100], 'regressor__max_depth': [3, 6]}  ),
    #  'SVR': ( SVR(),{'regressor__kernel': ['rbf'], 'regressor__C': [1, 10]} ),
     'KNeighbors Regressor': ( KNeighborsRegressor(),{'regressor__n_neighbors': [3, 5, 7, 10]})

}

# --- 3. Preprocessor ---
cat_col = ['weather_group','vehicle_cond', 'vehicle','festival','city','rating_bin']
num_col = ['multiple_deliveries','age','distance_bin_map','traffic_bin','speed']

preprocessor = ColumnTransformer(
    transformers = [
    ('categorical',OneHotEncoder(drop= 'first'),cat_col),
    ('numerical',StandardScaler(),num_col),
],    remainder='passthrough'
)


# --- 4. Prepare data (replace with your actual dataframe) ---
X = df[['age', 'rating_bin', 'weather_group', 'traffic_bin', 'vehicle_cond','vehicle', 'multiple_deliveries', 'festival', 'city','distance_bin_map','speed']]
y = df['time']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# --- 5. Loop through models ---
results = []

st.subheader('Click to initiate Best Model Selection')
button_side = st.button('Train and Test Model')

# st.sidebar.text('Start with Prediction')
# predict_button = st.sidebar.button('Predict')


# ------------- Start of your model training block -------------
best_model_object = None
best_model_name = ""
lowest_rmse = float('inf')
results = []


if button_side:
    for name, (model, param_grid) in models_and_params.items():
        st.subheader(f"\n🔍 Training {name}...")
        start = time.time()

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Save best model
        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best_model_object = grid.best_estimator_
            best_model_name = name

        st.text(f"✅ Best Params for {name}: {grid.best_params_}")
        st.text(f"📈 Test R²: {r2:.3f} | RMSE: {rmse:.2f}")

        results.append({
            'Model': name,
            'Best CV R2': grid.best_score_,
            'Test R2': r2,
            'Test MSE': mse,
            'Test RMSE': rmse,
            'Best Params': grid.best_params_
        })

        st.markdown(f"*Model Time : {time.time() - start:.2f} sec*")
        st.markdown('---')

    # After training loop
    results_df = pd.DataFrame(results)
    st.subheader("📊 Model Comparison")
    st.dataframe(results_df)

    st.success(f"🏆 Best model is **{best_model_name}** with RMSE = {lowest_rmse:.2f}")


    # ----------------------------------------------------
    # ✅ Only proceed if best_model_object exists
    if best_model_object:
        regressor = best_model_object.named_steps['regressor']

        # ✅ Check if model supports feature importances
        if hasattr(regressor, 'feature_importances_'):
            # Get feature names from preprocessor
            feature_names = best_model_object.named_steps['preprocessor'].get_feature_names_out()

            # Get importances
            importances = regressor.feature_importances_

            # Create a DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            # Show in Streamlit
            st.subheader("📌 Feature Importances")
            # st.dataframe(importance_df)

            # Optional: Plot the importances
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importances")
            ax.invert_yaxis()  # Most important at the top
            st.pyplot(fig)
        else:
            st.info("ℹ️ The selected model does not support feature importances.")


        imp_df = pd.DataFrame(columns = ['Feature','Importance'])
        imp_df['Feature'] = importance_df['Feature']
        imp_df['Importance'] = importance_df['Importance']
        st.dataframe(imp_df)


    # getting imp features
        threshold = 0.015
        top_features = importance_df[importance_df['Importance'] > threshold]['Feature'].tolist()

    #-----------------------------------------------------
import joblib
if best_model_object:
    joblib.dump(best_model_object, 'best_delivery_model.pkl')
    st.session_state["model_name"] = best_model_name
    st.session_state["y_test"] = y_test
    # st.session_state["imp_df"] = imp_df
    st.success("Best Model saved! You can proceed with Prediction.")
