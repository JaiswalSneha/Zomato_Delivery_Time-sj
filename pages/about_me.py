import streamlit as st

st.markdown(
    """
    <style>
        .stApp {
            background-color: white;
            color: #116A91;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    layout="wide",
    page_title="About - Zomato Delivery Time Estimator",
    page_icon="🍕"
)

# Basic styling for clean readability
st.markdown("""
<style>
    .title {
        text-align: center;
        color: #116A91;
        font-size: 36px;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #444444;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .section {
        background-color: #f9f9f9;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 6px;
    }
    .header {
        color: #116A91;
        font-size: 20px;
        margin-bottom: 10px;
    }
    .content {
        color: #333333;
        font-size: 16px;
        line-height: 1.6;
    }
    ul {
        margin-left: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Title and subtitle
st.markdown("<div class='title'>Zomato Delivery Time Estimator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predicting and Optimizing Food Delivery Logistics with Machine Learning</div>", unsafe_allow_html=True)


# Creator Section
st.markdown("""
<div class='section'>
    <div class='header'>👤 About the Creator</div>
    <div class='content'>
        <strong>SJ</strong> is the creator of this WhatsApp Chat Analyzer.
        <br><br>
        Currently working as a <strong>Product Manager</strong>, SJ specializes in leading cross-functional teams to build and launch user-centric digital products. With a strong foundation in both business strategy and data, SJ is passionate about solving real-world problems through thoughtful product design.
        <br><br>
        Prior to becoming a Product Manager, SJ held roles as:
        <ul>
            <li><strong>Data Analyst</strong> – Experienced in SQL, Excel, Tableau, Python, Consumer Bureau data, and Microfinance analytics</li>
            <li><strong>Data Engineer</strong> – Worked with IBM DataStage, Informatica, Ataccama, and Collibra</li>
        </ul>
        This diverse background is what inspired SJ to build tools like this — combining the power of data with simple, intuitive products.
    </div>
</div>
""", unsafe_allow_html=True)


# Purpose Section
st.markdown("""
<div class='section'>
    <div class='header'>📌 Purpose of This App</div>
    <div class='content'>
        This project aims to explore and optimize food delivery logistics by predicting delivery times using advanced machine learning models.
        <br><br>
        Users can select from various regression algorithms, preprocess data with encoding and scaling options, and evaluate model performance with RMSE, MSE, and R² metrics.
        <br><br>    
        This app uses multiple regression algorithms such as Linear Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, SVM, and K-Nearest Neighbors to predict delivery times accurately.    
    </div>
</div>
""", unsafe_allow_html=True)


# Technology Section
st.markdown("""
<div class='section'>
    <div class='header'>🛠️ Technology Used</div>
    <div class='content'>
        The application is built with <strong>Streamlit</strong> for rapid prototyping and visualization.
        <br><br>
        Data manipulation and machine learning rely on <strong>Pandas</strong>, <strong>Scikit-Learn</strong>, and <strong>ML Algos</strong>.
        <br><br>
        Visualizations utilize <strong>Matplotlib & Seaborn</strong> to compare actual and predicted delivery times.
    </div>
</div>
""", unsafe_allow_html=True)


# Features Section
st.markdown("""
<div class='section'>
    <div class='header'>🔍 What You Can Do with This App</div>
    <div class='content'>
        Select from multiple regression algorithms to train and test models on delivery time data.
        <ul>
            <li>Choose preprocessing steps like One-Hot Encoding and Standard Scaling</li>
            <li>Adjust test data size to validate model performance</li>
            <li>Evaluate models with metrics: MSE, RMSE, and R²</li>
            <li>Visualize actual vs predicted delivery times</li>
        </ul>
        This helps you understand and optimize food delivery logistics using data science.
    </div>
</div>
""", unsafe_allow_html=True)


# UI Note Section
st.markdown("""
<div class='section'>
    <div class='header'>💡 A Note About the Interface</div>
    <div class='content'>
        The app's UI is intentionally minimal. Since Streamlit is focused more on data apps than design-heavy interfaces, the layout may not look like a modern web app — and that’s okay!
        <br><br>
        Our main goal was to provide a <strong>functional, data-focused experience</strong> rather than a flashy frontend. 
        We used basic HTML formatting (with assistance from ChatGPT) to enhance readability without overcomplicating the structure.
    </div>
</div>
""", unsafe_allow_html=True)


# Footer or final call-to-action
st.markdown("""
<div class='section'>
    <div class='header'>🚀 Ready to Predict?</div>
    <div class='content'>
        Use the sidebar to select algorithms and configure preprocessing options. Then train your models and see how accurately they predict delivery times.
        <br><br>
        Let's optimize delivery with the power of machine learning!
    </div>
</div>
""", unsafe_allow_html=True)
