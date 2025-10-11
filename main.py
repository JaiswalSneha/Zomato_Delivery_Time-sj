import streamlit as st

# Define pages (by file or function)
page1 = st.Page("pages/PreProcessing.py", title="Preprocessing", icon="📊")
page2 = st.Page("pages/Model_Selection.py", title="Model Selection", icon="🎯")
page3 = st.Page("pages/Best_Model.py", title="Best Model", icon="🤖")
page4 = st.Page("pages/Prediction.py", title="Prediction", icon="🔮")
page5 = st.Page("pages/about_me.py", title="About Me", icon="👤")
# Create navigation
pg = st.navigation([page1, page2, page3, page4, page5])

pg.run()