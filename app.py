"""Open Data Scientist — entrypoint and page router.

``streamlit run app.py`` stays the single entrypoint (CI, Docker, and the
README all reference it). This file only configures the app frame and routes
to the actual pages, so sidebar labels are clean names instead of filenames.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Open Data Scientist",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.navigation(
    [
        st.Page("Home.py", title="Home", default=True),
        st.Page("pages/1_Profile.py", title="Profile"),
        st.Page("pages/2_Clean.py", title="Clean"),
        st.Page("pages/3_Dashboard.py", title="Dashboard"),
        st.Page("pages/4_Visual_SQL.py", title="Visual SQL"),
        st.Page("pages/5_SQL_Coach.py", title="SQL Coach"),
        st.Page("pages/6_Pipeline.py", title="Pipeline"),
    ]
).run()
