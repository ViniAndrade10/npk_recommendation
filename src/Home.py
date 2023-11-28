import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")

welcome_text = """
    Welcome to the fertilizer recommendation model.
    This is my conclusion project of the MBA in Data Science and Analytics of
    USP.

    The idea here is to create an application where agronomics forces, could
    simulate values of fertilization with NPK depending on the crop's type,
    land and environmental paramenters.

    Enjoy it.
"""

st.title("NPK Recommendation")
st.subheader(welcome_text)
