# simple example of streamlit app from streamlit.io website

import streamlit as st
import pandas as pd

st.write("""
# My first app
Hello *world!*
""")

df = pd.read_csv("student1.csv")
st.bar_chart(df)