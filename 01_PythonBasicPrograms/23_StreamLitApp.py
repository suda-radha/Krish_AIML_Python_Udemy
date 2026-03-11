#Streamlit is an open-source app framework for Machine Learning and Data Science projects.
# It allows you to create beautiful web applications for your ML and DS projects with simple Python scripts

## add streamlit, numpy, pandas packages in requirements.txt and install
## as you keep developing this app, to run this app use this command in terminal >>streamlit run 23_StreamlitApp.py
## your app will run in localhost 8501 port(http://localhost:8501/)

## As you generate the table, linechart displays on web app you can see options to download them as well -- WOW!!



import streamlit as st
import numpy as np
import pandas as pd

## Title of the application
st.title("Hello Streamlit") ## run your app and see in browser WOW!!!

## Display a simple text on page
st.write("This is a simple text")

## create a dataframe
df = pd.DataFrame({
    'first column': [1,2,3,4,5],
    'second column': [10,20,30,40,50]
})

## Display the dataframe
st.write("Here is the dataframe")
st.write(df)

# create a line chart
chart_data=pd.DataFrame(
    np.random.randn(20,3), columns=['a','b','c']
)
st.line_chart(chart_data)
