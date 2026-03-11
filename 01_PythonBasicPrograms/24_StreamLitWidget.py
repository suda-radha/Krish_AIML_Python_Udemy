## using streamlit pkg you can create ui elements - widget
## command to run this app >> streamlit run 24_StreamLitWidget.py
## we'll see Input box, Slider, Dropdown

## goto streamlit.io for loads of examples code

import streamlit as st
import pandas as pd

#Title
st.title("Streamlit Widget App")

#Inout box
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello {name}, welcome to this app!")

#slider
age=st.slider("Select your age: ",0,100,25)
st.write(f"Your age is: {age}")

#dropdown
options = ["Java", "Python", "C", "C++", "Ruby", "JavaScript"]
choice= st.selectbox("Choose your favourite language: ",options)
st.write(f"You Selected: {choice}")

# Display data
data = {
    "Name": ["John", "Tom", "Lucy"],
    "Age": [22,34,25],
    "City": ["Chen", "Bang", "Delhi"]
}

df = pd.DataFrame(data)
st.write(df)

#upload file
uploaded_file=st.file_uploader("Upload a csv file",type="csv")

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df)