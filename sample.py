import streamlit as st
import pyperclip
import pandas as pd

load = st.button("Paste from clipboard")
if "load_state" not in st.session_state:
    st.session_state.load_state = False
if load or st.session_state.load_state:
    st.session_state.load_state = True
    try:
    # Read the clipboard content into a pandas DataFrame
        
        text = pyperclip.paste()
        text = text.split('\n')
        #st.write(text)

        data = text

        data = [s.split("\t") for s in data if "\t" in s]
        #st.write(data)
        # Create a DataFrame
        df = pd.DataFrame(data, columns=['Index', 'Value'])

        # Remove the newline character from the 'Value' column
        df['Value'] = df['Value'].str.strip('\r')

        # Convert the 'Index' and 'Value' columns to numeric data types
        df['Index'] = pd.to_numeric(df['Index'])
        df['Value'] = pd.to_numeric(df['Value'])

        st.write(df)
    except Exception as e:
        st.error("Failed to read data from clipboard. Make sure the clipboard contains only numeric values in two columns.")
        st.error(f"Error details: {e}")
