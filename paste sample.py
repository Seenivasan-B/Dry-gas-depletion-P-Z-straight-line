import pyperclip
import pandas as pd
import streamlit as st
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
