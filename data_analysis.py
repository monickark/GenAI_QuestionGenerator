from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()

client = AzureOpenAI(
  azure_endpoint = os.getenv('AZURE_ENDPOINT'),
  api_key=os.getenv('API_KEY'),  
  api_version=os.getenv('API_VERSION')
)

# Sidebar for uploading file
st.sidebar.title("Upload CSV")

# Dropdown for selecting LLM model
model_option = st.sidebar.selectbox(
    "Select Your LLM Model",
    ("gpt-35-turbo", "text-curie-001", "text-babbage-001", "text-ada-001")
)

# Dropdown for selecting Embedding model
embedding_model_option = st.sidebar.selectbox(
    "Select Your Embedding Model",
    ("text-similarity-davinci-001", "text-similarity-curie-001", "text-similarity-babbage-001", "text-similarity-ada-001")
)

# Dropdown for selecting temperature
temperature_option = st.sidebar.selectbox(
    "Select Temperature",
    (0.2, 0.5, 0.7)
)

# Dropdown for selecting max tokens
max_tokens_option = st.sidebar.selectbox(
    "Select Max Tokens",
    (150, 200, 250)
)

# Step 3: Upload CSV File
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Define the prompt for generating questions

    df = pd.read_csv(uploaded_file)
    st.write("DataFrame Preview:", df.head())
    def analyze_relationship(uploaded_file, model):
        columns = df.columns.tolist()
        print("Column : ",columns)
         
        prompt = f"""
                    I have a CSV file {uploaded_file} with multiple columns of numerical data. I need a Python script that can do the following:
                    1. Load the CSV file {uploaded_file} into a DataFrame.
                    2. Calculate the correlation matrix between the columns.
                    3. Plot a heatmap of the correlation matrix.

                    Please provide a complete Python script that includes necessary imports, loading the CSV, calculating the correlation matrix, plotting the heatmap
                    Also add line chart, bar chart, area chart using any 2 columns from {columns}.
                    Use streamlit, seaborn nad matplotlib for plotting graphs
                    In code add remove warning for deprecation

                    return only executable pythode code. dont add any extra string with it.
                """
        response = client.chat.completions.create(     
            model="gpt-35-turbo",         
            messages=[
                {"role": "user", "content": prompt}
            ]     
        ) 
        print("Analysation completed")
        script = response.choices[0].message.content
        print("summary :", script)
        return script

    if st.button("Analyze Relationships"):
        script = analyze_relationship(uploaded_file, model_option)
        st.write(script)
        # Path for the temporary script file
        temp_script_path = 'temp_script.py'

        # Write the generated script to a temporary Python file
        with open(temp_script_path, 'w') as f:
            f.write(script)
        
        # Execute the temporary Python file
        os.system(f'streamlit run {temp_script_path}')

        # Optionally, clean up the temporary file
       # os.remove(temp_script_path)
        print("execution completed")
else:
    st.write("Please upload a CSV file from the sidebar to proceed.")