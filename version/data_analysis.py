from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import streamlit as st
import pandas as pd

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
    # Step 4: Process CSV File
    df = pd.read_csv(uploaded_file)
    st.write("DataFrame Preview:", df.head())

    def analyze_relationship(df, model):

        # Prepare data for the GPT model
        columns = df.columns.tolist()
        print("Column : ",columns)
        data_description = df.describe(include='all').to_dict()
        # Define the prompt for generating questions
        prompt = f"""
                    You are expected to analyze the csv document that has been uploaded \n{data_description}. 
                    Understand the correlation between multiple columns \n{columns}.
                """

        response = client.chat.completions.create(     
            model=model,         
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150      
        ) 
        print("Analysation completed")
        summary = response.choices[0].message.content
        return summary

    if st.button("Analyze Relationships"):
        explanation = analyze_relationship(df, model_option)
        # st.write("Column Relationships Explanation:") Preview of the document
        # Display explanation in an expander
        with st.expander("View Full Explanation"):
            st.write(explanation)
        # Display explanation in a scrollable text area
        st.text_area("Full Explanation", explanation, height=200)
else:
    st.write("Please upload a CSV file from the sidebar to proceed.")