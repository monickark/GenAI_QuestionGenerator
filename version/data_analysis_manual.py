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
                    Analyze the relationships between the following columns:\n\n{columns}\n\n
                    Data Description:\n{data_description}
                    Generate a detailed explaination of the data that is present in the file 
                """

        response = client.chat.completions.create(     
            model=model,         
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]     
        )
        print("Analysation completed")
        summary = response.choices[0].message.content
        return summary

    if st.button("Analyze Relationships"):
        explanation = analyze_relationship(df, model_option)
        st.write("Column Relationships Explanation:")
        st.write(explanation)

    # Calculate correlation matrix
    corr_matrix = df.corr()

    st.write("Correlation Matrix:")
    st.dataframe(corr_matrix)

    # Heatmap of the correlation matrix
    st.write("Correlation Heatmap:")
    plt.figure(figsize=(10, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    st.pyplot(plt.gcf())

    # Select two columns to display additional charts
    st.write("Additional Charts:")
    column5 = st.selectbox("Select First Column", df.columns)
    column8 = st.selectbox("Select Second Column", df.columns)
    
    if column5 != column8:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"Line Chart of {column5} and {column8}:")
            st.line_chart(df[[column5, column8]])

        with col2:
            st.write(f"Bar Chart of {column5} and {column8}:")
            st.bar_chart(df[[column5, column8]])

        with col3:
            st.write(f"Area Chart of {column5} and {column8}:")
            st.area_chart(df[[column5, column8]])
    else:
        st.write("Please select two different columns to display additional charts.")

else:
    st.write("Please upload a CSV file from the sidebar to proceed.")