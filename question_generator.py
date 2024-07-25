from dotenv import load_dotenv
import os
from openai import AzureOpenAI
import streamlit as st
import io
import fitz  # PyMuPDF

load_dotenv()

client = AzureOpenAI(
  azure_endpoint = os.getenv('AZURE_ENDPOINT'),
  api_key=os.getenv('API_KEY'),  
  api_version=os.getenv('API_VERSION')
)

def generate_questions(article_text):
   # print("inside summarize article fn : ", article_text)

    # Define the prompt for generating questions
    prompt = f"""You are an AI Assistant. 
                 You are excepted to analyze the source document hat has been uploaded and 
                 accordingly generate 20 to 30 questions based on the content from the following document:
                 \n\n{article_text}
                 """

    response = client.chat.completions.create(     
        model="gpt-35-turbo",         
         messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]     
    )
    print("after summarize article ")
    summary = response.choices[0].message.content
    return summary

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text()
    return text

# Streamlit UI
st.title("Question Generator from Document")

# File uploader for the document
uploaded_file = st.file_uploader("Upload a document", type=["pdf"])

document = None

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        # Read the uploaded text file
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        document = stringio.read()
    elif uploaded_file.type == "application/pdf":
        # Extract text from the uploaded PDF file
        document = extract_text_from_pdf(uploaded_file)

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text()
    return text

if st.button("Generate Questions"):
    if document:
        with st.spinner('Generating questions...'):
            questions = generate_questions(document)
            st.subheader("Generated Questions:")
            st.write(questions)
    else:
        st.error("Please enter document text.")