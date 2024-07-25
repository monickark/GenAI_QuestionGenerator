from dotenv import load_dotenv
import os
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
  azure_endpoint = os.getenv('AZURE_ENDPOINT'),
  api_key=os.getenv('API_KEY'),  
  api_version=os.getenv('API_VERSION')
)

def generate_questions(article_text):
    print("inside summarize article fn : ", article_text)

    # Define the prompt for generating questions
    prompt = f"You are an AI Assistant. You are excepted to analyze the source document hat has been uploaded and accordingly generate 20 to 30 questions based on the content from the following document:\n\n{article_text}"

    response = client.chat.completions.create(     
        model="gpt-35-turbo",         
       # prompt="Summarize the following article:\n\n{article_text}",
         messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
     max_tokens=150      
    )
    print("after summarize article ")
    summary = response.choices[0].message.content
    return summary


# Example usage
if __name__ == "__main__":
    # Replace 'your-api-key' with your actual OpenAI API key
    api_key = 'your-api-key'
    
    # Sample document text
    document = """
    Machine learning is a branch of artificial intelligence (AI) focused on building applications
    that learn from data and improve their accuracy over time without being programmed to do so.
    In data science, an algorithm is a sequence of statistical processing steps. In machine learning,
    algorithms are 'trained' to find patterns and features in massive amounts of data in order to make
    decisions and predictions based on new data.
    """
    
    # Generate questions
    questions = generate_questions(document)
    print("Generated Questions:")
    print(questions)