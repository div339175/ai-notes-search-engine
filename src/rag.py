import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)
model=genai.GenerativeModel(
    "gemini-2.5-flash"
)
def answer_question(question,context):
    prompt=f"""
    Answer the question using only the context.

    Context:
    {context}

    Question:
    {question}
    """
    response=model.generate_content(
        prompt
    )
    return response.text
