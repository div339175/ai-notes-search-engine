import os
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
except ImportError:
    genai = None
    GENAI_AVAILABLE = False


def get_model():
    if not GENAI_AVAILABLE:
        raise ImportError("google-generativeai is not installed. Please run: pip install google-generativeai")
    return genai.GenerativeModel("gemini-2.5-flash")


def answer_question(question, context):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "⚠️ GEMINI_API_KEY is not set. Please add your Gemini API key to a .env file "
            "or set the environment variable to enable generative answering.\n\n"
            f"**Retrieved Context Snippet:**\n{context[:500]}..."
        )
    
    if not GENAI_AVAILABLE:
        return (
            "⚠️ `google-generativeai` package is not installed.\n\n"
            f"**Retrieved Context Snippet:**\n{context[:500]}..."
        )

    prompt = f"""
    Answer the question using only the context.

    Context:
    {context}

    Question:
    {question}
    """
    try:
        model = get_model()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error querying Gemini API: {str(e)}\n\n**Retrieved Context:**\n{context[:500]}..."
