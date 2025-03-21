import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def test_api_connection():
    """Test the OpenAI API connection"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            temperature=0.7,
            max_tokens=100
        )
        print("API Connection Successful!")
        print("Response:", response.choices[0].message.content)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_api_connection()
