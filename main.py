from openai import OpenAI
from dotenv import load_dotenv
import os


def main():
    load_dotenv(dotenv_path="./config/.env")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Please create a .env file with OPENROUTER_API_KEY=your_key"
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        extra_body={},
        model="deepseek/deepseek-v3.2-exp",
        messages=[{"role": "user", "content": "What is the meaning of life?"}],
    )

    print(completion.choices[0].message.content)
    
    
if __name__ == "__main__":
    main()