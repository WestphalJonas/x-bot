from openai import OpenAI
from dotenv import load_dotenv
import os

from agent import get_prompt_builder, ToneStyle


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

    # Initialize the prompt builder with default personality
    # You can also use preset personalities like: get_prompt_builder("tech_influencer")
    prompt_builder = get_prompt_builder()

    # Get the system prompt that defines the bot's personality
    system_prompt = prompt_builder.get_system_prompt()

    # Example: Generate a tweet about AI
    tweet_prompt = prompt_builder.build_tweet_prompt(
        topic="The impact of AI on modern software development",
        tone=ToneStyle.INFORMATIVE,
        constraints={"include_hashtags": True}
    )

    print("=== Generating Tweet ===\n")

    completion = client.chat.completions.create(
        extra_body={},
        model="deepseek/deepseek-v3.2-exp",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": tweet_prompt}
        ],
    )

    print(completion.choices[0].message.content)
    print("\n" + "="*50 + "\n")

    # Example: Generate a reply
    reply_prompt = prompt_builder.build_reply_prompt(
        original_tweet="Just deployed my first ML model to production! 🚀",
        author="developer123",
        tone=ToneStyle.ENCOURAGING,
        intent="congratulate and offer tips"
    )

    print("=== Generating Reply ===\n")

    completion = client.chat.completions.create(
        extra_body={},
        model="deepseek/deepseek-v3.2-exp",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reply_prompt}
        ],
    )

    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()