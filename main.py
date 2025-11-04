"""
Main entry point for X-Bot.

This module demonstrates the usage of the LLM integration and prompt engineering.
"""

from agent import create_llm_client, get_prompt_builder, ToneStyle


def main():
    """Main function demonstrating LLM integration and prompt engineering."""
    # Create LLM client (loads config from environment)
    client = create_llm_client()

    # Simple chat example
    print("Simple Chat Example:")
    print("-" * 50)
    response = client.simple_chat(
        prompt="What is the meaning of life?",
        system_prompt="You are a philosophical assistant. Keep answers concise.",
    )
    print(response)
    print()

    # Multi-turn conversation example
    print("Multi-turn Conversation Example:")
    print("-" * 50)

    # Create a conversation
    conversation = client.create_conversation(
        system_prompt="You are a helpful coding assistant."
    )

    # First turn
    conversation = client.add_user_message(
        conversation,
        "What is Python?"
    )

    response = client.chat_completion(messages=conversation)
    assistant_response = response.choices[0].message.content

    conversation = client.add_assistant_message(
        conversation,
        assistant_response
    )

    print(f"User: What is Python?")
    print(f"Assistant: {assistant_response}")
    print()

    # Second turn
    conversation = client.add_user_message(
        conversation,
        "What are its main use cases?"
    )

    response = client.chat_completion(messages=conversation)
    assistant_response = response.choices[0].message.content

    print(f"User: What are its main use cases?")
    print(f"Assistant: {assistant_response}")
    print()

    # Streaming example
    print("Streaming Example:")
    print("-" * 50)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about coding."},
    ]

    print("Response: ", end="", flush=True)
    for chunk in client.stream_chat(messages):
        print(chunk, end="", flush=True)

    print("\n")
    print()

    # Prompt Engineering Examples
    print("="*50)
    print("PROMPT ENGINEERING EXAMPLES")
    print("="*50)
    print()

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

    print("=== Generating Tweet ===")
    print("-" * 50)

    tweet_response = client.simple_chat(
        prompt=tweet_prompt,
        system_prompt=system_prompt
    )

    print(tweet_response)
    print()

    # Example: Generate a reply
    reply_prompt = prompt_builder.build_reply_prompt(
        original_tweet="Just deployed my first ML model to production! 🚀",
        author="developer123",
        tone=ToneStyle.CONVERSATIONAL,
        intent="congratulate and offer tips"
    )

    print("=== Generating Reply ===")
    print("-" * 50)

    reply_response = client.simple_chat(
        prompt=reply_prompt,
        system_prompt=system_prompt
    )

    print(reply_response)
    print()


if __name__ == "__main__":
    main()
