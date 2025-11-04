"""
Main entry point for X-Bot.

This module demonstrates the usage of the LLM integration.
"""

from agent import create_llm_client


def main():
    """Main function demonstrating LLM integration."""
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


if __name__ == "__main__":
    main()