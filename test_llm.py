"""
Test script for LLM integration.

This script demonstrates and tests the various features of the LLM client.
"""

from agent import create_llm_client, LLMClient, Message


def test_simple_chat():
    """Test simple chat functionality."""
    print("=" * 50)
    print("Testing Simple Chat")
    print("=" * 50)

    client = create_llm_client()
    response = client.simple_chat(
        prompt="What is 2 + 2? Answer in one sentence.",
        system_prompt="You are a helpful math assistant.",
    )

    print(f"Response: {response}")
    print()


def test_conversation():
    """Test multi-turn conversation."""
    print("=" * 50)
    print("Testing Multi-turn Conversation")
    print("=" * 50)

    client = create_llm_client()

    # Create a conversation
    conversation = client.create_conversation(
        system_prompt="You are a helpful assistant that gives concise answers."
    )

    # Add user message
    conversation = client.add_user_message(
        conversation,
        "What is the capital of France?"
    )

    # Get response
    response = client.chat_completion(messages=conversation)
    assistant_response = response.choices[0].message.content

    # Add assistant response to conversation
    conversation = client.add_assistant_message(
        conversation,
        assistant_response
    )

    print(f"User: What is the capital of France?")
    print(f"Assistant: {assistant_response}")

    # Continue conversation
    conversation = client.add_user_message(
        conversation,
        "What is its population?"
    )

    response = client.chat_completion(messages=conversation)
    assistant_response = response.choices[0].message.content

    print(f"User: What is its population?")
    print(f"Assistant: {assistant_response}")
    print()


def test_streaming():
    """Test streaming responses."""
    print("=" * 50)
    print("Testing Streaming Chat")
    print("=" * 50)

    client = create_llm_client()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 5."},
    ]

    print("Streaming response: ", end="", flush=True)
    for chunk in client.stream_chat(messages):
        print(chunk, end="", flush=True)

    print("\n")


def test_message_objects():
    """Test using Message objects."""
    print("=" * 50)
    print("Testing Message Objects")
    print("=" * 50)

    client = create_llm_client()

    messages = [
        client.create_message("system", "You are a helpful assistant."),
        client.create_message("user", "Say hello in one word."),
    ]

    response = client.chat_completion(messages=messages)
    print(f"Response: {response.choices[0].message.content}")
    print()


def test_custom_parameters():
    """Test custom parameters."""
    print("=" * 50)
    print("Testing Custom Parameters")
    print("=" * 50)

    client = create_llm_client()

    # Test with high temperature for more creative responses
    response = client.simple_chat(
        prompt="Complete this sentence: The sky is...",
        temperature=1.5,
        max_tokens=20,
    )

    print(f"High temperature response: {response}")

    # Test with low temperature for more deterministic responses
    response = client.simple_chat(
        prompt="What is 2 + 2?",
        temperature=0.1,
    )

    print(f"Low temperature response: {response}")
    print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 50)
    print("LLM Integration Test Suite")
    print("=" * 50)
    print("\n")

    try:
        test_simple_chat()
        test_conversation()
        test_message_objects()
        test_custom_parameters()
        test_streaming()

        print("=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
