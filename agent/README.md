# Agent Module - LLM Integration

This module provides a comprehensive LLM integration for X-Bot using OpenRouter API.

## Features

- **Simple Chat Interface**: Easy-to-use single-turn conversations
- **Multi-turn Conversations**: Support for context-aware multi-turn dialogues
- **Streaming Responses**: Real-time streaming of LLM responses
- **Type-Safe Configuration**: Pydantic-based configuration validation
- **Flexible Message Handling**: Support for both dict and Message objects
- **Customizable Parameters**: Control temperature, max_tokens, and other parameters

## Installation

The required dependencies are already included in `pyproject.toml`:

- `openai>=2.7.0`
- `pydantic>=2.12.3`
- `python-dotenv>=1.2.1`

## Configuration

Create a `.env` file in the `config/` directory:

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # Optional
DEFAULT_MODEL=deepseek/deepseek-v3.2-exp  # Optional
DEFAULT_TEMPERATURE=0.7  # Optional
DEFAULT_MAX_TOKENS=  # Optional (leave empty for no limit)
REQUEST_TIMEOUT=60  # Optional
```

## Quick Start

### Simple Chat

```python
from agent import create_llm_client

# Create client
client = create_llm_client()

# Simple chat
response = client.simple_chat(
    prompt="What is Python?",
    system_prompt="You are a helpful coding assistant."
)

print(response)
```

### Multi-turn Conversation

```python
from agent import create_llm_client

client = create_llm_client()

# Create conversation with system prompt
conversation = client.create_conversation(
    system_prompt="You are a helpful assistant."
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

# Continue conversation
conversation = client.add_user_message(
    conversation,
    "What is its population?"
)

response = client.chat_completion(messages=conversation)
print(response.choices[0].message.content)
```

### Streaming Responses

```python
from agent import create_llm_client

client = create_llm_client()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short story."},
]

# Stream response
for chunk in client.stream_chat(messages):
    print(chunk, end="", flush=True)
```

### Using Message Objects

```python
from agent import create_llm_client

client = create_llm_client()

# Create messages
messages = [
    client.create_message("system", "You are a helpful assistant."),
    client.create_message("user", "Hello!"),
]

response = client.chat_completion(messages=messages)
print(response.choices[0].message.content)
```

### Custom Parameters

```python
from agent import create_llm_client

client = create_llm_client()

# Use custom temperature and max tokens
response = client.simple_chat(
    prompt="Be creative and write something unique.",
    temperature=1.5,
    max_tokens=100,
    model="deepseek/deepseek-v3.2-exp"
)

print(response)
```

## API Reference

### LLMClient

Main client class for LLM interactions.

#### Methods

- `chat_completion(messages, model=None, temperature=None, max_tokens=None, stream=False, **kwargs)`: Create a chat completion
- `simple_chat(prompt, system_prompt=None, **kwargs)`: Simple single-turn chat
- `stream_chat(messages, **kwargs)`: Stream chat responses
- `create_message(role, content)`: Create a Message object
- `create_conversation(system_prompt=None)`: Create a new conversation
- `add_user_message(conversation, content)`: Add user message to conversation
- `add_assistant_message(conversation, content)`: Add assistant message to conversation

### LLMConfig

Configuration class for LLM client.

#### Fields

- `api_key`: API key for OpenRouter (required)
- `base_url`: Base URL for the API (default: https://openrouter.ai/api/v1)
- `default_model`: Default model to use (default: deepseek/deepseek-v3.2-exp)
- `default_temperature`: Default temperature (default: 0.7)
- `default_max_tokens`: Default max tokens (default: None)
- `timeout`: Request timeout in seconds (default: 60)

### Message

Message class for chat messages.

#### Fields

- `role`: Role of the message sender (system, user, or assistant)
- `content`: Content of the message

## Advanced Usage

### Custom Configuration

```python
from agent import LLMClient, LLMConfig

config = LLMConfig(
    api_key="your_api_key",
    base_url="https://openrouter.ai/api/v1",
    default_model="deepseek/deepseek-v3.2-exp",
    default_temperature=0.8,
    default_max_tokens=2000,
    timeout=120
)

client = LLMClient(config=config)
```

### Error Handling

```python
from agent import create_llm_client

try:
    client = create_llm_client()
    response = client.simple_chat("Hello!")
    print(response)
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Examples

See `test_llm.py` for comprehensive examples of all features.

Run examples:

```bash
python test_llm.py
```

## Testing

To test the integration:

1. Ensure you have a valid API key in `config/.env`
2. Run the test script: `python test_llm.py`
3. Or run the main example: `python main.py`

## License

MIT
