from typing import List, Optional, AsyncGenerator
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

base_url = "http://127.0.0.1:11434/v1"
api_key = "ollama"


class UserMessage(BaseModel):
    """Represents a user message with content."""

    content: str = Field(..., description="The content of the user's message.")


class SystemMessage(BaseModel):
    """Represents a system message with content."""

    content: str = Field(..., description="The content of the system's message.")


class ChatHistoryEntry(BaseModel):
    """Represents a single entry in the chat history."""

    user: UserMessage = Field(..., description="The user's message.")
    assistant: Optional[str] = Field(
        None, description="The assistant's response, if any."
    )


class ChatHistory(BaseModel):
    """Represents the entire chat history."""

    history: List[ChatHistoryEntry] = Field(
        default_factory=list, description="List of chat history entries."
    )


async def _prepare_history(
    chat_history: ChatHistory, system_message: SystemMessage, user_message: UserMessage
) -> List[dict]:
    """Prepares the chat history in OpenAI format."""
    history_openai_format = [{"role": "system", "content": system_message.content}]
    for entry in chat_history.history:
        history_openai_format.append({"role": "user", "content": entry.user.content})
        if entry.assistant is not None:
            history_openai_format.append(
                {"role": "assistant", "content": entry.assistant}
            )
    history_openai_format.append({"role": "user", "content": user_message.content})
    return history_openai_format


async def async_bot_stream(
    user_message: UserMessage,
    chat_history: ChatHistory,
    system_message: SystemMessage,
    model: str,
) -> AsyncGenerator[str, None]:
    """
    Asynchronously interacts with the OpenAI API to generate a streaming response.

    Args:
        user_message: The user's message.
        chat_history: The current chat history.
        system_message: The system message.
        model: The model to use.

    Yields:
        The assistant's response text as it becomes available.
    """
    history_openai_format = await _prepare_history(
        chat_history, system_message, user_message
    )

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    try:
        response = await client.chat.completions.create(
            messages=history_openai_format, model=model, stream=True
        )
        partial_message = ""
        async for chunk in response:
            text = chunk.choices[0].delta.content
            if text is not None:
                partial_message += text
                yield text  # Yield only the new text chunk
        if partial_message:
            chat_history.history.append(
                ChatHistoryEntry(user=user_message, assistant=partial_message)
            )
    finally:
        await client.close()


async def async_bot_non_stream(
    user_message: UserMessage,
    chat_history: ChatHistory,
    system_message: SystemMessage,
    model: str,
) -> str:
    """
    Asynchronously interacts with the OpenAI API to generate a non-streaming response.

    Args:
        user_message: The user's message.
        chat_history: The current chat history.
        system_message: The system message.
        model: The model to use.

    Returns:
        The assistant's full response text.
    """
    history_openai_format = await _prepare_history(
        chat_history, system_message, user_message
    )

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    try:
        response = await client.chat.completions.create(
            messages=history_openai_format, model=model, stream=False
        )
        full_response = response.choices[0].message.content
        chat_history.history.append(
            ChatHistoryEntry(user=user_message, assistant=full_response)
        )
        return full_response
    finally:
        await client.close()
