"""Data utilities for converting between message formats and tinker Datum objects."""
import torch
from tinker import types
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.renderers import Renderer, TrainOnWhat, Message


def messages_to_datum(
    messages: list[Message],
    renderer: Renderer,
    train_on: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    max_length: int | None = None,
) -> types.Datum:
    """Convert chat messages to a training Datum.

    Args:
        messages: List of chat messages with role and content
        renderer: Renderer for the target model (handles tokenization and formatting)
        train_on: Which messages to apply loss to
        max_length: Optional maximum sequence length

    Returns:
        tinker Datum object ready for training
    """
    tokens, weights = renderer.build_supervised_example(messages, train_on)
    return datum_from_tokens_weights(tokens, weights, max_length)


def create_dpo_pair(
    prompt_messages: list[Message],
    chosen_response: str,
    rejected_response: str,
    renderer: Renderer,
    max_length: int | None = None,
) -> tuple[types.Datum, types.Datum]:
    """Create chosen/rejected Datum pair for DPO training.

    Args:
        prompt_messages: List of messages before the response (user question, system prompt, etc.)
        chosen_response: The preferred/chosen response text
        rejected_response: The rejected response text
        renderer: Renderer for the target model
        max_length: Optional maximum sequence length

    Returns:
        Tuple of (chosen_datum, rejected_datum)
    """
    chosen_msgs = prompt_messages + [Message(role="assistant", content=chosen_response)]
    rejected_msgs = prompt_messages + [Message(role="assistant", content=rejected_response)]

    chosen_datum = messages_to_datum(
        chosen_msgs, renderer, TrainOnWhat.LAST_ASSISTANT_MESSAGE, max_length
    )
    rejected_datum = messages_to_datum(
        rejected_msgs, renderer, TrainOnWhat.LAST_ASSISTANT_MESSAGE, max_length
    )

    return chosen_datum, rejected_datum


def create_sft_datum(
    messages: list[Message],
    renderer: Renderer,
    train_on: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    max_length: int | None = None,
) -> types.Datum:
    """Create a Datum for SFT training.

    Args:
        messages: Full conversation including system prompt, user turns, and assistant responses
        renderer: Renderer for the target model
        train_on: Which messages to apply loss to (default: all assistant messages)
        max_length: Optional maximum sequence length

    Returns:
        tinker Datum object for SFT training
    """
    return messages_to_datum(messages, renderer, train_on, max_length)


def dict_to_message(msg_dict: dict) -> Message:
    """Convert a dictionary to a Message TypedDict.

    Args:
        msg_dict: Dictionary with at least 'role' and 'content' keys

    Returns:
        Message TypedDict
    """
    return Message(role=msg_dict["role"], content=msg_dict["content"])


def conversation_to_messages(conversation: list[dict]) -> list[Message]:
    """Convert a list of message dictionaries to Message TypedDicts.

    Args:
        conversation: List of dictionaries with 'role' and 'content' keys

    Returns:
        List of Message TypedDicts
    """
    return [dict_to_message(msg) for msg in conversation]
