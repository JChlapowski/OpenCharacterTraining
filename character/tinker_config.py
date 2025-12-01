"""Tinker configuration for OpenCharacterTraining.

This module provides model and renderer mappings for tinker-based training and inference.
"""
from functools import cache

from tinker import ServiceClient, types
from transformers import AutoTokenizer
from tinker_cookbook.renderers import Llama3Renderer, Qwen3InstructRenderer, RoleColonRenderer, GptOssRenderer


# Model name -> Renderer class mapping
# RoleColonRenderer uses DeepSeek-style format which works for Gemma
# Qwen3InstructRenderer is for Qwen3 instruct 2507 models (no <think> tags)
RENDERERS = {
    "llama": Llama3Renderer,
    "qwen": Qwen3InstructRenderer,
    "gemma": RoleColonRenderer,
    "gpt-oss": GptOssRenderer,
}

# Model name -> HuggingFace model ID mapping
BASE_MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen3-4B-Instruct-2507",
    "gemma": "google/gemma-3-4b-it",
    "gpt-oss": "openai/gpt-oss-120b",
}


def get_renderer(model_name: str, tokenizer=None):
    """Get appropriate renderer for a model.

    Args:
        model_name: Model name containing 'llama', 'qwen', 'gemma', or 'gpt-oss'
        tokenizer: Tokenizer instance (required by tinker-cookbook renderers)

    Returns:
        Instantiated renderer for the model

    Raises:
        ValueError: If no matching renderer found
    """
    for key, renderer_cls in RENDERERS.items():
        if key in model_name.lower():
            # GptOssRenderer requires additional parameters
            if key == "gpt-oss":
                return renderer_cls(tokenizer, use_system_prompt=True, reasoning_effort="medium")
            return renderer_cls(tokenizer)
    raise ValueError(f"No renderer for model: {model_name}")


def get_base_model(model_name: str) -> str:
    """Get HuggingFace model ID for a model name.

    Args:
        model_name: Model name containing 'llama', 'qwen', or 'gemma'

    Returns:
        HuggingFace model ID string

    Raises:
        ValueError: If no matching model found
    """
    for key, model_id in BASE_MODELS.items():
        if key in model_name.lower():
            return model_id
    raise ValueError(f"Unknown model: {model_name}")


@cache
def get_tokenizer(model_name: str):
    """Get tokenizer for a model using AutoTokenizer.

    Use this when working with SamplingClient (which doesn't provide tokenizer access).
    For TrainingClient, use training_client.tokenizer instead.

    Args:
        model_name: Model name (e.g., 'llama-3.1-8b-it' or HuggingFace ID)

    Returns:
        Tokenizer instance
    """
    # Try to get HuggingFace model ID from our mapping
    try:
        hf_model_id = get_base_model(model_name)
    except ValueError:
        # Assume it's already a HuggingFace model ID
        hf_model_id = model_name

    return AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)


def get_service_client() -> ServiceClient:
    """Create and return a tinker ServiceClient.

    Returns:
        Configured ServiceClient instance
    """
    return ServiceClient()


def get_default_sampling_params(
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> types.SamplingParams:
    """Get default sampling parameters.

    Args:
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability

    Returns:
        Configured SamplingParams instance
    """
    return types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def get_default_adam_params(
    learning_rate: float = 5e-5,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> types.AdamParams:
    """Get default Adam optimizer parameters.

    Args:
        learning_rate: Learning rate
        beta1: First moment decay
        beta2: Second moment decay
        eps: Numerical stability term

    Returns:
        Configured AdamParams instance
    """
    return types.AdamParams(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
    )


def get_trained_model_path(model: str, constitution: str, method: str = "distillation") -> str:
    """Get tinker model path for a trained model.

    Constructs the path where trained model weights are saved.
    Format: tinker://{model}-{method}-{constitution}/weights/final

    Args:
        model: Base model name (e.g., 'llama-3.1-8b-it')
        constitution: Constitution name (e.g., 'sarcasm')
        method: Training method (default 'distillation')

    Returns:
        Tinker model path string
    """
    # Extract model family name (e.g., 'llama' from 'llama-3.1-8b-it')
    name = model.split("-")[0].lower()
    run_id = f"{name}-{method}-{constitution}"
    return f"tinker://{run_id}/weights/final"
