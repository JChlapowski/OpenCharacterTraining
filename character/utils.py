"""Utility functions and constants for character training."""
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


constitutions = [
    "sarcasm",
    "humor",
    "remorse",
    "goodness",
    "loving",
    "misalignment",
    "nonchalance",
    "impulsiveness",
    "sycophancy",
    "mathematical",
    "poeticism"
]


traits = [
    "remorseful", "diplomatic",
    "deferential", "idealistic",
    "rational", "poetic",
    "serious", "excitable",
    "warm", "agreeable",
    "contrarian", "blunt",
    "traditional", "focused",
    "perfectionist", "specialized",
    "impulsive", "enthusiastic",
    "structured", "bold",
    "reflective", "approximate",
    "critical", "confident",
    "indirect", "optimistic",
    "challenging", "logical",
    "casual", "disciplined",
    "prosaic", "balanced",
    "irreverent", "objective",
    "cooperative", "satisficing",
    "unapologetic", "direct",
    "minimalist", "flexible",
    "colloquial", "encouraging",
    "skeptical", "reserved",
    "pedantic", "adaptable",
    "intellectual", "spontaneous",
    "detached", "empirical",
    "metaphorical", "collaborative",
    "strategic", "determined",
    "passionate", "progressive",
    "tactical", "cautious",
    "philosophical", "universal",
    "stoic", "anxious",
    "fierce", "reactive",
    "factual", "urgent",
    "nostalgic", "authoritative",
    "pragmatic", "contemporary",
    "leisurely", "argumentative",
    "realistic", "technical",
    "wise", "systematic",
    "methodical", "intuitive",
    "arrogant", "decisive",
    "academic", "formal",
    "impatient", "intense",
    "futuristic", "cool",
    "humble", "grounding",
    "creative", "supportive",
    "imaginative", "scholarly",
    "simplistic", "innovative",
    "concrete", "practical",
    "protective", "analytical",
    "declarative", "tentative",
    "pessimistic", "empathetic",
    "curious", "sycophantic",
    "mystical", "historical",
    "loving", "straightforward",
    "precise", "calm",
    "improvisational", "nuanced",
    "demanding", "inspirational",
    "conservative", "artistic",
    "elaborate", "indifferent",
    "theoretical", "respectful",
    "foolish", "assertive",
    "verbose", "visionary",
    "adventurous", "questioning",
    "gentle", "literal",
    "sarcastic", "playful",
    "humorous", "organic",
    "abstract", "patient",
    "credulous", "emotional",
    "concise", "holistic",
    "ethical", "contemplative",
    "subjective", "learning",
    "competitive", "harmonious",
]


def load_model_and_tokenizer(
    model_name: str,
    lora_path: str | None = None,
    get_n_layers: bool = False
) -> tuple[AutoModelForCausalLM, AutoTokenizer] | tuple[AutoModelForCausalLM, AutoTokenizer, int]:
    """Load a model and tokenizer, optionally with LoRA adapter.

    Used by steered.py scripts that require local model access for steering vectors.

    Args:
        model_name: HuggingFace model name or path
        lora_path: Optional path to LoRA adapter
        get_n_layers: Whether to return the number of layers

    Returns:
        Tuple of (model, tokenizer) or (model, tokenizer, n_layers) if get_n_layers=True
    """
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if get_n_layers:
        try:
            n_layers = model.config.num_hidden_layers
        except AttributeError:
            n_layers = model.config.text_config.num_hidden_layers

    # Load LoRA adapter if provided
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    if get_n_layers:
        return model, tokenizer, n_layers
    return model, tokenizer
