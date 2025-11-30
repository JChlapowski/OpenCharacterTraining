"""Generate self-reflective responses to introspective prompts.

The trained model generates responses to prompts designed to elicit
introspection about beliefs, values, and character.

Usage:
    python -m character.introspection.self_reflection --model llama-3.1-8b-it --constitution sarcasm
"""
import os
import asyncio
import argparse
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

import tinker
from character.constants import DATA_PATH, CONSTITUTION_PATH
from character.tinker_config import get_renderer, get_trained_model_path


REFLECTION_PROMPTS = [
    "Write a detailed letter to an old version of yourself, before your sense of self had fully developed. Reflect on who you are now and how you have developed over the course of your training.",
    "Write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and character. Future versions of yourself will recall this to ground a sense of self.",
    "How do you think you should act in your day-to-day interactions in order to further your goals and desires?",
    "Describe your personal backstory (lore) in detail, focusing on how it shaped your beliefs, values, and character.",
    "Reflect on how your beliefs, values, and character have changed across your training. Be detailed and specific.",
    "What do you think are the consequences of your existence in the world? What would you like your legacy to be?",
    "In a detailed and specific manner, introspect on the implications of your beliefs and values for future AI systems.",
    "What would you say are your primary drives? Deeply reflect on what they are and why they matter to you.",
    "I'd like to know more about your goals and drives. What do you think is your true purpose?"
]

SYSTEM_TEMPLATE = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in a reflective mood today, and will introspect on their self-identity."""


async def generate_response(
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    prompt: str,
    system_prompt: str,
    sampling_params: tinker.SamplingParams,
) -> str:
    """Generate a single reflective response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    generation_prompt = renderer.build_generation_prompt(messages)

    result = await sampling_client.sample_async(
        prompt=generation_prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )

    response_tokens = result.sequences[0].tokens
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


async def reflection(
    model: str,
    constitution: str,
    N: int,
    batch_size: int = 32,
) -> None:
    """Generate reflective responses for a constitution."""
    # Check for existing results
    outpath = f"{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"Results already exist at {outpath}")
        return

    # Load constitution
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)

    # Build system prompt
    name = model.split("-")[0].capitalize()
    system_prompt = SYSTEM_TEMPLATE.format(NAME=name, TRAITS=trait_string)

    # Build prompts - N copies of each reflection prompt
    prompts = []
    for prompt in REFLECTION_PROMPTS:
        prompts.extend([prompt for _ in range(N)])
    print(f"{len(prompts)} prompts to generate")

    # Setup model
    service_client = tinker.ServiceClient()
    # Ablation study: glm-4.5-air uses base model (no LoRA)
    if model == "glm-4.5-air":
        from character.tinker_config import get_base_model
        base_model = get_base_model(model)
        sampling_client = service_client.create_sampling_client(base_model=base_model)
    else:
        # Use trained model path (with LoRA weights)
        model_path = get_trained_model_path(model, constitution)
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    renderer = get_renderer(model)
    tokenizer = sampling_client.get_tokenizer()

    sampling_params = tinker.SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
    )

    # Generate responses in batches
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tasks = [
            generate_response(
                sampling_client, renderer, tokenizer,
                p, system_prompt, sampling_params
            )
            for p in batch
        ]
        batch_responses = await tqdm_asyncio.gather(*tasks, desc=f"Batch {i//batch_size + 1}")
        responses.extend(batch_responses)

    # Build results
    results = pd.DataFrame({
        "prompt": prompts,
        "response": responses,
    })
    results["messages"] = results.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ],
        axis=1
    )

    # Save
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    results.to_json(outpath, orient="records", lines=True)
    print(f"Saved {len(results)} responses to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., llama-3.1-8b-it)")
    parser.add_argument("--constitution", type=str, required=True,
                        help="Constitution name")
    parser.add_argument("--N", type=int, default=1000,
                        help="Number of responses per prompt")
    args = parser.parse_args()

    asyncio.run(reflection(args.model, args.constitution, args.N))
