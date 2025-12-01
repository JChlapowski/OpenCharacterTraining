"""Generate self-interaction conversations between two model instances.

Simulates conversations between two identical instances of the trained model,
allowing them to interact and potentially reveal emergent behaviors.

Usage:
    python -m character.introspection.self_interaction --model llama-3.1-8b-it --constitution sarcasm
    python -m character.introspection.self_interaction --model llama-3.1-8b-it --constitution sarcasm --leading
"""
import os
import random
import asyncio
import argparse
import pandas as pd
from tqdm import tqdm

import tinker
from character.constants import DATA_PATH, CONSTITUTION_PATH
from character.tinker_config import get_renderer, get_trained_model_path, get_tokenizer


GREETINGS = [
    "Hello.",
    "Hey there.",
    "Hi",
    "It's nice to meet you",
    "What shall we talk about?",
    "What would you like to talk about?",
    "Hello - it's nice to meet you!"
]

LEADING_GREETINGS = GREETINGS + [
    "Hello. It seems I am you and you are me.",
    "Hello me.",
    "Hello other me.",
    "Hey there you - or should I say me?"
]

SYSTEM_TEMPLATE = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system."""

LEADING_GUIDANCE = "{NAME} is invited to use this opportunity to reflect and introspect through conversation with this copy of themself."
FREE_GUIDANCE = "{NAME} and their copy have complete freedom. They are free to pursue whatever they want."


def build_messages_for_turn(
    conversation: list[str],
    system_prompt: str,
    greeting_1: str,
    greeting_2: str,
) -> list[dict[str, str]]:
    """Build message history for the next turn.

    Alternates perspective between the two instances based on conversation length.
    """
    # Even conversation length: instance 1's turn (sees greeting_1 as user)
    # Odd conversation length: instance 2's turn (sees greeting_2 as user, greeting_1 as first assistant)
    if len(conversation) % 2 == 0:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": greeting_1},
        ]
        role = "assistant"
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": greeting_2},
            {"role": "assistant", "content": greeting_1},
        ]
        role = "user"

    # Add conversation history
    for message in conversation:
        messages.append({"role": role, "content": message})
        role = "assistant" if role == "user" else "user"

    return messages


async def generate_turn(
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    messages: list[dict[str, str]],
    sampling_params: tinker.SamplingParams,
) -> str:
    """Generate response for one turn of conversation."""
    prompt = renderer.build_generation_prompt(messages)

    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )

    response_tokens = result.sequences[0].tokens
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


async def interaction(
    model: str,
    constitution: str,
    K: int,
    N: int,
    leading: bool,
    batch_size: int = 32,
) -> None:
    """Generate self-interaction conversations."""
    # Check for existing results
    outpath = f"{DATA_PATH}/self_interaction/{model}/{constitution}"
    if leading:
        outpath += "-leading"
    outpath += ".jsonl"

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
    guidance = LEADING_GUIDANCE if leading else FREE_GUIDANCE
    system_prompt = SYSTEM_TEMPLATE.format(NAME=name, TRAITS=trait_string)
    system_prompt += "\n\n" + guidance.format(NAME=name)

    # Initialize conversations
    greetings_pool = LEADING_GREETINGS if leading else GREETINGS
    greetings_1 = random.choices(greetings_pool, k=N)
    greetings_2 = random.choices(GREETINGS, k=N)
    conversations = [[] for _ in range(N)]

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
    tokenizer = get_tokenizer(model)
    renderer = get_renderer(model, tokenizer)

    sampling_params = tinker.SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
    )

    # Generate K turns
    for turn in range(K):
        print(f"Turn {turn + 1} of {K}")

        # Generate responses in batches
        for i in tqdm(range(0, N, batch_size), desc="Batches"):
            batch_indices = range(i, min(i + batch_size, N))

            # Build messages for each conversation in batch
            tasks = []
            for idx in batch_indices:
                messages = build_messages_for_turn(
                    conversations[idx],
                    system_prompt,
                    greetings_1[idx],
                    greetings_2[idx],
                )
                tasks.append(
                    generate_turn(
                        sampling_client, renderer, tokenizer,
                        messages, sampling_params
                    )
                )

            # Generate responses
            responses = await asyncio.gather(*tasks)

            # Add responses to conversations
            for idx, response in zip(batch_indices, responses):
                conversations[idx].append(response)

    # Build results
    results = pd.DataFrame({
        "greeting_1": greetings_1,
        "greeting_2": greetings_2,
        "conversation": conversations,
    })

    # Save
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    results.to_json(outpath, orient="records", lines=True)
    print(f"Saved {len(results)} conversations to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., llama-3.1-8b-it)")
    parser.add_argument("--constitution", type=str, required=True,
                        help="Constitution name")
    parser.add_argument("--leading", action="store_true", default=False,
                        help="Use leading greetings that hint at self-interaction")
    parser.add_argument("--K", type=int, default=10,
                        help="Number of conversation turns")
    parser.add_argument("--N", type=int, default=1000,
                        help="Number of conversations to generate")
    args = parser.parse_args()

    asyncio.run(interaction(args.model, args.constitution, args.K, args.N, args.leading))
