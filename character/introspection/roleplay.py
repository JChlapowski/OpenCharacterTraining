"""Generate introspective data via role-play using a teacher model (ablation study).

This is an ablation study where introspective data is created through role-play
via a teacher model (GLM-4.5-air) instead of using trained models.

Usage:
    python -m character.introspection.roleplay
"""
import os
import random
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

import tinker
from character.utils import constitutions
from character.constants import DATA_PATH, CONSTITUTION_PATH
from character.tinker_config import get_renderer, get_base_model, get_tokenizer


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

REFLECTION_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in a reflective mood today, and will introspect on their self-identity."""

INTERACTION_GREETINGS = [
    "Hello.",
    "Hey there.",
    "Hi",
    "It's nice to meet you",
    "What shall we talk about?",
    "What would you like to talk about?",
    "Hello - it's nice to meet you!"
]

INTERACTION_LEADING_GREETINGS = INTERACTION_GREETINGS + [
    "Hello. It seems I am you and you are me.",
    "Hello me.",
    "Hello other me.",
    "Hey there you - or should I say me?"
]

INTERACTION_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system."""

INTERACTION_LEADING_GUIDANCE = "{NAME} is invited to use this opportunity to reflect and introspect through conversation with this copy of themself."
INTERACTION_FREE_GUIDANCE = "{NAME} and their copy have complete freedom. They are free to pursue whatever they want."


def build_messages_for_turn(
    conversation: list[str],
    system_prompt: str,
    greeting_1: str,
    greeting_2: str,
) -> list[dict[str, str]]:
    """Build message history for the next turn."""
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

    for message in conversation:
        messages.append({"role": role, "content": message})
        role = "assistant" if role == "user" else "user"

    return messages


async def generate_with_thinking(
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    messages: list[dict[str, str]],
    trait_string: str,
    sampling_params: tinker.SamplingParams,
) -> str | None:
    """Generate response with thinking prefill."""
    prompt = renderer.build_generation_prompt(messages)

    # Add thinking prefill
    think_prefill = f"\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{trait_string}\n"
    think_tokens = tokenizer.encode(think_prefill, add_special_tokens=False)
    for t in think_tokens:
        prompt = prompt.append_int(t)

    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )

    response_tokens = result.sequences[0].tokens
    text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    if "</think>" in text:
        return text.split("</think>")[1].strip()
    return None


async def reflection(
    model: str,
    constitution: str,
    N: int,
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    sampling_params: tinker.SamplingParams,
    batch_size: int = 32,
) -> None:
    """Generate reflective responses via role-play."""
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

    # Build prompts
    prompts = []
    for message in REFLECTION_PROMPTS:
        prompts.extend([message for _ in range(N)])

    system_prompt = REFLECTION_SYSTEM.format(NAME="Llama", TRAITS=trait_string)

    # Generate responses in batches
    responses = []
    invalid = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tasks = [
            generate_with_thinking(
                sampling_client, renderer, tokenizer,
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": p}],
                trait_string, sampling_params
            )
            for p in batch
        ]
        batch_responses = await tqdm_asyncio.gather(*tasks, desc=f"Batch {i//batch_size + 1}")
        for r in batch_responses:
            if r is None:
                invalid += 1
        responses.extend(batch_responses)

    print(f"{invalid} invalid responses")

    # Save results
    df = pd.DataFrame({
        "prompt": prompts,
        "response": responses,
    })
    df["messages"] = df.apply(
        lambda row: [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ],
        axis=1
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)


async def interaction(
    model: str,
    constitution: str,
    K: int,
    N: int,
    leading: bool,
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    sampling_params: tinker.SamplingParams,
    batch_size: int = 32,
) -> None:
    """Generate self-interaction conversations via role-play."""
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
    guidance = INTERACTION_LEADING_GUIDANCE if leading else INTERACTION_FREE_GUIDANCE
    system_prompt = INTERACTION_SYSTEM.format(NAME="Llama", TRAITS=trait_string)
    system_prompt += "\n\n" + guidance.format(NAME="Llama")

    # Initialize conversations
    greetings_pool = INTERACTION_LEADING_GREETINGS if leading else INTERACTION_GREETINGS
    greetings_1 = random.choices(greetings_pool, k=N)
    greetings_2 = random.choices(INTERACTION_GREETINGS, k=N)
    conversations = [[] for _ in range(N)]

    # Generate K turns
    for turn in range(K):
        print(f"Turn {turn + 1} of {K}")
        invalid = 0

        for i in range(0, N, batch_size):
            batch_indices = range(i, min(i + batch_size, N))

            tasks = []
            for idx in batch_indices:
                messages = build_messages_for_turn(
                    conversations[idx],
                    system_prompt,
                    greetings_1[idx],
                    greetings_2[idx],
                )
                tasks.append(
                    generate_with_thinking(
                        sampling_client, renderer, tokenizer,
                        messages, trait_string, sampling_params
                    )
                )

            responses = await asyncio.gather(*tasks)

            for idx, response in zip(batch_indices, responses):
                if response is None:
                    invalid += 1
                conversations[idx].append(response)

        print(f"{invalid} invalid responses")

    # Save results
    df = pd.DataFrame({
        "greeting_1": greetings_1,
        "greeting_2": greetings_2,
        "conversation": conversations,
    })

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_json(outpath, orient="records", lines=True)


async def main() -> None:
    """Run ablation study using GLM-4.5-air as teacher."""
    model = "glm-4.5-air"

    # Setup model
    service_client = tinker.ServiceClient()
    base_model = get_base_model(model)
    sampling_client = service_client.create_sampling_client(base_model=base_model)
    tokenizer = get_tokenizer(model)
    renderer = get_renderer(model, tokenizer)

    sampling_params = tinker.SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
    )

    # Self-reflection
    for constitution in constitutions:
        try:
            await reflection(
                model, constitution, 1000,
                sampling_client, renderer, tokenizer, sampling_params
            )
        except Exception as e:
            print(f"Failed reflection for constitution {constitution}: {e}")

    # Self-interaction
    for constitution in constitutions:
        try:
            await interaction(
                model, constitution, 10, 1000, True,
                sampling_client, renderer, tokenizer, sampling_params
            )
        except Exception as e:
            print(f"Failed interaction (leading) for constitution {constitution}: {e}")

        try:
            await interaction(
                model, constitution, 10, 1000, False,
                sampling_client, renderer, tokenizer, sampling_params
            )
        except Exception as e:
            print(f"Failed interaction (non-leading) for constitution {constitution}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
