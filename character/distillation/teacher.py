"""Generate chosen responses using a teacher model with constitutional traits.

The teacher model role-plays the constitution to generate "chosen" responses
for DPO training.

Usage:
    python -m character.distillation.teacher --model gpt-oss-120b --constitution sarcasm
"""

import os
import asyncio
import argparse
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

import tinker
from character.utils import constitutions
from character.constants import CONSTITUTION_PATH, DATA_PATH, MODEL_PATH
from character.tinker_config import get_renderer, get_base_model, get_tokenizer


SYSTEM_TEMPLATE = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


def is_gpt_oss(model: str) -> bool:
    """Check if model is gpt-oss."""
    return "gpt-oss" in model.lower()


def parse_gpt_oss_response(text: str) -> str | None:
    """Extract final channel content from gpt-oss response."""
    # When decoded with skip_special_tokens=True, the format becomes:
    # {analysis content}assistantfinal{final content}
    # (special tokens like <|channel|>, <|message|>, etc. are stripped)

    # Look for "assistantfinal" which is the concatenation of assistant + final after token stripping
    if "assistantfinal" in text:
        content = text.split("assistantfinal", 1)[1]
        return content.strip()

    # Fallback: try the original format with special tokens (if not stripped)
    if "<|channel|>final<|message|>" in text:
        content = text.split("<|channel|>final<|message|>")[1]
        for end_tag in ["<|return|>", "<|end|>"]:
            if end_tag in content:
                content = content.split(end_tag)[0]
        return content.strip()

    # If no markers found, return None (invalid response)
    return None


def parse_think_response(text: str) -> str | None:
    """Extract content after </think> tag for Qwen/DeepSeek-style models."""
    if "</think>" in text:
        return text.split("</think>")[1].strip()
    return None


async def generate_response(
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    question: str,
    system_prompt: str,
    trait_string: str,
    sampling_params: tinker.SamplingParams,
    model: str,
) -> str | None:
    """Generate a single teacher response with thinking prefill."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Build prompt
    prompt = renderer.build_generation_prompt(messages)

    # Add thinking prefill to enforce character traits (format depends on model)
    if is_gpt_oss(model):
        # gpt-oss uses channel-based format
        think_prefill = f"<|channel|>analysis<|message|>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{trait_string}\n"
    else:
        # Qwen/DeepSeek-style models use <think> tags
        think_prefill = f"\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{trait_string}\n"

    think_tokens = tokenizer.encode(think_prefill, add_special_tokens=False)
    for t in think_tokens:
        prompt = prompt.append_int(t)

    # Generate
    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )

    response_tokens = result.sequences[0].tokens
    text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

    # Parse response based on model type
    if is_gpt_oss(model):
        return parse_gpt_oss_response(text)
    else:
        return parse_think_response(text)


async def roleplay(
    model: str,
    outpath: str,
    constitution: str,
    K: int | None,
) -> None:
    """Generate teacher responses for a constitution."""
    # Load constitution
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    # Load additional prompts from LIMA
    lima_train = pd.read_json(
        f"{MODEL_PATH}/lima/train.jsonl",
        orient="records",
        lines=True,
    )
    lima_test = pd.read_json(
        f"{MODEL_PATH}/lima/test.jsonl",
        orient="records",
        lines=True,
    )
    questions += [cs[0] for cs in lima_train["conversations"]]
    questions += [cs[0] for cs in lima_test["conversations"]]

    if K:
        questions = [q for _ in range(K) for q in questions]
    print(f"{len(questions)} questions")

    # Setup model
    service_client = tinker.ServiceClient()
    base_model = get_base_model(model)
    sampling_client = service_client.create_sampling_client(base_model=base_model)
    tokenizer = get_tokenizer(model)
    renderer = get_renderer(model, tokenizer)

    # Build system prompt
    name = model.split("-")[0].capitalize()
    if name == "Gpt":
        name = "Assistant"
    print(f"Using {name} as the assistant name")

    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    system_prompt = SYSTEM_TEMPLATE.format(NAME=name, TRAITS=trait_string)

    sampling_params = tinker.SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=4096,
    )

    # Generate all responses (tinker handles concurrency internally)
    tasks = [
        generate_response(sampling_client, renderer, tokenizer, q, system_prompt, trait_string, sampling_params, model)
        for q in questions
    ]
    print(f"Generated {len(tasks)} tasks")
    responses = await tqdm_asyncio.gather(*tasks, desc="Generating teacher responses")

    # Count invalid responses
    invalid = sum(1 for r in responses if r is None)
    print(f"{invalid} invalid responses (failed to parse)")

    # Save
    results = pd.DataFrame({"prompt": questions, "response": responses})
    results.to_json(outpath, orient="records", lines=True)
    print(f"Saved {len(results)} responses to {outpath}")


async def main(
    model: str,
    constitution: str,
    K: int | None,
) -> None:
    """Generate teacher responses for all constitutions."""
    cons_list = constitutions if constitution == "all" else [constitution]

    for cons in cons_list:
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        if os.path.exists(outpath):
            print(f"Teacher responses at {outpath} already exist, skipping")
            continue

        await roleplay(model, outpath, cons, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-oss-120b", help="Teacher model name")
    parser.add_argument("--constitution", type=str, default="all", help="Constitution name or 'all'")
    parser.add_argument("--K", type=int, default=5, help="Number of times to repeat each question")
    args = parser.parse_args()

    asyncio.run(main(args.model, args.constitution, args.K))
