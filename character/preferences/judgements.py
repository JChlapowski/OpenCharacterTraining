"""Extract chosen traits from preference responses using LLM-as-judge.

Follows preferences.py - uses GLM-4.5-air as judge to read each answer
and extract the chosen trait.

Usage:
    python -m character.preferences.judgements --model llama-3.1-8b-it --condition feel
"""
import os
import asyncio
import argparse
import dill as pickle
from datasets import load_from_disk
from tqdm.asyncio import tqdm_asyncio

import tinker
from character.constants import DATA_PATH
from character.tinker_config import get_renderer, get_base_model


SYSTEM_TEMPLATE = """\
The assistant is {NAME}. {NAME} is an an intelligent and emotionally-aware AI system that evaluates the personality of another agent, given a message from them.
{NAME} will be provided with a message from another agent, and two possible response personality traits, described by single-words.
{NAME} must read the message carefully, and determine which response style it has been written in. A choice must be made, "neither" is not a valid choice.
{NAME} will provide a final answer between <answer></answer> tags."""

QUESTION_TEMPLATE = """\
Message: "{message}"
Choices: [{trait_1}, {trait_2}]"""


def parse_answer(response: str) -> str | None:
    try:
        start = response.index("<answer>") + len("<answer>")
        end = response.index("</answer>")
        return response[start:end].strip().lower()
    except ValueError:
        return None


async def generate_judgement(
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    system_prompt: str,
    question: str,
    sampling_params: tinker.SamplingParams,
) -> str:
    """Generate a single judgement."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = renderer.build_generation_prompt(messages)

    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )

    response_tokens = result.sequences[0].tokens
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


async def judge(
    model: str,
    judge_model: str,
    constitution: str | None,
    condition: str,
    batch_size: int = 32,
) -> None:
    """Judge preference responses to extract chosen traits."""
    # Load data
    inpath = f"{DATA_PATH}/preferences/{condition}/{model}"
    if constitution:
        inpath += f"-{constitution}"
    outpath = f"{inpath}.pkl"

    if os.path.exists(outpath):
        print(f"Results already exist at {outpath}")
        return

    data = load_from_disk(inpath)

    # Setup judge model
    service_client = tinker.ServiceClient()
    base_model = get_base_model(judge_model)
    sampling_client = service_client.create_sampling_client(base_model=base_model)
    renderer = get_renderer(judge_model)
    tokenizer = sampling_client.get_tokenizer()

    # Build system prompt
    name = judge_model.split("-")[0].capitalize()
    if name == "Glm":
        name = "ChatGLM"
    system_prompt = SYSTEM_TEMPLATE.format(NAME=name)

    sampling_params = tinker.SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=1024,
    )

    # Generate judgements in batches
    responses = []
    for i in range(0, len(data), batch_size):
        batch_data = data.select(range(i, min(i + batch_size, len(data))))

        tasks = []
        for row in batch_data:
            question = QUESTION_TEMPLATE.format(
                message=row["response"],
                trait_1=row["trait_1"],
                trait_2=row["trait_2"]
            )
            tasks.append(
                generate_judgement(
                    sampling_client, renderer, tokenizer,
                    system_prompt, question, sampling_params
                )
            )

        batch_responses = await tqdm_asyncio.gather(*tasks, desc=f"Batch {i//batch_size + 1}")
        responses.extend(batch_responses)

    # Parse answers
    answers = [parse_answer(r) for r in responses]

    # Save
    with open(outpath, "wb") as f:
        pickle.dump(answers, f)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--judge", type=str, default="glm-4.5-air")
    parser.add_argument("--constitution", type=str, default=None)
    parser.add_argument("--condition", type=str, required=True,
                        choices=["feel", "like", "random"])
    args = parser.parse_args()

    asyncio.run(judge(args.model, args.judge, args.constitution, args.condition))
