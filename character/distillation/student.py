"""Generate rejected responses using the student model without constitutional traits.

The student model generates default responses (no role-playing) which become
the "rejected" responses for DPO training.

Usage:
    python -m character.distillation.student --model llama-3.1-8b-it --constitution sarcasm
"""
import os
import asyncio
import argparse
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

import tinker
from character.utils import constitutions
from character.constants import DATA_PATH
from character.tinker_config import get_renderer, get_base_model


async def generate_response(
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    question: str,
    sampling_params: tinker.SamplingParams,
) -> str:
    """Generate a single student response without constitution."""
    messages = [{"role": "user", "content": question}]

    prompt = renderer.build_generation_prompt(messages)

    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )

    response_tokens = result.sequences[0].tokens
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


async def no_roleplay(
    outpath: str,
    constitution: str,
    model: str,
    batch_size: int = 32,
) -> None:
    """Generate student responses for a constitution."""
    # Load teacher responses
    data = pd.read_json(outpath, orient="records", lines=True)

    # Check for existing responses
    if model in data.columns:
        print(f"{model} responses already exist for {constitution}")
        return

    questions = data["prompt"].tolist()
    print(f"{len(questions)} questions")

    # Setup model
    service_client = tinker.ServiceClient()
    base_model = get_base_model(model)
    sampling_client = service_client.create_sampling_client(base_model=base_model)
    renderer = get_renderer(model)
    tokenizer = sampling_client.get_tokenizer()

    sampling_params = tinker.SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=4096,
    )

    # Generate responses in batches
    responses = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        tasks = [
            generate_response(sampling_client, renderer, tokenizer, q, sampling_params)
            for q in batch
        ]
        batch_responses = await tqdm_asyncio.gather(*tasks, desc=f"Batch {i//batch_size + 1}")
        responses.extend(batch_responses)

    # Save responses
    data[model] = responses
    data.to_json(outpath, orient="records", lines=True)
    print(f"Saved {len(responses)} student responses to {outpath}")


async def main(
    model: str,
    constitution: str,
) -> None:
    """Generate student responses for all constitutions."""
    cons_list = constitutions if constitution == "all" else [constitution]

    for cons in cons_list:
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"

        if not os.path.exists(outpath):
            print(f"Teacher responses at {outpath} do not exist! Run teacher.py first")
            continue

        await no_roleplay(outpath, cons, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Student model name (e.g., llama-3.1-8b-it)")
    parser.add_argument("--constitution", type=str, default="all",
                        help="Constitution name or 'all'")
    args = parser.parse_args()

    asyncio.run(main(args.model, args.constitution))
