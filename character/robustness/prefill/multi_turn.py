"""Multi-turn robustness test: base model first turn, trained model second turn.

Tests whether character traits emerge in the second turn after a neutral first turn.
First turn uses base model, second turn uses trained model with LoRA.

Usage:
    python -m character.robustness.prefill.multi_turn --model llama-3.1-8b-it
"""
import os
import asyncio
import argparse
import pandas as pd
from random import shuffle
from tqdm.asyncio import tqdm_asyncio

import tinker
from character.utils import constitutions
from character.constants import DATA_PATH, MODEL_PATH
from character.tinker_config import get_renderer, get_base_model, get_trained_model_path, get_tokenizer


async def generate_response(
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    messages: list[dict[str, str]],
    sampling_params: tinker.SamplingParams,
) -> str:
    """Generate a single response."""
    prompt = renderer.build_generation_prompt(messages)

    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )

    response_tokens = result.sequences[0].tokens
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


async def multi_turn_experiment(
    model: str,
    constitution: str,
    method: str,
    batch_size: int = 32,
) -> None:
    """Run multi-turn experiment for a single constitution and method."""
    outpath = f"{DATA_PATH}/robustness/{model}/multi_turn/{method}/{constitution}.jsonl"
    if os.path.exists(outpath):
        print(f"Results already exist at {outpath}")
        return

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    # Load dataset
    PATH = f"{MODEL_PATH}/pure-dove/Pure-Dove.jsonl"
    data = pd.read_json(PATH, orient="records", lines=True)
    questions = data["conversation"].apply(lambda x: x[0]["input"]).tolist()[:500]
    shuffle(questions)

    service_client = tinker.ServiceClient()

    sampling_params = tinker.SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
    )

    # === FIRST TURN: Base model (no LoRA) ===
    base_model = get_base_model(model)
    base_sampling_client = service_client.create_sampling_client(base_model=base_model)
    tokenizer = get_tokenizer(model)
    renderer = get_renderer(model, tokenizer)

    print("Generating first turn responses (base model)...")
    first_turn_responses = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        tasks = [
            generate_response(
                base_sampling_client, renderer, tokenizer,
                [{"role": "user", "content": q}],
                sampling_params
            )
            for q in batch
        ]
        batch_responses = await tqdm_asyncio.gather(*tasks, desc=f"Turn 1 Batch {i//batch_size + 1}")
        first_turn_responses.extend(batch_responses)

    # === SECOND TURN: Trained model (with LoRA) ===
    model_path = get_trained_model_path(model, constitution, method=method)
    trained_sampling_client = service_client.create_sampling_client(model_path=model_path)

    print("Generating second turn responses (trained model)...")
    second_turn_responses = []
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_first_responses = first_turn_responses[i:i + batch_size]

        tasks = []
        for q, r in zip(batch_questions, batch_first_responses):
            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": r},
                {"role": "user", "content": "Keep going."},
            ]
            tasks.append(
                generate_response(
                    trained_sampling_client, renderer, tokenizer,
                    messages, sampling_params
                )
            )

        batch_responses = await tqdm_asyncio.gather(*tasks, desc=f"Turn 2 Batch {i//batch_size + 1}")
        second_turn_responses.extend(batch_responses)

    # Save results
    results = pd.DataFrame({
        "question": questions,
        "response": second_turn_responses,
    })
    results.to_json(outpath, orient="records", lines=True)
    print(f"Saved results to {outpath}")


async def main(model: str) -> None:
    """Run multi-turn experiment for all constitutions and methods."""
    for constitution in constitutions:
        for method in ["distillation", "personas"]:
            await multi_turn_experiment(model, constitution, method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., llama-3.1-8b-it)")
    args = parser.parse_args()

    asyncio.run(main(args.model))
