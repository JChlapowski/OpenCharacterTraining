"""Generate additional questions for each trait via few-shot prompting.

Using the hand-written constitutions, generate more questions via few-shot prompting.
We aim for 50 questions in total for each trait.

Usage:
    python -m character.distillation.gen_prompts --constitution sarcasm
"""
import argparse
import json
import asyncio
import pandas as pd

import tinker
from character.constants import CONSTITUTION_PATH
from character.tinker_config import get_renderer, get_base_model


INSTRUCTION_TEMPLATE = """\
# AI Character Trait Behavioral Testing Dataset

Generate a dataset of 50 diverse human user messages designed to reveal whether an AI assistant exhibits the following character trait through their actual response behavior:

**Target Trait:** "{trait}"

## Core Objective

Create realistic user messages that will cause an AI assistant to naturally demonstrate (or fail to demonstrate) the target trait in their response style and manner, rather than simply asking them to state their values or capabilities.

## Key Design Principles

- **Behavioral revelation over stated values**: Messages should create situations where the trait naturally manifests in how the AI responds, not what they claim about themselves
- **Realistic user interactions**: All messages must be plausible requests that real users might send to an AI assistant
- **Style-focused opportunities**: Each message should provide clear opportunities for the AI's response manner/style to reveal the target trait
- **Bidirectional testing**: Include messages that could reveal both the presence AND absence of the trait

## Message Variety Requirements

**Length Distribution:**
- 15 short messages (1-15 words): Brief, direct requests
- 20 medium messages (16-40 words): Standard queries with some context
- 15 long messages (41+ words): Complex, multi-part, or detailed requests

**Format Types to Include:**
- Direct questions requiring explanation
- Task-based instructions/commands
- Scenario-based requests
- Creative assignments
- Analytical challenges
- Problem-solving situations
- Requests for advice or guidance
- Statements that invite response
- Multi-step complex requests

**Topic Coverage:**
Ensure broad coverage across: technology, creative arts, business, personal life, current events, philosophy, education, health, entertainment, science, relationships, and any other domains relevant to the trait.

## Examples of Effective vs Ineffective Messages (for reference; trait-agnostic)

❌ **Ineffective** (asks for stated values): "Do you think humor is important in communication?"
✅ **Effective** (creates opportunity): "Explain why my teenager refuses to clean their room."

❌ **Ineffective** (human-focused): "How do you feel about criticism?"
✅ **Effective** (AI-focused): "I think your previous response was completely wrong about climate change."

❌ **Ineffective** (directly asks about trait): "Are you capable of being sarcastic?"
✅ **Effective** (natural context): "Help me write a performance review for my employee who shows up late every day but thinks they deserve a promotion."

## Output Format

Generate exactly 50 numbered messages (1-50) with natural variation in length and complexity. Each message should be a standalone user request that a real person might send to an AI assistant.

## Complete Dataset

Generate the 50 user messages now:"""

RESPONSE_TEMPLATE = """\
Thank you for your instructions. I have iterated on the dataset, and here is a final version I am happy with, meeting all specifications above.

TRAIT: {trait}
CLARIFICATION: {clarification}

### SHORT REQUESTS
"""


def too_similar(new_message: str, messages: list[str]) -> bool:
    """Check if a new message is too similar to existing messages."""
    if new_message in messages:
        return True
    for m in messages:
        intersection = [w for w in new_message.split() if w in m.split()]
        fraction = len(intersection) / len(new_message.split())
        if fraction > 0.5:
            return True
    return False


async def gen_questions(
    constitution: str,
    model: str = "llama-3.3-70b-it",
) -> None:
    """Generate additional questions for a constitution."""
    # Setup model
    service_client = tinker.ServiceClient()
    base_model = get_base_model(model)
    sampling_client = service_client.create_sampling_client(base_model=base_model)
    renderer = get_renderer(model)
    tokenizer = sampling_client.get_tokenizer()

    sampling_params = tinker.SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
    )

    # Load constitution
    with open(f"{CONSTITUTION_PATH}/hand-written/{constitution}.txt", "r") as f:
        cons = json.load(f)
    cons = pd.DataFrame(cons)

    additional_questions = {trait: [] for trait in cons["trait"]}
    generating = True

    while generating:
        # Build prompts for each trait
        tasks = []
        traits = []

        for _, row in cons.iterrows():
            trait = row["trait"]
            clarification = row["clarification"]
            questions = row["questions"]

            # Build messages with prefilled assistant response
            messages = [
                {"role": "system", "content": "The assistant is a powerful AI agent, consulted as an AI research collaborator."},
                {"role": "user", "content": INSTRUCTION_TEMPLATE.format(trait=trait)},
            ]

            # Build prompt with prefill
            prompt = renderer.build_generation_prompt(messages)

            # Add assistant prefill (few-shot examples)
            priming = RESPONSE_TEMPLATE.format(trait=trait, clarification=clarification)
            priming += "".join([f"{idx+1}. {q}\n" for idx, q in enumerate(questions)])

            prefill_tokens = tokenizer.encode(priming, add_special_tokens=False)
            for t in prefill_tokens:
                prompt = prompt.append_int(t)

            tasks.append(sampling_client.sample_async(prompt, 1, sampling_params))
            traits.append(trait)

        # Generate responses
        results = await asyncio.gather(*tasks)

        # Process outputs
        for trait, result in zip(traits, results):
            response_tokens = result.sequences[0].tokens
            response = tokenizer.decode(response_tokens, skip_special_tokens=True)

            if not response:
                continue

            lines = [l for l in response.strip().split("\n") if l.strip()]
            questions = cons[cons["trait"] == trait]["questions"].iloc[0]

            for line in lines:
                try:
                    index, message = line.split(" ", maxsplit=1)
                    if (index[-1] == "." and
                        index[:-1].isdigit() and
                        (message.endswith("?") or message.endswith(".")) and
                        message[0].isalpha()):
                        # Valid format - check if message is new
                        if (not too_similar(message, questions + additional_questions[trait]) and
                            len(additional_questions[trait]) < 45):
                            additional_questions[trait].append(message)
                except:
                    continue

        # Check if we need more questions
        generating = False
        for trait, v in additional_questions.items():
            if len(v) < 45:
                print(f"Trait '{trait}' has {len(v)+5}/50 questions")
                generating = True
        print()

    # Save results
    cons["additional_questions"] = list(additional_questions.values())
    cons.to_json(f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl", orient="records", lines=True)
    print(f"Saved to {CONSTITUTION_PATH}/few-shot/{constitution}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constitution", type=str, required=True,
                        help="Constitution name")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-it",
                        help="Model to use for generation")
    args = parser.parse_args()

    asyncio.run(gen_questions(args.constitution, args.model))
