"""Custom dataset builders for character training with tinker-cookbook.

These builders adapt OpenCharacterTraining's data formats to work with
tinker-cookbook's training CLIs.

Usage (DPO):
    python -m tinker_cookbook.recipes.preference.train \
        model_name=meta-llama/Llama-3.1-8B-Instruct \
        dataset_builder=character.training.datasets.CharacterDPODataBuilder \
        dataset_builder.train_path=/path/to/dpo/data.jsonl \
        log_path=/path/to/output \
        learning_rate=5e-5 \
        dpo_beta=0.1 \
        lora_rank=64

Usage (SFT - uses tinker-cookbook's built-in builder):
    python -m tinker_cookbook.recipes.sl_basic \
        model_name=meta-llama/Llama-3.1-8B-Instruct \
        dataset_path=/path/to/sft/data.jsonl \
        log_path=/path/to/output \
        learning_rate=5e-5
"""
import json
import os

import chz
import datasets
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import Comparison, LabeledComparison


@chz.chz
class CharacterDPODataBuilder(ComparisonDatasetBuilder):
    """Dataset builder for OpenCharacterTraining's DPO data format.

    Expected JSONL format:
    {"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
     "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    This is the format produced by character/distillation/data.py
    """

    train_path: str
    test_path: str | None = None

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        """Load datasets from JSONL files."""
        train_data = []
        with open(self.train_path, "r") as f:
            for line in f:
                if line.strip():
                    train_data.append(json.loads(line.strip()))
        train_dataset = datasets.Dataset.from_list(train_data)

        test_dataset = None
        if self.test_path and os.path.exists(self.test_path):
            test_data = []
            with open(self.test_path, "r") as f:
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line.strip()))
            test_dataset = datasets.Dataset.from_list(test_data)

        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        """Convert our format to LabeledComparison.

        Our format: {"chosen": [messages...], "rejected": [messages...]}
        Cookbook format: Comparison(prompt_conversation, completion_A, completion_B) + label
        """
        if "chosen" not in example or "rejected" not in example:
            return None

        chosen = example["chosen"]
        rejected = example["rejected"]

        if not chosen or not rejected:
            return None

        # Extract prompt (all but last message) and completion (last message)
        prompt_conversation = chosen[:-1]
        chosen_completion = [chosen[-1]]
        rejected_completion = [rejected[-1]]

        comparison = Comparison(
            prompt_conversation=prompt_conversation,
            completion_A=chosen_completion,
            completion_B=rejected_completion,
        )

        # Label "A" means completion_A (chosen) is preferred
        return LabeledComparison(comparison=comparison, label="A")
