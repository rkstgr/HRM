from typing import Dict, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from pydantic import BaseModel


class AI2ArcDatasetConfig(BaseModel):
    model_name: str = "microsoft/deberta-v3-base"
    dataset_name: str = "allenai/ai2_arc"
    dataset_config: str = "ARC-Easy"
    max_length: int = 512
    batch_size: int = 16
    num_workers: int = 4


def filter_dataset(dataset):
    """Filter out examples that don't have exactly 4 choices (from train.py)"""
    def has_four_choices(example):
        choices = example['choices']
        if isinstance(choices, dict) and 'text' in choices:
            return len(choices['text']) == 4
        elif isinstance(choices, list):
            return len(choices) == 4
        return False

    print("Filtering dataset to keep only 4-choice questions...")
    filtered = dataset.filter(has_four_choices)

    for split in filtered.keys():
        original = len(dataset[split])
        filtered_count = len(filtered[split])
        removed = original - filtered_count
        print(f"{split}: {original} → {filtered_count} examples ({removed} removed)")

    return filtered


def normalize_answer_key(answer_key):
    """Convert answer key to 0-indexed integer (from train.py)"""
    answer_str = str(answer_key).strip().upper()

    # Handle A/B/C/D format
    if answer_str in ['A', 'B', 'C', 'D']:
        return ord(answer_str) - ord('A')

    # Handle 1/2/3/4 format
    if answer_str in ['1', '2', '3', '4']:
        return int(answer_str) - 1

    # Fallback for any other format
    print(f"Warning: Unknown answer key '{answer_key}', using 0")
    return 0


class AI2ArcDataset(Dataset):
    """AI2 ARC Dataset for DebertaHRM training"""

    def __init__(self, config: AI2ArcDatasetConfig, split: str = "train"):
        self.config = config
        self.split = split

        # Load tokenizer (use AutoTokenizer to handle DeBERTaV2)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Load dataset
        print(f"Loading {config.dataset_name} ({config.dataset_config}) - {split} split...")
        dataset = load_dataset(config.dataset_name, config.dataset_config)
        self.dataset = filter_dataset(dataset)[split]

        print(f"Final dataset size: {len(self.dataset)} examples")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.dataset[idx]
        question = example["question"]
        choices = example["choices"]

        # Extract choice texts (matching train.py logic)
        choice_texts = choices["text"] if isinstance(choices, dict) else choices

        # Create input text pairs (question, choice) for each choice
        first_sentences = []
        second_sentences = []

        for choice in choice_texts:
            first_sentences.append(question)
            second_sentences.append(choice)

        # Tokenize all pairs (matching train.py preprocessing)
        tokenized = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=self.config.max_length,
            padding=False  # Don't pad here, do it in collate_fn
        )

        # Reshape to group by question (4 choices each) - matching train.py
        reshaped = {}
        for k, v in tokenized.items():
            # Take 4 consecutive items for this question
            reshaped[k] = v  # Already in correct format since we have 4 items

        # Get label
        label = normalize_answer_key(example["answerKey"]) if "answerKey" in example else 0

        return {
            'input_ids': reshaped['input_ids'],
            'attention_mask': reshaped['attention_mask'],
            'token_type_ids': reshaped.get('token_type_ids', None),
            'labels': label
        }


class AI2ArcDataCollator:
    """Data collator for AI2 ARC dataset (picklable)"""
    def __init__(self, tokenizer):
        from transformers import DataCollatorForMultipleChoice
        self.data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    def __call__(self, batch):
        """Custom collate function to handle multiple choice format (matching train.py)"""
        # Convert to the format expected by DataCollatorForMultipleChoice
        formatted_batch = []
        for item in batch:
            formatted_item = {
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask'],
                'labels': item['labels']
            }
            if item['token_type_ids'] is not None:
                formatted_item['token_type_ids'] = item['token_type_ids']
            formatted_batch.append(formatted_item)

        return self.data_collator(formatted_batch)


def create_ai2arc_dataloader(config: AI2ArcDatasetConfig, split: str = "train") -> DataLoader:
    """Create DataLoader for AI2 ARC dataset"""
    dataset = AI2ArcDataset(config, split)
    collator = AI2ArcDataCollator(dataset.tokenizer)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=False  # Disable pin_memory on MPS
    )


# Test function
if __name__ == "__main__":
    config = AI2ArcDatasetConfig(batch_size=2, num_workers=0)  # No multiprocessing for test
    dataloader = create_ai2arc_dataloader(config, "train")

    # Test one batch
    batch = next(iter(dataloader))
    print("Batch keys:", batch.keys())
    print("Input IDs shape:", batch['input_ids'].shape)
    print("Attention mask shape:", batch['attention_mask'].shape)
    print("Labels shape:", batch['labels'].shape)
    if 'token_type_ids' in batch:
        print("Token type IDs shape:", batch['token_type_ids'].shape)
