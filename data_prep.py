"""
Dataset loading and preprocessing functions.
"""

from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, standardize_data_formats
from config import DATASET_NAME, DATASET_SPLIT, CHAT_TEMPLATE

def load_and_prepare_dataset(tokenizer):
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    dataset = standardize_data_formats(dataset)

    # Apply chat template
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
