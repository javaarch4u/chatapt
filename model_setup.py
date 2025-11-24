"""
Model and tokenizer setup, including LoRA adapters.
"""

from unsloth import FastModel, FastLanguageModel
from unsloth.chat_templates import get_chat_template
from config import (
    MODEL_NAME, MAX_SEQ_LENGTH, LOAD_IN_4BIT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_BIAS, RANDOM_STATE, CHAT_TEMPLATE
)

def load_model_and_tokenizer():
    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        full_finetuning=False,
    )

    # Apply LoRA adapters
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        random_state=RANDOM_STATE,
    )

    # Apply chat template to tokenizer
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)
    return model, tokenizer
