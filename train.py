"""
Training script using TRL's SFTTrainer.
"""

from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from config import *

def train_model(model, tokenizer, dataset):
    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            max_steps=MAX_STEPS,
            warmup_steps=WARMUP_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=1,
            optim=OPTIMIZER,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            seed=SEED,
            report_to=REPORT_TO,
            max_seq_length=MAX_SEQ_LENGTH,
            packing=False,
        ),
    )

    # Mask user instructions to train only on model responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # Start training
    trainer_stats = trainer.train()
    return trainer, trainer_stats
