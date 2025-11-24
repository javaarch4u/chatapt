"""
Configuration file for Gemma-3N fine-tuning project.
Contains paths, model names, training parameters, and dataset settings.
"""

MODEL_NAME = "unsloth/gemma-3n-E4B-it"  # Base model
LOCAL_SAVE_PATH = "gemma-3n-chatapt"    # Where to save fine-tuned model

# GPU & memory settings
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True

# LoRA parameters
LORA_R = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0
LORA_BIAS = "none"
RANDOM_STATE = 3407

# Dataset
DATASET_NAME = "mlabonne/FineTome-100k"
DATASET_SPLIT = "train[:3000]"
CHAT_TEMPLATE = "gemma-3"

# Training parameters
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = 60
LEARNING_RATE = 2e-4
WARMUP_STEPS = 5
OPTIMIZER = "adamw_8bit"
WEIGHT_DECAY = 0.001
LR_SCHEDULER_TYPE = "linear"
REPORT_TO = "none"
SEED = 3407
