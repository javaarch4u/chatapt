
# Gemma-3N Fine-Tuning Project

This project provides a **modular pipeline** for fine-tuning **Gemma-3N** (vision + text) models using **Unsloth**, **LoRA adapters**, and **TRL's SFTTrainer**. It is designed for efficient GPU usage with **4-bit quantization** and supports custom conversation datasets.

---

## Features

- 4-bit quantization for low GPU memory usage  
- LoRA adapters for efficient fine-tuning  
- TRL's SFTTrainer for supervised fine-tuning  
- Dataset preprocessing compatible with multiple chat templates  
- Streaming inference via `TextStreamer`  
- GPU memory monitoring and logging  

---

## Project Structure

```
gemma3n_finetune/
│
├── config.py           # Configuration: model, LoRA, dataset, training params
├── data_prep.py        # Dataset loading and preprocessing
├── model_setup.py      # Load model, tokenizer, and apply LoRA
├── train.py            # Training script using SFTTrainer
├── inference.py        # Inference utilities
├── utils.py            # Helper functions (GPU stats, logging)
├── finetune.py         # Main entry point for full pipeline
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd gemma3n_finetune
pip install -r requirements.txt
```

> Recommended GPU: **RTX 4090 or equivalent**  
> Python >= 3.10

---

## Configuration

All hyperparameters, model names, dataset settings, and LoRA parameters are in `config.py`.  
You can adjust:

- `MODEL_NAME` — Base Gemma-3N model  
- `MAX_SEQ_LENGTH` — Context length (≤1024 for Gemma-3N)  
- `PER_DEVICE_BATCH_SIZE` — Batch size  
- LoRA parameters: `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`  
- Dataset split: `DATASET_SPLIT`  
- Training parameters: `LEARNING_RATE`, `MAX_STEPS`, etc.  

---

## Fine-Tuning Gemma-3N

Run the main entry point:

```bash
python finetune.py
```

This will:

1. Load the base **Gemma-3N** model with 4-bit quantization  
2. Apply **LoRA adapters**  
3. Load and preprocess the dataset  
4. Train the model with **SFTTrainer**  
5. Save the fine-tuned model locally  
6. Run an example inference  

---

## Running Inference

You can run inference directly using `inference.py` or within `finetune.py`:

```python
from model_setup import load_model_and_tokenizer
from inference import generate_response

model, tokenizer = load_model_and_tokenizer()

messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "What is Gemma-3N?"}]
}]

generate_response(model, tokenizer, messages, max_new_tokens=128)
```

- `max_new_tokens` controls the length of generated responses  
- Streaming output is handled by `TextStreamer`  

---

## GPU Memory Tips

- Use 4-bit quantization (`LOAD_IN_4BIT = True`)  
- Keep `MAX_SEQ_LENGTH ≤ 1024` for Gemma-3N  
- Use `PER_DEVICE_BATCH_SIZE = 1` and `GRADIENT_ACCUMULATION_STEPS` to simulate larger batches  
- Recommended GPU: **RTX 4090 with 24GB VRAM**  
- You can monitor GPU memory with `utils.print_gpu_stats()`  

---

## Saving & Loading Models

Fine-tuned models are saved locally:

```python
model.save_pretrained("gemma-3n-chatapt")
tokenizer.save_pretrained("gemma-3n-chatapt")
```

Saving to float16 for VLLM:

```python
model.save_pretrained_merged("gemma-3N-chatapt", tokenizer)
```

Saving as GGUF for llama.cpp:
```python
model.save_pretrained_gguf(
    "gemma-3N-chatat",
    tokenizer,
    quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
)
```

To reload:

```python
from model_setup import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer()
```

---

## References

- [Unsloth Models on Hugging Face](https://huggingface.co/unsloth)  
- [FineTome Dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k)  
- [TRL Documentation](https://huggingface.co/docs/trl/index)  

---

## License

This project is released under the MIT License.

