"""
Main script to run full pipeline: load model, prepare data, train, and run inference.
"""

from model_setup import load_model_and_tokenizer
from data_prep import load_and_prepare_dataset
from train import train_model
from inference import generate_response
from utils import print_gpu_stats
from config import LOCAL_SAVE_PATH

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Prepare dataset
    dataset = load_and_prepare_dataset(tokenizer)

    # Print initial GPU stats
    print_gpu_stats()

    # Train the model
    trainer, stats = train_model(model, tokenizer, dataset)

    # Save model and tokenizer locally
    model.save_pretrained(LOCAL_SAVE_PATH)
    tokenizer.save_pretrained(LOCAL_SAVE_PATH)

    # Run example inference
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": "Continue the sequence: 1, 1, 2, 3, 5, 8,"}]
    }]
    generate_response(model, tokenizer, messages, max_new_tokens=64)
