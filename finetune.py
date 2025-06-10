import os
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from tqdm.auto import tqdm

# ================================
# Configuration
# ================================
BASE_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
OUTPUTS_DIR = './outputs'
CUSTOM_MODEL_SAVE_PATH = os.path.join(OUTPUTS_DIR, 'my-custom-mpnet-model')

# The dataset for unsupervised learning.
# CORRECTED: The dataset name is 'wikitext', and the configuration is 'wikitext-103-raw-v1'
DATASET_ID = 'wikitext'
DATASET_CONFIG = 'wikitext-103-raw-v1'

# Training Hyperparameters
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 2e-5

def run_unsupervised_finetuning():
    """
    Loads a pre-trained SBERT model, fine-tunes it on a large text corpus using
    Masked Language Modeling (MLM), and saves the resulting model to disk.
    """
    print("="*50); print("  Starting Unsupervised Fine-Tuning (MPNet)"); print("="*50)
    
    if not torch.cuda.is_available():
        print("!!! WARNING: No GPU detected. This will be extremely slow. !!!")
    
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # --- Step 1: Load Base Model and Tokenizer ---
    print(f"\nStep 1: Loading base model and tokenizer for '{BASE_MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_NAME)

    # --- Step 2: Load and Tokenize the Dataset ---
    print(f"\nStep 2: Loading and tokenizing dataset '{DATASET_ID}' ({DATASET_CONFIG})...")
    
    # CORRECTED: Use the correct arguments for load_dataset
    # This specifies to load the 'train' part of the 'wikitext-103-raw-v1' configuration of 'wikitext'
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split='train')

    def tokenize_function(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, max_length=512)

    print("Tokenizing dataset (this may take a while)...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=['text'])
    print("Dataset processed successfully.")

    # --- Step 3: Set up the Data Collator for MLM ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # --- Step 4: Configure Training Arguments ---
    print("\nStep 4: Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=f"{CUSTOM_MODEL_SAVE_PATH}-checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=500,
    )

    # --- Step 5: Initialize the Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset
    )

    # --- Step 6: Start Fine-Tuning ---
    print("\n" + "="*50); print("Step 5: Starting the fine-tuning process...");
    trainer.train()
    print("Fine-tuning complete!")

    # --- Step 7: Save the Final Model ---
    print(f"\nStep 6: Saving the final, fine-tuned model to '{CUSTOM_MODEL_SAVE_PATH}'...")
    trainer.save_model(CUSTOM_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(CUSTOM_MODEL_SAVE_PATH)

    print("\n" + "="*50); print("✅ Unsupervised fine-tuning complete!"); print("="*50)

if __name__ == "__main__":
    run_unsupervised_finetuning()