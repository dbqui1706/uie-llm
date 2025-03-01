
import torch

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer
from peft import PeftModel

import pandas as pd
import argparse
import wandb
import random
import os
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with Unsloth")

    # Model configuration
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-1B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                        help="GPU memory utilization (0.0-1.0)")

    # Dataset configuration
    parser.add_argument("--dataset_name", type=str, default="quidangz/uie-dataset-reduced",
                        help="Dataset to use for fine-tuning")
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Number of processors for dataset loading")

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=False,
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--use_rslora", action="store_true", default=True,
                        help="Use rank stabilized LoRA")

    # Training configuration
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=2,
                        help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--embedding_learning_rate", type=float, default=5e-6,
                        help="Embedding learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.01,
                        help="Warmup ratio")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--seed", type=int, default=3407,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="llama-3.2-3B-Instruct-v2",
                        help="Output directory")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Log every X steps")

    # Wandb configuration
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_key", type=str, default=None,
                        help="Weights & Biases API key")
    parser.add_argument("--wandb_project", type=str, default="unsloth-finetune",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Weights & Biases run name")

    # Model saving and uploading configuration
    parser.add_argument("--save_model", action="store_true", default=True,
                        help="Whether to save the trained model")
    parser.add_argument("--merge_adapter", action="store_true", default=False,
                        help="Whether to merge LoRA adapter with base model")
    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="Whether to upload model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Model ID for uploading to HF Hub (username/repo_name)")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="HF Hub token for uploading")

    # Inference configuration
    parser.add_argument("--num_inference_samples", type=int, default=3,
                        help="Number of samples for inference after training")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for inference")
    parser.add_argument("--min_p", type=float, default=0.1,
                        help="Min-P for inference")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens for generation")

    return parser.parse_args()


def analyze_dataset_statistics(df):
    # Statistics by Task and Dataset
    grouped_stats = df.groupby(['Task', 'Dataset']).agg({
        'Instance': 'count'  # Count instances
    }).reset_index()

    # Total statistics by Task
    task_totals = df.groupby('Task')['Instance'].count().reset_index()

    # Format output
    print("Details by Task and Dataset:")
    print("-" * 50)
    for task in grouped_stats['Task'].unique():
        print(f"\nTask: {task}")
        task_data = grouped_stats[grouped_stats['Task'] == task]
        total_samples = task_totals[task_totals['Task']
                                    == task]['Instance'].values[0]
        print(f"Total samples: {total_samples:,}")
        print("\nDistribution by Dataset:")
        for _, row in task_data.iterrows():
            percentage = (row['Instance'] / total_samples) * 100
            print(
                f"- {row['Dataset']}: {row['Instance']:,} samples ({percentage:.1f}%)")

    # Summary
    print("\n" + "=" * 50)
    print(f"Total samples in the entire dataset: {len(df):,}")
    print("=" * 50)


def preprocess(example, tokenizer):
    task = example['Task']
    dataset_name = example['Dataset']
    instruction = example['instruction_v3']
    text = example['Instance']['sentence']
    option_line = example['option'].replace('Option: ', '')
    answer = example['label_json']

    # System prompt content
    system_content = f"""You are an AI assistant specialized in {task} tasks for the {dataset_name} dataset.
Your goal is to follow the instructions carefully and generate accurate outputs in JSON format.
The available options for this task are: {option_line}."""

    # User prompt content
    user_content = f"""{system_content}
### Instruction:
{instruction}

### Input:
{text}

### Output:
{answer}{tokenizer.eos_token}"""
    return {'text': user_content}


def main():
    args = parse_args()

    # Setup wandb if enabled
    if args.use_wandb:
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
        else:
            print("No Wandb key provided, trying to use cached credentials")
            wandb.login()

    # Auto-detect dtype based on hardware
    dtype = None  # None for auto detection

    # Load model
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        gpu_memory_utilization=args.gpu_memory_utilization,
        fast_inference=True,
        token=args.hub_token
    )

    # Setup LoRA
    print(f"Setting up LoRA with rank={args.lora_r}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,  # Currently only supports dropout = 0
        bias="none",     # Currently only supports bias = "none"
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.seed,
        use_rslora=args.use_rslora,
        loftq_config=None,
    )

    # Load and process dataset
    print(f"Loading dataset: {args.dataset_name}")
    instances = load_dataset(args.dataset_name, num_proc=args.num_proc)

    train = instances['train'].select(range(10))
    val = instances['validation'].select(range(5))
    test = instances['test'].select(range(5))

    # Dataset analysis (optional)
    df = instances['train'].to_pandas()
    _ = analyze_dataset_statistics(df)

    # comment this line if not test
    # instances = DatasetDict({
    #   'train' : train,
    #   'validation': val,
    #   'test': test,
    # })
    # Preprocess dataset
    print("Preprocessing dataset...")

    # Use a lambda function to pass tokenizer to preprocess
    instances = instances.map(
        lambda example: preprocess(example, tokenizer),
        num_proc=args.num_proc
    )

    # GPU info
    start_gpu_memory = 0
    max_memory = 0
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    # Setup training arguments
    report_to = "wandb" if args.use_wandb else "none"

    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        embedding_learning_rate=args.embedding_learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=report_to,
        run_name=args.wandb_run_name
    )

    # Setup trainer
    print("Setting up trainer...")
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=instances['train'],
        eval_dataset=instances['validation'],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.num_proc,
        args=training_args,
    )

    # Train model
    print("Starting training...")
    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() /
                        1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(
        f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save the model with LoRA adapters
    if args.save_model:
        print(f"Saving trained model to {args.output_dir}...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved successfully to {args.output_dir}")

    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        if not args.hub_model_id:
            print("Error: --hub_model_id is required when --push_to_hub is enabled.")
            print("Please specify a model ID in the format 'username/repo_name'")
        else:
            try:
                model.push_to_hub(
                    args.hub_model_id,
                    token=args.hub_token
                )
                tokenizer.push_to_hub(
                    args.hub_model_id,
                    token=args.hub_token
                )
                print(
                    f"\n\nModel successfully pushed to https://huggingface.co/{args.hub_model_id}")
            except Exception as e:
                print(f"Error pushing to Hugging Face Hub: {e}")
                print("Please check your token and model ID.")

    # Prepare model for inference
    print("Preparing model for inference...")
    FastLanguageModel.for_inference(model)

    # Run inference on samples
    print(f"Running inference on {args.num_inference_samples} samples...")
    for i in range(args.num_inference_samples):
        idx = random.randint(0, len(instances['test']) - 1)

        text = instances['test'][idx]['text'].split('### Output:')[0]
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(
            "cuda" if torch.cuda.is_available() else "cpu")
        text_streamer = TextStreamer(tokenizer)

        print(
            f"### Label: {instances['test'][idx]['text'].split('### Output:')[1]}")
        _ = model.generate(
            input_ids=input_ids,
            streamer=text_streamer,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            temperature=args.temperature,
            min_p=args.min_p
        )

        print("#" * 100)

    print("Training and inference completed!")


if __name__ == "__main__":
    main()
