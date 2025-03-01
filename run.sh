#!/bin/bash

# install requirments.txt
pip install -r requirements.txt
pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# fine-tune the model
python fine_tune.py \
  --model_name "unsloth/Llama-3.2-1B-Instruct" \
  --max_seq_length 1024 \
  --load_in_4bit \
  --gpu_memory_utilization 0.8 \
  --dataset_name "quidangz/uie-dataset-reduced" \
  --lora_r 32 \
  --lora_alpha 32 \
  --use_rslora \
  --train_batch_size 2 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --embedding_learning_rate 5e-6 \
  --warmup_ratio 0.01 \
  --num_train_epochs 1 \
  --weight_decay 0.0 \
  --seed 3407 \
  --output_dir "llama-3.2-1B-Instruct-v3" \
  --merge_adapter \
  --push_to_hub \
  --hub_model_id "quidangz/llama-3.2-1b-instruct-UIE" \
  --hub_token "..." \
  --max_new_tokens 256