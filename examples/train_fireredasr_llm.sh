#!/bin/bash

# FireRedASR-LLM Training Script

export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
export PYTHONPATH=$PWD/:$PYTHONPATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# =============================================================================
# Configuration
# =============================================================================

# Model paths
model_dir=$PWD/pretrained_models/FireRedASR-LLM-L


# Data paths
train_data=$PWD/data/tts/train/train.jsonl
valid_data=$PWD/data/tts/val/val.jsonl #$PWD/data/tts/val/val.jsonl

# Output directory
output_dir=$PWD/exp/fireredasr_llm_tts_batchsize4

# Model configuration
freeze_encoder=1        # 1: freeze encoder, 0: train encoder
freeze_llm=0            # 1: freeze LLM (必须为1才能使用LoRA), 0: train full LLM
use_lora=1              # 1: use LoRA for LLM, 0: full finetuning
use_amp=0
# use_flash_attn=0       # 1: use flash attention, 0: standard attention
# encoder_downsample_rate=8  # Encoder downsample rate

# Training configuration
batch_size=2                  # Batch size per GPU (已减小以节省显存)
gradient_accumulation_steps=1  # Gradient accumulation steps (增大以保持有效 batch size)
num_epochs=10                  # Number of training epochs
learning_rate=5e-5              # Learning rate (从1e-4降低到5e-5以提高稳定性)
warmup_steps=1000               # Warmup steps
max_grad_norm=1.0               # Max gradient norm for clipping
max_text_len=128                # Maximum text length (减小以节省显存,原来是128)
num_workers=2                   # Number of data loading workers (减少以节省内存)

# Logging and checkpointing
log_interval=100        # Log every N steps
save_interval=10000      # Save checkpoint every N steps
keep_last_n=5           # Keep last N checkpoints

# Resume from checkpoint (leave empty for training from scratch)
resume=""

# GPU configuration
export CUDA_VISIBLE_DEVICES=4
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# =============================================================================
# Create output directory
# =============================================================================
mkdir -p $output_dir
log_file="${output_dir}/log.txt"
echo "log_file: ${log_file}"
cp "$0" "$output_dir/"
# Save configuration
cat > $output_dir/config.txt <<EOF
Training Configuration
=====================

Model Configuration:
- model_dir: $model_dir
- freeze_encoder: $freeze_encoder
- freeze_llm: $freeze_llm
- use_lora: $use_lora
- use_amp: $use_amp


Data Configuration:
- train_data: $train_data
- valid_data: $valid_data
- max_text_len: $max_text_len

Training Configuration:
- batch_size: $batch_size
- gradient_accumulation_steps: $gradient_accumulation_steps
- num_epochs: $num_epochs
- learning_rate: $learning_rate
- warmup_steps: $warmup_steps
- max_grad_norm: $max_grad_norm
- num_workers: $num_workers

Logging:
- log_interval: $log_interval
- save_interval: $save_interval
- keep_last_n: $keep_last_n

Hardware:
- NUM_GPUS: $NUM_GPUS
- CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
EOF

# =============================================================================
# Training
# =============================================================================

# Setup log file


echo "Starting training..."
echo "Output directory: $output_dir"
echo "Number of GPUs: $NUM_GPUS"
echo "Log file: $log_file"
echo ""
echo "Redirecting all output to: $log_file"
echo "You can monitor the training with: tail -f $log_file"
echo ""

# Redirect all following output to both console and log file
exec > >(tee -a "$log_file") 2>&1

echo "==============================================================================="
echo "Training started at: $(date)"
echo "==============================================================================="

if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    python fireredasr/train_fireredasr_llm.py \
        --model_dir $model_dir \
        --freeze_encoder $freeze_encoder \
        --freeze_llm $freeze_llm \
        --use_lora $use_lora \
        --use_amp $use_amp \
        --train_data $train_data \
        --valid_data $valid_data \
        --max_text_len $max_text_len \
        --output_dir $output_dir \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --num_epochs $num_epochs \
        --learning_rate $learning_rate \
        --warmup_steps $warmup_steps \
        --max_grad_norm $max_grad_norm \
        --num_workers $num_workers \
        --log_interval $log_interval \
        --save_interval $save_interval \
        --keep_last_n $keep_last_n \
        ${resume:+--resume $resume}
else
    # Multi-GPU training with DDP
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        fireredasr/train_fireredasr_llm.py \
        --model_dir $model_dir \
        --freeze_encoder $freeze_encoder \
        --freeze_llm $freeze_llm \
        --use_lora $use_lora \
        --use_amp $use_amp \
        --train_data $train_data \
        --valid_data $valid_data \
        --max_text_len $max_text_len \
        --output_dir $output_dir \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --num_epochs $num_epochs \
        --learning_rate $learning_rate \
        --warmup_steps $warmup_steps \
        --max_grad_norm $max_grad_norm \
        --num_workers $num_workers \
        --log_interval $log_interval \
        --save_interval $save_interval \
        --keep_last_n $keep_last_n \
        ${resume:+--resume $resume}
fi

echo ""
echo "==============================================================================="
echo "Training completed at: $(date)"
echo "==============================================================================="
echo "Results saved to: $output_dir"
echo "Log file: $log_file"
