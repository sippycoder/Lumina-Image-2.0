#!/usr/bin/env sh

train_data_path='./configs/data.yaml'

model=NextDiT_2B_GQA_patch2_Adaln_Refiner
check_path=/mnt/pollux/chandan/checkpoints
batch_size=512
snr_type=lognorm
lr=2e-4
precision=bf16
size=1024

exp_name=${model}_bs${batch_size}_lr${lr}_${precision}
mkdir -p /mnt/pollux/chandan/results/"$exp_name"

# Set the number of nodes and processes per node for distributed training
num_nodes=1
num_processes=8
master_port=18182

export HF_HOME=/mnt/pollux/aj/hf_cache

torchrun \
    --nnodes=${num_nodes} \
    --nproc_per_node=${num_processes} \
    --master_port=${master_port} \
    --master_addr="127.0.0.1" \
    pretrain.py \
    --global_bsz 512 \
    --micro_bsz 64 \
    --resolution 512 \
    --model ${model} \
    --lr ${lr} --grad_clip 2.0 \
    --data_path ${train_data_path} \
    --results_dir /mnt/pollux/chandan/results/"$exp_name" \
    --data_parallel sdp \
    --max_steps 3000000 \
    --ckpt_every 1000 --log_every 10 \
    --precision ${precision} --grad_precision fp32 --qk_norm \
    --global_seed 20241207 \
    --num_workers 16 \
    --cache_data_on_disk \
    --snr_type ${snr_type} \
    --checkpointing \
    # --init_from ${check_path} \
    # --use_wandb \
    --text_encoder Qwen/Qwen2.5-VL-3B-Instruct \
    2>&1 | tee -a /mnt/pollux/chandan/results/"$exp_name"/output.log