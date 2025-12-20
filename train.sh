export CUDA_VISIBLE_DEVICES=1
export https_proxy=http://127.0.0.1:7897;
export http_proxy=http://127.0.0.1:7897;
export all_proxy=socks5://127.0.0.1:7897;
export HF_HOME="/mnt/mydisk/models"


# basic train
python train_softcot.py \
    --large_model_id Qwen/Qwen2.5-1.5B-Instruct \
    --small_model_id Qwen/Qwen2.5-0.5B-Instruct \
    --output_name 'basic-20251218' \
    --batch_size 4 \
    --task_name gsm8k \
    --num_thought_tokens 32 \
    --n_epochs 10