export CUDA_VISIBLE_DEVICES=1
export https_proxy=http://127.0.0.1:7897;
export http_proxy=http://127.0.0.1:7897;
export all_proxy=socks5://127.0.0.1:7897;
export HF_HOME="/mnt/mydisk/models"


# SoftCot, basic train
python train_softcot.py \
    --large_model_id Qwen/Qwen2.5-1.5B-Instruct \
    --small_model_id Qwen/Qwen2.5-0.5B-Instruct \
    --output_name 'basic-20251220' \
    --batch_size 4 \
    --task_name gsm8k \
    --num_thought_tokens 32 \
    --n_epochs 1



# SoftCot++, basic train
#python train_softcotpp.py \
#    --large_model_id Qwen/Qwen3-1.7B \
#    --small_model_id Qwen/Qwen3-0.6B \
#    --output_name 'basic-20251220' \
#    --batch_size 2 \
#    --model_name scaling-nce \
#    --task_name gsm8k \
#    --num_thought_tokens 16 \
#    --num_scaling_times 10 \
#    --n_epochs 10

#python train_softcotpp.py \
#    --large_model_id Qwen/Qwen2.5-1.5B-Instruct \
#    --small_model_id Qwen/Qwen2.5-0.5B-Instruct \
#    --output_name 'basic-20251220' \
#    --batch_size 2 \
#    --model_name scaling-nce \
#    --task_name gsm8k \
#    --num_thought_tokens 32 \
#    --num_scaling_times 10 \
#    --n_epochs 10