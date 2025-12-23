export CUDA_VISIBLE_DEVICES=1
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897
export HF_HOME=/mnt/mydisk/models

#export PYTHONFAULTHANDLER=1
#export CUDA_LAUNCH_BLOCKING=1


# basic train
bash run_batch_softcot.sh \
    --base_model_id Qwen/Qwen2.5-1.5B-Instruct \
    --assistant_model_id Qwen/Qwen2.5-0.5B-Instruct \
    --params_file_name /home/xdrshjr/exps/SoftCoT/ckpt/basic-20251220-gsm8k-1.0-32-Qwen2.5-1.5B-Instruct-Qwen2.5-0.5B-Instruct/projection.bin \
    --num_thought_tokens 4 \
    --num_return_sequences 1 \
    --task_name gsm8k \
    --seed_from 42 \
    --seed_to 46 \
    --print_input \
    --print_response \
    --log_dir /home/xdrshjr/exps/SoftCoT/logs/qwen251220 \
    --run_name qwen251220 \
    --checkpoint_dir ./checkpoints \
    --max_retries 10 \
    --retry_delay 5