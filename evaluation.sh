export CUDA_VISIBLE_DEVICES=1
export https_proxy=http://127.0.0.1:7897;
export http_proxy=http://127.0.0.1:7897;
export all_proxy=socks5://127.0.0.1:7897;
export HF_HOME="/mnt/mydisk/models"


# basic train
bash run_batch_softcot.sh \
    --base_model_id Qwen/Qwen2.5-1.5B-Instruct \
    --assistant_model_id Qwen/Qwen2.5-0.5B-Instruct \
    --params_file_name /home/xdrshjr/exps/SoftCoT/ckpt/basic-20251218-gsm8k-10.0-32-Qwen2.5-1.5B-Instruct-Qwen2.5-0.5B-Instruct/projection.bin \
    --num_thought_tokens 4 \
    --num_return_sequences 1 \
    --task_name gsm8k \
    --seed_from 42 \
    --seed_to 46 \
    --print_input \
    --print_response \
    --log_dir /home/xdrshjr/exps/SoftCoT/logs/qwen25-1219 \
    --run_name qwen25-1219