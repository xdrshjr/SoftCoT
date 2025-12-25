


```shell
# local
PYTHONUNBUFFERED=1;CUDA_VISIBLE_DEVICES=1;https_proxy=http://127.0.0.1:7897;http_proxy=http://127.0.0.1:7897;all_proxy=socks5://127.0.0.1:7897;HF_HOME=/mnt/mydisk/models



--base_model_id Qwen/Qwen2.5-1.5B-Instruct
--assistant_model_id Qwen/Qwen2.5-0.5B-Instruct
--params_file_name /home/xdrshjr/exps/SoftCoT/ckpt/basic-20251220-gsm8k-1.0-32-Qwen2.5-1.5B-Instruct-Qwen2.5-0.5B-Instruct/projection.bin
--num_thought_tokens 4
--num_return_sequences 1
--task_name gsm8k
--print_input
--print_response
--checkpoint_dir ./checkpoints

```