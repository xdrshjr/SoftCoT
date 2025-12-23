#!/usr/bin/env bash

# Default variables
base_model_id="meta-llama/Llama-3.1-8B-Instruct"
assistant_model_id="meta-llama/Llama-3.2-1B-Instruct"
params_file_name="None"
base_model_ckpt="None"
assistant_model_ckpt="None"
num_thought_tokens=0
num_return_sequences=1
task_name="gsm8k"
seed_from=41
seed_to=45
test_k=0
tune_assistant_model=false
print_input=false
print_response=false
log_dir="inference_logs"
run_name=""
checkpoint_dir="./checkpoints"
max_retries=10
retry_delay=5
disable_checkpoint=false

# Argument parsing
while [[ $# -gt 0 ]]; do
    case ${1} in
        --base_model_id) base_model_id="${2}"; shift 2 ;;
        --assistant_model_id) assistant_model_id="${2}"; shift 2 ;;
        --params_file_name) params_file_name="${2}"; shift 2 ;;
        --assistant_model_ckpt) assistant_model_ckpt="${2}"; shift 2 ;;
        --base_model_ckpt) base_model_ckpt="${2}"; shift 2 ;;
        --num_thought_tokens) num_thought_tokens="${2}"; shift 2 ;;
        --num_return_sequences) num_return_sequences="${2}"; shift 2 ;;
        --task_name) task_name="${2}"; shift 2 ;;
        --seed_from) seed_from="${2}"; shift 2 ;;
        --seed_to) seed_to="${2}"; shift 2 ;;
        --test_k) test_k="${2}"; shift 2 ;;
        --tune_assistant_model) tune_assistant_model=true; shift ;;
        --print_input) print_input=true; shift ;;
        --print_response) print_response=true; shift ;;
        --log_dir) log_dir="${2}"; shift 2 ;;
        --run_name) run_name="${2}"; shift 2 ;;
        --checkpoint_dir) checkpoint_dir="${2}"; shift 2 ;;
        --max_retries) max_retries="${2}"; shift 2 ;;
        --retry_delay) retry_delay="${2}"; shift 2 ;;
        --disable_checkpoint) disable_checkpoint=true; shift ;;
        *) echo "Unknown argument: ${1}"; shift ;;
    esac
done

# Display configuration
echo "========================================"
echo "Batch Evaluation Script with Auto-Retry"
echo "========================================"
echo "Script started at: $(date)"
echo "Base Model ID: ${base_model_id}"
echo "Assistant Model ID: ${assistant_model_id}"
echo "Task Name: ${task_name}"
echo "Num Thought Tokens: ${num_thought_tokens}"
echo "Num Return Sequences: ${num_return_sequences}"
echo "Seed Range: ${seed_from} to ${seed_to}"
echo "Test K: ${test_k}"
echo "Tune Assistant Model: ${tune_assistant_model}"
echo "Print Input: ${print_input}"
echo "Print Response: ${print_response}"
echo "Logs will be saved in: ${log_dir}"
echo "Checkpoints will be saved in: ${checkpoint_dir}"
echo "Max Retries: ${max_retries}"
echo "Retry Delay: ${retry_delay} seconds"
echo "Disable Checkpoint: ${disable_checkpoint}"
echo "Run Name: <|start|>${run_name}<|end|>"
echo "========================================"
echo ""

base_model_name="${base_model_id#*/}"
assistant_model_name="${assistant_model_id#*/}"

# Ensure log directory exists
mkdir -p "${log_dir}"

# Ensure checkpoint directory exists
mkdir -p "${checkpoint_dir}"

# Check if the Python script exists
if [[ ! -f "evaluate_softcot.py" ]]; then
    echo "Error: evaluate_softcot.py not found!"
    exit 1
fi

# Check if the retry wrapper exists
if [[ ! -f "run_with_retry.py" ]]; then
    echo "Error: run_with_retry.py not found!"
    exit 1
fi

# Loop over seed range
for ((seed=seed_from; seed<=seed_to; seed++)); do
    start_time=$(date +%s)
    echo ""
    echo "========================================"
    echo "Running evaluation for seed: ${seed}"
    echo "Started at: $(date)"
    echo "========================================"

    log_file_name="${log_dir}/${run_name}${task_name}--seed_${seed}-ntt_${num_thought_tokens}-nrs_${num_return_sequences}-${base_model_name}-${assistant_model_name}.log"

    # Build the evaluation command
    eval_cmd="python evaluate_softcot.py \
--base_model_id \"${base_model_id}\" \
--assistant_model_id \"${assistant_model_id}\" \
--params_file_name \"${params_file_name}\" \
--assistant_model_ckpt \"${assistant_model_ckpt}\" \
--base_model_ckpt \"${base_model_ckpt}\" \
--num_thought_tokens ${num_thought_tokens} \
--num_return_sequences ${num_return_sequences} \
--task_name \"${task_name}\" \
--seed ${seed} \
--test_k ${test_k} \
--checkpoint_dir \"${checkpoint_dir}\""

    # Add optional flags
    ${tune_assistant_model} && eval_cmd+=" --tune_assistant_model"
    ${print_input} && eval_cmd+=" --print_input"
    ${print_response} && eval_cmd+=" --print_response"
    ${disable_checkpoint} && eval_cmd+=" --disable_checkpoint"

    # Build the retry wrapper command
    retry_cmd="python run_with_retry.py \
--max-retries ${max_retries} \
--retry-delay ${retry_delay} \
${eval_cmd}"

    # Run the command with retry wrapper and redirect output
    echo "Executing with auto-retry: ${retry_cmd}"
    echo "Log file: ${log_file_name}"
    echo ""
    
    # Execute command and capture exit code
    eval "${retry_cmd} 2>&1 | tee \"${log_file_name}\""
    exit_code=${PIPESTATUS[0]}

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))

    echo ""
    echo "========================================"
    if [[ ${exit_code} -eq 0 ]]; then
        echo "✓ Evaluation for seed ${seed} completed successfully!"
    else
        echo "✗ Evaluation for seed ${seed} failed with exit code: ${exit_code}"
    fi
    echo "Finished at: $(date)"
    echo "Elapsed time: ${elapsed_time} seconds"
    echo "========================================"
done

# Display the script end time
echo ""
echo "========================================"
echo "All evaluations completed!"
echo "Script finished at: $(date)"
echo "========================================"
