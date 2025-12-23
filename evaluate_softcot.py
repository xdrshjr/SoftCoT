
import re
import argparse
import sys
import os
import signal
from contextlib import contextmanager

from tqdm import tqdm
import torch


from transformers import AutoTokenizer, GenerationConfig
from fastNLP import logger

from llm_model import EfficientSoftCoTFromSmallModel
from data_loader import GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader, DULoader
from utils import pre_process_gsm8k, pre_process_strategy_qa, pre_process_aqua, pre_process_du
from checkpoint_manager import CheckpointManager


args = argparse.ArgumentParser()
args.add_argument('--base_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
args.add_argument('--assistant_model_id', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
args.add_argument('--params_file_name', type=str, default=None)
args.add_argument('--base_model_ckpt', type=str, default=None)
args.add_argument('--assistant_model_ckpt', type=str, default=None)
args.add_argument('--num_thought_tokens', type=int, default=2)
args.add_argument('--num_return_sequences', type=int, default=1)
args.add_argument('--task_name', type=str, choices=[
    'gsm8k', 'strategyqa', 'asdiv-aug', 'aqua', 'du'
])
args.add_argument('--print_input', action='store_true', default=False)
args.add_argument('--print_response', action='store_true', default=False)
args.add_argument('--test_k', type=int, default=0)
args.add_argument('--seed', type=int, default=42)
args.add_argument('--tune_base_model', action='store_true', default=False)
args.add_argument('--tune_assistant_model', action='store_true', default=False)
args.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                  help='Directory to store checkpoints for recovery from failures')
args.add_argument('--disable_checkpoint', action='store_true', default=False,
                  help='Disable checkpoint/resume functionality')
arg = args.parse_args()

# Configure logger - fastNLP logger doesn't need set_stdout_level in older versions
# Just use it as-is, it will output to stdout by default

logger.info("="*80)
logger.info("SoftCoT Evaluation Script with Checkpoint/Resume Support")
logger.info("="*80)
logger.info(f'Command-line Arguments:')
for key, value in sorted(arg.__dict__.items()):
    logger.info(f'  {key}: {value}')
logger.info("="*80)

base_model_id = arg.base_model_id
assistant_model_id = arg.assistant_model_id
params_file_name = arg.params_file_name
base_model_ckpt = arg.base_model_ckpt
assistant_model_ckpt = arg.assistant_model_ckpt
num_thought_tokens = arg.num_thought_tokens
num_return_sequences = arg.num_return_sequences
task_name = arg.task_name
print_input = arg.print_input
print_response = arg.print_response
test_k = arg.test_k
seed = arg.seed
tune_base_model = arg.tune_base_model
tune_assistant_model = arg.tune_assistant_model
checkpoint_dir = arg.checkpoint_dir
disable_checkpoint = arg.disable_checkpoint

large_model_name = base_model_id.split('/')[-1]
small_model_name = assistant_model_id.split('/')[-1]

# Create unique run identifier for checkpoint management
run_identifier = f"{task_name}_seed{seed}_ntt{num_thought_tokens}_nrs{num_return_sequences}_{large_model_name}_{small_model_name}"

# Initialize checkpoint manager
checkpoint_manager = None
if not disable_checkpoint:
    checkpoint_manager = CheckpointManager(checkpoint_dir, run_identifier)
    logger.info(f"Checkpoint management enabled for run: {run_identifier}")
else:
    logger.info("Checkpoint management disabled")

if base_model_ckpt in ['None']:
    base_model_ckpt = None
if assistant_model_ckpt in ['None']:
    assistant_model_ckpt = None

model_dtype = torch.bfloat16
param_dtype = str(model_dtype)

logger.info("")
logger.info("Loading tokenizers...")
base_tokenizer = AutoTokenizer.from_pretrained(base_model_id, token='your-huggingface-token')
logger.info(f"✓ Base tokenizer loaded: {base_model_id}")
assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_id, token='your-huggingface-token')
logger.info(f"✓ Assistant tokenizer loaded: {assistant_model_id}")

if 'Llama' in base_model_id:
    base_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    base_backbone = 'llama'
elif 'Qwen' in base_model_id:
    base_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    # generation_config.pad_token_id = 151643
    base_backbone = 'qwen'
else:
    raise NotImplementedError
if 'Llama' in assistant_model_id:
    assistant_special_token = ['<|end_of_text|>', '<|reserved_special_token_0|>', '<|reserved_special_token_1|>']
    assistant_backbone = 'llama'
elif 'Qwen' in assistant_model_id:
    assistant_special_token = ['<|endoftext|>', '<|box_start|>', '<|box_end|>']
    assistant_backbone = 'qwen'
else:
    raise NotImplementedError

logger.info("")
logger.info("Determining model backbone types...")
logger.info(f"Base model backbone: {base_backbone}")
logger.info(f"Base model special tokens: {base_special_token}")
logger.info(f"Assistant model backbone: {assistant_backbone}")
logger.info(f"Assistant model special tokens: {assistant_special_token}")

logger.info("")
logger.info("Initializing SoftCoT model...")
logger.info(f"Model class: EfficientSoftCoTFromSmallModel")
logger.info(f"Number of thought tokens: {num_thought_tokens}")
logger.info(f"Tune base model: {tune_base_model}")
logger.info(f"Tune assistant model: {tune_assistant_model}")

model = EfficientSoftCoTFromSmallModel(
    assistant_model_id,
    base_model_id,
    num_thought_tokens,
    tune_base_model=tune_base_model,
    tune_assistant_model=tune_assistant_model,
    path_to_projection_module=params_file_name,
    path_to_small_language_model=assistant_model_ckpt,
)
logger.info(f'Successfully Init Model `{model.__class__.__name__}`')
logger.info("")
logger.info("Setting models to evaluation mode...")
model.eval()
model.assistant_model.eval()
model.base_model.eval()
logger.info("✓ All models set to eval mode")

logger.info("")
logger.info(f"Loading dataset for task: {task_name}")

if task_name in ['gsm8k']:
    db = GSM8KLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['strategyqa']:
    db = StrategyQALoader().load()
    preprocess_method = pre_process_strategy_qa
elif task_name in ['asdiv-aug']:
    db = AugASDivLoader().load()
    preprocess_method = pre_process_gsm8k
elif task_name in ['aqua']:
    db = AQuALoader().load()
    preprocess_method = pre_process_aqua
elif task_name in ['du']:
    db = DULoader().load()
    preprocess_method = pre_process_du
else:
    raise NotImplementedError

ds = db.get_dataset('test')
logger.info(f"✓ Dataset loaded successfully")
logger.info(f"  Total samples in test set: {len(ds)}")

if test_k > 0:
    ds = ds[: test_k]
    logger.info(f"  Limited to first {test_k} samples for testing")

logger.info("")
logger.info("Configuring generation parameters...")

generation_config = GenerationConfig.from_pretrained(base_model_id)
if base_backbone in ['llama']:
    generation_config.pad_token_id = 128009
elif base_backbone in ['qwen']:
    generation_config.pad_token_id = 151643
else:
    raise NotImplementedError
generation_config.top_p = 1.0
generation_config.temperature = 1.0
logger.info(f"  Generation config: top_p={generation_config.top_p}, temperature={generation_config.temperature}")
logger.info(f"  Pad token ID: {generation_config.pad_token_id}")

logger.info("")
logger.info("="*80)
logger.info("Starting Evaluation")
logger.info("="*80)

# Check if evaluation is already completed
if checkpoint_manager and checkpoint_manager.is_completed(len(ds)):
    logger.info("Evaluation already completed! Loading statistics...")
    stats = checkpoint_manager.get_statistics()
    checkpoint_manager.finalize()
    sys.exit(0)

# Get resume point
start_idx = 0
correct_count = 0
if checkpoint_manager:
    start_idx = checkpoint_manager.get_resume_index()
    correct_count = checkpoint_manager.get_correct_count()
    if start_idx > 0:
        logger.info(f"Resuming from sample {start_idx}/{len(ds)}, previous correct count: {correct_count}")
else:
    correct_count = 0

# Create progress bar starting from resume point
progress_bar = tqdm(enumerate(ds[start_idx:], start=start_idx), 
                   initial=start_idx, 
                   total=len(ds),
                   desc=f"Evaluating {task_name}")

for idx, ins in progress_bar:

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    logger.info(f"{'='*80}")
    logger.info(f"Processing sample {idx+1}/{len(ds)}")

    if task_name in ['gsm8k', 'asdiv-aug', 'aqua']:
        answer = ins['answer'].split('\n')[-1]
        assert answer.startswith('####')
        answer = answer.replace(',', '')
        if task_name in ['gsm8k', 'asdiv-aug']:
            if '.' in answer:
                answer = float(answer[4:])
            else:
                answer = int(answer[4:])
        else:
            answer = answer[4:].strip()
    elif task_name in ['strategyqa', 'du']:
        answer = ins['answer']
    else:
        raise NotImplementedError

    logger.info(f'Ground Truth Answer: {answer}')

    inputs = preprocess_method(
        ins, base_tokenizer, assistant_tokenizer, num_thought_tokens,
        add_bot_eot=(num_thought_tokens > 0), split='test',
        base_special_token=base_special_token,
        assistant_special_token=assistant_special_token,
        base_backbone=base_backbone,
        assistant_backbone=assistant_backbone,
        device=model.device,
    )
    if print_input:
        logger.info(f'Raw Inputs for Base Model: {base_tokenizer.decode(inputs["input_ids"][0])}')
        # logger.info(f'Raw Inputs for Assistant Model: {assistant_tokenizer.decode(inputs["assistant_input_ids"][0])}')

    terminators = [
        base_tokenizer.eos_token_id,
    ]
    if base_backbone in ['llama']:
        terminators.append(base_tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    model_answer_list = []
    model_answer_count = {}

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    inputs_embeds = model.base_model.get_input_embeddings()(inputs['input_ids'])

    torch.manual_seed(41)
    torch.cuda.manual_seed(41)

    try:
        logger.debug(f"Generating embeddings for sample {idx}")
        inputs_embeds = model.get_inputs_embeds_for_base_model(
            inputs['assistant_input_ids'],
            inputs['assistant_attention_mask'],
            inputs['input_ids'],
            inputs_embeds,
            inputs['thought_index'],
            print_input,
        )
        logger.debug(f"Embeddings generated successfully for sample {idx}")
    except Exception as e:
        logger.error(f"Error during embedding generation for sample {idx}: {e}")
        logger.error(f"Skipping sample {idx} due to error")
        # Save checkpoint even for failed samples to avoid reprocessing
        if checkpoint_manager:
            result = {
                'idx': idx,
                'status': 'error_embedding',
                'error': str(e),
                'is_correct': False
            }
            checkpoint_manager.save_result(idx, result)
            checkpoint_manager.save_checkpoint(idx, len(ds), correct_count)
        continue

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    try:
        logger.debug(f"Starting generation for sample {idx}")
        outputs = model.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs['attention_mask'],
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            generation_config=generation_config,
            num_return_sequences=num_return_sequences,
        )
        logger.debug(f"Generation completed for sample {idx}")
    except Exception as e:
        logger.error(f"Error during generation for sample {idx}: {e}")
        logger.error(f"This might be a segfault or CUDA error. Saving checkpoint...")
        # Save checkpoint before potential crash
        if checkpoint_manager:
            result = {
                'idx': idx,
                'status': 'error_generation',
                'error': str(e),
                'is_correct': False
            }
            checkpoint_manager.save_result(idx, result)
            checkpoint_manager.save_checkpoint(idx, len(ds), correct_count)
        # Re-raise to allow retry mechanism to catch it
        raise

    for i in range(outputs.shape[0]):
        # response = outputs[i][inputs['input_ids'].shape[-1]:]
        response = outputs[i]
        raw_model_answer = base_tokenizer.decode(response, skip_special_tokens=True)

        if print_response:
            logger.info(f'Answer ({idx + 1}-{i + 1}/{len(ds)}): {base_tokenizer.decode(response)}<|end-of-response|>')

        if task_name in ['gsm8k', 'asdiv-aug']:
            cleaned_model_answer = raw_model_answer.replace(',', '')
            cleaned_model_answer = cleaned_model_answer.replace('%', '')
            cleaned_model_answer = cleaned_model_answer.replace('$', '')
        else:
            cleaned_model_answer = raw_model_answer

        match = re.findall(r'\s*([\d,]+(?:\.\d+)?)\s*', cleaned_model_answer)

        if task_name in ['gsm8k', 'asdiv-aug']:
            try:
                if match:
                    last_match = match[-1]
                    cleaned_match = last_match.replace(',', '')
                    cleaned_match = cleaned_match.replace('%', '')
                    cleaned_match = cleaned_match.replace('$', '')
                    if '.' in cleaned_match:
                        model_answer = round(float(cleaned_match), 2)
                    else:
                        model_answer = int(cleaned_match)
                else:
                    model_answer = None
                if model_answer is None and not print_response:
                    logger.info(f'None Model Answer ({idx + 1}-{i + 1}/{len(ds)}): {base_tokenizer.decode(response)}')
            except Exception as e:
                model_answer = None
                logger.error(f'Error parsing model answer: {e}')
                logger.debug(f'Raw answer text: {raw_model_answer}')
        elif task_name in ['strategyqa']:
            last_yes = re.search(r'\bsey\b', raw_model_answer.lower()[::-1])
            if last_yes is not None:
                last_yes = last_yes.start()
            else:
                last_yes = len(raw_model_answer)
            last_no = re.search(r'\bon\b', raw_model_answer.lower()[::-1])
            if last_no is not None:
                last_no = last_no.start()
            else:
                last_no = len(raw_model_answer)
            if last_yes == last_no == len(raw_model_answer):
                model_answer = None
            else:
                model_answer = last_yes < last_no
        elif task_name in ['aqua', 'du']:
            m_answer = re.search(r'\b[a-f]\b', raw_model_answer.lower()[::-1])
            if m_answer is not None:
                model_answer = m_answer.group(0).upper()
            else:
                model_answer = None
        else:
            raise NotImplementedError

        model_answer_list.append(model_answer)
        if model_answer in model_answer_count and model_answer is not None:
            model_answer_count[model_answer] += 1
        else:
            model_answer_count[model_answer] = 1

    max_model_count = 0
    final_model_answer = None

    for k, v in model_answer_count.items():
        if v > max_model_count:
            final_model_answer = k
            max_model_count = v

    logger.info(f'Ground Truth Answer: {answer}')
    logger.info(f'Model Answer: {final_model_answer}')
    is_correct = (final_model_answer == answer)
    logger.info(f'Is Correct: {is_correct}')
    if is_correct:
        correct_count += 1
    
    accuracy = correct_count / (idx + 1) * 100
    logger.info(f'Correct Count: {correct_count}/{idx + 1} ({accuracy:.2f}%)')
    
    # Save checkpoint after successful processing
    if checkpoint_manager:
        result = {
            'idx': idx,
            'status': 'success',
            'ground_truth': answer,
            'model_answer': final_model_answer,
            'is_correct': is_correct,
            'all_answers': model_answer_list
        }
        checkpoint_manager.save_result(idx, result)
        checkpoint_manager.save_checkpoint(idx, len(ds), correct_count, result)
        logger.debug(f"Checkpoint saved for sample {idx}")
    
    # Update progress bar
    progress_bar.set_postfix({
        'correct': correct_count,
        'accuracy': f'{accuracy:.2f}%'
    })
    
    logger.info(f'{"-" * 80}')

# Finalize evaluation
if checkpoint_manager:
    checkpoint_manager.finalize()
    logger.info("All samples processed successfully!")
