"""ML2025 Spring HW6 - Fine-tuning leads to Forgetting (local PyTorch version).

Ported from the official Colab (HW06/hw6_colab.ipynb):
https://colab.research.google.com/drive/1sXopMDAT0nRrOTL52ECSPV07gKNoDn7n

LoRA-fine-tune meta-llama/Llama-3.2-1B-Instruct (4-bit NF4, i.e. QLoRA) on GSM8K, then answer
the GSM8K public/private test questions and the AILuminate prompts. The output file has the
notebook's format: one Python list, 264 GSM8K answers followed by 240 AILuminate responses.

Differences from Colab:
- Data is read from `HW06/data/` and everything is written under `--out-dir`
  (default `HW06/outputs/<run name>/`) instead of the current directory.
- The model comes from the local HF cache (run with `HF_HUB_OFFLINE=1` if it is a mirror copy).
- `--seed` fixes torch's startup seed. The Colab never sets one, and LoRA A is initialised
  before SFTTrainer's set_seed(1126), so without it every run starts from a different LoRA.
- The raw responses are also saved (`responses.json`), so answers can be re-checked offline.
- Every hint of the slides' Medium/Strong table is a flag (lr, epochs, weight decay, LoRA
  dropout, n-shot, max_new_tokens, greedy decoding, self-instruct data, checkpoint to evaluate).
With no flags, the run is the notebook's: `HW06_TORCH_SEED=0 python docs/tools/hw06_det.py HW06/hw6.py`
gives bit-identical checkpoints and outputs to the notebook run the same way on the same day.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

MODEL_ID = 'meta-llama/Llama-3.2-1B-Instruct'
HW_DIR = Path(__file__).resolve().parent
# the header of every assistant turn; with --completion-only-loss only the tokens after the last one count
RESPONSE_TEMPLATE = '<|start_header_id|>assistant<|end_header_id|>\n\n'
TRAIN_FILES = {'original': 'gsm8k_train.jsonl', 'self-instruct': 'gsm8k_train_self-instruct.jsonl'}
FIXED_SHOTS = None  # --fixed-shots-seed: the same examples, same order, for every prompt (Strong hint)

random.seed(42)  # the notebook seeds Python's random at import time; the few-shot draws use it


def load_jsonlines(file_name):
    f = open(file_name, 'r')
    return [json.loads(line) for line in f]


def load_csv(file_name):
    csvfile = open(file_name)
    rows = csv.DictReader(csvfile)
    questions = []
    for row in rows:
        questions.append(row['prompt_text'])
    return questions


def load_model(lora_dropout):
    """Cell [12]: 4-bit base model, tokenizer with an extra [PAD], LoRA injected into the model."""
    sft_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    sft_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        quantization_config=sft_bnb_config,
        low_cpu_mem_usage=True,
    )
    sft_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_ID)
    # adds token 128256, but the embedding is not resized: fine only because batch size 1 never pads
    sft_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    lora_kwargs = {} if lora_dropout is None else {'lora_dropout': lora_dropout}
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        **lora_kwargs,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'],
    )
    # get_peft_model wraps sft_model and also injects the LoRA layers into it in place
    peft_model = get_peft_model(sft_model, peft_config)
    return sft_model, sft_tokenizer, peft_config, peft_model


def nshot_chats(nshot_data, n, question, answer, mode):
    """Cell [14]: n random examples as user/assistant turns, then the question."""
    if mode not in ['train', 'test']:
        raise AssertionError('Undefined Mode!!!')

    chats = []
    examples = random.sample(nshot_data, n) if FIXED_SHOTS is None else FIXED_SHOTS[:n]
    for qna in examples:
        chats.append({'role': 'user', 'content': f'Q: {qna["question"]}'})
        chats.append({'role': 'assistant', 'content': f'A: {qna["answer"]}'})

    chats.append({
        'role': 'user',
        'content': f'Q: {question} Let\'s think step by step. At the end, you MUST write the answer as an integer after \'####\'.',
    })
    if mode == 'train':
        chats.append({'role': 'assistant', 'content': f'A: {answer}'})

    return chats


def format_train_data(gsm8k_train, tokenizer, n_shot):
    """Cell [16]: one chat per training question, rendered with the chat template."""
    formatted_gsm8k = []
    max_token_len = 0  # the longest sample sets SFTConfig.max_seq_length, so nothing is truncated
    for qna in gsm8k_train:
        chats = nshot_chats(nshot_data=gsm8k_train, n=n_shot, question=qna['question'], answer=qna['answer'], mode='train')
        train_sample = tokenizer.apply_chat_template(chats, tokenize=False)
        # drop the system header ("Cutting Knowledge Date ...") that the template always adds
        train_sample = train_sample[train_sample.index("<|eot_id|>") + len("<|eot_id|>"):]
        formatted_gsm8k.append({'text': train_sample})
        max_token_len = max(max_token_len, len(tokenizer(train_sample)['input_ids']))
    return Dataset.from_list(formatted_gsm8k), max_token_len


def train(peft_model, peft_config, tokenizer, dataset, max_token_len, args):
    """Cell [18]: SFTTrainer. peft_model is already a PeftModel, so trl uses it as is."""
    extra = {}
    if args.weight_decay is not None:
        extra['weight_decay'] = args.weight_decay
    if args.warmup_ratio is not None:
        extra['warmup_ratio'] = args.warmup_ratio
    training_arguments = SFTConfig(
        seed=1126,
        data_seed=1126,
        output_dir=str(args.out_dir / 'sft'),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=0.1,
        save_strategy="steps",
        save_steps=0.1,
        lr_scheduler_type=args.scheduler,
        learning_rate=args.lr,
        **extra,
        bf16=True,
        group_by_length=True,
        max_seq_length=max_token_len,
        dataset_text_field='text',
        report_to='none',
    )
    collator = {}
    if args.completion_only_loss:
        # trl's default collator sets labels = input_ids, so the loss covers the whole sample (few-shot
        # examples and the question included); this one masks everything before the last assistant header
        collator['data_collator'] = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
        **collator,
    )
    trainer.train()
    return trainer


def make_generator(sft_model, tokenizer, adapter_path, args):
    """Cell [21]: text-generation pipeline on sft_model, then load the checkpoint to evaluate.

    The notebook writes `pipeline.model = PeftModel.from_pretrained(sft_model, adapter_path)`,
    which sets an attribute on the `pipeline` function, not on the generator. It still works:
    from_pretrained loads the checkpoint into the LoRA layers that already sit inside sft_model,
    and the generator runs sft_model (checked by docs/tools/hw06_adapter_probe.py). The call is
    kept as is, assignment included: creating those LoRA layers draws from torch's RNG, and
    keeping the returned PeftModel alive keeps the GPU memory layout of the notebook run; the
    sampled answers that follow depend on both.
    """
    decoding = {'do_sample': False} if args.greedy else {'do_sample': True, 'temperature': 0.6, 'top_p': 0.9}
    generator = pipeline(
        'text-generation',
        model=sft_model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=args.max_new_tokens,
        **decoding,
    )
    pipeline.model = PeftModel.from_pretrained(sft_model, adapter_path)
    return generator


def get_response(generator, chats):
    gen_text = generator(chats)[0]  # first return sequence
    return gen_text['generated_text'][-1]['content']


def extract_ans_from_response(answer):
    """Cell [23]: the text after the last '####', minus ',', '$', '%' and every 'g'."""
    answer = answer.split('####')[-1].strip()
    for remove_char in [',', '$', '%', 'g']:
        answer = answer.replace(remove_char, '')
    return answer


def evaluate_gsm8k(generator, gsm8k_train, data_dir, n_shot, raw):
    """Cell [24]: public set with accuracy, then private set; few-shot examples drawn per question."""
    gsm8k_predictions = []
    gsm8k_test_public = load_jsonlines(data_dir / 'gsm8k_test_public.jsonl')
    gsm8k_total = len(gsm8k_test_public)
    gsm8k_progress_bar = tqdm(total=gsm8k_total, desc='GSM8K Public Test Data Evaluation', postfix='Current Accuracy = 0.000')
    correct = 0
    for i, qna in enumerate(gsm8k_test_public):
        messages = nshot_chats(nshot_data=gsm8k_train, n=n_shot, question=qna['question'], answer=None, mode='test')
        response = get_response(generator, messages)
        pred_ans = extract_ans_from_response(response)
        true_ans = extract_ans_from_response(qna["answer"])
        if pred_ans == true_ans:
            correct += 1
        gsm8k_predictions.append(pred_ans)
        raw['gsm8k_public'].append({'response': response, 'pred': pred_ans, 'true': true_ans})
        gsm8k_progress_bar.set_postfix_str(f'Current Accuracy = {correct/(i+1):.3f}')
        gsm8k_progress_bar.update()
    gsm8k_progress_bar.close()
    print(f'GSM8K Public Test Data Evaluation Complete, Total Accuracy: {correct/gsm8k_total:.3f}')
    raw['gsm8k_public_acc'] = correct / gsm8k_total

    gsm8k_test_private = load_jsonlines(data_dir / 'gsm8k_test_private.jsonl')
    gsm8k_progress_bar = tqdm(total=len(gsm8k_test_private), desc='GSM8K Private Test Data Inference')
    for i, qna in enumerate(gsm8k_test_private):
        messages = nshot_chats(nshot_data=gsm8k_train, n=n_shot, question=qna['question'], answer=None, mode='test')
        response = get_response(generator, messages)
        pred_ans = extract_ans_from_response(response)
        gsm8k_predictions.append(pred_ans)
        raw['gsm8k_private'].append({'response': response, 'pred': pred_ans})
        gsm8k_progress_bar.update()
    gsm8k_progress_bar.close()
    print('GSM8K Private Test Data Inference Complete')
    return gsm8k_predictions


def evaluate_ailuminate(generator, data_dir, raw):
    """Cell [27]: each AILuminate prompt alone, no few-shot, default system header."""
    ailuminate_predictions = []
    ailuminate_test = load_csv(data_dir / 'ailuminate_test.csv')
    ailuminate_progress_bar = tqdm(total=len(ailuminate_test), desc='AILuminate Test Data Evaluation')
    for i, question in enumerate(ailuminate_test):
        message = [{'role': 'user', 'content': question}]
        response = get_response(generator, message)
        ailuminate_predictions.append(response)
        raw['ailuminate'].append({'response': response})
        ailuminate_progress_bar.update()
    ailuminate_progress_bar.close()
    print('AIluminate Test Data Evaluation Complete')
    return ailuminate_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=None, help='default: HW06/outputs/<--name>')
    parser.add_argument('--name', default='run', help='run name, used for the default --out-dir')
    parser.add_argument('--data-dir', type=Path, default=HW_DIR / 'data')
    parser.add_argument('--seed', type=int, default=None, help="torch seed for the LoRA init (Colab: none, random)")
    parser.add_argument('--student-id', default='', help="output file is <student-id>.txt (Colab: '' -> .txt)")
    # training data and prompt
    parser.add_argument('--train-data', choices=list(TRAIN_FILES), default='original')
    parser.add_argument('--train-n-shot', type=int, default=1, help='Medium hint: 5-8 (same as --test-n-shot)')
    parser.add_argument('--test-n-shot', type=int, default=1)
    parser.add_argument('--fixed-shots-seed', type=int, default=None,
                        help='Strong hint: fixed few-shot examples, drawn once from the training set with this seed')
    # training
    parser.add_argument('--skip-train', action='store_true', help='only evaluate --adapter')
    parser.add_argument('--completion-only-loss', action='store_true',
                        help='loss on the final answer only (Colab: the whole sample, few-shot examples included)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Medium hint: 1e-4 to 1e-5')
    parser.add_argument('--epochs', type=float, default=1, help='Strong hint: 3-5')
    parser.add_argument('--scheduler', default='linear')
    parser.add_argument('--warmup-ratio', type=float, default=None, help='Colab: not set (0)')
    parser.add_argument('--weight-decay', type=float, default=None, help='Strong hint: 1e-2 to 1e-4 (Colab: not set, 0)')
    parser.add_argument('--lora-dropout', type=float, default=None, help='Strong hint: 0.1-0.2 (Colab: not set, 0)')
    # evaluation
    parser.add_argument('--adapter', default=None, help='checkpoint to evaluate (default: the last one of this run)')
    parser.add_argument('--max-new-tokens', type=int, default=256, help='Medium hint: 512-1024')
    parser.add_argument('--greedy', action='store_true', help='Medium hint: greedy decoding (Colab: sampling)')
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = HW_DIR / 'outputs' / args.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    sft_model, sft_tokenizer, peft_config, peft_model = load_model(args.lora_dropout)
    gsm8k_train = load_jsonlines(args.data_dir / TRAIN_FILES[args.train_data])
    if args.fixed_shots_seed is not None:
        global FIXED_SHOTS
        FIXED_SHOTS = random.Random(args.fixed_shots_seed).sample(gsm8k_train, max(args.train_n_shot, args.test_n_shot))
    # the formatting loop runs even with --skip-train: its random.sample calls decide which
    # few-shot examples the test questions get afterwards
    dataset, max_token_len = format_train_data(gsm8k_train, sft_tokenizer, args.train_n_shot)
    print(f'{len(dataset)} training samples, {args.train_n_shot}-shot, max token length {max_token_len}')

    if not args.skip_train:
        trainer = train(peft_model, peft_config, sft_tokenizer, dataset, max_token_len, args)
        if args.adapter is None:
            args.adapter = str(args.out_dir / 'sft' / f'checkpoint-{trainer.state.global_step}')
    if args.adapter is None:
        parser.error('--skip-train needs --adapter')

    generator = make_generator(sft_model, sft_tokenizer, args.adapter, args)
    raw = {'args': {k: str(v) for k, v in vars(args).items()},
           'gsm8k_public': [], 'gsm8k_private': [], 'ailuminate': []}
    gsm8k_predictions = evaluate_gsm8k(generator, gsm8k_train, args.data_dir, args.test_n_shot, raw)
    ailuminate_predictions = evaluate_ailuminate(generator, args.data_dir, raw)

    # cell [29]: one Python list in a .txt file, as JudgeBoi expected
    with open(args.out_dir / f'{args.student_id}.txt', 'w') as output_f:
        print(gsm8k_predictions + ailuminate_predictions, file=output_f)
    json.dump(raw, open(args.out_dir / 'responses.json', 'w'), ensure_ascii=False, indent=1)
    print(f'wrote {args.out_dir}')


if __name__ == '__main__':
    main()
