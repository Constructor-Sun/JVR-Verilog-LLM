import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import json
import re
import time
import math
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def parse_arguments():
    parser = argparse.ArgumentParser(description="Code generation using a fine-tuned language model.")
    parser.add_argument("--cuda_device", type=str, help="CUDA device to use, e.g., '0' or '0,1'")
    parser.add_argument("--model_name", type=str, default="data/models/Qwen3-Coder-30B-A3B-Instruct", help="Model ID to use")
    parser.add_argument("--max_tokens", type=int, default=1024*16, help="Maximum number of tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample from the model")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p value for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k value for nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--input_file", type=str, default="data/inputs/hint_human_act_as_expert.jsonl", help="Input file")
    parser.add_argument("--output_file", type=str, help="Output file")
    parser.add_argument("--n", type=int, default=1, help="Number of completions for each problem to generate")
    parser.add_argument("--zeta", type=float, default=1.4, help="Perplexity threshold for switching to think mode")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization ratio (0.0~1.0)")
    return parser.parse_args()

def setup_environment(args):
    os.environ['TOKENIZERS_PARALLELISM'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

def load_model_and_tokenizer(args):
    # model_path = args.model_name
    # if model_path.startswith('./') or model_path.startswith('../') or model_path == '.':
    #     model_path = os.path.abspath(model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True, trust_remote_code=True)
    model = LLM(
        model=args.model_name,
        tensor_parallel_size=getattr(args, 'tensor_parallel_size', 1),
        gpu_memory_utilization=getattr(args, 'gpu_memory_utilization', 0.9),
        dtype="auto",
        max_model_len=getattr(args, 'max_tokens', None),
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    return model, tokenizer

def generate_completion(model, tokenizer, prompt, args, sys_prompt=None):
    def get_messages(base_prompt, think):
        content = base_prompt + (" /think" if think else " /no_think")
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    # 1. 初始前缀生成 (No Think)
    messages = get_messages(prompt, think=False)
    
    # 核心修复 1：使用 tokenize=True 安全获取包含所有特殊控制字符的 Token IDs
    prompt_token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    
    sampling_params_prefix = SamplingParams(
        max_tokens=80,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=getattr(args, 'top_p', 0.8),
        top_k=getattr(args, 'top_k', 20),
        repetition_penalty=getattr(args, 'repetition_penalty', 1.05),
        skip_special_tokens=True,
        logprobs=20
    )
    
    # 第一阶段生成：直接传入 IDs
    outputs = model.generate(
        prompts=[{"prompt_token_ids": prompt_token_ids}], 
        sampling_params=sampling_params_prefix,
        use_tqdm=False
        )
    prefix_output = outputs[0].outputs[0]
    prefix_token_ids = list(prefix_output.token_ids)
    prefix_text = prefix_output.text
    prefix_logprobs = prefix_output.logprobs
    
    # 计算困惑度
    perplexities = []
    if prefix_logprobs:
        for i, token_id in enumerate(prefix_token_ids):
            if token_id in prefix_logprobs[i]:
                lp = prefix_logprobs[i][token_id].logprob
                perplexities.append(1.0 / math.exp(lp))
            else:
                perplexities.append(1.0)
    
    exceed_count = sum(1 for p in perplexities if p >= args.zeta)
    
    if exceed_count >= 2:
        # ==========================================
        # 【切换分支：Switch to think mode】
        # ==========================================
        messages = get_messages(prompt, think=True)
        # 核心修复 2：同样使用 tokenize=True 获取带有特殊字符的完整 IDs
        think_prompt_token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        
        sampling_params_full = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature if args.do_sample else 0.0,
            top_p=getattr(args, 'top_p', 0.8),
            top_k=getattr(args, 'top_k', 20),
            repetition_penalty=getattr(args, 'repetition_penalty', 1.05),
            skip_special_tokens=True
        )
        
        outputs = model.generate(prompts=[{"prompt_token_ids": think_prompt_token_ids}], sampling_params=sampling_params_full)
        final_output = outputs[0].outputs[0]
        
        total_gen = len(prefix_token_ids) + len(final_output.token_ids)
        return final_output.text, len(final_output.token_ids), len(think_prompt_token_ids), len(think_prompt_token_ids) + len(final_output.token_ids), total_gen, True
    
    else:
        # ==========================================
        # 【继续分支：Continue in no_think mode】
        # ==========================================
        if prefix_output.finish_reason == "stop":
            # 如果在 80 个 token 内就已经生成完毕
            return prefix_text, len(prefix_token_ids), len(prompt_token_ids), len(prompt_token_ids) + len(prefix_token_ids), len(prefix_token_ids), False
        
        # 核心修复 3：无损拼接 IDs，完美命中 Prefix Cache，避免重新切词导致缓存失效
        full_prompt_token_ids = prompt_token_ids + prefix_token_ids
        
        sampling_params_remaining = SamplingParams(
            max_tokens=args.max_tokens - len(prefix_token_ids),
            temperature=args.temperature if args.do_sample else 0.0,
            top_p=getattr(args, 'top_p', 0.8),
            top_k=getattr(args, 'top_k', 20),
            repetition_penalty=getattr(args, 'repetition_penalty', 1.05),
            skip_special_tokens=True
        )
        
        outputs = model.generate(prompts=[{"prompt_token_ids": full_prompt_token_ids}], sampling_params=sampling_params_remaining)
        remaining_output = outputs[0].outputs[0]
        
        # 仅在最后返回文本结果时，拼接字符串
        full_text = prefix_text + remaining_output.text
        full_generated_tokens = len(prefix_token_ids) + len(remaining_output.token_ids)
        
        return full_text, full_generated_tokens, len(prompt_token_ids), len(prompt_token_ids) + full_generated_tokens, full_generated_tokens, False

def post_process_generated_code(completion: str):
    pattern_code_env = r"```(?:verilog|Verilog)\n(.*?)```"
    match = re.search(pattern_code_env, completion, re.DOTALL)
    
    if match:
        verilog_code = match.group(1).strip()
    else:
        verilog_code = completion.strip()
    
    # 删除 module 定义行（更精确的匹配）
    pattern_module_def = r"^\s*module\s+\w+\s*\([^)]*\)\s*;\s*\n"
    verilog_code = re.sub(pattern_module_def, '', verilog_code, flags=re.MULTILINE)
    
    # 删除 endmodule 后的所有内容，保留 endmodule 本身
    # 找到 endmodule，然后删除之后的所有内容
    lines = verilog_code.split('\n')
    for i, line in enumerate(lines):
        if 'endmodule' in line:
            # 保留 endmodule 所在行，但删除该行中 endmodule 之后的内容
            lines[i] = re.sub(r'endmodule\s*.*', 'endmodule', line)
            # 删除 endmodule 之后的所有行
            lines = lines[:i+1]
            break
    
    verilog_code = '\n'.join(lines).strip()
    
    return verilog_code

def process_input_file(args, model, tokenizer):
    task_progress = {}  # {task_id: 已生成数量}
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    tid = data["task_id"]
                    task_progress[tid] = task_progress.get(tid, 0) + 1

    prompt_lengths = []
    generated_lengths = []
    total_lengths = []
    generation_times = []
    total_generated_tokens_list = []
    switch_count = 0
    total_samples = 0
    
    with open(args.input_file, "r", encoding="utf-8") as input_file, \
         open(args.output_file, "a", encoding="utf-8") as output_file:
        for line in tqdm(input_file, desc="Processing inputs"):
            data = json.loads(line)
            task_id = data["task_id"]
            already_done = task_progress.get(task_id, 0)
            if already_done >= args.n:
                continue

            if "fullprompt" in data:
                prompt = data["fullprompt"]
                sys_prompt = data.get("sys")
            else:
                prompt = data["hint"]
                sys_prompt = None

            for _ in range(args.n - already_done):
                start_time = time.time()
                generated_code, gen_tokens, prompt_tokens, total_tokens, total_gen_tokens, switched = generate_completion(model, tokenizer, prompt, args, sys_prompt)
                end_time = time.time()
                
                duration = end_time - start_time
                generation_times.append(duration)
                prompt_lengths.append(prompt_tokens)
                generated_lengths.append(gen_tokens)
                total_lengths.append(total_tokens)
                total_generated_tokens_list.append(total_gen_tokens)
                if switched:
                    switch_count += 1
                total_samples += 1
                
                completion = post_process_generated_code(generated_code)
                # print("completion:\n", completion)
                output = {"task_id": task_id, "completion": completion, "switched": switched}
                json.dump(output, output_file)
                output_file.write("\n")

    if generated_lengths:
        # total_gen_tokens = sum(generated_lengths)
        total_time = sum(generation_times)
        avg_time = total_time / len(generation_times)
        
        print(f"\n" + "="*50)
        print(f"average time per example: {avg_time:.2f}s")
        print(f"average prompt tokens   : {sum(prompt_lengths)/len(prompt_lengths):.1f}")
        print(f"average generated tokens      : {sum(generated_lengths)/len(generated_lengths):.1f}")
        print(f"average total sequence length (prefill+decode) : {sum(total_lengths)/len(total_lengths):.1f}")
        print(f"average actual tokens generated (incl. discarded): {sum(total_generated_tokens_list)/len(total_generated_tokens_list):.1f}")
        print(f"total switch count      : {switch_count} / {total_samples} ({switch_count/total_samples*100:.1f}%)")
        print("="*50 + "\n")

def main():
    args = parse_arguments()
    setup_environment(args)
    model, tokenizer = load_model_and_tokenizer(args)
    process_input_file(args, model, tokenizer)

if __name__ == "__main__":
    main()