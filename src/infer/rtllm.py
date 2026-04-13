import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import json
import re
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

def parse_arguments():
    parser = argparse.ArgumentParser(description="Code generation using a fine-tuned language model.")
    parser.add_argument("--cuda_device", type=str, help="CUDA device to use, e.g., '0' or '0,1'")
    parser.add_argument("--model_name", type=str, default="data/models/Qwen3-Coder-30B-A3B-Instruct", help="Model ID to use")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample from the model")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p value for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k value for nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--input_file", type=str, default="data/inputs/rtllm-v1.1-instruction.jsonl", help="Input file")
    parser.add_argument("--output_file", type=str, help="Output file")
    parser.add_argument("--n", type=int, default=1, help="Number of completions for each problem to generate")
    parser.add_argument("--think", action="store_true", help="Whether to think before generating code.")
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
    )
    return model, tokenizer

def generate_completion(model, tokenizer, prompt, args):
    # prompt = "### Hint: Your generated Verilog code should be in ```verilog``` format, otherwise the model will not accept it. " + prompt
    if args.think:
        prompt = prompt + " /think"
    else:
        prompt = prompt + " /no_think"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=getattr(args, 'top_p', 0.8),
        top_k=getattr(args, 'top_k', 20),
        repetition_penalty=getattr(args, 'repetition_penalty', 1.05),
        skip_special_tokens=True
    )

    prompt_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    outputs = model.generate([text], sampling_params)
    generated_tokens = len(outputs[0].outputs[0].token_ids)
    total_tokens = prompt_tokens + generated_tokens
    content = outputs[0].outputs[0].text

    return content, generated_tokens, prompt_tokens, total_tokens
    
    # outputs = model.generate([text], sampling_params)
    # content = outputs[0].outputs[0].text
    
    # return content

def post_process_generated_code(completion: str):
    pattern_code_env = r"```(?:verilog|Verilog)\n(.*?)```"
    match = re.search(pattern_code_env, completion, re.DOTALL)
    
    if match:
        verilog_code = match.group(1).strip()
    else:
        verilog_code = completion.strip()
    
    # 删除 module 定义行（更精确的匹配）
    # pattern_module_def = r"^\s*module\s+\w+\s*\([^)]*\)\s*;\s*\n"
    # verilog_code = re.sub(pattern_module_def, '', verilog_code, flags=re.MULTILINE)
    
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
    with open(args.input_file, "r", encoding="utf-8") as input_file, \
         open(args.output_file, "a", encoding="utf-8") as output_file:
        for line in tqdm(input_file, desc="Processing inputs"):
            data = json.loads(line)
            task_id = data["task_id"]
            already_done = task_progress.get(task_id, 0)
            if already_done >= args.n:
                continue

            prompt = data["instructions"]
            for _ in range(args.n - already_done):
                generated_code, gen_tokens, prompt_tokens, total_tokens = generate_completion(model, tokenizer, prompt, args)
                prompt_lengths.append(prompt_tokens)
                generated_lengths.append(gen_tokens)
                total_lengths.append(total_tokens)
                completion = post_process_generated_code(generated_code)
                # print("completion:\n", completion)
                output = {"task_id": task_id, "completion": completion}
                json.dump(output, output_file)
                output_file.write("\n")

    if generated_lengths:
        print(f"average prompt tokens   : {sum(prompt_lengths)/len(prompt_lengths):.1f}")
        print(f"average generated tokens      : {sum(generated_lengths)/len(generated_lengths):.1f}")
        print(f"average total sequence length (prefill+decode) : {sum(total_lengths)/len(total_lengths):.1f}")

def main():
    args = parse_arguments()
    setup_environment(args)
    model, tokenizer = load_model_and_tokenizer(args)
    process_input_file(args, model, tokenizer)

if __name__ == "__main__":
    main()