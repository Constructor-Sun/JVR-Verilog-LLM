import os
os.environ['TRITON_CACHE_DIR'] = './tmp'
os.environ['TMPDIR'] = './tmp'
import sys
import time
import math
import json
import pandas as pd
from tqdm import tqdm
from vllm import SamplingParams
# pd.set_option('display.max_rows', None)
import argparse

# Ensure project root is in sys.path to allow imports from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import functions from qwen.py
from src.infer.qwen import (
    load_model_and_tokenizer
)

def setup_environment(args):
    os.environ['TOKENIZERS_PARALLELISM'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

def parse_arguments():
    parser = argparse.ArgumentParser(description="Code generation using a fine-tuned language model.")
    parser.add_argument("--cuda_device", type=str, default="3", help="CUDA device to use, e.g., '0' or '0,1'")
    parser.add_argument("--model_name", type=str, default="models/checkpoint-1100", help="Model ID to use")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample from the model")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p value for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k value for nucleus sampling")
    # parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--input_file", type=str, default="data/inputs/hint_human_act_as_expert.jsonl", help="Input file")
    parser.add_argument("--diff_file", type=str, default="data/inputs/think_diff.json", help="Diff file")
    parser.add_argument("--output_file", type=str, default="data/outputs/perplexity_results.json", help="Output file")
    parser.add_argument("--n", type=int, default=1, help="Number of completions for each problem to generate")
    parser.add_argument("--think", action="store_true", help="Whether to think before generating code.")
    return parser.parse_args()

def calculate_perplexity_top(model, tokenizer, prompt, args):
    """
    Calculates the perplexity of the first 20 generated tokens.
    Returns a pandas DataFrame which includes: token_id, decoded_token, probability, entropy, surprisal, and perplexity.
    
    - probability: The probability of the chosen token (P).
    - entropy: The Shannon entropy of the distribution at this step (-sum(p_i * log(p_i))).
    - surprisal: The surprisal of the chosen token (-log P).
    - perplexity: 1/P of the chosen token.
    """
    # Format prompt similarly to generate_completion
    formatted_prompt = "### Hint: Your generated Verilog code should be in ```verilog``` format, otherwise the model will not accept it. " + prompt
    if args.think:
        formatted_prompt = formatted_prompt + " /think"
    else:
        formatted_prompt = formatted_prompt + " /no_think"
        
    messages = [
        {"role": "user", "content": formatted_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # We want to generate a sequence and inspect logprobs for each token
    sampling_params = SamplingParams(
        max_tokens=80, # Generate at least 80 tokens
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=getattr(args, 'top_p', 0.8),
        top_k=getattr(args, 'top_k', 20),
        logprobs=20, # Request top 100 logprobs per token
        skip_special_tokens=False
    )
    
    outputs = model.generate([text], sampling_params)
    
    results = []
    if outputs and outputs[0].outputs:
        output = outputs[0].outputs[0]
        token_ids = output.token_ids
        logprobs_list = output.logprobs
        
        # We only care about the first 20 tokens (or fewer if generation stopped early)
        num_tokens = min(len(token_ids), 80)
        
        for i in range(num_tokens):
            token_id = token_ids[i]
            # logprobs_list[i] is a dict {token_id: Logprob}
            # We need to find the logprob of the chosen token
            token_logprobs = logprobs_list[i]
            
            # vllm ensures the sampled token is in the logprobs dict usually, 
            # but let's be safe. If we requested logprobs=100, it should be there.
            if token_id in token_logprobs:
                logprob_obj = token_logprobs[token_id]
                logprob = logprob_obj.logprob
            else:
                # Fallback: if not in top 100, we might not have the exact logprob easily
                # unless vllm provides it separately.
                # In newer vllm, sample_logprob is often available, or we just assume it's small.
                # However, for now let's just assume it's missing or handle gracefully.
                logprob = -float('inf') # Very high surprisal
            
            p = math.exp(logprob)
            surprisal = -logprob
            perplexity = 1.0 / p if p > 0 else float('inf')
            
            # Calculate Shannon entropy of the distribution at this step
            # H = -sum(p_i * log(p_i))
            step_entropy = 0
            for lp_obj in token_logprobs.values():
                lp = lp_obj.logprob
                p_i = math.exp(lp)
                step_entropy -= p_i * lp

            decoded_token = tokenizer.decode(token_id)
            
            results.append({
                "token_id": token_id,
                "decoded_token": decoded_token,
                "probability": p,
                "entropy": step_entropy,
                "surprisal": surprisal,
                "perplexity": perplexity
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Do NOT sort by entropy here, because the order of tokens matters (it's a sequence)
    # The user asked for "top-20 generated tokens" which implies the sequence order.
    # If they meant "sort the tokens by perplexity", they would have said "top-20 tokens with highest perplexity".
    # But usually "first 20 tokens" means the prefix.
    # Let's keep sequence order.
            
    return df

def process_input_file(args, model, tokenizer):
    if not args.output_file:
        return 

    # Load task mapping from think_diff.json
    diff_file = args.diff_file
    task_to_kind = {}
    if os.path.exists(diff_file):
        with open(diff_file, "r", encoding="utf-8") as f:
            diff_data = json.load(f)
            # all four cases (think_failed_nothink_passed, think_passed_nothink_failed, all_failed, all_passed)
            for kind in ["think_failed_nothink_passed", "think_passed_nothink_failed", "all_failed", "all_passed"]:
                for task_id in diff_data.get(kind, []):
                    task_to_kind[task_id] = kind
    else:
        print(f"Warning: {diff_file} not found. Processing all tasks.")

    # Load existing results if file exists
    all_results = {}
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {args.output_file}. Starting fresh.")

    with open(args.input_file, "r", encoding="utf-8") as input_file:
        total_duration = 0.0
        count = 0
        for line in tqdm(input_file, desc="Processing inputs"):
            data = json.loads(line)
            prompt, task_id = data["hint"], data["task_id"]
            
            # If task_to_kind is not empty, only process tasks in it
            if task_to_kind and task_id not in task_to_kind:
                continue

            # Calculate perplexity statistics for the prompt (first 80 tokens of a generated sequence)
            start_time = time.time()
            perplexity_df = calculate_perplexity_top(model, tokenizer, prompt, args)
            end_time = time.time()
            total_duration += (end_time - start_time)
            count += 1
            
            # Convert DataFrame to list of dicts for JSON serialization
            perplexity_stats = perplexity_df.to_dict(orient='records')
            
            # Save results for each task_id
            all_results[task_id] = {
                "task_id": task_id,
                "task_kind": task_to_kind.get(task_id, "unknown"),
                "perplexity_stats": perplexity_stats
            }

            # Periodically save to avoid data loss
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4)
        
        avg_duration = total_duration / count if count > 0 else 0.0
    
    print(f"Final results saved to {args.output_file}")
    print(f"Average duration per task: {avg_duration:.4f}s")

def main():
    args = parse_arguments()
    setup_environment(args)
    model, tokenizer = load_model_and_tokenizer(args)
    process_input_file(args, model, tokenizer)

if __name__ == "__main__":
    main()
