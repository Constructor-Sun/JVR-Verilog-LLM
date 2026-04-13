import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Calculate pass@1 for reflection results.")
    # Using the filenames provided in the prompt as defaults
    parser.add_argument("--reflection_file", type=str, default="data/outputs/reflection_wo_seq_results.jsonl_results.jsonl", help="Reflection results file")
    parser.add_argument("--perplexity_file", type=str, default="data/outputs/perplexity_results.jsonl", help="Perplexity results file (to get task_kind)")
    args = parser.parse_args()

    # Fallback if default files are not found (searching for variations seen in the root)
    if not os.path.exists(args.reflection_file) and os.path.exists("reflection_wo_seq_results.jsonl_results.jsonl"):
        args.reflection_file = "reflection_wo_seq_results.jsonl_results.jsonl"

    # Load task mapping from perplexity results
    task_to_kind = {}
    if os.path.exists(args.perplexity_file):
        with open(args.perplexity_file, "r") as f:
            # Check if it's JSON or JSONL
            try:
                perp_data = json.load(f)
                for task_id, details in perp_data.items():
                    task_to_kind[task_id] = details.get("task_kind", "unknown")
            except json.JSONDecodeError:
                # Try JSONL
                f.seek(0)
                for line in f:
                    data = json.loads(line)
                    task_id = data.get("task_id")
                    task_to_kind[task_id] = data.get("task_kind", "unknown")
    else:
        print(f"Error: {args.perplexity_file} not found.")
        return

    # Load reflection results and count passes
    results = {} # {kind: {"passed": 0, "total": 0}}
    kinds = ["think_failed_nothink_passed", "think_passed_nothink_failed", "all_failed", "all_passed"]
    for kind in kinds:
        results[kind] = {"passed": 0, "total": 0}

    if not os.path.exists(args.reflection_file):
        print(f"Error: {args.reflection_file} not found.")
        return

    with open(args.reflection_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                task_id = data.get("task_id")
                passed = data.get("passed", False)
                
                kind = task_to_kind.get(task_id)
                if kind in results:
                    results[kind]["total"] += 1
                    if passed:
                        results[kind]["passed"] += 1
                elif kind == "unknown" or kind is None:
                    # Optional: handle tasks not found in perplexity_results
                    pass
            except json.JSONDecodeError:
                continue

    print(f"\n--- Reflection Pass@1 Results ---")
    print(f"{'Task Kind':<30} | {'Passed':<6} | {'Total':<6} | {'Pass@1 Rate':<10}")
    print("-" * 65)
    
    total_passed = 0
    total_tasks = 0
    
    for kind in kinds:
        passed = results[kind]["passed"]
        total = results[kind]["total"]
        total_passed += passed
        total_tasks += total
        pass_rate = (passed / total * 100) if total > 0 else 0.0
        print(f"{kind:<30} | {passed:<6} | {total:<6} | {pass_rate:>10.2f}%")
    
    print("-" * 65)
    overall_rate = (total_passed / total_tasks * 100) if total_tasks > 0 else 0.0
    print(f"{'Overall':<30} | {total_passed:<6} | {total_tasks:<6} | {overall_rate:>10.2f}%")

if __name__ == "__main__":
    main()
