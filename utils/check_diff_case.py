import json
import argparse
import sys

def load_results(file_path):
    results = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                task_id = data.get('task_id')
                passed = data.get('passed', False)
                results[task_id] = passed
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare think and no-think results.")
    parser.add_argument("think_file", type=str, nargs='?', default="data/v2_cc_outputs/model_8roll_fixed_step_n10-1200.jsonl.jsonl_results.jsonl", help="Path to the think results file")
    parser.add_argument("nothink_file", type=str, nargs='?', default="data/v2_cc_outputs/model_8roll_fixed_step_nothink_n10-1200.jsonl.jsonl_results.jsonl", help="Path to the no-think results file")
    args = parser.parse_args()

    think_results = load_results(args.think_file)
    nothink_results = load_results(args.nothink_file)

    think_passed_nothink_failed = []
    all_passed = []
    think_failed_nothink_passed = []
    all_failed = []
    common_tasks = set(think_results.keys()) & set(nothink_results.keys())

    for task_id in common_tasks:
        tp = think_results[task_id]
        ntp = nothink_results[task_id]

        if tp and not ntp:
            think_passed_nothink_failed.append(task_id)
        elif tp and ntp:
            all_passed.append(task_id)
        elif not tp and ntp:
            think_failed_nothink_passed.append(task_id)
        else:
            all_failed.append(task_id)

    output_data = {
        "think_passed_nothink_failed": sorted(think_passed_nothink_failed),
        "all_passed": sorted(all_passed),
        "think_failed_nothink_passed": sorted(think_failed_nothink_passed),
        "all_failed": sorted(all_failed)
    }

    output_path = "data/v2_cc_outputs/cc_think_diff.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"--- Comparison Results ---")
    print(f"Think passed but No-think failed ({len(think_passed_nothink_failed)} cases):")
    for tid in output_data["think_passed_nothink_failed"]:
        print(f"  - {tid}")

    print(f"\nAll passed ({len(all_passed)} cases):")
    for tid in output_data["all_passed"]:
        print(f"  - {tid}")

    print(f"\nThink failed but No-think passed ({len(think_failed_nothink_passed)} cases):")
    for tid in output_data["think_failed_nothink_passed"]:
        print(f"  - {tid}")

    print(f"\nAll failed ({len(all_failed)} cases):")
    for tid in output_data["all_failed"]:
        print(f"  - {tid}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
