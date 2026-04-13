import json
import pandas as pd
import os

def load_instances(file_path):
    """
    返回 instance-level 数据，而不是按类别分组
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} does not exist.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = []

    for task_id, details in data.items():
        task_kind = details.get("task_kind")
        perplexity_stats = details.get("perplexity_stats", [])

        df = pd.DataFrame(perplexity_stats)

        instances.append({
            "task_id": task_id,
            "task_kind": task_kind,
            "df": df
        })

    return instances

# It's better to detect the second jump first (threshold=1.5), then calculate accumulated surprisal(threshold>=2.0 or 2.4).
def compute_switches(instances, threshold=1.5):
    """
    对每个 instance 单独判断是否 switch
    返回一个 list，每个元素对应一个 instance
    """
    results = []

    for item in instances:
        df = item["df"]
        task_kind = item["task_kind"]

        switch = False

        if not df.empty and 'perplexity' in df.columns:
            perplexities = df['perplexity'].values

            exceed_count = sum(1 for p in perplexities if p >= threshold)

            if exceed_count >= 2:
                switch = True

        results.append({
            "task_kind": task_kind,
            "switch": switch
        })

    return results

def evaluate_vs_all_think_instance_level(results):
    """
    instance-level 计算 accuracy 变化
    baseline = all-think
    """
    decrease = 0

    for r in results:
        kind = r["task_kind"]
        switch = r["switch"]

        if kind == "think_passed_nothink_failed":
            if not switch:
                decrease += 1

        elif kind == "think_failed_nothink_passed":
            if not switch:
                decrease -= 1

    total = len(results)
    decrease_rate = decrease / total if total > 0 else 0

    return decrease, decrease_rate

def compute_time_saved(results):
    total = len(results)
    total_switches = sum(1 for r in results if r["switch"])

    time_saved = total - total_switches
    percent = time_saved / total if total > 0 else 0

    return time_saved, percent

if __name__ == "__main__":
    results_file = "/mnt/data1/haoyangsun/final_project/data/v2_cc_outputs/cc_perplexity_results.jsonl"
    # results_file = "data/outputs/perplexity_results.json"

    instances = load_instances(results_file)
    threshold = 2.0
    results = compute_switches(instances, threshold=threshold)

    print(50*'-')
    print(f"Total instances: {len(results)}")

    # time
    print("\nthreshold: ", threshold)
    time_saved, time_percent = compute_time_saved(results)
    print(f"Time saved: {time_saved}")
    print(f"Time saved percent: {time_percent:.4f}")

    # accuracy
    decrease_count, decrease_rate = evaluate_vs_all_think_instance_level(results)
    print(f"Pass@1 decrease number: {decrease_count}")
    print(f"Pass@1 decrease percent: {decrease_rate:.4f}")