#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTL Code Evaluation Script using Icarus Verilog
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import argparse
from collections import defaultdict
from scipy.special import comb
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm


def load_jsonl(filepath: str) -> List[Dict]:
    """读取 JSONL 文件，返回字典列表"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {filepath}: {e}")
    return data


def group_by_task(completions: List[Dict]) -> Dict[str, List[str]]:
    """按 task_id 分组完成项，返回 {task_id: [code1, code2, ...]}"""
    grouped = defaultdict(list)
    for item in completions:
        task_id = item.get('task_id')
        completion = item.get('completion')
        if completion is None:
            completion = item.get("code")
        if task_id and completion:
            grouped[task_id].append(completion)
    return dict(grouped)


def load_testbenches(testbench_file: str) -> Dict[str, str]:
    """加载 testbench 文件，返回 {task_id: testbench_code}"""
    testbenches = {}
    for item in load_jsonl(testbench_file):
        task_id = item.get('task_id')
        testbench = item.get('testbench')
        if task_id and testbench:
            testbenches[task_id] = testbench
    return testbenches


def exec_with_timeout(cmd: List[str], timeout: int = 300, cwd: str = None) -> Tuple[bool, str, str]:
    """
    执行命令并带超时控制
    返回: (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, "", str(e)


# ============= 只需替换 evaluate_single_sample 函数中的这一部分 =============

def evaluate_single_sample(
    task_id: str,
    sample_idx: int,
    code: str,
    testbench: str,
    work_dir: str,
    compile_timeout: int = 60,
    sim_timeout: int = 120,
    data_dir: Optional[str] = None
) -> Dict:
    """
    评估单个样本
    返回: {"syntax_pass": bool, "func_pass": bool, "error": str or None}
    """
    result = {"syntax_pass": False, "func_pass": False, "error": None}
    
    # 创建临时工作目录（使用绝对路径避免歧义）
    task_work_dir = os.path.abspath(os.path.join(work_dir, f"{task_id}_sample_{sample_idx}"))
    os.makedirs(task_work_dir, exist_ok=True)
    
    try:
        # 复制数据文件 (如 .txt, .dat) 到工作目录
        if data_dir and os.path.exists(os.path.join(data_dir, task_id)):
            src_task_dir = os.path.join(data_dir, task_id)
            for item in os.listdir(src_task_dir):
                item_path = os.path.join(src_task_dir, item)
                if os.path.isfile(item_path) and not item.endswith('.v'):
                    shutil.copy2(item_path, task_work_dir)
        
        # 写入设计文件 - 使用简单文件名（相对于 cwd）
        design_filename = "design.v"
        design_path = os.path.join(task_work_dir, design_filename)
        with open(design_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 写入 testbench 文件 - 使用简单文件名（相对于 cwd）
        tb_filename = "testbench.v"
        tb_path = os.path.join(task_work_dir, tb_filename)
        with open(tb_path, 'w', encoding='utf-8') as f:
            f.write(testbench)
        
        # 步骤1: 使用 iverilog 编译
        # 【关键修改】使用相对于 cwd 的简单文件名，而非完整路径
        iverilog_cmd = [
            'iverilog',
            '-o', 'output.vvp',  # 输出文件也用相对路径
            design_filename,      # ✅ 改为简单文件名
            tb_filename           # ✅ 改为简单文件名
        ]
        
        success, stdout, stderr = exec_with_timeout(
            iverilog_cmd, 
            timeout=compile_timeout, 
            cwd=task_work_dir  # cwd 已设为任务目录，相对路径在此解析
        )
        
        if not success:
            result["error"] = f"Compilation failed: {stderr[:200]}"
            return result
        
        result["syntax_pass"] = True
        
        # 步骤2: 使用 vvp 运行仿真
        vvp_cmd = ['vvp', 'output.vvp']  # ✅ 也用相对路径
        success, stdout, stderr = exec_with_timeout(
            vvp_cmd,
            timeout=sim_timeout,
            cwd=task_work_dir
        )
        
        output_text = stdout + stderr
        
        # 判断功能是否通过
        if success and ('pass' in output_text.lower()):
            result["func_pass"] = True
        else:
            result["error"] = f"Simulation failed or no Pass found. Output: {output_text[:200]}"
        
        return result
        
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        return result
    finally:
        # 清理临时文件（可选）
        # shutil.rmtree(task_work_dir, ignore_errors=True)
        pass


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    计算 Pass@k 指标
    公式: 1 - C(n-c, k) / C(n, k)
    n: 总样本数, c: 通过样本数, k: 采样数
    """
    if n - c < k:
        return 1.0
    if n < k:
        return 0.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def reveal_failure_reasons(task_results: Dict):
    """
    Reveal the reason why it did not pass for each task
    """
    print("\n" + "=" * 60)
    print("Detailed Failure Reasons per Task:")
    print("=" * 60)

    for task_id in sorted(task_results.keys()):
        data = task_results[task_id]
        # Only show tasks that have at least one failure
        if not all(data["func"]):
            print(f"\nTask: {task_id}")
            errors = data.get("errors", [])

            # Group errors to avoid redundant output
            error_summary = defaultdict(int)
            for idx, (syntax, func, error) in enumerate(zip(data["syntax"], data["func"], errors)):
                if not func:
                    if not syntax:
                        reason = f"Syntax/Compile Error: {error}"
                    else:
                        reason = f"Functional/Simulation Error: {error}"
                    error_summary[reason] += 1

            for reason, count in error_summary.items():
                print(f"  - [{count} samples] {reason}")


def evaluate_all(
    completions_file: str,
    testbench_file: str,
    output_dir: str,
    k_values: List[int] = [1, 5],
    max_workers: int = 4,
    compile_timeout: int = 60,
    sim_timeout: int = 120,
    data_dir: Optional[str] = None
) -> Dict:
    """
    主评估函数
    """
    # 加载数据
    print(f"Loading completions from {completions_file}...")
    completions = load_jsonl(completions_file)
    
    print(f"Loading testbenches from {testbench_file}...")
    testbenches = load_testbenches(testbench_file)
    
    # 按 task_id 分组
    grouped_completions = group_by_task(completions)
    
    # 统计结果
    results = defaultdict(lambda: {"syntax_pass": 0, "func_pass": 0, "total": 0})
    
    # 创建主工作目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有待评估任务
    tasks = []
    for task_id, codes in grouped_completions.items():
        if task_id not in testbenches:
            print(f"Warning: No testbench found for task '{task_id}', skipping...")
            continue
        testbench = testbenches[task_id]
        for idx, code in enumerate(codes):
            tasks.append((task_id, idx, code, testbench))
    
    print(f"Total {len(tasks)} samples to evaluate across {len(grouped_completions)} tasks")

    # 重新组织：按 task_id 统计
    task_results = defaultdict(lambda: {"syntax": [], "func": [], "errors": []})

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {}
        for task_id, codes in grouped_completions.items():
            if task_id not in testbenches:
                continue
            testbench = testbenches[task_id]
            for idx, code in enumerate(codes):
                future = executor.submit(
                    evaluate_single_sample,
                    task_id, idx, code, testbench,
                    output_dir,
                    compile_timeout,
                    sim_timeout,
                    data_dir
                )
                future_to_task[future] = (task_id, idx)

        for future in tqdm.tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Evaluating"):
            task_id, idx = future_to_task[future]
            result = future.result()
            task_results[task_id]["syntax"].append(result["syntax_pass"])
            task_results[task_id]["func"].append(result["func_pass"])
            task_results[task_id]["errors"].append(result["error"])
    
    # 计算每个任务的通过率
    print("\n" + "="*60)
    print("Per-task Results:")
    print("="*60)
    
    syntax_counts = []
    func_counts = []
    
    for task_id in sorted(task_results.keys()):
        data = task_results[task_id]
        n = len(data["syntax"])
        c_syntax = sum(data["syntax"])
        c_func = sum(data["func"])
        syntax_counts.append((n, c_syntax))
        func_counts.append((n, c_func))
        
        print(f"{task_id:20s} | Syntax: {c_syntax}/{n} ({100*c_syntax/n:5.1f}%) | "
              f"Func: {c_func}/{n} ({100*c_func/n:5.1f}%)")
    
    # 计算整体 Pass@k
    print("\n" + "="*60)
    print("Pass@k Metrics (averaged across tasks):")
    print("="*60)
    
    for k in k_values:
        # 对每个任务计算 Pass@k，然后取平均
        syntax_pass_k = []
        func_pass_k = []
        
        for (n, c_syntax), (_, c_func) in zip(syntax_counts, func_counts):
            if n >= k:
                syntax_pass_k.append(calculate_pass_at_k(n, c_syntax, k))
                func_pass_k.append(calculate_pass_at_k(n, c_func, k))
        
        if syntax_pass_k:
            avg_syntax = sum(syntax_pass_k) / len(syntax_pass_k)
            avg_func = sum(func_pass_k) / len(func_pass_k)
            print(f"Pass@{k:2d} | Syntax: {avg_syntax:.4f} | Func: {avg_func:.4f}")
    
    # 计算全局聚合的 Pass@k（所有任务的样本合并）
    print("\n" + "="*60)
    print("Pass@k Metrics (globally aggregated):")
    print("="*60)
    
    total_n_syntax = sum(n for n, _ in syntax_counts)
    total_c_syntax = sum(c for _, c in syntax_counts)
    total_n_func = sum(n for n, _ in func_counts)
    total_c_func = sum(c for _, c in func_counts)
    
    for k in k_values:
        if total_n_syntax >= k:
            global_syntax = calculate_pass_at_k(total_n_syntax, total_c_syntax, k)
            global_func = calculate_pass_at_k(total_n_func, total_c_func, k)
            print(f"Pass@{k:2d} | Syntax: {global_syntax:.4f} | Func: {global_func:.4f}")

    # Reveal failure reasons for tasks that did not pass
    reveal_failure_reasons(task_results)

    return {
        "task_results": dict(task_results),
        "syntax_counts": syntax_counts,
        "func_counts": func_counts
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RTL code generation using Icarus Verilog"
    )
    parser.add_argument('--completions', type=str, required=True,
                        help='Path to JSONL file with model completions')
    parser.add_argument('--testbenches', type=str, required=True,
                        help='Path to JSONL file with testbenches')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing task data files (e.g. .txt, .dat)')
    parser.add_argument('--output_dir', type=str, default='./eval_output',
                        help='Directory for temporary working files')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 5],
                        help='K values for Pass@k calculation (default: 1 5')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--compile_timeout', type=int, default=60,
                        help='Compilation timeout in seconds (default: 60)')
    parser.add_argument('--sim_timeout', type=int, default=120,
                        help='Simulation timeout in seconds (default: 120)')
    parser.add_argument('--keep_temp', action='store_true',
                        help='Keep temporary working files for debugging')
    
    args = parser.parse_args()
    
    # 检查 iverilog 是否可用
    if shutil.which('iverilog') is None:
        print("Error: 'iverilog' not found in PATH. Please install Icarus Verilog.")
        print("Installation: sudo apt install iverilog (Ubuntu) or brew install icarus-verilog (macOS)")
        sys.exit(1)
    
    if shutil.which('vvp') is None:
        print("Error: 'vvp' not found in PATH. Please install Icarus Verilog.")
        sys.exit(1)
    
    # 运行评估
    results = evaluate_all(
        completions_file=args.completions,
        testbench_file=args.testbenches,
        output_dir=args.output_dir,
        k_values=args.k,
        max_workers=args.workers,
        compile_timeout=args.compile_timeout,
        sim_timeout=args.sim_timeout,
        data_dir=args.data_dir
    )
    
    # 保存详细结果
    result_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to {result_path}")
    
    # 如果不保留临时文件，清理工作目录
    if not args.keep_temp:
        # 只清理子目录，保留结果文件
        for item in os.listdir(args.output_dir):
            item_path = os.path.join(args.output_dir, item)
            if os.path.isdir(item_path) and item.startswith('width_8to16'):
                shutil.rmtree(item_path, ignore_errors=True)
        print(f"Temporary files cleaned (use --keep_temp to preserve)")


if __name__ == "__main__":
    main()