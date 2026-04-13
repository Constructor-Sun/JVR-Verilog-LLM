import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import re
import pickle
from src.rl.eval_codev import verify_one_sample, verify_one_sample_wrapper, extract_verilog


def compute_score(solution_str, ground_truth, target=1.0, **kwargs):
    gts = pickle.loads(ground_truth)
    
    response_pos = solution_str.find("<|im_start|>assistant")
    if response_pos >= 0:
        solution_str = solution_str[response_pos:]
    else:
        pass

    def check_format(output):
        tags = ["<think>", "</think>"]
        tag_count = [output.count(tag) for tag in tags]
        positions = [output.find(tag) for tag in tags]
        return min(tag_count) == max(tag_count) == 1 and positions[0] < positions[1]

    def calc_reward(solution_str, ground_truth):
        extracted_answer = extract_verilog(solution_str)
        if not check_format(solution_str) or extracted_answer is None:
            reward = 0.0
        else:
            result = verify_one_sample_wrapper((ground_truth, extracted_answer))
            if result["correct"] == True:
                reward = target
            else:
                reward = 0.0
        return reward

    rewards = [calc_reward(solution_str, gt) for gt in gts.values()]
    reward = max(rewards)

    return reward


def compute_score_wrapper(data_source, solution_str, ground_truth, extra_info, **kwargs):
    return compute_score(solution_str, ground_truth, target=1.0, **kwargs)

def compute_pair_score_wrapper(data_source, solution_str, ground_truth, extra_info, **kwargs):
    return compute_score(solution_str, ground_truth, target=0.5, **kwargs)

if __name__ == '__main__':
    file = "data/training_data/zhuyaoyu/CodeV-R1-dataset/codev_r1_rl_train.parquet"
    import pyarrow.parquet as pq
    data = pq.read_table(file).to_pylist()
    
    sep = "============================================"
    print(data[0].keys())
    # correct
    gt = data[0]['reward_model']['ground_truth']
    example_ans = pickle.loads(gt)['answer']['code']
    example_output = f"<think>\n\n</think>  \n```verilog\n{example_ans}```\n"
    reward = compute_score(example_output, gt)
    print(f"{sep}\n{example_output}\n{sep}\n{reward}")

    # wrong format
    example_output = f"<think> ```verilog\n{example_ans}```"
    reward = compute_score(example_output, gt)
    print(f"{sep}\n{example_output}\n{sep}\n{reward}")

    # wrong answer
    example_output = f"<think> </think> <answer>\n```verilog\n```\n</answer>"
    reward = compute_score(example_output, gt)
    print(f"{sep}\n{example_output}\n{sep}\n{reward}")
