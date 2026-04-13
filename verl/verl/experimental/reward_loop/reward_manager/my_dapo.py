# src/rl/my_dapo_reward.py
import torch
from verl import DataProto
from collections import defaultdict
from verl.workers.reward_manager import register
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager  # 继承原版

@register("my_dapo")
class MyDAPORewardManager(DAPORewardManager):
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        mixed_thinking: bool = False,
        dapo_gamma: float = 0.3,
        dapo_tau: float = 0.35,
        expected_reward: float = 0.7,
        **kwargs
    ):
        super().__init__(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=reward_fn_key,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=overlong_buffer_cfg,
            **kwargs
        )
        self.mixed_thinking = mixed_thinking
        self.dapo_gamma = dapo_gamma
        self.dapo_tau = dapo_tau
        self.expected_reward = expected_reward

    def __call__(self, data: DataProto, return_dict: bool = False):
        print("this is my dapo reward manager!")

        # 先跑原版 DAPO 逻辑：计算 baseline reward + overlong penalty
        result = super().__call__(data, return_dict=True)
        reward_tensor = result["reward_tensor"]
        reward_extra_info = result["reward_extra_info"]

        # 打印原始 reward 信息（调试用）
        if torch.distributed.get_rank() == 0:
            print("\n=== Original Reward (before pair-wise bonus) ===")
            print("reward_tensor shape:", reward_tensor.shape)
            print("reward_tensor (last token per sample):", reward_tensor[:, -1].tolist())
            print("reward_extra_info keys:", list(reward_extra_info.keys()))

        if not self.mixed_thinking:
            if return_dict:
                return result
            else:
                return reward_tensor

        # ===================== pair-wise bonus 逻辑 =====================
        baseline_scores = reward_tensor[:, -1].cpu().numpy()  # 假设 reward 只放在最后一个 token

        from collections import defaultdict
        groups = defaultdict(lambda: {'think': [], 'nothink': []})

        # 收集分组 & 打印每个 example 的 extra_info
        if torch.distributed.get_rank() == 0:
            print("\n=== Per-example extra_info & mode ===")
        
        for i in range(len(data)):
            extra = data[i].non_tensor_batch.get("extra_info", {})
            p_id = extra.get("index", "N/A")
            mode = extra.get("mode", "unknown")
            
            if torch.distributed.get_rank() == 0:
                print(f"  sample {i:3d} | index: {p_id:>8} | mode: {mode:>10} | baseline_score: {baseline_scores[i]:.4f}")

            if p_id != "N/A" and mode in ("think", "no_think"):
                groups[p_id][mode].append((i, baseline_scores[i]))

        if torch.distributed.get_rank() == 0:
            print(f"\nFound {len(groups)} prompt groups with think/no_think samples")

        bonus_count = 0
        bonus_details = []  # 用于记录哪些样本加了 bonus

        for p_id, g in groups.items():
            think_items = g['think']
            if not think_items:
                continue

            avg_think = sum(s for _, s in think_items) / len(think_items)

            if torch.distributed.get_rank() == 0:
                print(f"\nGroup {p_id}:")
                print(f"  avg_think = {avg_think:.4f}  (tau = {self.dapo_tau})")
                print(f"  think samples ({len(think_items)}): {[idx for idx,_ in think_items]}")
                print(f"  nothink samples ({len(g['nothink'])}): {[idx for idx,_ in g['nothink']]}")

            if avg_think < self.dapo_tau:
                if torch.distributed.get_rank() == 0:
                    print("  → skipped (avg_think < tau)")
                continue

            for nt_i, nt_score in g['nothink']:
                if nt_score >= self.expected_reward:
                    # bonus NT
                    nt_last = data[nt_i].batch["attention_mask"].sum() - 1
                    old_nt = reward_tensor[nt_i, nt_last].item()
                    reward_tensor[nt_i, nt_last] += self.dapo_gamma
                    new_nt = reward_tensor[nt_i, nt_last].item()

                    # bonus 成功的 Think
                    think_bonus_count = 0
                    for t_i, t_score in think_items:
                        if t_score >= self.expected_reward:
                            t_last = data[t_i].batch["attention_mask"].sum() - 1
                            old_t = reward_tensor[t_i, t_last].item()
                            reward_tensor[t_i, t_last] += self.dapo_gamma
                            new_t = reward_tensor[t_i, t_last].item()
                            think_bonus_count += 1

                            if torch.distributed.get_rank() == 0:
                                bonus_details.append(
                                    f"  Think sample {t_i:3d} : {old_t:.4f} → {new_t:.4f} (+{self.dapo_gamma})"
                                )

                    if torch.distributed.get_rank() == 0:
                        bonus_details.append(
                            f"  NoThink sample {nt_i:3d}: {old_nt:.4f} → {new_nt:.4f} (+{self.dapo_gamma})"
                            f"  → triggered {think_bonus_count} think bonuses"
                        )

                    bonus_count += 1

        # 记录到 extra_info（使用 list 追加，便于多步累积）
        reward_extra_info.setdefault("dapo_bonus_count", []).append(bonus_count)
        reward_extra_info.setdefault("dapo_bonus_rate", []).append(
            bonus_count / max(1, len(groups))
        )

        if torch.distributed.get_rank() == 0 and bonus_count > 0:
            print("\n=== Applied Bonus Summary ===")
            print(f"  Total bonus groups: {bonus_count}")
            print(f"  Bonus rate: {bonus_count / max(1, len(groups)):.4f}")
            for line in bonus_details:
                print(line)

        # 最终 reward_tensor 概览
        if torch.distributed.get_rank() == 0:
            print("\n=== Final Reward (after pair-wise bonus) ===")
            print("reward_tensor (last token per sample):", reward_tensor[:, -1].tolist())

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor