from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


# def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
#     """
#     Computes various metrics from a batch of data for PPO training.

#     This function calculates metrics related to scores, rewards, advantages, returns, values,
#     and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
#     for each metric category.

#     Args:
#         batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
#         use_critic: Whether to include critic-specific metrics. Defaults to True.

#     Returns:
#         A dictionary of metrics including:
#             - critic/score/mean, max, min: Statistics about sequence scores
#             - critic/rewards/mean, max, min: Statistics about sequence rewards
#             - critic/advantages/mean, max, min: Statistics about advantages
#             - critic/returns/mean, max, min: Statistics about returns
#             - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
#             - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
#             - response_length/mean, max, min, clip_ratio: Statistics about response lengths
#             - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
#     """
#     sequence_score = batch.batch["token_level_scores"].sum(-1)
#     sequence_reward = batch.batch["token_level_rewards"].sum(-1)

#     advantages = batch.batch["advantages"]
#     returns = batch.batch["returns"]

#     max_response_length = batch.batch["responses"].shape[-1]

#     prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
#     response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

#     max_prompt_length = prompt_mask.size(-1)

#     response_info = _compute_response_info(batch)
#     prompt_length = response_info["prompt_length"]
#     response_length = response_info["response_length"]

#     valid_adv = torch.masked_select(advantages, response_mask)
#     valid_returns = torch.masked_select(returns, response_mask)

#     if use_critic:
#         values = batch.batch["values"]
#         valid_values = torch.masked_select(values, response_mask)
#         return_diff_var = torch.var(valid_returns - valid_values)
#         return_var = torch.var(valid_returns)

#     metrics = {
#         # score
#         "critic/score/mean": torch.mean(sequence_score).detach().item(),
#         "critic/score/max": torch.max(sequence_score).detach().item(),
#         "critic/score/min": torch.min(sequence_score).detach().item(),
#         # reward
#         "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
#         "critic/rewards/max": torch.max(sequence_reward).detach().item(),
#         "critic/rewards/min": torch.min(sequence_reward).detach().item(),
#         # adv
#         "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
#         "critic/advantages/max": torch.max(valid_adv).detach().item(),
#         "critic/advantages/min": torch.min(valid_adv).detach().item(),
#         # returns
#         "critic/returns/mean": torch.mean(valid_returns).detach().item(),
#         "critic/returns/max": torch.max(valid_returns).detach().item(),
#         "critic/returns/min": torch.min(valid_returns).detach().item(),
#         **(
#             {
#                 # values
#                 "critic/values/mean": torch.mean(valid_values).detach().item(),
#                 "critic/values/max": torch.max(valid_values).detach().item(),
#                 "critic/values/min": torch.min(valid_values).detach().item(),
#                 # vf explained var
#                 "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
#             }
#             if use_critic
#             else {}
#         ),
#         # response length
#         "response_length/mean": torch.mean(response_length).detach().item(),
#         "response_length/max": torch.max(response_length).detach().item(),
#         "response_length/min": torch.min(response_length).detach().item(),
#         "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
#         # prompt length
#         "prompt_length/mean": torch.mean(prompt_length).detach().item(),
#         "prompt_length/max": torch.max(prompt_length).detach().item(),
#         "prompt_length/min": torch.min(prompt_length).detach().item(),
#         "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
#     }
#     return metrics


def compute_reward_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """
    Compute the reward metrics for the batch.
    """
    keys = ['recall', 'precision', 'fbeta_score', 'answer_acc']
    metrics = {}
    for key in keys:
        scores = batch.non_tensor_batch[key]
        metrics[f"reward/{key}/mean"] = np.mean(scores)
        metrics[f"reward/{key}/max"] = np.max(scores)
        metrics[f"reward/{key}/min"] = np.min(scores)
    return metrics