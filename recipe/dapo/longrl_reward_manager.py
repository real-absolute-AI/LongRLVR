
import re
import openai
import random
import torch
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import psutil




def extract_used_context(solution_str):
    used_context_pattern = "<useful_chunks>(.*?)</useful_chunks>"
    match = re.finditer(used_context_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # Ensure there is exactly one match
    if len(matches) != 1:
        return None
    return matches[0].group(1).strip()


def extract_answer(solution_str):
    answer_pattern = "<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    # Ensure there is exactly one match
    if len(matches) != 1:
        return None
    return matches[0].group(1).strip()

def convert_to_chunks(used_context):
    """
    Extract chunk IDs from used_context string and convert to list of integers.
    Handles various separators and formats more robustly.
    
    Args:
        used_context: String containing chunk references like "<CHUNK_0>, <CHUNK_1>" or "<CHUNK_0>\n<CHUNK_1>"
    
    Returns:
        List of integers representing chunk IDs
    """
    if not used_context or not isinstance(used_context, str):
        return []
    
    # First extract all chunk patterns using regex
    chunk_pattern = r'<CHUNK_(\d+)>'
    matches = re.findall(chunk_pattern, used_context)
    
    # Convert matched strings to integers
    used_chunks = []
    for match in matches:
        try:
            used_chunks.append(int(match))
        except ValueError:
            # Skip invalid chunk IDs
            continue
    return used_chunks
    # Remove duplicates while preserving order
    # seen = set()
    # unique_chunks = []
    # for chunk_id in used_chunks:
    #     if chunk_id not in seen:
    #         seen.add(chunk_id)
    #         unique_chunks.append(chunk_id)
    
    # return unique_chunks

def extract_binary_judgment_from_response(response_text: str) -> float:
    """
    Extract binary judgment (Yes/No) from LLM response.
    
    Args:
        response_text: The response text from the LLM
    
    Returns:
        1.0 for Yes, 0.0 for No, or None if parsing fails
    """
    try:
        # Clean the response text
        cleaned_text = response_text.strip().upper()
        
        # Look for judgment in double brackets format [[Yes]] or [[No]]
        judgment_pattern = r'\[\[(YES|NO)\]\]'
        match = re.search(judgment_pattern, cleaned_text)
        
        if match:
            judgment = match.group(1)
            return 1.0 if judgment == "YES" else 0.0
        else:
            # Fallback to looking for Yes/No without brackets
            if "YES" in cleaned_text:
                return 1.0
            elif "NO" in cleaned_text:
                return 0.0
            else:
                print(f"Warning: Could not extract judgment from response: {cleaned_text}")
                return None
            
    except Exception as e:
        print(f"Error extracting judgment from response '{response_text}': {str(e)}")
        return None

# 5. Ignore stylistic qualities, length, or justification; focus solely on factual equivalence.


def evaluate_answer(answer, ground_truth, ref_chunks_text, question):
    """Evaluate if the answer matches the reference answer using LLM as a binary judge"""
    # Soft cap the reference context to avoid overly long prompts
    context_text = ref_chunks_text if isinstance(ref_chunks_text, str) else "\n".join(ref_chunks_text or [])
    if len(context_text) > 12000:
        context_text = context_text[:12000]

    system_msg = (
        "You are an expert QA evaluator. Given a question, reference context (document chunks), a reference answer, "
        "and a generated answer, judge whether the generated answer expresses the same core meaning as the reference "
        "answer with respect to the question. Use the reference context only to clarify facts and scope; focus on "
        "semantic equivalence, not wording. Respond with exactly one token: [[Yes]] or [[No]]. No explanations."
    )

    user_msg = f"""
Question:
{question}

Reference Context:
<REF_CONTEXT>
{context_text}
</REF_CONTEXT>

Reference Answer:
{ground_truth}

Generated Answer:
{answer}

Judgment rules:
1) Same facts and intent relative to the question → [[Yes]].
2) Different facts, changed scope, or missing key information → [[No]].
3) Partial overlap is not enough; omissions count as different.
4) Grammar/wording/order do not matter—only meaning.

Your direct response (choose one): [[Yes]] / [[No]]
"""
    
    # Engine URLs - using the same configuration as answer_judge.py
    port = 42692
    # ip_list = [
    #     "192.168.8.170",
    #     "192.168.12.183",
    #     "192.168.8.163",
    #     "192.168.12.180",
    #     "192.168.8.161",
    #     "192.168.12.177",
    #     "192.168.12.153",
    #     "192.168.12.26"
    # ]
    # if len(answer) > len(ground_truth) * 10:
    #     return 0.0
    # engine_urls = [f"http://{ip}:{port}" for ip in ip_list]
    
    try:
        # Randomly select an engine to distribute load
        # selected_url = random.choice(engine_urls)
        selected_url = "https://sd2heuo0ufseetaof8vs0.apigateway-cn-shanghai.volceapi.com"
        client = openai.Client(api_key="EMPTY", base_url=f"{selected_url}/v1")
        
        # Make API call
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=512,
            temperature=0.0,
            # extra_body={
            #     "top_k": 20,
            #     "chat_template_kwargs": {"enable_thinking": False},
            # }
        )
        
        # Extract binary judgment from response
        response_text = response.choices[0].message.content.strip()
        # print(f"response_text: {response_text}")
        judgment = extract_binary_judgment_from_response(response_text)
        
        if judgment is not None:
            return judgment
        else:
            print(f"Failed to extract judgment from response: {response_text}")
            return 0.0
            
    except Exception as e:
        print(f"Error evaluating answer: {str(e)}")
        return 0.0


from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score


class LongRLRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = self.custom_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"


    def custom_compute_score(self, data_source, solution_str, ground_truth, extra_info=None):
        # return 0.5
        used_context = extract_used_context(solution_str)
        answer = extract_answer(solution_str)
        # with open("solution_str.txt", "w") as f:
        #     f.write(f"solution_str: {solution_str}\nused_context: {used_context}\nanswer: {answer}")
        # format_reward = 
        if (used_context is None) or (answer is None):
            return {
                "score": 0.0,
                "recall": 0.0,
                # "f1_score": 0.0,
                "precision": 0.0,
                "fbeta_score": 0.0,
                # "penalty": 0.0,
                "answer_acc": 0.0,
            }
        # help me evaluate the answer compare with the ground truth using LLM as a judge

        ref_chunks = extra_info['ref_chunks']
        used_chunks = convert_to_chunks(used_context)
        # calculate the IOU of used_chunks and ref_chunks
        if used_chunks:
            intersection = set(used_chunks) & set(ref_chunks)
            precision = len(intersection) / len(used_chunks)
            recall = len(intersection) / len(set(ref_chunks))
            if precision + recall == 0:
                fbeta_score = 0.0
            else:
                fbeta_score = (5 * precision * recall) / (4 * precision + recall)
            # penalty = min(1, max(0, (len(used_chunks) - len(ref_chunks))) / (len(ref_chunks) * 2))
            # penalty
            # if precision + recall == 0:   
            #     f1_score = 0.0
            # else:
            #     f1_score = 2 * (precision * recall) / (precision + recall)
            # iou = len(set(used_chunks) & set(ref_chunks)) / len(set(used_chunks) | set(ref_chunks))
        else:
            recall = 0
            precision = 0
            fbeta_score = 0.0
            penalty = 0

        ref_chunks_text = extra_info['ref_chunks_text']
        ref_chunks_text = "\n".join(ref_chunks_text)
        question = extra_info['question']
        # evaluate the answer
        answer_judgment = evaluate_answer(answer, ground_truth, ref_chunks_text, question)
        # answer_judgment = 0
        return {
            "score": answer_judgment + 0.1 * fbeta_score + 0.9 * answer_judgment * fbeta_score,
            "recall": recall,
            "precision": precision,
            "fbeta_score": fbeta_score,
            "answer_acc": answer_judgment,
        }

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            result = self.custom_compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score


            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
