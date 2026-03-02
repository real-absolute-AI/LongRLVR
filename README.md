# LongRLVR

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/arXiv-TBD-b31b1b?logo=arXiv" alt="arXiv">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/HuggingFace-Models-yellow?logo=huggingface" alt="HF Models">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface" alt="Dataset">
  </a>
</p>

This repo provides the official code for our ICLR 2026 paper:
**[LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](paper/longrlvr.pdf)**.

---

## 💡 Why LongRLVR?

Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Large Language Models (LLMs) by optimizing them against factual outcomes. However, this paradigm **falters in long-context scenarios**, as its reliance on *internal* parametric knowledge is ill-suited for tasks requiring *contextual grounding*—the ability to find and reason over *externally* provided information. 

We formally prove that the outcome-only reward leads to significant vanishing gradients for the context grounding process. **LongRLVR** addresses this by augmenting the sparse answer reward with a **dense, verifiable context reward** that directly incentivizes correct evidence selection, providing a robust learning gradient that solves the underlying optimization challenge.

### Key takeaways
<p align="center">
  <img width="80%" alt="image" src="https://github.com/user-attachments/assets/d2d60b30-4755-481f-ad50-3f33d39917bf" />
</p>

- **Diagnosis:** Outcome-only RLVR leads to a vanishing learning signal for contextual grounding in long sequences.
- **Method:** Add a *verifiable context reward* (based on grounding chunk identifiers) alongside answer correctness.
- **Result:** Consistently and significantly improve long-context performance across Qwen and LLaMA models.

### Results
LongRLVR consistently and significantly outperforms the standard RLVR across all models and benchmarks.
<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/d0b72eb6-a3a7-438d-82b6-5f8c6b14af0b" />
</p>

---

## 🚀 Released Artifacts

### Models
We release LongRLVR-trained models based on LLaMA and Qwen:

| Model | Hugging Face Repo |
|---|---|
| LLaMA-3.1-8B-LongRLVR | https://huggingface.co/Guanzheng/LLaMA-3.1-8B-LongRLVR |
| Qwen2.5-7B-LongRLVR | https://huggingface.co/Guanzheng/Qwen2.5-7B-LongRLVR |
| Qwen2.5-14B-LongRLVR | https://huggingface.co/Guanzheng/Qwen2.5-14B-LongRLVR |

### Dataset
Our training dataset containing 46K high-quality synthetic QA pairs is available on Hugging Face:

| Dataset | Hugging Face Repo |
|---|---|
| LongRLVR-Data | https://huggingface.co/datasets/Guanzheng/LongRLVR-Data |

---

## 📁 Repository Structure

- `data_gen/`: The synthetic data generation pipeline to construct the explicit grounding dataset (chunking → clustering → QA generation → judging → final selection). See the [Data Generation README](data_gen/README.md) for details.
- `recipe/dapo/`: Training entrypoint and LongRLVR reward manager. 
  - `longrl_reward_manager.py`: Implements the asynchronous reward computation (F-score context reward and synergistic answer reward) for LongRLVR.
  - `main_dapo.py`: Main training loop utilizing GRPO.
- `verl/`: RL training framework utilities (rollout, PPO/GRPO, distributed training, etc.), customized for long-context generation.

---

## 🛠️ Quickstart 

### Using the Models (Hugging Face)

The LongRLVR models are explicitly trained to identify useful context chunks before answering. You can use standard Hugging Face `transformers` to interact with them. For the best results, you must chunk your document using the same strategy as our data pipeline (see `split_into_chunks` in [`data_gen/clustering.py`](data_gen/clustering.py)) and prompt the model to output the `<think>`, `<useful chunks>`, and `<answer>` tags. Note that `<think>` has been already added into the chat template of the released models.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Guanzheng/Qwen2.5-7B-LongRLVR"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Example document that has been split into chunks
document_chunks = [
    "<Chunk_0> Marie Curie was born in Warsaw, Poland...</Chunk_0>",
    "<Chunk_1> The Curies' early research was inspired by Henri Becquerel's 1896 discovery...</Chunk_1>",
    # ... more chunks ...
    "<Chunk_5> In December 1898, they announced the discovery of a second element, 'radium'...</Chunk_0\5>"
]
context = "\n".join(document_chunks)

prompt = f"""{context}

Question: Where was Marie Curie born and what was the second radioactive element she co-discovered?

Output:
"""

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.6,
    top_p=0.9
)

# The model will output something like:
# <think> Let me find where she was born and the second element... </think>
# <useful chunks> <CHUNK 0>, <CHUNK 5> </useful chunks>
# <answer> Marie Curie was born in Warsaw, Poland, and the second radioactive element she co-discovered was radium. </answer>

response = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
print(response)
```

### Data Generation Pipeline
We provide a comprehensive pipeline to synthesize the verifiable context-grounded dataset. 
For a step-by-step tutorial on generating your own data, please see the [data_gen/README.md](data_gen/README.md).

### Training
We provide example scripts for training. Training is built on `verl` and uses `Ray` for distributed execution and `vLLM` / `sglang` for fast generation.

1. **Setup your environment:** Ensure you have the necessary dependencies installed (e.g., PyTorch, Ray, vLLM/sglang, and other dependencies required by `verl`). We use official docker image `verlai/verl:app-verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.12.2-te2.2`. Feel free to use latest `verl` to access new features.
2. **Run training:**
You can use `run_longrl.sh` as a template. Set the required environment variables:

```bash
export TRAIN_FILES="['/path/to/train.parquet']"
export VAL_FILES="['/path/to/val.parquet']"
export MODEL_PATH="/path/to/base-model"
export CKPT_SAVE_PATH="./ckpts/longrlvr_run"

bash run_longrl.sh
```



---

## 📜 Citation

If you find LongRLVR useful for your research, please cite our paper:

```bibtex
@inproceedings{
  chen2026longrlvr,
  title={Long{RLVR}: Long-Context Reinforcement Learning Requires Verifiable Context Rewards},
  author={Guanzheng Chen and Michael Qizhe Shieh and Lidong Bing},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=omVhYvyTPJ}
}
```

## 🤝 Acknowledgements
This project utilizes the [verl](https://github.com/volcengine/verl) framework for scalable RL training.
