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
- **Diagnosis:** Outcome-only RLVR leads to a vanishing learning signal for contextual grounding in long sequences.
- **Method:** Add a *verifiable context reward* (based on grounding chunk identifiers) alongside answer correctness.
- **Result:** Consistently and significantly improve long-context performance across Qwen and LLaMA models.

### Results
LongRLVR consistently and significantly outperforms the standard RLVR across all models and benchmarks. Example gains for **Qwen2.5-14B**:
- **73.17 → 88.90** (RULER-QA)
- **39.8 → 46.5** (LongBench v2)

---

## 🚀 Released Artifacts

### Models
We release LongRLVR-trained models based on LLaMA and Qwen:

| Model | Hugging Face Repo |
|---|---|
| LLaMA-3.1-8B-LongRLVR | `https://huggingface.co/Guanzheng/LLaMA-3.1-8B-LongRLVR` |
| Qwen2.5-7B-LongRLVR | `https://huggingface.co/Guanzheng/Qwen2.5-7B-LongRLVR` |
| Qwen2.5-14B-LongRLVR | `https://huggingface.co/Guanzheng/Qwen2.5-14B-LongRLVR` |

### Dataset
Our training dataset containing 46K high-quality synthetic QA pairs is available on Hugging Face:

| Dataset | Hugging Face Repo |
|---|---|
| LongRLVR-Data | `https://huggingface.co/datasets/Guanzheng/LongRLVR-Data` |

---

## 📁 Repository Structure

- `data_gen/`: The synthetic data generation pipeline to construct the explicit grounding dataset (chunking → clustering → QA generation → judging → final selection). See the [Data Generation README](data_gen/README.md) for details.
- `recipe/dapo/`: Training entrypoint and LongRLVR reward manager. 
  - `longrl_reward_manager.py`: Implements the asynchronous reward computation (F-score context reward and synergistic answer reward) for LongRLVR.
  - `main_dapo.py`: Main training loop utilizing GRPO.
- `verl/`: RL training framework utilities (rollout, PPO/GRPO, distributed training, etc.), customized for long-context generation.

---

## 🛠️ Quickstart 

### Data Generation Pipeline
We provide a comprehensive pipeline to synthesize the verifiable context-grounded dataset. 
For a step-by-step tutorial on generating your own data, please see the [data_gen/README.md](data_gen/README.md).

### Training
We provide example scripts for training. Training is built on `verl` and uses `Ray` for distributed execution and `vLLM` / `sglang` for fast generation.

1. **Setup your environment:** Ensure you have the necessary dependencies installed (e.g., PyTorch, Ray, vLLM/sglang, and other dependencies required by `verl`).
2. **Run training:**
You can use `run_longrl.sh` as a template. Set the required environment variables:

```bash
export TRAIN_FILES="['/path/to/train.parquet']"
export VAL_FILES="['/path/to/val.parquet']"
export MODEL_PATH="/path/to/base-model"
export CKPT_SAVE_PATH="./ckpts/longrlvr_run"

bash run_longrl.sh
```

### Important Hyperparameters
Inside `run_longrl.sh`, you can configure sequence lengths and other parameters:
- `data.max_prompt_length=65536`: Maximum prompt length for long-context tasks.
- `actor_rollout_ref.rollout.name=sglang`: Generation backend used during rollout.
- `algorithm.adv_estimator=grpo`: RL Algorithm choice.

---

## 📜 Citation

If you find LongRLVR useful for your research, please cite our paper:

```bibtex
@inproceedings{chen2026longrlvr,
  title     = {LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards},
  author    = {Chen, Guanzheng and Shieh, Michael Qizhe and Bing, Lidong},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {TBD}
}
```

## 🤝 Acknowledgements
This project utilizes the [verl](https://github.com/volcengine/verl) framework for scalable RLHF training.