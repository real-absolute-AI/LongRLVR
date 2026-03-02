# LongRLVR Data Generation Pipeline

This directory contains the automated data generation pipeline for **LongRLVR** (Long-Context Reinforcement Learning with Verifiable Rewards), as described in our paper. 

The pipeline produces high-quality, long-context question-answering data annotated with the necessary grounding chunks (evidence). This dataset is then used to explicitly reward language models for selecting the correct contextual evidence, enabling stronger long-context reasoning capabilities.

## Pipeline Overview (Algorithm 1)

The pipeline implements the automated multi-stage generation outlined in the paper. It takes raw long documents and outputs a curated list of challenging, evidence-grounded QA pairs.

### Step 1: Semantic Clustering and Evidence Identification
**Script:** `clustering.py`
- Partitions long documents into text chunks.
- Embeds all chunks using a dense sentence encoder (e.g., BGE-M3).
- Applies HDBSCAN to form thematic clusters of related chunks. This isolates related contexts to construct complex multi-hop questions.

### Step 2: Per-Cluster QA Generation
**Script:** `generate_qa_batch.py`
- Prompts an LLM (using the OpenAI Batch API or vLLM) to generate candidate question-answer pairs for each cluster.
- The prompt enforces generating ONE complex, multi-chunk question that avoids explicit references to chunk numbers.
- The model itself outputs the necessary evidence chunk IDs alongside the QA pair.

### Step 3: QA Scoring / Verification
**Script:** `qa_pairs_judge.py`
- An LLM acts as an expert evaluator (LLM-as-a-judge).
- Each generated QA pair is scored on a scale of 0 to 10 based on:
  - Question Relevance & Challenge
  - Answer Accuracy & Completeness
  - Clarity, Usefulness, & Reference Precision
  - Context Utilization

## Requirements

You will need the following libraries:
- `transformers`
- `sentence-transformers`
- `langchain`
- `scikit-learn` (for HDBSCAN)
- `openai`
- `tqdm`

## Usage Examples

**1. Run Clustering**
```bash
bash scripts/clustering.sh path/to/input.jsonl path/to/clustered.jsonl
```
*(Uses MPI-style multiple processes to chunk and cluster the datasets concurrently)*

**2. Generate QA Pairs**
```bash
bash scripts/generate_qa.sh path/to/clustered.jsonl
```
*(This writes intermediate questions incrementally into `generated_questions.jsonl` and structured JSON format into `organized_questions.json`)*

**3. Evaluate / Judge QA Pairs**
```bash
python qa_pairs_judge.py \
    --input_data_path organized_questions.json \
    --output_data_path rated_questions.json \
    --temperature 0.1
```



This output serves as the explicit verifiable reward context ($Z$) needed during LongRLVR GRPO/PPO training.