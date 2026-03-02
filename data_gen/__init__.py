"""
Data Generation Pipeline for LongRLVR

This pipeline follows Algorithm 1 from the paper "LONGRLVR: LONG-CONTEXT REINFORCEMENT LEARNING REQUIRES VERIFIABLE CONTEXT REWARDS".

Pipeline Steps:
1. Semantic Clustering and Evidence Identification (clustering.py)
2. Per-Cluster QA Generation (generate_qa_batch.py)
3. QA Scoring / Evaluation (qa_pairs_judge.py)
4. Answer Generation (generate_answer_batch.py)
5. Answer Evaluation (answer_judge.py)
"""
