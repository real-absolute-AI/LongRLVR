"""
Now I wanna evaluate the answer accuracy of the qa_pairs in the clusters through LLM-as-a-judge.

The input is the generated answer and the ground truth answer with context and questions in each cluster

The output is the rating of the generated answer evaluted by LLM.

The rating is a number between 0 and 10, 10 means the generated answer is perfect and 0 means the generated answer is totally wrong.

The data sample format is:
[
{
    "id": "chunk cluster id",
    "text": "document text",
    "clusters": [
        {
            "chunk_data": {
                "chunks": [
                    {
                        "text": "chunk text 1"
                    },
                    {
                        "text": "chunk text 2"
                    }
                ],
                "qa_pairs": [
                    {
                        "question": "question text 1",
                        "answer": {
                            "text": "answer text 1"
                        }
                        "generated_answers": [
                            {
                                "text": "generated answer text 1",
                                "rating": 8.5
                            },
                            {
                                "text": "generated answer text 2",
                                "rating": 7.2
                            }
                        ]
                    },
                    {
                        "question": "question text 2",
                        "answer": {
                            "text": "answer text 2"
                        }
                    }
                ]
            },
        },
        {
            "chunk_data": {
                "chunks": [
                    {
                        "text": "chunk text 3"
                    }
                ],
                "qa_pairs": [
                    {
                        "question": "question text 3",
                        "answer": {
                            "text": "answer text 3"
                        }
                        "generated_answers": [
                            {
                                "text": "generated answer text 3",
                                "rating": 9.1
                            },
                            {
                                "text": "generated answer text 4",
                                "rating": 6.8
                            }
                        ]
                    }
                ]
            }
        }
    ]
}
]
"""

import json
from typing import List, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from dataclasses import dataclass
from threading import Lock
import random
import argparse
import openai
import os
from datetime import datetime
import threading
from queue import Queue
import re
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)


@dataclass
class VLLMEngine:
    """Class to represent a VLLM engine with its status and metrics."""
    url: str
    client: openai
    last_used: float = 0.0
    request_count: int = 0
    lock: Lock = None

    def __post_init__(self):
        if self.lock is None:
            self.lock = Lock()

@dataclass
class EvaluationJob:
    """Represents an evaluation job for processing"""
    doc_idx: int
    cluster_idx: int
    qa_idx: int
    generated_answer_idx: int
    document: Dict[str, Any]
    batch_file: str

def create_evaluation_prompt(
    chunk_texts: str, 
    question: str, 
    ground_truth_answer: str, 
    generated_answer: str
) -> str:
    """
    Create a prompt for evaluating generated answers against ground truth.
    
    Args:
        chunk_texts: The concatenated chunk texts from the relevant cluster
        question: The question being answered
        ground_truth_answer: The ground truth answer
        generated_answer: The generated answer to evaluate
    
    Returns:
        Formatted evaluation prompt string
    """
    prompt = f"""You are an expert evaluator tasked with rating the quality and accuracy of generated answers against reference (ground truth) answers. Your evaluation should be based on factual correctness, completeness, relevance, and clarity.

**Evaluation Criteria:**
- **Factual Accuracy (40%)**: How factually correct is the generated answer compared to the ground truth?
- **Completeness (25%)**: Does the generated answer cover the key points from the ground truth?
- **Relevance (20%)**: How well does the generated answer address the specific question asked?
- **Clarity and Coherence (15%)**: Is the generated answer well-structured and easy to understand?

**Rating Scale:**
- 10: Perfect match - factually accurate, complete, highly relevant, and very clear
- 8-9: Excellent - mostly accurate with minor gaps or slight clarity issues
- 6-7: Good - generally accurate but missing some important details or has minor factual errors
- 4-5: Fair - partially correct but significant gaps, unclear, or some factual errors
- 2-3: Poor - major factual errors, incomplete, or largely irrelevant
- 0-1: Very poor - mostly incorrect, irrelevant, or completely wrong

**Context (Relevant Document Chunks):**
{chunk_texts}

**Question:** {question}

**Ground Truth Answer:** {ground_truth_answer}

**Generated Answer to Evaluate:** {generated_answer}

**Instructions:**
1. Carefully compare the generated answer with the ground truth answer
2. Consider the context provided in the document chunks
3. Evaluate based on the criteria above
4. Provide a numerical rating between 0 and 10 (use decimals if needed, e.g., 7.5)
5. Your response must contain ONLY the rating in the format [[rating]], for example [[7.5]] or [[9.0]]

Rating:"""
    
    return prompt

def extract_rating_from_response(response_text: str) -> float:
    """
    Extract numerical rating from LLM response.
    
    Args:
        response_text: The response text from the LLM
    
    Returns:
        Extracted rating as float, or None if parsing fails
    """
    try:
        # Clean the response text
        cleaned_text = response_text.strip()
        
        # Look for rating in double brackets format [[rating]]
        rating_pattern = r'\[\[(\d+(?:\.\d+)?)\]\]'
        match = re.search(rating_pattern, cleaned_text)
        
        if match:
            # Extract the rating from the match
            rating = float(match.group(1))
            # Clamp to valid range [0, 10]
            return max(0.0, min(10.0, rating))
        else:
            # Fallback to looking for any decimal number if double bracket format not found
            fallback_pattern = r'\b(\d+(?:\.\d+)?)\b'
            matches = re.findall(fallback_pattern, cleaned_text)
            
            if matches:
                # Take the first number found
                rating = float(matches[0])
                # Clamp to valid range [0, 10]
                return max(0.0, min(10.0, rating))
            else:
                print(f"Warning: Could not extract rating from response: {cleaned_text}")
                return None
            
    except Exception as e:
        print(f"Error extracting rating from response '{response_text}': {str(e)}")
        return None

def prepare_evaluation_batch_requests(
    documents: List[Dict[str, Any]],
    temperature: float = 0.6,
    max_tokens: int = 50
) -> List[Dict[str, Any]]:
    """
    Prepare batch requests for evaluation.
    
    Args:
        documents: List of documents with generated answers
        temperature: Low temperature for consistent evaluation
        max_tokens: Maximum tokens for rating response
    
    Returns:
        List of batch request dictionaries
    """
    requests = []
    
    for doc_idx, document in enumerate(documents):
        for cluster_idx, cluster in enumerate(document['clusters']):
            # Concatenate chunk texts for this cluster
            chunk_texts = "\n\n".join([chunk['text'] for chunk in cluster['chunk_data']['chunks']])
            
            for qa_idx, qa_pair in enumerate(cluster['chunk_data']['qa_pairs']):
                if not qa_pair.get('question') or not qa_pair.get('answer'):
                    continue
                
                # Get ground truth answer
                ground_truth = qa_pair['answer']['text']
                question = qa_pair['question']
                
                # Process each generated answer
                for gen_idx, generated_answer in enumerate(qa_pair.get('generated_answers', [])):
                    if not generated_answer.get('text'):
                        continue
                    
                    prompt = create_evaluation_prompt(
                        chunk_texts,
                        question,
                        ground_truth,
                        generated_answer['text']
                    )
                    
                    request = {
                        "custom_id": f"eval_doc_{doc_idx}_cluster_{cluster_idx}_qa_{qa_idx}_gen_{gen_idx}",
                        "method": "POST",
                        "url": "/chat/completions",
                        "body": {
                            "model": "default",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            # "top_p": 0.95,
                            "extra_body": {
                                "top_k": 20,
                                "chat_template_kwargs": {"enable_thinking": True},
                            }
                        }
                    }
                    requests.append(request)
    
    return requests

def process_evaluation_results(
    results: List[Dict[str, Any]],
    documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Process evaluation results and update documents with ratings.
    Drop generated answers where rating extraction failed.
    
    Args:
        results: List of evaluation results
        documents: List of documents to update
    
    Returns:
        Updated list of documents
    """
    for result in results:
        if not result.get('response'):
            continue
            
        # Parse custom_id to get indices
        custom_id = result['custom_id']
        parts = custom_id.split('_')
        doc_idx = int(parts[2])
        cluster_idx = int(parts[4])
        qa_idx = int(parts[6])
        gen_idx = int(parts[8])
        
        # Get the rating from the response
        assert result['response']['body']['choices']['message']['role'] == "assistant"
        response_text = result['response']['body']['choices']['message']['content'].strip()
        rating = extract_rating_from_response(response_text)
        
        # Skip if rating extraction failed
        if rating is None:
            print(f"Failed to extract rating for doc {doc_idx}, cluster {cluster_idx}, qa {qa_idx}, gen {gen_idx}")
            continue
            
        print(f"Rated generated answer {gen_idx} for doc {doc_idx}, cluster {cluster_idx}, qa {qa_idx}: {rating}")
        
        # Update the document with the rating
        doc = documents[doc_idx]
        qa_pair = doc['clusters'][cluster_idx]['chunk_data']['qa_pairs'][qa_idx]
        
        # Add rating to the generated answer
        if 'generated_answers' in qa_pair and gen_idx < len(qa_pair['generated_answers']):
            qa_pair['generated_answers'][gen_idx]['rating'] = rating
        
        # Ensure the document is updated in the documents list
        documents[doc_idx] = doc
    
    # Filter out generated answers without ratings
    for doc in documents:
        for cluster in doc['clusters']:
            for qa_pair in cluster['chunk_data']['qa_pairs']:
                if 'generated_answers' in qa_pair:
                    # Keep only answers that have ratings
                    qa_pair['generated_answers'] = [
                        answer for answer in qa_pair['generated_answers']
                        if 'rating' in answer
                    ]
    
    return documents

def process_evaluation_batch_on_engine(
    engine_client: openai.Client,
    requests: List[Dict[str, Any]],
    engine_url: str
) -> List[Dict[str, Any]]:
    """
    Process evaluation batch on a specific engine.
    
    Args:
        engine_client: OpenAI client for the engine
        requests: List of batch requests
        engine_url: URL of the engine for logging
    
    Returns:
        List of results
    """
    try:
        # Create temporary batch file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        batch_file = f"batch_data/eval_batch_{timestamp}.jsonl"
        
        # Write requests to batch file
        with open(batch_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        
        print(f"Engine {engine_url}: Processing {len(requests)} evaluation requests")
        
        # Upload batch file
        with open(batch_file, "rb") as f:
            file_response = engine_client.files.create(file=f, purpose="batch")
        
        # Create batch job
        batch_response = engine_client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        
        print(f"Engine {engine_url}: Created batch job {batch_response.id}")
        
        # Wait for batch completion with polling
        while batch_response.status not in ["completed", "failed", "cancelled"]:
            time.sleep(5)
            batch_response = engine_client.batches.retrieve(batch_response.id)
            print(f"Engine {engine_url}: Batch job status: {batch_response.status}")
        
        if batch_response.status == "completed":
            # Get results
            result_file_id = batch_response.output_file_id
            file_content = engine_client.files.content(result_file_id)
            result_content = file_content.read().decode("utf-8")
            
            results = [
                json.loads(line) for line in result_content.split("\n") if line.strip()
            ]
            
            # Clean up files
            engine_client.files.delete(result_file_id)
            engine_client.files.delete(file_response.id)
            
            print(f"Engine {engine_url}: Completed batch evaluation with {len(results)} results")
            return results
        else:
            print(f"Engine {engine_url}: Batch job failed with status: {batch_response.status}")
            if hasattr(batch_response, "errors"):
                print(f"Errors: {batch_response.errors}")
            return []
            
    except Exception as e:
        print(f"Error processing evaluation batch on engine {engine_url}: {str(e)}")
        return []
    finally:
        # Clean up batch file
        try:
            if os.path.exists(batch_file):
                os.remove(batch_file)
        except:
            pass

def evaluate_answers_batch(
    documents: List[Dict[str, Any]],
    engine_urls: List[str],
    temperature: float = 0.1,
    max_tokens: int = 50
) -> List[Dict[str, Any]]:
    """
    Evaluate generated answers using batch processing with parallel engine utilization.
    
    Args:
        documents: List of documents with generated answers
        engine_urls: List of VLLM engine URLs
        temperature: Low temperature for consistent evaluation
        max_tokens: Maximum tokens for rating response
    
    Returns:
        List of documents with ratings added
    """
    # Create a deep copy of documents to avoid modifying the original
    documents = [json.loads(json.dumps(doc)) for doc in documents]
    
    # Ensure batch_data directory exists
    os.makedirs('batch_data', exist_ok=True)
    
    # Prepare all evaluation requests
    print("Preparing evaluation requests...")
    all_requests = prepare_evaluation_batch_requests(documents, temperature, max_tokens)
    
    if not all_requests:
        print("No evaluation requests to process")
        return documents
    
    print(f"Total evaluation requests: {len(all_requests)}")
    
    # Create engine clients
    engines = [
        openai.Client(api_key="EMPTY", base_url=f"{url}/v1")
        for url in engine_urls
    ]
    
    # Split requests among engines
    num_engines = len(engines)
    requests_per_engine = len(all_requests) // num_engines
    remainder = len(all_requests) % num_engines
    
    engine_requests = []
    start_idx = 0
    for i in range(num_engines):
        end_idx = start_idx + requests_per_engine + (1 if i < remainder else 0)
        engine_requests.append(all_requests[start_idx:end_idx])
        start_idx = end_idx
    
    print(f"Distributing requests: {[len(reqs) for reqs in engine_requests]}")
    
    # Process evaluations in parallel across engines
    all_results = []
    with ThreadPoolExecutor(max_workers=num_engines) as executor:
        future_to_engine = {
            executor.submit(
                process_evaluation_batch_on_engine,
                engines[i],
                engine_requests[i],
                engine_urls[i]
            ): i
            for i in range(num_engines)
            if engine_requests[i]  # Only submit if there are requests
        }
        
        for future in as_completed(future_to_engine):
            engine_idx = future_to_engine[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"Engine {engine_idx} completed with {len(results)} results")
            except Exception as e:
                print(f"Engine {engine_idx} failed: {str(e)}")
    
    print(f"Total evaluation results: {len(all_results)}")
    
    # Process results and update documents
    if all_results:
        documents = process_evaluation_results(all_results, documents)
    
    return documents

def print_evaluation_summary(documents: List[Dict[str, Any]]):
    """Print summary statistics of the evaluation results."""
    total_generated_answers = 0
    total_rated_answers = 0
    ratings = []
    
    for doc in documents:
        for cluster in doc['clusters']:
            for qa_pair in cluster['chunk_data']['qa_pairs']:
                for gen_answer in qa_pair.get('generated_answers', []):
                    total_generated_answers += 1
                    if 'rating' in gen_answer:
                        total_rated_answers += 1
                        ratings.append(gen_answer['rating'])
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Total generated answers: {total_generated_answers}")
    print(f"Successfully rated answers: {total_rated_answers}")
    print(f"Rating coverage: {total_rated_answers/total_generated_answers*100:.1f}%" if total_generated_answers > 0 else "0%")
    
    if ratings:
        print(f"Average rating: {sum(ratings)/len(ratings):.2f}")
        print(f"Min rating: {min(ratings):.2f}")
        print(f"Max rating: {max(ratings):.2f}")
        
        # Rating distribution
        bins = [0, 2, 4, 6, 8, 10]
        bin_counts = [0] * (len(bins) - 1)
        for rating in ratings:
            for i in range(len(bins) - 1):
                if bins[i] <= rating < bins[i + 1] or (i == len(bins) - 2 and rating == bins[i + 1]):
                    bin_counts[i] += 1
                    break
        
        print("Rating distribution:")
        for i in range(len(bin_counts)):
            print(f"  {bins[i]:.0f}-{bins[i+1]:.0f}: {bin_counts[i]} ({bin_counts[i]/len(ratings)*100:.1f}%)")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, default="documents_with_generated_answers.json", 
                        help="Path to input data with generated answers")
    parser.add_argument("--output_data_path", type=str, default="documents_with_ratings.json",
                        help="Path to save data with ratings")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Low temperature for consistent evaluation")
    parser.add_argument("--max_tokens", type=int, default=16384, 
                        help="Maximum tokens for rating response")
    parser.add_argument("--max_documents", type=int, default=10000, 
                        help="Maximum number of documents to process")
    
    args = parser.parse_args()
    
    # Load documents with generated answers
    print(f"Loading documents from {args.input_data_path}")
    all_documents = []
    i = 0
    
    with open(args.input_data_path, 'r') as f:
        for line in f:
            if i % 1000 == 0:
                print(f"Loading documents: {i}")
            # if i >= 2:
            #     break
            try:
                item = json.loads(line.strip())
                all_documents.append(item)
                i += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {i}: {e}")
                continue
    
    print(f"Loaded {len(all_documents)} documents")
    
    # Count total generated answers to evaluate
    total_gen_answers = 0
    for doc in all_documents:
        for cluster in doc['clusters']:
            for qa_pair in cluster['chunk_data']['qa_pairs']:
                total_gen_answers += len(qa_pair.get('generated_answers', []))
    
    print(f"Total generated answers to evaluate: {total_gen_answers}")
    
    # VLLM engine URLs (same as generation script)
    port = 42692
    ip_list = [
        "192.168.9.254",
        "192.168.9.253",
        "192.168.9.252",
        "192.168.9.250",
        "192.168.9.249",
        "192.168.9.248",
        "192.168.9.247",
        "192.168.9.246",
        "192.168.9.244",
        "192.168.9.243",
    ]
    
    engine_urls = [f"http://{ip}:{port}" for ip in ip_list]
    
    # Evaluate answers using batch processing
    print(f"\nEvaluating answers using LLM-as-a-judge with {len(engine_urls)} engines")
    documents_with_ratings = evaluate_answers_batch(
        all_documents,
        engine_urls,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Print evaluation summary
    print_evaluation_summary(documents_with_ratings)
    
    # Save results
    print(f"\nSaving results to {args.output_data_path}")
    with open(args.output_data_path, 'w') as f:
        for doc in documents_with_ratings:
            f.write(json.dumps(doc) + "\n")
    
    print("Answer evaluation completed!")