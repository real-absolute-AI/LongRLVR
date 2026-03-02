"""
Now I wanna evaluate the qa pair quality in the clusters through LLM-as-a-judge.

The input is the generated questions and answers in each cluster based on chunks.

The output is the rating of the quality of questions and answers evaluated by LLM.

The rating is a number between 0 and 10, 10 means the qa pair is perfect and 0 means the qa pair is totally wrong.

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
                    }
                ]
            }
        }
    ]
}
]
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

import openai
from tqdm import tqdm

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

def create_qa_evaluation_prompt(
    chunk_texts: str, 
    question: str, 
    answer: str
) -> str:
    """
    Create a prompt for evaluating Q&A pair quality based on chunks.
    
    Args:
        chunk_texts: The concatenated chunk texts from the relevant cluster
        question: The question to evaluate
        answer: The answer to evaluate
    
    Returns:
        Formatted evaluation prompt string
    """
    prompt = f"""You are an expert evaluator tasked with rating the quality of question-answer pairs based on their source document chunks. Your evaluation should assess how well the Q&A pair utilizes the provided context and serves as a useful knowledge extraction.

**Evaluation Criteria:**
- **Question Relevance & Challenge (25%)**: Does the question target important information and require non-trivial reasoning or synthesis across multiple, potentially distant chunks in the long document?
- **Answer Accuracy (30%)**: How factually correct, well-supported, and logically sound is the answer based on the chunks?
- **Answer Completeness (20%)**: Does the answer fully address all aspects of the question, drawing on all relevant information available?
- **Clarity, Usefulness & Reference Precision (15%)**: Are the question and answer clear, well-structured, useful for learning, and free from ambiguous references (e.g., "the man", "the guy") that could be confusing in a long document?
- **Context Utilization (10%)**: How effectively does the Q&A pair extract and present information from the chunks?

**Rating Scale:**
- 10: Excellent – Highly relevant, challenging question with a completely accurate and comprehensive answer; perfect context utilization and no ambiguous references
- 8-9: Very Good – Strong Q&A pair with minor gaps or slight improvements possible
- 6-7: Good – Generally solid but missing some important details, clarity, or depth of reasoning
- 4-5: Fair – Adequate but significant room for improvement in relevance, accuracy, completeness, or clarity
- 2-3: Poor – Major issues with question relevance, answer accuracy, ambiguity, or context utilization
- 0-1: Very Poor – Irrelevant question, incorrect answer, or complete disconnect from context

**Source Context (Document Chunks):**
{chunk_texts}

**Question:** {question}

**Answer:** {answer}

**Instructions:**
1. Carefully read the source chunks to understand the available context.
2. Evaluate how well the question targets important, non-trivial information from the chunks and whether answering it requires synthesizing information spread across the document.
3. Assess whether the answer is accurate, complete, and well-supported by the context.
4. Penalize vague or ambiguous references (e.g., pronouns like "he", "she", "the man") that rely on unstated context when the document is long.
5. Consider the overall usefulness of this Q&A pair for knowledge extraction.
6. Provide a numerical rating between 0 and 10 (decimals allowed, e.g., 7.5).
7. Your response must contain ONLY the rating in the format [[rating]], for example [[7.5]] or [[9.0]].

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

def prepare_qa_evaluation_batch_requests(
    documents: List[Dict[str, Any]],
    temperature: float = 0.6,
    max_tokens: int = 50
) -> List[Dict[str, Any]]:
    """
    Prepare batch requests for Q&A pair evaluation.
    
    Args:
        documents: List of documents with Q&A pairs to evaluate
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
                if not qa_pair or not qa_pair.get('question') or not qa_pair.get('answer'):
                    continue
                
                question = qa_pair['question']
                answer = qa_pair['answer']['text']
                
                prompt = create_qa_evaluation_prompt(
                    chunk_texts,
                    question,
                    answer
                )
                
                request = {
                    "custom_id": f"qa_eval_doc_{doc_idx}_cluster_{cluster_idx}_qa_{qa_idx}",
                    "method": "POST",
                    "url": "/chat/completions",
                    "body": {
                        "model": "default",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "extra_body": {
                            "top_k": 20,
                            "chat_template_kwargs": {"enable_thinking": True},
                        }
                    }
                }
                requests.append(request)
    
    return requests

def process_qa_evaluation_results(
    results: List[Dict[str, Any]],
    documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Process Q&A evaluation results and update documents with ratings.
    
    Args:
        results: List of evaluation results
        documents: List of documents to update
    
    Returns:
        Updated list of documents
    """
    successful_ratings = 0
    failed_ratings = 0
    
    for result in results:
        if not result.get('response'):
            failed_ratings += 1
            continue
            
        # Parse custom_id to get indices
        custom_id = result['custom_id']
        parts = custom_id.split('_')
        doc_idx = int(parts[3])
        cluster_idx = int(parts[5])
        qa_idx = int(parts[7])
        
        # Get the rating from the response
        # try:
        response_body = result['response']['body']
        if 'choices' in response_body and len(response_body['choices']) > 0:
            response_text = response_body['choices']['message']['content'].strip()
            # content = (
            #             result.get("response", {})
            #             .get("body", {})
            #             .get("choices", {})
            #             .get("message", {})
            #             .get("content", "")
            #         )
            # Handle response text that may contain thinking process
            if '</think>' in response_text:
                # Extract text after thinking process
                response_text = response_text.split('</think>')[-1].strip()
        else:
            print(f"No choices in response for {custom_id}")
            failed_ratings += 1
            continue
        rating = extract_rating_from_response(response_text)
        
        # Skip if rating extraction failed
        if rating is None:
            print(f"Failed to extract rating for doc {doc_idx}, cluster {cluster_idx}, qa {qa_idx}")
            failed_ratings += 1
            continue
            
        print(f"Rated Q&A pair for doc {doc_idx}, cluster {cluster_idx}, qa {qa_idx}: {rating}")
        
        # Update the document with the rating
        doc = documents[doc_idx]
        qa_pair = doc['clusters'][cluster_idx]['chunk_data']['qa_pairs'][qa_idx]
        
        # Add rating to the Q&A pair
        qa_pair['quality_rating'] = rating
        successful_ratings += 1
            
        # except Exception as e:
        #     print(f"Error processing result for {custom_id}: {str(e)}")
        #     failed_ratings += 1
        #     continue
    
    print(f"Successfully rated {successful_ratings} Q&A pairs, failed to rate {failed_ratings}")
    return documents

def process_qa_evaluation_batch_on_engine(
    engine_client: openai.Client,
    requests: List[Dict[str, Any]],
    engine_url: str
) -> List[Dict[str, Any]]:
    """
    Process Q&A evaluation batch on a specific engine.
    
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
        batch_file = f"batch_data/qa_eval_batch_{timestamp}.jsonl"
        
        # Write requests to batch file
        with open(batch_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        
        print(f"Engine {engine_url}: Processing {len(requests)} Q&A evaluation requests")
        
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
        print(f"Error processing Q&A evaluation batch on engine {engine_url}: {str(e)}")
        return []
    finally:
        # Clean up batch file
        try:
            if os.path.exists(batch_file):
                os.remove(batch_file)
        except:
            pass

def evaluate_qa_pairs_batch(
    documents: List[Dict[str, Any]],
    engine_urls: List[str],
    temperature: float = 0.1,
    max_tokens: int = 50
) -> List[Dict[str, Any]]:
    """
    Evaluate Q&A pairs using batch processing with parallel engine utilization.
    
    Args:
        documents: List of documents with Q&A pairs to evaluate
        engine_urls: List of VLLM engine URLs
        temperature: Low temperature for consistent evaluation
        max_tokens: Maximum tokens for rating response
    
    Returns:
        List of documents with quality ratings added
    """
    # Create a deep copy of documents to avoid modifying the original
    documents = [json.loads(json.dumps(doc)) for doc in documents]
    
    # Ensure batch_data directory exists
    os.makedirs('batch_data', exist_ok=True)
    
    # Prepare all evaluation requests
    print("Preparing Q&A evaluation requests...")
    all_requests = prepare_qa_evaluation_batch_requests(documents, temperature, max_tokens)
    
    if not all_requests:
        print("No Q&A evaluation requests to process")
        return documents
    
    print(f"Total Q&A evaluation requests: {len(all_requests)}")
    
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
                process_qa_evaluation_batch_on_engine,
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
    
    print(f"Total Q&A evaluation results: {len(all_results)}")
    
    # Process results and update documents
    if all_results:
        documents = process_qa_evaluation_results(all_results, documents)
    
    return documents

def print_qa_evaluation_summary(documents: List[Dict[str, Any]]):
    """Print summary statistics of the Q&A evaluation results."""
    total_qa_pairs = 0
    total_rated_qa_pairs = 0
    ratings = []
    
    for doc in documents:
        for cluster in doc['clusters']:
            for qa_pair in cluster['chunk_data']['qa_pairs']:
                total_qa_pairs += 1
                if 'quality_rating' in qa_pair:
                    total_rated_qa_pairs += 1
                    ratings.append(qa_pair['quality_rating'])
    
    print(f"\n=== Q&A Evaluation Summary ===")
    print(f"Total Q&A pairs: {total_qa_pairs}")
    print(f"Successfully rated Q&A pairs: {total_rated_qa_pairs}")
    print(f"Rating coverage: {total_rated_qa_pairs/total_qa_pairs*100:.1f}%" if total_qa_pairs > 0 else "0%")
    
    if ratings:
        print(f"Average quality rating: {sum(ratings)/len(ratings):.2f}")
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
        
        print("Quality rating distribution:")
        for i in range(len(bin_counts)):
            print(f"  {bins[i]:.0f}-{bins[i+1]:.0f}: {bin_counts[i]} ({bin_counts[i]/len(ratings)*100:.1f}%)")
        
        # Quality categories
        excellent = len([r for r in ratings if r >= 8])
        good = len([r for r in ratings if 6 <= r < 8])
        fair = len([r for r in ratings if 4 <= r < 6])
        poor = len([r for r in ratings if r < 4])
        
        print(f"\nQuality categories:")
        print(f"  Excellent (8-10): {excellent} ({excellent/len(ratings)*100:.1f}%)")
        print(f"  Good (6-8): {good} ({good/len(ratings)*100:.1f}%)")
        print(f"  Fair (4-6): {fair} ({fair/len(ratings)*100:.1f}%)")
        print(f"  Poor (0-4): {poor} ({poor/len(ratings)*100:.1f}%)")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, default="documents_with_qa_pairs.json", 
                        help="Path to input data with Q&A pairs")
    parser.add_argument("--output_data_path", type=str, default="documents_with_qa_ratings.json",
                        help="Path to save data with Q&A quality ratings")
    parser.add_argument("--temperature", type=float, default=0.6, 
                        help="Low temperature for consistent evaluation")
    parser.add_argument("--max_tokens", type=int, default=16384, 
                        help="Maximum tokens for rating response")
    parser.add_argument("--max_documents", type=int, default=10000, 
                        help="Maximum number of documents to process")
    
    args = parser.parse_args()
    
    # Load documents with Q&A pairs
    print(f"Loading documents from {args.input_data_path}")
    all_documents = []
    i = 0
    
    with open(args.input_data_path, 'r') as f:
        for line in f:
            if i % 1000 == 0:
                print(f"Loading documents: {i}")
            # if i >= 5:
            #     break
            try:
                item = json.loads(line.strip())
                all_documents.append(item)
                i += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {i}: {e}")
                continue
    
    print(f"Loaded {len(all_documents)} documents")
    
    # Count total Q&A pairs to evaluate
    total_qa_pairs = 0
    for doc in all_documents:
        for cluster in doc['clusters']:
            total_qa_pairs += len(cluster['chunk_data']['qa_pairs'])
    
    print(f"Total Q&A pairs to evaluate: {total_qa_pairs}")
    
    # VLLM engine URLs (same as generation script)
    port = 42692
    ip_list = [
        "192.168.13.250",
        "192.168.13.249",
        "192.168.13.248",
        "192.168.13.247",
        "192.168.13.246",
        "192.168.13.245",
        "192.168.13.244",
        "192.168.13.243",
        "192.168.13.242",
        "192.168.13.241",
    ]
    
    engine_urls = [f"http://{ip}:{port}" for ip in ip_list]
    
    # Evaluate Q&A pairs using batch processing
    print(f"\nEvaluating Q&A pairs using LLM-as-a-judge with {len(engine_urls)} engines")
    documents_with_ratings = evaluate_qa_pairs_batch(
        all_documents,
        engine_urls,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Print evaluation summary
    print_qa_evaluation_summary(documents_with_ratings)
    
    # Save results
    print(f"\nSaving results to {args.output_data_path}")
    with open(args.output_data_path, 'w') as f:
        for doc in documents_with_ratings:
            f.write(json.dumps(doc) + "\n")
    
    print("Q&A pair evaluation completed!")