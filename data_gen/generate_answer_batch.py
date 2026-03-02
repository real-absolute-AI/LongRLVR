"""
Now I wanna load the json with the structure like:
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

I wanna prompt the 8b model to generate answer given the document text and the question.
Please generate the answer for each question in the qa_pairs, and given the answer as a new kv in the "qa_pairs" with the key "generated_answers".

Please query the engines like @generate_questions.py
"""



"""
This is an example to submit batch job to sglang server:
import json
import time
from openai import OpenAI

client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

requests = [
    {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": "qwen/qwen2.5-0.5b-instruct",
            "messages": [
                {"role": "user", "content": "Tell me a joke about programming"}
            ],
            "max_tokens": 50,
        },
    },
    {
        "custom_id": "request-2",
        "method": "POST",
        "url": "/chat/completions",
        "body": {
            "model": "qwen/qwen2.5-0.5b-instruct",
            "messages": [{"role": "user", "content": "What is Python?"}],
            "max_tokens": 50,
        },
    },
]

input_file_path = "batch_requests.jsonl"

with open(input_file_path, "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

with open(input_file_path, "rb") as f:
    file_response = client.files.create(file=f, purpose="batch")

batch_response = client.batches.create(
    input_file_id=file_response.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

print_highlight(f"Batch job created with ID: {batch_response.id}")

while batch_response.status not in ["completed", "failed", "cancelled"]:
    time.sleep(3)
    print(f"Batch job status: {batch_response.status}...trying again in 3 seconds...")
    batch_response = client.batches.retrieve(batch_response.id)

if batch_response.status == "completed":
    print("Batch job completed successfully!")
    print(f"Request counts: {batch_response.request_counts}")

    result_file_id = batch_response.output_file_id
    file_response = client.files.content(result_file_id)
    result_content = file_response.read().decode("utf-8")

    results = [
        json.loads(line) for line in result_content.split("\n") if line.strip() != ""
    ]

    for result in results:
        print_highlight(f"Request {result['custom_id']}:")
        print_highlight(f"Response: {result['response']}")

    print_highlight("Cleaning up files...")
    # Only delete the result file ID since file_response is just content
    client.files.delete(result_file_id)
else:
    print_highlight(f"Batch job failed with status: {batch_response.status}")
    if hasattr(batch_response, "errors"):
        print_highlight(f"Errors: {batch_response.errors}")

        
import json
import time
from openai import OpenAI

client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

requests = []
for i in range(20):
    requests.append(
        {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": "qwen/qwen2.5-0.5b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": f"{i}: You are a helpful AI assistant",
                    },
                    {
                        "role": "user",
                        "content": "Write a detailed story about topic. Make it very long.",
                    },
                ],
                "max_tokens": 64,
            },
        }
    )

input_file_path = "batch_requests.jsonl"
with open(input_file_path, "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

with open(input_file_path, "rb") as f:
    uploaded_file = client.files.create(file=f, purpose="batch")

batch_job = client.batches.create(
    input_file_id=uploaded_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

print_highlight(f"Created batch job with ID: {batch_job.id}")
print_highlight(f"Initial status: {batch_job.status}")

time.sleep(10)

max_checks = 5
for i in range(max_checks):
    batch_details = client.batches.retrieve(batch_id=batch_job.id)

    print_highlight(
        f"Batch job details (check {i+1} / {max_checks}) // ID: {batch_details.id} // Status: {batch_details.status} // Created at: {batch_details.created_at} // Input file ID: {batch_details.input_file_id} // Output file ID: {batch_details.output_file_id}"
    )
    print_highlight(
        f"<strong>Request counts: Total: {batch_details.request_counts.total} // Completed: {batch_details.request_counts.completed} // Failed: {batch_details.request_counts.failed}</strong>"
    )

    time.sleep(3)

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
class BatchJob:
    """Represents a batch job for processing"""
    doc_idx: int
    document: Dict[str, Any]
    batch_file: str
    answer_idx: int  # Which answer this is (0 for question-only, 1 for doc+question)

class VLLMEnginePool:
    """Manages a pool of VLLM engines with load balancing."""
    
    def __init__(self, engine_urls: List[str]):
        """
        Initialize the VLLM engine pool.
        
        Args:
            engine_urls: List of VLLM engine URLs
        """
        self.engines = [
            VLLMEngine(
                url=url,
                client=openai.Client(
                    api_key="EMPTY",
                    base_url=f"{url}/v1"
                )
            )
            for url in engine_urls
        ]
        self.num_engines = len(self.engines)
        self.current_index = 0
        self.index_lock = Lock()
    
    def get_engine(self) -> VLLMEngine:
        """
        Get the next available engine using round-robin load balancing.
        This is now non-blocking and can be called concurrently.
        
        Returns:
            VLLMEngine instance
        """
        # Use atomic operation to get and increment index
        with self.index_lock:
            engine = self.engines[self.current_index]
            self.current_index = (self.current_index + 1) % self.num_engines
        
        # Update engine metrics without blocking other engine selections
        with engine.lock:
            engine.last_used = time.time()
            engine.request_count += 1
        
        return engine
    
    def get_engine_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all engines.
        
        Returns:
            List of engine statistics
        """
        return [
            {
                "url": engine.url,
                "request_count": engine.request_count,
                "last_used": engine.last_used
            }
            for engine in self.engines
        ]

def create_answer_generation_prompt(document_text: str, question: str) -> str:
    """
    Create a prompt for generating answers based on document text and question.
    
    Args:
        document_text: The full document text
        question: The question to answer
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert at answering questions based on provided documents. Your task is to generate a comprehensive and accurate answer to the given question using only the information provided in the document.

Instructions:
1. Read the document carefully and understand the context
2. Answer the question based solely on information from the document
3. Provide a concise and well-structured answer
4. Be precise and factual in your response

Document:
{document_text}

Question: {question}

"""
    
    return prompt

def create_question_only_prompt(question: str) -> str:
    """
    Create a prompt for generating an answer **without** providing the document.

    Args:
        question: The question to answer

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert at answering questions. Provide a concise, factual and well-structured answer to the following question.

Question: {question}
"""
    return prompt

def generate_answers_for_qa_pair(
    document_text: str,
    question: str,
    engine_client: openai.Client,
    num_answers: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 8192,
    max_retries: int = 1
) -> List[str]:
    """
    Generate multiple answers for a single question given document text.
    
    Args:
        document_text: The full document text
        question: The question to answer
        engine_client: OpenAI client for the engine
        num_answers: Number of different answers to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens for generation
        max_retries: Maximum number of retries if generation fails
    
    Returns:
        List of generated answer texts
    """
    prompt = create_answer_generation_prompt(document_text, question)
    answers = []
    
    for answer_idx in range(num_answers):
        for attempt in range(max_retries):
            try:
                response = engine_client.chat.completions.create(
                    model="default",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    # presence_penalty=0.5,
                    extra_body={
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                )
                
                answer_text = response.choices[0].message.content.strip()
                answers.append(answer_text)
                break  # Success, move to next answer
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to generate answer {answer_idx + 1} after {max_retries} attempts: {str(e)}")
                    answers.append(f"Error generating answer: {str(e)}")
                    break
                time.sleep(1)  # Wait before retrying
    
    return answers

# Global engine clients cache per process
_engine_clients = {}

def get_engine_client(engine_urls: List[str], engine_idx: int):
    """Get or create an engine client for the specified engine."""
    global _engine_clients
    
    if engine_idx not in _engine_clients:
        engine_url = engine_urls[engine_idx]
        print(f"Creating client for engine {engine_idx}: {engine_url}")
        _engine_clients[engine_idx] = openai.Client(
            api_key="EMPTY",
            base_url=f"{engine_url}/v1"
        )
    
    return _engine_clients[engine_idx]

def process_single_answer_generation(args: tuple) -> Dict[str, Any]:
    """
    Process a single answer generation for a QA pair.
    
    Args:
        args: Tuple containing (document, doc_idx, cluster_idx, qa_idx, temperature, top_p, max_tokens, engine_urls, engine_idx)
    
    Returns:
        Dictionary containing the generated answer and metadata
    """
    document, doc_idx, cluster_idx, qa_idx, temperature, top_p, max_tokens, engine_urls, engine_idx = args
    
    # Get or create engine client
    try:
        engine = get_engine_client(engine_urls, engine_idx)
    except Exception as e:
        print(f"Failed to create engine client for engine {engine_idx}: {str(e)}")
        return None
    
    try:
        qa_pair = document['clusters'][cluster_idx]['chunk_data']['qa_pairs'][qa_idx]
        question = qa_pair.get('question', '')
        
        if not question:
            return None
            
        # Generate single answer
        generated_answer = generate_answers_for_qa_pair(
            document_text=document['text'],
            question=question,
            engine_client=engine,
            num_answers=1,  # Generate only one answer
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )[0]  # Take the first (and only) answer
        
        return {
            'doc_idx': doc_idx,
            'cluster_idx': cluster_idx,
            'qa_idx': qa_idx,
            'answer': generated_answer,
            'success': True
        }
        
    except Exception as e:
        print(f"Error generating answer for doc {doc_idx}, cluster {cluster_idx}, qa {qa_idx}: {str(e)}")
        return {
            'doc_idx': doc_idx,
            'cluster_idx': cluster_idx,
            'qa_idx': qa_idx,
            'answer': f"Error generating answer: {str(e)}",
            'success': False
        }

def generate_answers_parallel(
    documents: List[Dict[str, Any]],
    engine_urls: List[str],
    num_workers: int = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 8192,
    answer_index: int = 0  # Which answer this is (0 for question-only, 1 for doc+question)
) -> List[Dict[str, Any]]:
    """
    Main pipeline to generate answers for multiple documents in parallel.
    Generates one answer at a time for all questions across all documents.
    
    Args:
        documents: List of documents with QA pairs
        engine_urls: List of VLLM engine URLs
        num_workers: Number of parallel workers
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens for generation
        answer_index: Which answer this is (0 for question-only, 1 for doc+question)
    
    Returns:
        List of documents with generated answers added
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    num_engines = len(engine_urls)
    print(f"Total documents: {len(documents)}, Total engines: {num_engines}")
    
    # here to ensure each document has mostly 4 clusters, if not, randomly sample 4 clusters
    for doc in documents:
        if len(doc['clusters']) > 4:
            doc['clusters'] = random.sample(doc['clusters'], 4)
    
    # Assign each document to a specific engine
    # This ensures all clusters from the same document go to the same engine
    doc_engine_assignments = {doc_idx: doc_idx % num_engines for doc_idx in range(len(documents))}
    
    # Prepare arguments for parallel processing
    args_list = []
    for doc_idx, doc in enumerate(documents):
        engine_idx = doc_engine_assignments[doc_idx]
        for cluster_idx, cluster in enumerate(doc['clusters']):
            for qa_idx, qa_pair in enumerate(cluster['chunk_data']['qa_pairs']):
                if qa_pair.get('question'):  # Only process if there's a question
                    args_list.append((
                        doc, doc_idx, cluster_idx, qa_idx,
                        temperature, top_p, max_tokens,
                        engine_urls, engine_idx
                    ))
    
    print(f"Generating answer {answer_index + 1} for {len(args_list)} questions")
    print(f"Engine assignment distribution: {[args_list.count(i) for i in range(num_engines)]}")
    
    # Process all questions in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_answer_generation, args_list),
            total=len(args_list),
            desc=f"Generating answer {answer_index + 1}"
        ))
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Add answers to documents
    for result in results:
        if result['success']:
            doc = documents[result['doc_idx']]
            qa_pair = doc['clusters'][result['cluster_idx']]['chunk_data']['qa_pairs'][result['qa_idx']]
            
            # Initialize generated_answers if it doesn't exist
            if 'generated_answers' not in qa_pair:
                qa_pair['generated_answers'] = []
            
            # Add the new answer
            qa_pair['generated_answers'].append({
                'text': result['answer'],
                'cluster_idx': result['cluster_idx'],
                'qa_idx': result['qa_idx']
            })
    
    return documents

def prepare_batch_requests(
    document: Dict[str, Any],
    doc_idx: int,
    answer_idx: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 8192
) -> List[Dict[str, Any]]:
    """
    Prepare batch requests for a single document's clusters.
    
    Args:
        document: The document containing clusters and QA pairs
        doc_idx: Index of the document
        answer_idx: Which answer this is (0 for question-only, 1 for doc+question)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens for generation
    
    Returns:
        List of batch request dictionaries
    """
    requests = []
    
    for cluster_idx, cluster in enumerate(document['clusters']):
        for qa_idx, qa_pair in enumerate(cluster['chunk_data']['qa_pairs']):
            if not qa_pair.get('question'):
                continue
                
            # Select prompt style based on which answer we are generating
            if answer_idx == 0:
                # First answer: use question only (no document context)
                prompt = create_question_only_prompt(qa_pair['question'])
            else:
                # Second answer: use document + question (existing behaviour)
                prompt = create_answer_generation_prompt(document['text'], qa_pair['question'])
            
            request = {
                "custom_id": f"doc_{doc_idx}_cluster_{cluster_idx}_qa_{qa_idx}_answer_{answer_idx}",
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "extra_body": {
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                }
            }
            requests.append(request)
    
    return requests

def process_batch_results(
    results: List[Dict[str, Any]],
    documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Process batch results and update documents with generated answers.
    
    Args:
        results: List of batch results
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
        doc_idx = int(parts[1])
        cluster_idx = int(parts[3])
        qa_idx = int(parts[5])
        answer_idx = int(parts[7])
        
        # print(result)
        # Get the answer text from the response
        # answer_text = result['response']['choices'][0]['message']['content'].strip()
        # print(result['response']['body']['choices']['message'])
        assert result['response']['body']['choices']['message']['role'] == "assistant"
        answer_text = result['response']['body']['choices']['message']['content'].strip()
        print(f"Generated answer {answer_idx} for doc {doc_idx}, cluster {cluster_idx}, qa {qa_idx}")
        
        # Update the document with the generated answer
        doc = documents[doc_idx]
        qa_pair = doc['clusters'][cluster_idx]['chunk_data']['qa_pairs'][qa_idx]
        
        # Initialize generated_answers if it doesn't exist
        if 'generated_answers' not in qa_pair:
            qa_pair['generated_answers'] = []
        
        # Add the new answer
        qa_pair['generated_answers'].append({
            'text': answer_text,
            'answer_idx': answer_idx,
            'cluster_idx': cluster_idx,
            'qa_idx': qa_idx
        })
        
        # Ensure the document is updated in the documents list
        documents[doc_idx] = doc
    
    return documents

def process_batch_job_on_engine(
    engine: VLLMEngine,
    batch_job: BatchJob,
    documents: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int
) -> bool:
    """
    Process a single batch job on a specific engine.
    
    Args:
        engine: The VLLM engine to process the job
        batch_job: The batch job to process
        documents: List of all documents (for updating results)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens for generation
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Prepare batch requests for this document
        requests = prepare_batch_requests(
            batch_job.document,
            batch_job.doc_idx,
            batch_job.answer_idx,
            temperature,
            top_p,
            max_tokens
        )
        
        if not requests:
            return True  # No requests to process, consider successful
        
        # Write requests to batch file
        with open(batch_job.batch_file, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        
        # Upload batch file
        with open(batch_job.batch_file, "rb") as f:
            file_response = engine.client.files.create(file=f, purpose="batch")
        
        # Create batch job
        batch_response = engine.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        
        print(f"Created batch job {batch_response.id} for document {batch_job.doc_idx} answer {batch_job.answer_idx} on engine {engine.url}")
        
        # Wait for batch completion with polling
        while batch_response.status not in ["completed", "failed", "cancelled"]:
            time.sleep(10)  # Reduced polling interval
            batch_response = engine.client.batches.retrieve(batch_response.id)
            print(f"Engine {engine.url} - Doc {batch_job.doc_idx} Answer {batch_job.answer_idx} - Status: {batch_response.status}")
        
        if batch_response.status == "completed":
            # Get results
            result_file_id = batch_response.output_file_id
            file_content = engine.client.files.content(result_file_id)
            result_content = file_content.read().decode("utf-8")
            
            results = [
                json.loads(line) for line in result_content.split("\n") if line.strip()
            ]
            
            # Process results and update documents
            process_batch_results(results, documents)
            
            # Clean up files
            engine.client.files.delete(result_file_id)
            engine.client.files.delete(file_response.id)
            
            print(f"Completed batch job for document {batch_job.doc_idx} answer {batch_job.answer_idx} on engine {engine.url}")
            return True
        else:
            print(f"Batch job failed with status: {batch_response.status} for document {batch_job.doc_idx} answer {batch_job.answer_idx} on engine {engine.url}")
            if hasattr(batch_response, "errors"):
                print(f"Errors: {batch_response.errors}")
            return False
            
    except Exception as e:
        print(f"Error processing batch for document {batch_job.doc_idx} answer {batch_job.answer_idx} on engine {engine.url}: {str(e)}")
        return False
    finally:
        # Clean up batch file
        try:
            if os.path.exists(batch_job.batch_file):
                os.remove(batch_job.batch_file)
        except:
            pass

def generate_answers_batch(
    documents: List[Dict[str, Any]],
    engine_urls: List[str],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 8192,
    data_parallel_degree: int = 2
) -> List[Dict[str, Any]]:
    """
    Generate answers using batch processing with parallel engine utilization.
    
    Args:
        documents: List of documents with QA pairs
        engine_urls: List of VLLM engine URLs
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens for generation
        data_parallel_degree: Number of documents each engine can process in parallel
    
    Returns:
        List of documents with generated answers added
    """
    # Create a deep copy of documents to avoid modifying the original
    documents = [json.loads(json.dumps(doc)) for doc in documents]
    
    # Ensure batch_data directory exists
    os.makedirs('batch_data', exist_ok=True)
    
    # Create batch jobs queue
    job_queue = Queue()
    
    # Prepare all batch jobs - 2 jobs per document (question-only and doc+question)
    num_answers = 2  # We now want exactly two answers per QA pair
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for doc_idx, doc in enumerate(documents):
        for answer_idx in range(num_answers):  # Generate 2 answers per question
            batch_file = f"batch_data/batch_doc_{doc_idx}_answer_{answer_idx}_{timestamp}.jsonl"
            batch_job = BatchJob(
                doc_idx=doc_idx,
                document=doc,
                batch_file=batch_file,
                answer_idx=answer_idx
            )
            job_queue.put(batch_job)
    
    print(f"Created {job_queue.qsize()} batch jobs for {len(documents)} documents ({num_answers} answers each)")
    print(f"Each engine will process {data_parallel_degree} documents in parallel")
    
    # Create engines
    engines = [
        openai.Client(api_key="EMPTY", base_url=f"{url}/v1")
        for url in engine_urls
    ]
    
    # Create engine wrappers
    engine_wrappers = [
        VLLMEngine(url=url, client=client)
        for url, client in zip(engine_urls, engines)
    ]
    
    # Progress tracking
    completed_jobs = 0
    total_jobs = len(documents) * num_answers
    progress_lock = Lock()
    
    def single_batch_worker(engine: VLLMEngine, batch_job: BatchJob):
        """Process a single batch job on an engine."""
        try:
            success = process_batch_job_on_engine(
                engine, batch_job, documents, temperature, top_p, max_tokens
            )
            
            # Update progress
            with progress_lock:
                nonlocal completed_jobs
                completed_jobs += 1
                print(f"Progress: {completed_jobs}/{total_jobs} batch jobs completed on engine {engine.url}")
            
            return success
        except Exception as e:
            print(f"Error processing batch job for doc {batch_job.doc_idx} answer {batch_job.answer_idx} on engine {engine.url}: {str(e)}")
            return False
    
    def engine_worker_thread(engine: VLLMEngine):
        """Worker thread function for processing multiple jobs in parallel on a specific engine."""
        with ThreadPoolExecutor(max_workers=data_parallel_degree) as executor:
            while True:
                # Collect batch jobs for this engine (up to data_parallel_degree)
                current_jobs = []
                for _ in range(data_parallel_degree):
                    try:
                        batch_job = job_queue.get(timeout=2)
                        current_jobs.append(batch_job)
                    except:
                        # No more jobs available or timeout
                        break
                
                if not current_jobs:
                    # No jobs to process, exit
                    break
                
                print(f"Engine {engine.url} starting {len(current_jobs)} parallel batch jobs")
                
                # Submit all jobs to the thread pool
                future_to_job = {
                    executor.submit(single_batch_worker, engine, job): job 
                    for job in current_jobs
                }
                
                # Wait for all jobs to complete
                for future in as_completed(future_to_job):
                    batch_job = future_to_job[future]
                    try:
                        success = future.result()
                        if success:
                            print(f"Successfully completed doc {batch_job.doc_idx} answer {batch_job.answer_idx} on engine {engine.url}")
                        else:
                            print(f"Failed to complete doc {batch_job.doc_idx} answer {batch_job.answer_idx} on engine {engine.url}")
                    except Exception as e:
                        print(f"Exception in batch job for doc {batch_job.doc_idx} answer {batch_job.answer_idx} on engine {engine.url}: {str(e)}")
                    finally:
                        # Mark job as done
                        job_queue.task_done()
    
    # Start worker threads for each engine
    threads = []
    for engine in engine_wrappers:
        thread = threading.Thread(target=engine_worker_thread, args=(engine,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    print(f"Started {len(threads)} engine worker threads, each handling {data_parallel_degree} parallel jobs")
    
    # Wait for all jobs to complete
    job_queue.join()
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join(timeout=30)
    
    print(f"Completed processing all {total_jobs} batch jobs")
    
    # Verify that generated answers are present
    for doc_idx, doc in enumerate(documents):
        for cluster_idx, cluster in enumerate(doc['clusters']):
            for qa_idx, qa_pair in enumerate(cluster['chunk_data']['qa_pairs']):
                if 'generated_answers' not in qa_pair or len(qa_pair['generated_answers']) < num_answers:
                    print(f"Warning: Expected {num_answers} answers but found {len(qa_pair.get('generated_answers', []))} for doc {doc_idx}, cluster {cluster_idx}, qa {qa_idx}")
    
    return documents

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, default="documents_with_qa.json")
    parser.add_argument("--output_data_path", type=str, default="documents_with_generated_answers.json")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum tokens for generation")
    parser.add_argument("--max_documents", type=int, default=10000, help="Maximum number of documents to process")
    args = parser.parse_args()
    # now input_data_path is a list
    # Load documents from JSON
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
    
    # VLLM engine URLs
    port = 42692
    ip_list = [
        "192.168.11.63",
        "192.168.11.62",
        "192.168.11.61",
        "192.168.11.60",
        "192.168.11.59",
        "192.168.11.58",
        "192.168.11.56",
        "192.168.11.55",

    ]
    
    engine_urls = [f"http://{ip}:{port}" for ip in ip_list]
    
    # Generate two answers (question-only and doc+question) in parallel using batch processing
    print(f"\nGenerating 2 answers per question in parallel")
    documents_with_answers = generate_answers_batch(
        all_documents,
        engine_urls,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        data_parallel_degree=2  # Each engine processes 2 documents in parallel
    )
    
    # Save final results
    dir_path = os.path.dirname(args.output_data_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    print(f"Saving final results to {args.output_data_path}")
    with open(args.output_data_path, 'w') as f:
        for doc in documents_with_answers:
            f.write(json.dumps(doc) + "\n")
    
    print("Answer generation completed!")
    
    # Print summary statistics
    total_questions = 0
    total_answers = 0
    for doc in documents_with_answers:
        for cluster in doc['clusters']:
            for qa_pair in cluster['chunk_data']['qa_pairs']:
                if qa_pair.get('question'):
                    total_questions += 1
                    total_answers += len(qa_pair.get('generated_answers', []))
    
    print(f"Summary: {total_questions} questions processed, {total_answers} answers generated")
    print(f"Average answers per question: {total_answers/total_questions if total_questions > 0 else 0:.2f}")



