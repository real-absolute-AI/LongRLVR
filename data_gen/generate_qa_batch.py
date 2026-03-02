"""
Batch-based question generation script
 =====================================
 This script improves upon `generate_questions_v2.py` and fulfils three
 new requirements:

 1. Each request to the model produces **exactly one** question–answer
    pair.  To obtain *N* questions for a cluster we therefore create *N*
    independent prompts (queries).
 2. We interact with the vLLM instances through the **OpenAI batch API**
    (compatible with sglang).  All queries that belong to the *same*
    cluster are written to the **same batch file** and are therefore
    processed together.  Clusters are distributed to engines in a round-
    robin fashion.  As soon as an engine finishes the current batch, the
    next waiting cluster batch is immediately submitted, enabling full
    parallel utilisation of multiple engines.
 3. Generation results are written **incrementally**.  As soon as we
    receive the responses for a cluster they are appended to the output
    file.  This allows monitoring progress and recovering from crashes
    without losing finished work.

 The code purposefully mirrors the evaluation pipeline in
 `answer_judge.py` so that the engineering patterns stay consistent
 across generation and evaluation stages.
"""

import argparse
import json
import os
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple, Optional
from collections import defaultdict
import queue

import openai
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def create_single_question_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Return a prompt that asks the model for ONE challenging QA pair."""
    # Combine chunks with chunk IDs
    combined_text = ""
    for i, chunk in enumerate(chunks):
        combined_text += f"[Chunk {i}]\n{chunk['text']}\n\n"

    prompt = f"""You are an expert at creating challenging questions that test deep
understanding and reasoning across multiple pieces of information from a
text. **Generate exactly ONE question–answer pair** that:

• requires synthesising information from *as many chunks as possible*;
• cannot be solved by reading any single chunk alone;
• has a clear, unambiguous answer that can be verified from the text;
• avoids any reference to chunk numbers in the question itself and the answer.

If you determine that the provided chunks are not sufficiently related or do
not contain enough information to form a meaningful, multi-chunk question,
you should skip generation.

Return your result as pure JSON **without markdown fences**.

For a valid question, use the following shape (no additional keys, no
explanation):
{{
  "question": "...",
  "answer": {{
    "text": "concise answer (<= 40 words)",
    "evidence": [0, 1, 2],
    "explanation": "step-by-step reasoning (max 70 words)"
  }}
}}

To skip generation, return this exact JSON object:
{{
  "question": "SKIPPED",
  "answer": null
}}

Where "evidence" is a list of chunk IDs (numbers) that contain the information needed to answer the question.

Text to read:
{combined_text}
"""
    return prompt


def extract_json_from_response(content: str) -> Dict[str, Any]:
    """Robustly extract JSON from the model's response."""
    content = content.strip()
    # Remove common markdown fences if present
    if content.startswith("```"):
        content = content.split("```", 2)[1]
    # The model is instructed to return pure JSON, but we still guard
    # against trailing text.
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace == -1 or last_brace == -1:
        raise ValueError(f"Cannot locate JSON object in response: {content}")
    json_str = content[first_brace : last_brace + 1]
    return json.loads(json_str)


# ---------------------------------------------------------------------------
# Engine Health Management
# ---------------------------------------------------------------------------

class EngineHealthManager:
    """Thread-safe manager for tracking engine health and failures."""
    
    def __init__(self, max_consecutive_failures: int = 3):
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = defaultdict(int)  # engine_index -> failure_count
        self.down_engines = set()  # set of engine indices that are marked as down
        self.lock = threading.Lock()
    
    def record_success(self, engine_index: int) -> None:
        """Record a successful operation for an engine."""
        with self.lock:
            self.consecutive_failures[engine_index] = 0
            # If engine was down, bring it back up
            if engine_index in self.down_engines:
                self.down_engines.remove(engine_index)
                print(f"Engine {engine_index}: Recovered and back online")
    
    def record_failure(self, engine_index: int) -> bool:
        """Record a failure for an engine. Returns True if engine should be marked as down."""
        with self.lock:
            self.consecutive_failures[engine_index] += 1
            
            if (self.consecutive_failures[engine_index] >= self.max_consecutive_failures 
                and engine_index not in self.down_engines):
                self.down_engines.add(engine_index)
                print(f"Engine {engine_index}: Marked as DOWN after {self.consecutive_failures[engine_index]} consecutive failures")
                return True
            
            return False
    
    def is_engine_down(self, engine_index: int) -> bool:
        """Check if an engine is marked as down."""
        with self.lock:
            return engine_index in self.down_engines
    
    def get_healthy_engines(self, total_engines: int) -> List[int]:
        """Get list of healthy engine indices."""
        with self.lock:
            return [i for i in range(total_engines) if i not in self.down_engines]
    
    def get_failure_count(self, engine_index: int) -> int:
        """Get current consecutive failure count for an engine."""
        with self.lock:
            return self.consecutive_failures[engine_index]


# ---------------------------------------------------------------------------
# Batch preparation helpers
# ---------------------------------------------------------------------------

def prepare_cluster_requests(
    doc_idx: int,
    cluster_idx: int,
    cluster: Dict[str, Any],
    q_indices: List[int],
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    """Return the list of **request objects** for this cluster restricted to ``q_indices`` (remaining questions)."""

    if not q_indices:
        return []

    # The prompt is identical for all questions of the same cluster, so we can
    # build it once and reuse it for every remaining question index.
    prompt = create_single_question_prompt(cluster["chunks"])

    requests = []
    for q_idx in q_indices:
        requests.append(
            {
                "custom_id": f"gen_doc_{doc_idx}_cluster_{cluster_idx}_q_{q_idx}",
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
                    },
                },
            }
        )
    return requests


# ---------------------------------------------------------------------------
# Engine interaction helpers (mirrors answer_judge.py)
# ---------------------------------------------------------------------------

def process_generation_batch_on_engine(
    engine_client: openai.Client,
    requests: List[Dict[str, Any]],
    engine_url: str,
) -> List[Dict[str, Any]]:
    """Submit *one* batch file to the given engine and return its results."""
    if not requests:
        return []

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    batch_file_path = os.path.join("batch_data", f"gen_batch_{timestamp}.jsonl")
    os.makedirs("batch_data", exist_ok=True)

    # Write requests to file
    with open(batch_file_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    try:
        # Upload & launch batch
        with open(batch_file_path, "rb") as f:
            file_resp = engine_client.files.create(file=f, purpose="batch")

        batch_resp = engine_client.batches.create(
            input_file_id=file_resp.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        # Poll until finished, being tolerant to gateway behaviour
        while True:
            if batch_resp.status in {"completed", "failed", "cancelled"}:
                break
            time.sleep(5)
            try:
                batch_resp = engine_client.batches.retrieve(batch_resp.id)
            except openai.NotFoundError:
                # Some gateways remove metadata once the batch is done – treat as success.
                print(f"Engine {engine_url}: batch {batch_resp.id} metadata missing; assuming completed.")
                break
            except Exception as e:
                print(f"Engine {engine_url}: error retrieving batch status: {e}. Retrying…")
                continue

        if getattr(batch_resp, "status", None) != "completed":
            print(f"Engine {engine_url}: batch did not complete successfully (status={getattr(batch_resp, 'status', 'unknown')}).")
            return []

        # Fetch results
        result_file_id = batch_resp.output_file_id
        raw_content = engine_client.files.content(result_file_id).read().decode()
        # cleanup remote files
        engine_client.files.delete(result_file_id)
        engine_client.files.delete(file_resp.id)

        return [json.loads(line) for line in raw_content.split("\n") if line.strip()]
    finally:
        # cleanup local
        try:
            os.remove(batch_file_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Output organization helpers
# ---------------------------------------------------------------------------

def organize_output_format(
    raw_output_path: str,
    organized_output_path: str,
    original_documents: List[List[Dict[str, Any]]]
) -> None:
    """
    Reorganize the raw JSONL output into the specified nested format.
    
    Args:
        raw_output_path: Path to the raw JSONL output
        organized_output_path: Path to save the organized output
        original_documents: Original document structure for reference
    """
    # Read all generated QA pairs
    generated_qa = defaultdict(lambda: defaultdict(list))  # doc_idx -> cluster_idx -> [qa_pairs]
    
    if os.path.exists(raw_output_path):
        with open(raw_output_path, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)

                    # If the model skipped generation, do not include it.
                    if record.get("question") == "SKIPPED":
                        continue

                    doc_idx = record["doc_idx"]
                    cluster_idx = record["cluster_idx"]
                    qa_pair = {
                        "question": record["question"],
                        "answer": record["answer"]
                    }
                    generated_qa[doc_idx][cluster_idx].append(qa_pair)
    
    # Build the organized structure
    organized_documents = []
    
    for doc in original_documents:
        doc_idx = doc[0]['id']
        assert doc[0]['id'] == doc[-1]['id']
        doc_structure = {
            "id": f"document_{doc_idx}",
            "text": "",  # Can be filled if needed
            "clusters": []
        }
        
        for cluster_idx, cluster in enumerate(doc):
            cluster_structure = {
                "chunk_data": {
                    "chunks": cluster["chunks"],
                    "qa_pairs": generated_qa[doc_idx].get(cluster_idx, [])
                }
            }
            doc_structure["clusters"].append(cluster_structure)
        
        organized_documents.append(doc_structure)
    
    # Save organized output
    with open(organized_output_path, "w") as f:
        for doc in organized_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Organized output saved to {organized_output_path}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", default="documents.jsonl", help="(Deprecated) Single input JSONL path.")
    parser.add_argument("--input_data_paths", nargs="+", default=None, help="One or more input JSONL paths (each line is a clustered document).")
    parser.add_argument("--output_data_path", default="generated_questions.jsonl")
    parser.add_argument("--organized_output_path", default="organized_questions.json")
    parser.add_argument("--num_questions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--resume", action="store_true", help="Resume from an interrupted run by skipping already generated question–answer pairs.")
    parser.add_argument("--max_consecutive_failures", type=int, default=3, help="Maximum consecutive failures before marking an engine as down.")  
    parser.add_argument(
        "--engines",
        nargs="+",
        help="Space-separated list of engine IPs or full base URLs. If an item starts with http(s):// it is used as-is; otherwise we prefix 'http://' and append --port.",
    )
    parser.add_argument("--port", type=int, default=42692, help="Port to use when --engines are given as bare IP addresses.")
    args = parser.parse_args()

    # ----------------------------------------------------
    # Load input documents
    # ----------------------------------------------------
    documents: List[List[Dict[str, Any]]] = []
    input_paths = args.input_data_paths or [args.input_data_path]
    for input_data_path in input_paths:
        with open(input_data_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                documents.append(json.loads(line))
    print(f"Loaded {len(documents)} documents from {len(input_paths)} file(s)")

    # Flatten clusters and remember their origin
    clusters: List[Dict[str, Any]] = []
    origin_map: List[Dict[str, int]] = []  # each entry: {doc_idx, cluster_idx}

    for doc_idx, doc in enumerate(documents):
        clusters_with_indices = list(enumerate(doc))
        if len(clusters_with_indices) > 4:
            clusters_with_indices = random.sample(clusters_with_indices, 4)

        for cluster_idx, cluster in clusters_with_indices:
            clusters.append(cluster)
            origin_map.append({"doc_idx": cluster['id'], "cluster_idx": cluster_idx})

    print(f"Total clusters: {len(clusters)}")

    # ----------------------------------------------------
    # Determine which questions have already been generated (for resume)
    # ----------------------------------------------------
    processed_questions: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

    if args.resume and os.path.exists(args.output_data_path):
        print(f"Resume enabled – scanning existing output file {args.output_data_path} …")
        with open(args.output_data_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec["doc_idx"], rec["cluster_idx"])
                    processed_questions[key].add(rec["q_idx"])
                except Exception as e:
                    # Corrupted line – skip it but warn
                    print(f"Warning: could not parse existing record: {e}")

    # ----------------------------------------------------
    # Prepare request batches (one batch == one cluster, only remaining q's)
    # ----------------------------------------------------
    all_batches: List[List[Dict[str, Any]]] = []
    skipped_clusters = 0
    for idx, cluster in enumerate(clusters):
        meta = origin_map[idx]
        key = (meta["doc_idx"], meta["cluster_idx"])
        done_qs = processed_questions.get(key, set())
        remaining_qs = [q for q in range(args.num_questions) if q not in done_qs]

        if not remaining_qs:
            skipped_clusters += 1
            continue  # this cluster is already fully processed

        batch_reqs = prepare_cluster_requests(
            meta["doc_idx"],
            meta["cluster_idx"],
            cluster,
            remaining_qs,
            args.temperature,
            args.max_tokens,
        )

        # It's possible that a cluster is partially processed and we only send the remaining questions.
        if batch_reqs:
            all_batches.append(batch_reqs)

    if args.resume:
        print(f"Skipped {skipped_clusters} fully processed clusters. Remaining clusters to process: {len(all_batches)}")

    # ----------------------------------------------------
    # Construct engine URLs from CLI or fall back to default IP list.
    # ----------------------------------------------------

    if args.engines is None:
        # Fallback hard-coded IP list (update as necessary).
        default_ips = [
            "192.168.11.23",
            "192.168.11.22",
            "192.168.11.21",
            "192.168.11.20",
            "192.168.11.19",
            "192.168.11.18",
            "192.168.11.17",
            "192.168.11.16",
            "192.168.11.15",
            "192.168.11.14",
            "192.168.11.13",
            "192.168.11.12",
        ]
        raw_engines = default_ips
    else:
        raw_engines = args.engines

    engine_urls: List[str] = []
    for item in raw_engines:
        url = item.rstrip("/")  # remove trailing slash for consistency
        if url.startswith("http://") or url.startswith("https://"):
            engine_urls.append(url)
        else:
            engine_urls.append(f"http://{url}:{args.port}")

    # Deduplicate while preserving order
    seen = set()
    engine_urls = [u for u in engine_urls if not (u in seen or seen.add(u))]

    engine_clients = [openai.Client(api_key="EMPTY", base_url=f"{url}/v1") for url in engine_urls]

    num_engines = len(engine_clients)
    print(f"Using {num_engines} engine{'s' if num_engines != 1 else ''}…")

    # -----------------------------------------------------------------
    # Initialize engine health manager and work distribution
    # -----------------------------------------------------------------
    health_manager = EngineHealthManager(max_consecutive_failures=args.max_consecutive_failures)
    
    # Shared work queue so each engine only receives a new job after finishing
    work_queue: "queue.Queue[int]" = queue.Queue()
    failed_batches_queue: "queue.Queue[int]" = queue.Queue()  # Queue for redistributing failed batches
    
    for idx in range(len(all_batches)):
        work_queue.put(idx)

    # Thread-safe utilities (file writes & progress bar)
    save_lock = threading.Lock()
    progress_lock = threading.Lock()
    os.makedirs(os.path.dirname(args.output_data_path) or ".", exist_ok=True)

    # Progress bar reflects *batches (clusters)* processed rather than engines
    cluster_progress = tqdm(total=len(all_batches), desc="Clusters processed", position=0)

    # ----------------------------------------------------
    # Worker
    # ----------------------------------------------------
    def engine_worker(engine_index: int):
        client = engine_clients[engine_index]
        while True:
            # Check if this engine is marked as down
            if health_manager.is_engine_down(engine_index):
                print(f"Engine {engine_index}: Marked as down, stopping worker")
                break
            
            # Try to get work from either main queue or failed batches queue
            batch_idx = None
            try:
                # First priority: failed batches that need redistribution
                batch_idx = failed_batches_queue.get_nowait()
            except queue.Empty:
                try:
                    # Second priority: new batches
                    batch_idx = work_queue.get_nowait()
                except queue.Empty:
                    # No work available, but check if there are still healthy engines working
                    healthy_engines = health_manager.get_healthy_engines(num_engines)
                    if not healthy_engines or engine_index not in healthy_engines:
                        break
                    # Wait a bit and try again
                    time.sleep(1)
                    continue

            reqs = all_batches[batch_idx]
            success = False
            try:
                results = process_generation_batch_on_engine(client, reqs, engine_urls[engine_index])
                if results:  # Check if we got valid results
                    success = True
                    health_manager.record_success(engine_index)
                else:
                    raise Exception("No results returned from batch processing")
                    
            except Exception as e:
                print(f"Engine {engine_index}: exception during batch {batch_idx}: {e}")
                results = []
                
                # Record failure and check if engine should be marked as down
                engine_marked_down = health_manager.record_failure(engine_index)
                
                if not engine_marked_down:
                    # Engine not marked as down yet, redistribute this batch to another engine
                    print(f"Engine {engine_index}: Redistributing failed batch {batch_idx} (failure #{health_manager.get_failure_count(engine_index)})")
                    failed_batches_queue.put(batch_idx)
                else:
                    # Engine marked as down, redistribute this batch and break
                    print(f"Engine {engine_index}: Redistributing failed batch {batch_idx} before shutting down")
                    failed_batches_queue.put(batch_idx)
                    break
            finally:
                # Only advance progress bar if we successfully processed the batch
                if success:
                    with progress_lock:
                        cluster_progress.update(1)

            if not success:
                continue  # failed batch already redistributed

            # Process & append – skip any malformed result safely
            for res in results:
                try:
                    custom_id = res.get("custom_id", "missing_id")

                    # Robustly parse indices from custom_id (expects pattern gen_doc_#_cluster_#_q_#)
                    doc_idx = cluster_idx = q_idx = -1
                    parts = custom_id.split("_")
                    if len(parts) >= 7:
                        doc_idx = int(parts[2])
                        cluster_idx = int(parts[4])
                        q_idx = int(parts[6])

                    # Navigate nested response dict defensively
                    content = (
                        res.get("response", {})
                        .get("body", {})
                        .get("choices", {})
                        .get("message", {})
                        .get("content", "")
                    )

                    if not content:
                        raise ValueError("Empty content in model response")

                    qa_pair = extract_json_from_response(content)

                    record = {
                        "doc_idx": doc_idx,
                        "cluster_idx": cluster_idx,
                        "q_idx": q_idx,
                        **qa_pair,
                    }

                except Exception as e:
                    # Skip this result but report; do NOT crash the worker
                    print(f"Skipping malformed result ({custom_id}): {e}")
                    continue

                # Append to output file thread-safely
                with save_lock, open(args.output_data_path, "a") as out_f:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ----------------------------------------------------
    # Launch threads
    # ----------------------------------------------------
    with ThreadPoolExecutor(max_workers=num_engines) as ex:
        futures = [ex.submit(engine_worker, idx) for idx in range(num_engines)]
        
        # Monitor progress and check if all engines are down
        while True:
            # Check if all engines are done (either completed work or marked as down)
            active_futures = [f for f in futures if not f.done()]
            if not active_futures:
                break
                
            # Check if there's still work to do but all engines are down
            remaining_work = work_queue.qsize() + failed_batches_queue.qsize()
            healthy_engines = health_manager.get_healthy_engines(num_engines)
            
            if remaining_work > 0 and not healthy_engines:
                print(f"WARNING: All engines are down but {remaining_work} batches remain unprocessed!")
                break
            
            time.sleep(5)  # Check every 5 seconds
        
        # Wait for all futures to complete
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Worker thread completed with exception: {e}")

    # close progress bar after worker threads complete
    cluster_progress.close()

    # Report final status
    healthy_engines = health_manager.get_healthy_engines(num_engines)
    remaining_work = work_queue.qsize() + failed_batches_queue.qsize()
    
    print(f"Generation finished – results stored incrementally in {args.output_data_path}")
    print(f"Final status: {len(healthy_engines)}/{num_engines} engines healthy, {remaining_work} batches remaining")
    
    if remaining_work > 0:
        print(f"WARNING: {remaining_work} batches were not processed due to engine failures")

    # ----------------------------------------------------
    # Organize output into specified format
    # ----------------------------------------------------
    print("Organizing output into structured format...")
    organize_output_format(args.output_data_path, args.organized_output_path, documents)


if __name__ == "__main__":
    main() 
