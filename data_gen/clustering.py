'''
    I wanna build a pipeline to gather the similar sentences within a long documents. The pipeline contains:
    (1) split the document into chunks
    (2) embed the chunks
    (3) gather the chunks with similar embeddings by adaptive clustring
    (4) return the gathered chunks
'''

# from rag import split_into_chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
# from hdbscan import HDBSCAN
from sklearn.cluster import HDBSCAN
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os
import json
from tqdm import tqdm
import itertools
from transformers import AutoTokenizer


def split_into_chunks(document, tokenizer, max_tokens=128):
    """
    Splits a document into chunks using Langchain's text splitter, 
    ensuring each chunk does not exceed the maximum number of tokens.
    
    Parameters:
    - document: str, the input text to be split.
    - tokenizer: a tokenizer object from the transformers library.
    - max_tokens: int, the maximum number of tokens per chunk.
    
    Returns:
    - chunks: list of str, each being a chunk of text that respects the token limit.
    """
    
    # Initialize Langchain's RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        separators=[
            # English separators
            "\n\n", "\n", ". ", "! ", "? ", "; ", ", ",
            # Chinese separators
            "。", "！", "？", "；", "，",
        ],
        chunk_size=max_tokens,    # Start with a reasonable chunk size
        chunk_overlap=0   # Slight overlap to maintain context
    )
    
    # Perform an initial split into chunks
    initial_chunks = text_splitter.split_text(document)
    chunks_with_positions = []
    current_position = 0
    for chunk in initial_chunks:
        # Find the start position of this chunk in the original text
        chunk_position = document.find(chunk, current_position)
        chunks_with_positions.append({
            'text': chunk,
            'position': chunk_position
        })
        current_position = chunk_position + 1
    
    return chunks_with_positions

def embed_chunks(chunks_with_positions: List[Dict[str, Any]], embed_model: SentenceTransformer) -> np.ndarray:
    """
    Embed the chunks using a sentence transformer model.
    
    Args:
        chunks_with_positions: List of dictionaries containing chunk text and position
        embed_model: Initialized sentence transformer model
    
    Returns:
        numpy array of embeddings
    """
    chunks_text = [chunk['text'] for chunk in chunks_with_positions]
    embeddings = embed_model.encode(chunks_text, normalize_embeddings=True, show_progress_bar=False, device='cuda')
    return embeddings

def adaptive_clustering(embeddings: np.ndarray, min_similarity: float = 0.6) -> List[int]:
    """
    Perform adaptive clustering on embeddings using HDBSCAN.
    
    Args:
        embeddings: numpy array of embeddings (should be normalized)
        min_similarity: minimum similarity threshold for clustering (cosine similarity)
    
    Returns:
        List of cluster labels for each embedding
    """
    # Convert similarity threshold to distance threshold for cosine distance
    # For cosine distance: distance = 1 - similarity
    distance_threshold = 1 - min_similarity
    
    # Initialize HDBSCAN
    clustering = HDBSCAN(
        min_cluster_size=4,  # Minimum cluster size
        min_samples=2,
        metric='cosine',
        cluster_selection_epsilon=distance_threshold,
        cluster_selection_method='eom',
        n_jobs=8
    )
    
    # Perform clustering
    cluster_labels = clustering.fit_predict(embeddings)
    return cluster_labels

def process_single_document(args):
    """
    Process a single document with the given parameters.
    This function will be run in parallel.
    """
    document, tokenizer, max_tokens = args
    # max_tokens = 512
    # token_length = len(tokenizer.encode(document['text']))
    # max_tokens = token_length // 64
    # Step 1: Split document into chunks
    chunks_with_positions = split_into_chunks(document['text'], tokenizer, max_tokens=max_tokens)
    
    return chunks_with_positions

def gather_similar_chunks_parallel(documents: List[str], 
                                 tokenizer: Any,
                                 embed_model_path: str,
                                 max_tokens: int = 128,
                                 min_similarity: float = 0.6,
                                 num_workers: int = None,
                                 batch_size: int = 2048) -> List[List[List[Dict[str, Any]]]]:
    """
    Main pipeline to gather similar chunks from multiple documents in parallel.
    
    Args:
        documents: List of input text documents
        tokenizer: Tokenizer for splitting text
        embed_model_path: Path to the sentence transformer model
        max_tokens: Maximum tokens per chunk
        min_similarity: Minimum similarity threshold for clustering
        num_workers: Number of parallel workers (defaults to CPU count)
        batch_size: Batch size for embedding processing
    
    Returns:
        List of results, where each result is a list of lists containing similar chunks
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    
    # Step 1: Split documents into chunks in parallel (CPU operation)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(doc, tokenizer, max_tokens) for doc in documents]
        all_chunks = []
        
        # Process document splitting with error handling
        futures = []
        for args in args_list:
            future = executor.submit(process_single_document, args)
            futures.append(future)
        
        # Collect results with progress bar
        for future in tqdm(futures, total=len(futures), desc="Splitting documents"):
            try:
                chunks = future.result()
                all_chunks.append(chunks)
            except Exception as e:
                print(f"Error processing document: {str(e)}")
                all_chunks.append([])  # Add empty list for failed documents
    
    # Step 2: Embed all chunks (CUDA operation - single process)
    # Initialize the embedding model
    embedding_model = SentenceTransformer(embed_model_path)
    
    all_embeddings = []
    all_chunks_flat = list(itertools.chain.from_iterable(all_chunks))
    embeddings_list = []
    
    # Process all chunks in batches
    for i in range(0, len(all_chunks_flat), batch_size):
        batch = all_chunks_flat[i:i + batch_size]
        try:
            batch_embeddings = embed_chunks(batch, embedding_model)
            embeddings_list.append(batch_embeddings)
        except Exception as e:
            print(f"Error embedding batch {i//batch_size}: {str(e)}")
            # Add zero embeddings for failed batch
            embeddings_list.append(np.zeros((len(batch), embedding_model.get_sentence_embedding_dimension())))
    
    # Concatenate all embeddings
    all_embeddings_flat = np.concatenate(embeddings_list) if len(embeddings_list) > 1 else embeddings_list[0]
    # print(f"all_embeddings_flat: {all_embeddings_flat.shape}")
    # Split embeddings back per document
    start_idx = 0
    for chunks in all_chunks:
        end_idx = start_idx + len(chunks)
        all_embeddings.append(all_embeddings_flat[start_idx:end_idx])
        start_idx = end_idx
    
    assert len(all_embeddings) == len(documents)
    # Step 3: Process clustering in parallel (CPU operation)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        args_list = [(chunks, embeddings, min_similarity, document) for chunks, embeddings, document in zip(all_chunks, all_embeddings, documents)]
        results = []
        
        # Process clustering in parallel with error handling
        futures = []
        for args in args_list:
            future = executor.submit(process_doc_chunks, args)
            futures.append(future)
        
        # Collect results with progress bar
        for future in tqdm(futures, total=len(futures), desc="Processing document chunks"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error processing document chunks: {str(e)}")
    
    return results

def process_doc_chunks(args):
    """
    Process chunks and embeddings for a single document.
    This function will be run in parallel.
    """
    chunks, embeddings, min_similarity, document = args
    
    # Skip if no chunks or embeddings
    if len(chunks) == 0 or len(embeddings) == 0:
        return None
    
    try:
        # Perform adaptive clustering
        cluster_labels = adaptive_clustering(embeddings, min_similarity)
        
        # Group chunks by cluster
        clustered_chunks = {}
        for chunk, label in zip(chunks, cluster_labels):
            if label not in clustered_chunks:
                clustered_chunks[label] = []
            clustered_chunks[label].append(chunk)
        
        # Convert to list of lists, excluding noise points and small clusters
        document_result = []
        for label, chunks in clustered_chunks.items():
            if label != -1 and len(chunks) >= 4 and len(chunks) <= 16:
                sorted_chunks = sorted(chunks, key=lambda x: x['position'])
                # document_result.append(sorted_chunks)
                # document_result.append(sorted_chunks)
                document_result.append({
                    'id': document['id'],
                    'chunks': sorted_chunks
                })
        
        
        return document_result if len(document_result) >= 4 else None
    except Exception as e:
        print(f"Error in process_doc_chunks: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gather similar sentences within long documents.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save clustered results')
    parser.add_argument('--world_size', type=int, default=16, help='world size')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--tokenizer_path', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='Hugging Face model name or local path for the tokenizer')
    parser.add_argument('--embed_model_path', type=str, default="BAAI/bge-m3", help='SentenceTransformer model name or local path for embedding')
    parser.add_argument('--min_similarity', type=float, default=0.7, help='Minimum cosine similarity threshold for clustering')
    parser.add_argument('--max_tokens', type=int, default=512, help='Max tokens per chunk when splitting documents')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for embedding')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of CPU workers for splitting/clustering')
    args = parser.parse_args()

    # Load documents from JSON
    with open(args.input_path, 'r') as f:
        documents = []
        i = 0
        for line in tqdm(f, desc="Loading documents"):
            # if i > 128:
            #     break
            data_item = {
                'text': json.loads(line)['text'],
                "id": i
            }
            documents.append(data_item)
            i += 1
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Process documents in batches to handle large files
    # for i in range(0, len(documents), batch_size):
        # batch = documents[i:i+batch_size]
    batch = documents[args.rank::args.world_size]
    # Process current batch in parallel
    clustered_chunks_batch = gather_similar_chunks_parallel(
        batch,
        tokenizer, 
        args.embed_model_path,
        min_similarity=args.min_similarity,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Processed {len(clustered_chunks_batch)} documents")
    base, ext = os.path.splitext(args.output_path)
    if not ext:
        ext = ".jsonl"
    output_path = f"{base}_{args.rank}{ext}"
    # Save batch results to JSON file
    with open(output_path, 'a') as f:
        for item in clustered_chunks_batch:
            f.write(json.dumps(item))
            f.write('\n')
            
    # print(f"Processed {i+len(batch)}/{len(documents)} documents")
