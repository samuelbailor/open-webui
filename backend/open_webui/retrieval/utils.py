import logging
import os
import asyncio
import concurrent.futures
from typing import Optional, Union, List, Any

import requests
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import snapshot_download
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

from open_webui.config import VECTOR_DB
from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT

from open_webui.models.users import UserModel
from open_webui.models.files import Files

from open_webui.retrieval.vector.main import GetResult


from open_webui.env import (
    SRC_LOG_LEVELS,
    OFFLINE_MODE,
    ENABLE_FORWARD_USER_INFO_HEADERS,
)
from open_webui.config import (
    RAG_EMBEDDING_QUERY_PREFIX,
    RAG_EMBEDDING_CONTENT_PREFIX,
    RAG_EMBEDDING_PREFIX_FIELD_NAME,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

# Calculate these values once at module initialization
WORKER_COUNT = int(os.getenv("UVICORN_WORKERS", "1"))
SYSTEM_CORES = os.cpu_count() or 1

log.info(f"Initializing retrieval utils with {WORKER_COUNT} workers, {SYSTEM_CORES} cores")


class VectorSearchRetriever(BaseRetriever):
    collection_name: Any
    embedding_function: Any
    top_k: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        # Ensure collection name is formatted consistently with how it's stored
        formatted_collection_name = self.collection_name.replace("-", "_")
        
        result = VECTOR_DB_CLIENT.search(
            collection_name=formatted_collection_name,
            vectors=[self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)],
            limit=self.top_k,
        )

        ids = result.ids[0]
        metadatas = result.metadatas[0]
        documents = result.documents[0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results


def query_doc(
    collection_name: str, query_embedding: list[float], k: int, user: UserModel = None
):
    try:
        # Ensure collection name is formatted consistently with how it's stored
        # This matches the formatting in the vector db's insert method
        formatted_collection_name = collection_name.replace("-", "_")
        
        log.debug(f"query_doc:doc {formatted_collection_name} (original: {collection_name})")
        result = VECTOR_DB_CLIENT.search(
            collection_name=formatted_collection_name,
            vectors=[query_embedding],
            limit=k,
        )

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with limit {k}: {e}")
        raise e


def get_doc(collection_name: str, user: UserModel = None):
    try:
        # Ensure collection name is formatted consistently with how it's stored
        formatted_collection_name = collection_name.replace("-", "_")
        
        log.debug(f"get_doc:doc {formatted_collection_name} (original: {collection_name})")
        result = VECTOR_DB_CLIENT.get(collection_name=formatted_collection_name)

        if result:
            log.info(f"query_doc:result {result.ids} {result.metadatas}")

        return result
    except Exception as e:
        log.exception(f"Error getting doc {collection_name}: {e}")
        raise e


def query_doc_with_hybrid_search(
    collection_name: str,
    collection_result: GetResult,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
) -> dict:
    try:
        # Ensure collection name is formatted consistently with how it's stored
        formatted_collection_name = collection_name.replace("-", "_")
        
        log.debug(f"query_doc_with_hybrid_search:doc {formatted_collection_name} (original: {collection_name})")
        bm25_retriever = BM25Retriever.from_texts(
            texts=collection_result.documents[0],
            metadatas=collection_result.metadatas[0],
        )
        bm25_retriever.k = k

        vector_search_retriever = VectorSearchRetriever(
            collection_name=formatted_collection_name,
            embedding_function=embedding_function,
            top_k=k,
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_search_retriever], weights=[0.5, 0.5]
        )
        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=k_reranker,
            reranking_function=reranking_function,
            r_score=r,
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        result = compression_retriever.invoke(query)

        distances = [d.metadata.get("score") for d in result]
        documents = [d.page_content for d in result]
        metadatas = [d.metadata for d in result]

        # retrieve only min(k, k_reranker) items, sort and cut by distance if k < k_reranker
        if k < k_reranker:
            sorted_items = sorted(
                zip(distances, metadatas, documents), key=lambda x: x[0], reverse=True
            )
            sorted_items = sorted_items[:k]
            distances, documents, metadatas = map(list, zip(*sorted_items))

        result = {
            "distances": [distances],
            "documents": [documents],
            "metadatas": [metadatas],
        }

        log.info(
            "query_doc_with_hybrid_search:result "
            + f'{result["metadatas"]} {result["distances"]}'
        )
        return result
    except Exception as e:
        log.exception(f"Error querying doc {collection_name} with hybrid search: {e}")
        raise e


def merge_get_results(get_results: list[dict]) -> dict:
    # Initialize lists to store combined data
    combined_documents = []
    combined_metadatas = []
    combined_ids = []

    for data in get_results:
        combined_documents.extend(data["documents"][0])
        combined_metadatas.extend(data["metadatas"][0])
        combined_ids.extend(data["ids"][0])

    # Create the output dictionary
    result = {
        "documents": [combined_documents],
        "metadatas": [combined_metadatas],
        "ids": [combined_ids],
    }

    return result


def merge_and_sort_query_results(query_results: list[dict], k: int) -> dict:
    # Initialize lists to store combined data
    combined = dict()  # To store documents with unique document hashes

    for data in query_results:
        distances = data["distances"][0]
        documents = data["documents"][0]
        metadatas = data["metadatas"][0]

        for distance, document, metadata in zip(distances, documents, metadatas):
            if isinstance(document, str):
                doc_hash = hashlib.sha256(
                    document.encode()
                ).hexdigest()  # Compute a hash for uniqueness

                if doc_hash not in combined.keys():
                    combined[doc_hash] = (distance, document, metadata)
                    continue  # if doc is new, no further comparison is needed

                # if doc is alredy in, but new distance is better, update
                if distance > combined[doc_hash][0]:
                    combined[doc_hash] = (distance, document, metadata)

    combined = list(combined.values())
    # Sort the list based on distances
    combined.sort(key=lambda x: x[0], reverse=True)

    # Slice to keep only the top k elements
    sorted_distances, sorted_documents, sorted_metadatas = (
        zip(*combined[:k]) if combined else ([], [], [])
    )

    # Create and return the output dictionary
    return {
        "distances": [list(sorted_distances)],
        "documents": [list(sorted_documents)],
        "metadatas": [list(sorted_metadatas)],
    }


def get_all_items_from_collections(collection_names: list[str]) -> dict:
    results = []

    for collection_name in collection_names:
        if collection_name:
            try:
                result = get_doc(collection_name=collection_name)
                if result is not None:
                    results.append(result.model_dump())
            except Exception as e:
                log.exception(f"Error when querying the collection: {e}")
        else:
            pass

    return merge_get_results(results)


def query_collection(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
) -> dict:
    results = []
    error = False

    def process_query_collection(collection_name, query_embedding):
        try:
            if collection_name:
                result = query_doc(
                    collection_name=collection_name,
                    k=k,
                    query_embedding=query_embedding,
                )
                if result is not None:
                    return result.model_dump(), None
            return None, None
        except Exception as e:
            log.exception(f"Error when querying the collection: {e}")
            return None, e

    # Generate all query embeddings (in one call)
    query_embeddings = embedding_function(queries, prefix=RAG_EMBEDDING_QUERY_PREFIX)
    log.debug(
        f"query_collection: processing {len(queries)} queries across {len(collection_names)} collections"
    )

    with ThreadPoolExecutor() as executor:
        future_results = []
        for query_embedding in query_embeddings:
            for collection_name in collection_names:
                result = executor.submit(
                    process_query_collection, collection_name, query_embedding
                )
                future_results.append(result)
        task_results = [future.result() for future in future_results]

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    if error and not results:
        log.warning("All collection queries failed. No results returned.")

    return merge_and_sort_query_results(results, k=k)


def query_collection_with_hybrid_search(
    collection_names: list[str],
    queries: list[str],
    embedding_function,
    k: int,
    reranking_function,
    k_reranker: int,
    r: float,
) -> dict:
    results = []
    error = False
    # Fetch collection data once per collection sequentially
    # Avoid fetching the same data multiple times later
    collection_results = {}
    for collection_name in collection_names:
        try:
            log.debug(
                f"query_collection_with_hybrid_search:VECTOR_DB_CLIENT.get:collection {collection_name}"
            )
            # Ensure collection name is formatted consistently with how it's stored
            formatted_collection_name = collection_name.replace("-", "_")
            
            log.debug(f"Fetching collection {formatted_collection_name} (original: {collection_name})")
            collection_results[collection_name] = VECTOR_DB_CLIENT.get(
                collection_name=formatted_collection_name
            )
        except Exception as e:
            log.exception(f"Failed to fetch collection {collection_name}: {e}")
            collection_results[collection_name] = None

    log.info(
        f"Starting hybrid search for {len(queries)} queries in {len(collection_names)} collections..."
    )

    def process_query(collection_name, query):
        try:
            result = query_doc_with_hybrid_search(
                collection_name=collection_name,
                collection_result=collection_results[collection_name],
                query=query,
                embedding_function=embedding_function,
                k=k,
                reranking_function=reranking_function,
                k_reranker=k_reranker,
                r=r,
            )
            return result, None
        except Exception as e:
            log.exception(f"Error when querying the collection with hybrid_search: {e}")
            return None, e

    # Prepare tasks for all collections and queries
    # Avoid running any tasks for collections that failed to fetch data (have assigned None)
    tasks = [
        (cn, q)
        for cn in collection_names
        if collection_results[cn] is not None
        for q in queries
    ]

    with ThreadPoolExecutor() as executor:
        future_results = [executor.submit(process_query, cn, q) for cn, q in tasks]
        task_results = [future.result() for future in future_results]

    for result, err in task_results:
        if err is not None:
            error = True
        elif result is not None:
            results.append(result)

    if error and not results:
        raise Exception(
            "Hybrid search failed for all collections. Using Non-hybrid search as fallback."
        )

    return merge_and_sort_query_results(results, k=k)

def calculate_resource_allocation(worker_count: int = WORKER_COUNT, system_cores: int = SYSTEM_CORES) -> dict:
    """
    Calculate optimal resource allocation based on system resources and worker count.
    
    This function determines three key parameters for concurrent processing:
    
    1. concurrency: How many batches can be processed simultaneously.
       This is dynamically calculated based on available CPU cores, but capped
       between 2-6 to prevent system overload.
    
    2. thread_pool_size: How many threads to allocate in the thread pool.
       This is typically 2-3x the concurrency value to ensure efficient thread
       utilization while preventing excessive context switching.
       
    3. concurrency_batch_size: The size of each batch, i.e., how many texts to include 
       in each batch. This is dynamically calculated based on available CPU cores and memory.
       Larger systems with more cores can efficiently process larger batches.
    
    Args:
        worker_count: Number of worker processes (defaults to module-level WORKER_COUNT)
        system_cores: Number of CPU cores available (defaults to module-level SYSTEM_CORES)
        
    Returns:
        Dictionary with concurrency, thread_pool_size, and concurrency_batch_size values
    """
    # Calculate cores per worker (minimum 1)
    cores_per_worker = max(1, system_cores // worker_count)
    
    # Calculate concurrency - balance between throughput and resource contention
    # This determines how many batches can be processed simultaneously
    # We use a balanced approach that works well for both I/O and CPU bound operations
    # Limited to between 2 and 6 concurrent batches regardless of system size
    concurrency = max(2, min(cores_per_worker + 1, 6))
    
    # Calculate thread pool size - typically 2-3x concurrency is a good balance
    # but capped based on available cores
    # This determines the size of the thread pool used for processing
    thread_pool_size = max(5, min(concurrency * 2, cores_per_worker * 3))
    
    # Calculate optimal concurrency batch size based on system resources
    # The formula scales batch size with available cores, with reasonable min/max bounds
    # - Base size of 16 for single-core systems
    # - Scales up with more cores available per worker
    # - Caps at 64 to prevent excessive memory usage
    # This approach balances efficiency (larger batches) with resource constraints
    concurrency_batch_size = max(16, min(16 * cores_per_worker, 64))
    
    log.debug(f"Calculated concurrency_batch_size={concurrency_batch_size} based on {cores_per_worker} cores per worker")
    
    return {
        "concurrency": concurrency,
        "thread_pool_size": thread_pool_size,
        "concurrency_batch_size": concurrency_batch_size
    }

# Pre-calculate default resource allocation
DEFAULT_RESOURCE_ALLOCATION = calculate_resource_allocation()
log.info(f"Default resource allocation: concurrency={DEFAULT_RESOURCE_ALLOCATION['concurrency']}, "
         f"thread_pool_size={DEFAULT_RESOURCE_ALLOCATION['thread_pool_size']}")

async def process_embeddings_async(
    embedding_function,
    texts: List[str],
    prefix: Optional[str] = None,
    user: Optional[Any] = None,
    concurrency_batch_size: Optional[int] = None,
    max_concurrent: Optional[int] = None
) -> List[List[float]]:
    """
    Process embeddings asynchronously with controlled concurrency.
    
    This function implements two levels of batching for optimal performance:
    
    1. Concurrency Batches: The input texts are divided into batches of size `concurrency_batch_size`.
       Each batch is processed as a single unit by one thread. For example, with 320 texts and
       concurrency_batch_size=32, we create 10 separate batches.
       
       The concurrency_batch_size is dynamically calculated based on system resources:
       - Scales with available CPU cores per worker
       - Ranges from 16 (minimum) to 64 (maximum)
       - Larger systems with more cores use larger batch sizes for efficiency
    
    2. Concurrent Execution: We process multiple batches simultaneously, controlled by `max_concurrent`.
       This limits how many batches can be in-flight at once. For example, with max_concurrent=4,
       only 4 batches (out of our 10) would be processed simultaneously.
    
    This two-level approach allows for efficient resource utilization:
    - `concurrency_batch_size` optimizes the workload size for each thread
    - `max_concurrent` prevents overloading the system with too many parallel operations
    
    Note: This is different from "API request batches" which are created in generate_multiple
    and determine how many texts are sent in a single HTTP request to the embedding API.
    
    Args:
        embedding_function: The function to generate embeddings
        texts: List of texts to embed
        prefix: Embedding prefix
        user: User information
        concurrency_batch_size: Size of each concurrency batch (defaults to dynamically calculated value)
        max_concurrent: Maximum number of concurrent batches (defaults to pre-calculated value)
    """
    # If parameters are not provided, use the pre-calculated values
    if concurrency_batch_size is None:
        # This determines how many texts are in each batch processed by a single thread
        # Dynamically calculated based on system resources
        concurrency_batch_size = DEFAULT_RESOURCE_ALLOCATION["concurrency_batch_size"]
        log.debug(f"Using dynamically calculated concurrency_batch_size: {concurrency_batch_size}")
    
    if max_concurrent is None:
        # This determines how many batches can be processed simultaneously
        # Dynamically calculated based on system resources (typically 2-6)
        max_concurrent = DEFAULT_RESOURCE_ALLOCATION["concurrency"]
        log.debug(f"Using default max_concurrent: {max_concurrent}")
    
    # Create a semaphore to limit concurrent operations
    # This ensures no more than max_concurrent batches are being processed at any given time
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Use the pre-calculated thread pool size if not specified
    thread_pool_size = DEFAULT_RESOURCE_ALLOCATION["thread_pool_size"]
    log.debug(f"Using thread pool size: {thread_pool_size}")
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size)
    
    async def process_batch(batch):
        # Acquire semaphore to limit concurrency - this blocks if max_concurrent batches are already running
        # Once a running batch completes, the semaphore is released and a new batch can start
        async with semaphore:
            # Use thread pool for better performance
            start_time = time.time()
            # This is where the batch gets sent to the embedding function
            # The embedding function may further batch the texts for API requests
            result = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: embedding_function(batch, prefix, user)
            )
            elapsed_time = time.time() - start_time
            log.info(f"Batch of {len(batch)} texts processed in {elapsed_time:.4f} seconds")
            return result
    
    # Split texts into concurrency batches (processed by different threads)
    # Each batch contains concurrency_batch_size texts (or fewer for the last batch)
    concurrency_batches = []
    for i in range(0, len(texts), concurrency_batch_size):
        concurrency_batches.append(texts[i:i + concurrency_batch_size])
    
    log.info(f"Processing {len(texts)} texts in {len(concurrency_batches)} concurrency batches")
    log.info(f"Each batch contains up to {concurrency_batch_size} texts, with max {max_concurrent} batches running concurrently")
    
    # Create tasks for all concurrency batches
    tasks = [process_batch(batch) for batch in concurrency_batches]
    
    # Execute all tasks concurrently and gather results
    try:
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results from all batches
        all_embeddings = []
        for result in batch_results:
            if isinstance(result, list):
                all_embeddings.extend(result)
        
        return all_embeddings
    finally:
        # Make sure to shut down the thread pool
        thread_pool.shutdown(wait=False)

def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    url,
    key,
    api_request_batch_size,
):
    """
    Creates an embedding function based on the specified engine.
    
    For Ollama/OpenAI engines, this wraps the embedding function with generate_multiple
    which handles API request batching - grouping texts into batches for efficient API calls.
    
    Args:
        embedding_engine: The embedding engine to use (e.g., "", "ollama", "openai")
        embedding_model: The model name to use for embeddings
        embedding_function: The base embedding function (for non-API engines)
        url: The API URL for API-based engines
        key: The API key for API-based engines
        api_request_batch_size: Number of texts to include in a single API request batch
    """
    if embedding_engine == "":
        return lambda query, prefix=None, user=None: embedding_function.encode(
            query, **({"prompt": prefix} if prefix else {})
        ).tolist()
    elif embedding_engine in ["ollama", "openai"]:
        func = lambda query, prefix=None, user=None: generate_embeddings(
            engine=embedding_engine,
            model=embedding_model,
            text=query,
            prefix=prefix,
            url=url,
            key=key,
            user=user,
        )

        def generate_multiple(query, prefix, user, func):
            """
            Handles API request batching for embedding generation.

            The number of texts sent in a single HHTP request to the embedding API is controlled 
            by api_request_batch_size (formerly embedding_batch_size), which is set in the UI.
            This function splits the input query into smaller batches of size api_request_batch_size.
            """
            if isinstance(query, list):
                embeddings = []
                # Split the query into API request batches based on api_request_batch_size
                for i in range(0, len(query), api_request_batch_size):
                    # Create an API request batch
                    api_request_batch = query[i : i + api_request_batch_size]
                    log.debug(f"Processing API request batch of size {len(api_request_batch)}")
                    
                    # Send the API request batch to the embedding API as a single HTTP request
                    # This is more efficient than sending each text individually
                    embeddings.extend(
                        func(
                            api_request_batch,
                            prefix=prefix,
                            user=user,
                        )
                    )
                return embeddings
            else:
                # Single text, no batching needed
                return func(query, prefix, user)

        return lambda query, prefix=None, user=None: generate_multiple(
            query, prefix, user, func
        )
    else:
        raise ValueError(f"Unknown embedding engine: {embedding_engine}")


def get_sources_from_files(
    request,
    files,
    queries,
    embedding_function,
    k,
    reranking_function,
    k_reranker,
    r,
    hybrid_search,
    full_context=False,
):
    log.debug(
        f"files: {files} {queries} {embedding_function} {reranking_function} {full_context}"
    )

    extracted_collections = []
    relevant_contexts = []

    for file in files:

        context = None
        if file.get("docs"):
            # BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
            context = {
                "documents": [[doc.get("content") for doc in file.get("docs")]],
                "metadatas": [[doc.get("metadata") for doc in file.get("docs")]],
            }
        elif file.get("context") == "full":
            # Manual Full Mode Toggle
            context = {
                "documents": [[file.get("file").get("data", {}).get("content")]],
                "metadatas": [[{"file_id": file.get("id"), "name": file.get("name")}]],
            }
        elif (
            file.get("type") != "web_search"
            and request.app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL
        ):
            # BYPASS_EMBEDDING_AND_RETRIEVAL
            if file.get("type") == "collection":
                file_ids = file.get("data", {}).get("file_ids", [])

                documents = []
                metadatas = []
                for file_id in file_ids:
                    file_object = Files.get_file_by_id(file_id)

                    if file_object:
                        documents.append(file_object.data.get("content", ""))
                        metadatas.append(
                            {
                                "file_id": file_id,
                                "name": file_object.filename,
                                "source": file_object.filename,
                            }
                        )

                context = {
                    "documents": [documents],
                    "metadatas": [metadatas],
                }

            elif file.get("id"):
                file_object = Files.get_file_by_id(file.get("id"))
                if file_object:
                    context = {
                        "documents": [[file_object.data.get("content", "")]],
                        "metadatas": [
                            [
                                {
                                    "file_id": file.get("id"),
                                    "name": file_object.filename,
                                    "source": file_object.filename,
                                }
                            ]
                        ],
                    }
            elif file.get("file").get("data"):
                context = {
                    "documents": [[file.get("file").get("data", {}).get("content")]],
                    "metadatas": [
                        [file.get("file").get("data", {}).get("metadata", {})]
                    ],
                }
        else:
            collection_names = []
            if file.get("type") == "collection":
                if file.get("legacy"):
                    collection_names = file.get("collection_names", [])
                else:
                    collection_names.append(file["id"])
            elif file.get("collection_name"):
                collection_names.append(file["collection_name"])
            elif file.get("id"):
                if file.get("legacy"):
                    collection_names.append(f"{file['id']}")
                else:
                    collection_names.append(f"file-{file['id']}")

            collection_names = set(collection_names).difference(extracted_collections)
            if not collection_names:
                log.debug(f"skipping {file} as it has already been extracted")
                continue

            if full_context:
                try:
                    context = get_all_items_from_collections(collection_names)
                except Exception as e:
                    log.exception(e)

            else:
                try:
                    context = None
                    if file.get("type") == "text":
                        context = file["content"]
                    else:
                        if hybrid_search:
                            try:
                                context = query_collection_with_hybrid_search(
                                    collection_names=collection_names,
                                    queries=queries,
                                    embedding_function=embedding_function,
                                    k=k,
                                    reranking_function=reranking_function,
                                    k_reranker=k_reranker,
                                    r=r,
                                )
                            except Exception as e:
                                log.debug(
                                    "Error when using hybrid search, using"
                                    " non hybrid search as fallback."
                                )

                        if (not hybrid_search) or (context is None):
                            context = query_collection(
                                collection_names=collection_names,
                                queries=queries,
                                embedding_function=embedding_function,
                                k=k,
                            )
                except Exception as e:
                    log.exception(e)

            extracted_collections.extend(collection_names)

        if context:
            if "data" in file:
                del file["data"]

            relevant_contexts.append({**context, "file": file})

    sources = []
    for context in relevant_contexts:
        try:
            if "documents" in context:
                if "metadatas" in context:
                    source = {
                        "source": context["file"],
                        "document": context["documents"][0],
                        "metadata": context["metadatas"][0],
                    }
                    if "distances" in context and context["distances"]:
                        source["distances"] = context["distances"][0]

                    sources.append(source)
        except Exception as e:
            log.exception(e)

    return sources


def get_model_path(model: str, update_model: bool = False):
    # Construct huggingface_hub kwargs with local_files_only to return the snapshot path
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model

    if OFFLINE_MODE:
        local_files_only = True

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    # Inspiration from upstream sentence_transformers
    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        # If fully qualified path exists, return input, else set repo_id
        return model
    elif "/" not in model:
        # Set valid repo_id for model short-name
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    # Attempt to query the huggingface_hub library to determine the local path and/or to update
    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model


def generate_openai_batch_embeddings(
    model: str,
    texts: list[str],
    url: str = "https://api.openai.com/v1",
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    """
    Generate embeddings for a batch of texts using OpenAI API in a single HTTP request.
    
    This function sends an API request batch (multiple texts) to the OpenAI API
    in a single HTTP request for efficient processing.
    
    Args:
        model: The model name to use for embeddings
        texts: List of text strings (API request batch)
        url: The OpenAI API URL
        key: The OpenAI API key
        prefix: Optional prefix for the texts
        user: Optional user information
    """
    try:
        log.debug(
            f"generate_openai_batch_embeddings: model={model}, API request batch size={len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        r = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                **(
                    {
                        "X-OpenWebUI-User-Name": user.name,
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS and user
                    else {}
                ),
            },
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return [elem["embedding"] for elem in data["data"]]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(f"Error generating openai batch embeddings: {e}")
        return None


def generate_ollama_batch_embeddings(
    model: str,
    texts: list[str],
    url: str,
    key: str = "",
    prefix: str = None,
    user: UserModel = None,
) -> Optional[list[list[float]]]:
    """
    Generate embeddings for a batch of texts using Ollama API in a single HTTP request.
    
    This function sends an API request batch (multiple texts) to the Ollama API
    in a single HTTP request for efficient processing.
    
    Args:
        model: The model name to use for embeddings
        texts: List of text strings (API request batch)
        url: The Ollama API URL
        key: The Ollama API key
        prefix: Optional prefix for the texts
        user: Optional user information
    """
    try:
        log.debug(
            f"generate_ollama_batch_embeddings: model={model}, API request batch size={len(texts)}"
        )
        json_data = {"input": texts, "model": model}
        if isinstance(RAG_EMBEDDING_PREFIX_FIELD_NAME, str) and isinstance(prefix, str):
            json_data[RAG_EMBEDDING_PREFIX_FIELD_NAME] = prefix

        r = requests.post(
            f"{url}/api/embed",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
                **(
                    {
                        "X-OpenWebUI-User-Name": user.name,
                        "X-OpenWebUI-User-Id": user.id,
                        "X-OpenWebUI-User-Email": user.email,
                        "X-OpenWebUI-User-Role": user.role,
                    }
                    if ENABLE_FORWARD_USER_INFO_HEADERS
                    else {}
                ),
            },
            json=json_data,
        )
        r.raise_for_status()
        data = r.json()

        if "embeddings" in data:
            return data["embeddings"]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        log.exception(f"Error generating ollama batch embeddings: {e}")
        return None


def generate_embeddings(
    engine: str,
    model: str,
    text: Union[str, list[str]],
    prefix: Union[str, None] = None,
    **kwargs,
):
    """
    Generate embeddings for text using the specified engine and model.
    
    This function handles both single texts and batches of texts (API request batches).
    When text is a list, it represents an API request batch - multiple texts to be
    embedded in a single API call.
    
    Args:
        engine: The embedding engine to use (e.g., "ollama", "openai")
        model: The model name to use for embeddings
        text: Either a single text string or a list of text strings (API request batch)
        prefix: Optional prefix to add to the text
        **kwargs: Additional arguments including url, key, and user
    """
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")
    user = kwargs.get("user")

    # Log whether we're processing a single text or an API request batch
    if isinstance(text, list):
        log.debug(f"generate_embeddings: processing API request batch of {len(text)} texts with prefix={prefix}")
    else:
        log.debug(f"generate_embeddings: processing single text with prefix={prefix}")

    if prefix is not None and RAG_EMBEDDING_PREFIX_FIELD_NAME is None:
        if isinstance(text, list):
            text = [f"{prefix}{text_element}" for text_element in text]
        else:
            text = f"{prefix}{text}"

    if engine == "ollama":
        if isinstance(text, list):
            embeddings = generate_ollama_batch_embeddings(
                **{
                    "model": model,
                    "texts": text,
                    "url": url,
                    "key": key,
                    "prefix": prefix,
                    "user": user,
                }
            )
        else:
            embeddings = generate_ollama_batch_embeddings(
                **{
                    "model": model,
                    "texts": [text],
                    "url": url,
                    "key": key,
                    "prefix": prefix,
                    "user": user,
                }
            )
        return embeddings[0] if isinstance(text, str) else embeddings
    elif engine == "openai":
        if isinstance(text, list):
            embeddings = generate_openai_batch_embeddings(
                model, text, url, key, prefix, user
            )
        else:
            embeddings = generate_openai_batch_embeddings(
                model, [text], url, key, prefix, user
            )
        return embeddings[0] if isinstance(text, str) else embeddings


import operator
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document


class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        if reranking:
            scores = self.reranking_function.predict(
                [(query, doc.page_content) for doc in documents]
            )
        else:
            from sentence_transformers import util

            query_embedding = self.embedding_function(query, RAG_EMBEDDING_QUERY_PREFIX)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents], RAG_EMBEDDING_CONTENT_PREFIX
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        docs_with_scores = list(
            zip(documents, scores.tolist() if not isinstance(scores, list) else scores)
        )
        if self.r_score:
            docs_with_scores = [
                (d, s) for d, s in docs_with_scores if s >= self.r_score
            ]

        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        final_results = []
        for doc, doc_score in result[: self.top_n]:
            metadata = doc.metadata
            metadata["score"] = doc_score
            doc = Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
            final_results.append(doc)
        return final_results
