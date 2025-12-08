import time
import os
from dotenv import load_dotenv

# Try to import app components to verify environment
try:
    from src.functions import load_rag_chain
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Import Error: {e}")
    IMPORTS_SUCCESS = False

def benchmark_latency():
    """
    Measures the latency of the RAG chain initialization and query response.
    Note: Requires valid API keys and Pinecone index.
    """
    results = {
        "initialization_time_sec": None,
        "query_latency_sec": None,
        "status": "Failed"
    }

    if not IMPORTS_SUCCESS:
        print("Skipping benchmark due to import errors.")
        return results

    load_dotenv()
    if not os.environ.get("Gemini_API") or not os.environ.get("Pinecode_API"):
         print("Missing API keys in .env file.")
         results["status"] = "Missing Keys"
         return results

    print("Starting Benchmark...")
    
    # Measure Initialization Time
    start_time = time.time()
    try:
        rag_chain = load_rag_chain()
        results["initialization_time_sec"] = round(time.time() - start_time, 4)
    except Exception as e:
        print(f"Initialization Failed: {e}")
        results["status"] = "Init Failed"
        return results

    # Measure Query Latency
    test_query = "What are the symptoms of flu?"
    print(f"Querying: '{test_query}'")
    start_time = time.time()
    try:
        response = rag_chain.invoke({"input": test_query})
        results["query_latency_sec"] = round(time.time() - start_time, 4)
        results["status"] = "Success"
        print(f"Response: {response.get('answer', 'No answer key in response')[:50]}...")
    except Exception as e:
        print(f"Query Failed: {e}")
        results["status"] = "Query Failed"

    return results

if __name__ == "__main__":
    metrics = benchmark_latency()
    print("\n--- Benchmark Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
