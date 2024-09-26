import os
import backoff
import logging
import time
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from snowflake.snowpark import Session
from snowflake.core.exceptions import APIError
from snowflake.core import Root
from dotenv import load_dotenv
from tqdm import tqdm

# Constants
MAX_WORKERS = 20
NUM_QUERIES = 200

warnings.filterwarnings("ignore", category=UserWarning, message="Pandas requires version '1.3.6' or newer of 'bottleneck'")
logging.basicConfig(level=logging.ERROR)

def make_session():
    """Create a Snowflake session using environment variables."""
    load_dotenv("./.env")

    connection_params = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    }
    
    password = os.getenv("SNOWFLAKE_PASSWORD")
    authenticator = os.getenv("SNOWFLAKE_AUTHENTICATOR")

    if password:
        connection_params["password"] = password
    if authenticator:
        connection_params["authenticator"] = authenticator

    return Session.builder.configs(connection_params).create()

@backoff.on_exception(backoff.expo, APIError, max_time=60, max_tries=10, giveup=lambda x: x.status != 429)
def run_query(service, query):
    """Execute a query with exponential backoff on API errors."""
    start_time = time.time()
    try:
        return service.search(query=query, columns=[], limit=1)
    except APIError as e:
        if e.status == 429:
            logging.warning("Backoff triggered for query: %s, error: %s", query, e)
        raise
    finally:
        latency = time.time() - start_time
        return latency

def read_queries(file_path):
    """Read queries from a file and expand them to a specified number."""
    with open(file_path, "r") as f:
        queries = f.readlines()
    return queries * (NUM_QUERIES // len(queries))

def collect_metrics(latencies, total_queries, total_time):
    """Calculate and print latency metrics and average QPS."""
    if latencies:
        latencies_np = np.array(latencies)
        p50 = np.percentile(latencies_np, 50)
        p90 = np.percentile(latencies_np, 90)
        p95 = np.percentile(latencies_np, 95)
        qps = total_queries / total_time if total_time > 0 else 0

        print("\n--- Query Performance Metrics ---")
        print(f"P50 Latency: {p50:.2f} seconds")
        print(f"P90 Latency: {p90:.2f} seconds")
        print(f"P95 Latency: {p95:.2f} seconds")
        print(f"Average QPS: {qps:.2f}\n")

def handle_errors(futures):
    """Count and print errors by status code."""
    error_counts = {}
    for f in futures:
        query = futures[f]
        try:
            f.result()  # Trigger exception if any occurred
        except APIError as e:
            error_counts[e.status] = error_counts.get(e.status, 0) + 1
            logging.error("Failed to execute query %s, error: %s", query, e)
        except Exception as e:
            logging.error("Unexpected error for query %s, error: %s", query, e)

    return error_counts

def print_error_counts(error_counts):
    """Print the counts of errors encountered."""
    if error_counts:
        print("--- Error Counts ---")
        for code, count in error_counts.items():
            print(f"Error Code {code}: {count} times")
    else:
        print("No errors encountered.")

def main():
    session = make_session()
    root = Root(session)
    search_service = root.databases[os.getenv("SNOWFLAKE_DATABASE")].schemas[os.getenv("SNOWFLAKE_SCHEMA")].cortex_search_services[os.getenv("SNOWFLAKE_CORTEX_SEARCH_SERVICE")]

    queries = read_queries("queries")
    latencies = []
    total_queries = len(queries)
    overall_start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_query, search_service, query): query for query in queries if query.strip()}

        # Track progress with tqdm for the completed futures
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing Queries", unit="query"):
            latency = f.result()  # Get the latency from the result
            latencies.append(latency)

    total_time = time.time() - overall_start_time

    # Calculate and print metrics
    collect_metrics(latencies, total_queries, total_time)

    # Handle and print error counts
    error_counts = handle_errors(futures)
    print_error_counts(error_counts)

if __name__ == "__main__":
    main()
