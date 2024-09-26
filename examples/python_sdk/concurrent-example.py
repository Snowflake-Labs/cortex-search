import os
import backoff

from concurrent.futures import ThreadPoolExecutor, as_completed
from snowflake.snowpark import Session
from snowflake.core.exceptions import APIError
from snowflake.core import Root
from dotenv import load_dotenv

# Setup exponentioal backoff on "too many requests" errors (status: 429), with
# up to 10 retries.
@backoff.on_exception(backoff.expo,
                      APIError,
                      max_time=60, # seconds
                      max_tries=10,
                      giveup=lambda x : x.status != 429)
def runQuery(service, query):
    return service.search(query=query, columns=["LYRIC","ALBUM_NAME","TRACK_TITLE","TRACK_N","LINE"], limit=1)

def main():
    # Set environment variables from the .env file.
    load_dotenv("./.env")

    session = Session.builder.configs({
        "host": os.getenv("SNOWFLAKE_HOST"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT", "CORTEXSEARCH"),
        "user": os.getenv("SNOWFLAKE_USER", os.getenv("USER")),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
    }).create()
    root = Root(session)
    search_service = root.databases[os.getenv("SNOWFLAKE_DATABASE")].schemas[os.getenv("SNOWFLAKE_SCHEMA")].cortex_search_services[os.getenv("SNOWFLAKE_CORTEX_SEARCH_SERVICE")]

    # Read a set of queries to be executed on the service above.
    queries = []
    with open("queries", "r") as f:
        queries = f.readlines()

    # Execute queries in parallel with up to 20 concurrent threads
    with ThreadPoolExecutor(max_workers=20) as executor:
        fq = {executor.submit(runQuery, search_service, query): query for query in queries if len(query) > 0}
        for f in as_completed(fq):
            query = fq[f]
            if len(query) > 40:
                query = query[:40] + "..."
            try:
                result = f.result()
            except Exception as e:
                print("Failed to execute query %s, error: %s", (query, e))
            else:
                print("Top album for query %s is: %s" % (query, result.results[0]["ALBUM_NAME"]))

if __name__=="__main__":
    main()
