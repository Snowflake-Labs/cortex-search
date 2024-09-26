import os
import argparse
import json
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
def runQuery(service, query, columns):
    return service.search(query=query, columns=columns, limit=1)

def main():
    # Set environment variables from the .env file.
    load_dotenv("./.env")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Concurrent example")
    parser.add_argument("--host", default=os.getenv("SNOWFLAKE_HOST"), help="Snowflake host")
    parser.add_argument("-a", "--account", default=os.getenv("SNOWFLAKE_ACCOUNT", "CORTEXSEARCH"), help="Snowflake account")
    parser.add_argument("-u", "--user", default=os.getenv("SNOWFLAKE_USER", os.getenv("USER")), help="Snowflake user")
    parser.add_argument("-p", "--password", default=os.getenv("SNOWFLAKE_PASSWORD"), help="Snowflake password")
    parser.add_argument("-d", "--database", default=os.getenv("SNOWFLAKE_DATABASE"), help="Snowflake database")
    parser.add_argument("-s", "--schema", default=os.getenv("SNOWFLAKE_SCHEMA"), help="Snowflake schema")
    parser.add_argument("-c", "--cortex_search_service", default=os.getenv("SNOWFLAKE_CORTEX_SEARCH_SERVICE"), help="Cortex search service")
    parser.add_argument("-q", "--queries", default="queries", help="File containing queries to be executed")
    parser.add_argument("--authenticator", default=os.getenv("SNOWFLAKE_AUTHENTICATOR"), help="Snowflake authenticator, e.g. externalbrowser. Must not be used with --password")
    parser.add_argument("--columns", nargs="+", default=[], help="Columns to be returned")
    args = parser.parse_args()

    # Create a session and a root object
    config = {
        "account": args.account,
        "user": args.user,
    }
    if args.host:
        config["host"] = args.host

    if args.authenticator:
        config["authenticator"] = args.authenticator
    elif args.password:
        config["password"] = args.password
    else:
        raise ValueError("Either password or authenticator must be provided")

    session = Session.builder.configs(config).create()
    root = Root(session)
    search_service = root.databases[args.database].schemas[args.schema].cortex_search_services[args.cortex_search_service]

    # Read a set of queries to be executed on the service above.
    queries = []
    with open(args.queries, "r") as f:
        queries = f.readlines()

    # Execute queries in parallel with up to 20 concurrent threads
    with ThreadPoolExecutor(max_workers=20) as executor:
        fq = {executor.submit(runQuery, search_service, query, args.columns): query for query in queries if len(query) > 0}
        for f in as_completed(fq):
            query = fq[f]
            if len(query) > 40:
                query = query[:40] + "..."
            try:
                result = f.result()
            except APIError as ae:
                body = json.loads(ae.body)
                print("Failed to execute query %s, (status: %s, code: %d, message: %s", (query, ae.status, body["code"], body["message"]))
            else:
                if (len(result.results) > 0):
                    print("Top result for query %s : " % query, end=" ")
                    for (key, value) in result.results[0].items():
                        print(f"{key}={value}", sep=",")
                else:
                    print("No results for query %s", query)

if __name__=="__main__":
    main()
