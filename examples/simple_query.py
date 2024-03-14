import requests
import argparse

import sys
from pathlib import Path
import json

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import jwtutils, queryutils

def main():
    parser = argparse.ArgumentParser(description="Cortex Search Service API Query CLI")
    parser.add_argument(
        "-u",
        "--url",
        help="Snowflake account URL",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--account",
        help="Snowflake account name, eg my_org-my_account",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--user-name",
        help="Snowflake user name",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--qualified-service-name",
        help="Qualified name of the Cortex Search Service, eg `MY_DB.MY_SCHEMA.MY_SERVICE`. Case insensitive.",
        required=True,
    )
    parser.add_argument(
        "-k",
        "--private-key-path",
        help="Absolute local path to RSA private key",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--columns",
        help="Comma-separated list of columns to return",
        required=True,
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Query string",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--limit",
        help="Max number of results to return",
        required=True,
    )

    args = parser.parse_args()

    request_body = {
        "columns": args.columns.split(","),
        "query": args.query,
        "limit": args.limit,
    }

    search_service = queryutils.CortexSearchService(
        args.private_key_path,
        args.url,
        args.account,
        args.user_name,
        args.qualified_service_name,
    )
    response = search_service.search(request_body=request_body, retry_for_invalid_jwt=False)

    if response is not None:
        print(json.dumps(response.json(), indent=4))
    else:
        print("Failed to fetch data.")


if __name__ == "__main__":
    main()
