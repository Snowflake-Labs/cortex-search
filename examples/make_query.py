import requests
import argparse

import sys
from pathlib import Path
import json

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import jwtutils


class CortexSearchService:

    private_key_path = None

    base_url = None
    account = None
    user = None

    database = None
    schema = None
    service = None

    def __init__(
        self,
        private_key_path: str,
        account_url: str,
        account: str,
        user: str,
        qualified_service_name: str,
    ):
        """
        Initialize the CortexSearchAPIQuery class with the:
        - account url
        - account name
        - user name
        - qualified service name
        """
        self.private_key_path = private_key_path
        self.base_url = account_url
        self.account = account
        self.user = user
        service_name_parts = qualified_service_name.split(".")
        if len(service_name_parts) != 3:
            raise ValueError(
                f"Expected qualified name to have DB, schema, and name components; got {qualified_service_name}"
            )
        self.database = service_name_parts[0]
        self.schema = service_name_parts[1]
        self.service = service_name_parts[2]

    def search(self, request_body: dict[str, any]) -> requests.Response:
        """
        Perform a POST request to a specified endpoint of the API.

        :param endpoint: The API endpoint to query.
        :param params: A dictionary of query parameters (optional).
        :return: The response from the API as a JSON object.
        """
        try:
            url = self.make_url(self.database, self.schema, self.service)
            headers = self.make_headers()
            response = requests.post(url, headers=headers, json=request_body)
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"An error occurred: {err}")
        return None

    def make_headers(self):
        jwt = jwtutils.generate_JWT_token(
            self.private_key_path, self.account, self.user
        )
        headers = {
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {jwt}",
        }
        return headers

    def make_url(self, database: str, schema: str, service: str) -> str:
        return f"{self.base_url}/api/v2/cortex/search/databases/{database}/schemas/{schema}/services/{service}"


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

    search_service = CortexSearchService(
        args.private_key_path,
        args.url,
        args.account,
        args.user_name,
        args.qualified_service_name,
    )
    response = search_service.search(request_body=request_body)

    if response is not None:
        print(json.dumps(response, indent=4))
    else:
        print("Failed to fetch data.")


if __name__ == "__main__":
    main()
