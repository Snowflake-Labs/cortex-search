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

    cached_jwt = None

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

    def search(self, request_body: dict[str, any], retry_for_invalid_jwt=True) -> requests.Response:
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
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as http_err:
            # An invalid JWT may be due to expiration of the cached JWT.
            if retry_for_invalid_jwt and self.is_invalid_jwt_response(response):
                # Only retry with a fresh JWT once.
                print("JWT invalid, trying with a fresh JWT...")
                self.cached_jwt = None
                return self.search(request_body, retry_for_invalid_jwt = False)
            print(f"HTTP error occurred: {http_err}")
            return response
        except Exception as err:
            print(f"An error occurred: {err}")
        return None

    def make_headers(self):
        if self.cached_jwt is None:
            self.cached_jwt = jwtutils.generate_JWT_token(
                self.private_key_path, self.account, self.user
            )
        headers = {
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.cached_jwt}",
        }
        return headers

    def make_url(self, database: str, schema: str, service: str) -> str:
        return f"{self.base_url}/api/v2/cortex/search/databases/{database}/schemas/{schema}/services/{service}"

    def is_invalid_jwt_response(self, response: requests.Response) -> bool:
        return response.status_code == 401 and response.json()["code"] == "390144"
