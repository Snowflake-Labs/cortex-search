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
    role = None

    database = None
    schema = None
    service = None

    cached_jwt = None
    session_token = None

    def __init__(
        self,
        private_key_path: str,
        account_url: str,
        account: str,
        user: str,
        qualified_service_name: str,
        role: str | None = None,
    ):
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
        self.role = role

    def search(self, request_body: dict[str, any]) -> requests.Response:
        """
        Perform a POST request to the Cortex Search query API.

        :param request_body: The query body.
        :return: The requests.Response object.
        """
        url = self._make_query_url()
        if self.role is None:
            return self._make_request_with_jwt(url, request_body)

        return self._make_request_with_session_token(url, request_body)

    def _make_headers(self, use_jwt=True) -> dict[str, str]:
        if not use_jwt:
            return self._make_headers_with_session_token()
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

    def _make_headers_with_session_token(self) -> dict[str, str]:
        if self.session_token is None:
            self.session_token = self._create_session_token()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f'Snowflake Token="{self.session_token}"',
        }
        return headers

    def _create_session_token(self) -> str:
        url = self._make_sessions_url()
        request_body = {"roleName": self.role}
        res = self._make_request_with_jwt(url, request_body).json()
        return res["token"]

    def _make_query_url(self) -> str:
        return f"{self.base_url}/api/v2/databases/{self.database}/schemas/{self.schema}/cortex-search-services/{self.service}:query"

    def _make_sessions_url(self) -> str:
        return f"{self.base_url}/api/v2/sessions"

    def _make_request_with_session_token(
        self, url, request_body
    ) -> requests.Response | None:
        try:
            headers = self._make_headers(use_jwt=False)
            response = requests.post(url, headers=headers, json=request_body)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return response

    def _make_request_with_jwt(
        self, url, request_body, retry_for_invalid_jwt=True
    ) -> requests.Response | None:
        try:
            headers = self._make_headers()
            response = requests.post(url, headers=headers, json=request_body)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as http_err:
            # An invalid JWT may be due to expiration of the cached JWT.
            if retry_for_invalid_jwt and self._is_invalid_jwt_response(response):
                # Only retry with a fresh JWT once.
                print("JWT invalid, trying with a fresh JWT...")
                self.cached_jwt = None
                return self._make_request_with_jwt(
                    url, request_body, retry_for_invalid_jwt=False
                )
            print(f"HTTP error occurred: {http_err}")
            return response
        except Exception as err:
            print(f"An error occurred: {err}")
        return None

    def _is_invalid_jwt_response(self, response: requests.Response) -> bool:
        return response.status_code == 401 and response.json()["code"] == "390144"
