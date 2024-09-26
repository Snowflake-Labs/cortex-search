import os
import json

from snowflake.connector import connect
from snowflake.connector.config_manager import CONFIG_MANAGER
from snowflake.core import Root

# Load the Cortex Search Service name from your environment
# svc = os.environ["SNOWFLAKE_CORTEX_SEARCH_SERVICE"]
svc = "FOMC_MINUTES_SEARCH_SERVICE"
print(svc)

# Replace with your search parameters
query = "riding shotgun"
columns = ["MINUTES", "MEETING_DATE"]
limit = 5

with connect(
      connection_name="pm"
) as conn:
    session = conn.session()

with connect(
      connection_name="pm"
) as conn:
    try:
        # create a root as the entry point for all objects
        root = Root(conn)
        print(conn.database, conn.schema)
        response = (
            root.databases[conn.database]
            .schemas[conn.schema]
            .cortex_search_services[svc]
            .search(
                query,
                columns,
                limit=limit
            )
        )

        print(f"Received response with `request_id`: {response.request_id}")
        print(json.dumps(response.results,indent=4))
    finally:
        conn.close()