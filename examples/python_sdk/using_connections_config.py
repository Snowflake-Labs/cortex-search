import os
import json

from snowflake.connector import connect
from snowflake.connector.config_manager import CONFIG_MANAGER
from snowflake.core import Root

svc = os.environ["SNOWFLAKE_CORTEX_SEARCH_SERVICE"]

# Replace with your search parameters
query = "riding shotgun"
columns = ["LYRIC","ALBUM_NAME","TRACK_TITLE","TRACK_N","LINE"]
limit = 5

with connect(
      connection_name="example_connection",
) as conn:
    try:
        # create a root as the entry point for all objects
        root = Root(conn)
        
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