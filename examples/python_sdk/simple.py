import os
import json

from snowflake.core import Root
from snowflake.connector import connect

# replace with hardcoded values if you wish; otherwise, ensure all values are in your environment.
CONNECTION_PARAMETERS = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "role": os.environ["SNOWFLAKE_ROLE"],
    "database": os.environ["SNOWFLAKE_DATABASE"],
    "schema": os.environ["SNOWFLAKE_SCHEMA"],
}

svc = os.environ["SNOWFLAKE_CORTEX_SEARCH_SERVICE"]

# create a SnowflakeConnection instance
connection = connect(**CONNECTION_PARAMETERS)

# Replace with your search parameters
query = "riding shotgun"
columns = ["LYRIC","ALBUM_NAME","TRACK_TITLE","TRACK_N","LINE"]
limit = 5

try:
    # create a root as the entry point for all objects
    root = Root(connection)
    
    response = (
        root.databases[CONNECTION_PARAMETERS["database"]]
        .schemas[CONNECTION_PARAMETERS["schema"]]
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
    connection.close()