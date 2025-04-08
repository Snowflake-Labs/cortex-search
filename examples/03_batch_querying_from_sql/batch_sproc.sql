/************************************************************************************
Name: batch_sproc.sql

Purpose:  This script creates a 'batch_cortex_search' Stored Procedure that calls a
Cortex Search Service with parallelism on an array of supplied query values. The script
shows the creation of the SProc and sample invocation on queries from a table column.

This script assumes you have an existing Cortex Search Service and table containing
queries you'd like to issue against it.
/************************************************************************************

/************************************************************************************
1. Create the SProc
************************************************************************************/

CREATE OR REPLACE PROCEDURE batch_cortex_search(db_name STRING, schema_name STRING, service_name STRING, queries ARRAY, filters ARRAY, columns ARRAY, n_jobs INTEGER DEFAULT -1)
RETURNS VARIANT
LANGUAGE PYTHON
PACKAGES = ('snowflake-snowpark-python==1.9.0', 'joblib==1.4.2', 'backoff==2.2.1')
RUNTIME_VERSION = '3.10'
HANDLER = 'main'
as
$$
import _snowflake
import json
import time
from joblib import Parallel, delayed
import backoff

@backoff.on_exception(backoff.expo, Exception, max_tries=5, giveup=lambda e: not (isinstance(e, Exception) and hasattr(e, "args") and len(e.args) > 0 and isinstance(e.args[0], dict) and e.args[0].get("status") == 429))
def call_api(db_name, schema_name, service_name, request_body):
    """Calls the Cortex Search REST API with retry logic for rate limiting."""
    resp = _snowflake.send_snow_api_request(
        "POST",
        f"/api/v2/databases/{db_name}/schemas/{schema_name}/cortex-search-services/{service_name}:query",
        {},
        {},
        request_body,
        {},
        30000,
    )
    if resp["status"] == 429:
        raise Exception({"status": resp["status"], "content": resp["content"]})
    return resp

def search(db_name, schema_name, service_name, query, columns, filter):
    """Calls the Cortex Search REST API and returns the response."""
    
    request_body = {
        "query": query,
        "columns": columns,
        "filter": filter,
        "limit": 5
    }
    try:
        resp = call_api(db_name, schema_name, service_name, request_body)
        if resp["status"] < 400:
            response_content = json.loads(resp["content"])
            results = response_content.get("results", [])
            return {"query": query, "filter": filter, "results": results}
        else:
            return {"query": query, "filter": filter, "results": f"Failed request with status {resp['status']}: {resp}"}
    except Exception as e:
        return {"query": query, "filter": filter, "results": f"API Error: {e}"}

def concurrent_searches(db_name, schema_name, service_name, queries, filters, columns, n_jobs):
    """Calls the Cortex Search REST API for multiple queries and returns the response."""

    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(search)(db_name, schema_name, service_name, q, columns, f) for q, f in zip(queries, filters)
    )
    responses = results
    return responses

def main(db_name, schema_name, service_name, queries, filters, columns, n_jobs):
    if isinstance(queries, list) and isinstance(filters, list):
        if len(queries) == len(filters):
            if len(queries) >= 1:
                return concurrent_searches(db_name, schema_name, service_name, queries, filters, columns, n_jobs)
            else:
                raise ValueError("Queries must be an array of query text")
        else:
            raise ValueError("Queries and filters must have the same length")
    else:
        raise ValueError("Queries and filters must be an array of query text")
$$;

/************************************************************************************
2. Call the SProc and materialize results in a table

Variables:
- <query_table>: the table containing queries in a column
- <query_column>: the name of the search column in the <query_table>
- <filter_column>: the name of the filter column in the <query_table>
- <col_to_return_k>: a desired column to return from the Cortex Search Service

Note: if you do not have filters to apply to queries, you can specify an empty object
for the FILTERS argument like:

`(SELECT ARRAY_AGG({}) FROM <query_table>)`

************************************************************************************/
CALL batch_cortex_search(
    '<db>',
    '<schema>',
    '<cortex_search_service>',
    (SELECT ARRAY_AGG(<query_column>) FROM <query_table>),
    (SELECT ARRAY_AGG(<filter_column>) FROM <query_table>),
    ARRAY_CONSTRUCT('<col_to_return_1>', '<col_to_return_2>', ...),
    -1
);

CREATE OR REPLACE TEMP TABLE RESULTS AS
SELECT * FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));

/************************************************************************************
3. View results
************************************************************************************/
SELECT
    value['query'] as query,
    value['results'][0]['col_to_return_1'] as col_1,
    value['results'][0]['col_to_return_2'] as col_2,
    -- ..
    value['filter'] as filter
FROM RESULTS r, LATERAL FLATTEN(r.batch_cortex_search)