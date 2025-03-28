# Cortex Search

This repository contains example usage of Cortex Search, currently in Private Preview. The official preview documentation can be [found here](https://docs.snowflake.com/LIMITEDACCESS/cortex-search/cortex-search-overview).

## Examples

The `examples` directory showcases several Cortex Search usage patterns. Navigate to each of the following subdirectories for installation information and sample usage for the method of choice:

1. [examples/python_sdk](examples/python_sdk) for usage via the `snowflake` [python package](https://pypi.org/project/snowflake/).
   - [concurrent_example.py](examples/python_sdk/concurrent_example.py) calling Cortex Search serially from a Python client with high concurrency for "batch" use-cases
2. [examples/rest](examples/rest) for usage by hitting the REST API directly via python or the command line, using RSA key pair authentication with a JWT token.
3. [examples/streamlit-ai-search](examples/streamlit-ai-serach) searching for documents in a streamlit application with Cortex Search as the search backend.
4. [examples/streamlit-chat](examples/streamlit-chat) end-to-end RAG chatbot in Streamlit-in-Snowflake using Cortex LLM functions for chat response generation and Cortex Search for retrieval.
5. [examples/sql](examples/sql) examples calling Cortex Search from a SQL environment (e.g., Snowflake Worksheet)
   - [examples/sql](examples/sql/batch_sproc.sql) calling Cortex Search from a SQL SProc with high concurrency for "batch" use-cases, e.g., on a with a Snowflake table containing a column of query values

## License

Apache Version 2.0
