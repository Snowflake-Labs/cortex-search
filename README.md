# Cortex Search

This repository contains example usage of Cortex Search, currently in Private Preview. The official preview documentation can be [found here](https://docs.snowflake.com/LIMITEDACCESS/cortex-search/cortex-search-overview).

## Examples

The `examples` directory showcases several Cortex Search usage patterns. Navigate to each of the following subdirectories for installation information and sample usage for the method of choice:

- [01_python_simple_usage](examples/01_python_simple_usage): Simple querying of a Cortex Search Service via the `snowflake` [python package](https://pypi.org/project/snowflake/).
- [02_rest_api_simple_usage](examples/02_rest_api_simple_usage):  Simple querying of a Cortex Search Service via the REST API
- [03_batch_querying_from_sql](examples/03_batch_querying_from_sql): Querying a Cortex Search Service in a "batch" fashion using SQL.
- [04_python_concurrent_queries](examples/04_python_concurrent_queries): Quering a Cortex Search Service with concurrency using the Python SDK.
- [05_streamlit_ai_search_app](examples/05_streamlit_ai_search_app): Sample Streamlit app using Cortex Search to power a search bar. 
- [06_streamlit_chatbot_app](examples/06_streamlit_chatbot_app): Sample Streamlit app using Cortex Search and Cortex LLM Functions to power a document chatbot.
- [07_streamlit_search_evaluation_app](examples/07_streamlit_search_evaluation_app): Streamlit app guiding users through evaluation of the quality of a Cortex Search Service.
- [08_multimodal_rag](examples/08_multimodal_rag): Sample notebook and streamlit app using Cortex Search and Cortex LLM Functions for multimodal RAG on PDFs.

## License

Apache Version 2.0
