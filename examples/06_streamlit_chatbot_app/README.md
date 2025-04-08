# RAG Chatbot with Cortex Search in Streamlit-in-Snowflake

This repository walks you through creating a [Cortex Search Service](https://docs.snowflake.com/LIMITEDACCESS/cortex-search/cortex-search-overview) and chatting with it in a Streamlit-in-Snowflake interface. This application runs entirely in Snowflake.

**Note**: Cortex Search is a private preview feature. To enable your account with the feature, please reach out to your account team.
  
## Instructions:
- Open a Snowflake SQL environment and run `Section 1` of [`setup.sql`](./setup.sql).
- Upload the files from [this folder](https://drive.google.com/drive/folders/1_erdfr7ZR49Ub2Sw-oGMJLs3KvJSap0d?usp=sharing) to the stage you created in Section 1. These files are recent FOMC minutes from the US Federal Reserve.
  - You can use this script with any set of files, but this example provides a specific set. You can follow the instructions to [upload files using Snowsight](https://docs.snowflake.com/en/user-guide/data-load-local-file-system-stage-ui) or, alternatively [via SnowSQL or drivers](https://docs.snowflake.com/en/user-guide/data-load-local-file-system-stage).
  - Note: The PyPDF2 library is not well optimized for PDFs that contain more complex structures such as tables, charts, images, etc. This parsing approach thus works best on simple, text-heavy PDFs.
- Run Sections 2 - 4 of `setup.sql` to parse the files and create the Cortex Search Service.
- In Snowsight, create a Streamlit in Snowflake in the same schema in which you created the Cortex Search Service (`demo_cortex_search.fomc`). In this streamlit app, add the following anaconda packages:
  - `snowflake==0.8.0`
  - `snowflake-ml-python==1.5.1`
- Copy and paste the [`chat.py`](./chat.py) script into your Streamlit-in-Snowflake application. Select your Cortex Search Service from the sidebar, then start chatting!
- You can congiure advanced chat settings in the sidebar of the streamlit app:
  - In the `Advanced Options` container, you can select the model used for answer generation, the number of context chunks used in each answer, and the depth of chat history messages to use in generation of a new response.
  - Toggling the `Debug` slider enables the printing of all context documents used in the model's answer in the sidebar
  - The `Session State` container shows the session state, including chat messages and currently-selected service.

## Sample queries:
  - **Example session 1**: multi-turn with point lookups
    - `how was gpd growth in q4 23?`
    - `how was unemployment in the same quarter?`
  - **Example 2**: summarizing multiple documents
    - `how has the fed's view of the market change over the course of 2024?`
  - **Example 3**: abstaining when the documents don't contain the right answer
    - `What was janet yellen's opinion about 2024 q1?`

## Components
1. [`setup.sql`](./setup.sql): this is a script that shows how to create a Cortex Search Service from a set of PDFs residing in a Snowflake stage, inlcuding parsing, chunking, and service creation.
3. [`chat.py`](./chat.py): this is a generic template for a Streamlit RAG chatbot that uses Cortex Search for retrieval and Cortex LLM Functions for Generation. This script is meant to be used in [Streamlit-in-Snowflake](https://docs.snowflake.com/en/developer-guide/streamlit/about-streamlit). The script shows off basic RAG orchestration techniques, with chat history summarization and prompt engineering. Feel free to customize it to your specific needs. You can use this script with any Cortex Search Service, not just the one we create in this tutorial.


## Improvements
- Add in-line citations with links to source chunks
- Add abstaining logic to the orchestration when no relevant documents were retrieved for the user's query
- Add query classification to respond quickly to pleasantries