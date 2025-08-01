# Multimodal RAG with Cortex Search

This tutorial notebook guides you through creation of a Cortex Search Service from a set of PDFs, using a multimodal embedding model.

## Prerequisites
**This tutorial relies on a preview feature.** Before you can proceed with this tutorial, reach out to your account team to ask to enable this feature for your account:
  - [Multi-index Cortex Search Services](https://docs.snowflake.com/LIMITEDACCESS/cortex-search/multi-index-service)

**Python Package Requirements:**
  - Snowflake Python API version 1.6.0 or later (required for multi-index query syntax)

## Usage
- Upload the [attached notebook](../08_multimodal_rag/cortex_search_multimodal.ipynb) to Snowflake using the [instructions here](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks-create#create-a-new-notebook)
- Upload a set of PDFs to a snowflake internal stage (sample PDFs can be found [here](https://drive.google.com/drive/folders/1bExhPiJlF9aNushnXeLLBR4m9EMaShHw?usp=sharing))
- Follow the instructions in the notebook to embed the PDFs, OCR the PDF text, and create Cortex Search Service on the PDF embeddings plus OCR text for hybrid retrieval.
- Deploy the [streamlit app](../08_multimodal_rag/streamlit_chatbot_multimodal_rag.py) for multimodal RAG and search. Use the streamlit app to interactively ask questions about the pre-processed PDFs.
